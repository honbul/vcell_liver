"""Measure source-only HCC transfer after profiling assay-by-tissue drug means."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .anchor_rank_cv import PREVIOUS_SCORE, old_forecasts
from .assay_aware_cv import source_grids
from .evaluate import score, score_across_cells
from .external import KEY, SOURCE
from .io import freeze_table, publish_json, sha256, verify_forecast
from .structured_cv import ASSAYS, attach, observations, split, with_assay
from .v0plus_cv import molecular_inputs, panel_targets, training_inputs, validation_roster
from .v0plus_score import crossed_patient_drug_ci

PENALTY = 4.
ANCHOR_PRIOR = 12.
PRIMARY = "within_tissue_hcc_anchor"
ARMS = ("within_tissue_global_anchor", PRIMARY, "structured_hcc_anchor")
BASELINES = {"drug_mean", "drug_ridge", "structured", "ridge_structured_average",
             "cellhit", "v0plus_inductive"}


def tissue_map(rows):
    values = rows.tissue_origin.fillna("").replace("", "unknown")
    return rows.assign(tissue=values.mask(rows.lineage_role.eq("adult_hcc"), "adult_hcc"))


def fit_anchors(rows, anchor):
    lineage_means = (rows.groupby([*SOURCE, "compound_id", "tissue", "lineage_group"])
                     .model_value.mean().rename("value").reset_index())
    means = lineage_means.groupby([*SOURCE, "compound_id", "tissue"]).value.agg(["mean", "size"])
    source_mean = rows.groupby([*SOURCE, "compound_id"]).model_value.mean()
    means = means.join(source_mean.rename("base"), on=[*SOURCE, "compound_id"], validate="many_to_one")
    means["shift"] = means["size"] / (means["size"] + ANCHOR_PRIOR) * (means["mean"] - means.base)
    source_count = source_mean.groupby(level="compound_id").size()
    shifts = means.groupby(["compound_id", "tissue"])["shift"].sum() / source_count
    return shifts.add(anchor, level="compound_id")


def tensor_membership(labels, cells, device):
    import torch

    values = labels.reindex(cells)
    if values.isna().any():
        raise ValueError("Missing tissue for a source model")
    codes = pd.Categorical(values)
    one_hot = torch.nn.functional.one_hot(torch.as_tensor(codes.codes.copy(), dtype=torch.long, device=device))
    return codes.categories, one_hot.T.float()


def remove_group_location(weight, membership):
    """Return the weighted orthogonal projector away from assay/tissue/drug intercepts."""
    root = weight.sqrt()
    totals = (membership @ weight).clamp_min(1e-12)

    def project(value):
        location = (membership @ (root * value)) / totals
        return value - root * (membership.T @ location)

    return project


def solve_profiled(cell, drug, weights, response, membership, penalty=PENALTY):
    """Solve the kernel in the observed row space after profiling group intercepts."""
    import torch

    root = weights.sqrt()
    project = remove_group_location(weights, membership)
    rhs = project(root * response)
    coefficient = torch.zeros_like(rhs)
    residual = rhs.clone()
    diag = penalty + weights * cell.diagonal()[:, None] * drug.diagonal()[None, :]
    direction = residual / diag
    energy = (direction * residual).sum()
    tolerance = max(1e-7, 5e-5 * rhs.norm().item())
    for step in range(220):
        merged = (root * project(direction)).sum(dim=0)
        product = project(root * (cell @ merged @ drug)) + penalty * direction
        update = energy / (direction * product).sum().clamp_min(1e-20)
        coefficient += update * direction
        residual -= update * product
        distance = residual.norm().item()
        if distance <= tolerance:
            return (root * project(coefficient)).sum(dim=0), {"steps": step + 1, "residual": distance}
        scaled = residual / diag
        next_energy = (scaled * residual).sum()
        direction = scaled + next_energy / energy * direction
        energy = next_energy
    raise RuntimeError(f"Profiled solve residual {distance:.5g} exceeds tolerance {tolerance:.5g}")


def kernel_forecast(fit, query, rna, edges, targets, auxiliary, device):
    from .structured_kernel import cell_kernels, drug_kernels

    _, anchor, rows, _ = panel_targets(fit)
    rows = tissue_map(rows)
    mapping = rows[["cell_id", "tissue"]].drop_duplicates().set_index("cell_id", verify_integrity=True).tissue
    fit_rna = rna.loc[sorted(fit.cell_id.unique())]
    query_rna = rna.loc[sorted(query.cell_id.unique())]
    cell_terms, _ = cell_kernels(fit_rna, query_rna, edges, auxiliary, device=device)
    chemicals, drug_terms = drug_kernels(rows, targets, device=device)
    cell, cross = cell_terms["full"]
    drug = drug_terms["structured"]
    tissue_names, membership = tensor_membership(mapping, fit_rna.index, device)
    weight, response = source_grids(rows, fit_rna.index, chemicals, device)
    alpha, stats = solve_profiled(cell, drug, weight, response, membership)
    column = chemicals.get_indexer(query.compound_id)
    row = query_rna.index.get_indexer(query.cell_id)
    check_tissue = tissue_names.get_indexer(query.tissue)
    mean = (membership @ (cell @ alpha @ drug)) / membership.sum(dim=1)[:, None]
    residual = (cross @ alpha @ drug).cpu().numpy()[row, column]
    if (check_tissue < 0).any():
        raise ValueError("Held-out HCC tissue has no source fit profiles")
    residual -= mean.cpu().numpy()[check_tissue, column]
    baseline = query.compound_id.map(anchor).to_numpy()
    adjusted = fit_anchors(rows, anchor).reindex(
        pd.MultiIndex.from_frame(query[["compound_id", "tissue"]]),
    ).to_numpy()
    adjusted = np.where(np.isfinite(adjusted), adjusted, baseline)
    diagnostic = stats | {
        "fit_hcc_lineages": int(rows.loc[rows.tissue.eq("adult_hcc"), "lineage_group"].nunique()),
        "fit_tissues": len(tissue_names),
        "hcc_anchor_abs_median_change": float(np.median(np.abs(adjusted - baseline))),
    }
    return baseline, adjusted, residual, diagnostic


def predict(args):
    import torch

    from .v0plus_neural import load_auxiliary

    if not torch.cuda.is_available():
        raise RuntimeError("The registered conditional kernel requires CUDA")
    previous = args.baseline / "score.json"
    if sha256(previous) != PREVIOUS_SCORE:
        raise ValueError("Unexpected structured development baseline")
    previous_hashes = json.loads(previous.read_text())["forecast_sha256"]
    archive, digest = old_forecasts(args.baseline, args.assay, args.fold, previous_hashes)
    train, rna, _ = training_inputs(args.bundle)
    fit, check = split(train, args.fold, args.assay)
    evaluation = check[check.lineage_role.eq("adult_hcc")]
    query = validation_roster(evaluation).assign(tissue="adult_hcc")
    priors, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, _ = load_auxiliary(args.auxiliary)
    baseline, adjusted, residual, diagnostic = kernel_forecast(
        fit, query, rna, priors, targets, auxiliary, torch.device("cuda"),
    )
    recalibration = query[KEY].assign(shift=adjusted - baseline, base=baseline)
    old = archive[archive.row_id.isin(query.row_id) & archive.method.isin(BASELINES)]
    selected = old[old.method.eq("structured")].merge(recalibration, on=KEY, validate="many_to_one")
    anchors = old[old.method.eq("drug_mean")].merge(recalibration, on=KEY, validate="many_to_one")
    if not np.allclose(anchors.predicted_value, anchors.base, atol=1e-9, rtol=1e-8):
        raise ValueError("New model and anchor ablation require the registered source scale")
    methods = [old, selected.assign(method=ARMS[2], predicted_value=selected.predicted_value + selected["shift"])]
    for method, location in zip(ARMS, (baseline, adjusted)):
        part = query[KEY].assign(method=method, predicted_value=location + residual)
        methods.append(with_assay(part, evaluation, args.fold))
    kept = [*KEY, "method", "track", "fold", "predicted_value"]
    result = pd.concat([part[kept] for part in methods], ignore_index=True)
    if not np.isfinite(result.predicted_value).all():
        raise ValueError("Missing prediction on registered HCC rows")
    freeze_table(result, args.out, provenance={
        "assay": args.assay, "fold": args.fold, "bundle_sha256": sha256(args.bundle / "receipt.json"),
        "baseline_sha256": digest, "query_lineages": int(evaluation.lineage_group.nunique()),
        "query_observations": len(evaluation), "diagnostics": diagnostic,
    })


def fold_centered_correlations(frame):
    """Accumulate within-fold cross-products, weighted by sample support."""
    units = (frame.groupby(["track", "method", "compound_id", "fold", "lineage_group"], as_index=False)
             [["endpoint_value", "predicted_value"]].mean())
    keys = ["track", "method", "compound_id", "fold"]
    by_fold = units.groupby(keys, sort=False)
    values = units[["endpoint_value", "predicted_value"]]
    ranks = by_fold[["endpoint_value", "predicted_value"]].rank()
    output = {}
    for name, data in (("pearson", values), ("spearman", ranks)):
        centered = data - data.groupby([units[k] for k in keys]).transform("mean")
        a, b = centered.to_numpy().T
        terms = units[keys].assign(n=1, aa=a*a, ab=a*b, bb=b*b).groupby(keys, as_index=False).sum()
        valid = terms[terms.n.ge(3)]
        totals = valid.groupby(keys[:-1], as_index=False)[["n", "aa", "ab", "bb"]].sum()
        totals = totals[totals.n.ge(5) & totals.aa.gt(1e-16)]
        den = np.sqrt(totals.aa * totals.bb)
        totals[name] = np.divide(totals.ab, den, out=np.zeros(len(den)), where=den.gt(1e-8))
        output[name] = totals.groupby(["track", "method"], as_index=False)[name].agg(["mean", "count"]).reset_index(
            drop=True,
        ).to_dict("records")
    return output


def score_mode(frame):
    methods = set(frame.method.unique())
    comparators = [base for base in ("drug_mean", "drug_ridge", "structured", "cellhit") if base in methods]
    contrasts = [(m, b) for m in ARMS for b in comparators]
    cell, _ = score(frame, endpoint_direction="higher", rank_only=True, comparisons=contrasts)
    identities = frame[["track", "row_id", "fold"]].drop_duplicates()
    if identities.duplicated(["track", "row_id"]).any():
        raise ValueError("A grouped prediction row was present in more than one outer fold")
    fold_drug, _ = score_across_cells(frame)
    oof_drug, _ = score_across_cells(frame.assign(fold=0))
    across = oof_drug.groupby(["track", "method"], as_index=False)[["spearman", "pearson"]].mean()
    within_fold = fold_drug.groupby(["track", "method"], as_index=False)[["spearman", "pearson"]].mean()
    intervals = [
        {"track": track, "candidate": PRIMARY, "baseline": base,
         "ci95": crossed_patient_drug_ci(part, PRIMARY, base, repeats=1200)}
        for track, part in frame.groupby("track") for base in comparators
    ]
    by_track = {}
    for track in sorted(frame.track.unique()):
        drug_intervals = [d["ci95"] for d in intervals if d["track"] == track
                          and d["baseline"] in {"drug_ridge", "structured"}]
        cell_intervals = [d["ci95"] for d in cell["paired_contrasts"] if d["track"] == track
                          and d["candidate"] == PRIMARY and d["baseline"] == "drug_mean"
                          and d["metric"] in {"spearman", "ndcg_at_5"}]
        by_track[track] = (len(drug_intervals) == len(cell_intervals) == 2
                           and all(len(d) == 2 for d in drug_intervals)
                           and all(v[0] > 0 for d in drug_intervals for v in d.values())
                           and all(d[0] > 0 for d in cell_intervals))
    return {"across_cell": across.to_dict("records"), "within_cell": cell["scores"],
            "within_fold_across_cell": within_fold.to_dict("records"),
            "across_cell_intervals": intervals, "within_cell_intervals": cell["paired_contrasts"],
            "fold_centered": fold_centered_correlations(frame), "track_gate": by_track}


def score_all(args):
    train, _, _ = training_inputs(args.bundle)
    bundle_hash = sha256(args.bundle / "receipt.json")
    modes, audits, hashes = {}, {}, {}
    for assay in ASSAYS:
        parts = []
        for fold in range(3):
            path = args.forecasts / f"{assay}-{fold}.parquet"
            frame, seal = verify_forecast(path)
            meta = seal["provenance"]
            if (meta["assay"], meta["fold"], meta["bundle_sha256"]) != (assay, fold, bundle_hash):
                raise ValueError("Unexpected grouped split in the frozen HCC prediction")
            _, check = split(train, fold, assay)
            check = check[check.lineage_role.eq("adult_hcc")]
            index = [*KEY, "track"]
            expected = set(observations(check).set_index(index).index)
            for _, method in frame.groupby("method"):
                if len(method) != len(expected) or set(method.set_index(index).index) != expected:
                    raise ValueError("Methods require exactly the same development HCC rows")
            parts.append(attach(frame, check))
            hashes[path.stem] = seal["sha256"]
            audits[path.stem] = meta
        modes[assay] = score_mode(pd.concat(parts, ignore_index=True))
    publish_json(args.out, {"modes": modes, "fold_audits": audits, "forecast_sha256": hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, options in {
        "predict": ("bundle", "baseline", "priors", "hallmark", "targets", "auxiliary", "out"),
        "score": ("bundle", "forecasts", "out"),
    }.items():
        command = commands.add_parser(name)
        for arg in options:
            command.add_argument("--" + arg, type=Path, required=True)
        if name == "predict":
            command.add_argument("--assay", choices=ASSAYS, required=True)
            command.add_argument("--fold", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    {"predict": predict, "score": score_all}[args.action](args)


if __name__ == "__main__":
    main()
