"""Separate shared and assay-specific drug response in the structured kernel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .evaluate import score, score_across_cells
from .external import KEY, SOURCE, _validate
from .io import freeze_table, publish_json, sha256, verify_forecast
from .structured_cv import ASSAYS, attach, observations, paired_intervals, split, with_assay
from .v0plus_cv import molecular_inputs, panel_targets, summarize, training_inputs, validation_roster

RATIOS = (.25, 1., 4.)
PENALTY = 4.
BASELINE_SHA256 = "10af6062118b6b3ad9eb205b2b465e7beeaad03d1be90035d6bef8cfacfdd521"
MAIN = "assay_shared"
BLEND = "assay_shared_ridge_average"


def source_grids(rows, cells, drugs, device):
    import torch

    groups = rows.groupby([*SOURCE, "compound_id"]).residual
    sizes = groups.transform("size")
    largest = rows.assign(size=sizes).groupby("compound_id")["size"].transform("max")
    counts = rows[[*SOURCE, "compound_id"]].drop_duplicates().groupby("compound_id").size()
    weighted = rows.assign(weight=largest / (sizes * rows.compound_id.map(counts)))
    shape = (len(cells), len(drugs))
    weights, response = [], []
    for _, source in weighted.groupby(SOURCE, sort=True):
        w, y = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
        index = cells.get_indexer(source.cell_id), drugs.get_indexer(source.compound_id)
        np.add.at(w, index, source.weight.to_numpy())
        np.add.at(y, index, (source.weight * source.residual).to_numpy())
        weights.append(w)
        response.append(np.divide(y, w, out=y, where=w > 0))
    return tuple(torch.as_tensor(np.stack(value), device=device) for value in (weights, response))


def solve_tasks(cell, drug, weight, response, private_ratio):
    """Fit K_cell × K_drug × (ones + private_ratio * identity) with missing outcomes."""
    import torch

    root = weight.sqrt()
    rhs = root * response
    diag = PENALTY + (1 + private_ratio) * weight * cell.diagonal()[:, None] * drug.diagonal()[None, :]
    coefficient = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual / diag
    energy = (direction * residual).sum()
    tolerance = max(1e-7, 5e-5 * rhs.norm().item())
    for iteration in range(220):
        via_biology = cell @ (root * direction) @ drug
        product = root * (via_biology.sum(dim=0) + private_ratio * via_biology) + PENALTY * direction
        update = energy / (direction * product).sum().clamp_min(1e-20)
        coefficient += update * direction
        residual -= update * product
        error = residual.norm().item()
        if error <= tolerance:
            return (root * coefficient).sum(dim=0), {"iterations": iteration + 1, "residual": error}
        scaled = residual / diag
        next_energy = (scaled * residual).sum()
        direction = scaled + (next_energy / energy) * direction
        energy = next_energy
    raise RuntimeError(f"Multitask solver did not reach residual tolerance {tolerance:.5g}, residual {error:.5g}")


def candidates(fit, query, rna, edges, targets, auxiliary, device, ratios):
    from .structured_kernel import cell_kernels, drug_kernels

    _validate(fit, query)
    _, anchor, rows, _ = panel_targets(fit)
    training_rna = rna.loc[sorted(fit.cell_id.unique())]
    query_rna = rna.loc[sorted(query.cell_id.unique())]
    cell, _ = cell_kernels(training_rna, query_rna, edges, auxiliary, device=device)
    drug_names, drug = drug_kernels(rows, targets, device=device)
    w, y = source_grids(rows, training_rna.index, drug_names, device)
    cell_kernel, to_query = cell["full"]
    drug_kernel = drug["structured"]
    idx = query_rna.index.get_indexer(query.cell_id), drug_names.get_indexer(query.compound_id)
    anchor = query.compound_id.map(anchor).to_numpy()
    parts, diagnostics = [], {}
    for ratio in ratios:
        alpha, report = solve_tasks(cell_kernel, drug_kernel, w, y, ratio)
        residual = (to_query @ alpha @ drug_kernel).cpu().numpy()[idx]
        parts.append(query[KEY].assign(method=f"{MAIN}/{ratio}", predicted_value=anchor + residual))
        diagnostics[str(ratio)] = report
    return pd.concat(parts, ignore_index=True), diagnostics


def predict(args):
    import torch

    from .v0plus_neural import load_auxiliary

    if not torch.cuda.is_available():
        raise RuntimeError("The registered assay kernel requires CUDA")
    device = torch.device("cuda")
    previous = args.baseline / "score.json"
    if sha256(previous) != BASELINE_SHA256:
        raise ValueError("Expected the previously registered structured kernel score receipt")
    train, rna, _ = training_inputs(args.bundle)
    fit, check = split(train, args.fold, args.assay)
    edges, targets, prior = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    inner = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=2).split(fit, groups=fit.lineage_group)):
        fit_a, check_b = fit.iloc[a], fit.iloc[b]
        check_b = check_b[check_b.compound_id.isin(fit_a.compound_id)]
        query = validation_roster(check_b)
        forecast, _ = candidates(fit_a, query, rna, edges, targets, auxiliary, device, RATIOS)
        inner.append(attach(with_assay(forecast, check_b, fold), check_b))
    cv = summarize(pd.concat(inner, ignore_index=True))
    utility = cv.assign(rank_utility=.5 * cv.spearman + .25 * cv.macro_spearman + .25 * cv.macro_ndcg_at_5)
    candidate = utility.groupby("method").rank_utility.mean().idxmax()
    ratio = float(candidate.split("/")[1])
    roster = validation_roster(check)
    new, solver = candidates(fit, roster, rna, edges, targets, auxiliary, device, [ratio])
    new = with_assay(new.assign(method=MAIN), check, args.fold)
    key = [*KEY, "track", "fold"]
    path = args.baseline / "forecasts" / f"{args.assay}-{args.fold}.parquet"
    archive, receipt = verify_forecast(path)
    if receipt["sha256"] != json.loads(previous.read_text())["forecast_sha256"][path.stem]:
        raise ValueError("The archived forecast was not the registered fold")
    kept = archive[archive.method.isin({
        "structured", "ridge_structured_average", "drug_mean", "drug_ridge", "cellhit", "v0plus_inductive",
    })]
    compare = kept[kept.method.eq("drug_ridge")][key + ["predicted_value"]]
    paired = new.merge(compare, on=key, validate="one_to_one", suffixes=("_shared", "_ridge"))
    if len(paired) != len(new) or len(paired) != len(compare):
        raise ValueError("Ridge and assay kernel were not evaluated on the same support")
    blend = paired[key].assign(method=BLEND,
                               predicted_value=(paired.predicted_value_shared + paired.predicted_value_ridge) / 2)
    forecast = pd.concat([kept, new, blend], ignore_index=True)
    if not np.isfinite(forecast.predicted_value).all():
        raise ValueError("Nonfinite forecast on the matched source grid")
    freeze_table(forecast, args.out, provenance={
        "assay": args.assay, "fold": args.fold, "selected_private_ratio": ratio, "solver": solver,
        "training_sources": sorted(fit.source_id.unique()), "forecast_lines": len(new),
        "baseline_sha256": receipt["sha256"], "bundle_sha256": sha256(args.bundle / "receipt.json"),
        "molecular_priors": prior, "auxiliary_receipt": aux_receipt,
        "inner_utilities": utility.groupby("method").rank_utility.mean().to_dict(),
    })


def score_frozen(args):
    train, _, _ = training_inputs(args.bundle)
    modes, hashes, settings = {}, {}, {}
    bundle_hash = sha256(args.bundle / "receipt.json")
    for assay in ASSAYS:
        folds = []
        settings[assay] = []
        for fold in range(3):
            path = args.forecasts / f"{assay}-{fold}.parquet"
            frame, seal = verify_forecast(path)
            record = seal["provenance"]
            if (record["assay"], record["fold"], record["bundle_sha256"]) != (assay, fold, bundle_hash):
                raise ValueError("Unexpected source split identity")
            _, check = split(train, fold, assay)
            expected = set(observations(check).set_index([*KEY, "track"]).index)
            if any(set(part.set_index([*KEY, "track"]).index) != expected
                   for _, part in frame.groupby("method")):
                raise ValueError("Incomplete common support in frozen assay-aware forecasts")
            settings[assay].append({key: record[key] for key in (
                "selected_private_ratio", "inner_utilities", "solver", "forecast_lines", "training_sources",
            )})
            hashes[path.stem] = seal["sha256"]
            folds.append(attach(frame, check))
        merged = pd.concat(folds, ignore_index=True)
        pairs = [(MAIN, "structured"), (MAIN, "drug_mean"), (BLEND, "ridge_structured_average"), (BLEND, "drug_mean")]
        stats, _ = score(merged, endpoint_direction="higher", rank_only=True, comparisons=pairs)
        drugs, _ = score_across_cells(merged)
        associations = drugs.groupby(["track", "method"], as_index=False)[["spearman", "pearson"]].mean()
        by_source = pd.DataFrame(stats["scores"]).merge(associations, on=["track", "method"], validate="one_to_one")
        columns = ["spearman", "pearson", "macro_spearman", "macro_ndcg_at_5", "macro_top5_overlap"]
        increments = [
            {"candidate": name, "baseline": base, "ci95": paired_intervals(merged, name, base)}
            for name, base in ((MAIN, "structured"), (BLEND, "ridge_structured_average"))
        ]
        first = [row["ci95"][0] for row in stats["paired_contrasts"]
                 if row["candidate"] == MAIN and row["baseline"] == "drug_mean"
                 and row["metric"] in {"spearman", "ndcg_at_5"}]
        modes[assay] = {
            "table": by_source.groupby("method", as_index=False)[columns].mean().to_dict("records"),
            "by_source": by_source.to_dict("records"), "intervals_across_drugs": stats["paired_contrasts"],
            "intervals_across_cells": increments,
            "gate_primary": increments[0]["ci95"]["spearman"][0] > 0
            and len(first) == 2 * merged.track.nunique() and min(first) > 0,
        }
    publish_json(args.out, {"modes": modes, "forecast_sha256": hashes, "settings": settings})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("predict")
    for key in ("bundle", "priors", "hallmark", "targets", "auxiliary", "baseline", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--assay", required=True, choices=ASSAYS)
    p.add_argument("--fold", required=True, type=int, choices=range(3))
    s = sub.add_parser("score")
    for key in ("bundle", "forecasts", "out"):
        s.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    {"predict": predict, "score": score_frozen}[args.action](args)


if __name__ == "__main__":
    main()
