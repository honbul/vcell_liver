"""Test whether added pharmacology improves on the corrected known-drug core."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .evaluate import score, score_across_cells
from .external import KEY, SOURCE, _validate, native_sensitivity
from .hcc_conditional_cv import fold_centered_correlations
from .io import freeze_table, publish_json, sha256, verify_forecast
from .structured_cv import ASSAYS, attach, observations, paired_intervals, split, with_assay
from .v0plus_cv import candidate_forecasts, molecular_inputs, panel_targets, training_inputs, validation_roster

RECEIPTS = {
    "original": "663eb5717b30c6f03ea43c04c4bb076bd233ea0c05405450e17f0c9e85f7fa86",
    "core": "d0ab7d0dac5d3c886495d48420d7dc448df019dc3b9196de898bab3c8b05a5d6",
    "extension": "412911e70d9b68e4d28c2229d8ed3e0b6531a1293ae6a5cc06fb0b289f4552b0",
}
PENALTY = 4.
RIDGE = ("drug_ridge", 1000., 512, 1.)
PRIMARY = "expanded_structured"
ARMS = (
    "historical_core", "corrected_core", PRIMARY, "expanded_generic_only", "corrected_ridge",
    "corrected_drug_mean", "expanded_ridge_average",
)
CONTRASTS = (
    ("corrected_core", "historical_core"),
    (PRIMARY, "corrected_core"),
    (PRIMARY, "expanded_generic_only"),
    (PRIMARY, "corrected_ridge"),
    ("expanded_ridge_average", "corrected_ridge"),
)


def registered_inputs(args, *, include_extension=False):
    for name, expected in RECEIPTS.items():
        if hasattr(args, name) and sha256(getattr(args, name) / "receipt.json") != expected:
            raise ValueError(f"Unexpected immutable source receipt: {name}")
    old, _, _ = training_inputs(args.original)
    clean, rna, correction = training_inputs(args.core)
    if correction["original_bundle_sha256"] != RECEIPTS["original"]:
        raise ValueError("Original and corrected cores are unrelated")
    extra = None
    if include_extension:
        metadata = json.loads((args.extension / "receipt.json").read_text())
        path = args.extension / "additional_rows.parquet"
        if (metadata["original_bundle_sha256"] != RECEIPTS["original"]
                or sha256(path) != metadata["artifacts"][path.name]):
            raise ValueError("Expanded training rows failed verification")
        extra = pd.read_parquet(path)
        if (set(extra.compound_id) & set(clean.compound_id)
                or not set(extra.lineage_group) <= set(old.lineage_group)
                or not set(extra.cell_id) <= set(old.cell_id)):
            raise ValueError("Extension changes the core chemical or lineage roster")
    return old, clean, rna, extra


def same_split(old, clean, fold, assay):
    history, historical_test = split(old, fold, assay)
    train = clean[clean.lineage_group.isin(history.lineage_group)]
    if assay != "pooled":
        train = train[train.source_id.ne(assay)]
    available = set(history.compound_id) & set(train.compound_id)
    check = clean[clean.row_id.isin(historical_test.row_id) & clean.compound_id.isin(available)]
    if (set(history.cell_id) != set(train.cell_id)
            or set(check.lineage_group) & set(train.lineage_group)
            or (assay != "pooled" and assay in set(train.source_id))):
        raise ValueError("Data scope would change the frozen fit lineages or assay")
    return history, train, check


def expand_aligned(extra, calibration):
    """Keep the corrected core calibration fixed when adding new drugs."""
    parameters = pd.DataFrame(calibration)[[*SOURCE, "slope", "intercept"]]
    rows = extra.merge(parameters, on=SOURCE, how="left", validate="many_to_one")
    if rows.slope.isna().any() or not rows.slope.gt(0).all():
        raise ValueError("A supplemental assay has no positive fit-only calibration")
    values = native_sensitivity(rows) * rows.slope + rows.intercept
    source_mean = values.groupby([rows[key] for key in [*SOURCE, "compound_id"]]).transform("mean")
    return rows.assign(model_value=values, residual=values - source_mean)


def expanded_targets(extra, core_targets, genes):
    symbols = set(genes)
    unique = extra[["compound_id", "drug_target_source_labels"]].drop_duplicates()
    extras = {}
    for compound, group in unique.groupby("compound_id", sort=True):
        terms = {term for value in group.drug_target_source_labels.dropna()
                 for term in re.split(r"[;,|/\s]+", value)}
        exact = sorted(terms & symbols)
        if exact:
            extras[compound] = exact
    return core_targets | extras, len(extras)


def structural_predictions(history, clean, extra, query, rna, edges, targets, auxiliary, device):
    import torch
    from .structured_kernel import cell_kernels, drug_kernels, solve_residuals, training_grid

    for fit in (history, clean):
        _validate(fit, query)
    fixed_rna = rna.loc[sorted(clean.cell_id.unique())]
    query_rna = rna.loc[sorted(query.cell_id.unique())]
    cell_k, aux_count = cell_kernels(fixed_rna, query_rna, edges, auxiliary, device=device)
    cell, cross = cell_k["full"]
    _, clean_anchor, clean_rows, calibration = panel_targets(clean)
    _, old_anchor, old_rows, _ = panel_targets(history)
    extra_rows = expand_aligned(extra, calibration)
    joint_rows = pd.concat([clean_rows, extra_rows], ignore_index=True)
    all_targets, new_known = expanded_targets(extra_rows, targets, rna.columns)
    row_index = query_rna.index.get_indexer(query.cell_id)
    diagnostics, parts = {}, {}

    def estimate(name, drug_names, drug, weight, response, anchor):
        columns = drug_names.get_indexer(query.compound_id)
        if (columns < 0).any():
            raise ValueError("No training support for part of the common query panel")
        coef, steps, error = solve_residuals(cell, drug, weight, response, PENALTY)
        raw = (cross @ coef @ drug).cpu().numpy()[row_index, columns]
        parts[name] = query.compound_id.map(anchor).to_numpy() + raw
        diagnostics[name] = {"iterations": steps, "solver_residual": error}

    for name, rows, anchor in (("historical_core", old_rows, old_anchor), ("corrected_core", clean_rows, clean_anchor)):
        drugs, kernels = drug_kernels(rows, targets, device=device)
        weight, response = training_grid(rows, fixed_rna.index, drugs, device=device)
        estimate(name, drugs, kernels["structured"], weight, response, anchor)

    drugs, kernels = drug_kernels(joint_rows, all_targets, device=device)
    active = torch.as_tensor(drugs.isin(clean.compound_id), device=device)
    same_panel = active[:, None] == active[None, :]
    kernel = kernels["structured"]
    weights, response = training_grid(joint_rows, fixed_rna.index, drugs, device=device)
    estimate(PRIMARY, drugs, kernel, weights, response, clean_anchor)
    estimate("expanded_generic_only", drugs, torch.where(same_panel, kernel, .10), weights, response, clean_anchor)
    parts["corrected_drug_mean"] = query.compound_id.map(clean_anchor).to_numpy()
    return parts, {
        "fit_core_drugs": int(clean.compound_id.nunique()), "fit_extended_drugs": len(drugs),
        "fit_core_rows": len(clean), "fit_extension_rows": len(extra),
        "extra_targets_mapped": new_known, "fit_rna_auxiliary_columns": aux_count, "solver": diagnostics,
    }


def predict(args):
    import torch
    from .v0plus_neural import load_auxiliary

    if not torch.cuda.is_available():
        raise RuntimeError("Expanded multitask kernels require the assigned G1 GPU")
    old, clean, rna, extension = registered_inputs(args, include_extension=True)
    history, train, check = same_split(old, clean, args.fold, args.assay)
    extra = extension[extension.cell_id.isin(train.cell_id) & extension.source_id.isin(train.source_id)]
    edges, targets, features = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_meta = load_auxiliary(args.auxiliary)
    query = validation_roster(check)
    method, report = structural_predictions(
        history, train, extra, query, rna, edges, targets, auxiliary, torch.device("cuda"),
    )
    ridge, _ = candidate_forecasts("linear", train, query, rna, edges, targets, selected=[RIDGE])
    if not ridge[KEY].equals(query[KEY]):
        raise ValueError("Corrected Ridge forecast reordered the common validation roster")
    method["corrected_ridge"] = ridge.predicted_value.to_numpy()
    method["expanded_ridge_average"] = .5 * (method[PRIMARY] + method["corrected_ridge"])
    if not all(np.isfinite(values).all() for values in method.values()):
        raise ValueError("Incomplete expanded model forecasts")
    raw = pd.concat([query[KEY].assign(method=name, predicted_value=method[name]) for name in ARMS])
    forecast = with_assay(raw, check, args.fold)
    freeze_table(forecast, args.out, provenance={
        "assay": args.assay, "fold": args.fold, "input_receipts": RECEIPTS, "scoring_rows": len(check),
        "hcc_scoring_rows": int(check.lineage_role.eq("adult_hcc").sum()), "biological_priors": features,
        "auxiliary_receipt": aux_meta, **report,
    })


def summarize_track(frame):
    ranking, cells = score(frame, endpoint_direction="higher", rank_only=True,
                           comparisons=[*CONTRASTS, (PRIMARY, "corrected_drug_mean")])
    sensitivity, _ = score_across_cells(frame)
    columns = {"spearman": "within_drug_spearman", "pearson": "within_drug_pearson"}
    by_track = (sensitivity.groupby(["track", "method"], as_index=False)[list(columns)].mean()
                .rename(columns=columns).merge(pd.DataFrame(ranking["scores"]), on=["track", "method"],
                                               validate="one_to_one"))
    output = {
        "metrics": by_track.fillna(np.nan).astype(object).where(pd.notna(by_track), None).to_dict("records"),
        "ranking_intervals": ranking["paired_contrasts"],
    }
    return output, cells, sensitivity


def score_mode(frame, adult_cells, affected):
    output, _, pan = summarize_track(frame)
    intervals = [{"candidate": a, "baseline": b, "ci95": paired_intervals(frame, a, b, repeats=900)}
                 for a, b in CONTRASTS]
    output["sensitivity_intervals"] = intervals
    hcc = frame[frame.cell_id.isin(adult_cells)]
    hcc_summary, _, _ = summarize_track(hcc)
    centered = fold_centered_correlations(hcc)
    hcc_summary["fold_centered"] = centered
    output["hcc"] = hcc_summary
    focus = pan[pan.compound_id.isin(affected) & pan.method.isin({"historical_core", "corrected_core", PRIMARY})]
    output["corrected_drugs"] = focus.astype(object).where(pd.notna(focus), None).to_dict("records")
    relevant = next(item["ci95"] for item in intervals if (item["candidate"], item["baseline"]) ==
                    (PRIMARY, "corrected_core"))
    margins = [row["ci95"][0] for row in output["ranking_intervals"]
               if row["candidate"] == PRIMARY and row["baseline"] == "corrected_core"
               and row["metric"] in ("spearman", "ndcg_at_5")]
    positive_ranks = [row["gain"] for row in output["ranking_intervals"]
                      if row["candidate"] == PRIMARY and row["baseline"] == "corrected_drug_mean"
                      and row["metric"] in ("spearman", "ndcg_at_5")]
    hcc_table = pd.DataFrame(centered["spearman"]).pivot(index="track", columns="method", values="mean")
    output["gate"] = {
        "both_sensitivity_lower_bounds_positive": min(interval[0] for interval in relevant.values()) > 0,
        "ranking_noninferior": len(margins) == 2 * frame.track.nunique() and min(margins) > -.005,
        "ranking_above_mean": len(positive_ranks) == 2 * frame.track.nunique() and min(positive_ranks) >= 0,
        "hcc_centered_spearman_nonnegative_change": bool(
            (hcc_table[PRIMARY] - hcc_table.corrected_core).mean() >= 0
        ),
    }
    output["gate_pass"] = all(output["gate"].values())
    return output


def evaluate(args):
    old, clean, _, _ = registered_inputs(args)
    affected = {value["compound_id"]: value["compound_label"] for value in
                json.loads((args.core / "receipt.json").read_text())["affected_source_compounds"]}
    adult = set(clean.loc[clean.lineage_role.eq("adult_hcc"), "cell_id"])
    scores, hashes, support = {}, {}, {}
    for assay in ASSAYS:
        parts = []
        for fold in range(3):
            file = args.forecasts / f"{assay}-{fold}.parquet"
            forecast, seal = verify_forecast(file)
            meta = seal["provenance"]
            _, _, check = same_split(old, clean, fold, assay)
            if (meta["assay"], meta["fold"], meta["input_receipts"]) != (assay, fold, RECEIPTS):
                raise ValueError("Unexpected split or data receipt in expansion freeze")
            keys = [*KEY, "track"]
            expected = set(observations(check).set_index(keys).index)
            if set(forecast.method) != set(ARMS):
                raise ValueError("A registered comparison method is missing")
            for _, subset in forecast.groupby("method"):
                if len(subset) != len(expected) or set(subset.set_index(keys).index) != expected:
                    raise ValueError("Methods do not share the registered corrected outcomes")
            parts.append(attach(forecast, check))
            hashes[file.stem] = seal["sha256"]
            support[file.stem] = meta
        scores[assay] = score_mode(pd.concat(parts, ignore_index=True), adult, affected)
    publish_json(args.out, {"modes": scores, "forecast_sha256": hashes, "support": support,
                            "corrected_compound_labels": affected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action, names in {
        "predict": ("extension", "priors", "hallmark", "targets", "auxiliary"),
        "score": ("forecasts",),
    }.items():
        child = commands.add_parser(action)
        for name in ("original", "core", "out", *names):
            child.add_argument("--" + name, type=Path, required=True)
        if action == "predict":
            child.add_argument("--assay", choices=ASSAYS, required=True)
            child.add_argument("--fold", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    {"predict": predict, "score": evaluate}[args.action](args)


if __name__ == "__main__":
    main()
