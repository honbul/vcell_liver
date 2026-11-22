"""Compare repeat-weighted PRISM dose summaries on equal training support."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .expanded_panel_cv import expanded_targets
from .external import KEY, SOURCE
from .hcc_conditional_cv import fold_centered_correlations
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import checked
from .licob_node_recovery import joined_outcomes
from .licob_prism_restored import OLD_INPUT, RANK_METRICS
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .structured_cv import attach, paired_intervals, with_assay
from .structured_kernel import cell_kernels, drug_kernels, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, training_inputs, validation_roster
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .vinorelbine_raw_doses import MANIFEST, prism_curves
from .vinorelbine_transfer import EXTENSION_SHA
from .yang_hcc_external import require_hash


PRISM = "PRISM20Q2"
ARMS = ("drug_mean", "fitted_auc", "equal_raw", "reliable_raw")
PRIMARY = ARMS[-1]
ASSAYS = ("CTRPv2", "GDSC2")
CASES = ("licob", *ASSAYS)
PAIR = ["compound_id", "cell_id"]


def prior(area, files):
    if checked(area, files) != OLD_INPUT:
        raise ValueError("Unexpected LICOB historical register")


def register(args):
    prior(args.historical, ("train", "query"))
    core, _, _ = training_inputs(args.core)
    fit = pd.read_parquet(args.historical / "train.parquet")
    query = pd.read_parquet(args.historical / "query.parquet")
    receipt = args.extension / "receipt.json"
    require_hash(receipt, EXTENSION_SHA)
    extra_path = args.extension / "additional_rows.parquet"
    require_hash(extra_path, json.loads(receipt.read_text())["artifacts"][extra_path.name])
    extra = pd.read_parquet(extra_path, filters=[("source_id", "==", PRISM)])
    exact = set(query.loc[query.remaining_unspecified_stereo.eq(0), "compound_id"])
    existing = fit[fit.source_id.eq(PRISM)]
    compounds = exact & (set(existing.compound_id) | set(extra.compound_id))
    if len(compounds) != 24:
        raise ValueError(f"Expected the 24 native exact PRISM parents, got {len(compounds)}")
    catalog = pd.concat([core[core.source_id.eq(PRISM)], extra], ignore_index=True)
    key = ["row_id", *PAIR]
    catalogue = catalog[catalog.compound_id.isin(compounds) & catalog.cell_id.isin(fit.cell_id)]
    identity = catalogue[["row_id", "native_compound_id", "source_run"]]
    meta = existing[existing.compound_id.isin(compounds)].merge(identity, on="row_id", validate="one_to_one")
    if len(meta) != existing.compound_id.isin(compounds).sum():
        raise ValueError("Missing registered PRISM plate protocol")
    new = catalogue[~catalogue.row_id.isin(fit.row_id)].assign(core_rows=False)
    if not new.row_id.is_unique:
        raise ValueError("Duplicate newly registered PRISM observations")
    raw_roster = pd.concat([meta, new], ignore_index=True)
    if raw_roster.duplicated(PAIR).any():
        raise ValueError("More than one protocol per PRISM cell and parent")
    require_hash(args.raw / "inputs.json", MANIFEST)
    for item in json.loads((args.raw / "inputs.json").read_text()):
        require_hash(args.raw / item["file"], item["sha256"])
    wells, support = prism_curves(raw_roster, args.raw)
    ordered = wells.sort_values("detection_plate")
    indices = [*PAIR, "dose"]
    grouped = ordered.groupby(indices).log_fold_change
    dose = grouped.agg(["size", "median", "first"]).rename(columns={"first": "a"})
    second = ordered[ordered.groupby(indices).cumcount().eq(1)].set_index(indices)
    dose["b"] = second.log_fold_change
    dose = dose[dose["size"].ge(2)].reset_index()
    support = support[support["nunique"].ge(6)]
    per_pair = dose.groupby(PAIR, as_index=False).dose.nunique()
    per_pair = per_pair[per_pair.dose.eq(per_pair.compound_id.map(support["nunique"]))]
    adequate = per_pair.groupby("compound_id").size().loc[lambda n: n.ge(100)].index
    per_pair = per_pair[per_pair.compound_id.isin(adequate)][PAIR]
    if len(adequate) < 8:
        raise ValueError("Too few compounds with 100 complete curves")
    complete = dose.merge(per_pair, on=PAIR, validate="many_to_one")
    complete[["median", "a", "b"]] = np.clip(np.exp2(complete[["median", "a", "b"]]), 0, 1)
    complete = complete.merge(raw_roster[[*PAIR, "lineage_group"]], on=PAIR, validate="many_to_one")
    fit = pd.concat([fit, new[fit.columns]], ignore_index=True)
    candidate = fit.source_id.eq(PRISM) & fit.compound_id.isin(adequate)
    allowed = fit.loc[candidate, key].merge(per_pair, on=PAIR, validate="one_to_one").row_id
    fit = fit[~candidate | fit.row_id.isin(allowed)].copy()
    if not fit.row_id.is_unique or fit.loc[fit.row_id.isin(allowed), ["endpoint_value"]].isna().any().any():
        raise ValueError("Invalid same-support fit")
    roles = core[["cell_id", "lineage_role"]].drop_duplicates().set_index("cell_id", verify_integrity=True)
    fit = fit.join(roles, on="cell_id", validate="many_to_one")
    if fit.lineage_role.isna().any():
        raise ValueError("Missing original training lineage role")
    data = args.run / "inputs"
    data.mkdir(exist_ok=False)
    files = {"wells": complete, "fit": fit}
    for name, table in files.items():
        table.to_parquet(data / (name + ".parquet"), index=False)
    publish_json(data / "receipt.json", {
        "historical": OLD_INPUT, "extension": EXTENSION_SHA, "raw": MANIFEST,
        "planned": sorted(compounds), "eligible": sorted(adequate),
        "fit_rows": len(fit), "unpooled_complete_cells": per_pair.groupby("compound_id").size().to_dict(),
        "retained_complete_points": len(complete), "new_registered_rows": len(new),
        "removed_incomplete_prism_rows": int(candidate.sum()) - len(allowed),
        "files": {name: sha256(data / (name + ".parquet")) for name in files},
    })


def load(root, keys):
    path = root / "inputs"
    card = path / "receipt.json"
    proof = json.loads(card.read_text())
    if (proof["historical"], proof["extension"], proof["raw"]) != (OLD_INPUT, EXTENSION_SHA, MANIFEST):
        raise ValueError("Unexpected registered assay inputs")
    result = []
    for name in keys:
        artifact = path / (name + ".parquet")
        require_hash(artifact, proof["files"][name])
        result.append(pd.read_parquet(artifact))
    return result, proof, sha256(card)


def grouped_cases(assay, data, query, eligible):
    if assay == "licob":
        return [(data, query[query.compound_id.isin(eligible)], None)]
    roster = data[["cell_id", "lineage_group"]].drop_duplicates().sort_values("cell_id")
    check = data[data.source_id.eq(assay) & data.compound_id.isin(eligible)]
    count = check.groupby("compound_id").lineage_group.nunique()
    check = check[check.compound_id.isin(count[count.ge(30)].index)]
    if check.compound_id.nunique() < 5:
        raise ValueError(f"{assay} has fewer than five compounds with 30 lineages")
    slices = []
    for train, held in GroupKFold(3).split(roster, groups=roster.lineage_group):
        fit = data[data.cell_id.isin(roster.iloc[train].cell_id) & data.source_id.ne(assay)]
        selected = check[check.cell_id.isin(roster.iloc[held].cell_id) & check.compound_id.isin(fit.compound_id)]
        slices.append((fit, validation_roster(selected), selected))
    return slices


def dose_labels(wells, rows):
    target = wells.merge(rows[PAIR], on=PAIR, validate="many_to_one")
    summary, metadata = [], {}
    for drug, block in target.groupby("compound_id"):
        grid = np.sort(block.dose.unique())
        intervals = np.diff(np.log(grid))
        weights = .5 * (np.r_[0., intervals] + np.r_[intervals, 0.])
        repeat = []
        for _, dose in block.groupby("dose", sort=True):
            units = dose.groupby("lineage_group")[["a", "b"]].mean()
            if len(units) < 30:
                raise ValueError(f"Too few fit lineage groups for reliability: {drug}")
            rho = units.a.corr(units.b, method="spearman") if units.nunique().gt(1).all() else 0.
            repeat.append(float(rho))
        reliable = weights * (.05 + np.maximum(repeat, 0.) ** 2)
        matrix = block.pivot(index="cell_id", columns="dose", values="median").reindex(columns=grid)
        if matrix.isna().any().any():
            raise ValueError("Incompletely registered dose axis")
        raw, weighted = weights / weights.sum(), reliable / reliable.sum()
        summary.append(pd.DataFrame({"cell_id": matrix.index, "compound_id": drug,
                                     "equal_raw": matrix.to_numpy() @ raw,
                                     PRIMARY: matrix.to_numpy() @ weighted}))
        metadata[drug] = {"rho": repeat, "doses": grid.tolist(),
                          "equal": raw.tolist(), "reliable": weighted.tolist(), "fit_cells": len(matrix)}
    return pd.concat(summary, ignore_index=True), metadata


def structural_fit(fit, query, rna, eligible, wells, edges, base_targets, auxiliary, device):
    if set(fit.lineage_group) & set(query.lineage_group) or set(fit.cell_id) & set(query.cell_id):
        raise ValueError("Fit and evaluation lineages overlap")
    _, anchor, responses, calibration = panel_targets(fit)
    reference, _ = expanded_targets(fit[~fit.compound_id.isin(base_targets)], base_targets, rna.columns)
    selected = responses.source_id.eq(PRISM) & responses.compound_id.isin(eligible)
    replacement, repeat = dose_labels(wells, responses.loc[selected])
    raw = responses.loc[selected, [*PAIR, *SOURCE, "residual"]].merge(replacement, on=PAIR,
                                                                      validate="one_to_one")
    slope = raw.merge(pd.DataFrame(calibration)[[*SOURCE, "slope"]], on=SOURCE, validate="many_to_one").slope
    if len(raw) != int(selected.sum()) or not np.isfinite(slope).all() or not slope.gt(0).all():
        raise ValueError("Incomplete calibration or raw curve join")
    # Preserve exact response order through the merge before assigning fitted targets.
    if not raw[PAIR].reset_index(drop=True).equals(responses.loc[selected, PAIR].reset_index(drop=True)):
        raise ValueError("Raw targets reordered the registered PRISM rows")
    by_compound = raw.compound_id
    fitted_scale = raw.groupby(by_compound).residual.transform("std")
    fit_x, test_x = (rna.loc[sorted(table.cell_id.unique())] for table in (responses, query))
    terms, _ = cell_kernels(fit_x, test_x, edges, auxiliary, device=device)
    cell, cross = terms["full"]
    drugs, kernels = drug_kernels(responses.drop_duplicates("compound_id"), reference, device=device)
    chemical = kernels["structured"]
    ri, ci = test_x.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    base = query.compound_id.map(anchor).to_numpy()
    if (ri < 0).any() or (ci < 0).any() or not np.isfinite(base).all():
        raise ValueError("Query outside the known compound fit")
    results = [query[KEY].assign(method="drug_mean", predicted_value=base)]
    solver = {}
    for name in ARMS[1:]:
        fit_rows = responses.copy()
        if name != "fitted_auc":
            y = -raw[name]
            groups = y.groupby(by_compound)
            fitted_ratio = fitted_scale / groups.transform("std").clip(lower=.02)
            factor = (fitted_ratio / slope).clip(.25, 4.) * slope
            fit_rows.loc[selected, "residual"] = ((y - groups.transform("mean")) * factor).to_numpy()
        weights, response = training_grid(fit_rows, fit_x.index, drugs, device=device)
        coef, steps, error = solve_residuals(cell, chemical, weights, response, 4.)
        offset = (cross @ coef @ chemical).cpu().numpy()[ri, ci]
        results.append(query[KEY].assign(method=name, predicted_value=base + offset))
        solver[name] = {"steps": steps, "residual": error}
    return pd.concat(results, ignore_index=True), {
        "repeat": repeat, "solver": solver, "fit_rows": len(fit), "fit_lineages": fit.lineage_group.nunique(),
        "prism_raw_rows": len(raw), "fit_only_calibration": calibration,
    }


def predict(args):
    (data, wells), support, receipt = load(args.run, ("fit", "wells"))
    prior(args.historical, ("rna", "query"))
    rna = pd.read_parquet(args.historical / "rna.parquet")
    query = pd.read_parquet(args.historical / "query.parquet")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    output, diagnostics = [], {}
    for fold, (fit, check, assay) in enumerate(grouped_cases(args.case, data, query, support["eligible"])):
        prediction, details = structural_fit(fit, check, rna, support["eligible"], wells, edges, targets, auxiliary,
                                              device_cuda())
        shared_support(prediction, check, ARMS)
        output.append(with_assay(prediction, assay, fold) if assay is not None
                      else prediction.assign(track="licob", fold=fold))
        diagnostics[str(fold)] = details
    freeze_table(pd.concat(output, ignore_index=True), args.run / f"{args.case}.parquet", provenance={
        "registration_sha256": receipt, "case": args.case, "diagnostics": diagnostics,
        "priors": priors, "auxiliary": aux["endpoints"],
    })


def summary(table, labels, *, external):
    pairs = [(PRIMARY, baseline) for baseline in ARMS[1:-1]]
    report, drugs, _ = axis_metrics(table, pairs)
    ci = crossed_patient_drug_ci if external else paired_intervals
    compared = {base: ci(table, PRIMARY, base) for _, base in pairs}
    gate = {}
    for _, base in pairs:
        ranks = {row["metric"]: row["ci95"][0] for row in report["paired_contrasts"]
                 if row["candidate"] == PRIMARY and row["baseline"] == base}
        gate[base] = (all(value[0] > 0 for value in compared[base].values())
                      and all(ranks.get(metric, -1.) > -.005 for metric in RANK_METRICS))
    strata = {}
    for name, ids in compound_strata(labels).items():
        data = drugs[drugs.compound_id.isin(ids)]
        if not data.empty:
            strata[name] = data.groupby("method")[["spearman", "pearson"]].mean().to_dict("index")
    return {"metrics": report, "ci95": compared, "gate": gate, "strata": strata, "pass": all(gate.values()),
            "drugs": table.compound_id.nunique(), "lineages": table.lineage_group.nunique(),
            "matched_observations": table.row_id.nunique(),
            "per_drug": drugs.astype(object).where(drugs.notna(), None).to_dict("records")}


def score(args):
    prior(args.historical, ("query",))
    query = pd.read_parquet(args.historical / "query.parquet")
    (fit,), support, receipt = load(args.run, ("fit",))
    results, hashes = {}, {}
    for name in CASES:
        rows, seal = verify_forecast(args.run / f"{name}.parquet")
        if seal["provenance"]["registration_sha256"] != receipt or seal["provenance"]["case"] != name:
            raise ValueError("Prediction registration changed")
        hashes[name] = seal["sha256"]
        if name == "licob":
            sub = query[query.compound_id.isin(support["eligible"])]
            require_hash(args.supplement, ASSET_HASHES["supplement"])
            shared_support(rows, sub, ARMS)
            metrics = adult_donor_outcomes(joined_outcomes(sub, rows, args.supplement))
            results[name] = summary(metrics, sub, external=True)
        else:
            parts = [attach(rows[rows.fold.eq(i)], labels)
                     for i, (_, _, labels) in enumerate(grouped_cases(name, fit, query, support["eligible"]))]
            data = pd.concat(parts, ignore_index=True)
            # Score assay-only labels, and separately describe adult HCC transfer.
            results[name] = summary(data, query, external=False)
            adult = data[data.cell_id.isin(fit.loc[fit.lineage_role.eq("adult_hcc"), "cell_id"])]
            results[name]["adult_hcc"] = {
                "lineages": adult.lineage_group.nunique(), "fold_centered": fold_centered_correlations(adult),
            }
    publish_json(args.run / "scores.json", {"registration": receipt, "forecast_sha256": hashes, "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name, task, keys in (("register", register, ("core", "extension", "raw")),
                             ("predict", predict, ("priors", "hallmark", "targets", "auxiliary")),
                             ("score", score, ("supplement",))):
        command = modes.add_parser(name)
        for key in ("run", "historical", *keys):
            command.add_argument("--" + key, type=Path, required=True)
        if name == "predict":
            command.add_argument("--case", choices=CASES, required=True)
        command.set_defaults(task=task)
    args = parser.parse_args()
    args.task(args)


if __name__ == "__main__":
    main()
