"""Register a separate drug screen and evaluate three- or four-source transfer."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from .cross_source_data import MODEL_COLS, gdsc_rows, read_raw, response_rows, roster_maps, roster_table
from .expanded_panel_cv import PENALTY, RECEIPTS, RIDGE, registered_inputs, same_split
from .external import KEY, SOURCE, align_endpoints
from .function_data import alias_token, new_output
from .hcc_conditional_cv import fold_centered_correlations
from .improve_gdsc_registry import DRUG_INFO_SHA256, matched_structures
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import checked as checked_cold
from .licob_dose_sources import COLD
from .pdo_transfer_cv import device_cuda
from .replicated_features import CORE, frozen_baseline
from .structured_cv import ASSAYS, attach, observations, paired_intervals, with_assay
from .v0plus_cv import candidate_forecasts, molecular_inputs, panel_targets, validation_roster
from .v0plus_score import axis_metrics, compound_strata
from .yang_hcc_external import require_hash

ASSAY = "GDSC1"
ADDED = ("gdsc1_equal", "gdsc1_quarter")
BASE = "corrected_core"
ANCHOR = "corrected_drug_mean"
TRACKS = (*ASSAYS, ASSAY)
TKIS = {"Cabozantinib", "Lenvatinib", "Regorafenib", "Sorafenib"}


def catalogue(core, official, raw, published, trusted_parents):
    ids = set(raw.DRUG_ID)
    names = (official.assign(aliases=lambda m: m.DRUG_NAME + "," + m.SYNONYMS)
             .groupby("DRUG_ID").aliases.agg(",".join).str.split(",")
             .map(lambda values: set(map(alias_token, values)) - {""}))
    raw_names = raw[["DRUG_ID", "DRUG_NAME"]].drop_duplicates().groupby("DRUG_ID").DRUG_NAME.agg(set)
    verified_labels = {native for native in ids
                       if set(map(alias_token, raw_names[native])) <= names.get(native, set())}
    matched, audit = matched_structures(published, official, ids & verified_labels)
    known = (core[core.source_id.eq("GDSC2")][["native_compound_id", "compound_id", "isomeric_smiles"]]
             .drop_duplicates().query("native_compound_id in @verified_labels"))
    if not known.native_compound_id.is_unique:
        raise ValueError("Ambiguous existing GDSC2 native identity")
    paired = known.merge(matched, on="native_compound_id", suffixes=("_known", "_publisher"))
    disagreement = paired[paired.compound_id_known.ne(paired.compound_id_publisher)]
    fresh = matched[~matched.native_compound_id.isin(known.native_compound_id)].copy()
    missing_stereo = fresh.isomeric_smiles.map(
        lambda smiles: any(item.specified != Chem.StereoSpecified.Specified
                           for item in Chem.FindPotentialStereo(Chem.MolFromSmiles(smiles))))
    fixed = pd.concat([known.assign(evidence="shared_verified_gdsc_native_id"),
                       fresh.loc[~missing_stereo].assign(evidence="improve_exact_structure")],
                      ignore_index=True)
    stable = fixed[fixed.compound_id.isin(trusted_parents)].copy()
    stable["source_compound_label"] = stable.native_compound_id.map(
        official.groupby("DRUG_ID").DRUG_NAME.first())
    summary = audit | {"source_ids": len(ids), "official_labels_agree": len(verified_labels),
                       "shared_verified_ids": len(known), "shared_ids_with_publisher_conflict": len(disagreement),
                       "excluded_unassigned_stereo": int(missing_stereo.sum()),
                       "bridged_native_ids": len(stable)}
    return stable, summary


def register(args):
    _, core, _, _ = registered_inputs(args)
    digest = checked_cold(args.cold, ("train",))
    if digest != COLD:
        raise ValueError("Unexpected expanded reference registry")
    require_hash(args.drug_info, DRUG_INFO_SHA256)
    evidence = json.loads((args.original / "receipt.json").read_text())["inputs"]
    ref = {name: Path(evidence[name]["path"]) for name in ("gdsc1", "gdsc_labels", "depmap_models")}
    for name, path in ref.items():
        require_hash(path, evidence[name]["sha256"])
    raw = read_raw(ref["gdsc1"], ["DRUG_ID", "DRUG_NAME", "SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME"])
    official = read_raw(ref["gdsc_labels"], ["DRUG_ID", "DRUG_NAME", "SYNONYMS", "TARGET"])
    published = pd.read_csv(args.drug_info, sep="\t", dtype=str, keep_default_na=False)
    trusted = set(pd.read_parquet(args.cold / "train.parquet", columns=["compound_id"]).compound_id)
    chemistry, audit = catalogue(core, official, raw, published, trusted)
    published_cells = read_raw(ref["depmap_models"], MODEL_COLS)
    ids = published_cells.set_index("ModelID", verify_integrity=True).SangerModelID
    lookup = roster_maps(roster_table(raw, ASSAY, "CELL_LINE_NAME", "SANGER_MODEL_ID", "COSMIC_ID"),
                         published_cells)
    lookup = lookup[lookup.cell_id.isin(core.cell_id) & lookup.source_key.ne("")
                    & lookup.cell_id.map(ids).eq(lookup.source_key)]
    raw_rows = gdsc_rows(ref["gdsc1"], lookup, chemistry[["native_compound_id", "compound_id"]])
    raw_rows["source_run"] = raw_rows.source_run.str.replace("GDSC2_", "GDSC1_", regex=False)
    single, _ = response_rows(raw_rows, ASSAY, "Syto60_or_resazurin_72h", 72, "Sanger_MGH_8.5_fitted_curve",
                              ["dose_min_uM", "dose_max_uM"])
    cells = core[["cell_id", "lineage_group", "lineage_role"]].drop_duplicates().set_index("cell_id",
                                                                                      verify_integrity=True)
    labels = chemistry[["native_compound_id", "compound_id", "isomeric_smiles", "evidence", "source_compound_label"]]
    targets = official.groupby("DRUG_ID").TARGET.first().rename("drug_target_source_labels")
    rows = (single.join(cells, on="cell_id", validate="many_to_one")
            .merge(labels, on=["native_compound_id", "compound_id"], validate="many_to_one")
            .join(targets, on="native_compound_id", validate="many_to_one"))
    rows = rows.assign(row_id=rows.source_id + "|" + rows.cell_id + "|" + rows.compound_id + "|" + rows.endpoint_id)
    rows = rows.assign(in_core=rows.compound_id.isin(core.compound_id))
    if not rows.row_id.is_unique or not np.isfinite(rows.endpoint_value).all():
        raise ValueError("GDSC1 must have unique finite supported responses")
    old = rows[rows.in_core]
    adult = old[old.lineage_role.eq("adult_hcc")]
    if old.compound_id.nunique() < 10 or len(adult.lineage_group.unique()) < 5:
        raise ValueError("The exact new assay does not support the registered assessment")
    parent_names = (core[["compound_id", "source_compound_label"]].drop_duplicates("compound_id")
                    .set_index("compound_id").source_compound_label)
    tki = rows[rows.compound_id.map(parent_names).isin(TKIS)]
    with new_output(args.out) as work:
        rows.sort_values("row_id").to_parquet(work / "observations.parquet", index=False)
        publish_json(work / "receipt.json", {
            "core_sha256": RECEIPTS["core"], "expanded_sha256": COLD, "structure_catalogue_sha256": DRUG_INFO_SHA256,
            "official_data": {name: evidence[name] for name in ref},
            "observation_sha256": sha256(work / "observations.parquet"),
            "chemistry": summary_json(audit), "all": data_counts(rows), "core": data_counts(old),
            "core_adult_hcc": data_counts(adult), "tki": {
                parent_names[key]: data_counts(g) for key, g in tki.groupby("compound_id")
            },
        })


def summary_json(audit):
    return {key: (sorted(value) if isinstance(value, set) else value) for key, value in audit.items()}


def data_counts(rows):
    return {"rows": len(rows), "compounds": rows.compound_id.nunique(),
            "lineages": rows.lineage_group.nunique(), "cells": rows.cell_id.nunique()}


def read_registration(path):
    proof = json.loads((path / "receipt.json").read_text())
    if (proof["core_sha256"], proof["expanded_sha256"]) != (RECEIPTS["core"], COLD):
        raise ValueError("Unexpected GDSC1 register")
    target = path / "observations.parquet"
    require_hash(target, proof["observation_sha256"])
    return pd.read_parquet(target), sha256(path / "receipt.json")


def split_bridge(history, clean, added, assay, fold):
    _, fit, check = same_split(history, clean, fold, "pooled" if assay == ASSAY else assay)
    added = added[added.in_core & added.compound_id.isin(fit.compound_id)]
    new_fit = added[added.lineage_group.isin(fit.lineage_group)]
    if assay == ASSAY:
        check = added[added.lineage_group.isin(check.lineage_group)]
        new_fit = new_fit.iloc[:0]
    if set(check.lineage_group) & (set(fit.lineage_group) | set(new_fit.lineage_group)):
        raise ValueError("Cross-screen train and test lineages overlap")
    return fit, new_fit, check


def model_values(fit, extra, roster, rna, edges, targets, aux, assay):
    import torch
    from .structured_kernel import KernelGrid, solve_residuals, training_grid

    device = device_cuda()
    _, anchor, native, calibration = panel_targets(fit)
    ref = calibration[0]["reference_source"], calibration[0]["reference_endpoint"]
    params = next(entry for entry in calibration if entry["source_id"] == ref[0]
                  and entry["endpoint_id"] == ref[1])
    if params["slope"] != 1:
        raise ValueError("Unexpected reference choice")
    fit_axis = rna.loc[sorted(fit.cell_id.unique())]
    query_axis = rna.loc[sorted(roster.cell_id.unique())]
    grid = KernelGrid.prepare(native, fit_axis, query_axis, edges, targets, aux, device=device)
    anchors = roster.compound_id.map(anchor).to_numpy()
    if assay == ASSAY:
        values, diagnostics = grid.forecast(roster, "structured", PENALTY)
        baseline = candidate_forecasts("linear", fit, roster, rna, edges, targets, selected=[RIDGE])[0]
        return {BASE: anchors + values, ANCHOR: anchors}, baseline.assign(method="corrected_ridge"), diagnostics
    if not set(extra.cell_id) <= set(fit_axis.index):
        # The previous arrays already use the exact fit-lineage support. Map additional measured
        # cell models on those lineages only by requiring an observed source-fit model.
        extra = extra[extra.cell_id.isin(fit_axis.index)]
    jointly, probe = align_endpoints(pd.concat([fit, extra], ignore_index=True), reference=ref)
    new = jointly[jointly.source_id.eq(ASSAY)].copy()
    new["residual"] = new.model_value - new.groupby([*SOURCE, "compound_id"]).model_value.transform("mean")
    augmented = pd.concat([native, new], ignore_index=True)
    cell, cross = grid.cell["full"]
    drug = grid.drug["structured"]
    ix = query_axis.index.get_indexer(roster.cell_id), grid.drugs.get_indexer(roster.compound_id)
    estimates, solve = {}, {}
    for name, gain in zip(ADDED, (1., .25)):
        weights, residual = training_grid(augmented, fit_axis.index, grid.drugs, device=device,
                                          row_weight=np.where(augmented.source_id.eq(ASSAY), gain, 1.))
        coef, steps, norm = solve_residuals(cell, drug, weights, residual, PENALTY)
        estimates[name] = anchors + (cross @ coef @ drug).cpu().numpy()[ix]
        solve[name] = {"iterations": steps, "residual_norm": norm}
    return estimates, None, {"gdsc1": next(row for row in probe if row["source_id"] == ASSAY),
                             "added_train_rows": len(new), "added_drugs": sorted(new.compound_id.unique()),
                             "solves": solve}


def predict(args):
    from .v0plus_neural import load_auxiliary

    old, clean, rna, _ = registered_inputs(args)
    registered, digest = read_registration(args.registration)
    fit, extra, outcomes = split_bridge(old, clean, registered, args.assay, args.fold)
    check = outcomes[[*KEY, *SOURCE, "isomeric_smiles"]]
    roster = validation_roster(check)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    aux, aux_receipt = load_auxiliary(args.auxiliary)
    values, ridge, diagnostics = model_values(fit, extra, roster, rna, edges, targets, aux, args.assay)
    predictions = [roster[KEY].assign(method=name, predicted_value=result) for name, result in values.items()]
    if ridge is not None:
        predictions.append(ridge)
    fresh = with_assay(pd.concat(predictions, ignore_index=True), check, args.fold)
    if args.assay != ASSAY:
        earlier, earlier_sha = frozen_baseline(args.baseline, args.assay, args.fold)
        fresh = pd.concat([fresh, earlier], ignore_index=True)
    else:
        earlier_sha = None
    if not np.isfinite(fresh.predicted_value).all():
        raise ValueError("Unexpected nonfinite forecast")
    freeze_table(fresh, args.out, provenance={"assay": args.assay, "fold": args.fold,
        "registration_sha256": digest, "core_sha256": RECEIPTS["core"], "reference_forecast_sha256": earlier_sha,
        "fit_count": len(fit), "query_count": len(check), "fit_sources": sorted(fit.source_id.unique()),
        "details": diagnostics, "priors": priors, "auxiliary": aux_receipt})


def summary(data, comparisons):
    metrics, _, _ = axis_metrics(data, comparisons)
    intervals = {f"{a}/{b}": paired_intervals(data, a, b) for a, b in comparisons}
    return {"metrics": metrics, "drug_ci95": intervals, "data": data_counts(data.drop_duplicates([*KEY, "track"]))}


def supported_summary(data):
    data = data.drop(columns="supported")
    _, by_drug, _ = axis_metrics(data, ())
    return (
        by_drug.groupby(["track", "method"])[["spearman", "pearson"]].mean()
        .dropna().reset_index().to_dict("records")
    )


def evaluate(args):
    old, clean, _, _ = registered_inputs(args)
    added, digest = read_registration(args.registration)
    adult = set(clean.loc[clean.lineage_role.eq("adult_hcc"), "cell_id"])
    tki = compound_strata(clean.drop_duplicates("compound_id"))["hcc_tki"]
    reports, hashes, provenance = {}, {}, {}
    for assay in TRACKS:
        pieces = []
        methods = CORE if assay == ASSAY else (*CORE, *ADDED)
        for fold in range(3):
            path = args.forecasts / f"{assay}-{fold}.parquet"
            rows, seal = verify_forecast(path)
            proof = seal["provenance"]
            if (proof["assay"], proof["fold"], proof["registration_sha256"], proof["core_sha256"]) != (
                    assay, fold, digest, RECEIPTS["core"]):
                raise ValueError("Unexpected frozen GDSC1 study partition")
            _, _, truth = split_bridge(old, clean, added, assay, fold)
            keys = [*KEY, "track"]
            expected = observations(truth).set_index(keys).sort_index().index
            if set(rows.method) != set(methods) or any(
                    not part.set_index(keys).sort_index().index.equals(expected)
                    for _, part in rows.groupby("method")):
                raise ValueError("Mismatched method identities")
            joined = attach(rows, truth)
            pieces.append(joined.assign(supported=joined.compound_id.isin(proof["details"].get("added_drugs", ()))))
            hashes[path.stem], provenance[path.stem] = seal["sha256"], proof
        frame = pd.concat(pieces, ignore_index=True)
        contrasts = ((BASE, "corrected_ridge"), (BASE, ANCHOR)) if assay == ASSAY else tuple(
            (a, b) for a in ADDED for b in (BASE, ANCHOR))
        result = summary(frame, contrasts)
        result["adult_hcc"] = fold_centered_correlations(frame[frame.cell_id.isin(adult)])
        result["adult_hcc_tki"] = fold_centered_correlations(frame[frame.cell_id.isin(adult)
                                                                 & frame.compound_id.isin(tki)])
        if assay != ASSAY:
            result["added_support"] = supported_summary(frame[frame.supported])
            rank_lows = [x["ci95"][0] for x in result["metrics"]["paired_contrasts"]
                         if x["candidate"] == ADDED[0] and x["baseline"] == BASE
                         and x["metric"] in ("spearman", "ndcg_at_5")]
            stats = pd.DataFrame(result["metrics"]["scores"]).set_index(["method", "track"])
            measures = ["macro_spearman", "macro_ndcg_at_5"]
            result["gate"] = {
                "within_drug": all(low > 0 for low, _ in result["drug_ci95"][f"{ADDED[0]}/{BASE}"].values()),
                "within_cell_noninferior": (len(rank_lows) == 2 * frame.track.nunique() and min(rank_lows) > -.005),
                "within_cell_at_least_mean": bool((stats.loc[ADDED[0], measures] >= stats.loc[ANCHOR, measures]).all(axis=None)),
            }
        reports[assay] = result
    passed = all(all(reports[assay]["gate"].values()) for assay in ("CTRPv2", "PRISM20Q2"))
    publish_json(args.out, {"input_sha256": digest, "prediction_sha256": hashes, "provenance": provenance,
                           "reports": reports, "promoted": passed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    fields = {"register": ("cold", "drug-info"), "predict": ("registration", "baseline", "priors", "hallmark",
                                                            "targets", "auxiliary"),
              "score": ("registration", "forecasts")}
    for stage, params in fields.items():
        task = commands.add_parser(stage)
        for name in ("original", "core", "out", *params):
            task.add_argument("--" + name, required=True, type=Path)
        if stage == "predict":
            task.add_argument("--assay", choices=TRACKS, required=True)
            task.add_argument("--fold", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    {"register": register, "predict": predict, "score": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
