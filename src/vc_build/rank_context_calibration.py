"""Calibrate the patient and drug axes of two fixed RNA ranking contexts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import across_cell_metrics, rank_metrics
from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table, publish_json, sha256
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, check_component, external_inputs,
)
from .rna_gene_panel import PDO_AXIS as RERANK, PDO_CONTEXT as NATIVE, PDO_INPUT, schema_columns
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, observations, paired_intervals, with_assay
from .structured_kernel import KernelGrid, drug_kernels, fit_cell_kernels, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

ARCHIVED_SCORE = {
    "source": "807d762639135f5cbf77fbe32cca3d200bb1f2acbe5797cb101d513697bb275c",
    "external": "0f40b2729536052b9f3b0d45eed04f48d9cee045af1a71a2a9edaab4e727941a",
}
PRIMARY = "rank_context_calibrated"
UNSEPARATED = "rank_context_raw_mix"
EQUAL = "rank_context_equal"
FIXED = "rank_context_native_source_offset"
TRANSDUCTIVE = "rank_context_native_query_offset"
ARM_NAMES = (NATIVE, RERANK, PRIMARY, UNSEPARATED, EQUAL, FIXED, TRANSDUCTIVE)
PREDICTION_KEYS = [*KEY, "track", "fold"]
GRID = (0., .25, .5, .75, 1.)
DRUG_METRICS = ("spearman", "pearson")
SAMPLE_METRICS = ("spearman", "ndcg_at_5", "top5_overlap")


def archive_component(args, inputs_sha):
    external = args.mode in EXTERNAL_TRACKS
    tag = args.mode if external else f"{args.mode}-{args.fold}"
    stage = "external" if external else "source"
    path = args.archive / f"{stage}_score.json"
    require_hash(path, ARCHIVED_SCORE[stage])
    checksums = json.loads(path.read_text())["forecast_sha256"]
    values, digest = check_component(args.archive / "forecasts" / f"{tag}.parquet",
                                     args.mode, args.fold, inputs_sha)
    if checksums[tag] != digest:
        raise ValueError("Ranking context forecast changed after scoring")
    methods = (*BASELINES, NATIVE, RERANK, *PUBLIC) if external else (*BASELINES, NATIVE, RERANK)
    return values[values.method.isin(methods)], digest


def bridge_axis(args):
    manifest = args.pdo / "receipt.json"
    require_hash(manifest, PDO_INPUT)
    meta = json.loads(manifest.read_text())
    return schema_columns(args.pdo / "rna.parquet", meta["files"]["rna.parquet"])


def inner_groups(fit):
    """Use a deterministic, balanced assignment of fit-only lineage groups."""
    groups = sorted(fit.lineage_group.unique(),
                    key=lambda x: hashlib.sha256(f"10492:{x}".encode()).hexdigest())
    return [groups[i::2] for i in range(2)]


def fit_pair(train, query, rna, axis, edges, targets, auxiliary, device):
    _validate(train, query)
    x = rna.loc[sorted(train.cell_id.unique())].sort_index(axis=1)
    q = rna.loc[sorted(query.cell_id.unique()), x.columns]
    _, anchor, rows, _ = panel_targets(train)
    drugs, kernels = drug_kernels(rows, targets, device=device)
    w, y = training_grid(rows, x.index, drugs, device=device)
    base = query.compound_id.map(anchor).to_numpy()
    result = []
    for method, features in ((NATIVE, x), (RERANK, x[axis])):
        basis, gram = fit_cell_kernels(features, edges, auxiliary, device=device,
                                      genes=axis if method == NATIVE else None)
        projected = basis.cross(q)["full"]
        grid = KernelGrid(q.index, drugs, {"full": (gram["full"], projected)},
                          kernels, w, y, basis.reference[-1].shape[1])
        residual, _ = grid.forecast(query, "structured", PENALTY)
        result.append(query[KEY].assign(method=method, predicted_value=base+residual))
    return pd.concat(result, ignore_index=True)


def matrix(rows):
    values = rows[rows.method.isin([NATIVE, RERANK])]
    wide = values.pivot(index=PREDICTION_KEYS, columns="method", values="predicted_value").reset_index()
    if len(wide) * 2 != len(values) or not np.isfinite(wide[[NATIVE, RERANK]].to_numpy()).all():
        raise ValueError("The two rank contexts must cover every forecast row")
    return wide.assign(delta=wide[RERANK] - wide[NATIVE])


def inner_development(fit, rna, axis, edges, targets, auxiliary, device):
    splits, metadata = [], []
    for fold, groups in enumerate(inner_groups(fit)):
        train, val = fit[~fit.lineage_group.isin(groups)], fit[fit.lineage_group.isin(groups)]
        val = val[val.compound_id.isin(train.compound_id)]
        query = validation_roster(val)
        prior = fit_pair(train, query, rna, axis, edges, targets, auxiliary, device)
        panel = matrix(with_assay(prior, val, fold))
        y = observations(val)
        scored = panel.merge(y, on=[*KEY, "track"], validate="one_to_one")
        if len(scored) != len(panel) or len(scored) != len(y):
            raise ValueError("Inner held observations differ from their roster")
        splits.append(scored)
        metadata.append({
            "fit_lineages": int(train.lineage_group.nunique()),
            "held_lineages": int(scored.lineage_group.nunique()), "observations": len(scored),
            "held_lineage_sha256": hashlib.sha256(("\n".join(sorted(groups)) + "\n").encode()).hexdigest(),
            "compounds": int(scored.compound_id.nunique()), "assays": sorted(scored.track.unique()),
        })
    return pd.concat(splits, ignore_index=True), metadata


def sensitivity_weight(dev):
    """Choose a shared blend using worst-assay improvement on paired validation units."""
    scores = []
    for (track, _, drug), rows in dev.groupby(["track", "fold", "compound_id"], sort=True):
        if rows.lineage_group.nunique() < 5:
            continue
        for w in GRID:
            out = across_cell_metrics(rows.assign(predicted_value=rows[NATIVE] + w * rows.delta))
            scores.append({"track": track, "compound_id": drug, "weight": w, **out})
    frame = pd.DataFrame(scores).dropna(subset=list(DRUG_METRICS))
    by_drug = frame.groupby(["track", "compound_id", "weight"])[list(DRUG_METRICS)].mean()
    by_assay = by_drug.groupby(["track", "weight"]).mean()
    assay = by_assay.index.get_level_values("track").unique()
    trace = {}
    for w in GRID:
        changes = by_assay.xs(w, level="weight") - by_assay.xs(0., level="weight")
        trace[str(w)] = changes.reindex(assay).to_dict(orient="index")
    viable = [(min(m for ag in trace[str(w)].values() for m in ag.values()), w) for w in GRID[1:]]
    chosen = max((value for value in viable if value[0] > 0.), default=(0., 0.),
                 key=lambda pair: (pair[0], -pair[1]))[1]
    return chosen, trace


def per_sample_ranking(dev, values):
    parts = []
    columns = ["track", "fold", "lineage_group", "cell_id"]
    for key, block in dev.assign(calibrated=values).groupby(columns, sort=True):
        if len(block) < 6:
            continue
        metrics = rank_metrics(block.endpoint_value.to_numpy(), block.calibrated.to_numpy())
        parts.append(dict(zip(columns, key)) | metrics)
    return (pd.DataFrame(parts).groupby(["track", "lineage_group"])[list(SAMPLE_METRICS)].mean()
            .groupby(level="track").mean())


def learn_calibration(dev, fit_drugs, *, splits):
    w, trace = sensitivity_weight(dev)
    d = (dev.groupby(["compound_id", "lineage_group"]).delta.mean().groupby(level="compound_id").mean()
         .reindex(fit_drugs, fill_value=0.))
    shifts = dev.compound_id.map(d)
    mixed = dev[NATIVE] + w * dev.delta
    base = per_sample_ranking(dev, mixed)
    rank_trace, valid = {}, []
    for a in sorted({0., .5, 1., w}):
        compare = per_sample_ranking(dev, mixed + (a - w) * shifts)
        change = compare - base
        rank_trace[str(a)] = change.to_dict(orient="index")
        if (change[["spearman", "top5_overlap"]] >= -.005).all(axis=None):
            valid.append((float(change.ndcg_at_5.mean()), a))
    a = max(valid, key=lambda v: (v[0], -abs(v[1] - w), -v[1]))[1]
    return {"blend_weight": w, "drug_offset_weight": a, "delta_by_drug": d.to_dict(),
            "sensitivity_selection": trace, "ranking_selection": rank_trace, "splits": splits}


def calibrated(archive, settings):
    view = matrix(archive)
    d = view.compound_id.map(settings["delta_by_drug"])
    if d.isna().any():
        raise ValueError("A query compound is outside the source calibration table")
    # Also freeze an unlabeled, cohort-conditional offset as a diagnostic control.
    q = view.groupby(["track", "compound_id"]).delta.transform("mean")
    w, a = settings["blend_weight"], settings["drug_offset_weight"]
    main = view[NATIVE] + w * view.delta
    values = {
        PRIMARY: main + (a - w) * d,
        UNSEPARATED: main,
        EQUAL: view[NATIVE] + .5 * view.delta,
        FIXED: view[NATIVE] + d,
        TRANSDUCTIVE: view[NATIVE] + q,
    }
    return pd.concat([view[PREDICTION_KEYS].assign(method=name, predicted_value=x) for name, x in values.items()],
                     ignore_index=True)


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, receipt = external_inputs(args, args.mode)
    else:
        old, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(old, clean, args.fold, args.mode)
        query = validation_roster(check)
        receipt = RECEIPTS["core"]
    original, ref = archive_component(args, receipt)
    if set(fit.lineage_group) & set(query.lineage_group):
        raise ValueError("A query lineage cannot enter the source calibration")
    axis = rna.columns.intersection(bridge_axis(args)).sort_values()
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    dev, splits = inner_development(fit, rna, axis, edges, targets, auxiliary, device_cuda())
    settings = learn_calibration(dev, sorted(fit.compound_id.unique()), splits=splits)
    settings.update({"mode": args.mode, "fold": args.fold, "core_sha256": RECEIPTS["core"],
                     "archive_forecast_sha256": ref, "pdo_receipt_sha256": PDO_INPUT, "genes": len(axis)})
    publish_json(args.calibration_out, settings)
    variants = calibrated(original, settings)
    joined = pd.concat([original, variants], ignore_index=True)
    methods = (*BASELINES, *ARM_NAMES, *PUBLIC) if external else (*BASELINES, *ARM_NAMES)
    wide = joined.pivot(index=PREDICTION_KEYS, columns="method", values="predicted_value")
    if set(wide) != set(methods) or len(wide) != (len(query) if external else len(check)) or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Calibration changed the matched prediction support")
    if external:
        shared_support(joined, query, methods)
    freeze_table(joined[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": receipt, "core_sha256": RECEIPTS["core"],
        "reference_sha256": ref, "calibration_sha256": sha256(args.calibration_out),
        "priors": priors, "auxiliary": aux,
    })


def assessor(frame, *, external):
    peers = (*BASELINES, *PUBLIC) if external else BASELINES
    checks = (*peers, NATIVE, RERANK, EQUAL)
    pairs = [(PRIMARY, c) for c in checks]
    pairs += [(FIXED, NATIVE), (TRANSDUCTIVE, NATIVE), (PRIMARY, UNSEPARATED)]
    metrics, _, _ = axis_metrics(frame, pairs)
    interval = crossed_patient_drug_ci if external else paired_intervals
    across = {name: interval(frame, PRIMARY, name, repeats=1000 if external else 800) for name in checks}
    ranks = {(p["baseline"], p["metric"]): p["ci95"]
             for p in metrics["paired_contrasts"] if p["candidate"] == PRIMARY}
    gates = {}
    for name in checks:
        margin = -.005 if name in (NATIVE, EQUAL) else 0.
        gates[name] = all(across[name][m][0] > margin for m in DRUG_METRICS) and all(
            ranks[name, m][0] > (0. if name == NATIVE and m == "ndcg_at_5" else -.005) for m in SAMPLE_METRICS)
    return {"metrics": metrics, "across_cell_ci95": across, "gates": gates,
            "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}


def verify_calibrations(args, tags):
    for tag in tags:
        stamp = json.loads((args.forecasts / f"{tag}.parquet.freeze.json").read_text())
        require_hash(args.calibration / f"{tag}.json", stamp["provenance"]["calibration_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("archive", "pdo", "licob", "priors", "hallmark", "targets", "auxiliary", "calibration-out"),
        "source": ("forecasts", "calibration"),
        "external": ("forecasts", "calibration", "licob", "limore-labels", "licob-labels"),
    }
    for name, keys in fields.items():
        p=stages.add_parser(name)
        for key in ("original","core",*keys,"out"):
            p.add_argument("--" + key, type=Path, required=True)
        if name == "predict":
            p.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            p.add_argument("--fold", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        verify_calibrations(args, (f"{mode}-{fold}" for mode in ASSAYS for fold in range(3)))
        score_source(args, assessor=assessor)
    else:
        verify_calibrations(args, EXTERNAL_TRACKS)
        score_external(args, assessor=assessor, arms=ARM_NAMES)


if __name__ == "__main__":
    main()
