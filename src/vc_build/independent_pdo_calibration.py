"""Estimate a fixed PDO offset independently of held samples and their outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import across_cell_metrics
from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY
from .io import freeze_table, publish_json, sha256
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .rank_context_calibration import (
    DRUG_METRICS, NATIVE, RERANK, SAMPLE_METRICS, archive_component, bridge_axis,
    fit_pair, inner_development, matrix,
)
from .response_component_benchmark import BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs
from .rna_gene_panel import digest
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, paired_intervals, with_assay
from .v0plus_cv import molecular_inputs, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import EXPOSED_DONORS, REGISTRATION, require_hash

WONG = "1b7871c110bf61a7dd22d374a95e5bb68e0cdad6831eeb8f6de54c8cd469cc0d"
PRIMARY = "pdo_reference_personalized"
PANEL_NATIVE = "pdo_reference_native"
PANEL_RERANK = "pdo_reference_reranked"
PERSONAL_ONLY = "pdo_reference_personal_only"
FIXED_ONLY = "pdo_reference_offset_only"
QUERY_ONLY = "pdo_reference_query_offset"
YANG_ONLY = "pdo_reference_yang_offset"
WONG_ONLY = "pdo_reference_wong_offset"
COMMON_ARMS = (PANEL_NATIVE, PANEL_RERANK, PRIMARY, PERSONAL_ONLY, FIXED_ONLY)
ALL_ARMS = (*COMMON_ARMS, QUERY_ONLY, YANG_ONLY, WONG_ONLY)
WEIGHTS = (0., .25, .5)
MIN_LINEAGES = 30
MIN_GAIN = .002


def read_reference(path, receipt_hash, file_key, excluded, expected):
    receipt = path / "receipt.json"
    require_hash(receipt, receipt_hash)
    files = json.loads(receipt.read_text())[file_key]
    for key in ("rna.parquet", "query.parquet"):
        require_hash(path / key, files[key])
    groups = pd.read_parquet(path / "query.parquet", columns=["cell_id", "lineage_group"]).drop_duplicates()
    groups = groups[~groups.lineage_group.isin(excluded)].set_index("cell_id", verify_integrity=True)
    rna = pd.read_parquet(path / "rna.parquet").loc[groups.index].sort_index(axis=1)
    if groups.lineage_group.nunique() != expected or not rna.columns.is_unique:
        raise ValueError("Unexpected PDO reference identities or gene axis")
    return rna, groups.reset_index()


def drug_weights(development, drugs):
    scores = []
    for (drug, assay, fold), block in development.groupby(["compound_id", "track", "fold"], sort=True):
        if block.lineage_group.nunique() < MIN_LINEAGES:
            continue
        for weight in WEIGHTS:
            result = across_cell_metrics(block.assign(predicted_value=block[NATIVE] + weight * block.delta))
            scores.append({"drug": drug, "assay": assay, "fold": fold, "weight": weight, **result})
    trace = pd.DataFrame(scores, columns=["drug", "assay", "fold", "weight", *DRUG_METRICS]).dropna()
    choices, records = {}, {}
    for drug, assays in trace.groupby("drug"):
        usable = [a.set_index(["fold", "weight"])[list(DRUG_METRICS)] for _, a in assays.groupby("assay")
                  if len(a) == 2 * len(WEIGHTS)]
        if len(usable) < 2:
            continue
        improvements = {w: min(float((a.xs(w, level="weight") - a.xs(0., level="weight")).mean().min())
                               for a in usable) for w in WEIGHTS}
        good = [w for w in WEIGHTS if improvements[w] >= MIN_GAIN]
        choices[drug] = max(good, key=lambda w: (improvements[w], -w), default=0.)
        records[drug] = {"assays": len(usable), "worst_gain": improvements}
    weight = pd.Series(choices, dtype=float).reindex(drugs, fill_value=0.)
    fit_mean = (development.groupby(["compound_id", "lineage_group"]).delta.mean().groupby(level=0).mean()
                .reindex(drugs, fill_value=0.))
    return weight, fit_mean, records


def reference_roster(groups, fit):
    compounds = fit[["compound_id", "isomeric_smiles"]].drop_duplicates()
    return groups.merge(compounds, how="cross").assign(row_id=lambda x: x.cell_id + "|" + x.compound_id)


def offsets(view):
    return view.groupby(["compound_id", "lineage_group"]).delta.mean().groupby(level=0).mean()


def arms(view, weights, centers, references):
    w = view.compound_id.map(weights)
    shift = view.compound_id.map(centers)
    if w.isna().any() or shift.isna().any():
        raise ValueError("A query compound is outside the independent calibration support")
    p = view[NATIVE]
    values = {
        PANEL_NATIVE: p,
        PANEL_RERANK: view[RERANK],
        PRIMARY: p + w * (view.delta - shift) + shift,
        PERSONAL_ONLY: p + w * view.delta,
        FIXED_ONLY: p + shift,
    }
    if references:
        values[QUERY_ONLY] = p + view.compound_id.map(offsets(view))
        for name, ref in references.items():
            values[name] = p + view.compound_id.map(ref)
    return pd.concat([view[[*KEY, "track", "fold"]].assign(method=name, predicted_value=y)
                      for name, y in values.items()], ignore_index=True)


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, seal = external_inputs(args, args.mode)
    else:
        old, core, rna, _ = registered_inputs(args)
        _, fit, check = same_split(old, core, args.fold, args.mode)
        query, seal = validation_roster(check), RECEIPTS["core"]
    base, reference = archive_component(args, seal)
    wrna, wgroups = read_reference(args.wong, WONG, "files", set(), 10)
    yrna, ygroups = read_reference(args.yang, REGISTRATION, "artifacts", EXPOSED_DONORS, 74)
    native_genes = rna.columns.intersection(wrna.columns).intersection(yrna.columns).sort_values()
    panel = native_genes.intersection(bridge_axis(args))
    if not 10_000 <= len(panel) < len(native_genes):
        raise ValueError("Insufficient or identical RNA axes")
    rna = rna[native_genes]
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    device = device_cuda()
    dev, splits = inner_development(fit, rna, panel, edges, targets, auxiliary, device)
    weights, fit_mean, selected = drug_weights(dev, sorted(fit.compound_id.unique()))
    extra, detail = {}, {}
    if external:
        for name, values, groups in ((YANG_ONLY, yrna, ygroups), (WONG_ONLY, wrna, wgroups)):
            roster = reference_roster(groups, fit)
            reference_axis = pd.concat([rna, values[native_genes]], verify_integrity=True)
            estimated = fit_pair(fit, roster, reference_axis, panel, edges, targets, auxiliary, device)
            profile = matrix(estimated.assign(track=name, fold=0))
            extra[name] = offsets(profile)
            detail[name] = {"donors": int(groups.lineage_group.nunique()), "models": len(groups)}
    center = (extra[YANG_ONLY] + extra[WONG_ONLY]) / 2 if args.mode == "LICOB" else fit_mean
    raw = fit_pair(fit, query, rna, panel, edges, targets, auxiliary, device)
    view = matrix(raw.assign(track=args.mode, fold=0) if external else with_assay(raw, check, args.fold))
    new = arms(view, weights, center, extra)
    joint = pd.concat([base, new], ignore_index=True)
    methods = (*BASELINES, *PUBLIC, NATIVE, RERANK, *ALL_ARMS) if external else (*BASELINES, NATIVE, RERANK, *COMMON_ARMS)
    wide = joint.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
    if set(wide) != set(methods) or len(wide) != (len(query) if external else len(check)) or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("PDO calibration changed the common scoring roster")
    if external:
        shared_support(joint, query, methods)
    publish_json(args.calibration_out, {
        "mode": args.mode, "fold": args.fold, "prior_forecast_sha256": reference,
        "inputs_sha256": seal, "references": detail, "reference_receipts": {"wong": WONG, "yang": REGISTRATION},
        "native_axis": {"n": len(native_genes), "sha256": digest(native_genes)},
        "restricted_axis": {"n": len(panel), "sha256": digest(panel)},
        "weights": weights.to_dict(), "fit_center": fit_mean.to_dict(), "primary_center": center.to_dict(),
        "offset_controls": {name: values.to_dict() for name, values in extra.items()},
        "query_center": offsets(view).to_dict() if external else {}, "selection": selected, "splits": splits,
    })
    freeze_table(joint[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": seal, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "calibration_sha256": sha256(args.calibration_out),
        "priors": priors, "auxiliary": aux,
    })


def assess(frame, *, external):
    peers = (*BASELINES, *PUBLIC) if external else BASELINES
    checks = (*peers, NATIVE, RERANK, PANEL_NATIVE, FIXED_ONLY)
    pairs = [(PRIMARY, name) for name in checks]
    pairs += [(FIXED_ONLY, PANEL_NATIVE), (PRIMARY, PERSONAL_ONLY)]
    if external:
        pairs += [(name, PANEL_NATIVE) for name in (QUERY_ONLY, WONG_ONLY, YANG_ONLY)]
    metrics, _, _ = axis_metrics(frame, pairs)
    interval = crossed_patient_drug_ci if external else paired_intervals
    across = {name: interval(frame, PRIMARY, name, repeats=1000 if external else 800) for name in checks}
    ranks = {(r["baseline"], r["metric"]): r["ci95"] for r in metrics["paired_contrasts"] if r["candidate"] == PRIMARY}
    is_pdo = frame.track.nunique() == 1 and frame.track.iloc[0] == "LICOB"
    gates = {name: all(across[name][m][0] > 0 for m in DRUG_METRICS)
             and all(ranks[name, m][0] > (0. if is_pdo and name == PANEL_NATIVE and m == "ndcg_at_5" else -.005)
                     for m in SAMPLE_METRICS) for name in checks}
    return {"metrics": metrics, "across_cell_ci95": across, "gates": gates, "groups": frame.lineage_group.nunique(),
            "observations": frame.row_id.nunique()}


def verify(args, names):
    for name in names:
        stamp = json.loads((args.forecasts / f"{name}.parquet.freeze.json").read_text())
        require_hash(args.calibration / f"{name}.json", stamp["provenance"]["calibration_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, extra in {
        "predict": ("archive", "pdo", "licob", "wong", "yang", "priors", "hallmark", "targets", "auxiliary",
                    "calibration-out"),
        "source": ("forecasts", "calibration"),
        "external": ("forecasts", "calibration", "licob", "limore-labels", "licob-labels"),
    }.items():
        command = stages.add_parser(stage)
        for name in ("original", "core", "out", *extra):
            command.add_argument(f"--{name}", type=Path, required=True)
        if stage == "predict":
            command.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            command.add_argument("--fold", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        verify(args, (f"{mode}-{fold}" for mode in ASSAYS for fold in range(3)))
        score_source(args, assessor=assess)
    else:
        verify(args, EXTERNAL_TRACKS)
        score_external(args, assessor=assess, arms=(NATIVE, RERANK, *ALL_ARMS))


if __name__ == "__main__":
    main()
