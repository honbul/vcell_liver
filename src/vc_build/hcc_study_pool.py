"""Evaluate shared organoid response residuals with one entire HCC study held out."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import join_observations
from .external import KEY
from .function_data import new_output
from .hcc_3d_reciprocal import checked_inputs
from .hcc_3d_study_transfer import DRUGS, recipient_outcomes, validated_inputs
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_full_axis import validated as validated_wide
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda, domain_kernel_terms
from .v0plus_cv import molecular_inputs
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_audit import COHORT as YANG
from .yang_hcc_external import require_hash

STUDIES = ("LICOB", "Wong", YANG)
RECIPROCAL = "d9778cf77d3732f81f8bbcbe4edc9c5a15aecf454c5047bc1f53278bfd629716"
RECIPROCAL_FORECAST = "df76a2992c07532cfac09fd2ce6d8fb846f7a21a7e7c4f9c40d2d84e6d775bad"
TRANSFER = "aed0f31111507989ab7c241888660d97bff687f39af62a3585ae6fcfd59e54a1"
YANG_FORECAST = "00c685ec949ed8c8abe4c135b390c56c2f24b9be8d193ed19074f00129094d79"
WIDE = "1f37cdff695ad7c9cbac06430a86b77d5ef5dfca5f9fa03d4557954b464102a0"
REFERENCE = ("2d_mean", "2d_ridge", "2d_cellhit", "2d_structured", "2d_teacher")
ARMS = ("pooled_3d", "study_balanced_3d", "balanced_residual", "balanced_interaction")
PRIMARY = "balanced_residual"
PENALTY = 2.


def known_methods(frame, source, *, prefix=""):
    names = REFERENCE if source == "Wong" else ("drug_mean", "drug_ridge", "cellhit", "structured", "ridge_structured_average")
    name_map = {prefix + name: reference for name, reference in zip(names, REFERENCE)}
    subset = frame[frame.method.isin(name_map)].copy()
    if source == "LICOB":
        subset.cell_id = "LICOB:" + subset.cell_id
    subset.method = subset.method.map(name_map)
    return subset.pivot(index=["cell_id", "compound_id"], columns="method", values="predicted_value")


def prepare(args):
    if checked_inputs(args.reciprocal)[1] != RECIPROCAL or validated_inputs(args.transfer) != TRANSFER:
        raise ValueError("Unexpected archived study input")
    if validated_wide(args.wide) != WIDE:
        raise ValueError("Unregistered recovered wider LICOB axis")
    wide, wide_seal = verify_forecast(args.wide_forecast)
    old, old_seal = verify_forecast(args.reciprocal_forecast)
    yang, yang_seal = verify_forecast(args.yang_forecast)
    if (wide_seal["provenance"]["registration_sha256"], old_seal["sha256"], yang_seal["sha256"]) != (
            WIDE, RECIPROCAL_FORECAST, YANG_FORECAST):
        raise ValueError("Unregistered matched 2D-only forecasts")
    wong = pd.read_parquet(args.reciprocal / "Wong.parquet")
    licob = pd.read_parquet(args.wide / "query.parquet")
    panel = set(wong.compound_id)
    licob = licob[licob.compound_id.isin(panel)].copy()
    licob["model_native"] = licob.cell_id
    licob.cell_id = "LICOB:" + licob.cell_id
    licob.lineage_group = "LICOB:" + licob.donor_id
    licob.row_id = licob.cell_id + "|" + licob.compound_id
    wong = wong.assign(new_donor=False)
    # Use the frozen RNA-only 3D application identities. The source-only teacher already covers all Yang cells.
    applications = pd.read_parquet(args.transfer / "application.parquet")
    yang_query = applications[applications.study.eq(YANG) & applications.compound_id.isin(DRUGS)].drop(columns="study")
    yang_query = yang_query.assign(new_donor=False,
        source_compound_label=yang_query.compound_id.map(DRUGS), drug_target_source_labels="kinase")
    parts = dict(zip(STUDIES, (licob, wong, yang_query)))
    forecasts = {"LICOB": known_methods(wide, "LICOB", prefix="wide/"),
                 "Wong": known_methods(old[old.track.eq("Wong")], "Wong"), YANG: known_methods(yang, YANG)}
    expected = {"LICOB": (43, 38, 21), "Wong": (10, 10, 21), YANG: (217, 74, 3)}
    for study, q in parts.items():
        q = q.join(forecasts[study], on=["cell_id", "compound_id"], validate="one_to_one")
        sizes = (q.cell_id.nunique(), q.lineage_group.nunique(), q.compound_id.nunique())
        if sizes != expected[study] or q.row_id.duplicated().any() or q[list(REFERENCE)].isna().any(axis=None):
            raise ValueError(f"Unmatched {study} registered rows or 2D-only forecasts")
        parts[study] = q
    lrna = pd.read_parquet(args.wide / "rna.parquet").rename(index=lambda label: "LICOB:"+label)
    others = pd.read_parquet(args.transfer / "rna.parquet")
    ids = sorted(set(wong.cell_id) | set(yang_query.cell_id))
    genes = sorted(set(lrna.columns) & set(others.columns))
    if len(genes) < 13000:
        raise ValueError("Insufficient shared HCC study RNA features")
    frames = {f"{name}.parquet": part for name, part in parts.items()}
    frames["rna.parquet"] = pd.concat([lrna[genes], others.loc[ids, genes]], verify_integrity=True)
    with new_output(args.out) as tmp:
        files = {}
        for name, table in frames.items():
            table.to_parquet(tmp / name, index=name == "rna.parquet")
            files[name] = sha256(tmp / name)
        publish_json(tmp / "receipt.json", {"prepared_from": [RECIPROCAL, TRANSFER, WIDE], "gene_count": len(genes),
            "reference_hashes": [old_seal["sha256"], yang_seal["sha256"], wide_seal["sha256"]],
            "model_support": expected, "artifacts": files})


def registered(path):
    file = path / "receipt.json"
    value = json.loads(file.read_text())
    if value["prepared_from"] != [RECIPROCAL, TRANSFER, WIDE]:
        raise ValueError("The registered archive identities changed")
    for name, digest in value["artifacts"].items():
        require_hash(path / name, digest)
    return sha256(file)


def data(args, study, *, full):
    query = pd.read_parquet(args.inputs / f"{study}.parquet")
    if study == "LICOB" and not full:
        query = query[~query.new_donor]
    if study == YANG:
        require_hash(args.transfer / "receipt.json", TRANSFER)
        document = json.loads((args.transfer / "receipt.json").read_text())
        path = args.transfer / "fit3d.parquet"
        require_hash(path, document["artifacts"][path.name])
        measured = pd.read_parquet(path, columns=["cell_id", "compound_id", "endpoint_value"])
        frame = query.merge(measured, on=["cell_id", "compound_id"], validate="one_to_one")
    else:
        path = args.licob_book if study == "LICOB" else args.wong_book
        if path is None:
            raise ValueError("Missing authorized training-study response file")
        frame = recipient_outcomes(study, query, path)
    if len(frame) != len(query) or not np.isfinite(frame.endpoint_value).all():
        raise ValueError(f"Missing outcomes in {study} registered response support")
    return frame.assign(study=study)


def donor_scales(train):
    donors = train.groupby(["study", "lineage_group", "compound_id"], as_index=False)[["endpoint_value", "contrast"]].mean()
    grouped = donors.groupby(["study", "compound_id"])
    center = grouped[["endpoint_value", "contrast"]].mean().add_suffix("_center")
    variation = grouped[["endpoint_value", "contrast"]].std(ddof=0).add_suffix("_scale")
    return center.join(variation).join(grouped.size().rename("donors_per_study"))


def estimates(train, query, rna, edges, targets, device):
    import torch

    from .structured_kernel import drug_kernels, solve_residuals

    if set(train.lineage_group) & set(query.lineage_group) or "endpoint_value" in query:
        raise ValueError("Held study or labels entered the prediction step")
    train = train.assign(contrast=train["2d_teacher"] - train["2d_mean"])
    scales = donor_scales(train)
    if not scales.donors_per_study.ge(5).all() or scales.endpoint_value_scale.le(1e-8).any():
        raise ValueError("Study/compound blocks must have measurable response variation")
    calibration = scales.contrast_scale.groupby("compound_id").mean().clip(lower=.020)
    train = train.join(scales, on=["study", "compound_id"])
    y = (train.endpoint_value - train.endpoint_value_center) / train.endpoint_value_scale
    t = (train.contrast - train.contrast_center) / train.compound_id.map(calibration)
    residual = y - t
    interaction = residual - residual.groupby(train.cell_id).transform("mean")
    donors = train.groupby("lineage_group").cell_id.nunique()
    per_donor = 1 / train.lineage_group.map(donors)
    weights = per_donor * train.groupby("compound_id").donors_per_study.transform("max") / train.donors_per_study
    weights = weights / train.groupby("compound_id").study.transform("nunique")
    fit_ids, held_ids = (pd.Index(sorted(frame.cell_id.unique())) for frame in (train, query))
    drugs, kernels = drug_kernels(train, targets, device=device)
    cell, cross = domain_kernel_terms(rna, fit_ids, held_ids, edges, targets, device)["nonlinear"]
    studies = train.drop_duplicates("cell_id").set_index("cell_id").study.reindex(fit_ids).to_numpy()
    training_kernel = cell * (1 + .5 * torch.as_tensor(np.equal.outer(studies, studies), device=device))
    at = fit_ids.get_indexer(train.cell_id), drugs.get_indexer(train.compound_id)
    query_at = held_ids.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if any((idx < 0).any() for idx in (*at, *query_at)):
        raise ValueError("The held panel contains a drug absent from all fit studies")

    def grid(values):
        matrix = torch.zeros((len(fit_ids), len(drugs)), device=device)
        matrix[at] = torch.as_tensor(values.to_numpy(np.float32), device=device)
        return matrix

    rows, notes = [], {}
    for method, target, weight, gram, anchor, multiplier in (
        ("pooled_3d", y, per_donor, cell, "2d_mean", 1.),
        ("study_balanced_3d", y, weights, training_kernel, "2d_mean", 1.),
        ("balanced_residual", residual, weights, training_kernel, "2d_teacher", .5),
        ("balanced_interaction", interaction, weights, training_kernel, "2d_teacher", .5),
    ):
        model, steps, err = solve_residuals(gram, kernels["structured"], grid(weight), grid(target), PENALTY)
        scores = (cross @ model @ kernels["structured"]).cpu().numpy()[query_at]
        prediction = query[anchor] + multiplier * query.compound_id.map(calibration) * scores
        rows.append(query[KEY].assign(method=method, predicted_value=prediction))
        notes[method] = {"solver_steps": steps, "solver_residual": err}
    references = [query[KEY].assign(method=name, predicted_value=query[name]) for name in REFERENCE]
    return pd.concat([*references, *rows]), {"fit_support": train.groupby("study").lineage_group.nunique().to_dict(),
        "fit_source_count_per_drug": scales.reset_index().groupby("compound_id").study.nunique().to_dict(), "solver": notes}


def predict(args):
    receipt = registered(args.inputs)
    if (args.holdout == "LICOB" and args.licob_book is not None) or (args.holdout == "Wong" and args.wong_book is not None):
        raise ValueError("Held study path was supplied to the predictor")
    train = pd.concat([data(args, study, full=False) for study in STUDIES if study != args.holdout], ignore_index=True)
    query = pd.read_parquet(args.inputs / f"{args.holdout}.parquet")
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    pred, notes = estimates(train, query, rna, edges, targets, device_cuda())
    shared_support(pred, query, (*REFERENCE, *ARMS))
    freeze_table(pred.assign(track=args.holdout, fold=0), args.out, provenance={"input_sha256": receipt,
        "holdout": args.holdout, "training_studies": sorted(set(STUDIES) - {args.holdout}), **notes})


def merge(args):
    receipt = registered(args.inputs)
    parts, component = [], {}
    for name in STUDIES:
        pred, stamp = verify_forecast(args.parts / f"{name}.parquet")
        p = stamp["provenance"]
        if (p["holdout"], p["input_sha256"], set(p["training_studies"])) != (name, receipt, set(STUDIES) - {name}):
            raise ValueError("Another study was not held out")
        parts.append(pred)
        component[name] = stamp["sha256"]
    freeze_table(pd.concat(parts), args.out, provenance={"input_sha256": receipt, "components": component})


def evaluate_scope(frame):
    base = frame[frame.method.eq("2d_mean")].groupby("compound_id").predicted_value
    if not base.nunique().eq(1).all():
        raise ValueError("A drug-only mean varies with the held sample")
    block = frame.groupby(["lineage_group", "compound_id", "track", "fold", "method"], as_index=False)[
        ["predicted_value", "endpoint_value"]].mean()
    is_mean = block.method.eq("2d_mean")
    block.loc[is_mean, "predicted_value"] = block.loc[is_mean, "compound_id"].map(base.first())
    block = block.assign(cell_id=block.lineage_group, row_id=block.lineage_group + "|" + block.compound_id)
    pairs = [(PRIMARY, name) for name in ("2d_teacher", "2d_ridge", "2d_cellhit", "study_balanced_3d", "pooled_3d")]
    pairs += [("balanced_interaction", PRIMARY)]
    full = axis_metrics(block, pairs)[0]
    count = block.compound_id.nunique()
    if count <= 5:
        for row in full["scores"]:
            row.pop("macro_ndcg_at_5", None)
            row.pop("macro_top5_overlap", None)
        full["paired_contrasts"] = [x for x in full["paired_contrasts"] if x["metric"] == "spearman"]
    cis = [{"candidate": a, "baseline": b, "ci95": crossed_patient_drug_ci(block, a, b)} for a, b in pairs]
    delta = next(row["ci95"] for row in cis if row["baseline"] == "2d_teacher")
    delta_rank = {row["metric"]: row["ci95"] for row in full["paired_contrasts"]
                  if row["candidate"] == PRIMARY and row["baseline"] == "2d_teacher"}
    passing = count > 5 and all(delta[m][0] > 0 for m in ("spearman", "pearson")) and all(
        delta_rank[m][0] > 0 for m in ("spearman", "ndcg_at_5"))
    return {"support": {"donors": block.lineage_group.nunique(), "drugs": count}, "report": full, "crossed_ci95": cis,
            "joint_direction_gate": passing if count > 5 else None}


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    receipt = registered(args.inputs)
    forecast, seal = verify_forecast(args.forecast)
    if seal["provenance"]["input_sha256"] != receipt:
        raise ValueError("Study holdout changed before scoring")
    scores = {}
    for study in STUDIES:
        truth = data(args, study, full=True)
        joined = join_observations(forecast[forecast.track.eq(study)], truth)
        scopes = {"all": joined, "three_TKI": joined[joined.compound_id.isin(DRUGS)]}
        if study == "LICOB":
            new = set(truth[truth.new_donor].lineage_group)
            scopes["nine_previous_confirmation_donors"] = joined[joined.lineage_group.isin(new)]
        scores[study] = {name: evaluate_scope(rows) for name, rows in scopes.items()}
    publish_json(args.out, {"forecast_sha256": seal["sha256"], "registration_sha256": receipt, "primary": PRIMARY,
        "all_21_drug_gate": all(scores[s]["all"]["joint_direction_gate"] for s in ("LICOB", "Wong")),
        "scores": scores})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    fields = {"prepare": ("reciprocal", "reciprocal_forecast", "transfer", "wide", "wide_forecast", "yang_forecast"),
              "predict": ("inputs", "transfer", "priors", "hallmark", "targets"), "merge": ("inputs", "parts"),
              "score": ("inputs", "transfer", "forecast")}
    for name, paths in fields.items():
        task = sub.add_parser(name)
        for key in (*paths, "out"):
            task.add_argument("--"+key.replace("_", "-"), required=True, type=Path)
        if name in ("predict", "score"):
            task.add_argument("--licob-book", type=Path)
            task.add_argument("--wong-book", type=Path)
        if name == "predict":
            task.add_argument("--holdout", required=True, choices=STUDIES)
    args = parser.parse_args()
    {"prepare": prepare, "predict": predict, "merge": merge, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
