"""Fit a same-assay ICC bridge and evaluate held HCC histology and studies."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook

from .culture_consistency import kernel_between
from .evaluate import join_observations
from .expanded_panel_cv import PENALTY, RECEIPTS
from .external import KEY, _validate
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_full_axis import RNA_HASH, validated
from .licob_node_recovery import joined_outcomes, recover
from .limore_corrected import OBSERVATIONS, shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import BASELINES, PUBLIC, external_inputs, external_reference
from .response_components import centered_kernels, orthogonal_parts
from .structured_kernel import drug_kernels, fit_cell_kernels, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, training_inputs
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes, paired_axis
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .wong_expanded import checked as checked_wong, observations as wong_observations
from .v0plus_wong import WORKBOOK_HASH
from .yang_hcc_external import require_hash

WONG_REFERENCE = "9330a821ea14ab99bde652f7ff66e84879d92ef63bd9e97a3fb3b41e38af9965"
TRACKS = ("LIMORE", "LICOB", "WONG")
BASE = "icc_matched_2d"
ANCHOR = "icc_anchor_only"
PRIMARY = "icc_anchor_and_residual"
NO_ANCHOR = "icc_residual_only"
SHUFFLE = "icc_donor_shuffle"
ARMS = (BASE, ANCHOR, PRIMARY, NO_ANCHOR, SHUFFLE)
PEERS = (*BASELINES, *PUBLIC)
WEIGHT = .5


def clinical_support(book):
    models, groups = {}, []
    for name, age, *_, histology in book["S1. Clinical information"].iter_rows(
            min_row=5, max_col=6, values_only=True):
        if isinstance(name, str) and name.endswith("were from the same patient sample"):
            groups.extend(re.findall(r"[A-Z]+\d+", name))
        if histology == "ICC" and re.fullmatch(r"ICCO\d+", str(name)):
            models[name] = age
    result = pd.to_numeric(pd.Series(models), errors="coerce")
    names = sorted(result[result.ge(18)].index)
    if len(names) != 12 or set(names) & set(groups):
        raise ValueError("Unexpected ICC age, histology or duplicate donor annotation")
    return names


def register(args):
    require_hash(args.public_rna, RNA_HASH)
    require_hash(args.node, ASSET_HASHES["node_archive"])
    require_hash(args.licob_labels, ASSET_HASHES["supplement"])
    hcc_receipt = validated(args.licob)
    hcc = pd.read_parquet(args.licob / "query.parquet")
    compounds = hcc[["compound_id", "response_column", "response_label"]].drop_duplicates()
    if len(compounds) != 53 or not compounds.compound_id.is_unique or not compounds.response_column.is_unique:
        raise ValueError("Unexpected S6 panel for the 2D core")
    rna = pd.read_parquet(args.public_rna)
    with closing(load_workbook(args.licob_labels, read_only=True, data_only=True)) as book:
        names = clinical_support(book)
        _, rows = paired_axis(book)
        if not set(names) <= set(rows) & set(rna.index) or set(names) & (set(hcc.cell_id) | set(hcc.donor_id)):
            raise ValueError("Crossed ICC and HCC roles")
        table = book["S6. AUC"]
        measure = []
        for name in names:
            n = rows[name]
            selected = next(table.iter_rows(min_row=n, max_row=n, values_only=True))
            if selected[0] != name:
                raise ValueError("The ICC S6 row identity changed")
            measure.append([selected[i] for i in compounds.response_column])
    values = pd.DataFrame(measure, index=pd.Index(names, name="cell_id"), columns=compounds.compound_id)
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Missing ICC response on the registered panel")
    rna = rna.loc[names]
    prefix, _ = recover(args.node)
    genes = rna.columns.intersection(prefix.index)
    match = float(np.max(np.abs(rna[genes].to_numpy() - prefix.loc[genes, names].to_numpy().T)))
    if len(genes) < 14_000 or match > 3e-7:
        raise ValueError("Public ICC RNA differs from the decodable original workbook prefix")
    with new_output(args.out) as folder:
        assets = {}
        for file, data in (("rna.parquet", rna), ("icc_sensitivity.parquet", -values)):
            path = folder / file
            data.to_parquet(path)
            assets[file] = sha256(path)
        publish_json(folder / "receipt.json", {"files": assets, "icc_models": len(names), "compounds": 53,
                     "public_rna_sha256": RNA_HASH, "licob_receipt_sha256": hcc_receipt,
                     "shared_node_genes": len(genes), "maximum_prefix_difference": match,
                     "supplement_sha256": ASSET_HASHES["supplement"],
                     "same_donor_footnotes_crossing_ICC_HCC": 0})


def icc_inputs(path, hcc):
    receipt = path / "receipt.json"
    metadata = json.loads(receipt.read_text())
    if metadata["licob_receipt_sha256"] != hcc or metadata["public_rna_sha256"] != RNA_HASH:
        raise ValueError("Unexpected ICC bridge registration")
    for name, digest in metadata["files"].items():
        require_hash(path / name, digest)
    expr = pd.read_parquet(path / "rna.parquet")
    values = pd.read_parquet(path / "icc_sensitivity.parquet")
    if not expr.index.equals(values.index) or values.shape != (12, 53):
        raise ValueError("The ICC bridge registration changed")
    return expr, values, sha256(receipt)


def query_inputs(args):
    if args.mode != "WONG":
        fit, roster, rna, seal = external_inputs(args, args.mode)
        compare, ref = external_reference(args, args.mode, roster)
        return fit, roster, rna, seal, compare, ref
    seal = checked_wong(args.wong, ("query.parquet", "rna.parquet"))
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    fit, rna, _ = training_inputs(args.core)
    query = pd.read_parquet(args.wong / "query.parquet")
    query = query[query.compound_id.isin(fit.compound_id)]
    panel = pd.read_parquet(args.wong / "rna.parquet").loc[sorted(query.cell_id.unique())]
    if len(query) != 270 or not query.previously_scored.all():
        raise ValueError("Unexpected core-supported Wong roster")
    compare, receipt = verify_forecast(args.wong_archive)
    if receipt["sha256"] != WONG_REFERENCE or receipt["provenance"]["registration_sha256"] != seal:
        raise ValueError("Unexpected Wong forecast registration")
    compare = compare[compare.method.isin(PEERS) & compare.row_id.isin(query.row_id)].assign(track="WONG", fold=0)
    genes = rna.columns.intersection(panel.columns)
    return fit, query, pd.concat([rna[genes], panel[genes]], verify_integrity=True), seal, compare, receipt["sha256"]


def affine_map(measurement, reference):
    def iqr(x):
        a, b = np.quantile(x, [.25, .75])
        return b - a

    x, y = measurement.mean(axis=0), reference.mean(axis=0)
    if min(iqr(x), iqr(y)) < 1e-8:
        raise ValueError("ICC to 2D potency map has insufficient variation")
    ratio = float(np.clip(iqr(y) / iqr(x), .25, 4))
    offset = float(np.median(y - ratio*x))
    return ratio * measurement + offset, {"slope": ratio, "offset": offset}


def centered_chemistry(kernel, indices):
    block = kernel[indices][:, indices]
    return centered_kernels(block, block)[0]


def corrections(cell, projected, drug, target):
    tensor = torch.as_tensor(target, device=cell.device, dtype=cell.dtype)
    coeff, n, residual = solve_residuals(cell, drug, torch.ones_like(tensor), tensor, PENALTY)
    return (projected @ coeff @ drug).cpu().numpy(), {"iterations": n, "solver_residual": residual}


def forecast(fit, query, rna, icc, measured, edges, targets, auxiliary):
    _validate(fit, query)
    if set(icc.index) & (set(fit.cell_id) | set(query.cell_id) | set(query.lineage_group)):
        raise ValueError("ICC and validation lineages must be disjoint")
    axis = rna.columns.intersection(icc.columns).sort_values()
    train = rna.loc[sorted(fit.cell_id.unique()), axis]
    held = rna.loc[sorted(query.cell_id.unique()), axis]
    icc = icc[axis]
    device = device_cuda()
    basis, gram = fit_cell_kernels(train, edges, auxiliary, device=device)
    _, anchor, aligned, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(aligned, targets, device=device)
    g = kernels["structured"]
    w, y = training_grid(aligned, train.index, drugs, device=device)
    coef, iterations, error = solve_residuals(gram["full"], g, w, y, PENALTY)
    b = (basis.cross(held)["full"] @ coef @ g).cpu().numpy() + anchor.loc[drugs].to_numpy()
    icc_b = (basis.cross(icc)["full"] @ coef @ g).cpu().numpy() + anchor.loc[drugs].to_numpy()
    indices = drugs.get_indexer(measured.columns)
    if (indices < 0).any():
        raise ValueError("Missing an ICC drug in the fit core")
    z, map_meta = affine_map(measured.to_numpy(), icc_b[:, indices])
    error_panel = z - icc_b[:, indices]
    _, shift, residual = orthogonal_parts(error_panel, np.ones_like(error_panel))
    reference, tested = basis.vectors(icc), basis.vectors(held)
    kc = kernel_between(basis, reference, reference)
    kq = kernel_between(basis, tested, reference)
    cell, cross = centered_kernels(kc, kq)
    ik = centered_chemistry(g, indices)
    fit_error, stats = corrections(cell, cross, ik, residual)
    order = np.random.default_rng(20327).permutation(len(residual))
    permutation = np.empty_like(order)
    permutation[order] = np.roll(order, 1)
    shuffled, null_stats = corrections(cell, cross, ik, residual[permutation])
    d = np.zeros(b.shape[1])
    d[indices] = WEIGHT * shift
    t, null = np.zeros_like(b), np.zeros_like(b)
    t[:, indices], null[:, indices] = WEIGHT * fit_error, WEIGHT * shuffled
    requested = held.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    values = {BASE: b, ANCHOR: b+d, PRIMARY: b+d+t, NO_ANCHOR: b+t, SHUFFLE: b+d+null}
    panel = pd.concat([query[KEY].assign(method=m, predicted_value=z[requested]) for m,z in values.items()],
                      ignore_index=True)
    return panel, pd.Series(d, index=drugs), {
        "genes": len(axis), "direct_query_drugs": len(set(query.compound_id) & set(measured.columns)),
        "affine": map_meta, "anchor_shift_rms": float(np.sqrt(np.mean(d[indices]**2))),
        "icc_interaction_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "predicted_interaction_rms": float(np.sqrt(np.mean(t[:, indices]**2))),
        "base_solver": {"iterations": iterations, "solver_residual": error},
        "residual_solver": stats, "shuffle_solver": null_stats,
    }


def predict(args):
    fit, query, rna, seal, prior, ref = query_inputs(args)
    icc, measured, icc_seal = icc_inputs(args.icc, validated(args.licob))
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    new, d, audit = forecast(fit, query, rna, icc, measured, edges, targets, auxiliary)
    same_info = prior.assign(method="icc_offset_" + prior.method,
                             predicted_value=prior.predicted_value + prior.compound_id.map(d))
    rows = pd.concat([prior, same_info, new.assign(track=args.mode, fold=0)], ignore_index=True)
    shared_support(rows, query, (*PEERS, *ARMS, *("icc_offset_"+x for x in PEERS)))
    freeze_table(rows, args.out, provenance={
        "mode": args.mode, "registration_sha256": seal, "icc_receipt_sha256": icc_seal,
        "reference_sha256": ref, "core_sha256": RECEIPTS["core"],
        "priors": priors, "auxiliary": aux, "icc_adaptation": audit,
    })


def score(args):
    results, hashes = {}, {}
    for mode in TRACKS:
        args.mode = mode
        _, query, _, registration, _, ref = query_inputs(args)
        rows, stamp = verify_forecast(args.forecasts / f"{mode}.parquet")
        info = stamp["provenance"]
        if (info["mode"], info["registration_sha256"], info["reference_sha256"], info["icc_receipt_sha256"]) != (
                mode, registration, ref, sha256(args.icc / "receipt.json")):
            raise ValueError("Unexpected ICC bridge forecast")
        if mode == "LIMORE":
            require_hash(args.limore_labels, OBSERVATIONS)
            joined = join_observations(rows, pd.read_parquet(args.limore_labels))
        elif mode == "LICOB":
            require_hash(args.licob_labels, ASSET_HASHES["supplement"])
            joined = adult_donor_outcomes(joined_outcomes(query, rows, args.licob_labels))
        else:
            require_hash(args.wong_labels, WORKBOOK_HASH)
            labels = wong_observations(query, args.wong_labels)
            joined = join_observations(rows[rows.row_id.isin(labels.row_id)], labels)
        controls = (BASE, ANCHOR, SHUFFLE, *("icc_offset_"+m for m in PEERS))
        pairs = [(PRIMARY, m) for m in controls] + [(ANCHOR, BASE), (NO_ANCHOR, BASE)]
        metrics, _, _ = axis_metrics(joined, pairs)
        interval = {m: crossed_patient_drug_ci(joined, PRIMARY, m, repeats=1000) for m in controls}
        rank = {(r["baseline"], r["metric"]): r["ci95"] for r in metrics["paired_contrasts"]
                if r["candidate"] == PRIMARY}
        gates = {m: all(interval[m][k][0] > 0 for k in ("spearman", "pearson"))
                 and all(rank[m,k][0] > -.005 for k in ("ndcg_at_5", "spearman", "top5_overlap")) for m in controls}
        strata = {name: {"drugs": part.compound_id.nunique(), "metrics": axis_metrics(part, ())[0]}
                  for name, drugs in compound_strata(query).items()
                  if name in ("hcc_tki","targeted_or_kinase","nonselective_cytotoxics")
                  and len(part := joined[joined.compound_id.isin(drugs)])}
        results[mode] = {"metrics": metrics, "across_cell_ci95": interval, "gates": gates,
                         "strata": strata, "groups": joined.lineage_group.nunique(), "observations": joined.row_id.nunique()}
        hashes[mode] = stamp["sha256"]
    publish_json(args.out, {"cohorts": results, "forecast_sha256": hashes,
                            "pass": all(all(x["gates"].values()) for x in results.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    common = ("original", "core", "licob", "archive", "wong", "wong-archive", "icc")
    for stage, fields in {
        "register": ("public-rna", "node", "licob", "licob-labels"),
        "predict": (*common, "priors", "hallmark", "targets", "auxiliary"),
        "score": (*common, "forecasts", "limore-labels", "licob-labels", "wong-labels"),
    }.items():
        p = stages.add_parser(stage)
        for field in ("out", *fields):
            p.add_argument("--"+field, type=Path, required=True)
        if stage == "predict":
            p.add_argument("--mode", choices=TRACKS, required=True)
    args = parser.parse_args()
    {"register": register, "predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
