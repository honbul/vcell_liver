"""Fit signed drug relationships from lineage- and assay-disjoint development responses."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate
from .io import freeze_table, publish_json
from .licob_node_recovery import joined_outcomes
from .limore_corrected import OBSERVATIONS, shared_support
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, check_component,
    external_inputs, external_reference, source_reference,
)
from .response_components import PENALTY, component_scores, interaction_intervals, orthogonal_parts
from .structured_cv import ASSAYS, attach, paired_intervals, with_assay
from .structured_kernel import KernelGrid, solve_residuals
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

PRIMARY = "response_interactions"
ARMS = (PRIMARY, "response_raw", "response_link_shuffle")
EXTERNAL_CONTROLS = ("structured", "response_link_shuffle", "drug_ridge", "cellhit")


def masked_fisher(values, mask, counts):
    available = np.maximum(counts, 1)
    values = np.where(mask, values, 0.)
    sums = values.T @ mask
    squared = (values ** 2).T @ mask - sums ** 2 / available
    product = values.T @ values - sums * sums.T / available
    denominator = np.sqrt(np.maximum(squared, 0) * np.maximum(squared.T, 0))
    valid = (counts >= 40) & (denominator > 1e-12)
    corr = np.divide(product, denominator, out=np.zeros_like(product), where=valid)
    np.fill_diagonal(valid, False)
    return np.arctanh(np.clip(corr, -.98, .98)), valid


def normalized_psd(fisher):
    relation = np.tanh((fisher + fisher.T) / 2)
    np.fill_diagonal(relation, 1.)
    values, vectors = np.linalg.eigh(relation)
    gram = (vectors * np.maximum(values, 0)) @ vectors.T
    diag = np.sqrt(np.diag(gram))
    if not (diag > 0).all():
        raise ValueError("Drug covariance projection has no diagonal support")
    return gram / (diag[:, None] * diag[None, :]), int((values < 0).sum())


def response_kernels(source, drugs):
    shape = (len(drugs), len(drugs))
    raw, adjusted = (np.zeros(shape) for _ in range(2))
    raw_n, adjusted_n = (np.zeros(shape, dtype=int) for _ in range(2))
    rng = np.random.default_rng(61019)
    assays = {}
    for assay, rows in source.groupby(SOURCE, sort=True):
        panel = rows.groupby(["lineage_group", "compound_id"]).residual.mean().unstack().reindex(columns=drugs)
        observed = panel.notna().to_numpy()
        mask = observed.astype(float)
        counts = mask.T @ mask
        values = panel.fillna(0.).to_numpy()
        shrink = counts / (counts + 100)
        z, okay = masked_fisher(values, mask, counts)
        raw += np.where(okay, z * shrink, 0.)
        raw_n += okay
        fitted = orthogonal_parts(values, mask)[2]
        z, okay = masked_fisher(fitted, mask, counts)
        null_sum = np.zeros(shape)
        null_count = np.zeros(shape, dtype=int)
        for _ in range(12):
            shuffled = np.zeros_like(values)
            for j in range(len(drugs)):
                present = observed[:, j]
                shuffled[present, j] = rng.permutation(values[present, j])
            score, valid = masked_fisher(orthogonal_parts(shuffled, mask)[2], mask, counts)
            null_sum += np.where(valid, score, 0.)
            null_count += valid
        okay &= null_count == 12
        adjusted += np.where(okay, (z - null_sum / np.maximum(null_count, 1)) * shrink, 0.)
        adjusted_n += okay
        assays["/".join(assay)] = {"lineages": len(panel), "adjusted_pairs": int(okay.sum() // 2)}

    raw, raw_clipped = normalized_psd(raw / np.maximum(raw_n, 1))
    interaction, clipped = normalized_psd(adjusted / np.maximum(adjusted_n, 1))
    supported = np.flatnonzero((adjusted_n > 0).any(axis=0))
    if len(supported) < 2:
        raise ValueError("Drug-response covariance is unsupported on this fit")
    index = np.arange(len(drugs))
    index[supported] = np.random.default_rng(82001).permutation(supported)
    shuffled = interaction[np.ix_(index, index)]
    return dict(zip(ARMS, (interaction, raw, shuffled))), {
        "source_assays": assays, "supported_drugs": len(supported),
        "pairs": int((adjusted_n > 0).sum() // 2), "multi_assay_pairs": int((adjusted_n > 1).sum() // 2),
        "adjusted_eigenvalues_clipped": clipped, "raw_eigenvalues_clipped": raw_clipped,
        "shuffled_drugs": int((index != np.arange(len(drugs))).sum()),
    }


def candidates(fit, query, rna, edges, targets, auxiliary):
    _validate(fit, query)
    if not torch.cuda.is_available():
        raise RuntimeError("Response-relatedness fit requires CUDA")
    device = torch.device("cuda")
    train_rna = rna.loc[sorted(fit.cell_id.unique())]
    query_rna = rna.loc[sorted(query.cell_id.unique())]
    _, anchor, source, _ = panel_targets(fit)
    grid = KernelGrid.prepare(source, train_rna, query_rna, edges, targets, auxiliary, device=device)
    kernels, info = response_kernels(source, grid.drugs)
    cell, cross = grid.cell["full"]
    structured = grid.drug["structured"]
    row, col = grid.cells.get_indexer(query.cell_id), grid.drugs.get_indexer(query.compound_id)
    baseline = query.compound_id.map(anchor).to_numpy()
    parts, solve = [], {}
    for name, values in kernels.items():
        related = torch.as_tensor(values, device=device, dtype=structured.dtype)
        drug = .60 * structured + .40 * related
        coefficients, step, residual = solve_residuals(cell, drug, grid.weights, grid.response, PENALTY)
        values = (cross @ coefficients @ drug).cpu().numpy()[row, col]
        parts.append(query[KEY].assign(method=name, predicted_value=baseline + values))
        solve[name] = {"iterations": step, "residual": residual}
    return pd.concat(parts, ignore_index=True), info | {"solver": solve}


def predict(args):
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    if args.mode in EXTERNAL_TRACKS:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, checksum = external_reference(args, args.mode, query)
        new, info = candidates(fit, query, rna, edges, targets, auxiliary)
        prediction = pd.concat([previous, new.assign(track=args.mode, fold=0)], ignore_index=True)
        shared_support(prediction, query, {*BASELINES, *PUBLIC, *ARMS})
    else:
        old, corrected, rna, _ = registered_inputs(args)
        _, fit, check = same_split(old, corrected, args.fold, args.mode)
        query = validation_roster(check)
        previous, checksum = source_reference(args.cv_archive, args.mode, args.fold)
        new, info = candidates(fit, query, rna, edges, targets, auxiliary)
        prediction = pd.concat([previous, with_assay(new, check, args.fold)[FORECAST_SCHEMA]], ignore_index=True)
        matrix = prediction.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
        if set(matrix.columns) != {*BASELINES, *ARMS} or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("New method and archived source competitors differ in observed support")
        registered = RECEIPTS["core"]
    freeze_table(prediction[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": checksum, "priors": priors, "auxiliary": aux, "relationships": info,
    })


def rank_gate(report, baseline):
    rows = {row["metric"]: row["ci95"] for row in report["paired_contrasts"]
            if row["candidate"] == PRIMARY and row["baseline"] == baseline}
    return all(rows[metric][0] > -.005 for metric in ("spearman", "ndcg_at_5"))


def score_source(args):
    old, clean, _, _ = registered_inputs(args)
    hashes, modes = {}, {}
    for mode in ASSAYS:
        pieces = []
        for fold in range(3):
            key = f"{mode}-{fold}"
            forecast, hashes[key] = check_component(args.forecasts / f"{key}.parquet", mode, fold, RECEIPTS["core"])
            _, _, check = same_split(old, clean, fold, mode)
            pieces.append(attach(forecast, check))
        modes[mode] = {}
        for track, frame in pd.concat(pieces, ignore_index=True).groupby("track"):
            pairs = [(PRIMARY, name) for name in ("structured", "response_raw", "response_link_shuffle")]
            report = axis_metrics(frame, pairs)[0]
            intervals = {b: paired_intervals(frame, a, b) for a, b in pairs}
            passed = all(low > 0 for low, _ in intervals["structured"].values()) and rank_gate(report, "structured")
            modes[mode][track] = {"metrics": report, "paired_ci95": intervals, "gate": passed}
    gates = sum(next(iter(modes[mode].values()))["gate"] for mode in ASSAYS if mode != "pooled")
    publish_json(args.out, {"source_gate": gates >= 2, "screen_holds_passed": gates, "modes": modes, "forecast_sha256": hashes})


def score_external(args):
    hashes, cohorts = {}, {}
    for study in EXTERNAL_TRACKS:
        _, query, _, registered = external_inputs(args, study)
        frame, hashes[study] = check_component(args.forecasts / f"{study}.parquet", study, 0, registered)
        shared_support(frame, query, {*BASELINES, *PUBLIC, *ARMS})
        if study == "LIMORE":
            require_hash(args.limore_labels, OBSERVATIONS)
            joined = join_observations(frame, pd.read_parquet(args.limore_labels))
        else:
            require_hash(args.licob_labels, ASSET_HASHES["supplement"])
            joined = adult_donor_outcomes(joined_outcomes(query, frame, args.licob_labels))
        contrasts = [(PRIMARY, name) for name in (*EXTERNAL_CONTROLS, "response_raw", "drug_mean")]
        overall = axis_metrics(joined, contrasts)[0]
        across = {b: crossed_patient_drug_ci(joined, a, b) for a, b in contrasts}
        interactions = {b: interaction_intervals(joined, PRIMARY, b) for b in EXTERNAL_CONTROLS}
        gate = {b: all(low > 0 for low, _ in across[b].values()) and interactions[b][0] > 0
                and rank_gate(overall, b) for b in EXTERNAL_CONTROLS}
        strata = {}
        groups = compound_strata(query)
        for name in ("hcc_tki", "targeted_or_kinase", "nonselective_cytotoxics"):
            drugs = groups[name]
            if drugs:
                subset = axis_metrics(joined[joined.compound_id.isin(drugs)], ())[0]
                if len(drugs) <= 5:
                    for row in subset["scores"]:
                        row.pop("macro_ndcg_at_5", None)
                        row.pop("macro_top5_overlap", None)
                strata[name] = {"drugs": len(drugs), "metrics": subset}
        cohorts[study] = {"groups": joined.lineage_group.nunique(), "rows": joined.row_id.nunique(),
                          "metrics": overall, "paired_ci95": across, "interaction_ci95": interactions,
                          "response_components": component_scores(joined, cell_axis="lineage_group"),
                          "gate": gate, "strata": strata}
    publish_json(args.out, {"external_gate": all(all(cohorts[study]["gate"].values()) for study in EXTERNAL_TRACKS),
                            "cohorts": cohorts, "forecast_sha256": hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary"),
        "source": ("forecasts",),
        "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }
    for action, names in fields.items():
        stage = stages.add_parser(action)
        for name in ("original", "core", *names, "out"):
            stage.add_argument(f"--{name}", required=True, type=Path)
        if action == "predict":
            stage.add_argument("--mode", required=True, choices=(*ASSAYS, *EXTERNAL_TRACKS))
            stage.add_argument("--fold", type=int, default=0, choices=range(3))
    args = parser.parse_args()
    {"predict": predict, "source": score_source, "external": score_external}[args.stage](args)


if __name__ == "__main__":
    main()
