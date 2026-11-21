"""Evaluate fixed structure models on compound- and replicate-held-out Sci-Plex3 wells."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import cho_factor, cho_solve

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import read_json, sha256, write_json

from .sciplex_replicate import DRAWS, LINES, N_GENES, centered_square, paired_dose, plate_profiles


DOSES = (10000, 1000)
ALPHAS = (0.1, 1.0, 10.0)
MODELS = ("tanimoto_ridge", "linear_ridge")
SEED = 130113


def chemistry(wells):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    rows, fps = [], []
    for drug, cultures in wells.loc[~wells.vehicle].groupby("drug"):
        smiles = cultures.SMILES.dropna().unique()
        if len(smiles) > 1:
            raise ValueError(f"Multiple released SMILES for catalog drug {drug}")
        with rdBase.BlockLogs():
            parent = canonical_parent(smiles[0]) if len(smiles) else None
            mol = Chem.MolFromSmiles(parent[1]) if parent else None
            key = Chem.MolToInchiKey(mol).split("-")[0] if mol else ""
        if not key:
            rows.append(dict(drug=drug, status="missing_or_invalid_structure", connectivity="", fingerprint=-1))
            continue
        rows.append(dict(drug=drug, status="valid", connectivity=key, fingerprint=len(fps)))
        fps.append(generator.GetFingerprintAsNumPy(mol))
    compounds = pd.DataFrame(rows).set_index("drug", verify_integrity=True)
    groups = sorted(compounds.loc[compounds.status.eq("valid"), "connectivity"].unique())
    universe = pd.Index(sorted(groups, key=lambda key: (hashlib.sha256(key.encode()).digest(), key)))
    order = pd.Series(np.arange(len(universe)) % 3, index=universe)
    compounds["fold"] = compounds.connectivity.map(order).fillna(-1).astype(int)
    return compounds, universe, np.asarray(fps, dtype=np.float64)


def kernels(fp):
    linear = fp @ fp.T
    mass = np.diag(linear)
    tanimoto = linear / np.maximum(mass[:, None] + mass[None, :] - linear, 1e-12)
    return {"tanimoto_ridge": tanimoto, "linear_ridge": linear}


def balanced_weights(groups):
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    return 1 / count[inverse]


def ridge(kernel, values, groups, train, query, alpha):
    fit = kernel[np.ix_(train, train)]
    held = kernel[np.ix_(query, train)]
    target = values[train]
    weight = balanced_weights(groups[train])
    weight /= weight.sum()
    fitmean = weight @ fit
    globalmean = fitmean @ weight
    fit = fit - fitmean[None, :] - fitmean[:, None] + globalmean
    held = held - fitmean - (held @ weight)[:, None] + globalmean
    targetmean = weight @ target
    scale = np.sqrt(weight * len(weight))
    system = scale[:, None] * fit * scale
    system[np.diag_indices_from(system)] += alpha
    coefficient = cho_solve(cho_factor(system), scale[:, None] * (target - targetmean))
    return (held * scale) @ coefficient + targetmean


def select_alpha(kernel, source, groups, folds, outer):
    totals = np.zeros(len(ALPHAS))
    for inner in sorted(set(folds) - {outer}):
        train = np.flatnonzero((folds != inner) & (folds != outer))
        validation = np.flatnonzero(folds == inner)
        actual = source[validation]
        weights = balanced_weights(groups[validation])
        for i, alpha in enumerate(ALPHAS):
            residual = ridge(kernel, source, groups, train, validation, alpha) - actual
            residual -= np.average(residual, axis=0, weights=weights)
            row_rmse = np.sqrt(np.mean(residual**2, axis=1))
            totals[i] += weights @ row_rmse
    return ALPHAS[int(np.argmin(totals))]


def score_fold(actual, donor, predictions, codes, draws):
    mass = draws[:, codes] * balanced_weights(codes)
    pool = mass / mass.sum(axis=1, keepdims=True)
    residual = {"B0": actual, "replicate_quarter": actual - 0.25 * donor,
                "replicate_full": actual - donor}
    residual.update({name: actual - prediction for name, prediction in predictions.items()})
    totals = {}
    for name, delta in residual.items():
        kernel = (delta @ delta.T) / N_GENES
        totals[name] = (mass * np.sqrt(centered_square(kernel, pool))).sum(axis=1)
    return totals, mass.sum(axis=1)


def cross_validate(source, target, cohort, catalogue_kernels, codes, draws):
    groups = cohort.connectivity.to_numpy()
    folds = cohort.fold.to_numpy()
    indices = cohort.fingerprint.to_numpy()
    name_to_kernel = {name: kernel[np.ix_(indices, indices)] for name, kernel in catalogue_kernels.items()}
    total = {name: np.zeros(len(draws)) for name in ("B0", "replicate_quarter", "replicate_full", *MODELS)}
    mass = np.zeros(len(draws))
    selection = []
    for outer in range(3):
        train = np.flatnonzero(folds != outer)
        query = np.flatnonzero(folds == outer)
        if len(np.unique(groups[query])) < 3:
            raise ValueError(f"Too few held-out compound groups in fold {outer}")
        predictions = {}
        for model, kernel in name_to_kernel.items():
            alpha = select_alpha(kernel, source, groups, folds, outer)
            predictions[model] = ridge(kernel, source, groups, train, query, alpha)
            selection.append(dict(outer_fold=outer, model=model, alpha=alpha,
                                  train_groups=len(np.unique(groups[train])), query_groups=len(np.unique(groups[query]))))
        sums, count = score_fold(target[query], source[query], predictions, codes[query], draws)
        for name in total:
            total[name] += sums[name]
        mass += count
    return {name: value / mass for name, value in total.items()}, selection


def summary_metrics(metrics, sample_count, groups, line, direction, dose):
    b0 = metrics["B0"]
    ceiling = b0 - metrics["replicate_quarter"]
    eligible = ceiling > 0
    rows = []
    for name, rmse in metrics.items():
        improvement = 100 * (b0 - rmse) / b0
        reach = np.divide(100 * (b0 - rmse), ceiling, out=np.full_like(ceiling, np.nan), where=eligible)
        rmse_ci = np.quantile(rmse[1:], (0.025, 0.975))
        improvement_ci = np.quantile(improvement[1:], (0.025, 0.975))
        reach_boot = reach[1:][eligible[1:]]
        reach_ci = np.quantile(reach_boot, (0.025, 0.975)) if len(reach_boot) else (np.nan, np.nan)
        rows.append(dict(dose_uM=dose / 1000, line=line, direction=direction, arm=name, groups=groups,
                         matched_pairs=sample_count, rmse=rmse[0], rmse_low=rmse_ci[0], rmse_high=rmse_ci[1],
                         b0_relative_reduction_pct=improvement[0], reduction_low=improvement_ci[0],
                         reduction_high=improvement_ci[1], repeat_gain_reached_pct=reach[0],
                         reach_low=reach_ci[0], reach_high=reach_ci[1],
                         positive_repeat_gain_bootstrap_fraction=eligible[1:].mean(),
                         positive_10uM_gate=(bool(improvement_ci[0] > 0)
                                             if dose == 10000 and name in MODELS
                                             and line == "equal_line_mean" and direction == "equal_direction" else None)))
    return rows


def merge_metrics(items):
    return {name: np.mean([values[name] for values in items], axis=0) for name in items[0]}


def study(args):
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Prespecified protocol hash mismatch")
    historical = read_json(args.h_receipt)
    cache = args.cache
    wells = pd.read_csv(cache / "wells.tsv", sep="\t", dtype={"drug": str}).set_index("Combo", verify_integrity=True)
    genes = pd.read_csv(cache / "genes_human.tsv", sep="\t")
    if set(wells.time_point) != {24} or set(wells.line) != set(LINES):
        raise ValueError("Expected exactly the three stage-h 24 h cell lines")
    compounds, universe, fp = chemistry(wells)
    args.output.mkdir(parents=True, exist_ok=False)
    compounds.reset_index().to_csv(args.output / "split.tsv", sep="\t", index=False)
    frozen_split = sha256(args.output / "split.tsv")
    prefit = dict(protocol_sha256=args.protocol_sha256, split_sha256=frozen_split,
                  stage_h_receipt_sha256=sha256(args.h_receipt), sorted_group_order=universe.tolist(),
                  code_sha256=sha256(Path(__file__)))
    write_json(args.output / "freeze.json", prefit)
    chemistry_kernels = kernels(fp)
    rng = np.random.default_rng(SEED)
    draws = np.vstack((np.ones(len(universe)), rng.exponential(size=(DRAWS, len(universe)))))
    catalogue = {drug: drug for drug in compounds.index}
    assessed, matched, controls, selected = {}, {}, {}, []
    for line in LINES:
        cultures = wells.loc[wells.line.eq(line)]
        expression, valid_plates, panel = plate_profiles(cache, line, cultures, len(genes))
        controls[line] = hashlib.sha256("\n".join(genes.id.iloc[panel].astype(str)).encode()).hexdigest()
        if controls[line] != historical["panel_sha256"][line]:
            raise ValueError(f"{line} vehicle panel differs from stage h")
        for dose in DOSES:
            drugs, x, y = paired_dose(line, dose, cultures, expression, valid_plates, catalogue)
            cohort = compounds.loc[drugs]
            available = cohort.status.eq("valid").to_numpy()
            cohort = cohort.iloc[np.flatnonzero(available)]
            x, y = x[available], y[available]
            codes = universe.get_indexer(cohort.connectivity)
            if np.any(codes < 0):
                raise ValueError("A matched group is missing from the frozen split")
            matched[line, dose] = dict(groups=set(cohort.connectivity), pairs=len(cohort), h_pairs=len(drugs))
            for direction, source, target in (("rep1_to_rep2", x, y), ("rep2_to_rep1", y, x)):
                metrics, picked = cross_validate(source, target, cohort, chemistry_kernels, codes, draws)
                assessed[line, dose, direction] = metrics
                selected.extend(dict(line=line, dose_uM=dose / 1000, direction=direction, **entry) for entry in picked)
            print(f"Completed {line} {dose} nM; {len(cohort)}/{len(drugs)} matched catalog pairs", flush=True)
    rows, pair_rows = [], []
    directions = ("rep1_to_rep2", "rep2_to_rep1")
    for dose in DOSES:
        line_means = []
        for line in LINES:
            pair = matched[line, dose]
            pair_rows.append(dict(line=line, dose_uM=dose / 1000, paired_wells_stage_h=pair["h_pairs"],
                                  paired_wells_chemistry=pair["pairs"], parent_groups=len(pair["groups"])))
            values = []
            for direction in directions:
                results = assessed[line, dose, direction]
                rows.extend(summary_metrics(results, pair["pairs"], len(pair["groups"]), line, direction, dose))
                values.append(results)
            line_metrics = merge_metrics(values)
            rows.extend(summary_metrics(line_metrics, pair["pairs"], len(pair["groups"]), line, "equal_direction", dose))
            line_means.append(line_metrics)
        for direction in directions:
            results = merge_metrics([assessed[line, dose, direction] for line in LINES])
            rows.extend(pooled_rows(results, matched, direction, dose))
        rows.extend(pooled_rows(merge_metrics(line_means), matched, "equal_direction", dose))
    pd.DataFrame(rows).to_csv(args.output / "metrics.tsv", sep="\t", index=False)
    pd.DataFrame(selected).to_csv(args.output / "alpha.tsv", sep="\t", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output / "coverage.tsv", sep="\t", index=False)
    inputs = [cache / name for name in ("wells.tsv", "genes_human.tsv")]
    inputs.extend(cache / f"{prefix}_{line}.{ext}" for line in LINES
                  for prefix, ext in (("obs", "parquet"), ("counts", "npz")))
    write_json(args.output / "receipt.json", dict(
        **prefit, cache_sha256={path.name: sha256(path) for path in inputs}, panel_sha256=controls,
        source="GSM4150378 / stage-h R10 QC cache; previously exposed",
        bootstrap=dict(draws=DRAWS, seed=SEED, method="shared exponential parent-group weights and pool recentering"),
        files={path.name: sha256(path) for path in args.output.iterdir()},
    ))


def pooled_rows(metrics, matched, direction, dose):
    entries = [matched[line, dose] for line in LINES]
    return summary_metrics(metrics, sum(x["pairs"] for x in entries),
                           len(set.union(*(x["groups"] for x in entries))),
                           "equal_line_mean", direction, dose)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--h-receipt", type=Path, required=True)
    study(parser.parse_args())


if __name__ == "__main__":
    main()
