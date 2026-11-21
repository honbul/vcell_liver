"""Measure Sci-Plex3 replicate well reliability without fitting a response model."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.r11_chemical_probe.common import sha256, write_json


LINES = ("A549", "K562", "MCF7")
DOSES = (10, 100, 1000, 10000)
N_GENES = 2000
MIN_CELLS = 20
DRAWS = 10000
SEED = 20261121


def plate_profiles(cache, line, wells, n_genes):
    obs = pd.read_parquet(cache / f"obs_{line}.parquet")
    if not obs.cell.is_unique:
        raise ValueError(f"Duplicate cell IDs in {line}")
    keys = ["culture_plate", "replicate", "drug", "vehicle", "dose"]
    expected = wells.loc[obs.Combo, keys].reset_index(drop=True)
    for col in keys:
        if not np.array_equal(obs[col].to_numpy(), expected[col].to_numpy()):
            raise ValueError(f"Cell and well metadata disagree for {line} {col}")
    indices = wells.index.get_indexer(obs.Combo)
    n_cells = np.bincount(indices, minlength=len(wells))
    if not np.array_equal(n_cells, wells.n_qc.to_numpy()):
        raise ValueError(f"QC counts do not match the R10 well receipt for {line}")
    membership = sparse.csr_matrix(
        (np.ones(len(obs), dtype=np.int32), (indices, np.arange(len(obs)))),
        shape=(len(wells), len(obs)),
    )
    counts = sparse.load_npz(cache / f"counts_{line}.npz")
    if counts.shape != (len(obs), n_genes):
        raise ValueError(f"Count shape does not match cell and gene metadata for {line}")
    if not np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), obs.n_umi_human.to_numpy()):
        raise ValueError(f"Count rows do not match the R10 cell metadata for {line}")
    values = (membership @ counts).toarray().astype(np.float64)
    totals = values.sum(axis=1, keepdims=True)
    log_cp10k = np.log1p(values * (10000.0 / np.maximum(totals, 1)))
    controls = wells.vehicle.to_numpy() & (totals[:, 0] > 0)
    if controls.sum() < 2:
        raise ValueError(f"Insufficient vehicle wells to select a panel for {line}")
    variance = log_cp10k[controls].var(axis=0)
    panel = np.argsort(-variance, kind="stable")[:N_GENES]
    if len(panel) != N_GENES or variance[panel[-1]] <= 0:
        raise ValueError(f"Only {int((variance > 0).sum())} variable genes in {line} vehicle wells")
    profile = log_cp10k[:, panel].copy()
    valid_plate = np.zeros(len(wells), dtype=bool)
    for _, ix in wells.groupby("culture_plate", sort=False).indices.items():
        ctrl = ix[controls[ix]]
        if len(ctrl) < 2:
            continue
        profile[ix] -= profile[ctrl].mean(axis=0)
        valid_plate[ix] = True
    return profile, valid_plate, panel


def paired_dose(line, dose, wells, values, valid_plate, identities):
    treated = wells.loc[
        ~wells.vehicle & wells.dose.eq(dose) & wells.n_qc.ge(MIN_CELLS) & valid_plate
    ].copy()
    treated["row"] = wells.index.get_indexer(treated.index)
    a = treated.loc[treated.replicate.eq("rep1")].set_index("drug", verify_integrity=True)
    b = treated.loc[treated.replicate.eq("rep2")].set_index("drug", verify_integrity=True)
    matched = a.index.intersection(b.index).sort_values()
    if len(matched) < 3:
        raise ValueError(f"Insufficient matched drugs for {line} {dose} nM")
    a, b = a.loc[matched], b.loc[matched]
    if set(a.culture_plate) & set(b.culture_plate):
        raise ValueError(f"Culture plates occur in both replicate sets for {line} {dose} nM")
    x, y = values[a.row.to_numpy()], values[b.row.to_numpy()]
    return np.asarray([identities[i] for i in matched]), x, y


def centered_square(kernel, weight):
    mean = weight @ kernel
    return np.maximum(np.diag(kernel) - 2 * mean + (mean * weight).sum(axis=1, keepdims=True), 0)


def evaluate(groups, x, y, draws, universe):
    codes = universe.get_indexer(groups)
    if (codes < 0).any():
        raise ValueError("Matched group missing from the draw universe")
    weight = draws[:, codes] / np.bincount(codes, minlength=len(universe))[codes]
    weight /= weight.sum(axis=1, keepdims=True)
    xx = centered_square(x @ x.T / N_GENES, weight)
    yy = centered_square(y @ y.T / N_GENES, weight)
    diff = x - y
    delta = centered_square(diff @ diff.T / N_GENES, weight)
    b0 = (weight * 0.5 * (np.sqrt(xx) + np.sqrt(yy))).sum(axis=1)
    rep = (weight * np.sqrt(delta)).sum(axis=1)
    xy = (xx + yy - delta) / 2
    reduced_a_to_b = np.sqrt(np.maximum(yy + 0.25**2 * xx - 0.5 * xy, 0))
    reduced_b_to_a = np.sqrt(np.maximum(xx + 0.25**2 * yy - 0.5 * xy, 0))
    reduced = (weight * 0.5 * (reduced_a_to_b + reduced_b_to_a)).sum(axis=1)
    valid = np.sqrt(xx * yy) > 1e-15
    cos = np.divide(xy, np.sqrt(xx * yy), out=np.zeros_like(xx), where=valid)
    coverage = (weight * valid).sum(axis=1)
    align = np.divide((weight * np.clip(cos, -1, 1)).sum(axis=1), coverage,
                      out=np.full(len(coverage), np.nan), where=coverage > 0)
    return {"b0_rmse": b0, "replicate_rmse": rep, "replicate_quarter_rmse": reduced,
            "cosine": align, "cosine_coverage": coverage}


def summarize(line, dose, groups, result):
    b0, rep = result["b0_rmse"], result["replicate_rmse"]
    result["rmse_reduction"] = b0 - rep
    result["relative_rmse_reduction_pct"] = 100 * (b0 - rep) / b0
    result["relative_quarter_rmse_reduction_pct"] = 100 * (b0 - result["replicate_quarter_rmse"]) / b0
    summary = {"line": line, "dose_nM": dose, "dose_uM": dose / 1000,
               "groups": len(set(groups)), "paired_wells": len(groups)}
    for key, values in result.items():
        low, high = np.nanquantile(values[1:], [0.025, 0.975])
        summary.update({key: float(values[0]), f"{key}_lower95": float(low), f"{key}_upper95": float(high)})
    return summary


def study(args):
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Prespecified protocol hash mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    cache = args.cache
    wells = pd.read_csv(cache / "wells.tsv", sep="\t", dtype={"drug": str}).set_index("Combo", verify_integrity=True)
    genes = pd.read_csv(cache / "genes_human.tsv", sep="\t")
    if set(wells.time_point) != {24} or set(wells.line) != set(LINES):
        raise ValueError("R10 count cache differs from the intended 24 h cell lines")
    identities = {}
    for drug, rows in wells.loc[~wells.vehicle].groupby("drug"):
        smiles = rows.SMILES.dropna().unique()
        if len(smiles) > 1:
            raise ValueError(f"Multiple released SMILES for one catalog drug {drug}")
        identities[drug] = f"smiles:{smiles[0]}" if len(smiles) else f"catalog:{drug}"
    paired, panels = {}, {}
    for line in LINES:
        line_wells = wells.loc[wells.line.eq(line)]
        profiles, valid, panel = plate_profiles(cache, line, line_wells, len(genes))
        panels[line] = genes.id.iloc[panel].astype(str).tolist()
        for dose in DOSES:
            paired[line, dose] = paired_dose(line, dose, line_wells, profiles, valid, identities)
    universe = pd.Index(sorted(set(identities.values())))
    bootstrap = np.ones((DRAWS + 1, len(universe)))
    bootstrap[1:] = np.random.default_rng(SEED).exponential(size=(DRAWS, len(universe)))
    rows = []
    for dose in DOSES:
        line_metrics, all_groups = [], []
        for line in LINES:
            groups, x, y = paired[line, dose]
            values = evaluate(groups, x, y, bootstrap, universe)
            line_metrics.append(values)
            all_groups.extend(groups)
        metrics = {key: np.mean([m[key] for m in line_metrics], axis=0) for key in line_metrics[0]}
        rows.append(summarize("equal_line_mean", dose, all_groups, metrics))
        for line, values in zip(LINES, line_metrics):
            rows.append(summarize(line, dose, paired[line, dose][0], values))
    inputs = [cache / name for name in ["wells.tsv", "genes_human.tsv", "b0_summary.json"]]
    inputs += [cache / f"{prefix}_{line}.{ext}" for line in LINES
               for prefix, ext in (("obs", "parquet"), ("counts", "npz"))]
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output / "by_dose.tsv", sep="\t", index=False)
    write_json(args.output / "receipt.json", {
        "source": "GSM4150378 Sci-Plex3 R10 raw-derived QC count cache (previously exposed)",
        "cache_sha256": {str(path): sha256(path) for path in inputs},
        "protocol_sha256": args.protocol_sha256,
        "code_sha256": sha256(Path(__file__)),
        "panel_sha256": {line: hashlib.sha256("\n".join(ids).encode()).hexdigest() for line, ids in panels.items()},
        "bootstrap": {"draws": DRAWS, "seed": SEED},
        "summary_sha256": sha256(args.output / "by_dose.tsv"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    study(parser.parse_args())


if __name__ == "__main__":
    main()
