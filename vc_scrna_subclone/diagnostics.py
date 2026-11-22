"""Audit frozen patient-cell scores with donor-level inference."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .cohorts import file_record

TECHNICAL = ["umi", "detected_genes", "mitochondrial_fraction", "s_phase", "g2m_phase"]
IDENTITIES = ["id", "donor", "source_sample", "site", "cohort"]


def ranks_to_residuals(y, design):
    if np.ptp(y) == 0:
        return np.zeros(len(y), dtype=np.float64)
    x = np.column_stack([np.ones(len(y)), design])
    solution = np.linalg.lstsq(x, y, rcond=None)[0]
    return y - x @ solution


def donor_partial_correlations(frame):
    records = []
    for donor, group in frame.groupby("donor", sort=True):
        if len(group) < 30:
            continue
        scores = rankdata(group.resistance.to_numpy())
        tech = rankdata(group[TECHNICAL].to_numpy(), axis=0)
        samples = pd.get_dummies(group.source_sample, drop_first=True).to_numpy()
        for i, name in enumerate(TECHNICAL):
            others = np.delete(tech, i, axis=1)
            design = np.column_stack([samples, others])
            rx = ranks_to_residuals(tech[:, i], design)
            ry = ranks_to_residuals(scores, design)
            scale = np.linalg.norm(rx) * np.linalg.norm(ry)
            if scale > 1e-8:
                records.append({"donor": donor, "covariate": name, "cells": len(group),
                                "partial_rho": float(np.dot(rx, ry) / scale)})
    return pd.DataFrame(records)


def summarize_effect(values, rng):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if not n:
        return {"donors": 0, "mean": None, "ci95": [None, None], "sign_flip_p": None}
    average = float(values.mean())
    boot = values[rng.integers(0, n, size=(3000, n))].mean(axis=1)
    bounds = np.quantile(boot, [0.025, 0.975]).tolist() if n > 1 else [None, None]
    if n <= 15:
        signs = np.asarray(list(product((-1., 1.), repeat=n)))
        null = signs @ values / n
        p = float(np.mean(np.abs(null) >= abs(average) - 1e-12))
    else:
        signs = rng.choice((-1., 1.), size=(9999, n))
        null = signs @ values / n
        p = float((1 + np.sum(np.abs(null) >= abs(average) - 1e-12)) / (1 + len(null)))
    return {"donors": n, "mean": average, "ci95": bounds, "sign_flip_p": p}


def paired_site_contrast(frame, threshold, site_col, first, second, min_cells):
    data = frame.assign(high_resistance=frame.resistance.ge(threshold))
    counts = (data.groupby(["donor", site_col]).high_resistance
              .agg(n_cells="size", fraction="mean").reset_index())
    qualified = counts.loc[counts.n_cells.ge(min_cells)]
    pairs = qualified.pivot(index="donor", columns=site_col, values="fraction")
    if first not in pairs or second not in pairs:
        return pd.DataFrame(columns=["donor", "difference"])
    result = (pairs[second] - pairs[first]).dropna().rename("difference")
    return result.reset_index()


def load_frozen(path, receipt):
    evidence = json.loads(receipt.read_text())
    if evidence["predictions"] != file_record(path):
        raise ValueError("Inference table differs from its frozen receipt")
    data = pd.read_parquet(path)
    if not {"id", "resistance", "analysis_role"} <= set(data) or data.id.duplicated().any():
        raise ValueError("Invalid forecast schema")
    if not data.analysis_role.eq("held_out").all() or not np.isfinite(data.resistance).all():
        raise ValueError("Use one independently held-out prediction per scored cell")
    return data[["id", "resistance"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("predictions", "receipt", "metadata", "output"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True,
                   help="Resistance threshold previously selected using fitting roles")
    p.add_argument("--threshold-receipt", type=Path, required=True)
    p.add_argument("--seed", type=int, default=51226)
    p.add_argument("--minimum-site-cells", type=int, default=20)
    a = p.parse_args()
    threshold_evidence = json.loads(a.threshold_receipt.read_text())
    if threshold_evidence["threshold"] != a.threshold or threshold_evidence["role"] != "training":
        raise ValueError("Threshold is absent from the frozen training decision")
    data = load_frozen(a.predictions, a.receipt)
    meta = pd.read_parquet(a.metadata)
    expected = set(IDENTITIES + TECHNICAL)
    if not expected <= set(meta) or meta.id.duplicated().any():
        raise ValueError("Invalid donor and QC table")
    scored = data.merge(meta.loc[:, IDENTITIES + TECHNICAL], on="id", validate="one_to_one")
    if len(scored) != len(data):
        raise ValueError("No QC identity for at least one forecast")
    ready = np.isfinite(scored[TECHNICAL]).all(axis=1)
    donor_rho = donor_partial_correlations(scored.loc[ready])
    rng = np.random.default_rng(a.seed)
    covariates = {}
    for name in TECHNICAL:
        subset = donor_rho.loc[donor_rho.covariate.eq(name), "partial_rho"] if len(donor_rho) else []
        covariates[name] = summarize_effect(subset, rng)
    liver = scored.loc[scored.cohort.eq("GSE149614")]
    contrasts = {}
    for site in ("PVTT", "Lymph"):
        paired = paired_site_contrast(liver, a.threshold, "site", "Tumor",
                                      site, a.minimum_site_cells)
        contrasts[site] = summarize_effect(paired.difference.to_numpy(dtype=float), rng)
    report = {
        "inputs": {p.name: file_record(p) for p in (a.predictions, a.receipt, a.metadata, a.threshold_receipt)},
        "held_out_cells": len(scored), "held_out_donors": int(scored.donor.nunique()),
        "missing_qc_cells": int((~ready).sum()), "threshold": a.threshold,
        "technical_partial_rho": covariates, "paired_primary_to_site_resistance_fraction": contrasts,
        "inference_unit": "donor",
        "qc_analysis": "Within-donor rank partial correlation adjusted for other QC variables and sample",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
