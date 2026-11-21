"""Quantify prospective adult HCC benchmark precision from frozen P4 summaries."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.r11_chemical_probe.common import read_json, sha256, write_json


METRICS = ("contrast_rmse", "r2_pooled")
STRATA = ("SNU-423", "HepG2-C3A", "liver")
REFERENCES = ("best-VC", "B0")
PRACTICAL = 0.5
MARGINAL_POWER = 0.90
ALTERNATIVES = (1., 2., 5.)
BLOCK_SD = (0., 0.25, 0.5, 1.)
BLOCK_COUNTS = (8, 12, 16, 24, 32)
PARENT_COUNTS = (48, 96, 192, 384, 768, 1536)


def pinned_table(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"P4 source checksum differs: {path}")
    return pd.read_csv(path, sep="\t")


def calibrate(points, paired):
    points = points.loc[points.cohort.eq("development")].set_index(
        ["dose_uM", "minimum_cells", "stratum", "arm"], verify_integrity=True)
    subset = paired.loc[paired.arm.eq("P4-sibling")].set_index(
        ["dose_uM", "minimum_cells", "stratum", "reference", "metric"], verify_integrity=True)
    records = []
    for stratum in STRATA:
        for minimum in (20, 200):
            slab = (5., minimum, stratum)
            candidate = points.loc[(*slab, "P4-sibling")]
            for comparison in REFERENCES:
                for name in METRICS:
                    row = subset.loc[(*slab, comparison, name)]
                    reference = points.loc[(*slab, row.best_at_point)]
                    difference = candidate[name] - reference[name]
                    if (candidate.parent_groups != reference.parent_groups or (minimum == 20 and candidate.parent_groups != 203)
                            or not np.isclose(difference, row.difference, atol=1e-12, rtol=0)):
                        raise ValueError("Calibration point or parent support does not match the frozen comparison")
                    scale = -100. / reference[name] if name == "contrast_rmse" else 100.
                    gain, (low, high) = difference * scale, sorted(scale * row[["ci99_low", "ci99_high"]])
                    width = max(gain - low, high - gain)
                    records.append(dict(stratum=stratum, minimum_cells=minimum,
                                        comparison=comparison, metric=name, reference=row.best_at_point,
                                        parents=int(candidate.parent_groups), observed_gain=gain,
                                        low99=low, high99=high, conservative_halfwidth=width,
                                        equivalent_parent_sd=width * np.sqrt(candidate.parent_groups) / norm.ppf(.995)))
    return pd.DataFrame(records)


def required_parents(parent_sd, block_sd, blocks, effect):
    quantile = t.ppf(.995, blocks - 1)
    power_z = norm.ppf(MARGINAL_POWER)
    max_se = min((effect - PRACTICAL) / power_z, effect / (quantile + power_z))
    remaining_variance = max_se**2 - block_sd**2 / blocks
    return math.ceil(parent_sd**2 / remaining_variance) if remaining_variance > 0 else math.inf


def pass_probability(parent_sd, block_sd, blocks, parents, effect):
    se = np.sqrt(parent_sd**2 / parents + block_sd**2 / blocks)
    boundary = max(PRACTICAL, t.ppf(.995, blocks - 1) * se)
    return norm.sf((boundary - effect) / se)


def workload(parents, blocks):
    if not math.isfinite(parents):
        return math.inf, math.inf, math.inf
    plates = 2 * max(blocks, math.ceil(3 * parents / 48))
    return 6 * parents, plates, 12 * plates


def design(calibration):
    primary = calibration.loc[calibration.stratum.eq("SNU-423") & calibration.minimum_cells.eq(20)
                              & calibration.comparison.eq("best-VC")]
    rmse_sd, r2_sd = primary.set_index("metric", verify_integrity=True).loc[list(METRICS), "equivalent_parent_sd"]
    minima, grid = [], []
    for effect in ALTERNATIVES:
        for blocks in BLOCK_COUNTS:
            for block_sd in BLOCK_SD:
                minima_rmse, minima_r2 = [required_parents(sd, block_sd, blocks, effect) for sd in (rmse_sd, r2_sd)]
                n = max(minima_rmse, minima_r2, 3)
                treated, plates, controls = workload(n, blocks)
                scenario = dict(true_gain_both=effect, independent_blocks_per_line=blocks, assumed_block_sd=block_sd)
                minima.append(dict(**scenario, minimum_parents_rmse=minima_rmse, minimum_parents_r2=minima_r2,
                                   minimum_parents_both=n, treated_wells_2_lines=treated,
                                   minimum_plates_2_lines=plates, minimum_vehicle_wells_2_lines=controls))
                for parents in PARENT_COUNTS:
                    rmse, r2 = [pass_probability(sd, block_sd, blocks, parents, effect) for sd in (rmse_sd, r2_sd)]
                    treated, plates, controls = workload(parents, blocks)
                    grid.append(dict(**scenario, parents=parents, pass_probability_rmse=rmse, pass_probability_r2=r2,
                                     joint_probability_lower_bound=max(0., rmse + r2 - 1.),
                                     meets_conservative_rule=min(rmse, r2) >= MARGINAL_POWER,
                                     treated_wells_2_lines=treated, minimum_plates_2_lines=plates,
                                     minimum_vehicle_wells_2_lines=controls))
    return pd.DataFrame(minima), pd.DataFrame(grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score_dir = args.source / "scored"
    receipt = read_json(score_dir / "receipt.json")
    if sha256(args.source / "freeze.json") != receipt["freeze_sha256"]:
        raise ValueError("P4 score does not bind the source freeze")
    points, paired = [pinned_table(score_dir / name, receipt["files"][name]) for name in ("points.tsv", "paired.tsv")]
    calibration = calibrate(points, paired)
    minima, grid = design(calibration)
    args.out.mkdir(parents=True, exist_ok=False)
    tables = {"calibration.tsv": calibration, "minima.tsv": minima, "power_grid.tsv": grid}
    for name, table in tables.items():
        table.to_csv(args.out / name, sep="\t", index=False, float_format="%.10g")
    write_json(args.out / "receipt.json", dict(
        protocol_sha256=sha256(args.protocol), code_sha256=sha256(Path(__file__)),
        p4_freeze_sha256=receipt["freeze_sha256"], p4_score_sha256=sha256(score_dir / "receipt.json"),
        measured_expression_opened=False, files={name: sha256(args.out / name) for name in tables},
    ))


if __name__ == "__main__":
    main()
