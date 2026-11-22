"""Compare adult HCC study-level compound orders on an exact registered panel."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .hcc_3d_study_transfer import recipient_outcomes
from .hcc_study_pool import registered
from .io import publish_json, sha256
from .limore_corrected import OBSERVATIONS
from .yang_hcc_external import require_hash

REGISTRY = "bd27132354dcf5537b717227099147514851ec0218d0c96920ec4c8a00a96a4c"
LIMORE_ROWS = "f6a9ae3e30ca5200b02b7a69b415dd8ef643e909c7a3585686cee4dca690fbe3"
WONG_BOOK = "724e0276c474c8afbdacb734c291f53e0399a77a4e473b5ae57751dd637841e7"
EXPECTED = {"LIMORE": 50, "LICOB": 38, "Wong": 10}
REPEATS = 2000


def matrix(frame, panel, name):
    frame = frame[frame.compound_id.isin(panel)]
    if frame.duplicated(["cell_id", "compound_id"]).any():
        raise ValueError(f"{name} contains duplicated model–compound pairs")
    wide = frame.groupby(["lineage_group", "compound_id"]).endpoint_value.mean().unstack()
    wide = wide.reindex(columns=panel)
    if wide.shape != (EXPECTED[name], 21) or not np.isfinite(wide).all(axis=None):
        raise ValueError(f"{name} lacks a complete measurement matrix on registered support")
    return wide.to_numpy()


def correlation(first, second):
    first = rankdata(first, axis=-1)
    second = rankdata(second, axis=-1)
    first -= first.mean(axis=-1, keepdims=True)
    second -= second.mean(axis=-1, keepdims=True)
    scale = np.sqrt(np.square(first).sum(axis=-1) * np.square(second).sum(axis=-1))
    if (scale == 0).any():
        raise ValueError("Undefined correlation for constant compound ranks")
    return (first * second).sum(axis=-1) / scale


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def signs(values, indices):
    return np.sign(values[..., indices[:, 0]] - values[..., indices[:, 1]])


def agreement(first, second):
    return (.5 + .5 * first * second).mean(axis=-1)


def comparisons(matrices, labels):
    random = np.random.default_rng(62305)
    indices = np.array(list(combinations(range(len(labels)), 2)))
    centers, draws, center_sign, draw_sign, stable = {}, {}, {}, {}, {}
    for name, values in matrices.items():
        n = len(values)
        centers[name] = values.mean(axis=0)
        draws[name] = values[random.integers(n, size=(REPEATS, n))].mean(axis=1)
        center_sign[name] = signs(centers[name], indices)
        draw_sign[name] = signs(draws[name], indices)
        stable[name] = (
            (draw_sign[name] == center_sign[name]).mean(axis=0) >= .975
        ) & (center_sign[name] != 0)

    results, boot = {}, {}
    for a, b in combinations(EXPECTED, 2):
        key = f"{a}:{b}"
        boot[key] = correlation(draws[a], draws[b])
        both = stable[a] & stable[b]
        opposite = both & (center_sign[a] != center_sign[b])
        top = [np.argsort(-centers[name], kind="stable")[:5] for name in (a, b)]
        results[key] = {
            "spearman": float(correlation(centers[a], centers[b])),
            "spearman_ci95": interval(boot[key]),
            "pair_agreement": float(agreement(center_sign[a], center_sign[b])),
            "pair_agreement_ci95": interval(agreement(draw_sign[a], draw_sign[b])),
            "top5_overlap": len(set(top[0]) & set(top[1])),
            "stable_in_both": int(both.sum()),
            "stable_reversals": [[labels[i], labels[j]] for i, j in indices[opposite]],
        }

    differences = {
        key: {"difference": results["LICOB:Wong"]["spearman"] - results[key]["spearman"],
              "ci95": interval(boot["LICOB:Wong"] - boot[key])}
        for key in ("LIMORE:LICOB", "LIMORE:Wong")
    }
    return {
        "comparisons": results,
        "3d_pair_minus": differences,
        "3d_rank_advantage": all(value["ci95"][0] > 0 for value in differences.values()),
        "top5": {name: [labels[i] for i in np.argsort(-center, kind="stable")[:5]]
                 for name, center in centers.items()},
        "stable_pairs": {name: int(mask.sum()) for name, mask in stable.items()},
        "cohort_compound_means": {name: dict(zip(labels, center.tolist())) for name, center in centers.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "limore", "limore-rows", "licob", "wong", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if registered(args.inputs) != REGISTRY:
        raise ValueError("Unregistered HCC study panel")
    require_hash(args.limore, OBSERVATIONS)
    require_hash(args.limore_rows, LIMORE_ROWS)
    require_hash(args.wong, WONG_BOOK)
    licob = pd.read_parquet(args.inputs / "LICOB.parquet")
    wong = pd.read_parquet(args.inputs / "Wong.parquet")
    names = wong[["compound_id", "source_compound_label"]].drop_duplicates().set_index(
        "compound_id", verify_integrity=True,
    ).source_compound_label.sort_index()
    if len(names) != 21 or set(names.index) != set(licob.compound_id):
        raise ValueError("Unexpected registered 21-parent panel")
    limore_rows = pd.read_parquet(
        args.limore_rows, columns=["record_id", "cell_id", "cell_group", "compound_id"],
    ).rename(columns={"cell_group": "lineage_group"})
    limore_values = pd.read_parquet(args.limore, columns=["record_id", "endpoint_value"])
    limore = limore_rows.merge(limore_values, on="record_id", how="outer", validate="one_to_one")
    if len(limore) != 3400 or limore.isna().any(axis=None):
        raise ValueError("LIMORE labels disagree with registered 50-line support")
    frames = {
        "LIMORE": limore,
        "LICOB": recipient_outcomes("LICOB", licob, args.licob),
        "Wong": recipient_outcomes("Wong", wong, args.wong),
    }
    matrices = {name: matrix(frame, names.index, name) for name, frame in frames.items()}
    publish_json(args.out, comparisons(matrices, names.tolist()) | {
        "cohort_groups": EXPECTED, "bootstrap_draws": REPEATS, "panel": names.to_dict(),
        "input_hashes": {"registry": REGISTRY, "LIMORE": OBSERVATIONS, "LIMORE_rows": LIMORE_ROWS,
                         "LICOB": sha256(args.licob), "Wong": WONG_BOOK},
    })


if __name__ == "__main__":
    main()
