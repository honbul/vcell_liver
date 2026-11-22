"""Reconstruct zero-interaction potency from released Sanger dose matrices."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

from numba import njit
import numpy as np
import pandas as pd

from .splits import record


KEY = ["BARCODE", "DRUGSET_ID", "cmatrix"]
WELLS = ["lib1_conc", "lib2_conc", "lib1_y", "lib2_y", "inhibition"]
STUDY = re.compile(r"combo_well_stats_(GDSC_\d{3}|Sandpiper-\d+)_\d+\w+\d+\.csv")


@njit(cache=True, nogil=True)
def curve_fit(dose, y, fix_zero, fixed_bottom=0.):
    """Profile a bounded log-logistic curve over its potency and slope."""
    logd = np.empty(len(dose))
    logd[0] = -np.inf
    logd[1:] = np.log(dose[1:])
    if np.max(y) - np.min(y) < 1e-6:
        return np.full_like(y, y[0])
    center_lo, center_hi = logd[1] - 1.4, logd[-1] + 1.4
    hill_lo, hill_hi = .18, 5.
    basis = np.zeros(len(y))
    best_pred = y.copy()
    best_loss, best_c, best_h = 1e6, 0., 1.
    for _ in range(3):
        centers, slopes = np.linspace(center_lo, center_hi, 11), np.linspace(hill_lo, hill_hi, 7)
        for center in centers:
            for slope in slopes:
                basis[1:] = 1. / (1. + np.exp(np.minimum(40., slope * (center - logd[1:]))))
                if fix_zero:
                    bottom = fixed_bottom
                    amplitude = np.dot(basis, y - bottom) / max(np.dot(basis, basis), 1e-8)
                else:
                    residual_basis = basis - np.mean(basis)
                    amplitude = np.dot(residual_basis, y - np.mean(y)) / max(
                        np.dot(residual_basis, residual_basis), 1e-8,
                    )
                    bottom = np.minimum(1.25, np.maximum(-.25, np.mean(y) - amplitude * np.mean(basis)))
                plateau = np.minimum(1.25, np.maximum(-.25, bottom + amplitude))
                predicted = bottom + (plateau - bottom) * basis
                loss = np.sum((predicted - y) ** 2)
                if loss < best_loss:
                    best_loss, best_c, best_h = loss, center, slope
                    best_pred = predicted
        center_step = (center_hi - center_lo) / 10
        slope_step = (hill_hi - hill_lo) / 6
        center_lo, center_hi = best_c - center_step, best_c + center_step
        hill_lo, hill_hi = max(.12, best_h - slope_step), min(8., best_h + slope_step)
    return best_pred


@njit(cache=True, nogil=True)
def zip_score(dose_a, dose_b, inhibition, single_a, single_b):
    dose_a = np.concatenate((np.zeros(1), dose_a))
    dose_b = np.concatenate((np.zeros(1), dose_b))
    mono_a = np.concatenate((np.zeros(1), single_a))
    mono_b = np.concatenate((np.zeros(1), single_b))
    fit_a = curve_fit(dose_a, mono_a, False)[1:]
    fit_b = curve_fit(dose_b, mono_b, False)[1:]
    by_a = np.empty_like(inhibition)
    by_b = np.empty_like(inhibition)
    for j in range(len(single_b)):
        y = np.concatenate((single_b[j:j+1], inhibition[:, j]))
        by_a[:, j] = curve_fit(dose_a, y, True, single_b[j])[1:]
    for i in range(len(single_a)):
        y = np.concatenate((single_a[i:i+1], inhibition[i, :]))
        by_b[i, :] = curve_fit(dose_b, y, True, single_a[i])[1:]
    null = fit_a.reshape((len(fit_a), 1)) + fit_b - fit_a.reshape((len(fit_a), 1)) * fit_b
    delta = .5 * (by_a + by_b) - null
    return 100 * np.mean(delta)


def matrices(archive, entry):
    # The archive is ordered by plate, combination and matrix. Retain any
    # trailing incomplete block at the boundary of each input chunk.
    carried = None
    with ZipFile(archive) as z, z.open(entry) as f:
        for chunk in pd.read_csv(f, usecols=[*KEY, *WELLS], chunksize=100_000, dtype=dict.fromkeys(KEY, str)):
            if carried is not None:
                chunk = pd.concat([carried, chunk], ignore_index=True)
            last = chunk.iloc[-1]
            tail = (chunk.BARCODE.eq(last.BARCODE) & chunk.DRUGSET_ID.eq(last.DRUGSET_ID)
                    & chunk.cmatrix.eq(last.cmatrix))
            carried = chunk[tail]
            yield from chunk[~tail].groupby(KEY, sort=False)
        if carried is not None:
            yield from carried.groupby(KEY, sort=False)


def per_study(archive, entry):
    part = STUDY.fullmatch(entry).group(1).lower().replace("_", "-")
    ready, failed = [], {}
    seen = set()
    for key, matrix in matrices(archive, entry):
        name = "Sanger:" + part + ":" + ":".join(key)
        if key in seen:
            raise ValueError(f"A Sanger dose matrix reopened after its registered block: {name}")
        seen.add(key)
        if matrix[WELLS].isna().any().any():
            failed["missing_wells"] = failed.get("missing_wells", 0) + 1
            continue
        a, b, one_a, one_b, measured = matrix[WELLS].to_numpy(dtype=np.float64).T
        aa, ai = np.unique(a, return_inverse=True)
        bb, bi = np.unique(b, return_inverse=True)
        ix = ai * len(bb) + bi
        if (min(len(aa), len(bb)) < 4 or aa[0] <= 0 or bb[0] <= 0
                or len(np.unique(ix)) != len(aa) * len(bb)):
            failed["incomplete_matrix"] = failed.get("incomplete_matrix", 0) + 1
            continue
        if not np.isfinite(matrix[WELLS].to_numpy()).all():
            failed["nonfinite_wells"] = failed.get("nonfinite_wells", 0) + 1
            continue
        # Publisher's observed matched-dose control values are present
        # as columns on combination wells. DMSO is the normalized zero.
        measured = (np.bincount(ix, weights=measured) / np.bincount(ix)).reshape(len(aa), len(bb))
        ma = np.bincount(ai, weights=one_a) / np.bincount(ai)
        mb = np.bincount(bi, weights=one_b) / np.bincount(bi)
        value = zip_score(aa, bb, measured, ma, mb)
        ready.append((name, value, len(aa), len(bb)))
    return ready, {"study": part, "matrices": len(seen), "scored": len(ready), "excluded": failed}


def run(a):
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    with ZipFile(a.wells) as z:
        entries = [p for p in z.namelist() if STUDY.fullmatch(p)]
    if len(entries) != 5:
        raise ValueError("Expected five complete Sanger well-level source studies")
    # Compiles once before the five worker threads enter the fitting loop.
    x = np.geomspace(.01, 10, 7)
    y = np.linspace(.01, .90, 7)
    zip_score(x, x, np.outer(y, y), y, y)
    with ThreadPoolExecutor(max_workers=min(5, a.workers)) as pool:
        futures = [pool.submit(per_study, a.wells, entry) for entry in entries]
        results = [future.result() for future in futures]
    export = pd.DataFrame([row for rows, _ in results for row in rows],
                          columns=["id", "zip", "dose_levels_a", "dose_levels_b"])
    if not export.id.is_unique or not len(export) or not np.isfinite(export.zip).all():
        raise ValueError("Reconstructed ZIP does not map to unique source assays")
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        stage = Path(d)
        file = stage / "sanger_zip.parquet"
        export.to_parquet(file, index=False)
        receipt = {
            "well_archive": record(a.wells), "studies": [audit for _, audit in results], "files": {file.name: record(file)},
            "algorithm": "Bounded four-parameter log-logistic single agent and conditional curves, "
                         "SynergyFinder ZIP reference and two-directional fit mean, reconstructed from "
                         "matched-dose observed source mono values and normalized zero control.",
            "primary_method": "https://doi.org/10.1016/j.csbj.2015.09.001",
            "author_implementation_reviewed": "https://github.com/bioc/synergyfinder/tree/90ba7b94c23d0560e8759dde2e7c3e2fd81bbb7c",
        }
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wells", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    if a.workers < 1:
        p.error("At least one worker is required")
    run(a)


if __name__ == "__main__":
    main()
