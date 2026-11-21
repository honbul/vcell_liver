"""Evaluate held-out-well reliability of two fixed Tahoe teacher targets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from src.r11_chemical_probe.common import sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE, table
from src.r13_optimization.objectives import row_products

from .ceilings import macro_weights
from .followup_data import PROTOCOL_SHA, array, load_cache


WELL = KEY[0]
TYPES = ("same_position_other_dose", "other_position_same_dose", "other_position_other_dose")
ARMS = ("shrinkage", "rank4")


def averages(rows, values, indices):
    part = rows.iloc[indices]
    weights = part.n_cells.to_numpy(dtype=float)
    plates = part.groupby("dose_uM").plate_id.transform("nunique").to_numpy()
    index = [part.dose_uM.to_numpy(), part.plate_id.to_numpy()]
    totals = pd.Series(weights).groupby(index).transform("sum").to_numpy()
    return np.average(values[indices], axis=0, weights=weights / totals / plates)


def shrink(rows, values, line, dose):
    idx = np.arange(len(rows))
    partition = [idx[rows[LINE].eq(line)], idx[rows.dose_uM.eq(dose)], idx]
    valid = [i for i, take in enumerate(partition) if len(take)]
    states = [averages(rows, values, partition[i]) for i in valid]
    return np.average(states, axis=0, weights=np.asarray([0.5, 0.25, 0.25])[valid])


def teachers(rows, values):
    rank = min(4, *values.shape)
    u, s, v = randomized_svd(values, n_components=rank, n_iter=7, random_state=130113)
    reduced = (u * s) @ v
    targets = {}
    for line in rows[LINE].unique():
        targets[line] = (shrink(rows, values, line, 0.5), shrink(rows, reduced, line, 0.5))
    return targets


def stats(rows, values):
    rows = rows.reset_index(drop=True)
    weights = macro_weights(rows)
    arms = tuple(values)
    sums = pd.DataFrame(values).multiply(weights, axis=0).groupby(rows[GROUP]).sum()
    observed = sums.mean()
    samples = np.random.default_rng(130113).integers(len(sums), size=(10000, len(sums)))
    boot = sums.to_numpy()[samples].mean(axis=1)
    return observed.to_dict(), {arm: boot[:, i] for i, arm in enumerate(arms)}


def comparison(point, boot, arm, base):
    def effect(candidate, ref):
        return 100 * (1 - candidate / ref)
    if point[base] <= 0 or np.any(boot[base] <= 0):
        return dict(percent=None, ci95=None)
    return dict(percent=float(effect(point[arm], point[base])),
                ci95=np.quantile(effect(boot[arm], boot[base]), [0.025, 0.975]).tolist())


def summarize(results, matching):
    part = results.loc[results.matching.eq(matching)].reset_index(drop=True)
    groups = int(part[GROUP].nunique())
    counts = dict(groups=groups, rows=len(part), wells=int(part[WELL].nunique()),
                  paired_wells=len(set(zip(part[WELL], part.donor_well))))
    if not groups:
        return counts
    available = part[["donor_lines", "donor_doses", "donor_plates"]].agg(["min", "median", "max"]).to_dict()
    cols = ["b0", "single", *ARMS, "single_cosine", *(f"{arm}_cosine" for arm in ARMS)]
    point, boot = stats(part, {name: part[name].to_numpy() for name in cols})
    scores = {arm: {base: comparison(point, boot, arm, base) for base in ("b0", "single")} for arm in ARMS}
    return dict(**counts, point=point, scores=scores, teacher_support=available,
                cosine_ci95={a: np.quantile(boot[a], [0.025, 0.975]).tolist() for a in cols if a.endswith("_cosine")})


def classify_donors(other, position):
    same = other.barcode_position.eq(position)
    dose = other.dose_uM.eq(0.5)
    return (~dose & same, dose & ~same, ~dose & ~same)


def fold_results(rows, values):
    records = []
    for _, group in rows.groupby(GROUP, sort=True):
        for plate, query in group.loc[group.dose_uM.eq(0.5)].groupby("plate_id", sort=True):
            pool = group.loc[group.plate_id.ne(plate)]
            if pool.empty or not set(query[LINE]) & set(pool[LINE]):
                continue
            prediction = teachers(pool.reset_index(drop=True), values[pool.index])
            coverage = dict(donor_lines=pool[LINE].nunique(), donor_doses=pool.dose_uM.nunique(),
                            donor_plates=pool.plate_id.nunique())
            for i, target in query.iterrows():
                other = pool.loc[pool[LINE].eq(target[LINE])].copy()
                if other.empty:
                    continue
                other["distance"] = np.abs(np.log10(other.dose_uM / 0.5))
                other = other.sort_values(["distance", "barcode_position", WELL])
                y = values[i:i+1]
                for category, mask in zip(TYPES, classify_donors(other, target.barcode_position)):
                    if not mask.any():
                        continue
                    j = other.index[mask][0]
                    donor, direct = pool.loc[j, WELL], values[j]
                    guesses = np.vstack((direct, *prediction[target[LINE]]))
                    match = dict(**coverage, matching=category, donor_well=donor,
                                 b0=float(np.linalg.norm(y) / np.sqrt(y.size)))
                    for arm, guess in zip(("single", *ARMS), guesses):
                        p = guess[None, :]
                        match[arm] = float(np.sqrt(row_products(y - 0.25 * p, y - 0.25 * p)[0]))
                        norm = np.linalg.norm(y) * np.linalg.norm(p)
                        match[f"{arm}_cosine"] = float((y @ guess)[0] / norm) if norm > 1e-14 else 0.
                    records.append({**target[[WELL, GROUP, LINE, "compound_entity_id", "n_cells"]].to_dict(), **match})
    return pd.DataFrame.from_records(records)


def run(args):
    load_cache(args.cache, args.protocol)
    frames = []
    for fold in range(3):
        folder = args.cache / f"fold{fold}"
        frames.append(fold_results(table(folder / "rows.tsv"), array(folder, "target_all")))
        print(f"validated teacher on fold {fold}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    reports = {match: summarize(result, match) for match in TYPES}
    gate = reports["other_position_same_dose"]
    passing = [
        arm for arm in ARMS if gate["groups"] >= 10 and
        all(gate["scores"][arm][base]["ci95"] is not None and
            gate["scores"][arm][base]["ci95"][0] > threshold
            for base, threshold in (("single", 1), ("b0", 0))
        )
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "teacher.json", dict(protocol_sha256=PROTOCOL_SHA, comparisons=reports,
               cache_receipt_sha256=sha256(args.cache / "receipt.json"), passing=passing,
               selected_teacher=passing[0] if passing else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "protocol", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
