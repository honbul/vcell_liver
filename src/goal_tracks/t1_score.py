"""Separately score SHA-verified Tahoe forecasts on untouched evaluation folds."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, LINE, group_weights
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.objectives import center, row_products
from src.r6_followup_v1.metrics import grams

from .t1_data import DOSES, LIVER, MIN_CELLS, MIN_GROUPS, PROTOCOL_SHA256, array, eligibility, slice_name, table, validate_protocol
from .t1_io import load_task, score_truth
from .t1_predict import REQUIRED


STRATA = ("all", "liver", "HepG2-C3A", "SNU-423")
STRATUM_LINES = {STRATA[1]: LIVER, STRATA[2]: (LIVER[0],), STRATA[3]: (LIVER[1],)}
DRAW_COUNT = 10_000


def registry(stage):
    return sorted(map(int, read_json(stage / "split.json")["outer"]))


def axis_draw(groups):
    draw = np.random.default_rng(130113).exponential(size=(DRAW_COUNT, len(groups)))
    draw /= draw.mean(axis=1, keepdims=True)
    return pd.Index(groups), draw


def point_numerator(weights_by_group, group_index, draw):
    return draw @ np.bincount(group_index, weights=weights_by_group, minlength=draw.shape[1])


def centered_error_grams(metric, prediction, observed):
    return grams(prediction.astype(np.float64) - observed, metric.pools)


def add_centered_bootstrap(rows, metric, kernel, draw, gi, weights, labels, result):
    for grain, ix, center_weight in zip(kernel, metric.pools, metric.centers):
        positive = draw[:, gi[ix]]
        normalized = positive * center_weight
        normalized /= normalized.sum(axis=1, keepdims=True)
        products = normalized @ grain
        variance = np.diag(grain) - 2 * products + (products * normalized).sum(axis=1, keepdims=True)
        weighted = positive * np.sqrt(np.maximum(variance, 0))
        line = rows[LINE].iloc[ix[0]]
        result[0] += weighted @ weights[0, ix]
        if line in LIVER:
            result[1] += weighted @ weights[1, ix]
            i = 2 + LIVER.index(line)
            result[i] += weighted @ weights[i, ix]


def search(metric, prediction, observed):
    top = np.zeros(len(metric.rows))
    for ix in metric.pools:
        names, labels = np.unique(metric.rows.iloc[ix][GROUP], return_inverse=True)
        p, t = prediction[ix], observed[ix]
        true = np.stack([t[labels == g].mean(axis=0) for g in range(len(names))])
        p_norm = np.linalg.norm(p, axis=1)
        t_norm = np.linalg.norm(true, axis=1)
        similarities = p @ true.T / np.maximum(1e-20, p_norm[:, None] * t_norm[None, :])
        tied = np.isclose(similarities, similarities.max(axis=1, keepdims=True), atol=1e-10, rtol=1e-8)
        top[ix] = tied[np.arange(len(ix)), labels] / tied.sum(axis=1)
    return top


class ScoreBook:
    """Shared exact R13-style macro scores and paired re-centered bootstrap."""

    def __init__(self, dose, minimum_cells, groups=None):
        self.dose, self.minimum_cells = dose, minimum_cells
        self.groups = groups
        if groups is not None:
            self.group_index, self.draw = axis_draw(groups)
        self.points = defaultdict(lambda: np.zeros(4))
        self.boot = defaultdict(lambda: np.zeros(DRAW_COUNT))
        self.mass = defaultdict(float)
        self.mass_boot = defaultdict(lambda: np.zeros(DRAW_COUNT))
        self.row_counts = defaultdict(int)
        self.group_counts = defaultdict(set)

    def add(self, rows, observed, predictions):
        if self.groups is None:
            raise ValueError("Pass the fixed split's parent ID universe to ScoreBook")
        rows = rows.reset_index(drop=True)
        metric = Contrast(rows, MIN_GROUPS)
        if len(metric.active) != len(rows):
            raise ValueError("Scoring pool below the registered support threshold")
        weights = np.zeros((len(STRATA), len(rows)))
        weights[0] = metric.primary_weight
        for k, stratum in enumerate(STRATA[1:], 1):
            idx = rows[LINE].isin(STRATUM_LINES[stratum])
            if idx.any():
                weights[k, idx] = group_weights(rows.loc[idx])
        gi = self.group_index.get_indexer(rows[GROUP])
        if np.any(gi < 0):
            raise ValueError("A group outside the frozen split entered scoring")
        for k, stratum in enumerate(STRATA):
            self.mass[stratum] += weights[k].sum()
            self.mass_boot[stratum] += point_numerator(weights[k], gi, self.draw)
            self.row_counts[stratum] += int((weights[k] > 0).sum())
            self.group_counts[stratum].update(rows.loc[weights[k] > 0, GROUP])
        obs_contrast = center(metric, observed)
        for arm, prediction in predictions.items():
            if prediction.shape != observed.shape:
                raise ValueError(f"Shape mismatch in {arm}")
            adjusted = center(metric, prediction)
            denom = np.sqrt(row_products(adjusted, adjusted) * row_products(obs_contrast, obs_contrast))
            cosine = np.divide(row_products(adjusted, obs_contrast), denom, out=np.zeros(len(rows)),
                               where=denom > 1e-14)
            diff = adjusted - obs_contrast
            stats = np.stack((np.sqrt(row_products(prediction - observed, prediction - observed)),
                              np.sqrt(row_products(diff, diff)), cosine, search(metric, adjusted, obs_contrast)))
            values = weights @ stats.T
            batch = np.zeros((len(STRATA), DRAW_COUNT))
            kernel = centered_error_grams(metric, prediction, observed)
            add_centered_bootstrap(rows, metric, kernel, self.draw, gi, weights, STRATA, batch)
            for k, stratum in enumerate(STRATA):
                self.points[stratum, arm] += values[k]
                self.boot[stratum, arm] += batch[k]

    def summary(self):
        points = []
        boots = {}
        for (stratum, arm), val in self.points.items():
            if self.mass[stratum] <= 0:
                continue
            values = val / self.mass[stratum]
            samples = self.boot[stratum, arm] / self.mass_boot[stratum]
            boots[stratum, arm] = samples
            ci = np.quantile(samples, [0.005, 0.995])
            points.append(dict(dose_uM=self.dose, minimum_cells=self.minimum_cells, stratum=stratum, arm=arm,
                 parent_groups=len(self.group_counts[stratum]), rows_roles=self.row_counts[stratum],
                 absolute_rmse=values[0], contrast_rmse=values[1], cosine=values[2], top1=values[3],
                 contrast_ci99_low=ci[0], contrast_ci99_high=ci[1]))
        frame = pd.DataFrame(points)
        pair = comparisons(frame, boots, ["B0", "B_pos-all"])
        return frame, pair, boots


def comparisons(points, boots, references):
    paired = []
    for row in points.itertuples(index=False):
        for ref in references:
            if row.arm == ref or (row.stratum, ref) not in boots:
                continue
            base = points.loc[points.stratum.eq(row.stratum) & points.arm.eq(ref), "contrast_rmse"].iloc[0]
            difference = boots[row.stratum, row.arm] - boots[row.stratum, ref]
            ci = np.quantile(difference, [.005, .995])
            paired.append(dict(dose_uM=row.dose_uM, minimum_cells=row.minimum_cells, stratum=row.stratum,
                               arm=row.arm, reference=ref, contrast_diff=row.contrast_rmse - base,
                               ci99_low=ci[0], ci99_high=ci[1], gain_pct=100 * (base - row.contrast_rmse) / base,
                               upper_below_zero=bool(ci[1] < 0)))
    return pd.DataFrame(paired)


def repeat(inputs, obs, pred):
    """Only score-stage plate 6 treatment makes this repeated reference possible."""
    cols = [GROUP, LINE, "barcode_position"]
    lookup = inputs.reset_index(names="row_index")
    source = lookup.loc[lookup.plate_id.eq("plate6")].set_index(cols, verify_integrity=True)
    target = lookup.loc[lookup.plate_id.eq("plate14")].set_index(cols, verify_integrity=True)
    common = target.join(source[["row_index"]], how="inner", rsuffix="_plate6").reset_index()
    if common.empty:
        return None
    target_rows = eligibility(inputs.iloc[common.row_index], 0)
    if target_rows.empty:
        return None
    select = target_rows.merge(common[[*cols, "row_index", "row_index_plate6"]], on=cols, validate="one_to_one")
    a, b = select.row_index.to_numpy(), select.row_index_plate6.to_numpy()
    rows = inputs.iloc[a].reset_index(drop=True)
    source_rows = inputs.iloc[b].reset_index(drop=True)
    repeat_residual = center(Contrast(source_rows, MIN_GROUPS), obs[b] - pred["B_pos-all"][b])
    result = {name: pred[name][a] for name in ("B0", "B_pos-all", "proposed-v0")}
    for g in (0.25, 1.0):
        result[f"plate6-residual-x{g:g}"] = result["B_pos-all"] + g * repeat_residual
    return rows, obs[a], result


def reach(points, boots):
    rows = []
    for stratum, group in points.groupby("stratum"):
        p = group.set_index("arm").contrast_rmse
        base = p["B_pos-all"]
        model = p["proposed-v0"]
        for reference in ("plate6-residual-x0.25", "plate6-residual-x1"):
            denom = base - p[reference]
            delta = boots[stratum, "B_pos-all"] - boots[stratum, reference]
            numerator = boots[stratum, "B_pos-all"] - boots[stratum, "proposed-v0"]
            den_ci = np.quantile(delta, [.005, .995])
            values = dict(stratum=stratum, reference=reference, repeat_gain_rmse=denom,
                          repeat_gain_ci99_low=den_ci[0], repeat_gain_ci99_high=den_ci[1],
                          model_gain_rmse=base - model, reach_pct=100 * (base-model) / denom if denom > 0 else None,
                          unstable=bool(den_ci[0] <= 0 <= den_ci[1]))
            if den_ci[0] > 0:
                lo, hi = np.quantile(100 * numerator / delta, [.005, .995])
                values.update(reach_ci99_low=lo, reach_ci99_high=hi)
            rows.append(values)
    return pd.DataFrame(rows)


def freeze(args):
    validate_protocol(args.protocol)
    manifest = {}
    expected = REQUIRED | ({"5uM-kernel-retuned"} if args.dose == 5 else set())
    for n in MIN_CELLS:
        for fold in range(3):
            for role in "AB":
                name = slice_name(args.dose, n, fold, role)
                path = args.predictions / name
                rec, _, _, _ = load_task(path, args.stage, args.protocol, args.dose, n, fold, role)
                if set(rec["models"]) != expected:
                    raise ValueError(f"Wrong forecast universe for {name}")
                manifest[name] = sha256(path / "receipt.json")
    write_json(args.freeze, {
        "protocol_sha256": PROTOCOL_SHA256,
        "stage_sha256": sha256(args.stage / "receipt.json"),
        "predict_code_sha256": sha256(Path(__file__).with_name("t1_predict.py")),
        "score_code_sha256": sha256(__file__),
        "io_code_sha256": sha256(Path(__file__).with_name("t1_io.py")),
        "dose": args.dose, "tasks": manifest, "models": sorted(expected),
        "predictions": str(args.predictions.resolve()),
    })


def score(args):
    frozen = read_json(args.freeze)
    values = (PROTOCOL_SHA256, sha256(args.stage / "receipt.json"), sha256(__file__),
              sha256(Path(__file__).with_name("t1_predict.py")), sha256(Path(__file__).with_name("t1_io.py")),
              args.dose, str(args.predictions.resolve()))
    keys = ("protocol_sha256", "stage_sha256", "score_code_sha256", "predict_code_sha256",
            "io_code_sha256", "dose", "predictions")
    if tuple(frozen[key] for key in keys) != values:
        raise ValueError("Frozen manifest does not match inputs or execution")
    args.out.mkdir(parents=True, exist_ok=False)
    boot_groups = registry(args.stage)
    support = []
    for n in MIN_CELLS:
        book = ScoreBook(args.dose, n, boot_groups)
        repeats = ScoreBook(args.dose, n, boot_groups) if args.dose == 5 else None
        for fold in range(3):
            for role in "AB":
                name = slice_name(args.dose, n, fold, role)
                path = args.predictions / name
                if sha256(path / "receipt.json") != frozen["tasks"][name]:
                    raise ValueError("Prediction receipt changed after freeze")
                rec, rows, panel, predictions = load_task(path, args.stage, args.protocol, args.dose, n, fold, role)
                obs = score_truth(args.stage, args.dose, n, fold, role, rows, panel)
                book.add(rows, obs, predictions)
                selection = rec["selection"]
                support.append(dict(dose_uM=args.dose, minimum_cells=n, fold=fold, role=role,
                                    rows=len(rows), groups=rec["held_parents"],
                                    same_dose_support=selection["B_pos"]["supported_rows"],
                                    all_doses_support=selection["B_pos-all"]["supported_rows"],
                                    same_dose_gain=selection["B_pos"]["gain"],
                                    all_doses_gain=selection["B_pos-all"]["gain"],
                                    combined_position_gain=selection["proposed-v0"]["position_gain"],
                                    combined_residual_gain=selection["proposed-v0"]["residual_gain"]))
                if repeats is not None:
                    pair = repeat(rows, obs, predictions)
                    if pair is not None:
                        repeats.add(*pair)
                print(f"scored {name}", flush=True)
        points, paired, _ = book.summary()
        points.to_csv(args.out / f"points_n{n}.tsv", sep="\t", index=False)
        paired.to_csv(args.out / f"paired_n{n}.tsv", sep="\t", index=False)
        if repeats is not None:
            rep_points, rep_paired, boots = repeats.summary()
            rep_points.to_csv(args.out / f"repeat_points_n{n}.tsv", sep="\t", index=False)
            rep_paired.to_csv(args.out / f"repeat_paired_n{n}.tsv", sep="\t", index=False)
            reach(rep_points, boots).to_csv(args.out / f"repeat_reach_n{n}.tsv", sep="\t", index=False)
    pd.DataFrame(support).to_csv(args.out / "position_and_selection.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(freeze_sha256=sha256(args.freeze),
                     files={p.name: sha256(p) for p in sorted(args.out.iterdir())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("stage", "protocol", "predictions", "freeze"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--dose", required=True, choices=DOSES, type=float)
    parser.add_argument("--out", type=Path)
    parser.add_argument("action", choices=("freeze", "score"))
    args = parser.parse_args()
    (freeze if args.action == "freeze" else score)(args)


if __name__ == "__main__":
    main()
