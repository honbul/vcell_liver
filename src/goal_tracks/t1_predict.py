"""Train fixed R13 families and a location-plus-chemical residual on each Tahoe slice."""

import argparse
from collections import defaultdict
from copy import copy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, POOL, weights_for
from src.r11_chemical_probe.metrics import Contrast
from src.r12_multiaxis.data import Fit
from src.r13_optimization import control, dose_transfer, kernels, mechanism, neural, transfer
from src.r13_optimization.objectives import GainScores, center, row_products

from .t1_data import DOSES, MIN_CELLS, MIN_GROUPS, DoseSource, eligibility, slice_name, validate_protocol
from .t1_io import write_task


GAIN = np.asarray((0, .125, .25, .5, .75, 1, 1.5, 2, 3, 4))
POS_GAIN = GAIN[:8]
FAMILIES = {"kernels": kernels, "mechanism": mechanism, "transfer": transfer, "neural": neural,
            "dose_transfer": dose_transfer}
REQUIRED = {"B0", "B_abs", "B_pos", "B_pos-all", "proposed-v0", "proposed-position", "proposed-residual",
            *FAMILIES}


class PositionBank:
    """Pool-centered same-position donors with query plate and parent removed."""

    def __init__(self, rows, observed):
        self.rows = rows
        self.observed = observed
        self.metric = Contrast(rows, MIN_GROUPS)
        self.centered = center(self.metric, observed).astype(np.float32)
        self.by_pos = defaultdict(list)
        locations, lines = rows.barcode_position.to_numpy(), rows[LINE].to_numpy()
        for i in self.metric.active:
            self.by_pos[locations[i], lines[i]].append(i)
        self.plates = rows.plate_id.to_numpy()
        self.groups = rows[GROUP].to_numpy()
        self.pool_of = {}
        self.withdraw = {}
        self.pool_sizes = {}
        for pool_id, (ix, center_weight) in enumerate(zip(self.metric.pools, self.metric.centers)):
            self.pool_of.update((i, pool_id) for i in ix)
            member = self.groups[ix]
            self.pool_sizes[pool_id] = len(np.unique(member))
            for group in np.unique(member):
                use = member == group
                weight = center_weight[use]
                # Subtracting a parent's weighted contribution changes the donor center.
                self.withdraw[pool_id, group] = (
                    (weight @ self.centered[ix[use]] / (center_weight.sum() - weight.sum())).astype(np.float32)
                )

    def predict(self, rows, exclude_parent=False):
        if not exclude_parent and set(rows[GROUP]) & set(self.rows[GROUP]):
            raise ValueError("Query parents in position donor fit")
        result = np.zeros((len(rows), self.observed.shape[1]), dtype=np.float32)
        counts = np.zeros(len(rows), dtype=np.int32)
        for i, (position, line, plate, group) in enumerate(zip(
                rows.barcode_position, rows[LINE], rows.plate_id, rows[GROUP])):
            donors = [d for d in self.by_pos[position, line] if self.plates[d] != plate and self.groups[d] != group]
            if exclude_parent:
                donors = [d for d in donors if self.pool_sizes[self.pool_of[d]] > MIN_GROUPS
                          or (self.pool_of[d], group) not in self.withdraw]
            if donors:
                d = np.asarray(donors)
                values = self.centered[d]
                if exclude_parent:
                    for j, donor in enumerate(d):
                        correction = self.withdraw.get((self.pool_of[donor], group))
                        if correction is not None:
                            values[j] += correction
                labels, mass = np.unique(self.groups[d], return_counts=True)
                weights = 1.0 / mass[np.searchsorted(labels, self.groups[d])]
                result[i] = np.average(values, axis=0, weights=weights)
                counts[i] = len(labels)
        return result, counts


def history(source, folder):
    task = f"task{2 * source.outer + int(source.role == 'B')}"
    freeze = read_json(folder / "freeze.json")
    path = folder / "frozen" / task / "receipt.json"
    if sha256(path) != freeze["task_receipts"][task]:
        raise ValueError("R13 selected settings receipt changed")
    rec = read_json(path)
    if rec["partition_sha256"] != source.split["manifest_sha256"]:
        raise ValueError("R13 selected settings use other compound groups")
    return {f: rec["winners"][f]["specification"] for f in FAMILIES}, sha256(path)


class DoseTransfer(dose_transfer.DoseTransferFit):
    def dose_summary(self, dose):
        # When transplanting a 0.5 recipe to an endpoint dose, the other two
        # registered doses replace its low and high auxiliary views in order.
        alternatives = [d for d in DOSES if d != self.fit.source.dose]
        alternate = dict(zip((0.05, 5.0), alternatives))[dose]
        return super().dose_summary(alternate)


def family_prediction(fit, family, spec, confirmation=False):
    if family == "dose_transfer":
        return DoseTransfer(fit).predict(spec)
    if family == "neural" and confirmation:
        return np.mean([neural.predict(fit, dict(spec, seed=k)) for k in (42, 43, 44)], axis=0)
    return FAMILIES[family].predict(fit, spec)


def positions(fit, groups):
    source = fit.source
    candidates = source.all_fit_rows
    target = candidates.loc[candidates[GROUP].isin(groups)]
    target = eligibility(target, source.minimum_cells)
    full = source.observed(target, fit.panel)
    same = target.dose_uM.eq(source.dose).to_numpy()
    all_bank = PositionBank(target, full)
    same_bank = PositionBank(target.loc[same].reset_index(drop=True), full[same])
    p, p_count = same_bank.predict(fit.query.rows)
    a, a_count = all_bank.predict(fit.query.rows)
    p_train, _ = all_bank.predict(fit.train.rows, exclude_parent=True)
    return p, a, p_train, p_count, a_count


def residual_prediction(fit, train_position, kernel_spec):
    modified = copy(fit)
    modified.residual = fit.residual - train_position
    if hasattr(modified, "_r13_kernel_fit"):
        del modified._r13_kernel_fit
    return kernels.predict(modified, kernel_spec) - modified.offset


def mix_specs():
    return [(weight, alpha) for weight, alpha in product((0.0, 0.3, 1.0), (.1, 1, 10))]


def retune_predictions(fit, mixes):
    engine = kernels.KernelFit(fit)
    ecfp, maccs = engine.chemical("ecfp"), engine.chemical("maccs")
    for weight, alpha in mixes:
        label = f"ecfp-maccs-{weight:g}"
        engine.kernels[label] = tuple((1 - weight) * a + weight * b for a, b in zip(ecfp, maccs))
        spec = dict(chemistry=label, alpha=alpha, context="none", target="contrast", method="ridge")
        yield engine.ridge(spec) - fit.offset


def joint_losses(rows, truth, p, residual):
    metric = Contrast(rows, MIN_GROUPS)
    y, x, r = (center(metric, value) for value in (truth, p, residual))
    yy, xx, rr = row_products(y, y), row_products(x, x), row_products(r, r)
    yx, yr, xr = row_products(y, x), row_products(y, r), row_products(x, r)
    # Use residual as the first axis for the specified tie-breaking order.
    a, b = GAIN[:, None, None], POS_GAIN[None, :, None]
    sq = yy + b*b*xx + a*a*rr - 2*b*yx - 2*a*yr + 2*a*b*xr
    return np.average(np.sqrt(sq.clip(0)), axis=-1, weights=metric.primary_weight)


def choose(loss):
    return np.unravel_index(np.argmax(loss <= loss.min() + 1e-8), loss.shape)


def predict(args):
    validate_protocol(args.protocol)
    cfg = config(args.config)
    source = DoseSource(cfg, args.assets, args.stage, args.dose, args.minimum_cells, args.fold, args.role)
    specs, recovered = history(source, args.history)
    totals = defaultdict(list)
    mix_grid = mix_specs() if source.dose == 5 else []
    for inner in (0, 1):
        train, validation = source.inner(inner)
        fit = Fit(source, train, validation, panel=source.inner_panel(inner))
        y = source.observed(fit.query.rows, fit.panel) - fit.query.control
        true = GainScores(fit.query.rows, y, fit.offset, MIN_GROUPS)
        zero = GainScores(fit.query.rows, y, np.zeros_like(y), MIN_GROUPS)
        mass = true.metric.primary_weight.sum()
        abs_mass = true.metric.all_weight.sum()
        pools = [control.predict(fit, s) for s in control.candidates()]
        totals["B_abs"].append(abs_mass * np.array([zero.points(p, np.array([1]))[1][0] for p in pools]))
        p, all_p, train_p, *_ = positions(fit, set(train[GROUP]))
        totals["B_pos"].append(mass * zero.points(p, POS_GAIN)[0])
        totals["B_pos-all"].append(mass * zero.points(all_p, POS_GAIN)[0])
        for family, spec in specs.items():
            adj = family_prediction(fit, family, spec) - fit.offset
            totals[family].append(mass * true.points(adj, GAIN)[0])
        resid = residual_prediction(fit, train_p, specs["kernels"])
        totals["proposed-v0"].append(mass * joint_losses(fit.query.rows, y, all_p, resid))
        if mix_grid:
            totals["5uM-kernel-retuned"].append(mass * np.stack([
                true.points(p, GAIN)[0] for p in retune_predictions(fit, mix_grid)
            ], axis=1))
        print(f"inner {inner} done", flush=True)
    errors = {name: np.sum(arrays, axis=0) for name, arrays in totals.items()}
    picked = {name: choose(loss) for name, loss in errors.items()}
    fit = Fit(source, source.fit_rows, source.query_rows, panel=source.panel)
    p, all_p, train_p, p_count, all_count = positions(fit, set(source.fit_rows[GROUP]))
    anchor_spec = control.candidates()[picked["B_abs"][0]]
    anchor = control.predict(fit, anchor_spec)
    resid = residual_prediction(fit, train_p, specs["kernels"])
    r_gain, p_gain = GAIN[picked["proposed-v0"][0]], POS_GAIN[picked["proposed-v0"][1]]
    predictions = {
        "B0": np.zeros_like(fit.offset), "B_abs": anchor,
        "B_pos": anchor + POS_GAIN[picked["B_pos"][0]] * p,
        "B_pos-all": anchor + POS_GAIN[picked["B_pos-all"][0]] * all_p,
        "proposed-v0": anchor + p_gain * all_p + r_gain * resid,
        "proposed-position": anchor + p_gain * all_p,
        "proposed-residual": anchor + r_gain * resid,
    }
    selections = {
        "historical_settings_sha256": recovered,
        "B_abs": anchor_spec,
        "B_pos": {"gain": float(POS_GAIN[picked["B_pos"][0]]), "supported_rows": int((p_count > 0).sum())},
        "B_pos-all": {"gain": float(POS_GAIN[picked["B_pos-all"][0]]), "supported_rows": int((all_count > 0).sum())},
        "proposed-v0": {"position_gain": float(p_gain), "residual_gain": float(r_gain),
                        "kernel": specs["kernels"], "fit_position_gain": 1.0},
    }
    for family, spec in specs.items():
        p = family_prediction(fit, family, spec, confirmation=True)
        gain = GAIN[picked[family][0]]
        predictions[family] = fit.offset + gain * (p - fit.offset)
        selections[family] = {"spec": spec, "gain": float(gain)}
    if mix_grid:
        gain, grid = picked["5uM-kernel-retuned"]
        weight, alpha = mix_grid[grid]
        p = next(retune_predictions(fit, [(weight, alpha)]))
        predictions["5uM-kernel-retuned"] = fit.offset + GAIN[gain] * p
        selections["5uM-kernel-retuned"] = {"gain": float(GAIN[gain]), "maccs_weight": weight, "alpha": alpha}
    for name, entry in selections.items():
        if name in errors:
            entry["inner_weighted_error"] = errors[name].tolist()
    write_task(args.out / slice_name(source.dose, source.minimum_cells, source.outer, source.role),
               source, fit, predictions, selections)
    print(f"frozen input written for {args.dose}/{args.minimum_cells}/{args.fold}/{args.role}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "assets", "stage", "history", "protocol", "out"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--dose", required=True, choices=DOSES, type=float)
    parser.add_argument("--minimum-cells", required=True, choices=MIN_CELLS, type=int)
    parser.add_argument("--fold", required=True, choices=range(3), type=int)
    parser.add_argument("--role", required=True, choices=("A", "B"))
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
