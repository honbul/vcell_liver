"""Trim plate-position donors on fit responses and freeze the shared context."""

import argparse
from functools import cached_property
from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import DoseSource, MIN_GROUPS, eligibility, slice_name
from src.goal_tracks.t1_predict import PositionBank
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE
from src.r11_chemical_probe.metrics import Contrast
from src.r12_multiaxis.data import Fit
from src.r13_optimization.objectives import center


TRIM_THRESHOLDS = (None, 2.0, 1.5, 1.25)
POS_GAINS = np.array([0, .125, .25, .5, .75, 1, 1.5, 2])
MODEL_GAINS = POS_GAINS.copy()


def energy_curves(rows, observed, position, direction=None):
    """SSE/SST over row/gene elements after original compound-balanced centering."""
    metric = Contrast(rows, MIN_GROUPS)
    target = center(metric, observed).astype(np.float64)[metric.active].ravel()
    pos = center(metric, position).astype(np.float64)[metric.active].ravel()
    sst = target @ target
    if sst <= 0:
        raise ValueError("No energy in the valid inner contrasts")
    p2 = pos @ pos
    yp = target @ pos
    g = POS_GAINS
    if direction is None:
        return 1 + (g*g*p2 - 2*g*yp) / sst
    vector = center(metric, direction).astype(np.float64)[metric.active].ravel()
    vg = MODEL_GAINS[None, :]
    pg = g[:, None]
    return 1 + ((pg*pg*p2 - 2*pg*yp) + (vg*vg*(vector @ vector) - 2*vg*(target @ vector))
                + 2*pg*vg*(pos @ vector)) / sst


class TrimBank(PositionBank):
    """Cache line-specific medians after withdrawing the receiver's fit parent."""

    @cached_property
    def active_by_line(self):
        active = self.metric.active
        lines = self.rows[LINE].to_numpy()[active]
        return {line: active[lines == line] for line in np.unique(lines)}

    @cached_property
    def norms(self):
        return np.linalg.norm(self.centered, axis=1)

    def donors_without(self, permitted, group):
        keep = self.groups[permitted] != group
        return np.array([d for d in permitted[keep]
                         if self.pool_sizes[self.pool_of[d]] > MIN_GROUPS
                         or (self.pool_of[d], group) not in self.withdraw], dtype=int)

    def corrected(self, donors, group):
        values = self.centered[donors]
        if group is not None:
            for i, donor in enumerate(donors):
                correction = self.withdraw.get((self.pool_of[donor], group))
                if correction is not None:
                    values[i] += correction
        return values

    def medians(self, training=False):
        result = {}
        for line, ids in self.active_by_line.items():
            result[line, None] = float(np.median(self.norms[ids]))
            if training:
                for group in np.unique(self.groups[ids]):
                    keep = self.donors_without(ids, group)
                    if not len(keep):
                        continue
                    norm = self.norms[keep].copy()
                    touched = np.array([(self.pool_of[d], group) in self.withdraw for d in keep])
                    if touched.any():
                        norm[touched] = np.linalg.norm(self.corrected(keep[touched], group), axis=1)
                    result[line, group] = float(np.median(norm))
        return result

    def trimmed(self, rows, thresholds, training=False):
        if not training and set(rows[GROUP]) & set(self.groups):
            raise ValueError("Query parents in trimmed position fit")
        pred = np.zeros((len(thresholds), len(rows), self.centered.shape[1]), dtype=np.float32)
        support = np.zeros((len(thresholds), len(rows)), dtype=np.int16)
        medians = self.medians(training) if any(t is not None for t in thresholds) else {}
        for i, (position, line, plate, group) in enumerate(zip(
                rows.barcode_position, rows[LINE], rows.plate_id, rows[GROUP])):
            candidates = np.asarray(self.by_pos[position, line], dtype=int)
            candidates = candidates[self.plates[candidates] != plate]
            if training:
                candidates = self.donors_without(candidates, group)
            else:
                group = None
            if not len(candidates):
                continue
            values = self.corrected(candidates, group)
            norm = np.linalg.norm(values, axis=1)
            med = medians.get((line, group), medians.get((line, None), 0.0))
            for j, threshold in enumerate(thresholds):
                keep = np.ones(len(norm), dtype=bool) if threshold is None else norm <= threshold * med
                if not keep.any():
                    continue
                labels = self.groups[candidates[keep]]
                parents, ids, counts = np.unique(labels, return_inverse=True, return_counts=True)
                pred[j, i] = np.average(values[keep], axis=0, weights=1 / counts[ids])
                support[j, i] = len(parents)
        return pred, support


def build_bank(source, fit):
    valid = source.all_fit_rows[GROUP].isin(set(fit.train.rows[GROUP]))
    donors = eligibility(source.all_fit_rows.loc[valid], source.minimum_cells)
    if set(donors[GROUP]) & set(fit.query.rows[GROUP]):
        raise ValueError("Query parent entered shared location bank")
    return TrimBank(donors, source.observed(donors, fit.panel))


def finish_context(path, source, protocol):
    contexts, losses = [], []
    for inner in (0, 1):
        split = source.inner(inner)
        fit = Fit(source, *split, panel=source.inner_panel(inner))
        bank = build_bank(source, fit)
        position, support = bank.trimmed(fit.query.rows, TRIM_THRESHOLDS)
        target = source.observed(fit.query.rows, fit.panel) - fit.query.control
        losses.append(np.stack([energy_curves(fit.query.rows, target, v) for v in position]))
        contexts.append((fit, bank, position, support))
    loss = np.mean(losses, axis=0)
    # Prefer a lower scale, then the least trimming when objectives tie numerically.
    tx, gx = sorted(np.ndindex(loss.shape), key=lambda ij: (round(float(loss[ij]), 10), ij[1], ij[0]))[0]
    threshold, gain = TRIM_THRESHOLDS[tx], POS_GAINS[gx]
    final = Fit(source, source.fit_rows, source.query_rows, panel=source.panel)
    final_bank = build_bank(source, final)
    vp, sp = final_bank.trimmed(final.query.rows, [threshold])
    contexts.append((final, final_bank, vp, sp))
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    counts = {}
    try:
        for (fit, bank, cpr, support), ix, name in zip(
                contexts, (tx, tx, 0), ("inner0", "inner1", "outer")):
            tp, ts = bank.trimmed(fit.train.rows, [threshold], training=True)
            arrays = dict(
                train_position=tp[0], query_position=cpr[ix],
                panel=fit.panel, train_keys=fit.train.rows[KEY].to_numpy(dtype=str),
                query_keys=fit.query.rows[KEY].to_numpy(dtype=str), train_support=ts[0],
                query_support=support[ix],
            )
            np.savez(tmp / f"{name}.npz", **arrays)
            counts[name] = int((support[ix] > 0).sum())
        write_json(tmp / "receipt.json", dict(
            protocol_sha256=sha256(protocol), stage_sha256=sha256(source.staged / "receipt.json"),
            scope=(source.dose, source.minimum_cells, source.outer, source.role),
            threshold=threshold, gain=float(gain), threshold_grid=list(TRIM_THRESHOLDS),
            gain_grid=POS_GAINS.tolist(), inner_energy_loss=loss.tolist(),
            supported_query_rows=counts, files={p.name: sha256(p) for p in sorted(tmp.iterdir())},
        ))
        if path.exists():
            raise FileExistsError(path)
        os.rename(tmp, path)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def load_context(root, source, fit, inner, scaled=True, protocol=None):
    path = Path(root) / slice_name(source.dose, source.minimum_cells, source.outer, source.role)
    stamp = read_json(path / "receipt.json")
    if (stamp["scope"] != [source.dose, source.minimum_cells, source.outer, source.role]
            or stamp["stage_sha256"] != sha256(source.staged / "receipt.json")):
        raise ValueError("Position context uses a different fit partition")
    if protocol is not None and stamp["protocol_sha256"] != sha256(protocol):
        raise ValueError("Position context uses a different forecast protocol")
    name = "outer.npz" if inner == -1 else f"inner{inner}.npz"
    if sha256(path / name) != stamp["files"][name]:
        raise ValueError("Shared location file changed")
    with np.load(path / name, allow_pickle=False) as data:
        if not (np.array_equal(data["panel"], fit.panel)
                and np.array_equal(data["train_keys"], fit.train.rows[KEY].to_numpy(dtype=str))
                and np.array_equal(data["query_keys"], fit.query.rows[KEY].to_numpy(dtype=str))):
            raise ValueError("Location arrays do not align to the registered fit")
        gain = stamp["gain"] if scaled else 1
        return data["train_position"] * gain, data["query_position"] * gain


def source_args(parser):
    for arg in ("config", "assets", "stage", "protocol"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--dose", type=float, choices=(.05, .5, 5), required=True)
    parser.add_argument("--minimum-cells", type=int, choices=(20, 200), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--role", choices=("A", "B"), required=True)


def load_source(args):
    return DoseSource(config(args.config), args.assets, args.stage, args.dose,
                      args.minimum_cells, args.fold, args.role)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_args(parser)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = load_source(args)
    case = slice_name(source.dose, source.minimum_cells, source.outer, source.role)
    finish_context(args.out / case, source, args.protocol)


if __name__ == "__main__":
    main()
