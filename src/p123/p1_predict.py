"""Predict Tahoe using a trim-position anchor and compound sibling residuals."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.goal_tracks.t1_data import MIN_GROUPS, slice_name
from src.goal_tracks.t1_io import load_task
from src.r11_chemical_probe.common import read_json, sha256
from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE
from src.r11_chemical_probe.metrics import Contrast
from src.r12_multiaxis.data import Fit
from src.r13_optimization.mechanism import annotation_similarity, annotations, bit_tanimoto
from src.r13_optimization.objectives import center

from .p123_io import write_prediction
from .p1_position import MODEL_GAINS, load_context, load_source, source_args


GRIDS = {
    "p1-tanimoto": (0.1, 1.0, 10.0),
    "p1-closest": (0.3, 0.5, 0.7),
    "p1-moa": ("targets", "fine", "mixed"),
}
REQUIRED = {"p1-trim", "p1-ensemble", "proposed-v1", *GRIDS}


class Siblings:
    """One identity per fitted parent and separate same-line residual profiles."""

    def __init__(self, fit, train_position):
        tr, te = fit.train.rows, fit.query.rows
        if set(tr[GROUP]) & set(te[GROUP]):
            raise ValueError("Held-out parent entered sibling labels")
        metric = Contrast(tr, MIN_GROUPS)
        target = center(metric, fit.train.delta - train_position).astype(np.float32)
        active = metric.active
        self.rows = te
        self.lines = sorted(tr.loc[active, LINE].unique())
        self.line_idx = pd.Index(self.lines).get_indexer(te[LINE])
        self.groups, _ = np.unique(tr.loc[active, GROUP], return_index=True)
        preferred = (tr.iloc[active].sort_values([GROUP, ENTITY], kind="stable")
                     .drop_duplicates(GROUP))
        names = preferred[ENTITY].to_numpy()
        self.entities, qfirst, self.qidx = np.unique(te[ENTITY], return_index=True, return_inverse=True)
        fp_ref = fit.train.chemical["E"][preferred.index].astype(np.float64)
        fp_query = fit.query.chemical["E"][qfirst].astype(np.float64)
        self.k_fit = bit_tanimoto(fp_ref, fp_ref)
        self.k_query = bit_tanimoto(fp_query, fp_ref)
        by_group = pd.Index(self.groups).get_indexer(tr.loc[active, GROUP])
        by_line = pd.Index(self.lines).get_indexer(tr.loc[active, LINE])
        cell_index = by_line * len(self.groups) + by_group
        weights = metric.primary_weight[active]
        mass = np.bincount(cell_index, weights=weights, minlength=len(self.groups)*len(self.lines))
        profiles = np.zeros((len(mass), target.shape[1]), dtype=np.float32)
        np.add.at(profiles, cell_index, target[active] * weights[:, None])
        self.support = (mass.reshape(len(self.lines), -1) > 0)
        self.profiles = (profiles / np.maximum(mass[:, None], 1e-20)).astype(np.float32).reshape(
            len(self.lines), len(self.groups), -1)
        meta = annotations(fit.source.cfg["tahoe_root"])
        ref, query = meta.reindex(names), meta.reindex(self.entities)
        self.jaccard = {
            field: annotation_similarity(ref, query, src)
            for field, src in (("targets", "targets"), ("fine", "moa-fine"))
        }
        self.jaccard["mixed"] = .5*(self.jaccard["targets"]+self.jaccard["fine"])

    def ridge(self, alpha):
        result = np.zeros((len(self.rows), self.profiles.shape[-1]), dtype=np.float32)
        for line in range(len(self.lines)):
            idx = np.flatnonzero(self.line_idx == line)
            if not len(idx):
                continue
            q, inverse = np.unique(self.qidx[idx], return_inverse=True)
            g = np.flatnonzero(self.support[line])
            gram = self.k_fit[np.ix_(g, g)].copy()
            cross = self.k_query[np.ix_(q, g)].copy()
            mean = gram.mean(axis=0)
            grand = mean.mean()
            gram -= mean[:, None] + mean - grand
            cross -= cross.mean(axis=1, keepdims=True) + mean - grand
            gram.flat[::len(gram)+1] += alpha
            outcome = self.profiles[line, g]
            outcome = outcome - outcome.mean(axis=0, keepdims=True)
            weights = cho_solve(cho_factor(gram, check_finite=False), outcome, check_finite=False)
            result[idx] = (cross @ weights)[inverse]
        return result

    def closest(self, similarities, minimum):
        result = np.zeros((len(self.rows), self.profiles.shape[-1]), dtype=np.float32)
        for line in range(len(self.lines)):
            idx = np.flatnonzero(self.line_idx == line)
            if not len(idx):
                continue
            q, inverse = np.unique(self.qidx[idx], return_inverse=True)
            g = np.flatnonzero(self.support[line])
            sim = similarities[np.ix_(q, g)]
            peak = sim.max(axis=1, keepdims=True)
            weights = np.isclose(sim, peak, rtol=0, atol=1e-10) & (peak >= minimum) & (peak > 0)
            result[idx] = ((weights @ self.profiles[line, g]) / weights.sum(
                axis=1, keepdims=True).clip(min=1))[inverse]
        return result

    def prediction(self, name, value):
        if name == "p1-tanimoto":
            return self.ridge(value)
        if name == "p1-closest":
            return self.closest(self.k_query, value)
        if name == "p1-moa":
            return self.closest(self.jaccard[value], 0)
        raise KeyError(name)


class InnerObjective:
    """Evaluate a unique chemical residual direction at the shared position gain."""

    def __init__(self, rows, observed, fallback):
        self.metric = Contrast(rows, MIN_GROUPS)
        self.error = self.flat(observed - fallback)
        observed = self.flat(observed)
        self.sst = observed @ observed
        if self.sst <= 0:
            raise ValueError("Zero contrast energy in inner validation")
        self.sse = self.error @ self.error

    def flat(self, values):
        return center(self.metric, values).astype(np.float64)[self.metric.active].ravel()

    def curve(self, vector):
        raw = self.flat(vector)
        return (self.sse - 2*MODEL_GAINS*(raw @ self.error) + MODEL_GAINS**2*(raw @ raw)) / self.sst


def combine(directions):
    return np.mean(list(directions), axis=0, dtype=np.float32)


def learn(source, location, protocol):
    contexts = []
    for i in (0, 1):
        fit = Fit(source, *source.inner(i), panel=source.inner_panel(i))
        tr, te = load_context(location, source, fit, i, protocol=protocol)
        target = source.observed(fit.query.rows, fit.panel) - fit.query.control
        contexts.append((Siblings(fit, tr), InnerObjective(fit.query.rows, target, te)))
    chosen, losses = {}, {}
    for family, grid in GRIDS.items():
        surface = np.zeros((len(grid), len(MODEL_GAINS)))
        for ix, value in enumerate(grid):
            for model, objective in contexts:
                surface[ix] += objective.curve(model.prediction(family, value)) / len(contexts)
        i, g = min(np.ndindex(surface.shape), key=lambda ij: (
            round(float(surface[ij]), 10), ij[1], ij[0]))
        chosen[family] = (grid[i], float(MODEL_GAINS[g]))
        losses[family] = float(surface[i, g])
    combo = np.mean([
        objective.curve(combine(model.prediction(name, p) for name, (p, _) in chosen.items()))
        for model, objective in contexts
    ], axis=0)
    i = min(range(len(combo)), key=lambda j: (round(float(combo[j]), 10), j))
    losses["p1-ensemble"] = float(combo[i])
    candidate_order = (*GRIDS, "p1-ensemble")
    choices_gain = {**{name: g for name, (_, g) in chosen.items()}, "p1-ensemble": float(MODEL_GAINS[i])}
    winner = min(candidate_order, key=lambda name: (
        round(losses[name], 10), choices_gain[name], candidate_order.index(name)))
    return chosen, float(MODEL_GAINS[i]), winner, losses


def prior_t1(source, prior):
    dose_root = prior / f"t1/first/d{source.dose:g}"
    manifest_path = dose_root / "freeze.json"
    contract = prior / "t1-code/reports/goal_tracks/PROTOCOL.md"
    entry = slice_name(source.dose, source.minimum_cells, source.outer, source.role)
    task = dose_root / "predictions" / entry
    manifest = read_json(manifest_path)
    if manifest["tasks"][entry] != sha256(task / "receipt.json"):
        raise ValueError("Historical B_abs received a changed T1 prediction")
    _, _, _, arrays = load_task(task, source.staged, contract, source.dose, source.minimum_cells,
                                source.outer, source.role)
    return arrays["B_abs"], sha256(manifest_path)


def predict(args):
    source = load_source(args)
    chosen, ensemble_gain, winner, losses = learn(source, args.position, args.protocol)
    fit = Fit(source, source.fit_rows, source.query_rows, panel=source.panel)
    tr, te = load_context(args.position, source, fit, -1, protocol=args.protocol)
    model = Siblings(fit, tr)
    anchor, prior_hash = prior_t1(source, args.prior)
    offset = anchor + te
    arms = {"p1-trim": offset}
    outputs = {name: model.prediction(name, value) for name, (value, _) in chosen.items()}
    for name, (_, gain) in chosen.items():
        arms[name] = offset + gain * outputs[name]
    arms["p1-ensemble"] = offset + ensemble_gain * combine(outputs.values())
    arms["proposed-v1"] = arms[winner]
    name = slice_name(source.dose, source.minimum_cells, source.outer, source.role)
    stamp = args.position / name / "receipt.json"
    selections = dict(
        chosen=chosen, ensemble_gain=ensemble_gain, winner=winner, inner_energy_losses=losses,
        position_receipt_sha256=sha256(stamp), t1_freeze_sha256=prior_hash,
        code_sha256=sha256(Path(__file__)),
    )
    write_prediction(args.out / name, source, fit, arms, selections, args.protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_args(parser)
    for key in ("position", "prior", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
