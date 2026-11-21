"""Pool response evidence across cell lines before learning unseen-drug effects."""

from functools import cached_property
from itertools import product

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.utils.extmath import randomized_svd

from src.r11_chemical_probe.dataset import CONTROL, ENTITY, GROUP, KEY, LINE


def candidates():
    """Prespecified small-sample models and matched ablations."""
    specs = []

    def add(chemistry, alpha, gene_basis, line_basis="none", line_rank=0,
            line_gain=0.0, pooling="mean", plate_adjust=False):
        tag = f"tr_{chemistry}_a{alpha}_{gene_basis}_{line_basis}{line_rank}_g{line_gain}_{pooling}"
        if plate_adjust:
            tag += "_plate"
        specs.append(dict(name=tag, chemistry=chemistry, alpha=alpha, gene_basis=gene_basis,
                          line_basis=line_basis, line_rank=line_rank, line_gain=line_gain,
                          pooling=pooling, plate_adjust=plate_adjust))

    for chemistry, alpha, gene_basis in product(("ecfp", "cls", "mixed"), (0.3, 3, 30, 300),
                                                 ("full", "pca32", "cross32")):
        add(chemistry, alpha, gene_basis)
    for mode, rank, alpha, gain in product(("empirical", "control"), (4, 12), (3, 30), (0.25, 1.0)):
        add("ecfp", alpha, "pca32", mode, rank, gain)
    for alpha, rank_gene in product((3, 30, 300), ("pca32", "cross32")):
        add("ecfp", alpha, rank_gene, "identity", 0, 0.25)
        add("ecfp", alpha, rank_gene, plate_adjust=True)
    for alpha, pooling in product((3, 30, 300), ("robust", "precision")):
        add("ecfp", alpha, "pca32", pooling=pooling)
    return specs


def mean_units(values, inverse, unit_count, weights=None):
    """Aggregate repeated rows without assigning repetitions extra influence."""
    weights = np.ones(len(inverse)) if weights is None else np.asarray(weights)
    count = np.bincount(inverse, weights=weights, minlength=unit_count)
    result = np.zeros((unit_count, values.shape[1]), dtype=np.float32)
    np.add.at(result, inverse, values * weights[:, None].astype(np.float32))
    np.divide(result, count[:, None], where=count[:, None] > 0, out=result)
    return result, count


def unit_norm(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def weighted_consensus(values, weight):
    return np.einsum("glp,gl->gp", values, weight, optimize=True) / np.maximum(
        weight.sum(axis=1, keepdims=True), 1e-8)


def reproducible_gene_basis(left, right, rank):
    """Keep directions with positive cross-view covariance across fit compounds."""
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    both = np.concatenate((left, right), axis=0)
    subrank = min(2 * rank, *both.shape)
    _, _, space = randomized_svd(both, n_components=subrank, random_state=42)
    a, b = left @ space.T, right @ space.T
    score = a.T @ b
    values, directions = eigh((score + score.T) * 0.5, check_finite=False)
    chosen = np.flatnonzero(values > max(values[-1] * 1e-6, 1e-9))[-rank:][::-1]
    return (space.T @ directions[:, chosen]).astype(np.float32)


class TransferFit:
    def __init__(self, fit):
        self.fit = fit
        train, query = fit.train.rows, fit.query.rows
        if set(train[GROUP]) & set(query[GROUP]) or set(train[KEY[0]]) & set(query[KEY[0]]):
            raise ValueError("Transfer fitting requires disjoint fit and query compounds and wells")

        self.groups, self.group_inverse = np.unique(train[GROUP], return_inverse=True)
        self.lines, line_inverse = np.unique(train[LINE], return_inverse=True)
        self.query_entities, self.query_entity_first, self.query_entity_inverse = np.unique(
            query[ENTITY], return_index=True, return_inverse=True)
        self.query_line_index = pd.Index(self.lines).get_indexer(query[LINE])
        n_groups, n_lines = len(self.groups), len(self.lines)
        aggregate, support = mean_units(fit.residual, self.group_inverse * n_lines + line_inverse,
                                        n_groups * n_lines, fit.weight)
        self.support = support.reshape(n_groups, n_lines) > 0
        values = aggregate.reshape(n_groups, n_lines, -1)
        line_average = values.sum(axis=0) / self.support.sum(axis=0)[:, None]
        self.values = ((values - line_average[None, :, :]) * self.support[:, :, None]).astype(np.float32)
        self.chemistry = {}
        self.solvers = {}
        self.responses = {}
        self.transformed = {}
        self.latent = {}
        self.gene_spaces = {}
        self.line_spaces = {}

    @cached_property
    def train_entities(self):
        train = self.fit.train.rows
        return np.flatnonzero(~train.duplicated([GROUP, ENTITY]).to_numpy())

    @cached_property
    def plate_projection(self):
        """Remove fitted plate-associated trends when learning chemical slopes."""
        train = self.fit.train.rows
        ix = np.flatnonzero(~train.duplicated([GROUP, KEY[0]]).to_numpy())
        plate = pd.get_dummies(train.iloc[ix]["plate_id"], dtype=np.float32).to_numpy()
        distribution, _ = mean_units(plate, self.group_inverse[ix], len(self.groups))
        p = distribution - distribution.mean(axis=0)
        return np.eye(len(self.groups), dtype=np.float32) - p @ np.linalg.solve(
            p.T @ p + np.eye(p.shape[1], dtype=np.float32) * 0.5, p.T)

    def features(self, chemistry):
        if chemistry not in self.chemistry:
            if chemistry == "mixed":
                a, q_a = self.features("ecfp")
                b, q_b = self.features("cls")
                self.chemistry[chemistry] = (np.column_stack((a, b)) / np.sqrt(np.float32(2)),
                                             np.column_stack((q_a, q_b)) / np.sqrt(np.float32(2)))
            else:
                feature = {"ecfp": "E", "cls": "C"}[chemistry]
                train = self.fit.train.chemical[feature][self.train_entities].astype(np.float32)
                query = self.fit.query.chemical[feature][self.query_entity_first].astype(np.float32)
                if chemistry == "cls":
                    mean = train.mean(axis=0)
                    train, query = train - mean, query - mean
                mean_by_group, _ = mean_units(unit_norm(train), self.group_inverse[self.train_entities],
                                              len(self.groups))
                mean = mean_by_group.mean(axis=0)
                self.chemistry[chemistry] = mean_by_group - mean, unit_norm(query) - mean
        return self.chemistry[chemistry]

    def solver(self, chemistry, plate_adjust):
        key = chemistry, plate_adjust
        if key not in self.solvers:
            train, query = self.features(chemistry)
            if plate_adjust:
                train = self.plate_projection @ train
            scale = np.maximum(np.square(train).sum(axis=1).mean(), 1e-8)
            eigenvalues, directions = eigh(train @ train.T / scale, check_finite=False)
            self.solvers[key] = (np.maximum(eigenvalues, 0), directions,
                                 ((query @ train.T / scale) @ directions).astype(np.float32))
        return self.solvers[key]

    @cached_property
    def ordinary_consensus(self):
        return weighted_consensus(self.values, self.support.astype(np.float32))

    def consensus(self, method):
        if method == "mean":
            return self.ordinary_consensus
        if method == "precision":
            deviation = self.values - self.ordinary_consensus[:, None]
            variance = (np.square(deviation).mean(axis=2) * self.support).sum(axis=0) / self.support.sum(axis=0)
            prior = max(float(np.median(variance)), 1e-9)
            weight = np.clip(prior / np.maximum(variance, 0.5 * prior), 0.5, 2)
            return weighted_consensus(self.values, self.support * weight[None])
        center = self.ordinary_consensus
        for _ in range(3):
            rms = np.sqrt(np.square(self.values - center[:, None]).mean(axis=2))
            typical = np.ma.median(np.ma.array(rms, mask=~self.support), axis=1).filled(0.0)
            weight = np.minimum(1.0, (1.25 * typical[:, None] + 1e-8) / (rms + 1e-8))
            center = weighted_consensus(self.values, self.support * weight)
        return center

    @cached_property
    def control_line_space(self):
        rows = self.fit.train.rows
        unique = np.flatnonzero(~rows.duplicated([CONTROL, LINE]).to_numpy())
        indexes = pd.Index(self.lines).get_indexer(rows.iloc[unique][LINE])
        mean_control, _ = mean_units(self.fit.train.control[unique], indexes, len(self.lines))
        return mean_control - mean_control.mean(axis=0)

    @cached_property
    def line_covariance(self):
        deviations = (self.values - self.ordinary_consensus[:, None]) * self.support[:, :, None]
        tasks = np.moveaxis(deviations, 1, 0).reshape(len(self.lines), -1)
        # Dividing by square root of task support keeps the estimate positive semidefinite.
        tasks /= np.sqrt(self.support.sum(axis=0))[:, None]
        cov = tasks @ tasks.T
        return cov - cov.mean(axis=0) - cov.mean(axis=1, keepdims=True) + cov.mean()

    def line_columns(self, mode, rank):
        n = len(self.lines)
        if mode == "none":
            return np.zeros((n, 0), dtype=np.float32)
        if mode == "identity":
            return np.eye(n, dtype=np.float32) - np.float32(1 / n)
        rank = min(rank, n - 1)
        if rank == 0:
            return np.zeros((n, 0), dtype=np.float32)
        if mode == "empirical":
            _, vectors = eigh(self.line_covariance, subset_by_index=(n - rank, n - 1), check_finite=False)
            return vectors[:, ::-1].astype(np.float32)
        vectors, _, _ = randomized_svd(self.control_line_space, n_components=rank, random_state=42)
        return vectors.astype(np.float32)

    def genes(self, basis):
        if basis == "full":
            return None
        rank = min(32, *self.ordinary_consensus.shape)
        if basis == "pca32":
            target = self.ordinary_consensus - self.ordinary_consensus.mean(axis=0)
            _, _, vectors = randomized_svd(target, n_components=rank, random_state=42)
            return vectors.T.astype(np.float32)
        even = np.asarray([i % 2 == 0 for i in range(len(self.lines))])
        use_left, use_right = self.support * even, self.support * ~even
        keep = use_left.any(axis=1) & use_right.any(axis=1)
        if keep.sum() < 3:
            return self.genes("pca32")
        left = weighted_consensus(self.values, use_left.astype(np.float32))[keep]
        right = weighted_consensus(self.values, use_right.astype(np.float32))[keep]
        return reproducible_gene_basis(left, right, rank)

    def targets(self, spec):
        key = tuple(spec[c] for c in ("gene_basis", "line_basis", "line_rank", "pooling"))
        if key not in self.responses:
            if spec["gene_basis"] not in self.gene_spaces:
                self.gene_spaces[spec["gene_basis"]] = self.genes(spec["gene_basis"])
            basis = self.gene_spaces[spec["gene_basis"]]
            center = self.consensus(spec["pooling"])
            line_key = spec["line_basis"], spec["line_rank"]
            if line_key not in self.line_spaces:
                self.line_spaces[line_key] = self.line_columns(*line_key)
            line = self.line_spaces[line_key]
            if line.shape[1]:
                delta = self.values - center[:, None]
                if basis is not None:
                    delta = delta @ basis
                task = np.empty((len(center), line.shape[1], delta.shape[-1]), dtype=np.float32)
                regularization = 0.25 * np.eye(line.shape[1], dtype=np.float32)
                for group, mask in enumerate(self.support):
                    load = line[mask]
                    task[group] = np.linalg.solve(load.T @ load + regularization, load.T @ delta[group, mask])
            if basis is not None:
                center = center @ basis
            output = np.concatenate((center[:, None], task), axis=1) if line.shape[1] else center[:, None]
            output -= output.mean(axis=0)
            self.responses[key] = (output.reshape(len(center), -1), output.shape[1:], basis, line)
        return key, self.responses[key]

    def predict(self, spec):
        key, (target, shape, basis, line) = self.targets(spec)
        if shape[1] == 0:
            return self.fit.offset.copy()
        label = spec["chemistry"], spec["plate_adjust"]
        eigenvalue, directions, query = self.solver(*label)
        transform_key = label, key
        if transform_key not in self.transformed:
            if spec["plate_adjust"]:
                target = self.plate_projection @ target
            self.transformed[transform_key] = directions.T @ target
        latent_key = transform_key, spec["alpha"]
        if latent_key not in self.latent:
            coordinates = self.transformed[transform_key] / (eigenvalue[:, None] + spec["alpha"])
            self.latent[latent_key] = (query @ coordinates).reshape(len(query), *shape)
        consensus = self.latent[latent_key]
        residual = consensus[self.query_entity_inverse, 0]
        if line.shape[1]:
            by_row = np.zeros((len(residual), line.shape[1]), dtype=np.float32)
            found = self.query_line_index >= 0
            by_row[found] = line[self.query_line_index[found]]
            task = consensus[self.query_entity_inverse, 1:]
            residual += spec["line_gain"] * np.einsum("nl,nlp->np", by_row, task, optimize=True)
        if basis is not None:
            residual = residual @ basis.T
        return (self.fit.offset + residual).astype(np.float32)


def predict(fit, spec):
    if not hasattr(fit, "_r13_transfer"):
        fit._r13_transfer = TransferFit(fit)
    return fit._r13_transfer.predict(spec)
