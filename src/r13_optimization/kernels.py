"""Compound-held-out kernel and nearest-neighbor prediction of the residual over the pool baseline."""

from functools import cached_property

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist

from src.r11_chemical_probe.dataset import CONTROL, ENTITY, GROUP, KEY, LINE, POOL, check_split, weights_for
from src.r11_chemical_probe.models import Interaction


REGULARIZATION = (0.001, 0.01, 0.1, 1.0)


def candidates():
    """Bounded path families with settings relative to normalized sample weight."""
    specs = []
    for chemistry in ("ecfp", "ecfp3", "maccs", "physchem", "chemberta", "mixed"):
        for alpha in REGULARIZATION:
            specs.append(dict(name=f"K_{chemistry}_{alpha:g}", method="ridge", chemistry=chemistry,
                              context="none", alpha=alpha, target="contrast"))
    for chemistry, context in (("ecfp", "line"), ("ecfp", "rna"), ("mixed", "line"),
                               ("mixed", "rna"), ("physchem", "rna")):
        for alpha in REGULARIZATION:
            specs.append(dict(name=f"K_{chemistry}_{context}_{alpha:g}", method="ridge", chemistry=chemistry,
                              context=context, alpha=alpha, target="contrast"))
    for chemistry, context in (("ecfp", "none"), ("ecfp", "rna"), ("mixed", "none"), ("mixed", "rna")):
        for alpha in (0.003, 0.03, 0.3):
            specs.append(dict(name=f"K_robust_{chemistry}_{context}_{alpha:g}", method="robust",
                              chemistry=chemistry, context=context, alpha=alpha, target="contrast"))
    for context in ("none", "rna"):
        for alpha in (0.01, 0.1):
            specs.append(dict(name=f"K_response_ecfp_{context}_{alpha:g}", method="ridge",
                              chemistry="ecfp", context=context, alpha=alpha, target="response"))
    for chemistry in ("ecfp", "maccs", "mixed"):
        for peers, power in ((3, 6), (8, 4), (16, 2)):
            for line_mix in (0.0, 0.75):
                specs.append(dict(name=f"KNN_{chemistry}_{peers}_{line_mix:g}", method="neighbor",
                                  chemistry=chemistry, peers=peers, power=power, line_mix=line_mix))
    return specs


def available(fit, spec):
    """Check optional published inputs by metadata alone."""
    chemistry = spec["chemistry"]
    if chemistry in ("chemberta", "mixed"):
        return "C" in fit.train.chemical and "C" in fit.query.chemical
    if chemistry in ("maccs", "physchem"):
        return all("smiles" in block.rows and block.rows.smiles.fillna("").ne("").all()
                   for block in (fit.train, fit.query))
    return True


def _tanimoto(left, right):
    hits = left @ right.T
    mass = (left**2).sum(axis=1)[:, None] + (right**2).sum(axis=1)
    return hits / np.maximum(mass - hits, 1e-12)


def _neighbor_weights(similarity, k, power):
    """Rank whole compound groups and split weight at a tied cutoff."""
    k = min(k, similarity.shape[1])
    boundary = np.partition(similarity, -k, axis=1)[:, -k, None]
    above = similarity > boundary
    ties = similarity == boundary
    share = (k - above.sum(axis=1)) / ties.sum(axis=1)
    return (above + ties * share[:, None]) * similarity**power


def _rbf(train, query, standardize=False):
    train, query = train.astype(np.float64), query.astype(np.float64)
    if standardize:
        mean, spread = train.mean(axis=0), train.std(axis=0)
        keep = spread > 1e-8
        train = (train[:, keep] - mean[keep]) / spread[keep]
        query = (query[:, keep] - mean[keep]) / spread[keep]
    within = cdist(train, train, metric="sqeuclidean")
    positive = within[within > 1e-12]
    width = np.median(positive) if len(positive) else 1.0
    return np.exp(-within / width), np.exp(-cdist(query, train, metric="sqeuclidean") / width)


def _molecule_features(smiles, chemistry):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors

    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    if any(mol is None for mol in molecules):
        raise ValueError("Kernel chemistry received an invalid canonical SMILES")
    if chemistry == "maccs":
        bits = np.zeros((len(molecules), 167), dtype=np.float64)
        for mol, vector in zip(molecules, bits):
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), vector)
        return bits
    funcs = (Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumHDonors,
             Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds, Descriptors.NumAromaticRings,
             Descriptors.NumAliphaticRings, Descriptors.NumSaturatedRings, Descriptors.FractionCSP3,
             Descriptors.NumHeteroatoms, Descriptors.HeavyAtomCount, rdMolDescriptors.CalcNumBridgeheadAtoms,
             rdMolDescriptors.CalcNumSpiroAtoms, Descriptors.MolMR)
    return np.asarray([[f(m) for f in funcs] for m in molecules])


def _features(train_kernel, query_kernel, rank=None):
    """Spectral kernel map whose landmarks are the unique training units."""
    values, vectors = eigh(train_kernel, check_finite=False)
    keep = np.flatnonzero(values > max(values[-1], 1.0) * 1e-10)
    if rank is not None:
        keep = keep[-rank:]
    roots = np.sqrt(values[keep])
    basis = vectors[:, keep]
    return basis * roots, query_kernel @ (basis / roots)


def _center_pool(values, pools, mass):
    out = np.empty_like(values)
    for pos in pools:
        out[pos] = values[pos] - np.average(values[pos], axis=0, weights=mass[pos])
    return out


class RidgePath:
    def __init__(self, design, target, weight):
        self.design = design
        self.target = target
        self.weight = weight
        gram, response = self.statistics(weight)
        values, self.basis = eigh(gram, check_finite=False)
        self.values = np.maximum(values, 0.0)
        self.response = self.basis.T @ response
        self.coefficients = {}

    def statistics(self, weight):
        weighted = self.design.T * weight
        return weighted @ self.design, weighted @ self.target

    def solve(self, alpha, robust=False):
        case = (alpha, robust)
        if case not in self.coefficients:
            coef = self.basis @ (self.response / (self.values[:, None] + alpha))
            if robust:
                rms = np.sqrt(np.mean(self.target**2, axis=1))
                scale = max(1e-8, np.dot(self.weight, rms))
                for _ in range(3):
                    error = self.target - self.design @ coef
                    spread = np.sqrt(np.mean(error**2, axis=1) + (0.05 * scale)**2)
                    gram, response = self.statistics(self.weight * scale / spread)
                    gram.flat[::len(gram) + 1] += alpha
                    coef = cho_solve(cho_factor(gram, check_finite=False), response, check_finite=False)
            self.coefficients[case] = coef
        return self.coefficients[case]


class KernelFit:
    def __init__(self, fit):
        check_split(fit.train, fit.query)
        self.train, self.query = fit.train, fit.query
        self.offset = fit.offset
        self.residual = np.asarray(fit.residual)
        self.weights = fit.weight / fit.weight.sum()
        self.pools = tuple(self.train.rows.groupby(POOL, sort=False).indices.values())
        pooled = self.train.rows.assign(_pool=self.train.rows.groupby(POOL, sort=False).ngroup())
        self.pool_weights = weights_for(pooled, ("_pool", GROUP, ENTITY, KEY[0]))
        self.drug_pos, self.train_drug = self._unit_positions(self.train.rows, (ENTITY,))
        self.query_pos, self.query_drug = self._unit_positions(self.query.rows, (ENTITY,))
        self.groups, self.group_index = np.unique(self.train.rows[GROUP], return_inverse=True)
        self.entity_group = self.group_index[self.drug_pos]
        self.kernels = {}
        self.designs = {}
        self.paths = {}
        self.neighbors = {}

    @staticmethod
    def _unit_positions(rows, fields):
        indices = rows.groupby(list(fields), sort=False, dropna=False)
        return np.flatnonzero(~rows.duplicated(list(fields))), indices.ngroup().to_numpy()

    @cached_property
    def target(self):
        return _center_pool(self.residual, self.pools, self.pool_weights)

    def chemical(self, chemistry):
        if chemistry not in self.kernels:
            if chemistry == "mixed":
                a, b = self.chemical("ecfp"), self.chemical("chemberta")
                pair = tuple((x + y) * 0.5 for x, y in zip(a, b))
            elif chemistry in ("maccs", "physchem"):
                tr = self.train.rows.iloc[self.drug_pos].smiles
                qu = self.query.rows.iloc[self.query_pos].smiles
                train = _molecule_features(tr, chemistry)
                query = _molecule_features(qu, chemistry)
                pair = (_tanimoto(train, train), _tanimoto(query, train)) if chemistry == "maccs" else (
                    _rbf(train, query, standardize=True))
            elif chemistry == "chemberta":
                train = self.train.chemical["C"][self.drug_pos]
                query = self.query.chemical["C"][self.query_pos]
                pair = _rbf(train, query)
            else:
                train = self.train.chemical["E"][self.drug_pos].astype(np.float64)
                query = self.query.chemical["E"][self.query_pos].astype(np.float64)
                power = 1 if chemistry == "ecfp" else 3
                pair = (_tanimoto(train, train)**power, _tanimoto(query, train)**power)
            self.kernels[chemistry] = pair
        return self.kernels[chemistry]

    @cached_property
    def rna(self):
        train_meta = self.train.rows[[CONTROL, LINE]].reset_index(drop=True)
        control_pos = np.flatnonzero(~train_meta.duplicated())
        train_rna = self.train.control[control_pos]
        train_line = train_meta.iloc[control_pos][LINE].to_numpy()
        means = {line: train_rna[train_line == line].mean(axis=0) for line in pd.unique(train_line)}
        tr_pos, tr_idx = self._unit_positions(self.train.rows, (LINE,))
        qu_pos, qu_idx = self._unit_positions(self.query.rows, (CONTROL, LINE))
        tr = np.stack([means[line] for line in self.train.rows.iloc[tr_pos][LINE]])
        qu = np.stack([means.get(line, self.query.control[i])
                       for i, line in zip(qu_pos, self.query.rows.iloc[qu_pos][LINE])])
        if len(tr) > 1:
            pca, _ = Interaction._fit(tr, 12)
            tr, qu = pca.transform(tr), pca.transform(qu)
        within, cross = _rbf(tr, qu)
        a, b = _features(within, cross, rank=8)
        return a[tr_idx], b[qu_idx]

    @cached_property
    def lines(self):
        train_lines = pd.Index(self.train.rows[LINE].unique())
        identity = np.eye(len(train_lines))
        train_index = train_lines.get_indexer(self.train.rows[LINE])
        query_index = train_lines.get_indexer(self.query.rows[LINE])
        queries = np.zeros((len(self.query.rows), len(train_lines)))
        ok = query_index >= 0
        queries[ok] = identity[query_index[ok]]
        return identity[train_index], queries

    def design(self, chemistry, context):
        case = (chemistry, context)
        if case not in self.designs:
            a, b = _features(*self.chemical(chemistry))
            x, z = a[self.train_drug], b[self.query_drug]
            if context != "none":
                ctr, cqu = self.rna if context == "rna" else self.lines
                limit = 16 if context == "rna" else 6
                atr = x[:, -limit:, None] * ctr[:, None, :]
                aqu = z[:, -limit:, None] * cqu[:, None, :]
                x = np.column_stack((x, atr.reshape(len(x), -1))) * np.sqrt(0.5)
                z = np.column_stack((z, aqu.reshape(len(z), -1))) * np.sqrt(0.5)
            mean = self.weights @ x
            self.designs[case] = (x - mean, z - mean)
        return self.designs[case]

    def ridge(self, spec):
        chemistry, context, target = (spec[k] for k in ("chemistry", "context", "target"))
        case = (chemistry, context, target)
        train, query = self.design(chemistry, context)
        if not train.shape[1]:
            return self.offset.copy()
        if case not in self.paths:
            x, y = train, self.residual
            if target == "contrast":
                x, y = _center_pool(train, self.pools, self.pool_weights), self.target
            self.paths[case] = RidgePath(x, y, self.weights)
        coef = self.paths[case].solve(spec["alpha"], robust=spec["method"] == "robust")
        return self.offset + (query @ coef).astype(np.float32)

    @cached_property
    def group_response(self):
        response = np.zeros((len(self.groups), self.target.shape[1]), dtype=np.float64)
        np.add.at(response, self.group_index, self.target * self.weights[:, None])
        mass = np.bincount(self.group_index, weights=self.weights, minlength=len(self.groups))
        return response / mass[:, None]

    @cached_property
    def line_response(self):
        out = {}
        for line, ix in self.train.rows.groupby(LINE, sort=False).indices.items():
            group, codes = np.unique(self.group_index[ix], return_inverse=True)
            value = np.zeros((len(group), self.target.shape[1]))
            np.add.at(value, codes, self.target[ix] * self.weights[ix, None])
            weight = np.bincount(codes, weights=self.weights[ix], minlength=len(group))
            out[line] = group, value / weight[:, None]
        return out

    def neighbor(self, spec):
        chemistry, k, power = (spec[field] for field in ("chemistry", "peers", "power"))
        case = (chemistry, k, power)
        if case not in self.neighbors:
            _, similarity = self.chemical(chemistry)
            group_similarity = np.zeros((len(similarity), len(self.groups)))
            for i, group in enumerate(self.entity_group):
                np.maximum(group_similarity[:, group], similarity[:, i], out=group_similarity[:, group])
            weight = _neighbor_weights(group_similarity, k, power)
            total = weight.sum(axis=1, keepdims=True)
            global_value = (weight @ self.group_response) / np.maximum(total, 1e-12)
            self.neighbors[case] = (global_value.astype(np.float32), group_similarity)
        global_value, group_similarity = self.neighbors[case]
        prediction = global_value[self.query_drug].copy()
        if spec["line_mix"]:
            for line, ix in self.query.rows.groupby(LINE, sort=False).indices.items():
                if line in self.line_response:
                    groups, value = self.line_response[line]
                    weight = _neighbor_weights(group_similarity[np.ix_(self.query_drug[ix], groups)], k, power)
                    total = weight.sum(axis=1, keepdims=True)
                    local = (weight @ value) / np.maximum(total, 1e-12)
                    ok = total[:, 0] > 0
                    prediction[ix[ok]] += spec["line_mix"] * (local[ok] - prediction[ix[ok]])
        return self.offset + prediction


def predict(fit, spec):
    """Use only train responses and same-role vehicle controls for held-out entities."""
    if not available(fit, spec):
        raise ValueError(f"Unavailable public input for {spec['name']}")
    engine = getattr(fit, "_r13_kernel_fit", None)
    if engine is None:
        engine = KernelFit(fit)
        fit._r13_kernel_fit = engine
    if spec["method"] == "neighbor":
        return engine.neighbor(spec)
    return engine.ridge(spec)
