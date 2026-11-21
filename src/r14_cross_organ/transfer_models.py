"""R14 chemistry ridge and neighbor learners with assay-specific gene support."""

from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.dataset import LINE
from src.r13_optimization.kernels import RidgePath, _features, _neighbor_weights, _tanimoto


ARMS = {
    "D0": ("snu",),
    "D1": ("snu", "tahoe_other"),
    "D2": ("snu", "tahoe_other", "sciplex3", "mixseq"),
    "D3": ("snu", "tahoe_other", "heparg"),
    "D4": ("snu", "tahoe_other", "sciplex3", "mixseq", "heparg"),
}
ALPHAS = (0.1, 1.0, 10.0)
NEIGHBORS = (3, 8)
GAINS = np.arange(9) / 8
POOL = ["source", "plate_id", LINE, "dose_pool", "time_h"]
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@lru_cache(maxsize=None)
def chemistry(smiles):
    canonical = canonical_parent(smiles)
    if not canonical:
        return None
    molecule = Chem.MolFromSmiles(canonical[1])
    key = Chem.MolToInchiKey(molecule).split("-")[0]
    if not key:
        return None
    ecfp = np.zeros(2048, dtype=np.float32)
    maccs = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(MORGAN.GetFingerprint(molecule), ecfp)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(molecule), maccs)
    return key, ecfp, maccs


@lru_cache(maxsize=None)
def scaffold(smiles):
    parent = canonical_parent(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=parent[1]) if parent else ""


def fingerprint(rows):
    known = rows[["parent_key", "smiles"]].drop_duplicates()
    keys = known.parent_key.drop_duplicates().to_list()
    found = {}
    for key, smiles in known.itertuples(index=False, name=None):
        value = chemistry(smiles)
        if not value or value[0] != key:
            raise ValueError(f"Invalid or inconsistent parent structure: {key}")
        found.setdefault(key, value[1:])
    return keys, tuple(np.stack([found[k][j] for k in keys]) for j in range(2))


def weight_rows(rows):
    source = rows.groupby("source").parent_key.transform("nunique").to_numpy()
    group = rows.groupby(["source", "parent_key"]).parent_key.transform("size").to_numpy()
    weights = 1 / (source * group)
    return weights / weights.sum()


def remove_pool(rows, values, weights=None):
    result = np.zeros_like(values)
    for ix in rows.groupby(POOL, sort=False).indices.values():
        pool = values[ix]
        w = np.ones(len(ix)) if weights is None else weights[ix]
        # An assay supplies the same gene mask within one pool.
        support = np.isfinite(pool[0, :])
        result[ix] = pool
        if support.any():
            result[np.ix_(ix, support)] -= np.average(pool[:, support], axis=0, weights=w)
    return result


def sufficient_rows(rows, values, control):
    counts = rows.groupby(POOL, sort=False).parent_key.transform("nunique")
    use = counts.ge(3).to_numpy()
    if not use.any():
        raise ValueError("Training data lack multi-compound pools")
    rows = rows.loc[use].reset_index(drop=True)
    values, control = values[use], control[use]
    group = rows.groupby(POOL + ["parent_key"], sort=False).parent_key.transform("size").to_numpy()
    y = remove_pool(rows, values, 1 / group)
    fields = ["source", "assay_id", "parent_key", LINE, "dose_uM"]
    groups = rows.groupby(fields, sort=False).indices.values()
    indices = list(groups)
    meta = rows.iloc[[idx[0] for idx in indices]].reset_index(drop=True)
    observed = np.stack([np.mean(y[idx], axis=0) for idx in indices])
    mean_control = np.array([np.mean(control[idx]) for idx in indices])
    return meta, observed, mean_control


def ridge_design(train, query, chemical_train, chemical_query, control_tr, control_qu, weight):
    train_index, query_index = [part[LINE].to_numpy() for part in (train, query)]
    all_lines = pd.Index(sorted(set(train_index) | set(query_index)))
    line_train = np.eye(len(all_lines))[all_lines.get_indexer(train_index)]
    line_query = np.eye(len(all_lines))[all_lines.get_indexer(query_index)]
    doses = [np.log10(np.maximum(x.dose_uM.to_numpy(dtype=float), 1e-8)) for x in (train, query)]
    covariates = [np.column_stack((a, b)) for a, b in zip(doses, (control_tr, control_qu))]
    mean = np.average(covariates[0], axis=0, weights=weight)
    std = np.sqrt(np.average((covariates[0] - mean) ** 2, axis=0, weights=weight))
    scales = np.where(std > 1e-6, std, 1)
    contexts = [np.column_stack((line, np.clip((cov - mean) / scales, -4, 4)))
                for line, cov in zip((line_train, line_query), covariates)]
    return tuple(np.column_stack((chem, (chem[:, -6:, None] * context[:, None, :]).reshape(len(chem), -1)))
                 for chem, context in zip((chemical_train, chemical_query), contexts))


def ridge_predictions(train, query, gram, cross, x_idx, z_idx, obs, c_tr, c_qu, weights):
    base, test = _features(gram, cross)
    x, z = ridge_design(train, query, base[x_idx], test[z_idx], c_tr, c_qu, weights)
    support = np.isfinite(obs)
    patterns, inverse = np.unique(support.T, axis=0, return_inverse=True)
    result = {("ridge", alpha): np.zeros((len(query), obs.shape[1]), dtype=np.float32) for alpha in ALPHAS}
    for i, fit in enumerate(patterns):
        if not fit.any():
            continue
        genes = np.flatnonzero(inverse == i)
        w = weights[fit] / weights[fit].sum()
        loc_x = x[fit]
        avg = w @ loc_x
        path = RidgePath(loc_x - avg, obs[np.ix_(fit, genes)], w)
        for alpha in ALPHAS:
            result["ridge", alpha][:, genes] = (z - avg) @ path.solve(alpha)
    return result


def neighbor_predictions(train, query, similarity, x_idx, z_idx, values, weights):
    result = {}
    finite = np.isfinite(values)
    y = np.nan_to_num(values)
    for cut in NEIGHBORS:
        parts = []
        for line in (None, *query[LINE].unique()):
            fit = np.ones(len(train), dtype=bool) if line is None else train[LINE].eq(line).to_numpy()
            ids = np.unique(x_idx[fit])
            if not len(ids):
                continue
            mapping = np.searchsorted(ids, x_idx[fit])
            target = np.zeros((len(ids), values.shape[1]))
            support = np.zeros_like(target)
            np.add.at(target, mapping, y[fit] * weights[fit, None])
            np.add.at(support, mapping, finite[fit] * weights[fit, None])
            available = support > 0
            target /= np.where(available, support, 1)
            applied = np.ones(len(query), dtype=bool) if line is None else query[LINE].eq(line).to_numpy()
            w = _neighbor_weights(similarity[np.ix_(z_idx[applied], ids)], cut, 2)
            sample = w @ target
            den = w @ available
            sample /= np.where(den > 0, den, 1)
            parts.append((applied, sample, den > 0))
        pred = parts[0][1]
        for use, sample, covered in parts[1:]:
            pred[use] = np.where(covered, 0.5 * (pred[use] + sample), pred[use])
        result["neighbor", cut] = pred.astype(np.float32)
    return result


def predict_grid(train, query, values, c_tr, c_qu):
    train, obs, c_tr = sufficient_rows(train.reset_index(drop=True), values, c_tr)
    weights = weight_rows(train)
    training, xfp = fingerprint(train)
    testing, zfp = fingerprint(query)
    x_idx = pd.Index(training).get_indexer(train.parent_key)
    z_idx = pd.Index(testing).get_indexer(query.parent_key)
    gram, cross = [sum(w * _tanimoto(a, b) for w, a, b in zip((0.7, 0.3), features, xfp))
                   for features in (xfp, zfp)]
    if set(training) & set(testing):
        raise ValueError("Held-out parent reached an R14 fit")
    result = ridge_predictions(train, query, gram, cross, x_idx, z_idx, obs, c_tr, c_qu, weights)
    result.update(neighbor_predictions(train, query, cross, x_idx, z_idx, obs, weights))
    return result, cross[z_idx].max(axis=1)
