"""Candidate families and fit-only position estimates for R15."""

from itertools import product

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.r13_optimization.kernels import _tanimoto

from .design import fold_id


GAINS = (0., .125, .25, .5, .75, 1.)


def parent_average(rows, y):
    groups, inverse = np.unique(rows.parent, return_inverse=True)
    total = np.zeros((len(groups), y.shape[1]), dtype=np.float32)
    np.add.at(total, inverse, y)
    return total / np.bincount(inverse)[:, None]


def train_position(train, target, y):
    """Different RNA plates and parents provide same-position donor effects."""
    residual = y.copy()
    for indices in train.groupby("plate_id").indices.values():
        residual[indices] -= residual[indices].mean(axis=0)
    coords = train.groupby("rna_well").indices
    tr_plate, tr_parent = train.plate_id.to_numpy(), train.parent.to_numpy()
    answers = np.zeros((len(target), y.shape[1]), dtype=np.float32)
    for i, row in enumerate(target.itertuples(index=False)):
        if row.rna_well not in coords:
            continue
        pool = coords[row.rna_well]
        pool = pool[(tr_plate[pool] != row.plate_id) & (tr_parent[pool] != row.parent)]
        if len(pool):
            indices = pd.Series(tr_parent[pool]).groupby(tr_parent[pool], sort=False).indices
            answers[i] = np.stack([residual[pool[ii]].mean(0) for ii in indices.values()]).mean(0)
    return answers


def cross_position(train, target, y):
    """For fit labels, use mutually exclusive position cross-fits by parent."""
    cross = np.asarray([fold_id(p, "pos", 3) for p in train.parent])
    same = np.zeros_like(y)
    for fold in range(3):
        excluded = cross == fold
        if excluded.any():
            same[excluded] = train_position(train.loc[~excluded], train.loc[excluded], y[~excluded])
    return same, train_position(train, target, y)


def loss(rows, truth, predicted):
    error = np.square(truth - predicted).mean(1)
    signal = np.square(truth).mean(1)
    by = pd.DataFrame({"parent": rows.parent.to_numpy(), "rmse": np.sqrt(error),
                       "b0": np.sqrt(signal)}).groupby("parent").mean()
    return float(.5 * (by.rmse.mean()/by.b0.mean() + error.sum()/signal.sum()))


def solve(k, y, alpha):
    gram = k.copy()
    gram.flat[::len(gram)+1] += alpha
    return cho_solve(cho_factor(gram, check_finite=False), y, check_finite=False)


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def neural(train, target, training, epochs, snapshot=None):
    import torch

    torch.manual_seed(1501)
    torch.set_num_threads(4)
    scale = max(.05, float(np.std(training)))
    x = torch.as_tensor(np.asarray(train, dtype=np.float32))
    z = torch.as_tensor(np.asarray(target, dtype=np.float32))
    truth = torch.as_tensor(np.asarray(training/scale, dtype=np.float32))
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.GELU(),
                               torch.nn.Dropout(.1), torch.nn.Linear(256, truth.shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    results = {}
    for epoch in range(1, max(epochs)+1):
        model.train()
        for select in torch.randperm(len(x)).split(128):
            error = (model(x[select]) - truth[select]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            optimizer.step()
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():
                results[epoch] = model(z).numpy()*scale
    if snapshot is not None:
        values = {str(k): v.detach().numpy() for k, v in model.state_dict().items()}
        np.savez_compressed(snapshot, scale=scale, **values)
    return results


def candidates(scenario):
    rules = [("tanimoto", a) for a in (.1, 1., 10.)]
    rules += [("sibling", a) for a in (.3, .5, .7)]
    if scenario == "B":
        rules += [("mixed", a, w) for a, w in product((.1, 1., 10.), (.25, .5, .75))]
    rules += [("mlp", e) for e in (5, 15)]
    return rules


def chemical_grid(train, query, train_x, query_x, y, scenario, train_prior, query_prior, rules, snapshot=None):
    response = parent_average(train, y)
    _, first = np.unique(train.parent, return_index=True)
    tx = train_x[first]
    _, qfirst, expand = np.unique(query.parent, return_index=True, return_inverse=True)
    qx = query_x[qfirst]
    gram, cross = _tanimoto(tx, tx), _tanimoto(qx, tx)
    if scenario == "B":
        tv, qv = unit(train_prior[first]), unit(query_prior[qfirst])
        rg, rc = tv @ tv.T, qv @ tv.T
        train_nn, query_nn = np.column_stack((tx, 4*tv)), np.column_stack((qx, 4*qv))
    else:
        train_nn, query_nn = tx, qx
    epochs = [rule[1] for rule in rules if rule[0] == "mlp"]
    nn = neural(train_nn, query_nn, response, epochs, snapshot) if epochs else {}
    results = {}
    for rule in rules:
        name = rule[0]
        if name == "sibling":
            peak = cross.max(axis=1, keepdims=True)
            keep = np.isclose(cross, peak, atol=1e-10, rtol=0) & (peak >= rule[1])
            pred = (keep @ response) / keep.sum(axis=1, keepdims=True).clip(min=1)
            if snapshot is not None:
                np.savez_compressed(snapshot, structure=tx, prototype=response, threshold=rule[1])
        elif name == "mixed":
            _, alpha, w = rule
            coef = solve((1-w)*gram+w*rg, response, alpha)
            pred = ((1-w)*cross+w*rc) @ coef
            if snapshot is not None:
                np.savez_compressed(snapshot, structure=tx, signature=tv, coefficients=coef, mixture=w, alpha=alpha)
        elif name == "tanimoto":
            coef = solve(gram, response, rule[1])
            pred = cross @ coef
            if snapshot is not None:
                np.savez_compressed(snapshot, structure=tx, coefficients=coef, alpha=rule[1])
        else:
            pred = nn[rule[1]]
        results[rule] = pred[expand].astype(np.float32)
    return results
