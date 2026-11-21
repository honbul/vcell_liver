"""Fit-only chemistry, treatment context and position predictions for HepaRG."""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.r13_optimization.kernels import _neighbor_weights, _tanimoto
from src.r14_cross_organ.heparg_eval import GAINS, center
from src.r14_cross_organ.heparg_model import ALPHA, NEIGHBORS
from src.r14_cross_organ.transfer_models import fingerprint


MODELS = (
    "dose_shared_ridge",
    "same_rank_ridge",
    "same_uM_decade_ridge",
    "target_moa_neighbors",
    "joint_dose_shared_ridge",
    "dose_response",
    "joint_dose_response",
)
PERM_SEEDS = (41, 42, 43)
POSITION_ARMS = (
    "B_pos",
    "B_pos_dose_matched",
    "B_context",
    "B_context_plus_position",
    *(f"B_pos_perm{seed}" for seed in PERM_SEEDS),
    *(f"B_context_plus_perm{seed}" for seed in PERM_SEEDS),
)
RESIDUAL = ("B_context_plus_position", *(f"B_context_plus_perm{seed}" for seed in PERM_SEEDS))
HILLS = (0.1, 1.0, 10.0, "relative_0.1max")


def strata(query):
    rank = query.dose_rank.to_numpy()
    dimension = query.dose_decade.to_numpy()
    return {
        "rank7": rank == 7,
        "rank8": rank == 8,
        "ge10uM": dimension >= 4,
        **{f"decade_{level}": dimension == level for level in range(6)},
    }


def parent_groups(query, flags):
    return {label: (flag, *np.unique(query.parent_key.to_numpy()[flag], return_inverse=True, return_counts=True)[1:])
            for label, flag in flags.items() if flag.any()}


def grouped_loss(group, estimates):
    keep, parent_index, count = group
    weight = 1.0 / count[parent_index]
    return weight @ estimates[keep] / len(count)


def design_gain(rows, ec50):
    condition = rows.dose_uM.to_numpy(dtype=np.float32)
    half = 0.1 * rows.max_uM.to_numpy(dtype=np.float32) if isinstance(ec50, str) else ec50
    return condition / (condition + half)


def aggregate(rows, y, key_fields):
    group_codes = rows.groupby(key_fields, sort=True).ngroup().to_numpy()
    first = np.full(group_codes.max() + 1, len(rows), dtype=np.int64)
    np.minimum.at(first, group_codes, np.arange(len(rows)))
    counts = np.bincount(group_codes)
    totals = np.zeros((len(first), y.shape[1]), dtype=np.float32)
    np.add.at(totals, group_codes, y)
    totals /= counts[:, None]
    return rows.iloc[first].reset_index(drop=True), totals


def hill_direction(rows, y, ec50):
    dose = design_gain(rows, ec50)
    by_source = rows.groupby(["dataset", "parent_key"], sort=True).indices
    first, response = [], []
    for indices in by_source.values():
        g = dose[indices]
        first.append(indices[0])
        response.append((g @ y[indices]) / max(float(g @ g), 1e-8))
    return aggregate(rows.iloc[first].reset_index(drop=True), np.asarray(response), ["parent_key"])


class Kernel:
    def __init__(self, rows):
        keys, fps = fingerprint(rows)
        self.index = pd.Index(keys)
        self.full = sum(coef * _tanimoto(fp, fp) for coef, fp in zip((0.7, 0.3), fps)).astype(np.float32)
        if not np.isfinite(self.full).all():
            raise ValueError("Invalid chemical Gram matrix")

    def design(self, fit, query):
        if set(fit) & set(query):
            raise ValueError("Held parent entered chemical fit")
        i, j = self.index.get_indexer(fit), self.index.get_indexer(query)
        if np.min(i) < 0 or np.min(j) < 0:
            raise ValueError("A structure was omitted from chemical Gram matrix")
        gram, cross = self.full[np.ix_(i, i)], self.full[np.ix_(j, i)]
        mean = gram.mean(axis=0)
        overall = mean.mean()
        return gram - mean[None, :] - mean[:, None] + overall, cross - mean[None, :] - cross.mean(axis=1, keepdims=True) + overall

    def paths(self, fitting, y, query, scalar=None):
        if fitting.parent_key.duplicated().any():
            raise ValueError("Every fitted direction must represent one parent")
        parents, index = np.unique(query.parent_key, return_inverse=True)
        gram, cross = self.design(fitting.parent_key, parents)
        yc = y - y.mean(axis=0)
        diag = np.diag_indices(len(gram))
        if scalar is not None:
            scalar = scalar[:, None]
        for alpha in ALPHA:
            reg = gram.copy()
            reg[diag] += alpha
            directions = cross @ cho_solve(cho_factor(reg, check_finite=False), yc, check_finite=False)
            pred = directions[index]
            if scalar is not None:
                pred *= scalar
            yield alpha, pred


def centered_blocks(query, parts, templates, kernel, matching_field, arm):
    for val, q_idx in query.groupby(matching_field, sort=True).indices.items():
        eligible = templates[0][matching_field].eq(val).to_numpy()
        if not eligible.any():
            continue
        train, y = aggregate(
            templates[0].loc[eligible].reset_index(drop=True), templates[1][eligible], ["parent_key"],
        )
        for alpha, pred in kernel.paths(train, y, query.iloc[q_idx]):
            parts[alpha][q_idx] = pred
    for alpha, prediction in parts.items():
        yield arm, (alpha,), center(query, prediction)


def annotation_paths(query, templates, label):
    masks = query.groupby("dose_rank", sort=True).indices
    pred = {k: np.zeros((len(query), templates[1].shape[1]), dtype=np.float32) for k in NEIGHBORS}
    for rank, q_idx in masks.items():
        eligible = templates[0].dose_rank.eq(rank).to_numpy()
        if not eligible.any():
            continue
        fit, y = aggregate(templates[0].loc[eligible].reset_index(drop=True), templates[1][eligible], ["parent_key"])
        keys, inv = np.unique(query.parent_key.iloc[q_idx], return_inverse=True)
        sim = np.asarray([[len(label[k] & label[j]) / len(label[k] | label[j]) if label[k] and label[j] else 0.0
                           for j in fit.parent_key] for k in keys], dtype=np.float32)
        for k in NEIGHBORS:
            weight = _neighbor_weights(sim, k, 2).astype(np.float32)
            denom = weight.sum(axis=1, keepdims=True)
            weight /= np.where(denom > 0, denom, 1)
            pred[k][q_idx] = (weight @ y)[inv]
    for k, prediction in pred.items():
        yield "target_moa_neighbors", (k,), center(query, prediction)


def candidates(train, y, query, catalogue, kernel):
    fields = ["dataset", "parent_key", "dose_rank", "dose_uM", "max_uM"]
    by_dose = aggregate(train, y, fields)
    own = by_dose[0].dataset.eq(query.dataset.iloc[0]).to_numpy()
    local = by_dose[0].loc[own].reset_index(drop=True), by_dose[1][own]
    for prefix, cohort in (("", local), ("joint_", by_dose)):
        rank_profiles = aggregate(*cohort, ["dataset", "parent_key", "dose_rank"])
        source_profiles = aggregate(*rank_profiles, ["dataset", "parent_key"])
        parent_profiles = aggregate(*source_profiles, ["parent_key"])
        for alpha, prediction in kernel.paths(*parent_profiles, query):
            yield f"{prefix}dose_shared_ridge", (alpha,), center(query, prediction)
        for ec50 in HILLS:
            vectors = hill_direction(*cohort, ec50)
            for alpha, prediction in kernel.paths(*vectors, query, design_gain(query, ec50)):
                yield f"{prefix}dose_response", (ec50, alpha), center(query, prediction)
    for field, arm in (("dose_rank", "same_rank_ridge"), ("dose_decade", "same_uM_decade_ridge")):
        parts = {alpha: np.zeros((len(query), y.shape[1]), dtype=np.float32) for alpha in ALPHA}
        yield from centered_blocks(query, parts, local, kernel, field, arm)
    yield from annotation_paths(query, local, catalogue)


def permute(rows, seed, dose_matched=False):
    rng = np.random.default_rng(seed)
    fields = ["plate", "dose_decade"] if dose_matched else ["plate"]
    source = rows.position.to_numpy()
    copied = source.copy()
    for ix in rows.groupby(fields, sort=True).indices.values():
        copied[ix] = rng.permutation(source[ix])
    return copied


def position_prediction(fit, y, query, fit_coordinate, query_coordinate, dose_matched=False):
    fit = fit.assign(coordinate=fit_coordinate)
    query = query.assign(coordinate=query_coordinate)
    fields = ["coordinate", "dose_decade"] if dose_matched else ["coordinate"]
    grouped, values = aggregate(fit, y, fields + ["plate"])
    guess = np.zeros((len(query), y.shape[1]), dtype=np.float32)
    support = np.zeros(len(query), dtype=bool)
    fitting = grouped.groupby(fields, sort=False).indices
    for loc, idx in query.groupby(fields, sort=False).indices.items():
        key = loc[0] if len(fields) == 1 and isinstance(loc, tuple) else loc
        if key not in fitting:
            continue
        donors = fitting[key]
        donor_plates = grouped.plate.to_numpy()[donors]
        summed = values[donors].sum(axis=0)
        for plate, local in query.iloc[idx].groupby("plate", sort=False).indices.items():
            row = idx[local]
            same = donor_plates == plate
            count = len(donors) - int(same.sum())
            if count:
                guess[row] = (summed - values[donors[same]].sum(axis=0)) / count
                support[row] = True
    return center(query, guess), support


def context_features(fit, query):
    def log_doses(rows):
        actual = np.log10(rows.dose_uM.to_numpy())
        maximum = np.log10(rows.max_uM.to_numpy())
        return np.column_stack((actual, maximum, actual - maximum, actual**2, maximum**2, (actual - maximum)**2))

    train, held = map(log_doses, (fit, query))
    avg, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std > 1e-6, std, 1)
    x, z = [np.clip((m - avg) / std, -4, 4) for m in (train, held)]
    for field in ("dose_rank", "chemical_family", "generic_scaffold"):
        scope = fit.groupby(field).parent_key.nunique()
        categories = scope[scope >= (5 if field == "generic_scaffold" else 1)].index.to_numpy()
        x = np.column_stack((x, fit[field].to_numpy()[:, None] == categories))
        z = np.column_stack((z, query[field].to_numpy()[:, None] == categories))
    return center(fit, x.astype(np.float32)), center(query, z.astype(np.float32))


def location_candidates(fit, y, query, coordinates, fit_indices, query_indices):
    x, z = context_features(fit, query)
    cov = x.T @ x
    cov[np.diag_indices(len(cov))] += 1
    c = cho_solve(cho_factor(cov, check_finite=False), x.T @ y, check_finite=False)
    yield "B_context", z @ c, np.ones(len(query), dtype=bool)
    raw = coordinates["position"]
    yield "B_pos", *position_prediction(fit, y, query, raw[fit_indices], raw[query_indices])
    yield "B_pos_dose_matched", *position_prediction(fit, y, query, raw[fit_indices], raw[query_indices], True)
    residual = y - x @ c
    yield "B_context_plus_position", *position_prediction(
        fit, residual, query, raw[fit_indices], raw[query_indices], True,
    )
    for seed in PERM_SEEDS:
        coordinate = coordinates[f"perm{seed}"]
        yield f"B_pos_perm{seed}", *position_prediction(fit, y, query, coordinate[fit_indices], coordinate[query_indices])
        coordinate = coordinates[f"matched_perm{seed}"]
        yield f"B_context_plus_perm{seed}", *position_prediction(
            fit, residual, query, coordinate[fit_indices], coordinate[query_indices], True,
        )


def gain_scores(obs, prediction, scales=GAINS, anchor=None):
    obs2 = np.einsum("ij,ij->i", obs, obs) / obs.shape[1]
    pred2 = np.einsum("ij,ij->i", prediction, prediction) / obs.shape[1]
    mixed = np.einsum("ij,ij->i", obs, prediction) / obs.shape[1]
    if anchor is None:
        errs = obs2[:, None] - 2 * mixed[:, None] * scales + pred2[:, None] * scales**2
    else:
        anchor2 = np.einsum("ij,ij->i", anchor, anchor) / obs.shape[1]
        obs_anchor = np.einsum("ij,ij->i", obs, anchor) / obs.shape[1]
        cross = np.einsum("ij,ij->i", anchor, prediction) / obs.shape[1]
        a = scales[None, :, None]
        g = scales[None, None, :]
        errs = (obs2[:, None, None] - 2 * obs_anchor[:, None, None] * a + anchor2[:, None, None] * a**2
                - 2 * mixed[:, None, None] * g + pred2[:, None, None] * g**2 + 2 * cross[:, None, None] * a * g)
    return np.sqrt(np.maximum(errs, 0))


def score_inner(obs, roles, groups, location, predictors):
    context = None
    summary = defaultdict(dict)
    for arm, pred, _ in location:
        if arm == "B_context":
            context = pred
        scores = gain_scores(obs, pred, anchor=context if arm in RESIDUAL else None)
        for label, group in groups.items():
            summary[label][(arm, ())] = grouped_loss(group, scores.reshape(len(obs), -1)).reshape(scores.shape[1:])
    for arm, config, pred in predictors:
        scores = gain_scores(obs, pred)
        for name in roles:
            if name in groups:
                summary[name][(arm, config)] = grouped_loss(groups[name], scores)
    return summary


def pick_settings(folds):
    result = {}
    for label in sorted(set().union(*(f.keys() for f in folds))):
        combined = defaultdict(list)
        for fold in folds:
            for names, values in fold.get(label, {}).items():
                combined[names].append(values)
        result[label] = best = {}
        context_idx = np.argmin(np.mean(combined["B_context", ()], axis=0))
        for (arm, model), parts in combined.items():
            avg = np.mean(parts, axis=0)
            if arm in RESIDUAL:
                avg = avg[context_idx]
            i = int(np.argmin(avg))
            record = (float(avg[i]), i, str(model))
            if arm not in best or record < best[arm]["tie"]:
                best[arm] = {"config": list(model), "gain": float(GAINS[i]), "tie": record}
    return result
