"""Paired rank, retrieval, and Top-5 diversity estimates on the identical native panel."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from nextgen_h2.assets import KEY
from nextgen_h2.metrics import ScoreBlock, center_interactions, correlation

DRAWS = 2000
RANKS = ("spearman", "interaction_spearman")
RETRIEVAL = ("ndcg_at_5", "precision_at_5", "unique_top5_sets")


class Resampling:
    def __init__(self, data, groups):
        rng = np.random.default_rng(495123)
        self.lineages = pd.Index(sorted(data.lineage_group.unique()))
        self.scaffolds = pd.Index(sorted(set(data[KEY].map(groups).fillna(data[KEY]))))
        self.lineage_weights = rng.exponential(size=(DRAWS, len(self.lineages)))
        self.scaffold_weights = rng.exponential(size=(DRAWS, len(self.scaffolds)))

    def panel(self, cells, drugs):
        l = self.lineages.get_indexer(cells)
        d = self.scaffolds.get_indexer(drugs)
        if (l < 0).any() or (d < 0).any():
            raise ValueError("Unexpected resampling cluster")
        return self.lineage_weights[:, l], self.scaffold_weights[:, d]


def weighted_average(draws, weights):
    return (weights * draws.T).sum(axis=1) / weights.sum(axis=1)


def weighted_midrank(values, weights):
    """Compute the empirical mid-distribution for each shared Bayesian cluster draw."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1]
    cumulative = np.c_[np.zeros(len(weights)), np.cumsum(weights[:, order], axis=1)]
    left = cumulative[:, starts]
    right = cumulative[:, np.r_[starts[1:], len(values)]]
    tied = np.repeat(.5*(left+right), np.diff(np.r_[starts, len(values)]), axis=1)
    centered = tied - .5*cumulative[:, -1:]
    return centered[:, np.argsort(order)]


def lineage_rank_samples(truth, prediction, eligible, lineage_weights):
    """Re-rank within each drug in each lineage draw. Residual axes are already centered."""
    out = np.zeros((len(prediction), len(eligible), DRAWS))
    for j, drug in enumerate(eligible):
        known = np.isfinite(truth[:, drug])
        y = truth[known, drug]
        w = lineage_weights[:, known]
        rank_y = weighted_midrank(y, w)
        y_norm = np.einsum("rc,rc->r", w, rank_y**2)
        previous = {}
        for a, pred in enumerate(prediction):
            score = pred[known, drug]
            order = rankdata(score).astype(np.int16).tobytes()
            if order not in previous:
                rank_x = weighted_midrank(score, w)
                x_norm = np.einsum("rc,rc->r", w, rank_x**2)
                norm = np.sqrt(np.maximum(x_norm * y_norm, 0.))
                xy = np.einsum("rc,rc->r", rank_y * rank_x, w)
                previous[order] = np.divide(xy, norm, out=np.zeros_like(norm), where=norm > 1e-12)
            out[a, j] = previous[order]
    return out


def panel_values(layout, name, prediction, interaction, profile_weights):
    matrix = layout.wide(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Scored column contains nonfinite values: {name}")
    personal = matrix if interaction is None else layout.wide(interaction)
    centered = center_interactions(personal)
    rank = correlation(layout.truth_rank, rankdata(matrix, axis=0, nan_policy="omit"))
    truth_centered = layout.residual
    joint = correlation(layout.residual_rank, rankdata(centered, axis=0, nan_policy="omit"))
    top = np.argsort(-np.where(np.isfinite(matrix), matrix, -np.inf), axis=1, kind="stable")[:, :5]
    ndcg = np.take_along_axis(layout.relevance, top, axis=1) @ layout.discount[:top.shape[1]]
    valid = np.flatnonzero(layout.cell_eligible)
    precision = (top[:, :, None] == layout.actual_top[:, None, :]).sum((1,2)) / 5
    point = {"spearman": rank, "interaction_spearman": joint}
    retrieval = {"ndcg_at_5": ndcg[valid]/layout.ideal[valid], "precision_at_5": precision[valid]}
    return matrix, centered, point, retrieval, top[valid], truth_centered


def diversity_bundle(tops, lineages, counts=None):
    sets, set_index = np.unique(np.sort(tops, axis=1), axis=0, return_inverse=True)
    distinct, lineage_index = np.unique(lineages, return_inverse=True)
    flags = np.zeros((len(distinct), len(sets)), dtype=np.float32)
    flags[lineage_index, set_index] = 1.
    if counts is None:
        counts = np.random.default_rng(961007).multinomial(len(distinct), np.full(len(distinct), 1./len(distinct)),
                                                          size=DRAWS).astype(np.float32)
    return len(sets), (counts @ flags > 0).sum(axis=1)


@dataclass
class Block:
    screen: str
    fold: int
    point: dict
    drawings: dict
    support: dict


def measure_block(frame, groups, fold, screen, cols, random, *, crossed=False, components=None):
    frame = frame.reset_index(drop=True)
    layout = ScoreBlock(frame, np.arange(len(frame)), "", fold, screen, groups, True)
    lin_w, drug_w = random.panel(layout.lineages, layout.clusters)
    drugs = np.flatnonzero(layout.drug_eligible)
    cells = np.flatnonzero(layout.cell_eligible)
    point, draws = defaultdict(dict), defaultdict(dict)
    panels, interactions, stats = [], [], []
    components = components or {}
    for arm in cols:
        extra = frame[components[arm]].to_numpy() if arm in components else None
        p, i, r, scores, top, _ = panel_values(layout, arm, frame[arm].to_numpy(), extra, lin_w)
        panels.append(p)
        interactions.append(i)
        stats.append(r)
        if len(cells):
            for name, score in scores.items():
                point[name][arm] = float(score.mean())
                draws[name, "lineage"][arm] = weighted_average(score[:,None], lin_w[:, cells])
            point["unique_top5_sets"][arm], draws["unique_top5_sets", "lineage"][arm] = diversity_bundle(
                top, layout.lineages[cells])
        for name, values in r.items():
            if len(drugs):
                score = values[drugs]
                point[name][arm] = float(score.mean())
                draws[name, "scaffold"][arm] = weighted_average(score[:,None], drug_w[:,drugs])
    if crossed and len(drugs):
        for name, y, pred in (("spearman", layout.truth, panels),
                              ("interaction_spearman", layout.residual, interactions)):
            all_values = lineage_rank_samples(y, pred, drugs, lin_w)
            for arm, values in zip(cols, all_values):
                draws[name, "lineage"][arm] = values.mean(axis=0)
                draws[name, "scaffold_x_lineage"][arm] = weighted_average(values, drug_w[:,drugs])
    return Block(screen, fold, point, draws, {
        "rows": len(frame), "parents": len(layout.drugs), "cells": len(layout.cells),
        "lineages": len(set(layout.lineages)), "eligible_drugs": len(drugs), "eligible_cells": len(cells),
    })


def measure(panels, groups, cols, *, crossed=False, components=None):
    panels = [(f, p) for f, p in panels if not p.empty]
    if not panels:
        return []
    random = Resampling(pd.concat([p[[KEY, "lineage_group"]] for _, p in panels], ignore_index=True), groups)
    return [measure_block(block, groups, fold, screen, cols, random, crossed=crossed, components=components)
            for fold, p in panels for screen, block in p.groupby("source_id", sort=True) if not block.empty]


def mean_across(values):
    by_screen = defaultdict(list)
    for screen, sample in values:
        by_screen[screen].append(sample)
    return np.mean([np.mean(v, axis=0) for v in by_screen.values()], axis=0)


def summarize(blocks, tags, metrics, pairs, *, crossed=False):
    points, intervals = [], []
    for metric in (*RANKS, *RETRIEVAL):
        entries = [b for b in blocks if metric in b.point]
        if not entries:
            continue
        for name, track in metrics.items():
            for b in entries:
                points.append(tags | track | {"metric": metric, "screen": b.screen, "fold": b.fold,
                    "value": b.point[metric][name], **b.support})
            points.append(tags | track | {"metric": metric, "screen":"macro","fold":"all",
                "value":float(mean_across((b.screen,b.point[metric][name]) for b in entries)),
                **{k:sum(b.support[k] for b in entries) for k in entries[0].support},
                "eligible_screen_folds": len(entries)})
        variants = ("lineage",) if metric in RETRIEVAL else (("scaffold","lineage","scaffold_x_lineage")
                                                             if crossed else ("scaffold",))
        for name, weights in pairs:
            delta = float(mean_across((b.screen,sum(v*b.point[metric][k] for k,v in weights.items())) for b in entries))
            for variant in variants:
                key = metric, variant
                samples = mean_across((b.screen,sum(v*b.drawings[key][k] for k,v in weights.items())) for b in entries)
                lo, hi = np.quantile(samples, [.025,.975])
                intervals.append(tags | {"contrast":name,"metric":metric,"resampling":variant,
                    "delta":delta,"lo95":float(lo),"hi95":float(hi),"draws":DRAWS,
                    "eligible_screen_folds":len(entries)})
    return points, intervals
