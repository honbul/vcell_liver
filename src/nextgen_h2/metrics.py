"""Native-screen cell ranking, within-profile retrieval and paired cluster effects."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .assets import KEY

DRUG = ["spearman", "pearson", "interaction_spearman", "interaction_pearson"]
CELL = ["ndcg_at_5", "precision_at_5"]
MIN_CELLS = 5


def correlation(y, p):
    valid = np.isfinite(y)
    count = valid.sum(axis=0).clip(min=1)
    centered = [np.where(valid, a - np.nansum(a, axis=0) / count, 0.) for a in (y, p)]
    norms = np.linalg.norm(centered[0], axis=0) * np.linalg.norm(centered[1], axis=0)
    cov = (centered[0] * centered[1]).sum(axis=0)
    return np.divide(cov, norms, out=np.zeros_like(cov), where=norms > 1e-12)


def center_interactions(matrix):
    present = np.isfinite(matrix)
    observed = np.nan_to_num(matrix)
    nd, nc = present.sum(axis=0), present.sum(axis=1)
    valid_d, valid_c = nd > 0, nc > 0
    drug = np.zeros(len(nd))
    result = np.zeros_like(matrix)
    for _ in range(250):
        cell = np.divide((observed - present * drug).sum(axis=1), nc, where=valid_c, out=np.zeros(len(nc)))
        change = np.divide((observed - present * cell[:, None]).sum(axis=0), nd,
                           where=valid_d, out=np.zeros(len(nd)))
        error = float(np.max(np.abs(change - drug)))
        drug = change
        if error < 1e-10:
            break
    result[present] = (observed - cell[:, None] - drug)[present]
    result[~present] = np.nan
    return result


class ScoreBlock:
    def __init__(self, data, index, scope, fold, screen, clusters, interaction):
        self.data = data.iloc[index]
        self.index = index
        self.scope, self.fold, self.screen, self.interaction = scope, fold, screen, interaction
        self.cells = pd.Index(sorted(self.data.cell_id.unique()))
        lineage = self.data[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id").lineage_group
        self.lineages = lineage.loc[self.cells].to_numpy()
        self.drugs = pd.Index(sorted(self.data[KEY].unique()))
        self.clusters = clusters.reindex(self.drugs).fillna(pd.Series(self.drugs, index=self.drugs)).to_numpy()
        self.position = self.cells.get_indexer(self.data.cell_id), self.drugs.get_indexer(self.data[KEY])
        if self.data.duplicated(["cell_id", KEY]).any():
            raise ValueError("One native endpoint per screen-cell-parent is required")
        self.truth = self.wide(self.data.truth.to_numpy())
        self.truth_rank = rankdata(self.truth, axis=0, nan_policy="omit")
        mask = np.isfinite(self.truth)
        lower = np.min(np.where(mask, self.truth, np.inf), axis=0)
        upper = np.max(np.where(mask, self.truth, -np.inf), axis=0)
        self.drug_eligible = (mask.sum(axis=0) >= MIN_CELLS) & (upper - lower > 1e-12)
        ordered = np.argsort(-np.where(mask, self.truth, -np.inf), axis=1, kind="stable")
        self.actual_top = ordered[:, :5]
        best = np.max(np.where(mask, self.truth, -np.inf), axis=1)
        worst = np.min(np.where(mask, self.truth, np.inf), axis=1)
        self.cell_eligible = (mask.sum(axis=1) >= 5) & (best - worst > 1e-12)
        self.relevance = np.where(mask, (self.truth - worst[:, None]) / np.maximum(best - worst, 1e-12)[:, None], 0.)
        self.discount = 1 / np.log2(np.arange(2, 7))
        self.ideal = np.take_along_axis(self.relevance, self.actual_top, axis=1) @ self.discount[:self.actual_top.shape[1]]
        if interaction:
            self.residual = center_interactions(self.truth)
            self.residual_rank = rankdata(self.residual, axis=0, nan_policy="omit")

    def wide(self, values):
        a = np.full((len(self.cells), len(self.drugs)), np.nan)
        a[self.position] = values
        return a

    def score(self, prediction, method, support):
        p = self.wide(prediction[self.index])
        if not np.isfinite(prediction[self.index]).all():
            raise ValueError("Forecast must share the exact finite observation roster")
        methods = {"spearman": correlation(self.truth_rank, rankdata(p, axis=0, nan_policy="omit")),
                   "pearson": correlation(self.truth, p)}
        if self.interaction:
            residual = center_interactions(p)
            methods.update(interaction_spearman=correlation(self.residual_rank, rankdata(residual, axis=0, nan_policy="omit")),
                           interaction_pearson=correlation(self.residual, residual))
        drugs = np.flatnonzero(self.drug_eligible)
        cells = np.flatnonzero(self.cell_eligible)
        top = np.argsort(-np.where(np.isfinite(p), p, -np.inf), axis=1, kind="stable")[:, :5]
        ndcg = np.take_along_axis(self.relevance, top, axis=1) @ self.discount[:top.shape[1]]
        equal = (top[:, :, None] == self.actual_top[:, None, :]).sum(axis=(1, 2)) / 5
        names = {"scope": self.scope, "fold": self.fold, "screen": self.screen, "arm": method, "support": support}
        ds = pd.DataFrame(names | {"unit": self.drugs[drugs], "cluster": self.clusters[drugs],
                                  "n": np.isfinite(self.truth[:, drugs]).sum(axis=0)}
                          | {k: v[drugs] for k, v in methods.items()})
        cs = pd.DataFrame(names | {"unit": self.cells[cells], "cluster": self.lineages[cells],
                                  "n": np.isfinite(self.truth[cells]).sum(axis=1),
                                  "ndcg_at_5": ndcg[cells] / self.ideal[cells], "precision_at_5": equal[cells]})
        return ds, cs


class Evaluator:
    def __init__(self, rows, clusters, fold, *, scope="DRP", interaction=False):
        self.rows = rows.reset_index(drop=True)
        self.clusters = clusters
        self.fold, self.scope, self.interaction = fold, scope, interaction
        self.layouts = {}

    def score(self, arm, predicted, selector=None, support="all"):
        if support not in self.layouts:
            selection = np.ones(len(self.rows), dtype=bool) if selector is None else np.asarray(selector)
            blocks = []
            for source, indices in self.rows.loc[selection].groupby("source_id").groups.items():
                blocks.append(ScoreBlock(self.rows, np.asarray(indices), self.scope, self.fold, source,
                                         self.clusters, self.interaction))
            self.layouts[support] = blocks
        return [block.score(predicted, arm, support) for block in self.layouts[support]]


def macro(frame, metric):
    return float(frame.groupby(["screen", "fold"])[metric].mean().groupby(level=0).mean().mean())


def contrast(frame, candidate, reference, metric, *, draws=2000, seed=94071):
    axis = ["scope", "fold", "screen", "support", "unit", "cluster"]
    c = frame[frame.arm.eq(candidate)].set_index(axis)[metric].sort_index()
    r = frame[frame.arm.eq(reference)].set_index(axis)[metric].sort_index()
    if not c.index.equals(r.index):
        raise ValueError(f"Paired support mismatch for {candidate} versus {reference}")
    diff = (c - r).dropna().rename("delta").reset_index()
    if not len(diff):
        return np.nan, np.nan, np.nan, 0
    clusters = pd.Index(sorted(diff.cluster.unique()))
    rng = np.random.default_rng(seed)
    weights = rng.exponential(1., size=(draws, len(clusters)))
    intervals = {}
    for (screen, _), rows in diff.groupby(["screen", "fold"]):
        # Take the weighted mean per fold and screen. Share draw weights across screens.
        count = rows.groupby("cluster").size().reindex(clusters, fill_value=0).to_numpy()
        total = rows.groupby("cluster").delta.sum().reindex(clusters, fill_value=0).to_numpy()
        sample = (weights @ total) / (weights @ count)
        intervals.setdefault(screen, []).append(sample)
    pooled = np.mean([np.mean(parts, axis=0) for parts in intervals.values()], axis=0)
    lo, hi = np.quantile(pooled, [.025, .975])
    return macro(diff, "delta"), float(lo), float(hi), len(clusters)


def stack(pairs):
    return (pd.concat([x[0] for x in pairs], ignore_index=True),
            pd.concat([x[1] for x in pairs], ignore_index=True))
