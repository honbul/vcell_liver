"""Auxiliary-dose transfer fitted exclusively on overlapping training compounds and lines."""

from functools import cached_property
from itertools import product

import numpy as np
import pandas as pd

from src.r11_chemical_probe.dataset import GROUP, KEY, LINE, group_weights
from src.r11_chemical_probe.models import PoolMean

from .transfer import TransferFit, mean_units, reproducible_gene_basis, weighted_consensus

DOSE_VIEWS = {"low": (0.05,), "high": (5.0,), "both": (0.05, 5.0)}


def candidates():
    """Optional transfer variants when the immutable fit-side dose assets exist."""
    specs = []
    for views, alpha in product(DOSE_VIEWS, (3, 30, 300)):
        shared = dict(chemistry="ecfp", alpha=alpha, line_basis="none", line_rank=0,
                      line_gain=0.0, plate_adjust=False)
        specs.append(dict(name=f"tr_dose_{views}_basis_a{alpha}", gene_basis=f"dose_{views}",
                          pooling="mean", **shared))
        for weight in (0.25, 1.0):
            specs.append(dict(name=f"tr_dose_{views}_blend{weight}_a{alpha}", gene_basis="pca32",
                              pooling=f"dose_{views}_{weight}", **shared))
    return specs


def leave_compound_out_scale(auxiliary, primary):
    """Fit a regularized direction-preserving dose scale from other compounds."""
    dot = (auxiliary * primary).mean(axis=1, dtype=np.float64)
    squared = np.square(auxiliary).mean(axis=1, dtype=np.float64)
    support = squared.sum() - squared
    prior = 0.1 * support / max(len(squared) - 1, 1)
    gain = np.clip((dot.sum() - dot) / np.maximum(support + prior, 1e-8), 0.0, 3.0)
    return auxiliary * gain[:, None].astype(np.float32)


class DoseTransferFit(TransferFit):
    def __init__(self, fit):
        super().__init__(fit)
        self.dose_summaries = {}
        self.dose_projection = {}

    @cached_property
    def dose_block(self):
        from src.r12_multiaxis.dose import DoseCache

        source = self.fit.source
        if not hasattr(source, "dose_cache"):
            source.dose_cache = DoseCache(source)
        result = source.dose_cache.block(self.fit)
        if not set(result.rows[GROUP]) <= set(self.groups):
            raise ValueError("Transfer dose rows include an out-of-training compound")
        if set(result.rows[KEY[0]]) & set(self.fit.query.rows[KEY[0]]):
            raise ValueError("Transfer dose rows overlap a query well")
        return result

    def dose_summary(self, dose):
        if dose not in self.dose_summaries:
            block = self.dose_block
            sub = block.take(block.rows.dose_uM.eq(dose).to_numpy() & block.rows[LINE].isin(self.lines).to_numpy())
            rows = sub.rows
            if rows[GROUP].nunique() < 3:
                raise ValueError(f"Insufficient fit compound overlap at auxiliary dose {dose}")
            delta = sub.delta
            mean = PoolMean(rows, delta).predict(rows, self.fit.cfg["offset_shrinkage"], leave_group_out=True)[0]
            group = pd.Index(self.groups).get_indexer(rows[GROUP])
            line = pd.Index(self.lines).get_indexer(rows[LINE])
            n_groups, n_lines = len(self.groups), len(self.lines)
            agg, count = mean_units(delta - mean, group * n_lines + line, n_groups * n_lines, group_weights(rows))
            values = agg.reshape(n_groups, n_lines, -1)
            support = count.reshape(n_groups, n_lines) > 0
            by_line = values.sum(axis=0) / np.maximum(support.sum(axis=0)[:, None], 1)
            values = ((values - by_line[None]) * support[:, :, None]).astype(np.float32)
            # Give both doses the same eligible line set for each compound.
            paired = (self.support & support).astype(np.float32)
            valid = paired.sum(axis=1) > 0
            summary = weighted_consensus(values, paired)
            primary = weighted_consensus(self.values, paired)
            self.dose_summaries[dose] = (valid, summary[valid], primary[valid])
        return self.dose_summaries[dose]

    def project_dose(self, dose):
        if dose not in self.dose_projection:
            valid, summary, primary = self.dose_summary(dose)
            self.dose_projection[dose] = valid, leave_compound_out_scale(summary, primary)
        return self.dose_projection[dose]

    def consensus(self, method):
        if not method.startswith("dose_"):
            return super().consensus(method)
        _, views, weight = method.split("_")
        weight = float(weight)
        primary = self.ordinary_consensus.copy()
        total = np.ones(len(self.groups), dtype=np.float32)
        for dose in DOSE_VIEWS[views]:
            valid, projection = self.project_dose(dose)
            primary[valid] += weight * projection
            total[valid] += weight
        return primary / total[:, None]

    def genes(self, basis):
        if not basis.startswith("dose_"):
            return super().genes(basis)
        views = DOSE_VIEWS[basis.removeprefix("dose_")]
        paired = [self.dose_summary(dose)[1:] for dose in views]
        left = np.concatenate([x[1] - x[1].mean(axis=0) for x in paired])
        right = np.concatenate([(x[0] - x[0].mean(axis=0)) /
                                max(float(np.sqrt(np.mean(np.square(x[0])))), 1e-8) for x in paired])
        return reproducible_gene_basis(left, right, min(32, *self.ordinary_consensus.shape))


def predict(fit, spec):
    if not hasattr(fit, "_r13_dose"):
        fit._r13_dose = DoseTransferFit(fit)
    return fit._r13_dose.predict(spec)
