"""F1 ablation data modes on top of the unchanged common TrainingData (D0/D1/D2 inherited)."""
import numpy as np

from common_train_data import TrainingData

MODES = ('D0', 'D1', 'D2', 'Dmean', 'Dcap')
CAP_REPEATS = 2


def cap_allocation(sizes, total, repeats=CAP_REPEATS):
    """Largest cap c with repeats*sum(min(n_i, c)) <= total, and per-intervention exposures."""
    sizes = np.asarray(sizes, dtype=np.int64)
    if repeats * sizes.sum() < total:
        raise ValueError('Repeat budget cannot fill the epoch')
    cap = max(c for c in range(1, int(sizes.max()) + 1)
              if repeats * np.minimum(sizes, c).sum() <= total)
    exposure = repeats * np.minimum(sizes, cap)
    capped = np.flatnonzero(sizes > cap)
    remainder = int(total - exposure.sum())
    if not 0 <= remainder < repeats * max(len(capped), 1):
        raise AssertionError('Unexpected cap remainder')
    return cap, exposure, capped, remainder


class AblationData(TrainingData):
    def __init__(self, path):
        super().__init__(path)
        # Dense mean over the full train control pool; constant input for Dmean.
        self.control_mean = np.asarray(self.control.mean(axis=0), dtype=np.float64).ravel().astype(
            np.float32)

    def schedule(self, mode, seed, epoch):
        if mode not in MODES:
            raise ValueError(mode)
        if mode in ('D0', 'D1', 'D2'):
            return super().schedule(mode, seed, epoch)
        if mode == 'Dmean':
            # Identical row order and control indices to D0; only the input changes in batch().
            return super().schedule('D0', seed, epoch)
        n = len(self.train_ids)
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2515]))
        conds = sorted(self.train_rows)
        sizes = np.array([len(self.train_rows[c]) for c in conds])
        cap, exposure, capped, remainder = cap_allocation(sizes, n)
        order = capped[rng.permutation(len(capped))]
        q, r = divmod(remainder, len(order)) if len(order) else (0, 0)
        exposure = exposure.copy()
        exposure[order] += q
        exposure[order[:r]] += 1
        parts = []
        for cond, count in zip(conds, exposure):
            rows = self.train_rows[cond]
            # Cycling one fresh permutation = consecutive non-duplicating passes.
            parts.append(np.resize(rng.permutation(rows), int(count)))
        rows = np.concatenate(parts)
        rng.shuffle(rows)
        assert len(rows) == n
        return rows, self.paired

    def batch(self, rows, pairs, panel=None, mode='D0'):
        x, y, targets = super().batch(rows, pairs, panel)
        if mode == 'Dmean':
            mean = self.control_mean if panel is None else self.control_mean[panel]
            x = np.broadcast_to(mean, x.shape).copy()
        return x, y, targets


def schedule_stats(data, mode, seed, epoch):
    rows, pairs = data.schedule(mode, seed, epoch)
    names = data.train_names[rows]
    exposures = np.array([np.sum(names == c) for c in sorted(data.train_rows)])
    repeats = np.unique(rows, return_counts=True)[1]
    out = dict(mode=mode, epoch=epoch, examples=int(len(rows)), updates=int(np.ceil(len(rows)/32)),
               distinct_cells=int(len(repeats)), max_cell_repeats=int(repeats.max()),
               cells_seen_twice=int(np.sum(repeats == 2)), cells_seen_3plus=int(np.sum(repeats >= 3)),
               intervention_exposure_min=int(exposures.min()),
               intervention_exposure_median=float(np.median(exposures)),
               intervention_exposure_max=int(exposures.max()),
               changed_pairs=float(np.mean(pairs != data.paired)))
    if mode == 'Dcap':
        sizes = [len(data.train_rows[c]) for c in sorted(data.train_rows)]
        cap, _, capped, remainder = cap_allocation(sizes, len(data.train_ids))
        out.update(cap=int(cap), capped_interventions=int(len(capped)), remainder=remainder)
    return out
