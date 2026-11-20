"""Shared train-only interaction Ridge and restored R6-style baselines."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA

from .dataset import CONTROL, ENTITY, GROUP, KEY, LINE, POOL, group_weights, weights_for


def keys(rows, cols):
    if not cols:
        return [()] * len(rows)
    return list(rows[list(cols)].itertuples(index=False, name=None))


class PoolMean:
    """Pool, line and global group-balanced means with whole-group exclusion."""

    def __init__(self, rows, target):
        self.stats = {}
        for columns in (POOL, [LINE], []):
            cluster = keys(rows, (*columns, GROUP))
            units = rows.assign(_cluster=cluster)
            hierarchy = ("_cluster", ENTITY, LINE, KEY[0]) if LINE not in columns else ("_cluster", ENTITY, KEY[0])
            weight = weights_for(units, hierarchy)
            sums, totals = {}, {}
            for i, (key, group) in enumerate(zip(keys(rows, columns), rows[GROUP])):
                constituent = key + (group,)
                if constituent not in sums:
                    sums[constituent] = np.zeros(target.shape[1])
                sums[constituent] += weight[i] * target[i]
            for constituent, vals in sums.items():
                key = constituent[:-1]
                if key not in totals:
                    totals[key] = [0, np.zeros(target.shape[1])]
                totals[key][0] += 1
                totals[key][1] += vals
            self.stats[tuple(columns)] = sums, totals
        self.n_genes = target.shape[1]

    def predict(self, rows, kappa, leave_group_out=False):
        out = np.empty((len(rows), self.n_genes), dtype=np.float32)
        support = np.zeros(len(rows), dtype=np.int64)
        levels = []
        for i, (pool, group) in enumerate(zip(keys(rows, POOL), rows[GROUP])):
            means = []
            for columns, key in ((POOL, pool), ([LINE], (pool[1],)), ([], ())):
                sums, totals = self.stats[tuple(columns)]
                n, total = totals.get(key, (0, 0))
                old = sums.get(key + (group,)) if leave_group_out else None
                if old is not None:
                    total = total - old
                    n -= 1
                means.append((n, total / n if n else None))
            available = [index for index, (n, _) in enumerate(means) if n]
            if not available:
                raise ValueError("Leave-group-out offset requires other training groups")
            line_index = next((idx for idx in available if idx > 0), available[0])
            line = means[line_index][1]
            count, pool_mean = means[0]
            support[i] = count
            if count and np.isfinite(kappa):
                fraction = count / (count + kappa)
                out[i] = fraction * pool_mean + (1 - fraction) * line
                levels.append("pool")
            else:
                out[i] = line
                levels.append(("pool", "line", "global")[line_index])
        return out, levels, support


class Scaling:
    def __init__(self, values):
        self.keep = np.flatnonzero(values.std(axis=0) > 1e-9)
        self.mean = values[:, self.keep].mean(axis=0)
        self.scale = values[:, self.keep].std(axis=0)

    def transform(self, values):
        return (values[:, self.keep] - self.mean) / self.scale


class Interaction:
    """Each fit gets distinct PCA/scalers fitted only on its unique inputs."""

    def __init__(self, block, arm, drug_dim, control_dim):
        self.arm = arm
        # Treat each entity and each physical control×line as a unit.
        unique_d = ~block.rows.duplicated(ENTITY)
        unique_c = ~block.rows.duplicated([CONTROL, LINE])
        self.drug = self._fit(block.chemical[arm][unique_d], drug_dim)
        self.control = self._fit(block.control[unique_c], control_dim)
        self.design = Scaling(self.raw(block))

    @staticmethod
    def _fit(values, maximum):
        values = values.astype(np.float64)
        n = min(maximum, len(values) - 1, values.shape[1])
        if n < 1:
            raise ValueError("PCA requires two nonduplicated input units")
        pca = PCA(n_components=n, svd_solver="randomized", random_state=42).fit(values)
        z = pca.transform(values)
        # The scaler drops zero-rank components and records the realized dimensions.
        return pca, Scaling(z)

    def blocks(self, block, drug=None):
        drug = block.chemical[self.arm] if drug is None else drug
        dp, ds = self.drug
        cp, cs = self.control
        return ds.transform(dp.transform(drug)), cs.transform(cp.transform(block.control))

    def raw(self, block, drug=None):
        drug, control = self.blocks(block, drug)
        return np.concatenate((drug, control, (drug[:, :, None] * control[:, None, :]).reshape(len(drug), -1)), axis=1)

    def transform(self, block, drug=None):
        return self.design.transform(self.raw(block, drug))


class FullInput:
    """Unreduced ECFP4 and mean control input retained for the B2 models."""

    def __init__(self, block):
        self.drug = Scaling(block.chemical["E"][~block.rows.duplicated(ENTITY)].astype(np.float64))
        self.control = Scaling(block.control[~block.rows.duplicated([CONTROL, LINE])])
        # Keep full ECFP bits; sparse occurrence does not increase their prior weight.
        self.drug.scale[:] = 1.0

    def transform(self, block):
        return np.concatenate((self.drug.transform(block.chemical["E"]), self.control.transform(block.control)), axis=1)


class WeightedRidge:
    def __init__(self, train, target, rows):
        weight = group_weights(rows)
        weight *= len(weight) / weight.sum()
        self.xmean = np.average(train, axis=0, weights=weight)
        self.ymean = np.average(target, axis=0, weights=weight)
        x = train - self.xmean
        xtw = x.T * weight
        self.gram = xtw @ x
        self.cross = xtw @ (target - self.ymean)

    def coefficients(self, alpha):
        regularized = self.gram.copy()
        regularized.ravel()[::len(regularized) + 1] += alpha
        return cho_solve(cho_factor(regularized, lower=True, check_finite=False), self.cross, check_finite=False)

    def predict(self, query, coef):
        return ((query - self.xmean) @ coef + self.ymean).astype(np.float32)
