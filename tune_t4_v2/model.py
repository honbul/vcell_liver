"""Persistent forecasting components for the frozen temporal model."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from sklearn.linear_model import Ridge

from .data import parent_weights
from .linear import conditions
from .transfer import centered_prior


@dataclass
class Adapter:
    center: np.ndarray
    x_mean: np.ndarray
    basis: np.ndarray | None = None
    model: object = None

    def forecast(self, baseline, rows, x):
        if self.basis is None:
            return baseline+self.center
        design = np.c_[(x-self.x_mean)@self.basis.T, conditions(rows, x, False)[:, :2]]
        return baseline+self.model.predict(design)


def fit_adapter(pan, train, rank, *, device):
    residuals = train.y-centered_prior(pan, train.x, train.rows, train.x, device=device)
    weights = parent_weights(train.rows)
    y_mean = np.average(residuals, axis=0, weights=weights)
    x_mean = train.x.mean(axis=0)
    if not rank:
        return Adapter(y_mean, x_mean)
    _, _, rotation = svd(train.x-x_mean, full_matrices=False, check_finite=False)
    basis = rotation[:min(rank, len(rotation)-1)]
    design = np.c_[(train.x-x_mean)@basis.T, conditions(train.rows, train.x, False)[:, :2]]
    estimator = Ridge(alpha=45.).fit(design, residuals, sample_weight=weights)
    return Adapter(y_mean, x_mean, basis, estimator)


@dataclass
class TemporalHead:
    policy: str
    linear: object
    nonlinear: object
    nonlinear_weight: float
    pan_input_mean: np.ndarray
    adapter: Adapter | None
    pan_weight: float
    train_scaffolds: set
    genes: np.ndarray
    seed: int

    def predict(self, rows, x, pan=None, *, device="cpu"):
        output = self.linear.predict(x, rows)
        if self.nonlinear is not None:
            output += self.nonlinear_weight * self.nonlinear.predict(rows, x, seed=self.seed+1, device=device)
        if self.pan_weight:
            if pan is None or not np.array_equal(pan.genes, self.genes):
                raise ValueError("The release pan checkpoint or gene axis is missing")
            prior = centered_prior(pan, self.pan_input_mean[None], rows, x, device=device)
            if self.adapter is not None:
                prior = self.adapter.forecast(prior, rows, x)
            output = (1.-self.pan_weight)*output + self.pan_weight*prior
        return output
