"""Input-preserving output heads and fit-only losses for the S/L ablations."""

from functools import lru_cache
from math import expm1, log, sqrt
from zlib import crc32

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.goal_tracks.t2_upstream import SEED
from src.r11_chemical_probe.dataset import CONTROL, GROUP, LINE, POOL
from src.r12_multiaxis.data import Design
from src.r12_multiaxis.network import fit_network

WEIGHTS = (.01, .1, 1.)
HUBER_SCALES = (.5, 1., 2.)
RANKS = (32, 64, 128)
CONTROL_POLICIES = ("drug_constant", "drug_shuffle", "context_constant", "context_shuffle")


class DirectionMagnitude(nn.Module):
    def __init__(self, vector):
        super().__init__()
        self.vector = vector
        self.magnitude = nn.Linear(vector.in_features, 1)
        nn.init.zeros_(self.magnitude.weight)
        nn.init.constant_(self.magnitude.bias, log(expm1(.1 * sqrt(vector.out_features))))

    def forward(self, state):
        direction = self.vector(state)
        length = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        magnitude = nn.functional.softplus(self.magnitude(state))
        return magnitude * direction / (length + 1e-6)


def separate_magnitude(head):
    if not isinstance(head.decoder, nn.Linear) or hasattr(head, "vectors"):
        raise ValueError("S2 requires the full-gene vector decoder")
    head.decoder = DirectionMagnitude(head.decoder)
    return head


def exact_pools(rows, measured_by_id):
    if not measured_by_id.index.is_unique:
        raise ValueError("L1 measured dose must be uniquely indexed by the physical LINCS instance")
    measured = rows.inst_id.map(measured_by_id)
    numeric = pd.to_numeric(measured, errors="coerce")
    if not np.isfinite(numeric).all() or numeric.le(0).any():
        raise ValueError("L1 needs exact positive measured µM support on the common fit axis")
    return rows.assign(dose_uM=measured.astype(str))


def shuffle_rows(values, rows, keys, scope):
    frame = rows[keys].reset_index(drop=True)
    unique = frame.drop_duplicates().sort_values(keys)
    first = unique.index.to_numpy()
    axis = pd.MultiIndex.from_frame(unique)
    expanded = axis.get_indexer(pd.MultiIndex.from_frame(frame))
    if (expanded < 0).any() or not np.array_equal(values[first][expanded], values):
        raise ValueError("A shuffled feature changed within its physical unit or parent")
    destination = np.arange(len(unique))
    strata = unique.groupby(LINE, sort=True).indices if LINE in keys else {"all": destination}
    for line, idx in strata.items():
        rng = np.random.default_rng(np.random.SeedSequence([1501, scope, crc32(line.encode())]))
        destination[idx] = rng.permutation(idx)
    return values[first[destination[expanded]]]


def feature_control(design, fit, policy):
    if policy is None:
        return design
    if policy not in CONTROL_POLICIES or len(design.train) != 2 or len(design.query) != 2:
        raise ValueError("The S/L feature control requires the fixed FiLM feature branches")
    branch = 0 if policy.startswith("drug") else 1
    keys = [GROUP] if branch == 0 else [CONTROL, LINE]

    def replace(values, rows, scope):
        result = list(values)
        result[branch] = (np.zeros_like(result[branch]) if policy.endswith("constant")
                          else shuffle_rows(result[branch], rows, keys, scope))
        return tuple(result)

    return Design(replace(design.train, fit.train.rows, 0), replace(design.query, fit.query.rows, 1),
                  response_head=design.response_head)


def matched_contrast(weight):
    """Squared contrast on parent-mean errors in matched multi-parent pools."""
    if weight not in WEIGHTS:
        raise ValueError("L1 weight must be on the registered grid")

    def prepare(rows):
        pool = rows.groupby(POOL, sort=False).ngroup().to_numpy(dtype=np.int64)
        parent, labels = pd.factorize(pd.MultiIndex.from_arrays([pool, rows[GROUP]]), sort=False)
        parent_pool = labels.get_level_values(0).to_numpy(dtype=np.int64)
        per_group = np.bincount(parent)
        per_pool = np.bincount(parent_pool)
        # The mean squared pair difference among k parent means equals 2k/(k-1) times their variance.
        correction = np.divide(2 * per_pool, np.maximum(1, per_pool - 1)) * (per_pool >= 2)

        @lru_cache(maxsize=2)
        def device_values(device, dtype):
            return (torch.as_tensor(parent, device=device), torch.as_tensor(parent_pool, device=device),
                    *(torch.as_tensor(x, device=device, dtype=dtype) for x in (per_group, per_pool, correction)))

        def objective(prediction, target):
            error = prediction - target
            group, grouping, ng, npool, pair_correction = device_values(error.device, error.dtype)
            group_mean = error.new_zeros((len(per_group), error.shape[1])).index_add(0, group, error) / ng[:, None]
            pool_mean = (error.new_zeros((len(per_pool), error.shape[1])).index_add(0, grouping, group_mean)
                         / npool[:, None])
            penalties = (group_mean - pool_mean[grouping]).square().mean(dim=1) * pair_correction[grouping]
            return error.square().mean(dim=1) + weight * penalties[group]

        return objective

    return prepare


def directional(residual, weight):
    """Use fit residual norms only to stabilize and weight the directional term."""
    residual = np.asarray(residual)
    if weight not in WEIGHTS or residual.ndim != 2 or not np.isfinite(residual).all():
        raise ValueError("L2 requires finite normalized fit residuals and a registered weight")
    magnitude = np.linalg.norm(residual, axis=1)
    positive = magnitude[magnitude > 0]
    floor = max(1e-6, float(np.quantile(positive, .1))) if len(positive) else 1e-6
    epsilon = max(1e-6, .1 * floor)

    def prepare(_rows):
        def objective(prediction, target):
            square = (prediction - target).square().mean(dim=1)
            reliable = (torch.linalg.vector_norm(target, dim=1) / floor).clamp(max=1.)
            norm_pred = (prediction.square().sum(dim=1) + epsilon ** 2).sqrt()
            norm_target = (target.square().sum(dim=1) + epsilon ** 2).sqrt()
            cosine = (prediction * target).sum(dim=1) / (norm_pred * norm_target)
            return square + weight * reliable * (1. - cosine.clamp(-1., 1.))

        return objective

    return prepare, floor


def huber(scale):
    """Match the MSE curvature near zero, using multiples of the fit residual scale."""
    if scale not in HUBER_SCALES:
        raise ValueError("L3 scale must be on the registered grid")

    def prepare(_rows):
        def objective(prediction, target):
            error = (prediction - target).abs()
            return torch.where(error <= scale, error.square(), scale * (2 * error - scale)).mean(dim=1)

        return objective

    return prepare


def fit_variant(fit, strategy, value, lr, epochs, training_pools, *, device="cpu", control=None):
    """Fit one predeclared S/L setting and capture gene arrays without target outcomes."""
    if (strategy not in {"S0", "S1", "S2", "S3", "L0", "L1", "L2", "L3"}
            or lr not in R12_LRS or not epochs or not set(epochs) <= set(R12_EPOCHS)
            or fit.query.treated is not None):
        raise ValueError("S/L requires a registered fit-only strategy, rate, and epoch grid")
    rank = value if strategy == "S3" else None
    if strategy == "S3" and (rank not in RANKS or rank > min(len(fit.panel), fit.train.rows[GROUP].nunique() - 1)):
        raise ValueError("S3 rank exceeds the registered grid or fit-parent support")
    if strategy != "S3" and not strategy.startswith("L") and value is not None:
        raise ValueError("Only S3 has an output-rank setting")
    direction_floor = None
    loss_factory = None
    if strategy == "L1":
        loss_factory = matched_contrast(value)
    elif strategy == "L2":
        loss_factory, direction_floor = directional(fit.residual / fit.scale, value)
    elif strategy == "L3":
        loss_factory = huber(value)
    elif value is not None and rank is None:
        raise ValueError("Unexpected setting for a registered MSE control")
    design = feature_control(fit.inputs("pca", rank=rank), fit, control)
    arch = "concat" if strategy == "S1" else "film"
    _, runtime, curve, _ = fit_network(
        fit, design, arch, "L0", lr, SEED, device, max(epochs), patience=False,
        head_transform=separate_magnitude if strategy == "S2" else None, loss_factory=loss_factory,
        training_pools=training_pools, inference_epochs=epochs,
    )
    return {row["epoch"]: row["prediction"] for row in curve if "prediction" in row}, dict(
        runtime, direction_floor=direction_floor,
    )
