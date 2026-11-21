"""Fit one optional-view kernel change while keeping the I-track anchor and FiLM fixed."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.r15_lincs.estimators import GAINS, loss, parent_average, solve
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_kernel_arms import ANNOTATION, annotation_matrix, fit_optional_profiles, iter_kernel_arms
from src.r19_plan.i_kernel_audit import assay_matrix

ALPHAS = (.1, 1., 10.)
WELL_FIELDS = ["study", "inst_id", "rna_plate", "rna_well"]


@dataclass(frozen=True)
class ViewInputs:
    parents: pd.Index
    structure: np.ndarray
    phenotypes: dict[str, np.ndarray]
    assays: pd.DataFrame
    annotations: pd.DataFrame


@dataclass(frozen=True)
class FitResidual:
    rows: pd.DataFrame
    values: np.ndarray


@dataclass(frozen=True)
class QueryInputs:
    rows: pd.DataFrame
    anchor: np.ndarray
    film: np.ndarray


@dataclass(frozen=True)
class KernelChoice:
    alpha: float
    gain: float
    loss: float


@dataclass(frozen=True)
class KernelForecast:
    full: np.ndarray
    kernel_only: np.ndarray
    view_count: np.ndarray
    views: tuple[str, ...]


def row_context(inputs, fit, query):
    """Validate well axes and learn every view transformation on this fit partition."""
    parents = inputs.parents
    if not parents.is_unique or not parents.is_monotonic_increasing:
        raise ValueError("The registered canonical parent axis must be unique and sorted")
    if set(inputs.phenotypes) != {"I2", "I4"}:
        raise ValueError("The registered I-track requires the approved I2 and I4 views")
    for rows in (fit.rows, query.rows):
        required = [*WELL_FIELDS, "parent"]
        if (not len(rows) or rows[required].isna().any().any() or rows[required].eq("").any().any()
                or rows.duplicated(WELL_FIELDS[:2]).any()
                or rows.duplicated(["study", "rna_plate", "rna_well"]).any()
                or not rows.parent.isin(parents).all()):
            raise ValueError("An I-track well is missing or duplicated on its registered axes")
    train_parents = parents.isin(fit.rows.parent)
    query_parents = parents.isin(query.rows.parent)
    if (train_parents & query_parents).any():
        raise ValueError("An I-track query parent entered the fitting partition")
    if np.ndim(query.anchor) != 2 or query.anchor.shape[1] == 0:
        raise ValueError("I-track forecasts require an ordered nonempty gene axis")
    genes = query.anchor.shape[1]
    anchor = checked_array(query.anchor, (len(query.rows), genes), "query anchor")
    film = checked_array(query.film, anchor.shape, "locked FiLM contribution")
    values = checked_array(fit.values, (len(fit.rows), genes), "fit-only well residuals")
    aggregate = parent_average(fit.rows, values)
    to_query = parents[query_parents].get_indexer(query.rows.parent)
    profiles = dict(inputs.phenotypes)
    profiles["I1"] = assay_matrix(inputs.assays, parents, train_parents)
    profiles[ANNOTATION] = annotation_matrix(inputs.annotations, parents, train_parents)
    mapped = fit_optional_profiles(profiles, train_parents)
    arms = iter_kernel_arms(inputs.structure, mapped, train_parents, query_parents)
    return arms, aggregate, to_query, anchor, film


def predict_residual(arm, values, alpha):
    return (arm.query_gram @ solve(arm.train_gram, values, alpha)).astype(np.float32)


def scaled_kernel(arm, raw, gain, fallback):
    scaled = gain * raw
    if arm.name.endswith(("_mask_only", "_morphology_only")):
        return scaled
    return arm.apply_structure_fallback(scaled, fallback)


def select_kernels(inputs, fit, select, validation_truth):
    """Select ridge and gain using only the independent inner-select target array."""
    arms, aggregate, to_query, anchor, film = row_context(inputs, fit, select)
    truth = checked_array(validation_truth, anchor.shape, "independent inner-select truth")
    if not truth.any():
        raise ValueError("The registered selection loss is undefined for all-zero truth")
    settings = {}
    base = None
    for arm in arms:
        candidates = {}
        for alpha in ALPHAS:
            raw = predict_residual(arm, aggregate, alpha)
            for gain in GAINS:
                scaled = scaled_kernel(arm, raw, gain, base) if base is not None else gain * raw
                candidates[alpha, gain] = loss(select.rows, truth, anchor + film + scaled[to_query])
        if not np.isfinite(list(candidates.values())).all():
            raise ValueError(f"Nonfinite I-track validation objective: {arm.name}")
        alpha, gain = min(candidates, key=lambda key: (round(candidates[key], 9), key[1], key[0]))
        settings[arm.name] = KernelChoice(alpha, gain, candidates[alpha, gain])
        if arm.name == "I0":
            base = gain * predict_residual(arm, aggregate, alpha)
    return settings


def refit_kernels(inputs, fit, query, settings):
    """Recompute view maps on outer-fit only, with no held RNA outcomes in this call."""
    arms, aggregate, to_query, anchor, film = row_context(inputs, fit, query)
    forecasts = {}
    base = None
    for arm in arms:
        choice = settings[arm.name]
        if choice.alpha not in ALPHAS or choice.gain not in GAINS or not np.isfinite(choice.loss):
            raise ValueError(f"An I-track choice is outside the registered grid: {arm.name}")
        raw = predict_residual(arm, aggregate, choice.alpha)
        scaled = scaled_kernel(arm, raw, choice.gain, base) if base is not None else choice.gain * raw
        if arm.name == "I0":
            base = scaled
        kernel_only = anchor + scaled[to_query]
        forecasts[arm.name] = KernelForecast(kernel_only + film, kernel_only,
                                             arm.query_view_count[to_query], arm.views)
    if set(forecasts) != set(settings):
        raise ValueError("Inner and outer I-track arm registries differ")
    return forecasts
