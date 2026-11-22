"""Source-only scalar and vehicle-based diagonal shift calibration for T1–T3."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.r19_plan.dev_selection import PhysicalControls, ResponseAxis, checked_array, physical_axis

SCALAR_MAX = 2.
TARGET_MIN_WELLS = 12
TARGET_MIN_PLATES = 3


class ControlSupportError(ValueError):
    """The registered physical fit controls cannot support a variance estimate."""


@dataclass(frozen=True)
class EffectDeltas:
    """Whole predicted or observed effects in a registered delta coordinate system."""

    axis: ResponseAxis
    coordinate: str
    values: np.ndarray


@dataclass(frozen=True)
class RegisteredControls:
    """Control expression in an independently matched assay and aggregation contract."""

    physical: PhysicalControls
    fit_wells: tuple[tuple[str, str], ...]
    scale_id: str


@dataclass(frozen=True)
class ScalarCalibration:
    genes: tuple[int, ...]
    coordinate: str
    gain: float


@dataclass(frozen=True)
class DiagonalCalibration:
    genes: tuple[int, ...]
    scale_id: str
    factors: np.ndarray


@dataclass(frozen=True)
class SourceHoldout:
    fit: np.ndarray
    held: np.ndarray


def checked_deltas(data, name):
    axis = data.axis
    if (not data.coordinate.strip() or not axis.genes or not axis.rows
            or len(set(axis.genes)) != len(axis.genes) or len(set(axis.rows)) != len(axis.rows)):
        raise ValueError("Effects must have a complete ordered physical row and gene axis")
    return checked_array(data.values, (len(axis.rows), len(axis.genes)), name)


def fit_source_scalar(source_oof, source_outcomes):
    """Fit clipped pooled energy least squares on source-only OOF effects.

    This is the ratio of two sums across rows and genes. It is not the
    minimizer of parent-macro RMSE, even if both use the same effect deltas.
    """
    if (source_oof.axis != source_outcomes.axis
            or source_oof.coordinate != source_outcomes.coordinate):
        raise ValueError("Source OOF predictions and outcomes differ in row, gene or coordinate order")
    predicted = checked_deltas(source_oof, "source OOF predictions").astype(np.float64)
    observed = checked_deltas(source_outcomes, "source OOF outcomes").astype(np.float64)
    denominator = np.einsum("ij,ij->", predicted, predicted)
    numerator = np.einsum("ij,ij->", predicted, observed)
    if not np.isfinite([denominator, numerator]).all():
        raise ValueError("The source OOF energy products exceed the numeric range")
    gain = 0.
    if numerator > 0. and denominator > 0.:
        gain = SCALAR_MAX if numerator / SCALAR_MAX >= denominator else float(numerator / denominator)
    return ScalarCalibration(source_oof.axis.genes, source_oof.coordinate, gain)


def source_holdout(rows, domain, outer_parents=None):
    """Hold a pseudo-target domain out and exclude query parents from every fit domain."""
    fields = ["domain", "parent"]
    if (not set(fields) <= set(rows) or rows[fields].isna().any().any()
            or rows[fields].astype(str).apply(lambda col: col.str.strip()).eq("").any().any()
            or rows.domain.nunique() < 3):
        raise ValueError("Cross-fitting requires at least three named source domains and canonical parents")
    inside = rows.domain.eq(domain)
    parents = set(rows.loc[inside, "parent"]) if outer_parents is None else set(outer_parents)
    if not parents or not parents <= set(rows.loc[inside, "parent"]):
        raise ValueError("Every query parent must be registered in the held source domain")
    fit = ~inside & ~rows.parent.isin(parents)
    held = inside & rows.parent.isin(parents)
    if rows.loc[fit, "domain"].nunique() < 2:
        raise ValueError("Parent blocking leaves fewer than two independent source fit domains")
    return SourceHoldout(np.flatnonzero(fit), np.flatnonzero(held))


def fit_held_source_scalar(source_oof, rows, domain, read_source_fit_outcomes, outer_parents=None):
    """Request only the allowed inner source outcomes from a row-scoped loader."""
    if not {"study", "inst_id"} <= set(rows) or (
        tuple(rows[["study", "inst_id"]].itertuples(index=False, name=None)) != source_oof.axis.rows
    ):
        raise ValueError("The source fold registry differs from the ordered OOF rows")
    predicted = checked_deltas(source_oof, "registered source OOF predictions")
    split = source_holdout(rows, domain, outer_parents)
    ids = tuple(source_oof.axis.rows[i] for i in split.fit)
    subset = EffectDeltas(ResponseAxis(ids, source_oof.axis.genes), source_oof.coordinate, predicted[split.fit])
    gain = fit_source_scalar(subset, read_source_fit_outcomes(ids))
    return gain, split


def control_variance(controls, *, target):
    physical = controls.physical
    wells = set(physical_axis(physical.wells))
    if wells != set(physical_axis(controls.fit_wells)):
        raise ValueError("Control inputs differ from the registered physical fit wells")
    plates = {plate for plate, _ in wells}
    if target and (len(wells) < TARGET_MIN_WELLS or len(plates) < TARGET_MIN_PLATES):
        raise ControlSupportError("T2/T3 need at least 12 target-fit physical wells on at least 3 plates")
    if len(wells) < 2:
        raise ControlSupportError("Source variance estimation needs at least two fit control wells")
    values = checked_array(physical.values, (len(wells), len(physical.genes)), "registered fit controls")
    variance = np.var(values, axis=0, ddof=1, dtype=np.float64)
    if not np.isfinite(variance).all():
        raise ValueError("Control variance exceeds the numeric range")
    return variance


def fit_diagonal(source_controls, target_fit_controls, *, alpha, floor, clip):
    """Apply a source-selected shrink/floor/clip rule to commensurate fit controls.

    scale_id must refer to a registered common assay/normalization and
    physical-well aggregation contract. Matching only the nominal log base
    cannot establish comparable physical-well variances.
    """
    genes = tuple(source_controls.physical.genes)
    scale = source_controls.scale_id
    if (not genes or len(set(genes)) != len(genes)
            or genes != tuple(target_fit_controls.physical.genes)
            or not scale.strip() or scale != target_fit_controls.scale_id):
        raise ValueError("Control genes and assay/aggregation scale must be matched")
    floor = np.broadcast_to(np.asarray(floor, dtype=np.float64), (len(genes),))
    if (not np.isfinite([alpha, clip]).all() or not 0. <= alpha <= 1. or clip < 0.
            or not np.isfinite(floor).all() or (floor <= 0.).any()):
        raise ValueError("T2 requires finite nonnegative shrink and clip, and positive variance floors")
    source = control_variance(source_controls, target=False)
    target = control_variance(target_fit_controls, target=True)
    log_ratio = np.logaddexp(np.log(target, where=target > 0., out=np.full_like(target, -np.inf)), np.log(floor))
    log_ratio -= np.logaddexp(np.log(source, where=source > 0., out=np.full_like(source, -np.inf)), np.log(floor))
    weights = np.exp(np.clip(.5 * alpha * log_ratio, -clip, clip))
    if not np.isfinite(weights).all() or (weights <= 0.).any():
        raise ValueError("T2 factors are outside the representable positive range")
    return DiagonalCalibration(genes, scale, weights)


def calibrate(prediction, *, scalar=None, diagonal=None):
    """Transform a whole effect delta with source-fitted coefficients only."""
    values = checked_deltas(prediction, "prediction delta")
    gain = 1.
    if scalar is not None:
        if prediction.axis.genes != scalar.genes or prediction.coordinate != scalar.coordinate:
            raise ValueError("The scalar was fitted on a different gene or effect coordinate")
        gain = scalar.gain
        if not np.isfinite(gain) or not 0. <= gain <= SCALAR_MAX:
            raise ValueError("The source scalar must be in [0, 2]")
    if diagonal is not None:
        if prediction.axis.genes != diagonal.genes or prediction.coordinate != diagonal.scale_id:
            raise ValueError("The diagonal was fitted on a different gene or effect coordinate")
        factors = checked_array(diagonal.factors, (len(diagonal.genes),), "diagonal factors")
        if (factors <= 0.).any():
            raise ValueError("T2 factors must be strictly positive")
        values = values * factors
    result = checked_array(gain * values, values.shape, "calibrated prediction delta")
    return EffectDeltas(prediction.axis, prediction.coordinate, result)


def refit_composed_scalar(source_oof, source_outcomes, source_diagonals, *, factor_axis):
    """Fit the T3 scalar after applying the appropriate OOF diagonal to each source row.

    OOF factors come from each row's allowed fit controls and a rule chosen
    without the outer evaluation domain or held parents.
    """
    values = checked_deltas(source_oof, "source OOF predictions")
    if factor_axis != source_oof.axis:
        raise ValueError("Source OOF factors are not on the ordered source row and gene axis")
    factors = checked_array(source_diagonals, values.shape, "source OOF diagonal factors")
    if (factors <= 0.).any():
        raise ValueError("Source OOF diagonal factors must be strictly positive")
    transformed = EffectDeltas(source_oof.axis, source_oof.coordinate, factors * values)
    return fit_source_scalar(transformed, source_outcomes)
