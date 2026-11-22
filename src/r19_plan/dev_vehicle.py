"""Fit-only vehicle augmentation and nuisance shrinkage for DEV D3."""

from collections import defaultdict
from dataclasses import dataclass
from zlib import crc32

import numpy as np

from src.r19_plan.dev_selection import PHASES, checked_array, physical_axis

RANKS = (1, 2)
STRENGTHS = (0., .25, .5)
WELL = tuple[str, str]


class ControlSupportError(ValueError):
    """The requested D3 arm lacks independent fit-control support."""


@dataclass(frozen=True)
class AugmentedFit:
    row_indices: np.ndarray
    view_ids: np.ndarray
    controls: np.ndarray
    residual: np.ndarray
    weights: np.ndarray
    view_wells: dict[str, tuple[tuple[WELL, ...], tuple[WELL, ...]]]


@dataclass(frozen=True)
class NuisanceBasis:
    genes: tuple[int, ...]
    vectors: np.ndarray
    singular_values: np.ndarray
    repeated_fit_plates: tuple[str, ...]
    physical_wells: int
    contrast_df: int

    @property
    def supported_ranks(self):
        return tuple(rank for rank in RANKS if rank <= len(self.vectors))


def fit_controls_only(controls, roles, genes):
    if len(genes) == 0 or len(set(genes)) != len(genes) or tuple(controls.genes) != tuple(genes):
        raise ValueError("The fit controls do not match the registered gene order")
    if set(roles) != set(PHASES):
        raise ValueError("D3 requires a registered physical axis for each control role")
    registered = {name: set(physical_axis(roles[name])) for name in PHASES}
    all_wells = set.union(*registered.values())
    if len(all_wells) != sum(map(len, registered.values())):
        raise ValueError("D3 physical control roles must be disjoint")
    wells = physical_axis(controls.wells)
    if set(wells) != registered["fit"]:
        raise ValueError("D3 requires expression from exactly the registered fit controls")
    values = checked_array(controls.values, (len(wells), len(genes)), "D3 fit controls")
    by_plate = defaultdict(list)
    order = sorted(range(len(wells)), key=lambda i: (
        wells[i][0], crc32(":".join(wells[i]).encode()), wells[i][1],
    ))
    for i in order:
        by_plate[wells[i][0]].append(i)
    return wells, values, by_plate, all_wells


def augment_fit(treatment_wells, genes, treated, anchor, weights, fit_controls, roles):
    """Duplicate each fit treatment across two disjoint control views with fixed total weight."""
    wells, values, by_plate, registered = fit_controls_only(fit_controls, roles, genes)
    treatment = physical_axis(treatment_wells)
    if set(treatment) & registered:
        raise ValueError("A D3 treatment row is a registered control well")
    n = len(treatment)
    shape = (n, len(genes))
    treated = checked_array(treated, shape, "D3 fit treatments")
    anchor = checked_array(anchor, shape, "D3 shared anchor")
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (n,) or not np.isfinite(weights).all()
            or (weights < 0).any() or not (weights > 0).any()):
        raise ValueError("D3 requires finite nonnegative treatment weights with positive mass")
    plates = [plate for plate, _ in treatment]
    insufficient = sorted(plate for plate in set(plates) if len(by_plate.get(plate, ())) < 2)
    if insufficient:
        raise ControlSupportError(f"BLOCKED_CONTROL_SUPPORT: D3b needs two fit wells per plate: {insufficient}")

    means, fractions, view_wells = {}, {}, {}
    for plate in sorted(set(plates)):
        indices = by_plate[plate]
        first, second = indices[::2], indices[1::2]
        means[plate] = np.stack((values[first].mean(axis=0), values[second].mean(axis=0)))
        fractions[plate] = len(first) / len(indices)
        view_wells[plate] = tuple(tuple(wells[i] for i in part) for part in (first, second))
    inputs = np.concatenate([means[plate] for plate in plates], axis=0)
    row_indices = np.repeat(np.arange(n), 2)
    first_weights = weights * np.array([fractions[plate] for plate in plates])
    view_weights = np.column_stack((first_weights, weights - first_weights)).ravel()
    return AugmentedFit(
        row_indices=row_indices, view_ids=np.tile([0, 1], n), controls=inputs,
        residual=treated[row_indices] - inputs - anchor[row_indices], weights=view_weights,
        view_wells=view_wells,
    )


def learn_nuisance_basis(fit_controls, roles, genes):
    """Estimate within-plate control covariance with equal weight per repeated fit plate."""
    genes = tuple(genes)
    _, values, by_plate, _ = fit_controls_only(fit_controls, roles, genes)
    repeated = {plate: indices for plate, indices in by_plate.items() if len(indices) >= 2}
    if len(repeated) < 2:
        raise ControlSupportError("BLOCKED_CONTROL_SUPPORT: D3c needs two independently repeated fit plates")
    counts = [len(indices) for indices in repeated.values()]
    centered = []
    for indices in repeated.values():
        plate = values[indices].astype(np.float64)
        centered.append((plate - plate.mean(axis=0)) / np.sqrt(len(plate) - 1))
    matrix = np.concatenate(centered)
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(matrix.dtype).eps * singular[0]
    rank = min(max(RANKS), int(np.count_nonzero(singular > tolerance)))
    if not rank:
        raise ControlSupportError("BLOCKED_CONTROL_SUPPORT: D3c has no nonzero within-plate fit contrast")
    vectors = right[:rank]
    peak = np.argmax(np.abs(vectors), axis=1)
    vectors = vectors * np.sign(vectors[np.arange(rank), peak])[:, None]
    return NuisanceBasis(genes, vectors, singular[:rank], tuple(repeated), sum(counts),
                         sum(count - 1 for count in counts))


def shrink_neural_residual(residual, anchor, genes, basis, rank, strength):
    """Shrink only the neural residual, then add the unchanged shared row anchor."""
    if tuple(genes) != basis.genes:
        raise ValueError("The neural forecast does not match the nuisance basis gene order")
    if rank not in RANKS or strength not in STRENGTHS:
        raise ValueError("The requested nuisance setting is outside the registered D3c grid")
    if rank not in basis.supported_ranks:
        raise ControlSupportError(f"BLOCKED_CONTROL_SUPPORT: D3c has insufficient fit contrast for rank {rank}")
    if getattr(residual, "ndim", 0) != 2:
        raise ValueError("D3c needs a two-dimensional neural residual")
    shape = (len(residual), len(genes))
    residual = checked_array(residual, shape, "D3 neural residual")
    anchor = checked_array(anchor, shape, "D3 shared anchor")
    if strength == 0:
        return anchor + residual
    vectors = basis.vectors[:rank]
    projection = (residual @ vectors.T) @ vectors
    return anchor + (residual - strength * projection).astype(residual.dtype)
