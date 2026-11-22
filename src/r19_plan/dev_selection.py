"""Compare registered DEV selection policies on the same gene-aligned checkpoint bank."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.r15_lincs.estimators import GAINS, loss

WELL_FIELDS = ("rna_plate", "rna_well")
PHASES = ("fit", "select", "score")
POLICIES = ("legacy_sequential", "joint")


@dataclass(frozen=True)
class ResponseAxis:
    rows: tuple[tuple[str, str], ...]
    genes: tuple[int, ...]


@dataclass(frozen=True)
class ComparisonDomain:
    validation_rows: pd.DataFrame
    genes: tuple[int, ...]
    inner_fit_parents: frozenset[str]
    outer_held_parents: frozenset[str]
    # Map fit, select and score to physical-well IDs. Only score IDs are needed.
    control_roles: dict[str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class Checkpoint:
    config_id: str
    parameters: tuple
    residual: np.ndarray


@dataclass(frozen=True)
class CheckpointBank:
    axis: ResponseAxis
    anchor: np.ndarray
    checkpoints: tuple[Checkpoint, ...]


@dataclass(frozen=True)
class ValidationTreatment:
    axis: ResponseAxis
    values: np.ndarray


@dataclass(frozen=True)
class PhysicalControls:
    wells: tuple[tuple[str, str], ...]
    genes: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class Selection:
    config_id: str
    parameters: tuple
    gain: float
    objective: float


def physical_axis(wells):
    frame = pd.DataFrame(wells, columns=WELL_FIELDS)
    if frame.empty or frame.isna().any().any():
        raise ValueError("Missing physical well identifiers")
    frame = frame.astype(str).apply(lambda column: column.str.strip().str.upper())
    frame["rna_well"] = frame.rna_well.str.replace(r"^([A-Z]+)([0-9])$", r"\g<1>0\2", regex=True)
    if frame.eq("").any().any() or frame.duplicated().any():
        raise ValueError("Empty or duplicate physical well identifiers")
    return tuple(frame.itertuples(index=False, name=None))


def checked_domain(domain, equal_controls):
    rows = domain.validation_rows
    required = ["study", "inst_id", "parent", *WELL_FIELDS]
    if (not set(required) <= set(rows) or rows.empty or rows[required].isna().any().any()
            or rows[required].astype(str).apply(lambda x: x.str.strip()).eq("").any().any()):
        raise ValueError("Missing validation row provenance")
    row_ids = tuple(rows[["study", "inst_id"]].itertuples(index=False, name=None))
    physical = set(physical_axis(rows[list(WELL_FIELDS)].itertuples(index=False, name=None)))
    if len(set(row_ids)) != len(rows):
        raise ValueError("Validation includes a duplicate physical treatment well")
    parent_sets = (domain.inner_fit_parents, set(rows.parent), domain.outer_held_parents)
    if not all(parent_sets) or len(set.union(*(set(group) for group in parent_sets))) != sum(map(len, parent_sets)):
        raise ValueError("Inner fit, inner selection and outer score parent sets must be nonempty and disjoint")
    if not domain.genes or len(set(domain.genes)) != len(domain.genes):
        raise ValueError("Invalid reference gene order")
    if set(domain.control_roles) != set(PHASES):
        raise ValueError("Fit, select and score require registered physical DMSO roles")
    roles = {role: set(physical_axis(domain.control_roles[role])) for role in PHASES}
    all_controls = set.union(*roles.values())
    if len(all_controls) != sum(map(len, roles.values())) or all_controls & physical:
        raise ValueError("Physical wells overlap across DMSO roles or with validation treatment")
    if equal_controls:
        fit, select, score = (Counter(plate for plate, _ in roles[role]) for role in PHASES)
        if not fit == select == score:
            raise ValueError("D1 requires equal numbers of physical controls for all three roles on each plate")
    return ResponseAxis(row_ids, domain.genes), roles


def checked_array(values, shape, label):
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite, floating-point well-by-gene array for {label}")
    return array


def checked_inputs(domain, bank, validation, equal_controls):
    axis, roles = checked_domain(domain, equal_controls)
    if bank.axis != axis or validation.axis != axis:
        raise ValueError("Checkpoint and treated response row or gene order differs from registered validation")
    shape = (len(axis.rows), len(axis.genes))
    treated = checked_array(validation.values, shape, "inner validation treatments")
    anchor = checked_array(bank.anchor, shape, "shared anchor")
    ids = [entry.config_id for entry in bank.checkpoints]
    keys = [entry.parameters for entry in bank.checkpoints]
    if (not ids or any(not isinstance(name, str) or not name.strip() for name in ids)
            or len(set(ids)) != len(ids) or len(set(keys)) != len(keys)):
        raise ValueError("Checkpoint identifiers and legacy keys must be unique")
    residual = {entry.config_id: checked_array(entry.residual, shape, entry.config_id)
                for entry in bank.checkpoints}
    return roles, treated, anchor, residual


def validation_delta(domain, treated, controls, allowed):
    wells = physical_axis(controls.wells)
    plates = domain.validation_rows.rna_plate.str.strip().str.upper().to_numpy()
    wanted = set(plates)
    expected = {well for well in allowed if well[0] in wanted}
    if (controls.genes != domain.genes or set(wells) != expected
            or {plate for plate, _ in wells} != wanted):
        raise ValueError("DMSO inputs differ from the permitted selection role or gene axis")
    values = checked_array(controls.values, (len(wells), len(domain.genes)), "validation DMSO wells")
    by_plate = defaultdict(list)
    for index, (plate, _) in enumerate(wells):
        by_plate[plate].append(index)
    means = {plate: values[indices].mean(axis=0) for plate, indices in by_plate.items()}
    return treated - np.stack([means[plate] for plate in plates])


def objectives(rows, delta, anchor, residual):
    if not delta.any():
        raise ValueError("The registered selection loss is undefined for all-zero validation truth")
    grid = {(name, gain): loss(rows, delta, anchor + gain * prediction)
            for name, prediction in residual.items() for gain in GAINS}
    if not np.isfinite(tuple(grid.values())).all():
        raise ValueError("Nonfinite registered selection loss")
    return grid


def pick_checkpoint(bank, grid, policy):
    entries = {entry.config_id: entry.parameters for entry in bank.checkpoints}
    if policy == "legacy_sequential":
        name = min(entries, key=lambda key: (round(grid[key, 1.], 9), entries[key]))
        gain = min(GAINS, key=lambda value: (round(grid[name, value], 9), value))
    elif policy == "joint":
        name, gain = min(grid, key=lambda key: (round(grid[key], 9), key[1], key[0]))
    else:
        raise ValueError(f"Unknown checkpoint selection policy: {policy}")
    return Selection(name, entries[name], gain, grid[name, gain])


def compare_d2(domain, bank, validation, select_controls):
    """Legacy R17 sequential versus DEV joint selection on one independent truth."""
    roles, treated, anchor, residual = checked_inputs(domain, bank, validation, equal_controls=False)
    delta = validation_delta(domain, treated, select_controls, roles["select"])
    grid = objectives(domain.validation_rows, delta, anchor, residual)
    return {policy: pick_checkpoint(bank, grid, policy) for policy in POLICIES}


def compare_d1(domain, bank, validation, fit_controls, select_controls, policy="joint"):
    """Change only validation DMSO while reusing the same bank, anchor, loss and policy."""
    roles, treated, anchor, residual = checked_inputs(domain, bank, validation, equal_controls=True)
    comparisons = {"shared": (fit_controls, roles["fit"]), "independent": (select_controls, roles["select"])}
    choices = {}
    for name, (controls, allowed) in comparisons.items():
        delta = validation_delta(domain, treated, controls, allowed)
        grid = objectives(domain.validation_rows, delta, anchor, residual)
        choices[name] = pick_checkpoint(bank, grid, policy)
    return choices
