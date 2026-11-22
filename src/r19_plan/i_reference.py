"""Derive the matched FiLM component and strong reference for the I-track ablation."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.r15_lincs.estimators import GAINS, loss
from src.r15_lincs.forecast import choose_proposed, fit_known, grid_chemical, position
from src.r16_lincs.predict import HEADS, gain_choice
from src.r17_lincs.predict import rotation
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_freeze import ROW_FIELDS, freeze_fold
from src.r19_plan.i_predict import FitResidual, QueryInputs, refit_kernels, select_kernels


@dataclass(frozen=True)
class MatchedReference:
    inner: FitResidual
    select: QueryInputs
    select_truth: np.ndarray
    outer: FitResidual
    query: QueryInputs
    forecast: np.ndarray
    settings: dict


def choose_blend(rows, truth, anchor, kernel, checkpoints):
    shape = truth.shape
    if not checkpoints or not truth.any():
        raise ValueError("I-track requires a checkpoint bank and nonzero independent selection truth")
    checked_array(anchor, shape, "registered selection anchor")
    checked_array(kernel, shape, "reference selection kernel")
    for params, array in checkpoints.items():
        checked_array(array, shape, f"checkpoint {params}")
    grid = {
        (params, a, b): loss(rows, truth, anchor + a * kernel + b * candidate)
        for params, candidate in checkpoints.items() for a, b in product(GAINS, repeat=2)
    }
    if not np.isfinite(list(grid.values())).all():
        raise ValueError("Nonfinite selection loss for the I-track reference")
    choice = min(grid, key=lambda key: (round(grid[key], 9), key[2], key[1], str(key[0])))
    return choice, grid[choice]


def fit_reference(args, bank, inner, validation, training, outer):
    """Use the established R17 family with the prospective DEV joint checkpoint rule."""
    if bank.scenario != "A":
        raise ValueError("The registered I-track reference excludes query donor RNA")
    if (set(inner.parent) & set(validation.parent) or set(training.parent) & set(outer.parent)
            or set(training.parent) != set(inner.parent) | set(validation.parent)):
        raise ValueError("I-track reference labels cross a registered parent partition")
    select_role, score_role = rotation(args.role)
    truth = bank.response(validation, select_role)
    y, inner_pos, val_pos = position(bank, inner, validation, args.role)
    pos_gain = gain_choice(validation, truth, val_pos, np.zeros_like(truth))
    raw, final_pos, query_pos = position(bank, training, outer, args.role)
    val_anchor, query_anchor = pos_gain * val_pos, pos_gain * query_pos
    inner_values, outer_values = y - pos_gain * inner_pos, raw - pos_gain * final_pos
    (rule, _), _ = choose_proposed(bank, inner, validation, y, truth, inner_pos, val_pos, pos_gain)
    val_kernel = grid_chemical(bank, inner, validation, inner_values, [rule])[rule]
    test_kernel = grid_chemical(bank, training, outer, outer_values, [rule])[rule]
    model = HEADS["blend_FiLM"]
    with TemporaryDirectory(prefix="i-fit-") as tmp:
        work = Path(tmp)
        candidates = {
            params: fit_known(args, bank, inner, validation, inner_values, None, val_anchor, model, work, params)
            for params in product(R12_LRS, R12_EPOCHS)
        }
        (params, kernel_gain, neural_gain), objective = choose_blend(validation, truth, val_anchor,
                                                                   val_kernel, candidates)
        final_neural = fit_known(args, bank, training, outer, outer_values, None, query_anchor, model, work, params)
    val_film = neural_gain * candidates[params]
    query_film = neural_gain * checked_array(final_neural, query_anchor.shape, "outer FiLM residual")
    return MatchedReference(
        inner=FitResidual(inner, inner_values), select=QueryInputs(validation, val_anchor, val_film),
        select_truth=truth, outer=FitResidual(training, outer_values),
        query=QueryInputs(outer, query_anchor, query_film), forecast=query_anchor + query_film + kernel_gain * test_kernel,
        settings=dict(fit_role=args.role, select_role=select_role, score_role=score_role, kernel_rule=rule,
                      kernel_gain=kernel_gain, neural_gain=neural_gain, checkpoint=params,
                      position_gain=pos_gain, selection_loss=objective),
    )


def predict_fold(args, bank, views, locked_inputs, query_axis):
    """Produce a frozen I-track slice from a predictor-only R17-style bank."""
    reference = fit_reference(args, bank, *bank.split())
    if not query_axis[ROW_FIELDS].reset_index(drop=True).equals(reference.query.rows[ROW_FIELDS]):
        raise ValueError("The I-track fit changed its registered query axis")
    settings = select_kernels(views, reference.inner, reference.select, reference.select_truth)
    forecasts = refit_kernels(views, reference.outer, reference.query, settings)
    path = args.out / f"f{args.fold}-{args.role}"
    return freeze_fold(path, query_axis, bank.landmarks, forecasts, reference.forecast,
                       settings, locked_inputs, args.protocol, reference_selection=reference.settings)
