"""Compute prespecified paired I5 outcomes after verifying the prediction freeze."""

import numpy as np
import pandas as pd

from src.r19_plan.i_score import (
    FOLDS, METRICS, MIN_FOLD_PARENTS, MIN_PARENTS, PHYSICAL, checked_array,
    checked_rows, paired_effects, parent_summaries, squared_rows,
)

COMPARISONS = {
    "I5_kernel": ("B_R17", "I0"),
    "I5_neural": ("B_R17", "I5_kernel", "I5_neural_mask_only", "I5_neural_parent_shuffle"),
}
ARMS = set(COMPARISONS) | {ref for refs in COMPARISONS.values() for ref in refs}


def i5_cohorts(rows, view_present):
    if set(view_present) != {"I2", "I4"}:
        raise ValueError("I5 coverage requires the Cell Painting and PRISM masks")
    masks = []
    for view in ("I2", "I4"):
        mask = np.asarray(view_present[view])
        if (mask.shape != (len(rows),) or mask.dtype.kind != "b"
                or pd.Series(mask).groupby(rows.parent).nunique().gt(1).any()):
            raise ValueError(f"I5 {view} availability must be Boolean and constant by parent")
        masks.append(mask)
    i2, i4 = masks
    return {
        "both_phenotypes": i2 & i4,
        "prism_available": i4,
        "whole_fallback": np.ones(len(rows), dtype=bool),
        "i2_only": i2 & ~i4,
        "absent": ~(i2 | i4),
    }


def i5_errors(truth, forecasts, absent):
    if set(forecasts) != ARMS:
        raise ValueError("I5 requires exactly the registered candidate and reference forecasts")
    structure = checked_array(forecasts["I0"], truth.shape, "I0")
    kernel = checked_array(forecasts["I5_kernel"], truth.shape, "I5_kernel")
    if not np.array_equal(kernel[absent], structure[absent]):
        raise ValueError("I5_kernel must use I0 when both phenotypes are absent")
    errors = {"I0": squared_rows(truth, structure), "I5_kernel": squared_rows(truth, kernel)}
    for arm in sorted(ARMS - errors.keys()):
        prediction = checked_array(forecasts[arm], truth.shape, arm)
        if arm.startswith("I5_neural") and not np.array_equal(prediction[absent], kernel[absent]):
            raise ValueError(f"{arm} must use I5_kernel when both phenotypes are absent")
        errors[arm] = squared_rows(truth, prediction)
    return errors


def score_i5(rows, truth, forecasts, view_present, *, expected_support=None, fit_parents_by_fold=None):
    """Return fixed-reference macro-RMSE and pooled R² effects on frozen score rows."""
    rows = checked_rows(rows, fit_parents_by_fold)
    truth = np.asarray(truth)
    if truth.ndim != 2 or truth.shape[0] != len(rows) or truth.shape[1] == 0:
        raise ValueError("I5 score truth must use the same nonempty well-by-gene axis as its predictions")
    truth = checked_array(truth, truth.shape, "I5 score truth")
    cohorts = i5_cohorts(rows, view_present)
    if expected_support is not None:
        for scope, count in expected_support.items():
            if scope not in cohorts or rows.loc[cohorts[scope], "parent"].nunique() != count:
                raise ValueError(f"I5 parent support differs from its registered count: {scope}")

    target = squared_rows(truth)
    errors = i5_errors(truth, forecasts, cohorts["absent"])
    result = []
    for scope, mask in cohorts.items():
        selected = rows.loc[mask]
        parents = selected.parent.nunique()
        fold_support = selected.groupby("fold").parent.nunique().reindex(FOLDS, fill_value=0)
        support = dict(
            scope=scope, parents=parents, rows=len(selected), physical_wells=len(selected.drop_duplicates(PHYSICAL)),
            plates=len(selected.drop_duplicates(PHYSICAL[:2])), roles=selected.role.nunique(),
            **{f"fold{fold}_parents": int(fold_support.loc[fold]) for fold in FOLDS},
            eligible=parents >= MIN_PARENTS and bool(fold_support.ge(MIN_FOLD_PARENTS).all()),
        )
        for arm, references in COMPARISONS.items():
            for reference in references:
                by_parent = parent_summaries(
                    selected, truth.shape[1], target[mask], errors[arm][mask], errors[reference][mask],
                )
                points, interval, valid = paired_effects(*by_parent)
                result.extend(
                    dict(support, arm=arm, reference=reference, metric=metric, arm_value=a, reference_value=r,
                         difference=a - r, low99=low, high99=high, valid_bootstrap_draws=int(draws))
                    for metric, (a, r), (low, high), draws in zip(METRICS, points, interval, valid, strict=True)
                )
    return pd.DataFrame(result)
