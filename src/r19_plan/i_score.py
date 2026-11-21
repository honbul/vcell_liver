"""Score frozen I-track predictions on paired held-parent support."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FOLDS = (0, 1, 2)
BASELINES = ("B_R17", "I0")
PHYSICAL = ["study", "rna_plate", "rna_well"]
SAMPLE = ["study", "inst_id"]
METRICS = ("rmse", "r2_pooled")
BOOTSTRAPS = 10_000
MIN_PARENTS = 100
MIN_FOLD_PARENTS = 20
EXTRA_REFERENCES = {
    "I1": ("I1_annotation_only",),
    "I2": ("I2_morphology_only",),
    "I3": ("I1", "I2"),
    "I4": ("I3",),
    "I3_all": ("I1", "I2", "I3", "I4"),
}


@dataclass(frozen=True)
class ViewCoverage:
    """Effective availability after fold-specific fit-only feature mapping."""

    total_views: int
    observed_counts: np.ndarray


def checked_rows(rows, fit_parents_by_fold):
    required = list(dict.fromkeys([*PHYSICAL, *SAMPLE, "parent", "role", "fold"]))
    if (rows.empty or not set(required) <= set(rows)
            or rows[required].isna().any().any()
            or rows[required].astype(str).apply(lambda col: col.str.strip()).eq("").any().any()):
        raise ValueError("I-track scoring requires complete, ordered physical well metadata")
    rows = rows.reset_index(drop=True).copy()
    folds = pd.to_numeric(rows.fold, errors="raise")
    if not folds.isin(FOLDS).all() or not rows.role.isin(list("ABC")).all():
        raise ValueError("I-track scoring rows have unregistered folds or DMSO roles")
    rows["fold"] = folds.astype(int)
    lineage = rows[[*PHYSICAL, "inst_id", "parent", "fold"]].drop_duplicates()
    if (rows.duplicated([*PHYSICAL, "role"]).any() or lineage.duplicated(PHYSICAL).any()
            or lineage.duplicated(SAMPLE).any()):
        raise ValueError("I-track well identifiers or parent assignments disagree across roles")
    if rows.groupby("parent").fold.nunique().gt(1).any():
        raise ValueError("A scoring parent occurs in multiple outer folds")
    if "known_scaffold" in rows and (not pd.api.types.is_bool_dtype(rows.known_scaffold)
            or rows.known_scaffold.isna().any() or rows.groupby("parent").known_scaffold.nunique().gt(1).any()):
        raise ValueError("Fit-scaffold availability must be Boolean and stable within each parent")
    if "scaffold_status" in rows and (
        not rows.scaffold_status.isin(("seen", "unseen", "acyclic")).all()
        or rows.groupby("parent").scaffold_status.nunique().gt(1).any()
    ):
        raise ValueError("Fit-scaffold status must be registered and stable within each parent")
    if fit_parents_by_fold is not None:
        if set(fit_parents_by_fold) != set(FOLDS):
            raise ValueError("The fit-parent registry must cover the three outer folds")
        for fold, block in rows.groupby("fold"):
            if set(block.parent) & set(fit_parents_by_fold[fold]):
                raise ValueError(f"Held parents crossed into the fit registry for fold {fold}")
    return rows


def coverage_arrays(rows, coverage, arms):
    if set(coverage) != set(arms) - set(BASELINES):
        raise ValueError("I-track view coverage must match the registered non-baseline arms")
    totals, counts = {}, {}
    for arm, evidence in coverage.items():
        count = np.asarray(evidence.observed_counts)
        total = evidence.total_views
        if (not isinstance(total, (int, np.integer)) or isinstance(total, (bool, np.bool_))
                or total < 1 or count.shape != (len(rows),) or count.dtype.kind not in "biu"
                or (count < 0).any() or (count > total).any()
                or pd.Series(count).groupby(rows.parent).nunique().gt(1).any()):
            raise ValueError(f"Invalid effective view counts for {arm}")
        totals[arm], counts[arm] = total, count
    for control in counts:
        for suffix in ("_mask_only", "_parent_shuffle"):
            if not control.endswith(suffix):
                continue
            arm = control.removesuffix(suffix)
            if (arm not in counts or totals[arm] != totals[control]
                    or not np.array_equal(counts[arm], counts[control])):
                raise ValueError(f"Control changed the view availability axis: {control}")
    return totals, counts


def uses_fallback(arm):
    return not arm.endswith(("_mask_only", "_morphology_only"))


def checked_array(array, shape, label):
    array = np.asarray(array)
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"Expected the same finite well-by-gene axis for {label}")
    return array


def squared_rows(truth, prediction=None):
    """Compute well moments once per arm, keeping temporary arrays bounded."""
    result = np.empty(len(truth), dtype=np.float64)
    for start in range(0, len(truth), 1024):
        stop = min(start + 1024, len(truth))
        if prediction is None:
            delta = truth[start:stop].astype(np.float64)
        else:
            delta = np.subtract(truth[start:stop], prediction[start:stop], dtype=np.float64)
        result[start:stop] = np.einsum("ij,ij->i", delta, delta)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite per-well squared energy")
    return result


def forecasts_sse(truth, forecasts, counts):
    baseline = checked_array(forecasts["I0"], truth.shape, "I0")
    result = {}
    for arm, prediction in forecasts.items():
        estimated = checked_array(prediction, truth.shape, arm)
        if arm in counts and uses_fallback(arm):
            absent = counts[arm] == 0
            if not np.array_equal(estimated[absent], baseline[absent]):
                raise ValueError(f"Unobserved {arm} wells must use the frozen I0 forecast")
        result[arm] = squared_rows(truth, estimated)
    return result


def parent_summaries(rows, genes, target_energy, arm_error, reference_error):
    codes, parents = pd.factorize(rows.parent, sort=True)
    parent_well_count = np.bincount(codes, minlength=len(parents))
    well_errors = np.column_stack((arm_error, reference_error))
    rmse = np.column_stack([
        np.bincount(codes, weights=np.sqrt(error / genes)) / parent_well_count
        for error in well_errors.T
    ])
    sse = np.column_stack([np.bincount(codes, weights=error) for error in well_errors.T])
    exposure = np.bincount(codes, weights=target_energy)
    return rmse, sse, exposure


def paired_effects(rmse, sse, energy):
    """Use the R16 parent RMSE and pooled R² definitions with fixed paired references."""
    n = len(energy)
    points, interval = np.full((2, 2), np.nan), np.full((2, 2), np.nan)
    valid = np.zeros(2, dtype=int)
    if n:
        points[0] = rmse.mean(axis=0)
        if energy.sum() > 0:
            points[1] = 1 - sse.sum(axis=0) / energy.sum()
    if n < 2:
        return points, interval, valid

    rng = np.random.default_rng(150104)
    draws = np.empty((2, BOOTSTRAPS))
    rmse_delta = rmse[:, 0] - rmse[:, 1]
    sse_delta = sse[:, 1] - sse[:, 0]
    for start in range(0, BOOTSTRAPS, 128):
        stop = min(start + 128, BOOTSTRAPS)
        sample = rng.integers(n, size=(stop - start, n))
        denominator = energy[sample].sum(axis=1)
        draws[0, start:stop] = rmse_delta[sample].mean(axis=1)
        draws[1, start:stop] = np.divide(sse_delta[sample].sum(axis=1), denominator,
                                       out=np.full(stop - start, np.nan), where=denominator > 0)
    valid = np.isfinite(draws).sum(axis=1)
    for metric in np.flatnonzero(valid == BOOTSTRAPS):
        interval[metric] = np.quantile(draws[metric], [0.005, 0.995])
    return points, interval, valid


def plate_sensitivity(rows, genes, energy, errors, subsets):
    """Recompute fixed-prediction paired points when leaving out one physical plate."""
    results = []
    for scope, arm, reference, keep in subsets:
        selected = rows.loc[keep]
        for study, plate in selected[PHYSICAL[:2]].drop_duplicates().itertuples(index=False, name=None):
            remaining = keep & ~(rows.study.eq(study) & rows.rna_plate.eq(plate)).to_numpy()
            metadata = rows.loc[remaining]
            if metadata.empty:
                continue
            rmse, sse, total = parent_summaries(metadata, genes, energy[remaining],
                                                errors[arm][remaining], errors[reference][remaining])
            rmse_effect = (rmse[:, 0] - rmse[:, 1]).mean()
            r2_effect = (sse[:, 1] - sse[:, 0]).sum() / total.sum() if total.sum() > 0. else np.nan
            results.append(dict(scope=scope, arm=arm, reference=reference, removed_study=study,
                                removed_rna_plate=plate, remaining_parents=len(total),
                                remaining_physical_wells=len(metadata.drop_duplicates(PHYSICAL)),
                                delta_rmse=rmse_effect, delta_r2_pooled=r2_effect))
    return pd.DataFrame(results)


def score_i_track(rows, truth, forecasts, coverage, *, kernel_only=None, fit_parents_by_fold=None):
    """Return paired tables on the frozen row/gene order for one registered cohort.

    Rows include study, inst_id, rna_plate, rna_well, parent, fold and role.
    Optional scaffold_status (seen, unseen, acyclic) or known_scaffold must use fit scaffolds.
    The caller verifies the relevant freeze and supplies the same score-role
    DMSO delta to all models. Coverage is keyed by every view-dependent arm.
    """
    rows = checked_rows(rows, fit_parents_by_fold)
    if not set(BASELINES) <= set(forecasts):
        raise ValueError("I-track comparisons require frozen B_R17 and I0")
    truth = np.asarray(truth)
    if truth.ndim != 2 or truth.shape[0] != len(rows) or truth.shape[1] == 0:
        raise ValueError("Expected a well-by-gene score target aligned to the frozen rows")
    truth = checked_array(truth, truth.shape, "score target")
    tables = {name: [] for name in ("same_support", "full_fallback", "strata", "controls")}
    energy = squared_rows(truth)
    totals, counts = coverage_arrays(rows, coverage, forecasts)
    sse = forecasts_sse(truth, forecasts, counts)
    all_wells = np.ones(len(rows), dtype=bool)
    cached_pairs = {}

    def complete(arm):
        return counts[arm] == totals[arm] if arm in counts else all_wells

    def record(table, scope, arm, reference, keep, errors=sse):
        key = (id(errors), arm, reference, np.packbits(keep).tobytes())
        if key not in cached_pairs:
            selected = rows.loc[keep]
            fold_count = selected.groupby("fold").parent.nunique().reindex(FOLDS, fill_value=0)
            parents = selected.parent.nunique()
            context = dict(
                arm=arm, reference=reference, parents=parents, rows=len(selected),
                physical_wells=len(selected.drop_duplicates(PHYSICAL)),
                plates=len(selected.drop_duplicates(PHYSICAL[:2])), roles=selected.role.nunique(),
                **{f"fold{fold}_parents": fold_count.loc[fold] for fold in FOLDS},
                eligible=parents >= MIN_PARENTS and bool(fold_count.ge(MIN_FOLD_PARENTS).all()),
            )
            moments = parent_summaries(selected, truth.shape[1], energy[keep], errors[arm][keep], errors[reference][keep])
            values, interval, valid = paired_effects(*moments)
            cached_pairs[key] = [
                dict(context, metric=metric, arm_value=a, reference_value=r, difference=a - r,
                     low99=lo, high99=hi, valid_bootstrap_draws=count)
                for metric, (a, r), (lo, hi), count in zip(METRICS, values, interval, valid, strict=True)
            ]
        tables[table].extend(dict(row, scope=scope) for row in cached_pairs[key])

    candidates = ["I0", *sorted(counts)]
    main_plate_subsets = []
    for arm in candidates:
        references = ("B_R17",) if arm == "I0" else BASELINES
        for reference in (*references, *EXTRA_REFERENCES.get(arm, ())):
            if reference in sse:
                record("same_support", "complete_shared", arm, reference, complete(arm) & complete(reference))
        for reference in references:
            if uses_fallback(arm):
                record("full_fallback", "all", arm, reference, all_wells)
            if arm in counts:
                count = counts[arm]
                record("strata", "views_absent", arm, reference, count == 0)
                record("strata", "views_partial", arm, reference, (count > 0) & ~complete(arm))
                record("strata", "views_complete", arm, reference, complete(arm))
            parent_scope = "unseen_parent" if fit_parents_by_fold is not None else "outer_held_parent"
            record("strata", parent_scope, arm, reference, all_wells)
            if "scaffold_status" in rows:
                for status in ("seen", "unseen", "acyclic"):
                    label = f"scaffold_{status}_in_fit" if status != "acyclic" else "acyclic"
                    record("strata", label, arm, reference, rows.scaffold_status.eq(status).to_numpy())
            elif "known_scaffold" in rows:
                known = rows.known_scaffold.to_numpy(dtype=bool)
                record("strata", "scaffold_seen_in_fit", arm, reference, known)
                record("strata", "scaffold_unseen_in_fit", arm, reference, ~known)
            if arm == "I0" or arm in EXTRA_REFERENCES:
                main_plate_subsets.append(("all", arm, reference, all_wells))
                if arm in counts:
                    main_plate_subsets.append(("complete_shared", arm, reference, complete(arm)))

        controls = (name for name in sorted(counts) if name.startswith(arm + "_")
                    and (name.endswith(("_mask_only", "_parent_shuffle", "_zero_fill_mask"))
                         or name.startswith(arm + "_without_")))
        for control in controls:
            shared = complete(arm) & complete(control)
            record("controls", "complete_shared", arm, control, shared)
            record("controls", "complete_shared", control, "I0", shared)
            record("controls", "all", control, "I0", all_wells)

    if kernel_only is not None:
        if set(kernel_only) != set(sse) - {"B_R17"}:
            raise ValueError("Kernel-only diagnostic arms must match the frozen combination inventory")
        tables["kernel_only"] = []
        kd = forecasts_sse(truth, kernel_only, counts)
        for arm in sorted(counts):
            record("kernel_only", "all", arm, "I0", all_wells, kd)
            record("kernel_only", "complete_shared", arm, "I0", complete(arm), kd)

    columns = ["scope", "arm", "reference", "parents", "rows", "physical_wells", "plates", "roles",
               *(f"fold{fold}_parents" for fold in FOLDS), "eligible", "metric", "arm_value",
               "reference_value", "difference", "low99", "high99", "valid_bootstrap_draws"]
    result = {name: pd.DataFrame(values, columns=columns) for name, values in tables.items()}
    result["plate_sensitivity"] = plate_sensitivity(rows, truth.shape[1], energy, sse, main_plate_subsets)
    return result
