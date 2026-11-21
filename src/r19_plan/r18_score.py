"""Score R18 after the complete frozen forecast passes a separate verification."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import LINE
from src.r15_lincs.data import table
from src.r15_lincs.forecast import PROPOSED
from src.r16_lincs.predict import CELL_GAIN, HEADS
from src.r16_lincs.score import collect, pool, repeats, summarize
from src.r17_lincs.score import cosine, moments
from src.r19_plan.r18_access_plan import PLATE, TARGET_FIELDS
from src.r19_plan.r18_score_contract import (
    ANCHORS, PREDICTION_FILES, PRIMARY, check_artifacts, check_freeze, check_slice_id, confirmation_label,
    forecast_manifest, reference_arms, role_names, summarize_case,
)

SCORE_FILES = {"score_controls.tsv", "score_labels.tsv"}
REPEAT_MODES = {"other_plate": False, "other_plate_and_position": True}
PREVIOUS = (
    ("R15", "r15_lincs/data/internal/scored", "HEPG2_6h_primary", "Two roles, original VC gains"),
    ("R16", "r16_lincs/data/internal/fair", "HEPG2_6h_primary", "Shared fit/select, fair gains"),
    ("R16", "r16_lincs/data/internal/blend", "HEPG2_6h_primary", "Shared fit/select"),
    ("R16", "r16_lincs/data/HEPG2/blend", "HEPG2_24h_all", "Shared fit/select, C1 measures"),
    ("R17", "r17_lincs/data/score", "HEPG2_6h_primary", "Three disjoint roles"),
)


def score_history(reports, comparisons):
    versions = []
    for version, relative, scope, design in PREVIOUS:
        folder = reports / relative
        registered = read_json(folder / "receipt.json")["files"]["paired.tsv"]
        path = folder / "paired.tsv"
        if sha256(path) != registered:
            raise ValueError(f"Historical paired table changed: {path}")
        historical = table(path)
        keep = historical.scope.eq(scope) & historical.arm.isin((PROPOSED, PRIMARY))
        versions.append(historical.loc[keep].assign(version=version, design=design, source=relative))
    updated = pd.DataFrame(comparisons)
    keep = updated.scope.eq("C1") & updated.arm.isin((PROPOSED, PRIMARY))
    versions.append(updated.loc[keep].assign(version="R18", design="Three disjoint roles, C1 measures",
                                             source="paired.tsv"))
    return pd.concat(versions, ignore_index=True)


def controls_for(rows, controls, values, role):
    """Replicate the R17 within-plate arithmetic mean of physical DMSO wells."""
    if not controls.role.eq(role).all() or len(controls) != len(values):
        raise ValueError(f"Rows or role changed on the {role} control axis")
    means = {key: values[positions].mean(axis=0)
             for key, positions in controls.groupby(PLATE, sort=False).indices.items()}
    plates = list(rows[PLATE].itertuples(index=False, name=None))
    if missing := set(plates) - means.keys():
        raise ValueError(f"Missing registered {role} controls: {sorted(missing)[:3]}")
    return np.stack([means[plate] for plate in plates])


def masks(case, rows):
    if case == "C2":
        for dose, indices in rows.groupby("dose", sort=False).indices.items():
            yield f"C2:{dose}", np.asarray(indices)
    else:
        yield case, np.arange(len(rows))


def supports_repeat(rows, donor, alternative_position):
    metadata = rows.assign(**{LINE: rows.cell_id},
                           plate_id=rows.study + ":" + rows.rna_plate)
    return repeats(metadata, donor, alternative_position)


def repeat_stats(groups):
    points, fractions = [], []
    for (scope, scenario, mode), by_arm in sorted(groups.items()):
        once, _ = summarize(by_arm, scope, scenario, (), {}, np.inf)
        scores = {row["arm"]: row for row in once}
        baseline, repeated = scores["B0"], scores["repeat"]
        points.extend(dict(mode=mode, **point) for point in once)
        for arm in (*HEADS, PROPOSED):
            rmse_gap = baseline["rmse"] - repeated["rmse"]
            r2_gap = repeated["r2_pooled"] - baseline["r2_pooled"]
            fractions.append(dict(
                scope=scope, scenario=scenario, mode=mode, arm=arm,
                parents=repeated["parents"], rows=repeated["rows"], rmse_gap=rmse_gap,
                r2_gap=r2_gap,
                rmse_fraction=(baseline["rmse"] - scores[arm]["rmse"]) / rmse_gap if rmse_gap > 0 else np.nan,
                r2_fraction=(scores[arm]["r2_pooled"] - baseline["r2_pooled"]) / r2_gap if r2_gap > 0 else np.nan,
            ))
    return points, fractions


def contribution_tables(groups, comparisons):
    """Keep the point-selected comparator fixed for descriptive parent sensitivity."""
    contributions, leave_out = [], []
    for comparison in comparisons:
        scope, scenario, arm = (comparison[key] for key in ("scope", "scenario", "arm"))
        if arm not in (PRIMARY, PROPOSED):
            continue
        reference = comparison["reference"]
        comparator, metric = comparison["best_at_point"], comparison["metric"]
        by_arm = groups[scope, scenario]
        candidate, base = (pool(by_arm[name]) for name in (arm, comparator))
        energy = base.sst.sum()
        if metric == "rmse":
            component = candidate.rmse - base.rmse
            denominator = pd.Series(1., index=base.index)
        else:
            component, denominator = base.sse - candidate.sse, base.sst
        ordered = component.abs().sort_values(ascending=False, kind="stable").index
        absolute = component.abs().sum()
        context = dict(scope=scope, scenario=scenario, arm=arm, reference=reference,
                       comparator=comparator, metric=metric)
        for rank, parent in enumerate(ordered, 1):
            change = component.loc[parent]
            contributions.append(dict(context, parent=parent, rank=rank,
                                      delta_component=change, rows=int(base.loc[parent, "rows"])))
        for top in (5, 10, 20):
            removed = ordered[:top]
            kept = ordered[top:]
            remaining_denominator = denominator.loc[kept].sum()
            difference = component.loc[kept].sum() / remaining_denominator if remaining_denominator > 0 else np.nan
            share = component.loc[removed].abs().sum() / absolute if absolute > 0 else np.nan
            leave_out.append(dict(context, removed=len(removed), requested_top=top, remaining=len(kept),
                                  absolute_contribution_share=share, difference=difference,
                                  total_target_energy=energy))
    return contributions, leave_out


def load_observed(args, slice_id, files, rows):
    # Import only after verifying the prediction freeze in score().
    from src.r19_plan.r18_input_bundle import materialize_expression

    score_plan = args.access / "score-plan" / slice_id
    check_artifacts(score_plan, SCORE_FILES, files["score"])
    score_tables = {name.removesuffix(".tsv"): table(score_plan / name)
                    for name in sorted(SCORE_FILES)}
    labels = score_tables["score_labels"]
    if not labels[TARGET_FIELDS].equals(rows[TARGET_FIELDS]):
        raise ValueError(f"The score observation axis differs from the frozen forecast: {slice_id}")
    _, _, _, fit = check_slice_id(slice_id)
    fit, _, score_role = role_names(fit)
    pred_plan = args.access / "predict-plan" / slice_id
    fit_controls = table(pred_plan / "fit_controls.tsv")
    assigned = fit_controls.merge(rows[PLATE].drop_duplicates(), on=PLATE, validate="many_to_one")
    score_tables["fit_controls"] = assigned
    observed, index_hash = materialize_expression(
        score_tables, args.index, args.index_reference, read_json(args.access_ref), args.landmarks)
    score_mean = controls_for(rows, score_tables["score_controls"], observed["score_controls"], score_role)
    fit_mean = controls_for(rows, assigned, observed["fit_controls"], fit)
    target = observed["score_labels"]
    return target - score_mean, target - fit_mean, score_mean - fit_mean, index_hash


def accumulate(groups, repeats_by, behavior, case, scenario, rows, truths, predictions):
    against_score, against_fit, vehicle = truths
    measured = {name: values for name, values in predictions.items() if name not in ANCHORS}
    for scope, indices in masks(case, rows):
        focus = rows.iloc[indices]
        truth = against_score[indices]
        arms = {name: pred[indices] for name, pred in measured.items()}
        collect(groups, (scope, scenario), focus, truth, arms)
        direction = vehicle[indices].astype(np.float64)
        for name, estimate in arms.items():
            if name in ("B0", "B_pos", CELL_GAIN):
                residual = estimate
            else:
                anchor = "blend_anchor" if name in HEADS else "anchor"
                residual = estimate - predictions[anchor][indices]
            behavior[scope, scenario, name] += (
                moments(estimate.astype(np.float64), direction),
                moments(residual.astype(np.float64), direction),
            )
        donor = against_fit[indices]
        for mode, other_position in REPEAT_MODES.items():
            valid, forecast = supports_repeat(focus, donor, other_position)
            repeat_arms = {name: arms[name][valid] for name in ("B0", PROPOSED, *HEADS)}
            repeat_arms["repeat"] = forecast[valid]
            collect(repeats_by, (scope, scenario, mode), focus.iloc[valid], truth[valid], repeat_arms)


def score(args):
    # The only paths visited before check_freeze are frozen predictions and prediction inputs.
    current = forecast_manifest(args.forecasts, args.access, args.access_ref, args.protocol, args.d0_ref)
    check_freeze(args.freeze_file, current, args.expected_sha256)
    if sha256(args.landmarks) != current["landmark_sha256"]:
        raise ValueError("Evaluator landmark order differs from the prediction freeze")
    if args.out.exists():
        raise FileExistsError(args.out)
    locked = read_json(args.access_ref)["files"]
    grouped, repeat_grouped = {}, {}
    behavior = defaultdict(lambda: np.zeros((2, 3), dtype=np.float64))
    selection, input_indices = [], {}
    for slice_id, stamp in current["prediction_receipts"].items():
        case, scenario, _, _ = check_slice_id(slice_id)
        predicted = args.forecasts / slice_id
        if sha256(predicted / "receipt.json") != stamp:
            raise ValueError(f"Forecast receipt changed after freeze verification: {slice_id}")
        check_artifacts(predicted, PREDICTION_FILES, read_json(predicted / "receipt.json")["files"])
        rows = table(predicted / "rows.tsv")
        truth, donor, vehicle, index_hash = load_observed(args, slice_id, locked[slice_id], rows)
        input_indices[slice_id] = index_hash
        with np.load(predicted / "forecast.npz", allow_pickle=False) as arrays:
            accumulate(grouped, repeat_grouped, behavior, case, scenario, rows, (truth, donor, vehicle),
                       dict(arrays))
        selection.append(table(predicted / "selection.tsv").assign(slice=slice_id))

    points, comparisons = [], []
    for (scope, scenario), records in sorted(grouped.items()):
        group_points, group_comparisons = summarize_case(records, scope, scenario)
        points.extend(group_points)
        comparisons.extend(group_comparisons)
    label = confirmation_label(comparisons)
    repeat_points, reach = repeat_stats(repeat_grouped)
    contributions, leave_out = contribution_tables(grouped, comparisons)
    vehicle_rows = [dict(scope=scope, scenario=scenario, arm=arm,
                         raw_cosine=cosine(stats[0]), residual_cosine=cosine(stats[1]))
                    for (scope, scenario, arm), stats in sorted(behavior.items())]
    tables = dict(points=points, paired=comparisons, repeat_points=repeat_points,
                  repeat_fraction=reach, contributions=contributions, robustness=leave_out,
                  vehicle_alignment=vehicle_rows, selection=pd.concat(selection, ignore_index=True),
                  history=score_history(args.reports, comparisons))
    args.out.mkdir(parents=True, exist_ok=False)
    for name, values in tables.items():
        pd.DataFrame(values).to_csv(args.out / f"{name}.tsv", sep="\t", index=False, na_rep="NA")
    (args.out / "decision.txt").write_text(label + "\n", encoding="utf-8")
    write_json(args.out / "receipt.json", dict(
        freeze_sha256=args.expected_sha256, evaluator_code_sha256=sha256(__file__),
        index_sha256=input_indices, slices=len(current["prediction_receipts"]),
        files={p.name: sha256(p) for p in sorted(args.out.iterdir())},
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("forecasts", "access", "access-ref", "protocol", "d0-ref", "freeze-file",
                 "index", "index-reference", "landmarks", "reports", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
