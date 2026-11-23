"""Lock physical grids and endpoint maps inside inner-fit, then freeze E2 forecasts."""
import argparse
from collections import Counter
from hashlib import sha256 as digest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from nextgen_h1_anchor.inputs import json_write
from nextgen_h1_v2.data import ROOT as H1
from nextgen_h1_v2.freeze import ANCHOR, DUAL
from nextgen_h1_v2.predict import verify_freeze
from vc_build.io import sha256

from .curves import CONTRACT, DOSE, PAIR, PARENT, SCREENS, checked, read_json, task_roster

BUDGET = 5
MODELS = {"AnchorOnly": ANCHOR, "Dual": DUAL}
ORDERS = {"AnchorOnly": "anchor", "Dual": "dual"}
CONDITIONS = {"Full-curve": None, "Reduced-8pt": 8, "Reduced-6pt": 6, "Reduced-3pt": 3,
              "Single-dose": 1}
CONTROL = ("B0", "B1")
RAW_REP = "Full-curve-1rep"


def axis_hash(values):
    return digest(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def choose_grid(fit_curves):
    counts = Counter()
    for row in fit_curves.itertuples():
        counts[tuple(row.grid), row.lineage] += 1
    support = Counter(grid for grid, _ in counts)
    return min(support, key=lambda g: (-support[g], g)) if support else None


def dose_indices(grid, number):
    if number is None:
        return np.arange(len(grid))
    if len(grid) < number:
        return None
    log_dose = np.log10(grid)
    x = (log_dose - log_dose[0]) / (log_dose[-1] - log_dose[0])
    if number == 1:
        return np.array([np.argmin(abs(x - 2 / 3))])
    targets = np.linspace(0, 1, number)
    # Monotone unique assignment with fixed endpoints and lexical tie breaks.
    paths = {0: (0., (0,))}
    for j in range(1, number - 1):
        allowed = range(j, len(grid) - number + j + 1)
        paths = {i: min((err + (x[i] - targets[j]) ** 2, positions + (i,))
                       for prev, (err, positions) in paths.items() if prev < i)
                 for i in allowed}
    return np.array((*min(paths.values())[1], len(grid) - 1))


def slope_intercept(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    if len(x) < 20 or np.var(x) < 1e-12 or np.var(y) < 1e-12:
        return None
    dx, dy = x - x.mean(), y - y.mean()
    slope = max(0., float(dx @ dy / (dx @ dx)))
    return {"n": len(x), "slope": slope, "intercept": float(y.mean() - slope*x.mean()),
            "rmse_fit": float(np.sqrt(np.mean((dy - slope*dx)**2)))}


def convert(values, calibrator):
    return calibrator["slope"] * values + calibrator["intercept"]


def auc(grid, viability, positions):
    if positions is None:
        return np.nan
    if len(positions) == 1:
        return float(-viability[positions[0]])
    x = np.log10(np.asarray(grid)[positions])
    return float(-trapezoid(viability[positions], x=x) / (x[-1] - x[0]))


def fits(grid, locked):
    return len(grid) == len(locked) and np.allclose(grid, locked, rtol=.02, atol=1e-12)


def endpoint_table(curves, grid, replicates):
    candidates = curves[curves.grid.map(lambda g: fits(g, grid))].copy()
    if replicates == 2:
        candidates = candidates[candidates.two_reps].copy()
    eligible = candidates.sort_values(["lineage", "cell_id", "curve_id"]).drop_duplicates("lineage")
    schedules = {key: dose_indices(grid, n) for key, n in CONDITIONS.items()}
    full = schedules["Full-curve"]
    rows = []
    for row in eligible.itertuples(index=False):
        first = np.asarray(row.first)
        both = (first + np.asarray(row.second)) / 2 if row.two_reps else None
        primary = both if replicates == 2 else first
        summaries = {label: auc(row.grid, primary, positions) for label, positions in schedules.items()}
        if both is not None:
            summaries[RAW_REP] = auc(row.grid, first, full)
            summaries["Full-curve-2rep"] = auc(row.grid, both, full)
            summaries["replicate_auc_abs_diff"] = abs(auc(row.grid, first, full) - auc(row.grid, np.asarray(row.second), full))
            summaries["replicate_viability_abs_diff"] = float(np.abs(first - np.asarray(row.second)).mean())
        rows.append({"lineage": row.lineage, **summaries})
    return pd.DataFrame(rows).set_index("lineage") if rows else pd.DataFrame(index=pd.Index([], name="lineage"))


def counts(table):
    return {str(k): int(v) for k, v in table.value_counts().sort_index().items()}


def panel_support(reg, raw, held_ids, panel):
    subset = reg[reg.lineage.isin(held_ids) & reg.compound_id.isin(panel)]
    keys = raw[raw.lineage.isin(held_ids) & raw.compound_id.isin(panel)]
    per_curve = keys.groupby(["cell_id", "compound_id", "curve_id"])[DOSE].nunique()
    per_pair = per_curve.groupby(level=[0, 1]).max()
    physical_reps = keys.groupby(["cell_id", "compound_id", "curve_id", DOSE]).rep_id.nunique()
    return {"exact_parents": len(panel), "registered_sample_parent_pairs": len(subset),
            "source_sample_ids_with_raw_keys": int(keys.native_sample.nunique()),
            "parents_with_raw_keys": int(keys.compound_id.nunique()),
            "registered_pairs_with_raw_keys": int(len(per_pair)),
            "registered_pairs_missing_raw_keys": int(len(subset) - len(per_pair)),
            "registered_pairs_max_grid_count_distribution": counts(per_pair),
            "well_ids_per_concentration_distribution": counts(physical_reps),
            "assay_time_key_distribution": counts(keys.time)}


def effect_scope(indices, q, group, archive, package, mapped, kinds, orders, *, repeats_alternative=False):
    result = {"index": indices, "q": q, "group": group[indices]}
    keep = pd.Index(archive["index"]).get_indexer(indices)
    if np.any(keep < 0) or not np.array_equal(archive["q"], q):
        raise ValueError("E2 roster differs from published Q8 support")
    result.update({b: archive[f"{b}/0"][keep] for b in CONTROL})
    for arm, model_id in MODELS.items():
        model = package["models"][model_id]
        order = model.order[:BUDGET]
        expected = tuple(package["panel"][order])
        if expected != orders[arm]:
            raise ValueError("K5 measurement roster differs from the frozen model")
        result[f"{arm}/Published-anchor"] = archive[f"{model_id}/5"][keep]
        actual = group[indices]
        for condition, measurement in kinds.items():
            if measurement == "Full-curve-2rep" and not repeats_alternative:
                raise ValueError("Repeat diagnostic requires an explicit support scope")
            if any(measurement not in mapped[d] for d in expected):
                continue
            block = np.stack([mapped[d][measurement].reindex(actual).to_numpy() for d in expected], axis=1)
            if not np.isfinite(block).all():
                continue
            packet = np.broadcast_to(model.ctx.mean[order], (len(group), BUDGET)).copy()
            packet[indices] = block
            values = model.predict(packet, BUDGET)[indices][:, q]
            if not np.isfinite(values).all() or values.shape != (len(actual), len(q)):
                raise ValueError("Frozen forecast failed to provide all original Q8 candidates")
            result[f"{arm}/{condition}"] = values
    return result


def well_cost(orders, grids, primary_reps, *, replicate_scope=False):
    cost = {}
    for arm, order in orders.items():
        points = [len(grids[drug]) if grids[drug] is not None else 0 for drug in order]
        if not all(points):
            cost[arm] = {condition: None for condition in CONDITIONS}
            continue
        reps = 2 if replicate_scope else primary_reps
        cost[arm] = {key: (sum(points) if number is None else number * len(points)) * reps + 12
                     if number is None or min(points) >= number else None for key, number in CONDITIONS.items()}
        if reps == 2:
            cost[arm][RAW_REP] = sum(points) + 12
    return cost


def save_scope(out, name, data, cost_records):
    path = out / f"{name}.npz"
    np.savez_compressed(path, **data)
    return {"name": name, "forecast_sha256": sha256(path), "groups_n": len(data["index"]),
            "group_sha256": axis_hash(data["group"]), "conditions": sorted(set(data) - {"index", "q", "group"}),
            "well_cost": cost_records}


def predict(task, source_dir, out):
    study = task.rsplit("-", 1)[0]
    primary_reps = 2 if study == "CTRPv2" else 1
    archive = H1 / "primary-a"
    original_primary = read_json(H1 / "prediction_receipt.json")
    signed_prediction = checked(archive / f"{task}.json", original_primary["prediction_receipts"][task])
    signature = read_json(signed_prediction)
    dev, meta, held, fit, manifest, original = task_roster(task)
    _, package = verify_freeze(H1 / "frozen-a", H1 / "frozen_receipt.json", task)
    archive_file = checked(archive / f"{task}.npz", signature["sha256"])
    with np.load(archive_file, allow_pickle=False) as stored:
        archived = dict(stored)
    if sha256(H1 / "frozen_receipt.json") != original_primary["frozen_sha256"]:
        raise ValueError("Archived prediction/model gates disagree")
    if not (np.array_equal(archived["group"], original["group"]) and
            np.array_equal(held["groups"], package["groups"]) and
            np.array_equal(archived["index"], original["index"]) and
            np.array_equal(archived["q"], original["q"])):
        raise ValueError("Archived primary and registered physical support differ")

    stamp = read_json(source_dir / "receipt.json")
    if stamp["screen"] != study or stamp["contract_sha256"] != sha256(CONTRACT) or stamp["query_values_exported"]:
        raise ValueError("Curve extraction differs from the locked contract")
    reg, raw, curves = (pd.read_parquet(checked(source_dir / f"{name}.parquet", stamp["files"][f"{name}.parquet"]))
                        for name in ("registered", "raw_keys", "curves"))
    keys = reg[[*PAIR, "lineage"]]
    curves = curves.merge(keys, on=PAIR, validate="many_to_one")
    raw = raw.merge(keys, on=PAIR, validate="many_to_one")
    groups = held["groups"]
    held0 = archived["index"]
    fit_ids, original_ids = dev["groups"][fit], groups[held0]
    q = archived["q"]
    if set(original_ids) & set(fit_ids):
        raise ValueError("Endpoint fit used an outer held lineage")
    orders = {arm: tuple(dev["panel"][np.asarray(manifest["orders"][mode][:BUDGET])])
              for arm, mode in ORDERS.items()}
    relevant = set.union(*map(set, orders.values()))
    if relevant & set(dev["panel"][q]) or not relevant <= set(manifest["a8"]):
        raise ValueError("Archive anchor measurement overlaps Q8")
    replay_max = {}
    for arm, model_id in MODELS.items():
        model = package["models"][model_id]
        native_cols = [manifest["a8"].index(drug) for drug in orders[arm]]
        packet = np.broadcast_to(model.ctx.mean[model.order[:BUDGET]], (len(groups), BUDGET)).copy()
        packet[held0] = original["x"][:, native_cols]
        native_forecast = model.predict(packet, BUDGET)[held0][:, q]
        error = float(np.max(np.abs(native_forecast - archived[f"{model_id}/5"])))
        if not np.isfinite(error) or error >= 1e-6:
            raise ValueError(f"Frozen H1-v2 native anchor replay failed for {arm}: {error}")
        replay_max[arm] = error
    grids, ledger, mapped, pair_map = {}, {}, {}, {}
    lineage_mask = set(fit_ids) | set(original_ids)
    held_raw = reg[reg.lineage.isin(original_ids)]
    original_atlas_raw = raw[raw.lineage.isin(dev["groups"]) & raw.compound_id.isin(dev["panel"])]
    atlas = {"fit_lineages": len(dev["groups"]), "panel_drugs": len(dev["panel"]),
             "native_measured_training_pairs": int(np.isfinite(dev["y"]).sum()),
             "raw_treatment_well_keys_found": len(original_atlas_raw),
             "raw_lineage_parent_pairs_found": int(original_atlas_raw[["lineage", "compound_id"]].drop_duplicates().shape[0]),
             "illustrative_12_controls_per_lineage": len(dev["groups"])*12,
             "atlas_reused_marginal_wells": 0}

    for drug in manifest["a8"]:
        axis = list(dev["panel"]).index(drug)
        source_rows = held_raw[held_raw.compound_id.eq(drug)]
        all_curves = curves[curves.compound_id.eq(drug) & curves.lineage.isin(lineage_mask)]
        training = all_curves[all_curves.lineage.isin(fit_ids)]
        if primary_reps == 2:
            training = training[training.two_reps]
        grid = choose_grid(training)
        grids[drug] = grid
        label = meta.loc[drug, "source_compound_label"]
        present = all_curves[all_curves.lineage.isin(original_ids)]
        registered_qc = raw[raw.compound_id.eq(drug) & raw.lineage.isin(original_ids)]
        identity_pairs = set(map(tuple, source_rows[PAIR].to_numpy()))
        raw_pairs = set(map(tuple, registered_qc[PAIR].drop_duplicates().to_numpy()))
        eligible_pairs = set(map(tuple, present[PAIR].drop_duplicates().to_numpy()))
        individually_complete = set()
        item = {
            "name": label, "native_drugs": sorted(source_rows[PARENT].unique()),
            "modes_k5": [arm for arm in orders if drug in orders[arm]],
            "held_original_lineages": len(original_ids), "held_registered_sample_pairs": len(identity_pairs),
            "held_pairs_missing_raw": len(identity_pairs - raw_pairs),
            "held_pairs_no_resolved_curve": len(raw_pairs - eligible_pairs),
            "held_lineages_resolved_curve": int(present.lineage.nunique()),
            "resolved_curves_per_held_lineage_distribution": counts(present.groupby("lineage").size()),
            "fit_lineages_with_resolved_curve_for_primary_reps": int(training.lineage.nunique()),
        }
        if grid is not None:
            for label_cond, n in CONDITIONS.items():
                ids = dose_indices(grid, n)
                item.setdefault("dose_schedules_uM", {})[label_cond] = np.asarray(grid)[ids].tolist() if ids is not None else None
            table = endpoint_table(all_curves, grid, primary_reps)
            rep_table = endpoint_table(all_curves, grid, 2)
            individually_complete = set(table.index) & set(original_ids)
            original_y = dev["y"][fit, axis]
            table_fit = table.reindex(fit_ids)
            native = slope_intercept(table_fit["Full-curve"].to_numpy(), original_y) if "Full-curve" in table_fit else None
            single = slope_intercept(table_fit["Single-dose"].to_numpy(), table_fit["Full-curve"].to_numpy()) if native else None
            if native:
                measured = {}
                for condition in CONDITIONS:
                    values = table[condition]
                    if condition == "Single-dose":
                        values = convert(values, single) if single else pd.Series(np.nan, index=values.index)
                    measured[condition] = convert(values, native)
                if RAW_REP in table:
                    measured[RAW_REP] = convert(table[RAW_REP], native)
                mapped[drug] = pd.DataFrame(measured)
                if RAW_REP in rep_table:
                    rep = rep_table[["Full-curve-2rep", RAW_REP]].copy()
                    for condition in rep:
                        rep[condition] = convert(rep[condition], native)
                    pair_map[drug] = rep
                held_rep = rep_table.reindex(original_ids)
                n_reps = int(held_rep["Full-curve-2rep"].notna().sum()) if "Full-curve-2rep" in held_rep else 0
                if n_reps:
                    item["measured_held_replicate_error"] = {
                        "pairs": n_reps,
                        "mean_abs_auc_between_wells": float(held_rep.replicate_auc_abs_diff.mean()),
                        "mean_abs_viability_between_wells": float(held_rep.replicate_viability_abs_diff.mean())}
            item.update({"native_calibration_fit_only": native, "single_to_full_fit_only": single,
                         "locked_fit_grid_lineages": int(training[training.grid.map(lambda g: fits(g, grid))].lineage.nunique()),
                         "held_lineages_on_locked_grid_with_primary_reps": len(individually_complete)})
        item["held_incomplete_on_locked_grid"] = len(original_ids) - len(individually_complete)
        ledger[drug] = item

    def eligible(tables, feature):
        support = [set(tables[d].index[tables[d][feature].notna()])
                   for d in relevant if d in tables and feature in tables[d]]
        chosen = set.intersection(*support) if len(support) == len(relevant) else set()
        return held0[np.isin(groups[held0], list(chosen))]

    use = eligible(mapped, "Full-curve")
    rep = eligible(pair_map, "Full-curve-2rep")
    out.mkdir()
    saved = {}
    main_kinds = {k: k for k in CONDITIONS}
    if primary_reps == 2:
        main_kinds[RAW_REP] = RAW_REP
    if len(use) >= 1:
        values = effect_scope(use, q, groups, archived, package, mapped, main_kinds, orders)
        saved["primary"] = save_scope(out, "primary", values, well_cost(orders, grids, primary_reps))
    if study == "GDSC2" and len(rep) >= 1:
        kinds = {"Full-curve": "Full-curve-2rep", RAW_REP: RAW_REP}
        values = effect_scope(rep, q, groups, archived, package, pair_map, kinds, orders, repeats_alternative=True)
        saved["two_rep_subset"] = save_scope(out, "two_rep_subset", values,
                                             well_cost(orders, grids, primary_reps, replicate_scope=True))

    receipt = {"task": task, "study": study, "endpoint": "native_oriented_AUC", "inner_fit_lineages": len(fit_ids),
        "inner_fit_lineage_sha256": axis_hash(fit_ids), "outer_lineage_sha256": axis_hash(groups),
        "archive_primary_forecast_sha256": signature["sha256"],
        "archive_model_sha256": manifest["package_sha256"],
        "archive_freeze_receipt_sha256": sha256(H1 / "frozen-a" / task / "freeze.json"),
        "archive_model_selection": {arm: manifest["choices"][mode]["champion"] for arm, mode in ORDERS.items()},
        "raw_source_receipt_sha256": sha256(source_dir / "receipt.json"),
        "contract_sha256": sha256(CONTRACT), "anchor_budget": BUDGET, "a8": manifest["a8"],
        "q8_parent_count": len(q), "q8_parent_sha256": axis_hash(dev["panel"][q]),
        "archived_original_held_count": len(held0), "common_curve_held_count": len(use),
        "original_held_count_with_complete_two_rep": len(rep),
        "held_exclusion_count": len(held0) - len(use),
        "same_original_q8": True, "primary_repeat_count": primary_reps,
        "k5_exact_parents": orders, "original_reference_atlas": atlas,
        "q8_metadata_support_audit": panel_support(reg, raw, original_ids, manifest["q8"]),
        "a8_curve_support_audit": ledger, "scopes": saved,
        "new_h1_model_fits": 0, "archived_native_forecast_replay_max_abs": replay_max,
        "outer_q_outcomes_opened": False}
    json_write(out / "receipt.json", receipt)
    print(task, len(use), "of", len(held0), "eligible; two-repeat", len(rep), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[f"{s}-{f}" for s in SCREENS for f in range(3)])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    predict(args.task, args.source, args.out)


if __name__ == "__main__":
    main()
