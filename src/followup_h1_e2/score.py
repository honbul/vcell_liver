"""Score sealed E2 forecasts once on the original fixed Q8 labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from nextgen_h1_anchor.inputs import json_write
from nextgen_h1_anchor.metrics import Retrieval, mean_valid
from nextgen_h1_v2.data import STUDY
from vc_build.io import sha256
from vc_build.v0plus_score import correlations

from .curves import CONTRACT, ROOT, SCREENS, checked, read_json
from .forecast import CONDITIONS, CONTROL, MODELS, RAW_REP, axis_hash

TASKS = tuple(f"{s}-{f}" for s in SCREENS for f in range(3))
METRICS = ("ndcg_at_5", "precision_at_5", "within_drug_spearman",
           "missed_true_top5_fraction", "selected_top5_actual_mean", "oracle_top5_regret")
PRIMARY = "primary"
REPEAT = "two_rep_subset"
FULL = "Full-curve"
LEGACY = "Published-anchor"


def load_scopes(root, task, receipt):
    scopes = {}
    for name, ref in receipt["scopes"].items():
        if name not in {PRIMARY, REPEAT}:
            raise ValueError(f"Unexpected forecast scope: {name}")
        with np.load(checked(root / task / f"{name}.npz", ref["forecast_sha256"]), allow_pickle=False) as rows:
            bundle = dict(rows)
        fields = set(bundle) - {"index", "q", "group"}
        if (len(bundle["index"]) != ref["groups_n"] or axis_hash(bundle["group"]) != ref["group_sha256"] or
            len(bundle["q"]) != receipt["q8_parent_count"] or sorted(fields) != ref["conditions"] or
            len(np.unique(bundle["index"])) != len(bundle["index"]) or
            len(np.unique(bundle["q"])) != len(bundle["q"]) or
            any(bundle[k].shape != (len(bundle["index"]), len(bundle["q"])) or
                not np.isfinite(bundle[k]).all() for k in fields)):
            raise ValueError(f"Invalid frozen predictions for {task}/{name}")
        scopes[name] = bundle
    return scopes


def seal(root, extracts, out):
    tasks, sources = {}, {}
    for screen in SCREENS:
        receipt = extracts / f"curves-{screen.lower()}-a/receipt.json"
        record = read_json(receipt)
        if record["query_values_exported"] or record["contract_sha256"] != sha256(CONTRACT):
            raise ValueError("Invalid source seal")
        for name, h in record["files"].items():
            checked(receipt.parent / name, h)
        sources[screen] = sha256(receipt)
    for task in TASKS:
        path = root / task / "receipt.json"
        ref = read_json(path)
        if (ref["task"] != task or ref["outer_q_outcomes_opened"] or ref["new_h1_model_fits"] or
            ref["raw_source_receipt_sha256"] != sources[ref["study"]] or not ref["same_original_q8"] or
            ref["contract_sha256"] != sha256(CONTRACT)):
            raise ValueError(f"Invalid label-blind seal: {task}")
        if max(ref["archived_native_forecast_replay_max_abs"].values()) >= 1e-6:
            raise ValueError(f"Archived H1-v2 inference replay failed: {task}")
        load_scopes(root, task, ref)
        tasks[task] = sha256(path)
    json_write(out, {"stage": "pre-score freeze", "folds": tasks, "sources": sources,
        "contract_sha256": sha256(CONTRACT), "outer_q_outcomes_opened": False})
    print("E2 label-blind seal verified", len(tasks), flush=True)


def row_summaries(y, p, retrieval, oracle):
    ndcg, precision = retrieval(p).T
    weights = Retrieval.top(np.where(retrieval.mask, p, -np.inf))
    selected = np.where(retrieval.valid, np.sum(weights * np.nan_to_num(y), axis=1) / 5, np.nan)
    return np.column_stack((ndcg, precision, 1-precision, selected, oracle-selected))


def compute_all(y, predictions, draws, seed):
    n = len(y)
    retrieval = Retrieval(y)
    oracle = np.sum(retrieval.truth_top * np.nan_to_num(y), axis=1) / 5
    forecasts = {key: np.where(retrieval.mask, p, np.nan) for key, p in predictions.items()}
    rows = {key: row_summaries(y, p, retrieval, oracle) for key, p in forecasts.items()}
    boot = np.random.default_rng(seed).integers(n, size=(draws, n))
    indices = np.vstack((np.arange(n), boot))
    estimates = {key: [] for key in forecasts}
    for block in np.array_split(indices, 1 + draws // 40):
        truth_ranks = rankdata(y[block], axis=1, nan_policy="omit")
        for key, p in forecasts.items():
            pred_rank = rankdata(p[block], axis=1, nan_policy="omit")
            rho = mean_valid(correlations(truth_ranks, pred_rank), 1)
            group_summary = mean_valid(rows[key][block], 1)
            estimates[key].append(np.column_stack((group_summary[:, :2], rho, group_summary[:, 2:])))
    return {label: np.concatenate(parts, axis=0) for label, parts in estimates.items()}


def single(input_):
    root, task, digest, draws, seed = input_
    path = checked(root / task / "receipt.json", digest)
    stamp = read_json(path)
    scopes = load_scopes(root, task, stamp)
    label_path = STUDY / f"labels/{task}.npz"
    original_label_stamp = read_json(STUDY / "receipt.json")["files"][f"labels/{task}.npz"]
    label_path = checked(label_path, original_label_stamp)
    results = {}
    with np.load(label_path, allow_pickle=False) as store:
        for name, forecasts in scopes.items():
            index, q, group = (forecasts.pop(key) for key in ("index", "q", "group"))
            if (not np.array_equal(store["groups"][index], group) or
                axis_hash(store["panel"][q]) != stamp["q8_parent_sha256"] or
                set(store["panel"][q]) & set(stamp["a8"])):
                raise ValueError("Verified outcomes do not match previously sealed Q8 keys")
            # Model scoring consumes this Q-only block. No measured anchor value is read from it.
            y = store["y"][np.ix_(index, q)]
            if np.any(np.isfinite(y).sum(1) < 15) or not Retrieval(y).valid.all():
                raise ValueError("Curve subset includes a lineage outside the archived eligible Q8 masks")
            scores = compute_all(y, forecasts, draws, seed)
            results[name] = {"estimates": scores, "n": len(y), "q": len(q),
                             "pairs": int(np.isfinite(y).sum()), "q8_mask_sha256": axis_hash(np.isfinite(y).ravel())}
    print(task, "scored on its original Q8 with raw curve eligibility", flush=True)
    return stamp, results, original_label_stamp


def interval(values):
    values = values[np.isfinite(values)]
    return tuple(np.quantile(values, [.025, .975])) if len(values) else (np.nan, np.nan)


def fold_average(matrices):
    return mean_valid(np.stack(matrices), 0)


def enough(indices, blocks, minimum):
    return bool(len(indices) and all(blocks[i]["n"] >= minimum for i in indices))


def tested_contrasts(names):
    for arm in MODELS:
        full = f"{arm}/{FULL}"
        for label in sorted(x for x in names if x.startswith(f"{arm}/")):
            condition = label.split("/")[1]
            for ref in CONTROL if condition != LEGACY else ():
                yield label, ref
            if label != full:
                yield label, full


def summary_rows(screen, level, scope, items, min_n):
    estimates, paired, settings = [], [], []
    folds = [stamp for stamp, _ in items]
    blocks = [out.get(scope, {}) for _, out in items]
    options = {method for block in blocks for method in block.get("estimates", {})}
    expected = set(CONTROL)
    expected |= {f"{arm}/{cond}" for arm in MODELS for cond in (list(CONDITIONS) + [LEGACY, RAW_REP]
                if scope == PRIMARY and screen == "CTRPv2" else
                [FULL, RAW_REP, LEGACY] if scope == REPEAT else list(CONDITIONS) + [LEGACY])}
    full_support = range(len(items))
    candidates = options | expected
    lookup = {}
    for name, baseline in tested_contrasts(candidates):
        keep = [i for i in full_support if {name, baseline} <= blocks[i].get("estimates", {}).keys()]
        common = len(keep) == len(items)
        differences = fold_average([blocks[i]["estimates"][name]-blocks[i]["estimates"][baseline] for i in keep]) if keep else None
        count = sum(blocks[i]["n"] for i in keep)
        nums = {}
        for j, metric in enumerate(METRICS):
            if differences is None:
                row = (np.nan, np.nan, np.nan, 0)
            else:
                x = differences[:, j]
                row = (x[0], *interval(x[1:]), int(np.isfinite(x[1:]).sum()))
            delta, lo, hi, draws = row
            status = "INFEASIBLE" if not common else "INCONCLUSIVE" if not enough(keep, blocks, min_n) or not draws else "ESTIMATED"
            paired.append({"screen": screen, "scope": scope, "level": level, "model_condition": name,
                           "reference": baseline, "metric": metric, "delta": delta, "lo95": lo, "hi95": hi,
                           "bootstrap_nonmissing": draws, "matched_folds": len(keep), "required_folds": len(items),
                           "lineages": count, "status": status})
            nums[metric] = row
        lookup[name, baseline] = (keep, nums)

    def criterium(name, comparator):
        keep, diff = lookup.get((name, comparator), ([], {}))
        if len(keep) != len(items):
            return "INFEASIBLE"
        if not enough(keep, blocks, min_n):
            return "INCONCLUSIVE"
        ndcg, rho = (diff[metric][1] for metric in ("ndcg_at_5", "within_drug_spearman"))
        if ndcg > 0:
            return "UTILITY-PASS"
        if rho > 0 and ndcg >= -.01:
            return "RHO-IMPROVEMENT_NDCG-NONINFERIOR"
        return "UTILITY-NOT-PASSED"

    for name in sorted(candidates):
        arm, condition = name.split("/") if "/" in name else ("No-anchors", name)
        keep = [i for i in full_support if name in blocks[i].get("estimates", {})]
        merged = fold_average([blocks[i]["estimates"][name] for i in keep]) if keep else None
        costs, reductions = [], []
        for i in keep:
            if arm == "No-anchors":
                costs.append(0)
            elif condition != LEGACY:
                per_fold = folds[i]["scopes"][scope]["well_cost"][arm]
                cost = per_fold[condition]
                costs.append(cost)
                reductions.append((per_fold[FULL]-cost)/per_fold[FULL])
        if merged is not None and len(keep) == len(items):
            status = "ESTIMATED" if enough(keep, blocks, min_n) else "INCONCLUSIVE"
        else:
            status = "INFEASIBLE"
        verdict = "REFERENCE" if condition == FULL else "NOT-APPLICABLE"
        against_b0 = against_b1 = combined = "NOT-APPLICABLE"
        if arm != "No-anchors" and condition != LEGACY:
            against_b0, against_b1 = (criterium(name, base) for base in CONTROL)
            joint = {against_b0, against_b1}
            if "INFEASIBLE" in joint:
                combined = "INFEASIBLE"
            elif "INCONCLUSIVE" in joint:
                combined = "INCONCLUSIVE"
            elif joint == {"UTILITY-PASS"}:
                combined = "UTILITY-PASS"
            elif "UTILITY-NOT-PASSED" in joint:
                combined = "UTILITY-NOT-PASSED"
            else:
                combined = "RHO-IMPROVEMENT_NDCG-NONINFERIOR"
            if condition != FULL:
                if status != "ESTIMATED":
                    verdict = status
                else:
                    _, diff = lookup[name, f"{arm}/{FULL}"]
                    verdict = ("COMPRESSION-PASS" if min(reductions) >= .20 and
                               diff["ndcg_at_5"][1] >= -.01 and diff["precision_at_5"][1] >= -.02
                               else "COMPRESSION-NOT-PASSED")

        support = {"screen": screen, "scope": scope, "level": level, "arm": arm, "condition": condition,
                   "status": status, "available_folds": len(keep), "required_folds": len(items),
                   "lineages": sum(blocks[i]["n"] for i in keep),
                   "observed_query_pairs": sum(blocks[i]["pairs"] for i in keep),
                   "q8_candidates_each_fold": folds[0]["q8_parent_count"],
                   "original_lineages_in_matched_folds": sum(folds[i]["archived_original_held_count"] for i in keep)}
        for j, key in enumerate(METRICS):
            value = (np.nan, np.nan, np.nan) if merged is None else (merged[0, j], *interval(merged[1:, j]))
            estimates.append(support | dict(zip(("value", "lo95", "hi95"), value)) | {"metric": key})
        settings.append(support | {"wells_min": min(costs, default=np.nan), "wells_max": max(costs, default=np.nan),
                          "well_reduction_min_fraction": min(reductions, default=np.nan),
                          "well_reduction_max_fraction": max(reductions, default=np.nan),
                          "fits_one_96well_accounting": max(costs) <= 96 if costs else None,
                          "compression": verdict, "utility_vs_b0": against_b0, "utility_vs_b1": against_b1,
                          "utility_vs_both": combined})
    return estimates, paired, settings


def score(root, gate_path, out):
    gate = read_json(gate_path)
    plan = read_json(CONTRACT)
    if set(gate["folds"]) != set(TASKS) or gate["outer_q_outcomes_opened"] or gate["contract_sha256"] != sha256(CONTRACT):
        raise ValueError("Invalid E2 pre-score seal")
    for task in TASKS:
        receipt = read_json(checked(root / task / "receipt.json", gate["folds"][task]))
        load_scopes(root, task, receipt)
    out.mkdir()
    claim = ROOT / "e2_score_claim.json"
    json_write(claim, {"gate_sha256": sha256(gate_path), "slurm_job": os.getenv("SLURM_JOB_ID"),
                       "purpose": "retrospective E2 on known data; original H1 score untouched"})
    draws = plan["bootstrap_draws"]
    tasks = [(root, task, gate["folds"][task], draws, plan["e2_seed"] + i) for i, task in enumerate(TASKS)]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(TASKS, pool.map(single, tasks)))
    assays, intervals, decisions, label_masks = [], [], [], {}
    for screen in SCREENS:
        group = [results[t][:2] for t in TASKS if t.startswith(screen + "-")]
        for scope in ((PRIMARY, REPEAT) if screen == "GDSC2" else (PRIMARY,)):
            for level, items in [("study", group), *((r["task"], [(r, s)]) for r, s in group)]:
                a, b, c = summary_rows(screen, level, scope, items, plan["minimum_held_lineages"])
                assays.extend(a)
                intervals.extend(b)
                decisions.extend(c)
    for task, (_, scores, label_hash) in results.items():
        label_masks[task] = {"label_sha256": label_hash, "scopes": {
            key: {k: scope[k] for k in ("n", "q", "pairs", "q8_mask_sha256")} for key, scope in scores.items()}}
    files = {"E2_METRICS.tsv": assays, "E2_PAIRED_95CI.tsv": intervals, "E2_COST_AND_GATES.tsv": decisions}
    for filename, rows in files.items():
        pd.DataFrame(rows).to_csv(out / filename, sep="\t", index=False, float_format="%.12g", na_rep="NA")
    json_write(out / "receipt.json", {"analysis": "E2_retrospective_measurement_compression", "score_passes": 1,
        "evaluation_exposure": "Original H1-v2 Q8 on previously exposed studies; complete raw support subsets reported",
        "sealed_forecast_gate_sha256": sha256(gate_path), "score_claim_sha256": sha256(claim),
        "contract_sha256": sha256(CONTRACT), "q8_label_masks": label_masks, "bootstrap_draws": draws,
        "paired_unit": "independent lineage, fixed candidate panel, resampled within original held fold, equal fold means",
        "query_values_in_forecast_stage": False, "h1_model_fit_count": 0,
        "files": {name: sha256(out / name) for name in files}})
    print("E2 score completed for all six folds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("seal", "score"):
        command = sub.add_parser(action)
        command.add_argument("forecast", type=Path)
        command.add_argument("second", type=Path)
        command.add_argument("out", type=Path)
    args = parser.parse_args()
    (seal if args.command == "seal" else score)(args.forecast, args.second, args.out)


if __name__ == "__main__":
    main()
