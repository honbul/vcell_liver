"""One official label read for all E4 tracks, then descriptive lambda curves in memory."""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from followup_h2.common import panel_hash, sha256
from followup_h2.inference import measure, summarize
from nextgen_h2.assets import KEY
from nextgen_h2.run import META, TRAIN, read_json
from nextgen_h2.score import external_labels
from vc_build.external import native_sensitivity
from vc_build.io import publish_json

from .data import OUTCOMES, V1_ASSETS, cpu_threads, registry, train_hash, utc_now
from .forecast import PERSONAL, adult_predictions, read_forecast, selection_lock
from .model import CALIBRATED, CONTROLS, FULL, KERNEL_ONLY, LOSO, PENALTIES, POOLED, Candidate
from .selection import InnerMetrics
from .test_pipeline import licob18

TRACK_NAMES = {"T1": "E4-Nested-CrossScreen", "T2": "E4-DRP-Nested", "T3": "E4-External"}
ORDERS = {
    "T1": (LOSO, POOLED, "H2-v1", "C1"),
    "T2": (FULL, KERNEL_ONLY, "H2-v2", "H2-v1", "C1", *CONTROLS),
    "T3": (FULL, KERNEL_ONLY, "H2-v2", "H2-v1", "C1"),
}
EXPOSURE = {
    "CrossScreen": "withheld_under_nested_E4",
    "DRP": "within_panel_nested_E4",
    "GDSC1": "E4_heldout_previous_H2_evaluation",
    "LIMORE": "previously_exposed_descriptive",
    "LICOB": "previously_exposed_descriptive",
    "LICOB18": "previously_exposed_descriptive",
}


def contrast_name(a, b):
    return f"{a} - {b}"


def evaluation(panels, groups, track, cohort):
    order = ORDERS[track]
    pairs = [(contrast_name(a, b), {a: 1., b: -1.}) for a, b in combinations(order, 2)]
    metrics, intervals, support = [], [], {}
    component = {} if track == "T1" else {KERNEL_ONLY: PERSONAL, FULL: PERSONAL}
    for scope in ("all", "lincs_observed", "lincs_unobserved"):
        subset = [(fold, data if scope == "all" else data.loc[
            data.lincs_observed.eq(scope == "lincs_observed")]) for fold, data in panels]
        rows = pd.concat([x for _, x in subset], ignore_index=True)
        if rows.empty:
            continue
        tags = {"track": TRACK_NAMES[track], "cohort": cohort, "support": scope, "exposure": EXPOSURE[cohort]}
        unique = {
            "rows": len(rows), "parents": rows[KEY].nunique(), "cells": rows.cell_id.nunique(),
            "scaffolds": rows[KEY].map(groups).fillna(rows[KEY]).nunique(),
            "lineages": rows.lineage_group.nunique(),
        }
        support[scope] = unique
        blocks = measure(subset, groups, order, components=component)
        arms = {a: {"arm": a} for a in order}
        m, d = summarize(blocks, tags, arms, pairs)
        for row in m:
            if row["screen"] == "macro":
                row.update(unique)
        metrics.extend(m)
        intervals.extend(row | {"screen": "macro", "fold": "all"} for row in d)
        for block in blocks:
            _, panel_ci = summarize([block], tags, arms, pairs)
            intervals.extend(row | {"screen": block.screen, "fold": block.fold} for row in panel_ci)
    print(TRACK_NAMES[track], cohort, support, flush=True)
    return metrics, intervals, support


def policy_gate(metrics, intervals, policy):
    primary = intervals[intervals.screen.eq("macro")]
    audit = {}

    def lookup(cohort, scope, metric, candidate, reference):
        row = primary.loc[primary.cohort.eq(cohort) & primary.support.eq(scope) & primary.metric.eq(metric)
                          & primary.contrast.eq(contrast_name(candidate, reference))]
        if len(row) != 1:
            raise ValueError(f"Missing locked contrast {cohort}/{scope}/{metric}/{candidate}/{reference}")
        found = row.iloc[0]
        return {key: float(found[key]) for key in ("delta", "lo95", "hi95")}

    cross_r = lookup("CrossScreen", "lincs_observed", "spearman", LOSO, POOLED)
    cross_n = lookup("CrossScreen", "all", "ndcg_at_5", LOSO, POOLED)
    screen_points = metrics[metrics.cohort.eq("CrossScreen") & metrics.screen.ne("macro")
                            & metrics.support.eq("lincs_observed") & metrics.metric.eq("spearman")]
    worst = {arm: float(screen_points.loc[screen_points.arm.eq(arm), "value"].min()) for arm in (LOSO, POOLED)}
    penalty = {arm: [rec["selected"]["penalty"] for name, rec in policy.items()
                     if name.startswith("T1/") and name.endswith(arm)] for arm in (LOSO, POOLED)}
    if any(len(x) != 3 for x in penalty.values()):
        raise ValueError("Expected three selections for both cross-screen methods")
    lens = {arm: [float(np.mean(vals)), float(np.median(vals))] for arm, vals in penalty.items()}
    changes = np.asarray(lens[LOSO]) - lens[POOLED]
    audit["track1"] = {
        "loso_minus_pooled_rho": cross_r, "loso_minus_pooled_ndcg": cross_n, "worst_screen_rho": worst,
        "lambda_values": penalty, "lambda_mean_and_median": lens, "rho_noninferior": cross_r["lo95"] >= -.005,
        "some_gain": cross_r["delta"] > 0 or cross_n["delta"] > 0 or worst[LOSO] > worst[POOLED],
        "regularization_non_decreasing_with_increase": bool((changes >= 0).all() and (changes > 0).any()),
    }

    gd_c = lookup("GDSC1", "lincs_observed", "spearman", FULL, "C1")
    gd_r = lookup("GDSC1", "lincs_observed", "spearman", FULL, "H2-v2")
    gd_n = lookup("GDSC1", "all", "ndcg_at_5", FULL, "H2-v2")
    audit["track3"] = {
        "gdsc1_rho_vs_c1": gd_c, "gdsc1_rho_vs_v2": gd_r, "gdsc1_ndcg_vs_v2": gd_n,
        "positive_vs_c1": gd_c["lo95"] > 0., "rho_noninferior_vs_v2": gd_r["lo95"] >= -.005,
        "ndcg_noninferior_vs_v2": gd_n["lo95"] >= -.005, "ndcg_point_improvement_above_02": gd_n["delta"] > .02,
    }

    controls = {a: lookup("DRP", "lincs_observed", "spearman", FULL, a) for a in ("C1", *CONTROLS)}
    drp_r = lookup("DRP", "lincs_observed", "spearman", FULL, "H2-v2")
    drp_n = lookup("DRP", "all", "ndcg_at_5", FULL, "H2-v2")
    audit["track2"] = {
        "rho_vs_controls": controls, "rho_vs_v2": drp_r, "ndcg_vs_v2": drp_n,
        "positive_vs_each_control": all(x["lo95"] > 0. for x in controls.values()),
        "rho_point_preserved_vs_v2": drp_r["delta"] >= -.010,
    }
    primary_go = all(audit["track1"][k] for k in (
        "rho_noninferior", "some_gain", "regularization_non_decreasing_with_increase"))
    primary_go &= all(audit["track3"][k] for k in (
        "positive_vs_c1", "rho_noninferior_vs_v2", "ndcg_noninferior_vs_v2"))
    primary_go &= all(audit["track2"][k] for k in ("positive_vs_each_control", "rho_point_preserved_vs_v2"))

    external = {}
    for cohort in ("GDSC1", "LIMORE", "LICOB18"):
        external[cohort] = {
            "rho": lookup(cohort, "lincs_observed", "spearman", FULL, "H2-v2"),
            "ndcg": lookup(cohort, "all", "ndcg_at_5", FULL, "H2-v2"),
        }
    clear_gain = cross_r["lo95"] > 0 or cross_n["lo95"] > 0 or any(
        row["lo95"] > 0 for contrasts in external.values() for row in contrasts.values())
    tradeoff = drp_r["delta"] < -.010 or drp_n["delta"] < -.030
    audit["partial"] = {"clear_cross_screen_or_external_gain": clear_gain,
                        "internal_tradeoff": tradeoff, "external": external}
    audit["decision"] = ("GO-LOSO-UPGRADE" if primary_go else
                         "PARTIAL-DUAL-PROFILE" if clear_gain and tradeoff else "NO-GO-LOSO")
    return audit


def lambda_curve(output, selection, grid, external_panels, groups):
    data = pd.read_csv(grid, sep="\t")
    locked_at = utc_now()
    rows = []
    for method, title in ((POOLED, "DRP Pooled-CV"), (LOSO, "DRP LOSO-CV")):
        role = f"T3/3/{method}"
        cfg = selection["policy"][role]["selected"]
        for p in PENALTIES:
            candidate = Candidate(**(cfg | {"penalty": p})).key
            block = data[data.selection.eq(role) & data.stage.eq(1) & data.candidate.eq(candidate)]
            value = block.loc[block.tier.eq("macro"), "value"].item()
            terms = block[block.tier.eq("screen")]
            rows.append({"dataset": title, "cohort": "DRP", "support": "lincs_observed", "lambda": p,
                         "spearman": value, "n_eligible_drug_screen_folds": int(terms.n.sum()),
                         "shape": cfg["shape"], "weight": cfg["weight"], "base": cfg["base"]})
    cfg = selection["policy"][f"T3/3/{LOSO}"]["selected"]
    for cohort, block in external_panels.items():
        for scope in ("all", "lincs_observed"):
            mask = None if scope == "all" else block.lincs_observed.to_numpy()
            scorer = InnerMetrics(block, groups, mask)
            for p in PENALTIES:
                vals = list(scorer.rank(block[f"lambda/{p:g}"].to_numpy()))
                rows.append({"dataset": cohort, "cohort": cohort, "support": scope, "lambda": p,
                             "spearman": float(np.mean([value for _, value, _ in vals])),
                             "n_eligible_drug_screen_folds": sum(n for _, _, n in vals),
                             "shape": cfg["shape"], "weight": cfg["weight"], "base": cfg["base"]})
    pd.DataFrame(rows).to_csv(output / "E4_LAMBDA_GENERALIZATION_CURVE.tsv", sep="\t", index=False)
    return locked_at


def run(preflight, selected, forecasts, tested, out):
    cpu_threads()
    split, selection, selection_sha = selection_lock(preflight, selected)
    test_file = tested / "test_receipt.json"
    tests = read_json(test_file)
    if (tests["status"] != "PASS" or tests["selection_sha256"] != selection_sha
            or tests["LOSO_Full_minus_Kernel_plus_V2_Int_within_drug_delta"] != 0.
            or tests["query_outcomes_opened"]
            or tests["passed_at_utc"] < selection["locked_at_utc"]):
        raise ValueError("Mandatory E4 tests are not locked")
    ready = []
    for i in range(len(split["evaluations"]) + 1):
        folder = forecasts / str(i)
        assert tests["forecast_receipt_sha256"][str(i)] == sha256(folder / "receipt.json")
        ready.append(read_forecast(folder, selection_sha))
    if sha256(TRAIN) != train_hash():
        raise ValueError("Source outcomes changed from the registered panel")
    reg = registry()

    # This exclusive guard is non-reentrant. Validation failures above do not consume a pass.
    out.mkdir(exist_ok=False, parents=True)
    started = utc_now()
    access = out / "outcome_access.json"
    publish_json(access, {
        "score_passes": 1, "opened_at_utc": started, "selection_sha256": selection_sha,
        "selection_locked_at_utc": selection["locked_at_utc"], "test_sha256": sha256(test_file),
        "forecast_receipts": tests["forecast_receipt_sha256"],
        "planned_label_reads": ["TRAIN", "GDSC1", "LIMORE", "LICOB"],
    })
    with TRAIN.open("rb") as source:
        raw = pd.read_parquet(source, columns=META + OUTCOMES).sort_values("row_id", ignore_index=True)
    assert panel_hash(raw) == split["source_roster_sha256"]
    values = pd.Series(native_sensitivity(raw).to_numpy(dtype=float), index=raw.row_id)
    print("One TRAIN outcome read", len(values), flush=True)
    by_track = defaultdict(list)
    for panel, note in ready[:-1]:
        by_track[note["track"]].append((note["fold"], panel.assign(truth=panel.row_id.map(values))))
    ext, labels_proof = ready[-1]
    labels, licence_table, ext_groups = external_labels(V1_ASSETS, read_json(V1_ASSETS / "data_receipt.json"))
    label_map = labels.set_index("row_id").truth
    assert set(label_map.index) == set(ext.row_id) and np.isfinite(label_map).all()
    print("One external native outcome read per cohort", len(labels), flush=True)
    by_cohort = {}
    for name, block in ext.assign(truth=ext.row_id.map(label_map)).groupby("source_id", sort=True):
        if name == "LICOB":
            block = adult_predictions(block, licence_table)
            by_cohort["LICOB18"] = licob18(block)
        by_cohort[name] = block

    metrics, paired, support = [], [], {}
    runs = [("T1", "CrossScreen", by_track["T1"], reg.scaffold_block),
            ("T2", "DRP", by_track["T2"], reg.scaffold_block),
            *(("T3", name, [(0, block)], ext_groups) for name, block in by_cohort.items())]
    for track, name, panels, groups in runs:
        m, p, size = evaluation(panels, groups, track, name)
        metrics.extend(m)
        paired.extend(p)
        support[name] = size
    metrics, paired = pd.DataFrame(metrics), pd.DataFrame(paired)
    comparison = policy_gate(metrics, paired, selection["policy"])
    metrics.to_csv(out / "E4_METRICS.tsv", sep="\t", index=False)
    paired.to_csv(out / "E4_PAIRED_95CI.tsv", sep="\t", index=False)
    assessment_done = utc_now()
    curve_started = lambda_curve(out, selection, selected / "E4_INNER_GRID.tsv", by_cohort, ext_groups)
    paths = ("E4_METRICS.tsv", "E4_PAIRED_95CI.tsv", "E4_LAMBDA_GENERALIZATION_CURVE.tsv")
    publish_json(out / "score_receipt.json", {
        "score_passes": 1, "outcome_access_sha256": sha256(access),
        "selection_sha256": selection_sha, "selection_locked_at_utc": selection["locked_at_utc"],
        "test_sha256": sha256(test_file), "started_at_utc": started, "metrics_sealed_at_utc": assessment_done,
        "curve_started_at_utc": curve_started, "finished_at_utc": utc_now(),
        "native_label_reads": {"TRAIN": 1, "GDSC1": 1, "LIMORE": 1, "LICOB": 1},
        "LICOB18": "subset of the same LICOB label read", "diagnostic_used_for_selection": False,
        "comparison": comparison, "supports": support, "forecast_receipts": tests["forecast_receipt_sha256"],
        "outputs": {name: sha256(out / name) for name in paths},
    })
    print(comparison["decision"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("preflight", "selection", "forecasts", "tests", "out"):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    run(args.preflight, args.selection, args.forecasts, args.tests, args.out)
