"""Verify the pushed report package after the one conditional score has finished."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, stdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from h3v2.grid import boundary_neighbors, identify

REQUIRED = {
    "PLAN_LOCK.md","DATA_SUPPORT_AND_EXPOSURE.md","ROUND1_TO_2_FAMILY_TUNING.md","RESULTS.md",
    "H3V2_METRICS.tsv","H3V2_PAIRED_95CI.tsv","H3V2_FAMILY_LEARNING_CURVES.tsv","H3V2_PANCANCER_BY_TISSUE.tsv",
    "model_card.md","verification.json","score_receipt.json","SLURM_JOBS.tsv",
}
FAMILIES = {"gbdt","film","attention","gnn","hybrid"}
SCORED_MODELS = FAMILIES | {"Champion"}
FIXED_MODELS = {"C1", "H2", "drug_mean", "drug_ridge", "Champion-ensemble"}
NATIVE_DRUG = {"spearman", "pearson", "interaction_spearman", "interaction_pearson"}
NATIVE_CELL = {"ndcg_at_5", "precision_at_5"}
HELD_METRICS = (NATIVE_DRUG - {"interaction_pearson"}) | NATIVE_CELL | {"within_cell_spearman"}
AXES = {"spearman":.01,"interaction_spearman":.02,"ndcg_at_5":.015}
COHORTS = {"GDSC1":39545,"LIMORE":3400,"LICOB":2888}
SCREENS = ("CTRPv2", "GDSC2", "PRISM20Q2")
SEEDS = {"11711", "11729", "11743"}
PARTITIONS = {f"{track}-{i}" for track in ("drug", "cell", *SCREENS) for i in range(3)}
CURVE_FIGURES = {"H3V2_FAMILY_LEARNING_CURVES.png", "H3V2_GRID_SATURATION.png"}
GRID_TABLE = "H3V2_GRID_SATURATION.tsv"
SCORE_TABLES = {"H3V2_METRICS.tsv", "H3V2_PAIRED_95CI.tsv", "H3V2_PANCANCER_BY_TISSUE.tsv",
                "H3V2_ERROR_COMPONENTS.tsv"}


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def rows(file):
    with file.open() as handle:
        return list(csv.DictReader(handle,delimiter="\t"))


def verify_frontier(grid, splits, replicated):
    groups = defaultdict(list)
    for row in grid:
        split, family, stage, candidate = (row[key] for key in ("split", "family", "round", "config_id"))
        config = json.loads(row["configuration"])
        if config["family"] != family or identify(config) != candidate:
            raise ValueError(f"Grid configuration identity changed for {split}/{family}/{stage}/{candidate}")
        if replicated[split, family, stage, candidate] != {"0", "1", "2"}:
            raise ValueError(f"Missing measured inner folds for {split}/{family}/{stage}/{candidate}")
        groups[split, family].append({
            "config": config, "id": candidate, "stage": stage, "frozen": row["frozen"] == "True",
            "order": (-float(row["utility"]), -float(row["ndcg_at_5"]), -float(row["spearman"]), candidate),
            "max_iteration": int(row["max_budget_pressure_iteration"]),
        })
    expected = {(split, family) for split in splits for family in FAMILIES}
    if set(groups) != expected:
        raise ValueError("The saturation grid does not cover every split and family")
    for (split, family), points in groups.items():
        identity = f"{split}/{family}"
        if len({point["id"] for point in points}) != len(points):
            raise ValueError(f"Duplicate settings in the saturation grid for {identity}")
        selected = min(points, key=lambda point: point["order"])
        lock = splits[split]["families"][family]
        if [point for point in points if point["frozen"]] != [selected] or any(
            selected[key] != lock[key] for key in ("config", "id", "stage")
        ):
            raise ValueError(f"Frozen candidate differs from the measured inner winner for {identity}")
        if boundary_neighbors(selected["config"], [point["config"] for point in points], selected["max_iteration"]):
            raise ValueError(f"Unmeasured saturation neighbors remain for {identity}")


def verify_curve_branches(grid, fits):
    measured = defaultdict(set)
    for row in fits:
        key = tuple(row[name] for name in ("split", "family", "round", "config_id", "inner"))
        neural = (row.get("train_huber"), row.get("val_huber"))
        trees = (row.get("train_objective"), row.get("validation_objective"))
        if all(value and isfinite(float(value)) for value in neural):
            measured[key].add("neural")
        if all(value and isfinite(float(value)) for value in trees):
            measured[key].add(row["branch"])
    for row in grid:
        key = tuple(row[name] for name in ("split", "family", "round", "config_id"))
        config = json.loads(row["configuration"])
        required = set(config["objective"].split("_")) if config["backend"] == "gbdt" else {"neural"}
        for fold in ("0", "1", "2"):
            if measured[(*key, fold)] != required:
                raise ValueError(f"Missing or unexpected learning curve branch for {'/'.join((*key, fold))}")


def verify_score_block(points, axes, label, *, held=False):
    """Check every model/axis identity and the score-level seed aggregates."""
    fixed = FIXED_MODELS
    replicated = SCORED_MODELS
    if held:
        fixed = fixed | {"structured_v1", "drug_ridge_v1", "drug_mean_v1"}
        replicated = replicated | {"H3v1"}
    arms = {(model, "fixed") for model in fixed}
    arms |= {(model, seed) for model in replicated for seed in SEEDS | {"mean"}}
    by_key = {(r["metric"], r["model"], r["seed"]): r for r in points}
    expected = {(axis, model, seed) for axis in axes for model, seed in arms}
    if len(points) != len(by_key) or set(by_key) != expected:
        raise ValueError(f"Missing, duplicate or unexpected score rows for {label}")
    for axis in axes:
        for model in replicated:
            values = [by_key[axis, model, seed]["value"] for seed in sorted(SEEDS)]
            values = [float(value) for value in values if value]
            aggregate = by_key[axis, model, "mean"]
            expected_stats = {"value": mean(values) if values else None,
                              "seed_sd": stdev(values) if len(values) > 1 else None}
            for name, expected_value in expected_stats.items():
                measured = float(aggregate[name]) if aggregate[name] else None
                if measured is None or expected_value is None:
                    matches = measured is expected_value
                else:
                    matches = isclose(measured, expected_value, rel_tol=1e-9, abs_tol=1e-12)
                if not matches:
                    raise ValueError(f"Incorrect {name} for {label}/{axis}/{model}")


def verify_reported_scores(root, receipt):
    metrics = defaultdict(list)
    for row in rows(root / "H3V2_METRICS.tsv"):
        metrics[row["scope"], row["support"], row["screen"]].append(row)
    internal = [("drug", support) for support in ("all", "lincs724", "structure462")]
    internal.append(("cell", "all"))
    native = [(scope, support, screen) for scope, support in internal for screen in ("macro", *SCREENS)]
    native += [(scope, "all", screen) for scope in COHORTS for screen in ("macro", scope)]
    for scope, support, screen in native:
        verify_score_block(metrics[scope, support, screen], NATIVE_DRUG | NATIVE_CELL,
                           f"{scope}/{support}/{screen}")
    for screen in ("macro", *SCREENS):
        verify_score_block(metrics["held", "core68", screen], HELD_METRICS, f"held/core68/{screen}", held=True)

    tissue_rows = defaultdict(list)
    for row in rows(root / "H3V2_PANCANCER_BY_TISSUE.tsv"):
        tissue_rows[row["tissue"]].append(row)
    roster = set(receipt["tissue_roster"])
    if len(roster) != 28 or set(tissue_rows) != roster:
        raise ValueError("Incomplete tissue stratification")
    for tissue, points in tissue_rows.items():
        if len(points) == 1 and points[0]["status"] == "insufficient_native_support":
            continue
        screens = defaultdict(list)
        for point in points:
            if (point["scope"], point["support"], point["status"]) != ("drug", "all", "estimated"):
                raise ValueError(f"Unexpected tissue score support for {tissue}")
            screens[point["screen"]].append(point)
        if "macro" not in screens:
            raise ValueError(f"Missing pooled tissue scores for {tissue}")
        for screen, group in screens.items():
            axes = {row["metric"] for row in group}
            if axes not in (NATIVE_DRUG, NATIVE_CELL, NATIVE_DRUG | NATIVE_CELL):
                raise ValueError(f"Incomplete set of native metrics for {tissue}/{screen}")
            verify_score_block(group, axes, f"{tissue}/{screen}")


def verify(root):
    for name in REQUIRED:
        if not (root/name).is_file() or (root/name).stat().st_size==0:
            raise ValueError(f"Missing or empty required deliverable: {name}")
    verify=json.loads((root/"verification.json").read_text())
    receipt=json.loads((root/"score_receipt.json").read_text())
    lock=json.loads((root/"final_lock.json").read_text())
    manifest=json.loads((root/"split_manifest.json").read_text())
    step0=json.loads((root/"step0_verification.json").read_text())
    if verify["outcomes_opened_by_independent_verification"] or verify["score_passes"]!=1 or receipt["score_passes"]!=1:
        raise ValueError("The audit cannot prove one-time score use")
    if verify["score_sha256"]!=digest(root/"score_receipt.json") or verify["frozen_sha256"]!=digest(root/"final_lock.json"):
        raise ValueError("Score/selection evidence differs from the published release")
    if verify["step0_sha256"]!=digest(root/"step0_verification.json"):
        raise ValueError("Feature parity evidence was not exported as verified")
    manifest_sha = digest(root / "split_manifest.json")
    if manifest_sha != verify["partition_manifest_sha256"] or manifest_sha != receipt["source_manifest_sha256"]:
        raise ValueError("Feature support changed")
    for name,sha in verify["deliverables"].items():
        if digest(root/name)!=sha:
            raise ValueError(f"Verified artifact changed while exporting {name}")
    if set(receipt["tables"]) != SCORE_TABLES:
        raise ValueError("Incomplete score table receipt")
    for name, sha in receipt["tables"].items():
        if verify["deliverables"].get(name) != sha:
            raise ValueError(f"Published score table differs from the single sealed pass: {name}")
    for name in CURVE_FIGURES:
        if name not in verify["deliverables"] or not (root / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Missing verified curve figure {name}")
    if GRID_TABLE not in verify["deliverables"]:
        raise ValueError("Missing verified saturation grid")
    if {key:receipt["parts"][key] for key in COHORTS}!=COHORTS or lock["outer_labels_opened"]:
        raise ValueError("Registered technical cohort support or outcome handling changed")
    if set(manifest["split_support"]) != PARTITIONS or set(lock["splits"]) != PARTITIONS:
        raise ValueError("The 15 registered internal partitions must all reach the freeze")
    if any(receipt["parts"][part] != entry["query_rows"] for part, entry in manifest["split_support"].items()):
        raise ValueError("A scored partition differs from its frozen row support")
    for track, expected in (("drug", 670830), ("cell", 670830), ("held", 45428)):
        prefixes = tuple(screen + "-" for screen in SCREENS) if track == "held" else (track + "-",)
        if sum(receipt["parts"][part] for part in PARTITIONS if part.startswith(prefixes)) != expected:
            raise ValueError(f"Incomplete measured row support for {track}")
    all_sites = PARTITIONS | {"external"}
    seed_models = {f"{family}__{seed}" for family in FAMILIES for seed in SEEDS}
    if (set(receipt["forecasts"]) != all_sites or verify["checkpoint_count"] != len(all_sites) * len(seed_models)
            or set(map(str, lock["seeds"])) != SEEDS):
        raise ValueError("The score omitted a frozen partition or seed")
    for part, proof in receipt["forecasts"].items():
        if set(proof["forecasts"]) != seed_models or set(proof["checkpoints"]) != seed_models:
            raise ValueError(f"Incomplete frozen model evidence for {part}")
    for label,line in zip(("가설","달라진 점","결과","판정","다음 결정"),(root/"RESULTS.md").read_text().splitlines()[:5]):
        if not line.startswith(f"- **{label}**"):
            raise ValueError("First five summary lines must precede the results tables")
    verify_reported_scores(root, receipt)
    fits=rows(root/"H3V2_FAMILY_LEARNING_CURVES.tsv")
    replicated = defaultdict(set)
    for row in fits:
        key = (row["split"], row["family"], row["round"], row["config_id"])
        if ((row.get("train_huber") and row.get("val_huber")) or
                (row.get("train_objective") and row.get("validation_objective"))):
            replicated[key].add(row["inner"])
    counts = defaultdict(int)
    for (split, family, round_name, config), folds in replicated.items():
        if folds == {"0", "1", "2"}:
            counts[split, family, round_name] += 1
    if any(counts[part, family, stage] < 2 for part in PARTITIONS for family in FAMILIES for stage in ("r1", "r2")):
        raise ValueError("A split or family is missing two measured configurations in a round")
    grid = rows(root / GRID_TABLE)
    verify_curve_branches(grid, fits)
    verify_frontier(grid, lock["splits"], replicated)
    trial_counts = Counter(row["split"] for row in grid)
    expected_inputs = {f"{split}/{entry['role']}": (split, entry)
                       for split, record in step0["folds"].items() for entry in record["feature_receipts"]
                       if entry["role"] != "outer"}
    input_evidence = verify["inner_fit_inputs"]
    if (set(expected_inputs) != {f"{split}/inner-{i}" for split in PARTITIONS for i in range(3)}
            or set(input_evidence) != set(expected_inputs)):
        raise ValueError("The completed feature audit does not cover every inner fold")
    for name, (split, entry) in expected_inputs.items():
        measured = input_evidence[name]
        if (measured["atlas_sha256"], measured["fit_rows"], measured["verified_trials"]) != (
            entry["atlas_sha256"], entry["fit_rows"], trial_counts[split]
        ):
            raise ValueError(f"Completed trial inputs disagree with the feature audit for {name}")
    checks=[r for r in rows(root/"H3V2_PAIRED_95CI.tsv") if r["baseline"]=="H2" and r["support"]=="all"]
    if set(receipt["verdicts"]) != FAMILIES | {"Champion"}:
        raise ValueError("Missing one or more preregistered promotion decisions")
    for model,outcome in receipt["verdicts"].items():
        evidence={(r["scope"],r["screen"],r["metric"]):r for r in checks if r["candidate"]==model}
        expected={}
        for axis,margin in AXES.items():
            here=evidence["drug","macro",axis]
            other=[evidence["drug","macro",name] for name in AXES if name!=axis]
            expected[axis]={
                "minimum_improvement_and_ci":float(here["delta"])>=margin and float(here["lower"])>0,
                "other_two_axes_protected":all(float(r["lower"])>=-.005 for r in other),
                "two_screens_improved":sum(float(evidence["drug",s,axis]["delta"])>0
                                           for s in SCREENS)>=2,
                "gdsc1_three_axes_protected":all(float(evidence["GDSC1","macro",m]["lower"])>=-.005 for m in AXES),
            }
        if outcome["by_axis"]!=expected or (outcome["verdict"]=="GO-MODEL")!=any(all(x.values()) for x in expected.values()):
            raise ValueError(f"The report did not enforce the registered gates: {model}")
    print(f"Verified {len(REQUIRED)} required reports, the closed inner grids and 28 tissue strata")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports",type=Path)
    verify(p.parse_args().reports)
