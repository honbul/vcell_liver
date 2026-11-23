"""One-time held-screen evaluation from immutable predictions and detached outcomes."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from vc_build.io import _publish, publish_json, sha256, verify_forecast

from .data import IDENTITY, ROOT, contract, read_json, read_part, split_id
from .score import ALL_METRICS, CORE_METRICS, gate, screen_scores, shared_draws

REFERENCES = ("structured", "drug_ridge", "drug_mean")
REPORT_TISSUE = ["status", "tissue", "screen", "model", "lineages", "rows", *ALL_METRICS]
KEY = [*IDENTITY, "fold"]
CLAIM = ROOT / "outer-score/score-pass.json"


def check_forecasts():
    lock_file = ROOT / "final-lock.json"
    lock = read_json(lock_file)
    for stage, digest in lock["inner_summary_sha256"].items():
        if sha256(ROOT / f"analysis/{stage}-summary.json") != digest:
            raise ValueError(f"Frozen inner selection evidence changed: {stage}")
    models = {arm["name"]: [f'{arm["name"]}__{seed}' for seed in lock["seeds"]] for arm in lock["arms"]}
    models |= {name: [name] for name in REFERENCES}
    columns = [name for members in models.values() for name in members]
    groups, references = [], {}
    for index in range(9):
        tag = split_id(index)
        path = ROOT / f"outer-forecast/{tag}/forecast.parquet"
        raw, seal = verify_forecast(path)
        prov = seal["provenance"]
        if (prov["split"], prov["lock_sha256"], prov["labels_loaded"]) != (tag, sha256(lock_file), False):
            raise ValueError("The outcome-free fit seal or lock differs")
        if set(raw.method) != set(columns) or sorted(raw.fold.unique()) != [index % 3]:
            raise ValueError("Frozen methods or historical fold changed")
        for stamp in prov["runs"]:
            chk = path.parent / stamp["checkpoint"]
            if not chk.exists() or sha256(chk) != stamp["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed")
        names = [row["checkpoint"].removesuffix(".pt") for row in prov["runs"]]
        if len(names) != len(set(names)) or set(names) != set(columns) - set(REFERENCES):
            raise ValueError("Missing or duplicate seed checkpoint")
        checks = pd.DataFrame(prov["runs"])
        expected = {row["name"]: row for row in lock["folds"][tag]["arms"]}
        if any(expected[entry["arm"]["name"]] != entry["arm"] for entry in prov["runs"]):
            raise ValueError("Outer fit used a different inner fold operating point")
        for metric in ("initial_sha256", "batch_order_sha256"):
            native = checks.loc[checks.arm.map(lambda arm: arm["batch"] == "pairs" and arm["head"] == "native")]
            if not native.groupby("seed")[metric].nunique().eq(1).all():
                raise ValueError(f"Matched outer candidate has changed {metric}")
        query = read_part(ROOT / f"data/splits/{tag}/query.parquet")
        q = query.assign(fold=index % 3).set_index(KEY)
        forecast = raw.pivot(index=KEY, columns="method", values="predicted_value")
        if (set(forecast.index) != set(q.index) or len(forecast) != len(q)
                or not np.isfinite(forecast.to_numpy()).all()):
            raise ValueError(f"Frozen evaluation support differs: {tag}")
        aligned = q[["tissue_origin"]].join(forecast, validate="one_to_one")
        groups.append((tag, aligned.reset_index()))
        references[tag] = {
            "forecast_sha256": seal["sha256"], "rows_per_method": len(q), "methods": len(columns),
            "archived_comparator_sha256": prov["historical_reference_sha256"],
            "checkpoints": {row["checkpoint"]: row["checkpoint_sha256"] for row in prov["runs"]},
        }
    return groups, models, columns, references


def with_labels(groups):
    for tag, forecast in groups:
        file = ROOT / "data/outer-labels" / f"{tag}.parquet"
        actual = read_part(file)
        joined = forecast.merge(actual, on=IDENTITY, validate="one_to_one", how="left", indicator=True)
        if not joined._merge.eq("both").all() or len(joined) != len(actual) or not np.isfinite(joined.observed).all():
            raise ValueError(f"Frozen forecasts and detached labels differ: {tag}")
        yield joined.drop(columns="_merge")


def contrasts(lock):
    a, b = lock["a"], lock["b"]
    result = []
    if not a["early_stop"]:
        for control in dict.fromkeys((a["matched_g1"], a["strongest_g1"])):
            result.append(("H3a", a["strongest_g2"], control, True))
        for control in REFERENCES:
            result.append(("H3a", a["strongest_g2"], control, False))
    if b["champion"]:
        for control in (a["strongest_g1"], *REFERENCES):
            result.append(("H3b", b["champion"], control, control in (a["strongest_g1"], "structured")))
    return result


def calculate(frame, layout, columns, lock, conf):
    samples = shared_draws(frame, conf["bootstraps"], conf["bootstrap_seed"])
    screen_results = screen_scores(frame, columns, samples, minimum=conf["minimum_validation_lineages_per_drug"])
    screens = conf["screens"]
    if sorted(screen_results) != sorted(screens):
        raise ValueError("Exactly three scored source screens are required")
    views = {**screen_results, "macro": {}}
    for member in columns:
        views["macro"][member] = (
            {axis: float(np.mean([screen_results[s][member][0][axis] for s in screens])) for axis in ALL_METRICS},
            {axis: np.mean([screen_results[s][member][1][axis] for s in screens], axis=0) for axis in CORE_METRICS}
        )

    rows, intervals, gates = [], [], []
    for screen, shelf in views.items():
        support = frame if screen == "macro" else frame.loc[frame.source_id.eq(screen)]
        for model, members in layout.items():
            points = [{axis: shelf[member][0][axis] for axis in ALL_METRICS} for member in members]
            for point, member in zip(points, members):
                seed = member.rsplit("__", 1)[1] if len(members) > 1 else "fixed"
                rows.extend({"screen": screen, "model": model, "seed": seed, "metric": axis,
                             "value": point[axis], "seed_sd": None,
                             "lineages": support.lineage_group.nunique(), "rows": len(support)} for axis in ALL_METRICS)
            if len(members) > 1:
                rows.extend({
                    "screen": screen, "model": model, "seed": "mean", "metric": axis,
                    "value": np.mean([x[axis] for x in points]), "seed_sd": np.std([x[axis] for x in points], ddof=1),
                    "lineages": support.lineage_group.nunique(), "rows": len(support),
                } for axis in ALL_METRICS)

    for hypothesis, candidate, baseline, required in contrasts(lock):
        deltas, bounds = {}, {}
        for screen, shelf in views.items():
            deltas[screen] = {}
            bounds[screen] = {}
            for axis in CORE_METRICS:
                cand, base = ([shelf[member] for member in layout[name]] for name in (candidate, baseline))
                delta = float(np.mean([e[0][axis] for e in cand]) - np.mean([e[0][axis] for e in base]))
                distribution = np.mean([e[1][axis] for e in cand], axis=0) - np.mean(
                    [e[1][axis] for e in base], axis=0)
                if not np.isfinite(distribution).all():
                    raise ValueError("Missing paired bootstrap replicate")
                lower, upper = np.quantile(distribution, [.025, .975])
                deltas[screen][axis] = delta
                bounds[screen][axis] = [float(lower), float(upper)]
                intervals.append({
                    "hypothesis": hypothesis, "candidate": candidate, "baseline": baseline, "required": required,
                    "screen": screen, "metric": axis, "delta": delta, "lower": lower, "upper": upper,
                    "replicates": len(distribution), "bootstrap_unit": "global_lineage",
                })
        criteria = gate(deltas, bounds["macro"], conf)
        gates.append({"hypothesis": hypothesis, "candidate": candidate, "baseline": baseline,
                      "required": required, "criteria": criteria, "pass": all(criteria.values())})
    decisions = {}
    for key in ("H3a", "H3b"):
        required_items = [item for item in gates if item["hypothesis"] == key and item["required"]]
        if required_items:
            criteria_met = all(item["pass"] for item in required_items)
            decisions[key] = {"verdict": "GO-DEV" if criteria_met else "NO-GO",
                              "stage": "outer", "candidate": required_items[0]["candidate"]}
        else:
            decisions[key] = {"verdict": "NO-GO", "stage": "inner", "candidate": None}
    return pd.DataFrame(rows), pd.DataFrame(intervals), gates, decisions


def tissue_table(frame, layout, decisions):
    selected = [d["candidate"] for d in decisions.values() if d["verdict"] == "GO-DEV"]
    if not selected:
        return pd.DataFrame([{"status": "SKIPPED_NO_GO_DEV"}], columns=REPORT_TISSUE)
    show = {name: layout[name] for name in dict.fromkeys([*selected, "structured", "drug_ridge", "drug_mean"])}
    cols = [member for members in show.values() for member in members]
    output = []
    for tissue, sub in frame.groupby("tissue_origin", dropna=False, sort=True):
        if sub.lineage_group.nunique() < 5:
            continue
        per_screen = screen_scores(sub, cols)
        for screen, shelf in per_screen.items():
            support = sub.loc[sub.source_id.eq(screen)]
            for name, members in show.items():
                output.append({"status": "DESCRIPTIVE", "tissue": tissue, "screen": screen, "model": name,
                               "lineages": support.lineage_group.nunique(), "rows": len(support),
                               **{axis: np.mean([shelf[member][0][axis] for member in members])
                                  for axis in ALL_METRICS}})
    return pd.DataFrame(output, columns=REPORT_TISSUE)


def evaluate(*, preflight=False):
    if CLAIM.exists():
        raise FileExistsError("Outer evaluation was already claimed and cannot be repeated")
    groups, layout, columns, hashes = check_forecasts()
    if preflight:
        print(f"Preflight verified {len(groups)} complete frozen fold manifests and loaded no labels", flush=True)
        return
    output = CLAIM.parent
    lock = ROOT / "final-lock.json"
    conf = contract()
    locked = read_json(lock)
    claim = {"score_passes": 1, "claimed_utc": datetime.now(timezone.utc).isoformat(),
             "final_lock_sha256": sha256(lock), "predictions": hashes}
    publish_json(CLAIM, claim)
    frame = pd.concat(with_labels(groups), ignore_index=True)
    score, differences, gates, decisions = calculate(frame, layout, columns, locked, conf)
    tables = {"H3_METRICS.tsv": score, "H3_PAIRED_95CI.tsv": differences,
              "H3_PANCANCER_BY_TISSUE.tsv": tissue_table(frame, layout, decisions)}
    for name, table in tables.items():
        _publish(output / name, lambda path: table.to_csv(path, index=False, sep="\t"))
    publish_json(output / "score_receipt.json", claim | {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest_sha256": sha256(ROOT / "data/manifest.json"),
        "score_claim_sha256": sha256(CLAIM),
        "actual_input_hashes": {tag: sha256(ROOT / "data/outer-labels" / f"{tag}.parquet") for tag, _ in groups},
        "output_sha256": {name: sha256(output / name) for name in tables},
        "native_observations": len(frame), "global_lineages": frame.lineage_group.nunique(),
        "screen_score_passes": {name: 1 for name in conf["screens"]},
        "seed_use": "mean of seed-level scores, sample SD; one common lineage draw per bootstrap and no seed as N",
        "gates": gates, "decisions": decisions,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    evaluate(preflight=parser.parse_args().preflight)
