"""Choose and hash inner-only operating points for each stage."""

import argparse

import numpy as np
import pandas as pd

from vc_build.io import _publish, publish_json, sha256

from .data import IDENTITY, ROOT, contract, read_json, split_id
from .fit import first_arms
from .score import ALL_METRICS, inner_scores
from .train import Arm

STAGES = ("round1", "round2-block", "round2-opt", "round2-head")
KEY = [*IDENTITY, "fold"]


def best(scores, models):
    return min(models, key=lambda name: (-scores[name]["ndcg_at_5"],
                                         -scores[name]["within_drug_spearman"],
                                         -scores[name]["precision_at_k"], name))


def options(stage):
    if stage == "round1":
        return first_arms()
    return [Arm(**x) for x in read_json(ROOT / f"{stage}-lock.json")["arms"]]


def measure(stage):
    models = [arm.name for arm in options(stage)]
    if stage == "round1":
        models += ["structured", "drug_mean"]
    metrics, hashes, runs, diag = [], {}, [], []
    for index in range(9):
        tag = split_id(index)
        root = ROOT / stage / tag
        receipt = read_json(root / "receipt.json")
        file = root / "forecast.parquet"
        if (receipt["stage"], receipt["split"], receipt["forecast_sha256"]) != (stage, tag, sha256(file)):
            raise ValueError("Inner forecast hash changed")
        hashes[tag] = receipt["forecast_sha256"]
        frame = pd.read_parquet(file)
        if sorted(frame.fold.unique()) != [0, 1] or frame.duplicated(KEY).any():
            raise ValueError("Inner fold forecasts do not share unique true rows")
        if stage != "round1":
            path = ROOT / f"round1/{tag}/forecast.parquet"
            sealed = read_json(ROOT / f"round1/{tag}/receipt.json")["forecast_sha256"]
            if sha256(path) != sealed:
                raise ValueError("Original inner roster changed")
            ref = pd.read_parquet(path, columns=[*KEY, "observed"])
            if not frame[[*KEY, "observed"]].equals(ref):
                raise ValueError("Inner stages do not share the registered scored observations")
        metrics.extend({"split": tag, "arm": name, **record}
                       for name, record in inner_scores(frame, models).items())
        for run in receipt["runs"]:
            if "seed" in run:
                runs.append({"split": tag, **run, "name": run["arm"]["name"]})
        for fold in range(2):
            path = root / f"gradients-inner{fold}.parquet"
            if stage in ("round1", "round2-block"):
                diag.append(pd.read_parquet(path))
                hashes[f"{tag}/gradients-inner{fold}"] = sha256(path)

    scores = pd.DataFrame(metrics)
    if not np.isfinite(scores[list(ALL_METRICS)].to_numpy()).all():
        raise ValueError("Unexpected unsupported inner aggregate")
    aggregates = scores.groupby("arm")[list(ALL_METRICS)].mean().to_dict(orient="index")
    output = ROOT / "analysis"
    _publish(output / f"{stage}-metrics.tsv", lambda path: scores.to_csv(path, sep="\t", index=False))
    run_frame = pd.DataFrame(runs)
    _publish(output / f"{stage}-runs.tsv", lambda path: run_frame.drop(
        columns="arm").to_csv(path, sep="\t", index=False))
    diag_summary = {}
    if diag:
        all_diag = pd.concat(diag, ignore_index=True)
        kept = all_diag.loc[all_diag.conflict.notna()].copy()
        kept["conflict"] = kept.conflict.astype(bool)
        kept["density"] = kept.batch_pairs / kept.rectangular_slots
        kept["eligible_cell_share"] = kept.eligible_cell_groups / kept.present_cell_groups
        groups = ["arm", "loss_a", "loss_b"]
        aggregate = dict(batches=("gradient_cosine", "size"), median_cosine=("gradient_cosine", "median"),
                         mean_cosine=("gradient_cosine", "mean"), median_norm_ratio=("norm_ratio", "median"),
                         median_weighted_ratio=("weighted_norm_ratio", "median"),
                         conflict_frequency=("conflict", "mean"), mean_valid_pairs=("batch_pairs", "mean"),
                         mean_density=("density", "mean"), mean_valid_drug_groups=("eligible_drug_groups", "mean"),
                         mean_valid_cell_groups=("eligible_cell_groups", "mean"),
                         mean_eligible_cell_share=("eligible_cell_share", "mean"))
        by_epoch = kept.groupby([*groups, "epoch"], as_index=False).agg(**aggregate)
        by_arm = kept.groupby(groups, as_index=False).agg(**aggregate)
        _publish(output / f"{stage}-gradient-by-epoch.tsv", lambda path: by_epoch.to_csv(path, sep="\t", index=False))
        _publish(output / f"{stage}-gradient-summary.tsv", lambda path: by_arm.to_csv(path, sep="\t", index=False))
        diag_summary = {
            "valid_pair_batches": len(kept),
            "all_pair_batches": len(all_diag),
            "g1_negative_cosine_frequency": float(kept.loc[kept.arm.str.startswith("G1"), "conflict"].mean()),
        }

    if stage == "round1":
        standard = run_frame.loc[run_frame.diagnostics]
        hashes_by_fold = standard.groupby(["split", "inner"])
        for key in ("initial_sha256", "batch_order_sha256", "pair_exposures", "updates", "shared_params"):
            if not hashes_by_fold[key].nunique().eq(1).all():
                raise ValueError(f"H3a immutable comparison violation: {key}")
        if len(standard) != 144 or not standard.shared_params.gt(0).all():
            raise ValueError("Incomplete H3a comparison")
        measured = run_frame[run_frame.timing_replay.eq(True)]
        if len(measured) != 4 or not measured.max_forecast_difference.le(1e-3).all():
            raise ValueError("Gradient audit changes the replay predictions")
    summary = {"arms": [arm.record() for arm in options(stage)], "scores": aggregates,
               "inputs_sha256": hashes, "diagnostics": diag_summary}
    publish_json(output / f"{stage}-summary.json", summary)
    return summary


def point_deltas(candidate, base):
    return {key: candidate[key] - base[key] for key in ALL_METRICS}


def choose(stage, summary):
    scores = summary["scores"]
    if stage == "round1":
        g1, g2 = [best(scores, [name for name in scores if name.startswith(prefix)])
                  for prefix in ("G1_", "G2_")]
        g1_at_g2 = g2.replace("G2", "G1")
        conflict = summary["diagnostics"]["g1_negative_cosine_frequency"]
        matching = {w: point_deltas(scores[f"G2_w{w}"], scores[f"G1_w{w}"]) for w in range(3)}
        stop = conflict < contract()["pilot_conflict_floor"] and max(
            result["ndcg_at_5"] for result in matching.values()) <= 0
        decision = {"strongest_g1": g1, "strongest_g2": g2, "matched_g1": g1_at_g2,
                    "early_stop": stop, "matched_weight_deltas": matching}
        publish_json(ROOT / "round1-decision.json", decision)
        lookup = {x.name: x for x in first_arms()}
        choose_block = [Arm(f"B_{source.method}_{batch}", source.method, source.weight, batch).record()
                        for source in (lookup[g1], lookup[g2]) for batch in ("block4x3", "block6x2")]
        publish_json(ROOT / "round2-block-lock.json", {"arms": choose_block, "preceding": decision})
        return

    base = read_json(ROOT / "round1-decision.json")
    reference = {x.name: x for x in first_arms()}
    weight = reference[base["strongest_g1"]].weight
    if stage == "round2-block":
        # Select the layout under the common H3b anchor optimizer.
        chosen = best(scores, [name for name in scores if name.startswith("B_sum_")])
        batch = chosen.removeprefix("B_sum_")
        arms = [Arm(f"O_{method}", method, weight, batch).record()
                for method in ("normalized", "gradnorm", "cagrad")]
        publish_json(ROOT / "round2-opt-lock.json", {"arms": arms, "best_block_under_g1": chosen})
        return

    if stage == "round2-opt":
        best_opt = best(scores, list(scores)).removeprefix("O_")
        batch = summary["arms"][0]["batch"]
        arms = [Arm(f"D_{head}_{method}", method, weight, batch, head).record()
                for method in ("sum", best_opt) for head in ("shared", "separate")]
        publish_json(ROOT / "round2-head-lock.json", {"arms": arms, "best_optimizer_under_block": best_opt})
        return

    books = [read_json(ROOT / f"analysis/{s}-summary.json") for s in STAGES]
    combined = {key: value for book in books[1:] for key, value in book["scores"].items()}
    arms = {item["name"]: item for book in books for item in book["arms"]}
    comparators = [base["strongest_g1"], "structured"]
    candidates = {}
    cfg = contract()["go"]
    for candidate, metrics in combined.items():
        changes = {name: point_deltas(metrics, books[0]["scores"][name]) for name in comparators}
        qualifying = all(d["ndcg_at_5"] >= cfg["ndcg_delta"]
                         and min(d["within_drug_spearman"], d["precision_at_k"]) >= cfg["protection"]
                         and d["residual_spearman"] > 0 for d in changes.values())
        candidates[candidate] = {"inner_gains": changes, "qualifying": qualifying}
    qualified = [name for name, row in candidates.items() if row["qualifying"]]
    champion = best(combined, qualified) if qualified else None
    best_unconstrained = best(combined, list(combined))
    outer = [base["strongest_g1"]]
    if not base["early_stop"]:
        outer.extend([base["strongest_g2"], base["matched_g1"]])
    if champion:
        outer.append(champion)
    record = {"a": base, "b": {"champion": champion, "best_by_ndcg": best_unconstrained,
                               "candidates": candidates, "inner_comparators": comparators},
              "seeds": contract()["seeds"], "arms": [arms[name] for name in dict.fromkeys(outer)],
              "inner_summary_sha256": {s: sha256(ROOT / f"analysis/{s}-summary.json") for s in STAGES}}
    publish_json(ROOT / "final-lock.json", record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    choose(args.stage, measure(args.stage))


if __name__ == "__main__":
    main()
