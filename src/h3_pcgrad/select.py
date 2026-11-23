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


def options(stage, split):
    if stage == "round1":
        return first_arms()
    return [Arm(**x) for x in read_json(ROOT / f"{stage}-lock.json")["splits"][split]["arms"]]


def publish_tsv(frame, path):
    content = frame.to_csv(sep="\t", index=False)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Partial inner evidence does not reproduce: {path}")
    else:
        _publish(path, lambda target: target.write_text(content))


def measure(stage):
    metrics, hashes, runs, diag, settings = [], {}, [], [], {}
    for index in range(9):
        tag = split_id(index)
        arms = options(stage, tag)
        settings[tag] = {arm.name: arm.record() for arm in arms}
        models = list(settings[tag])
        if stage == "round1":
            models += ["structured", "drug_mean"]
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
    publish_tsv(scores, output / f"{stage}-metrics.tsv")
    run_frame = pd.DataFrame(runs)
    publish_tsv(run_frame.drop(columns="arm"), output / f"{stage}-runs.tsv")
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
        publish_tsv(by_epoch, output / f"{stage}-gradient-by-epoch.tsv")
        publish_tsv(by_arm, output / f"{stage}-gradient-summary.tsv")
        baseline_prefix = "G1" if stage == "round1" else "B_sum"
        reference = kept.loc[kept.arm.str.startswith(baseline_prefix)]
        if reference.empty:
            raise ValueError("No weighted-sum reference in gradient diagnostics")
        diag_summary = {
            "valid_pair_batches": len(kept),
            "all_pair_batches": len(all_diag),
            "g1_negative_cosine_frequency": float(reference.conflict.mean()),
            "g1_frequency_by_split": reference.groupby("split").conflict.mean().to_dict(),
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
    summary = {"settings_by_split": settings, "scores": aggregates,
               "scores_by_split": {split: table.set_index("arm")[list(ALL_METRICS)].to_dict(orient="index")
                                   for split, table in scores.groupby("split")},
               "inputs_sha256": hashes, "diagnostics": diag_summary}
    publish_json(output / f"{stage}-summary.json", summary)
    return summary


def point_deltas(candidate, base):
    return {key: candidate[key] - base[key] for key in ALL_METRICS}


def select_round1(summary):
    decisions, block_options = {}, {}
    for tag, scores in summary["scores_by_split"].items():
        g1, g2 = [best(scores, [name for name in scores if name.startswith(prefix)])
                  for prefix in ("G1_", "G2_")]
        match = g2.replace("G2", "G1")
        conflict = summary["diagnostics"]["g1_frequency_by_split"][tag]
        matching = {w: point_deltas(scores[f"G2_w{w}"], scores[f"G1_w{w}"]) for w in range(3)}
        stop = conflict < contract()["pilot_conflict_floor"] and max(
            result["ndcg_at_5"] for result in matching.values()) <= 0
        decisions[tag] = {"strongest_g1": g1, "strongest_g2": g2, "matched_g1": match,
                          "pilot_stop": stop, "matched_weight_deltas": matching}
        sources = [Arm(**summary["settings_by_split"][tag][name]) for name in (g1, g2)]
        block_options[tag] = {"arms": [
            Arm(f"B_{arm.method}_{batch}", arm.method, arm.weight, batch).record()
            for arm in sources for batch in ("block4x3", "block6x2")
        ]}
    publish_json(ROOT / "round1-decision.json", {"splits": decisions})
    publish_json(ROOT / "round2-block-lock.json", {"splits": block_options})


def select_optimization(stage, summary, first):
    following = {}
    for tag, scores in summary["scores_by_split"].items():
        local_g1 = first["splits"][tag]["strongest_g1"]
        weight = next(arm.weight for arm in first_arms() if arm.name == local_g1)
        if stage == "round2-block":
            chosen = best(scores, [name for name in scores if name.startswith("B_sum_")])
            batch = chosen.removeprefix("B_sum_")
            arms = [Arm(f"O_{method}", method, weight, batch).record()
                    for method in ("normalized", "gradnorm", "cagrad")]
        else:
            chosen = best(scores, list(scores))
            best_opt = chosen.removeprefix("O_")
            batch = summary["settings_by_split"][tag][chosen]["batch"]
            arms = [Arm(f"D_{head}_{label}", method, weight, batch, head).record()
                    for label, method in (("sum", "sum"), ("adaptive", best_opt))
                    for head in ("shared", "separate")]
        following[tag] = {"arms": arms, "selected_prior_arm": chosen}
    next_stage = "round2-opt" if stage == "round2-block" else "round2-head"
    publish_json(ROOT / f"{next_stage}-lock.json", {"splits": following})


def final_lock(first):
    books = [read_json(ROOT / f"analysis/{stage}-summary.json") for stage in STAGES]
    cfg = contract()["go"]
    outer, choices = {}, {}
    stop_a = all(value["pilot_stop"] for value in first["splits"].values())
    for tag, a in first["splits"].items():
        initial = books[0]["scores_by_split"][tag]
        experiments = {name: row for book in books[1:] for name, row in book["scores_by_split"][tag].items()}
        arms = {name: row for book in books for name, row in book["settings_by_split"][tag].items()}
        comparisons = (a["strongest_g1"], "structured")
        candidates = {}
        for candidate, metrics in experiments.items():
            gains = {name: point_deltas(metrics, initial[name]) for name in comparisons}
            eligible = all(d["ndcg_at_5"] >= cfg["ndcg_delta"]
                           and min(d["within_drug_spearman"], d["precision_at_k"]) >= cfg["protection"]
                           and d["residual_spearman"] > 0 for d in gains.values())
            candidates[candidate] = {"inner_gains": gains, "qualifying": eligible}
        available = [name for name, row in candidates.items() if row["qualifying"]]
        champion = best(experiments, available) if available else None
        choices[tag] = {"champion": champion, "best_by_ndcg": best(experiments, list(experiments)),
                        "candidates": candidates, "inner_comparators": list(comparisons)}
        options = [arms[a["strongest_g1"]] | {"name": "G1_best"}]
        if not stop_a:
            options.extend([arms[a["strongest_g2"]] | {"name": "G2"},
                            arms[a["matched_g1"]] | {"name": "G1_matched"}])
        if champion:
            choices[tag]["selected_configuration"] = arms[champion]
            options.append(arms[champion] | {"name": "H3b"})
        outer[tag] = {"arms": options}

    successful = sum(row["champion"] is not None for row in choices.values())
    # A complete candidate policy has to furnish a challenger in every historical fold.
    if successful != len(choices):
        for value in outer.values():
            value["arms"] = [arm for arm in value["arms"] if arm["name"] != "H3b"]
    names = [arm["name"] for arm in next(iter(outer.values()))["arms"]]
    record = {
        "a": {"strongest_g1": "G1_best", "strongest_g2": "G2", "matched_g1": "G1_matched",
              "early_stop": stop_a, "fold_choices": first["splits"]},
        "b": {"champion": "H3b" if "H3b" in names else None, "qualified_folds": successful,
              "fold_choices": choices},
        "arms": [{"name": name} for name in names], "folds": outer, "seeds": contract()["seeds"],
        "selection_scope": "all choices fitted and selected only inside their corresponding outer fit split",
        "inner_summary_sha256": {s: sha256(ROOT / f"analysis/{s}-summary.json") for s in STAGES},
    }
    publish_json(ROOT / "final-lock.json", record)


def choose(stage, summary):
    if stage == "round1":
        select_round1(summary)
        return
    first = read_json(ROOT / "round1-decision.json")
    if stage in ("round2-block", "round2-opt"):
        select_optimization(stage, summary, first)
    else:
        final_lock(first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    choose(args.stage, measure(args.stage))


if __name__ == "__main__":
    main()
