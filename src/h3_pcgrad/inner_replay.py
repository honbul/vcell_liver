"""Verify the additive residual correction leaves registered model selection intact."""

import argparse

import pandas as pd

from vc_build.io import publish_json, sha256
from .data import ROOT, contract, read_json
from .score import ALL_METRICS, RESIDUAL, inner_scores
from .select import STAGES, best


def verify(stages, output):
    gates, scorebooks = {}, {}
    for stage in stages:
        book = read_json(ROOT / f"analysis/{stage}-summary.json")
        scorebooks[stage] = {}
        for split, scores in book["scores_by_split"].items():
            entry = ROOT / stage / split / "forecast.parquet"
            if sha256(entry) != book["inputs_sha256"][split]:
                raise ValueError("Cannot replay an unfrozen inner forecast")
            frame = pd.read_parquet(entry)
            updated = inner_scores(frame, list(scores))
            errors = [abs(updated[arm][metric] - value)
                      for arm, metrics in scores.items() if arm != "drug_mean"
                      for metric, value in metrics.items()]
            gap = max(errors)
            if gap > 1e-9:
                raise ValueError(f"Residual threshold changes a candidate in {stage}/{split}: {gap}")
            ranking_models = [name for name in scores if name not in ("structured", "drug_mean")]
            if best(updated, ranking_models) != best(scores, ranking_models):
                raise ValueError("The numerical correction changes the highest ranked candidate")
            if stage == "round1":
                for prefix in ("G1_", "G2_"):
                    family = [name for name in ranking_models if name.startswith(prefix)]
                    if best(updated, family) != best(scores, family):
                        raise ValueError("The numerical correction changes the matched comparison")
            scorebooks[stage][split] = updated
            additive = None
            if "drug_mean" in scores:
                block = frame.groupby(["source_id", "endpoint_id", "fold", "compound_id"]).drug_mean
                if not block.nunique().eq(1).all() or updated["drug_mean"][RESIDUAL] != 0:
                    raise ValueError("Drug mean does not have an exactly additive surface")
                additive = {"initial": scores["drug_mean"][RESIDUAL], "corrected": 0.0}
            gates[f"{stage}/{split}"] = {"non_drug_mean_max_delta": gap, "drug_mean_residual": additive}
    decisions_checked = False
    if tuple(stages) == STAGES:
        lock = read_json(ROOT / "final-lock.json")
        cfg = contract()["go"]
        for split, study in lock["b"]["fold_choices"].items():
            baseline = scorebooks["round1"][split]
            candidates = {name: row for stage in STAGES[1:] for name, row in scorebooks[stage][split].items()}
            for name, evidence in study["candidates"].items():
                gains = [{metric: candidates[name][metric] - baseline[ref][metric] for metric in ALL_METRICS}
                         for ref in study["inner_comparators"]]
                qualifies = all(delta["ndcg_at_5"] >= cfg["ndcg_delta"]
                                and min(delta["within_drug_spearman"], delta["precision_at_k"]) >= cfg["protection"]
                                and delta[RESIDUAL] > 0 for delta in gains)
                if qualifies != evidence["qualifying"]:
                    raise ValueError(f"Numerical threshold changes H3b eligibility: {split}/{name}")
        decisions_checked = True
    publish_json(output, {"only_inner_outcomes_loaded": True, "comparisons": gates,
                          "h3b_eligibility_verified": decisions_checked})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1-only", action="store_true")
    args = parser.parse_args()
    stages = STAGES[:1] if args.round1_only else STAGES
    receipt = "inner-round1-additive-replay.json" if args.round1_only else "inner-all-additive-replay.json"
    verify(stages, ROOT / "analysis" / receipt)
