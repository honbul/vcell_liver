"""Verify the additive residual correction leaves registered model selection intact."""

import argparse

import pandas as pd

from vc_build.io import publish_json, sha256
from .data import ROOT, read_json
from .score import RESIDUAL, inner_scores
from .select import STAGES


def verify(stages, output):
    gates = {}
    for stage in stages:
        book = read_json(ROOT / f"analysis/{stage}-summary.json")
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
            if gap > 1e-12:
                raise ValueError(f"Residual threshold changes a candidate in {stage}/{split}: {gap}")
            additive = None
            if "drug_mean" in scores:
                block = frame.groupby(["source_id", "endpoint_id", "fold", "compound_id"]).drug_mean
                if not block.nunique().eq(1).all() or updated["drug_mean"][RESIDUAL] != 0:
                    raise ValueError("Drug mean does not have an exactly additive surface")
                additive = {"initial": scores["drug_mean"][RESIDUAL], "corrected": 0.0}
            gates[f"{stage}/{split}"] = {"non_drug_mean_max_delta": gap, "drug_mean_residual": additive}
    publish_json(output, {"only_inner_outcomes_loaded": True, "comparisons": gates})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1-only", action="store_true")
    args = parser.parse_args()
    stages = STAGES[:1] if args.round1_only else STAGES
    receipt = "inner-round1-additive-replay.json" if args.round1_only else "inner-all-additive-replay.json"
    verify(stages, ROOT / "analysis" / receipt)
