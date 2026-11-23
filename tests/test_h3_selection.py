import tempfile
from pathlib import Path

from h3_pcgrad import select


def sample(ndcg):
    return {"ndcg_at_5": ndcg, "within_drug_spearman": .2, "precision_at_k": .3,
            "residual_spearman": .1, "within_drug_pearson": .2, "within_cell_spearman": .3}


def test_other_outer_folds_cannot_change_local_selection():
    arm_map = {arm.name: arm.record() for arm in select.first_arms()}
    tags = ["CTRPv2-0", "GDSC2-2"]
    scores = {name: sample(.73 + index / 100) for index, name in enumerate(arm_map)}
    summary = {
        "scores_by_split": {tag: dict(scores) for tag in tags},
        "settings_by_split": dict.fromkeys(tags, arm_map),
        "diagnostics": {"g1_frequency_by_split": dict.fromkeys(tags, .1)},
    }
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        original = select.ROOT
        try:
            select.ROOT = Path(first)
            select.select_round1(summary)
            stable = select.read_json(select.ROOT / "round2-block-lock.json")["splits"]
            summary["scores_by_split"][tags[0]]["G1_w0"] = sample(.99)
            summary["scores_by_split"][tags[0]]["G2_w0"] = sample(.99)
            select.ROOT = Path(second)
            select.select_round1(summary)
            changed = select.read_json(select.ROOT / "round2-block-lock.json")["splits"]
            assert stable[tags[0]] != changed[tags[0]]
            assert stable[tags[1]] == changed[tags[1]]
        finally:
            select.ROOT = original


if __name__ == "__main__":
    test_other_outer_folds_cannot_change_local_selection()
    print("Matched nested selection isolation passed")
