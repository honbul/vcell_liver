import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from h3_pcgrad.evaluate import calculate
from h3_pcgrad.score import WeightedRanking, gate, weighted_association


def test_bootstrap_average_ties_equal_literal_lineage_replication():
    y = np.array([1, 3, 4, 1, 5, 4, 7, 2])
    p = np.array([3, 4, 1, 2, 2, 7, 6, 1])
    counts = np.array([[2, 0, 3, 1, 2, 3, 1, 2], [3, 4, 5, 6, 0, 1, 0, 3]])
    yr = WeightedRanking(y).ranks(counts)
    pr = WeightedRanking(p).ranks(counts)
    got = weighted_association(yr, pr, counts)
    for k, draws in enumerate(counts):
        draw = np.repeat(np.arange(len(draws)), draws)
        np.testing.assert_array_equal(yr[k, draw], rankdata(y[draw]))
        np.testing.assert_allclose(got[k], spearmanr(y[draw], p[draw]).statistic)


def config():
    return {
        "screens": ["CTRPv2", "GDSC2", "PRISM20Q2"], "bootstraps": 40, "bootstrap_seed": 81,
        "minimum_validation_lineages_per_drug": 5,
        "go": {"ndcg_delta": .01, "protection": -.01, "screen_floor": -.02, "improved_screens": 2},
    }


def test_exact_gates_for_tie_boundaries_and_screen_harm():
    cfg = config()
    points = {key: {"ndcg_at_5": val, "within_drug_spearman": -.02, "precision_at_k": -.02}
              for key, val in zip(["macro", *cfg["screens"]], [.01, .02, .01, 0])}
    ci = {"ndcg_at_5": [.00001, .03], "within_drug_spearman": [-.01, .03], "precision_at_k": [-.01, .03]}
    assert all(gate(points, ci, cfg).values())
    points["PRISM20Q2"]["precision_at_k"] = -.0200001
    assert not gate(points, ci, cfg)["no_severe_loss"]
    ci["ndcg_at_5"][0] = 0
    assert not gate(points, ci, cfg)["ndcg_ci_positive"]


def test_three_screens_one_scoring_contract_on_synthetic_data():
    rng = np.random.default_rng(1823)
    cfg = config()
    records = []
    for screen in cfg["screens"]:
        for row in range(18):
            for drug in range(7):
                if (row + drug) % 7 == 0:
                    continue
                value = rng.normal()
                records.append({"source_id": screen, "endpoint_id": "test", "fold": int(row >= 9),
                                "compound_id": f"d{drug}", "lineage_group": f"g{row}",
                                "observed": value, "G2__1": value, "G2__2": value * .1, "G2__3": value * 2,
                                "G1__1": value, "G1__2": value * .1, "G1__3": value * 2,
                                "structured": rng.normal(), "drug_mean": rng.normal(), "drug_ridge": rng.normal()})
    frame = pd.DataFrame(records)
    layout = {name: [f"{name}__{i}" for i in (1, 2, 3)] for name in ("G1", "G2")}
    layout |= {name: [name] for name in ("structured", "drug_mean", "drug_ridge")}
    cols = [key for members in layout.values() for key in members]
    lock = {"a": {"early_stop": False, "strongest_g2": "G2", "matched_g1": "G1", "strongest_g1": "G1"},
            "b": {"champion": None}}
    metrics, diffs, gates, decisions = calculate(frame, layout, cols, lock, cfg)
    assert len(metrics) == 4 * (2 * 4 + 3) * 6
    same = diffs.loc[diffs.baseline.eq("G1")]
    assert same[["delta", "lower", "upper"]].abs().max().max() == 0
    assert set(same.replicates) == {40}
    assert decisions["H3a"] == {"verdict": "NO-GO", "candidate": "G2", "stage": "outer"}
    assert decisions["H3b"]["stage"] == "inner"
    assert gates[0]["criteria"]["no_severe_loss"]


if __name__ == "__main__":
    test_bootstrap_average_ties_equal_literal_lineage_replication()
    test_exact_gates_for_tie_boundaries_and_screen_harm()
    test_three_screens_one_scoring_contract_on_synthetic_data()
    print("3 H3 score tests passed")
