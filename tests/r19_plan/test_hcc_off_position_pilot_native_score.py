import numpy as np

from src.r19_plan.hcc_off_position_pilot_native_score import ARMS, compare_metrics


def test_public_reference_is_reselected_after_each_culture_draw():
    centers = (np.array([.2, .3, .4, .5, .6, .7]), np.array([.5, .4, .3, .2, .1, .0]))
    samples = (
        np.array([[.2, .3, .4, .5, .6, .7], [.2, .4, .25, .5, .6, .7]]),
        np.array([[.5, .4, .3, .2, .1, .0], [.5, .1, .35, .2, .0, .3], [np.nan]*6]),
    )

    rows, wins = compare_metrics(centers, samples, {"resampling": "parent_and_culture"})

    best = [row for row in rows if row["reference"] == "best_VC"]
    assert [row["best_at_point"] for row in best] == [ARMS[1], ARMS[1]]
    assert all(row["valid_draws"] == 2 for row in best)
    assert all(row["wins"] == 1 for row in wins if row["model"] in ARMS[1:3])


def test_degenerate_r2_and_perfect_reference_have_no_invalid_relative_interval():
    center = (np.zeros(6), np.full(6, np.nan))
    samples = (np.zeros((2, 6)), np.full((2, 6), np.nan))

    rows, _ = compare_metrics(center, samples, {"resampling": "parent"})

    assert all(np.isnan(r["relative_gain"]) for r in rows)
    assert all(r["valid_draws"] == 0 for r in rows if r["metric"] == "pooled_R2")
