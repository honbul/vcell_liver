"""Test the archived scale-only contrast diagnostic without fitting a model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.r19_plan import t0_archive


def test_scale_ceiling_includes_network_direction_and_zero_bounds():
    direction = pd.DataFrame(dict(
        minimum_cells=20, cohort="test", stratum="liver", arm=["positive", "negative", "zero"],
        pooled_cosine=[.3, -.4, 0.], prediction_to_observed_rms=[2., .5, 0.],
    ))
    points = direction[t0_archive.KEY].assign(sst=10., sse=[38., 16.5, 10.], r2_pooled=[-2.8, -.65, 0.])
    result = t0_archive.decompose(points, direction).set_index("arm").loc[direction.arm]
    np.testing.assert_allclose(result.diagnostic_nonnegative_optimum, [.15, 0., 0.])
    np.testing.assert_allclose(result.diagnostic_fixed_direction_r2_ceiling, [.09, 0., 0.])
    np.testing.assert_allclose(result.r2_at_equal_norm, [-.4, -1.8, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match="different support"):
        t0_archive.decompose(points, direction.iloc[:-1])
    with pytest.raises(ValueError, match="disagree"):
        t0_archive.decompose(points.assign(sse=10.), direction)


def test_original_r15_freeze_matches_archive_and_all_rows(tmp_path):
    archive = Path(__file__).resolve().parents[2] / "reports" / "r15_lincs" / "data" / "external"
    out = tmp_path / "score-only"
    t0_archive.run(archive, out)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["archived_support_rows"] == 224 and receipt["maximum_absolute_reconstruction_error"] < 1e-12
    assert receipt["score_only"] and not receipt["training"] and not receipt["treatment_arrays_read"]
    gaps = pd.read_csv(out / "baseline_gaps.tsv", sep="\t")
    case = gaps.query(
        "minimum_cells == 20 and cohort == 'LINCS-supported' and stratum == 'liver' "
        "and arm == 'R15-B' and reference == 'B_pos-all'"
    ).iloc[0]
    assert case.parent_groups == 108
    assert case.scale_ceiling_minus_reference_raw_r2 == pytest.approx(-.009052006187839759)
