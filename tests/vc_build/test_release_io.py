import json

import numpy as np
import pandas as pd
import pytest

from vc_build.calibration import absolute_error_quantile
from vc_build.evaluate import join_observations, rank_metrics, score
from vc_build.io import freeze_table, verify_forecast
from vc_build.predict import dose_uM, load_request


def test_freeze_rejects_overwrite_outcomes_and_tampering(tmp_path):
    data = pd.DataFrame([dict(row_id="01", track="C", fold=0, method="mean", predicted_value=0.2)])
    path = tmp_path / "predictions.tsv"
    written = freeze_table(data, path, provenance={"split_sha256": "123"})
    loaded, verified = verify_forecast(path)
    assert written == verified
    assert loaded.loc[0, "row_id"] == "01"
    with pytest.raises(FileExistsError):
        freeze_table(data, path, provenance={})
    with pytest.raises(ValueError, match="observed outcomes"):
        freeze_table(data.assign(endpoint_value=0), tmp_path / "unsealed.tsv", provenance={})
    path.write_text(path.read_text().replace("0.2", "0.3"))
    with pytest.raises(ValueError, match="freeze receipt"):
        verify_forecast(path)


def test_score_common_support_and_small_candidate_pools():
    forecasts = pd.DataFrame([
        dict(row_id=row, track="C", fold=0, cell_id="cell", lineage_group="family",
             method=method, compound_id=row, predicted_value=value)
        for row, a, b in [("0", 0., 0.), ("1", 1., 2.), ("2", 2., np.nan)]
        for method, value in [("full", a), ("partial", b)]
    ])
    observations = pd.DataFrame(dict(row_id=["0", "1", "2"], endpoint_value=[0., 1., 3.]))
    joined = join_observations(forecasts, observations)
    summary, cells = score(joined, endpoint_direction="higher")
    assert all(method["common_rows"] == 2 for method in summary["scores"])
    assert cells.set_index("method").loc["full", "mae"] == 0.
    assert rank_metrics([1, 2, 3], [1, 2, 3])["ndcg_at_5"] == 1.
    assert rank_metrics([1, 2, 3], [1, 2, 3])["top5_overlap"] == 1.


def test_concentration_duplicate_genes_and_small_calibration(tmp_path):
    assert dose_uM(7000, "nM") == 7
    assert dose_uM(0, "µM") == 0
    for value, unit in [(-1, "uM"), (1, "mg"), (float("inf"), "M"), (True, "uM")]:
        with pytest.raises(ValueError):
            dose_uM(value, unit)
    request = tmp_path / "query.json"
    request.write_text('{"genes": {"TP53": 1, "TP53": 2}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_request(request)
    with pytest.raises(ValueError, match="Insufficient independent"):
        absolute_error_quantile([1], [2])
    assert absolute_error_quantile(np.arange(10), np.arange(10) + 1) == 1
