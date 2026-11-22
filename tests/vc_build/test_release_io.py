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


def test_bulk_counts_gene_lengths_use_mapped_symbols_and_keep_missing(tmp_path):
    from vc_build.v0plus_cli import read_expression

    counts = tmp_path / "counts.tsv"
    counts.write_text("sample_id\tENSG1\tENSG2\tENSG3\ntumor\t50\t100\t10000\n")
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("input_id\tgene_symbol\nENSG1\tA\nENSG2\tB\nENSG3\tC\n")
    lengths = tmp_path / "lengths.tsv"
    lengths.write_text("gene\tlength_bp\nA\t1000\nB\t4000\n")

    cpm = read_expression(counts, "bulk_counts", mapping).loc["tumor"]
    adjusted = read_expression(counts, "bulk_counts", mapping, lengths).loc["tumor"]
    assert cpm["A"] < cpm["B"] < cpm["C"]
    assert 2 ** adjusted["A"] - 1 == pytest.approx(1e6 * 2 / 3)
    assert 2 ** adjusted["B"] - 1 == pytest.approx(1e6 / 3)
    assert adjusted.isna().to_dict() == {"A": False, "B": False, "C": True}
    with pytest.raises(ValueError, match="requires --scale bulk_counts"):
        read_expression(counts, "bulk_tpm", mapping, lengths)
    lengths.write_text("gene\tlength_bp\nA\t0\nB\t4000\n")
    with pytest.raises(ValueError, match="finite positive"):
        read_expression(counts, "bulk_counts", mapping, lengths)


def test_pdx_repeat_distinguishes_reported_arm_weights_from_missing_values():
    from vc_build.shuen_pdx import repeat_agreement

    columns = ["model_id", "tc", "control", "treatment"]
    left = pd.DataFrame([
        ("A", 0.2, 1.0, 0.2),
        ("B", 0.3, None, None),
        ("C", 0.4, None, None),
    ], columns=columns)
    right = pd.DataFrame([
        ("A", 0.2, 1.0, 0.2),
        ("B", 0.3, None, None),
        ("C", 0.5, None, None),
    ], columns=columns)
    results = repeat_agreement(left, right, {"A", "B"})
    assert results["all"]["identical_tc"] == 2
    assert results["all"]["identical_reported_arm_weights"] == 1
    assert results["nonidentical"]["models"] == 1
