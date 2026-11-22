from argparse import Namespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan import d1_inputs, d12_predict, d12_workflow
from src.r19_plan.d1_access import CONTROL_KEYS, SLICES, register_slice
from src.r19_plan.i_preflight import write_plan
from src.r19_plan.i_stage import stage_prediction
from src.r19_plan.r18_input_bundle import EXPRESSION, GENE_IDS, SAMPLE_IDS
from src.r19_plan.r18_matrix_index import metadata_index
from test_i_stage import indexed


def test_stage_reads_only_balanced_fit_select_and_keeps_held_score_separate(indexed, tmp_path):
    _, paths, original, plans, _, outcomes, _ = indexed
    cache, vehicles, plan, reference, slice_id = paths
    stage_prediction(*paths, original)
    fit, select, score = (plans["predict"]["fit_controls"], plans["predict"]["select_controls"],
                          plans["score"]["score_controls"])
    chosen = pd.concat((fit.iloc[[0, 2]], select, score.iloc[:1]), ignore_index=True)[CONTROL_KEYS]
    sample_ids = [*chosen.inst_id, "not-admitted"]
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    matrix = tmp_path / "level3.gctx"
    with h5py.File(matrix, "w") as hdf:
        hdf[SAMPLE_IDS], hdf[GENE_IDS], hdf[EXPRESSION] = np.asarray(sample_ids, dtype="S"), [1, 2, 3], values
    indexed_controls, axes, record = metadata_index(matrix, chosen, np.array([3, 1]))
    tables, _ = register_slice(plan, reference, slice_id, indexed_controls)
    access, out = tmp_path / "balanced", tmp_path / "d1-input"
    files = {phase: write_plan(access / (phase + "-plan") / slice_id,
                              {name + ".tsv": frame for name, frame in inventory.items()})
             for phase, inventory in tables.items()}
    axes.to_csv(access / "gene_indices.tsv", sep="\t", index=False)
    freeze = access / "receipt.json"
    write_json(freeze, dict(
        metadata_only=True, expression_read=False, source_access_sha256=sha256(reference),
        landmarks_sha256=sha256(cache / "landmarks.tsv"), matrix=record,
        slice_files={sid: files if sid == slice_id else {} for sid in SLICES},
        files={"gene_indices.tsv": sha256(access / "gene_indices.tsv")},
    ))

    reads = []
    original_extract = d1_inputs.extract_matrix

    def observe(record, wanted, axis, genes):
        reads.extend(wanted.inst_id)
        return original_extract(record, wanted, axis, genes)

    (access / "score-plan").rename(access / "withheld")
    with patch.object(d1_inputs, "extract_matrix", side_effect=observe):
        d1_inputs.stage(Namespace(access=access, reference=freeze, approved_bundle=original,
                                  approved_reference=reference, slice=slice_id, out=out))
    reopened = d1_inputs.read_bundle(out, freeze, slice_id)
    assert reads == sample_ids[:3]
    np.testing.assert_array_equal(reopened.arrays["fit_controls"], values[[0, 1]][:, [2, 0]])
    np.testing.assert_array_equal(reopened.arrays["select_controls"], values[[2]][:, [2, 0]])
    assert "score_labels" not in reopened.arrays
    (access / "withheld").rename(access / "score-plan")
    chosen_query = pd.read_csv(original / "query_features.tsv", sep="\t", dtype=str, keep_default_na=False)
    args = Namespace(cache=cache, vehicles=vehicles, access=access, original_plan=plan)
    truth = d12_workflow.score_truth(args, read_json(reference), read_json(freeze), {slice_id: chosen_query},
                                    reopened.landmarks)
    np.testing.assert_array_equal(truth, outcomes[[2, 0]] - values[3, [2, 0]])


def test_predicts_shared_and_independent_using_one_bank_and_outer_refit(monkeypatch):
    rows = pd.DataFrame({"parent": ["p"], "study": ["G"], "inst_id": ["w"]})
    split = tuple(rows.assign(parent=tag, inst_id=tag) for tag in ("fit", "select", "training", "held"))
    bank = Namespace(fit="A", select="B", landmarks=pd.DataFrame({"pr_gene_id": [1, 2]}), split=lambda: split,
                     response=lambda rows, role: np.full((len(rows), 2), 4. if role == "A" else 1.))
    both_zero = np.zeros((1, 2))
    monkeypatch.setattr(d12_predict, "position", lambda b, f, q, r: (both_zero, both_zero, both_zero))
    parameters = (.001, 5)
    monkeypatch.setattr(d12_predict, "checkpoint_predictions", lambda *args: {parameters: np.full((1, 2), 4.)})
    with patch.object(d12_predict, "fit_known", return_value=np.full((1, 2), 2.)) as fit:
        forecasts, selection = d12_predict.fit_policies(Namespace(model="FiLM_R12"), bank)
    assert fit.call_count == 1
    assert selection["D1_shared"]["gain"] == 1.
    assert selection["D1_independent"]["gain"] == .25
    assert selection["D1_shared"]["vehicle_cosine"] == pytest.approx(1.)
    np.testing.assert_array_equal(forecasts["D1_shared"], [[2., 2.]])
    np.testing.assert_array_equal(forecasts["D1_independent"], [[.5, .5]])
    np.testing.assert_array_equal(forecasts["D1_independent"], forecasts["D2_joint"])


def test_score_refuses_wrong_freeze_hash_before_expression(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    args = Namespace(action="score", freeze=freeze, out=tmp_path / "out", expected_sha256="0" * 64)
    with patch.object(d12_workflow, "score_truth", side_effect=AssertionError("score opened")):
        with pytest.raises(ValueError, match="freeze digest"):
            d12_workflow.run(args)


def test_adapted_models_unwrap_checkpoint_arrays_and_ignore_resource_tuple(monkeypatch, tmp_path):
    params = d12_predict.EPOCHS["chemCPA_P0"]
    observations = np.ones((1, 2))
    monkeypatch.setattr(d12_predict, "config", lambda _: {})
    monkeypatch.setattr(d12_predict, "FitSource", lambda *_a, **_k: None)
    monkeypatch.setattr(d12_predict, "Fit", lambda *_a, **_k: Namespace(query=Namespace(treated=None)))

    def upstream(_name):
        return lambda fit, work, epochs, cb: ({epoch: cb(observations * epoch) for epoch in epochs}, {})

    monkeypatch.setattr(d12_predict, "upstream", upstream)
    bank = Namespace(fit="A", landmarks=pd.DataFrame({"pr_gene_id": [1, 2]}))
    args = Namespace(model="chemCPA_P0", assets=tmp_path, config=tmp_path)
    predictions = d12_predict.checkpoint_predictions(args, bank, None, None, observations, observations, tmp_path)
    assert set(predictions) == {(epoch,) for epoch in params}
    np.testing.assert_array_equal(predictions[(params[-1],)], observations * params[-1])
