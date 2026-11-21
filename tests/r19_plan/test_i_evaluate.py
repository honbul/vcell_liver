"""Exercise the I-track score handoff with nine synthetic frozen shards."""

import json

import numpy as np
import pandas as pd
import pytest

from src.r11_chemical_probe.common import sha256
from src.r19_plan.i_evaluate import evaluate_i_run
from src.r19_plan.i_freeze import freeze_fold, freeze_run
from src.r19_plan.i_predict import KernelChoice, KernelForecast


def frozen_example(tmp_path, *, changed_registry=False, wrong_inventory=False):
    protocol = tmp_path / "i.md"
    protocol.write_text("Synthetic I-track score handoff\n")
    path = tmp_path / "forecasts"
    genes = pd.DataFrame({"pr_gene_id": [4, 5]})
    queries, truth = {}, {}
    for fold in range(3):
        for rotation, role in enumerate("ABC"):
            split = f"f{fold}-{role}"
            rows = pd.DataFrame({
                "study": "synthetic", "inst_id": [f"p{fold}-a", f"p{fold}-b"],
                "rna_plate": f"plate{fold}", "rna_well": ["A01", "A02"],
                "parent": [f"parent{fold}a", f"parent{fold}b"], "fold": fold, "role": role,
                "scaffold_status": ["seen", "unseen"],
            })
            queries[fold, role] = rows
            observed = np.repeat([[1 + fold + rotation], [5 + fold + rotation]], 2, axis=1).astype(np.float32)
            truth[split] = observed
            zero = np.zeros_like(observed)
            full = np.stack((observed[0], zero[1]))
            kernel = np.stack((observed[0] - .5, zero[1]))
            optional_view = "I2" if changed_registry and split == "f2-C" else "I1"
            arms = {
                "I0": KernelForecast(zero, zero, np.array([0, 0]), ()),
                "I1": KernelForecast(full, kernel, np.array([1, 0]), (optional_view,)),
            }
            choices = {arm: KernelChoice(.1, .5, .8) for arm in arms}
            freeze_fold(path / split, rows, genes, arms, zero + .25, choices,
                        {"inputs": "a" * 64}, protocol)
    if wrong_inventory:
        last = path / "f2-C"
        artifact = last / "kernel_only.npz"
        np.savez_compressed(artifact, I0=np.zeros((2, 2)))
        receipt = last / "receipt.json"
        stamp = json.loads(receipt.read_text())
        stamp["files"][artifact.name] = sha256(artifact)
        receipt.write_text(json.dumps(stamp))
    manifest = tmp_path / "freeze.json"
    digest = freeze_run(manifest, path, queries, genes, protocol)
    fit = {fold: {p for (other, _), row in queries.items() if other != fold for p in row.parent}
           for fold in range(3)}
    return (manifest, digest, path, queries, genes, protocol), truth, fit


def no_read(*args):
    pytest.fail("held-score delta accessed before all frozen slices were verified")


def test_i_evaluate_scores_verified_nine_slices_in_the_registered_order(tmp_path):
    args, truth, fit = frozen_example(tmp_path)
    queries = args[3]
    calls = []

    def read_score_delta(slice_id, expected_rows):
        fold, role = int(slice_id[1]), slice_id[-1]
        pd.testing.assert_frame_equal(expected_rows, queries[fold, role])
        assert expected_rows is not queries[fold, role]
        calls.append(slice_id)
        return truth[slice_id]

    results = evaluate_i_run(*args, read_score_delta=read_score_delta, fit_parents_by_fold=fit)
    assert calls == list(truth)
    full = results["full_fallback"].set_index(["arm", "reference", "metric"])
    i1 = full.loc["I1", "I0", "rmse"]
    assert (i1.parents, i1.rows, i1.physical_wells, i1.roles) == (6, 18, 6, 3)
    assert (i1.arm_value, i1.reference_value) == pytest.approx((3.5, 5.))
    matched = results["same_support"].set_index(["arm", "reference", "metric"])
    assert matched.loc["I1", "I0", "rmse"].reference_value == pytest.approx(3.)
    kernel = results["kernel_only"].set_index(["scope", "arm", "reference", "metric"])
    assert kernel.loc["complete_shared", "I1", "I0", "rmse"].arm_value == pytest.approx(.5)


def test_corrupt_final_slice_and_reordered_genes_fail_before_any_score_read(tmp_path):
    args, _, fit = frozen_example(tmp_path)
    wrong_gene_order = (*args[:4], args[4].iloc[::-1], args[5])
    with pytest.raises(ValueError, match="gene order"):
        evaluate_i_run(*wrong_gene_order, read_score_delta=no_read, fit_parents_by_fold=fit)
    with (args[2] / "f2-C" / "coverage.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        evaluate_i_run(*args, read_score_delta=no_read, fit_parents_by_fold=fit)


@pytest.mark.parametrize(("option", "error"), [
    ("changed_registry", "arm/view registry differs across slices"),
    ("wrong_inventory", "Frozen kernel_only.npz arm inventory differs"),
])
def test_even_frozen_registry_and_archive_disagreements_fail_before_score_reads(tmp_path, option, error):
    args, _, fit = frozen_example(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=error):
        evaluate_i_run(*args, read_score_delta=no_read, fit_parents_by_fold=fit)


def test_holdout_against_fit_registry_before_score_reader_is_called(tmp_path):
    args, _, fit = frozen_example(tmp_path)
    fit[2].add("parent2a")
    with pytest.raises(ValueError, match="crossed into the fit registry"):
        evaluate_i_run(*args, read_score_delta=no_read, fit_parents_by_fold=fit)
