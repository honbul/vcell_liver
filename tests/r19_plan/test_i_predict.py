"""Exercise the I-track fit/select, fallback and prediction freeze on synthetic responses."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.r19_plan.i_freeze import freeze_fold, freeze_run, verify_fold, verify_run
from src.r19_plan.i_predict import FitResidual, QueryInputs, ViewInputs, refit_kernels, select_kernels


def wells(parents, prefix):
    return pd.DataFrame({
        "study": "synthetic", "parent": parents,
        "inst_id": [f"{prefix}{i}" for i in range(len(parents))],
        "rna_plate": prefix, "rna_well": [f"A{i + 1:02d}" for i in range(len(parents))],
        "fold": 1, "role": "A",
    })


def example():
    parents = pd.Index([f"p{i}" for i in range(8)])
    numbers = np.arange(len(parents), dtype=float)
    profile = np.stack((numbers, numbers ** 2, np.cos(numbers)), axis=1)
    painting, prism = profile.copy(), profile.copy()
    painting[7] = np.nan
    prism[6:] = np.nan
    assays = pd.DataFrame({
        "parent": list(parents[:6]), "feature_key": [("target", "B")] * 6, "pchembl_value": numbers[:6]
    })
    annotations = assays[["parent", "feature_key"]].copy()
    inputs = ViewInputs(parents, np.eye(len(parents)) * .7 + .3, {"I2": painting, "I4": prism}, assays, annotations)
    inner = FitResidual(wells(["p2", "p1", "p0", "p1"], "fit"),
                        np.array([[1., 2.], [1., 3.], [2., 1.], [3., 2.]]))
    select = QueryInputs(wells(["p3", "p4"], "select"),
                         np.array([[.1, .2], [.4, .3]]), np.array([[.05, .1], [.2, .3]]))
    truth = np.array([[1.6, .2], [.4, 1.3]])
    outer = FitResidual(pd.concat((inner.rows, select.rows), ignore_index=True),
                        np.concatenate((inner.values, truth - select.anchor)))
    query = QueryInputs(wells(["p7", "p5", "p6", "p7"], "query"),
                        np.array([[1., 0.], [2., 1.], [1., .5], [.4, .7]]),
                        np.array([[.1, .2], [.3, .4], [.9, .2], [.2, .6]]))
    return inputs, inner, select, truth, outer, query


def test_matched_kernel_fit_keeps_anchor_film_and_exact_outer_fallback():
    inputs, inner, select, truth, outer, query = example()
    settings = select_kernels(inputs, inner, select, truth)
    result = refit_kernels(inputs, outer, query, settings)
    assert len(result) == 25
    np.testing.assert_array_equal(result["I3_all"].view_count, [0, 3, 1, 0])
    np.testing.assert_array_equal(result["I3"].full[[0, 3]], result["I0"].full[[0, 3]])
    for arm in result.values():
        np.testing.assert_allclose(arm.full - arm.kernel_only, query.film)
    assert not np.array_equal(result["I2_mask_only"].full[[0, 3]], result["I0"].full[[0, 3]])


def test_inner_and_outer_roles_fail_closed():
    inputs, inner, select, truth, outer, query = example()
    with pytest.raises(ValueError, match="query parent"):
        select_kernels(inputs, outer, select, truth)
    with pytest.raises(ValueError, match="finite"):
        select_kernels(inputs, inner, select, np.full_like(truth, np.nan))
    settings = select_kernels(inputs, inner, select, truth)
    settings["I1"] = replace(settings["I1"], gain=.9)
    with pytest.raises(ValueError, match="registered grid"):
        refit_kernels(inputs, outer, query, settings)


def test_freeze_checks_registered_axes_and_detects_changed_outputs(tmp_path):
    inputs, inner, select, truth, outer, query = example()
    settings = select_kernels(inputs, inner, select, truth)
    result = refit_kernels(inputs, outer, query, settings)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Locked synthetic I track\n")
    genes = pd.DataFrame({"pr_gene_id": [2, 3]})
    path = tmp_path / "frozen"
    digest = freeze_fold(path, query.rows, genes, result, query.anchor, settings, {"inputs": "f" * 64}, protocol)
    receipt = verify_fold(path, digest, query.rows, genes, protocol)
    assert receipt["views"]["I3"] == ["I1", "I2"]
    with np.load(path / "forecasts.npz", allow_pickle=False) as forecasts:
        np.testing.assert_array_equal(forecasts["B_R17"], query.anchor)
    with pytest.raises(ValueError, match="score gene order"):
        verify_fold(path, digest, query.rows, genes.iloc[::-1], protocol)
    with (path / "coverage.npz").open("ab") as file:
        file.write(b"modified")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_fold(path, digest, query.rows, genes, protocol)


def test_run_freeze_requires_all_rotations_of_each_disjoint_outer_fold(tmp_path):
    inputs, inner, select, truth, outer, query = example()
    settings = select_kernels(inputs, inner, select, truth)
    result = refit_kernels(inputs, outer, query, settings)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Locked nine slices\n")
    genes = pd.DataFrame({"pr_gene_id": [2, 3]})
    forecasts = tmp_path / "forecasts"
    manifest = tmp_path / "freeze.json"
    queries = {}
    for fold in range(3):
        for role in "ABC":
            rows = query.rows.assign(fold=fold, role=role, parent=query.rows.parent + str(fold),
                                     inst_id=query.rows.inst_id + str(fold), rna_plate=query.rows.rna_plate + str(fold))
            queries[fold, role] = rows
            freeze_fold(forecasts / f"f{fold}-{role}", rows, genes, result, query.anchor, settings,
                        {"inputs": "f" * 64}, protocol)
    with pytest.raises(ValueError, match="all nine"):
        freeze_run(manifest, forecasts, {key: value for key, value in queries.items() if key != (1, "C")},
                   genes, protocol)
    digest = freeze_run(manifest, forecasts, queries, genes, protocol)
    assert len(verify_run(manifest, digest, forecasts, queries, genes, protocol)["folds"]) == 9
    changed = dict(queries)
    changed[0, "B"] = changed[0, "B"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="share their physical query"):
        verify_run(manifest, digest, forecasts, changed, genes, protocol)
