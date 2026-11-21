"""Exercise I-track expression extraction exclusively on synthetic read-only caches."""

from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(not all(find_spec(name) for name in ("h5py", "rdkit")),
                                reason="requires the I-track LINCS metadata environment")


@pytest.fixture
def indexed(tmp_path):
    from src.r11_chemical_probe.common import sha256, write_json
    from src.r19_plan import i_stage

    cache, vehicles, plan = (tmp_path / name for name in ("cache", "vehicles", "plan"))
    pin, out = tmp_path / "expected.json", tmp_path / "staged"
    for folder in (cache, vehicles, plan):
        folder.mkdir()
    slice_id = "f0-A"
    target = pd.DataFrame([
        dict(inst_id=f"t{i}", sig_id=f"s{i}", parent=f"p{i}", cell_id="HEPG2", hours="6", dose="10",
             rna_plate=plate, rna_well=f"A{i + 1:02d}")
        for i, plate in enumerate("QPQPQP")
    ])
    gene_path = cache / "landmarks.tsv"
    gene_path.write_text("pr_gene_id\tpr_gene_symbol\n3\tGENE3\n1\tGENE1\n")
    cache_rows = cache / "target_rows.tsv"
    target.assign(index=range(len(target))).to_csv(cache_rows, sep="\t", index=False)
    outcomes = np.arange(len(target) * 2, dtype=np.float32).reshape(len(target), 2)
    outcomes[[3, 5]] = np.nan
    np.save(cache / "target_wells.npy", outcomes)
    pd.DataFrame({"rna_plate": ["P", "Q"]}).to_csv(vehicles / "vehicles.tsv", sep="\t", index=False)
    controls = np.arange(12, dtype=np.float32).reshape(2, 3, 2) * 2
    controls[1, 1] = np.nan
    np.save(vehicles / "vehicles.npy", controls)
    role_sha = "a" * 64
    write_json(cache / "receipt.json", dict(study=i_stage.GEO, target_wells=len(target), files={
        path.name: sha256(path) for path in (gene_path, cache_rows, cache / "target_wells.npy")}))
    write_json(vehicles / "receipt.json", dict(audit_sha256=role_sha, files={
        name: sha256(vehicles / name) for name in ("vehicles.tsv", "vehicles.npy")}))

    selected = target.assign(cache_index=range(len(target)), study=i_stage.GEO)
    fit = selected.iloc[[4, 1]].assign(inner_phase=["inner_fit", "select"], outer_fold=[2, 1])
    labels = selected.iloc[[2, 0]].assign(outer_fold=0, scaffold_status=["acyclic", "seen"])
    query = labels.drop(columns="cache_index")
    role_rows = pd.DataFrame([
        dict(inst_id=f"{plate}{role}{i}", rna_plate=plate, rna_well=f"{role}B{i:02d}",
             study=i_stage.GEO, cache_plate_index="PQ".index(plate), cache_role_index="ABC".index(role), role=role)
        for role, plate, i in (("A", "Q", 1), ("A", "Q", 2), ("A", "P", 1),
                               ("B", "P", 1), ("C", "Q", 1), ("C", "Q", 2))
    ])
    plans = {
        "predict": {"inner_and_outer_labels": fit, "query_features": query,
                    "fit_controls": role_rows.iloc[:3], "select_controls": role_rows.iloc[[3]]},
        "score": {"score_labels": labels, "score_controls": role_rows.iloc[4:]},
    }

    def pin_plan():
        manifests = {}
        for phase, tables in plans.items():
            folder = plan / (phase + "-plan") / slice_id
            folder.mkdir(parents=True, exist_ok=True)
            for name, rows in tables.items():
                rows.to_csv(folder / (name + ".tsv"), sep="\t", index=False)
            manifests[phase] = {file.name: sha256(file) for file in folder.iterdir()}
        (plan / "receipt.json").unlink(missing_ok=True)
        write_json(plan / "receipt.json", dict(
            metadata_only=True, expression_read=False, fitted=False, R18_protected_input_overlap=0,
            slices=9, files={key: manifests if key == slice_id else {} for key in i_stage.SLICES},
            cache_metadata_sha256=sha256(cache / "receipt.json"), role_metadata_sha256=role_sha,
            control_cache_metadata_sha256=sha256(vehicles / "receipt.json"), landmarks_sha256=sha256(gene_path),
        ))
        pin.write_bytes((plan / "receipt.json").read_bytes())

    pin_plan()
    return i_stage, (cache, vehicles, plan, pin, slice_id), out, plans, pin_plan, outcomes, controls


def recorded_reads(module, observed):
    load = np.load
    digest = module.sha256

    class Logged:
        def __init__(self, file, data):
            self.file, self.data = file, data
            self.shape, self.dtype = data.shape, data.dtype

        def __getitem__(self, index):
            picks = index if isinstance(index, tuple) else (index,)
            observed.append((self.file.name, tuple(map(tuple, picks))))
            return self.data[index]

    def trace_load(file, *args, **kwargs):
        value = load(file, *args, **kwargs)
        if isinstance(file, Path) and file.suffix == ".npy":
            assert isinstance(value, np.memmap) and value.mode == "r"
            return Logged(file, value)
        return value

    def trace_digest(file):
        assert Path(file).suffix != ".npy"
        return digest(file)

    return patch.object(module, "sha256", trace_digest), patch.object(module.np, "load", trace_load)


def test_predictor_exact_reads_and_separate_bundle(indexed):
    module, paths, out, _, _, outcomes, controls = indexed
    _, _, plan, pin, slice_id = paths
    # A predictor stage must work with no access to the score files.
    (plan / "score-plan").rename(plan / "hidden")
    observed = []
    hashes, loads = recorded_reads(module, observed)
    with hashes, loads:
        stamp = module.stage_prediction(*paths, out)
        bundle = module.read_prediction_bundle(out, pin, slice_id)
    assert bundle.input_receipts["stage"] == stamp
    assert tuple(bundle.landmarks.pr_gene_id) == (3, 1)
    assert set(bundle.arrays) == set(module.ARRAYS)
    assert list(bundle.tables["query_features"].inst_id) == ["t2", "t0"]
    assert "cache_index" not in bundle.tables["query_features"]
    np.testing.assert_array_equal(bundle.arrays["inner_and_outer_labels"], outcomes[[4, 1]])
    np.testing.assert_array_equal(bundle.arrays["fit_controls"], controls[[1, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(bundle.arrays["select_controls"], controls[[0], [1]])
    assert observed == [("target_wells.npy", ((4, 1),)),
                        ("vehicles.npy", ((1, 0), (0, 0))), ("vehicles.npy", ((0,), (1,)))]
    with (out / "expression.npz").open("ab") as file:
        file.write(b"modified")
    with pytest.raises(ValueError, match="Changed registered"):
        module.read_prediction_bundle(out, pin, slice_id)


def test_score_loads_only_held_treatments_and_score_controls(indexed):
    module, paths, _, _, _, outcomes, controls = indexed
    query = module.read_query_metadata(*paths[2:])
    genes = pd.DataFrame({"pr_gene_id": [3, 1], "pr_gene_symbol": ["GENE3", "GENE1"]})
    observed = []
    hashes, loads = recorded_reads(module, observed)
    with hashes, loads:
        result = module.load_score_delta(*paths, query, genes)
    assert observed == [("target_wells.npy", ((2, 0),)), ("vehicles.npy", ((1,), (2,)))]
    np.testing.assert_array_equal(result, outcomes[[2, 0]] - controls[1, 2])
    with patch.object(module.np, "load", side_effect=AssertionError("opened expression before score-axis check")):
        with pytest.raises(ValueError, match="registered query axis"):
            module.load_score_delta(*paths, query.iloc[::-1], genes)
        with pytest.raises(ValueError, match="frozen gene axis"):
            module.load_score_delta(*paths, query, genes.iloc[::-1])


def test_changed_source_roles_and_labels_fail_before_opening_expression(indexed):
    module, paths, _, plans, pin_plan, _, _ = indexed
    with patch.object(module.np, "load", side_effect=AssertionError("opened expression before metadata check")):
        original = plans["predict"]["fit_controls"]
        plans["predict"]["fit_controls"] = original.assign(cache_role_index=2)
        pin_plan()
        with pytest.raises(ValueError, match="DMSO index"):
            module.read_prediction_inputs(*paths)
        plans["predict"]["fit_controls"] = original
        plans["predict"]["inner_and_outer_labels"] = plans["predict"]["inner_and_outer_labels"].assign(
            cache_index=[2, 1])
        pin_plan()
        with pytest.raises(ValueError, match="locked treatment index"):
            module.read_prediction_inputs(*paths)
        plans["predict"]["query_features"] = plans["predict"]["query_features"].assign(cache_index=0)
        pin_plan()
        with pytest.raises(ValueError, match="only registered metadata"):
            module.read_prediction_inputs(*paths)
