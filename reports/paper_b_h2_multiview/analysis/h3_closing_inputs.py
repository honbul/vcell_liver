"""Read-only inputs and provenance for the H3-v2 closing audit."""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h2.assets import COLD, GDSC, KEY, LIMORE, SUPPLEMENT, parquet_lock
from nextgen_h2.run import META, TRAIN
from nextgen_h2.score import adult_licob, external_labels
from vc_build.io import publish_json, sha256

BASE = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
HOME = BASE / "paper_b_h2_posthoc"
H3 = BASE / "nextgen_v2/h3_v2"
E5 = BASE / "followup_20261124/h2_e5_upgrade"
E1 = BASE / "followup_20261124/h2_e1e3"
ASSETS = BASE / "nextgen_h123/h2_multiview/assets-v3"
INPUTS = {}
H3_SCORE = "55e9731925dd8f83a1b51aeb01d9831bd8d0c9540a590a5bc1ddb826ea7f10c4"
E5_SCORE = "93c2c4a30e1c606cae7297925999613fd5c4787b28b69e13f78b1b014d3da73c"
ROSTER = "44b53f6506360786a279f5b38d4dd12d648fc85063f0a74748b1b27ab24b1d52"


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") not in {"cpu", "cpu-c3"}:
        raise RuntimeError("Scoring is restricted to the commissioned G1 CPU Slurm jobs")


def pin(path, expected=None):
    path = Path(path)
    result = INPUTS.get(str(path))
    if result is None:
        result = sha256(path)
        INPUTS[str(path)] = result
    if expected is not None and result != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {result} != {expected}")
    return result


def read(path, expected=None):
    pin(path, expected)
    return json.loads(path.read_text())


def read_frame(path, expected=None, **kwargs):
    pin(path, expected)
    return pd.read_parquet(path, **kwargs)


def pin_modules(names):
    for name in names:
        pin(Path(importlib.import_module(name).__file__))


def align(left, right, keys=META):
    """Restore the left row order and enforce full identity, including endpoint and lineage."""
    if not left.row_id.is_unique or not right.row_id.is_unique or len(left) != len(right):
        raise ValueError("Forecasts lack one-to-one row identity")
    if np.array_equal(left.row_id.to_numpy(), right.row_id.to_numpy()):
        aligned = right.reset_index(drop=True)
    else:
        aligned = right.set_index("row_id", drop=False).loc[left.row_id].reset_index(drop=True)
    for key in keys:
        if not np.array_equal(left[key].to_numpy(), aligned[key].to_numpy()):
            raise ValueError(f"Exact row identity differs in {key}")
    return aligned


def external_inputs():
    asset = read(ASSETS / "data_receipt.json")
    cold = read(COLD / "receipt.json")
    lim = read(LIMORE / "receipt.json")["artifacts"]
    for name in ("test_inputs/rows.parquet", "test_labels/outcomes.parquet"):
        pin(LIMORE / name, lim[name]["sha256"])
    pin(COLD / "query.parquet", cold["files"]["query.parquet"])
    pin(SUPPLEMENT, cold["workbook_sha256"])
    pin(GDSC, asset["gdsc"]["raw_sha256"])
    pin(ASSETS / "gdsc_query.parquet", asset["files"]["gdsc_query.parquet"])
    return external_labels(ASSETS, asset)


def h3_metadata():
    score = read(H3 / "score/score_receipt.json", H3_SCORE)
    assert score["score_passes"] == 1
    read(H3 / "score/consumed.json", score["claim_sha256"])
    manifest = read(H3 / "splits/manifest.json", score["source_manifest_sha256"])
    lock = read(H3 / "locks/frozen.json", score["locked_operating_points_sha256"])
    assert lock["outer_labels_opened"] is False
    for name in ("final_lock.json", "verification.json", "RESULTS.md", "inner_selection_ledger.json"):
        pin(H3 / "reports-final-delivery" / name)
    return score, manifest, lock


def h3_frame(split, score, manifest, lock, external=None):
    from h3v2.grid import FAMILIES, SEEDS

    directory = H3 / "final" / split
    anchors = score["forecasts"][split]
    ref = read(directory / "references/receipt.json")
    assert ref["forecast_sha256"] == anchors["reference"]
    assert ref["lock_sha256"] == score["locked_operating_points_sha256"] and not ref["outer_labels_loaded"]
    reference = read_frame(directory / "references/forecast.parquet", anchors["reference"])
    assert len(reference) == ref["rows"] and reference.row_id.is_unique
    predictors = {}
    for family in FAMILIES:
        for seed in SEEDS:
            name = f"{family}__{seed}"
            group = directory / family / str(seed)
            receipt = read(group / "receipt.json")
            assert receipt["forecast_sha256"] == anchors["forecasts"][name]
            assert (receipt["split"], receipt["family"], receipt["seed"]) == (split, family, seed)
            assert not receipt["outer_labels_loaded"] and receipt["lock_sha256"] == ref["lock_sha256"]
            pred = read_frame(group / "forecast.parquet", anchors["forecasts"][name])
            predictors[name] = align(reference, pred).predicted.to_numpy()
    state = lock["external"] if split == "external" else lock["splits"][split]
    predictors.update({f"Champion__{s}": predictors[f'{state["champion"]}__{s}'] for s in SEEDS})
    frame = reference.assign(**predictors)
    frame["Champion-ensemble"] = frame[[f"Champion__{s}" for s in SEEDS]].mean(axis=1)
    if split == "external":
        labels = external
        assert labels.row_id.is_unique and set(labels.row_id) == set(frame.row_id)
    else:
        support = manifest["split_support"][split]
        query = read_frame(H3 / "splits" / split / "query.parquet", support["query_sha256"], columns=META)
        align(reference, query)
        labels = align(reference, read_frame(H3 / "outcomes" / f"{split}.parquet", support["label_sha256"]))
    out = frame.merge(labels[["row_id", "truth"]], on="row_id", validate="one_to_one")
    assert np.isfinite(out.truth).all()
    return out


def registry():
    asset = read(ASSETS / "data_receipt.json")
    return read_frame(ASSETS / "folds.parquet", asset["files"]["folds.parquet"]).scaffold_block


def compare(actual, path, keys, floats, expected_sha):
    pin(path, expected_sha)
    archived = pd.read_csv(path, sep="\t", float_precision="round_trip", dtype={k:str for k in keys})
    actual = pd.DataFrame(actual).astype({k:str for k in keys})
    for d in (archived, actual):
        for key in keys:
            d[key] = d[key].fillna("").replace("nan", "")
    if set(archived) != set(actual):
        raise ValueError(f"Column mismatch: archive={list(archived)}, replay={list(actual)}")
    a, b = (d.set_index(keys, verify_integrity=True).sort_index() for d in (archived, actual))
    pd.testing.assert_index_equal(a.index, b.index)
    errors, worst = {}, {}
    for field in a:
        if field not in floats:
            pd.testing.assert_series_equal(a[field], b[field], check_dtype=False, check_exact=True)
            continue
        x, y = a[field].to_numpy(float), b[field].to_numpy(float)
        np.testing.assert_array_equal(np.isfinite(x), np.isfinite(y))
        keep = np.isfinite(x)
        delta = np.abs(x[keep] - y[keep])
        errors[field] = float(delta.max(initial=0))
        if len(delta):
            j = np.flatnonzero(keep)[delta.argmax()]
            worst[field] = {"key": list(a.index[j]), "archive": float(x[j]), "replay": float(y[j])}
    return {"rows":len(a), "max_abs_diff":max(errors.values()), "fields":errors, "worst":worst}


def job_meta():
    return {"id": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
            "node": os.environ.get("SLURMD_NODENAME"), "cores": os.environ.get("SLURM_CPUS_PER_TASK")}
