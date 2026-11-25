"""Read frozen E5 inputs and validate the full registered 991-lineage scorer."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from followup_h2.common import panel_hash, sha256
from followup_h2.inference import DRAWS
from followup_h2_e4.data import OUTCOMES, V1_ASSETS, cpu_threads, registry, train_hash
from followup_h2_e5.forecast import choice_lock, read_frozen
from followup_h2_e5.score import EXTERNAL, aggregates, licob18, score_panel
from nextgen_h2.assets import KEY
from nextgen_h2.run import META, TRAIN
from nextgen_h2.score import external_labels
from vc_build.external import native_sensitivity
from vc_build.io import publish_json

BASE = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
HOME = BASE / "paper_b_h2_posthoc"
E5 = BASE / "followup_20261124/h2_e5_upgrade"
E3 = BASE / "followup_20261124/h2_e1e3"
TABLES = HOME / "tables"
ARCHIVE = E5 / "score"
PNAMES = ("E5_METRICS.tsv", "E5_PAIRED_95CI.tsv")
FORECAST_INDICES = tuple(range(7))
E5_SELECTION = "2422af579a1a9c6a04cdec721bc6a58c1aa92a18b088b6c75a6a760dc3cc6b86"
E5_SCORE = "93c2c4a30e1c606cae7297925999613fd5c4787b28b69e13f78b1b014d3da73c"


def read(path):
    return json.loads(path.read_text())


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def archived_inputs():
    path = ARCHIVE / "score_receipt.json"
    assert sha256(path) == E5_SCORE
    score = read(path)
    files = {n: sha256(ARCHIVE / n) for n in PNAMES}
    assert all(score["output_sha256"][k] == h for k, h in files.items())
    assert score["score_passes"] == 1 and score["comparison"]["decision"] == "GO-V3-PARETO-UPGRADE"
    return score, files


def panels():
    """Use outcome identity and the registered LICOB donor aggregation unchanged."""
    score, _ = archived_inputs()
    split, _, selection_sha = choice_lock(E5 / "preflight", E5 / "selection")
    assert selection_sha == score["selection_sha256"] == E5_SELECTION
    prepared = {}
    for i in FORECAST_INDICES:
        d = E5 / "forecasts" / str(i)
        assert sha256(d / "receipt.json") == score["forecast_receipt_sha256"][str(i)]
        prepared[i] = read_frozen(d, selection_sha)
    assert sha256(TRAIN) == train_hash()
    with TRAIN.open("rb") as h:
        source = pd.read_parquet(h, columns=[*META, *OUTCOMES]).sort_values("row_id", ignore_index=True)
    assert panel_hash(source) == split["source_roster_sha256"]
    assert source.row_id.is_unique and source.lineage_group.nunique() == 991
    truth = pd.Series(native_sensitivity(source).to_numpy(), index=source.row_id)
    del source
    by_track = defaultdict(list)
    for i, (p, note) in prepared.items():
        if i < 6:
            assert p.row_id.is_unique and set(p.row_id) <= set(truth.index)
            by_track[note["track"]].append((note["fold"], p.assign(truth=p.row_id.map(truth))))
    for partitions in by_track.values():
        row_ids = pd.concat([p.row_id for _, p in partitions])
        assert row_ids.is_unique and set(row_ids) == set(truth.index)
    labels, license_table, ext_groups = external_labels(V1_ASSETS, read(V1_ASSETS / "data_receipt.json"))
    ext = prepared[6][0]
    label_map = labels.set_index("row_id").truth
    assert label_map.index.is_unique and set(label_map.index) == set(ext.row_id)
    assert np.isfinite(label_map).all() and np.isfinite(truth).all()
    cohorts = {}
    for name, block in ext.assign(truth=ext.row_id.map(label_map)).groupby("source_id", sort=True):
        if name == "LICOB":
            block = aggregates(block, license_table)
            cohorts["LICOB18"] = licob18(block)
        cohorts[name] = block
    source_groups = registry().scaffold_block
    return [
        ("T1", "CrossScreen", by_track["T1"], source_groups),
        ("T2", "DRP", by_track["T2"], source_groups),
        *(("T3", c, [(0, cohorts[c])], ext_groups) for c in EXTERNAL),
    ]


def comparison(actual, file, keys, floats):
    """Check all archived keys, nonnumeric fields, missingness, and numeric values."""
    archived = pd.read_csv(file, sep="\t", float_precision="round_trip")
    actual = pd.DataFrame(actual)
    assert set(actual.columns) == set(archived.columns)
    for d in (archived, actual):
        d["fold"] = d["fold"].astype(str)
    a, b = (d.set_index(keys, verify_integrity=True).sort_index() for d in (archived, actual))
    pd.testing.assert_index_equal(a.index, b.index)
    abs_max = {}
    for col in a:
        if col not in floats:
            pd.testing.assert_series_equal(a[col], b[col], check_dtype=False, check_exact=True)
            continue
        v, u = a[col].to_numpy(float), b[col].to_numpy(float)
        assert np.isfinite(v).all() and np.isfinite(u).all()
        abs_max[col] = float(np.max(np.abs(v - u)))
    return {"rows": len(a), "max_abs_diff": max(abs_max.values()), "fields": abs_max}


def replay():
    start = timestamp()
    assert DRAWS == 2000
    pinned, hashes = archived_inputs()
    metrics, intervals = [], []
    for track, cohort, p, groups in panels():
        m, ci, support = score_panel(p, groups, track, cohort)
        assert support == pinned["supports"][cohort]
        metrics += m
        intervals += ci
    keys = ["track", "cohort", "support", "screen", "fold", "metric"]
    result = {
        PNAMES[0]: comparison(metrics, ARCHIVE / PNAMES[0], keys + ["arm"], ["value"]),
        PNAMES[1]: comparison(intervals, ARCHIVE / PNAMES[1], keys + ["contrast", "resampling"],
                              ["delta", "lo95", "hi95"]),
    }
    maximum = max(x["max_abs_diff"] for x in result.values())
    print(result, flush=True)
    assert maximum <= 1e-9, result
    evidence = {"status": "PASS", "started_at_utc": start, "passed_at_utc": timestamp(),
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "lineage_groups": 991,
                "draws": DRAWS, "seed": 495123, "max_abs_diff": maximum, "threshold": 1e-9,
                "retrained_models": 0, "refits": 0, "reselected_policies": 0,
                "archive_sha256": hashes, "archive_score_receipt_sha256": E5_SCORE,
                "archive_selection_sha256": E5_SELECTION, "comparison": result}
    publish_json(TABLES / "e5_trackb_machine_replay_receipt.json", evidence)


def require_replay():
    p = TABLES / "e5_trackb_machine_replay_receipt.json"
    r = read(p)
    assert r["status"] == "PASS" and r["max_abs_diff"] <= 1e-9 and r["lineage_groups"] == 991
    assert archived_inputs()[1] == r["archive_sha256"]
    assert r["comparison"][PNAMES[0]]["rows"] == 4200 and r["comparison"][PNAMES[1]]["rows"] == 12420
    return r


if __name__ == "__main__":
    cpu_threads()
    replay()
