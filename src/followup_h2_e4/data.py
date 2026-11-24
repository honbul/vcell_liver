"""Outcome-blind roster, source-only features, and pinned data utilities for E4."""
from datetime import datetime, timezone
from pathlib import Path
import os
import sys

import pandas as pd
import torch

from followup_h2.common import (
    AUXILIARY, COLD, HALLMARK, PRIOR, STORAGE, TARGET, V1_ASSETS, V1_TARGETS, V2,
    parquet_lock, sha256,
)
from nextgen_h2.assets import GDSC, KEY, LIMORE, SUPPLEMENT
from nextgen_h2.model import Inputs, molecular_features
from nextgen_h2.replay import network_deny
from nextgen_h2.run import META, TRAIN, read_json
from nextgen_h2_v2.assets import FOLDS_SHA
from vc_build.v0plus_cv import molecular_inputs
from vc_build.v0plus_neural import load_auxiliary

ROOT = STORAGE / "followup_20261124/h2_e4_loso"
REPORTS = Path(__file__).resolve().parents[2] / "reports/followup_h2_e4_loso"
PLAN = REPORTS / "PLAN_LOCK.md"
SCREENS = {"CTRPv2": 285842, "GDSC2": 83995, "PRISM20Q2": 300993}
CPU = torch.device("cpu")
OUTCOMES = ["endpoint_value", "endpoint_direction"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cpu_threads():
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def internal_only():
    """Deny Python opens of external rosters, outcomes and mixed external forecasts."""
    blocked = (
        LIMORE, SUPPLEMENT, GDSC, COLD / "query.parquet",
        V1_ASSETS / "gdsc_query.parquet", V2 / "study/forecasts/3",
        STORAGE / "nextgen_h123/h2_multiview/study-v2/forecasts/3",
        STORAGE / "followup_20261124/h2_e1e3/e1-components/3",
        STORAGE / "followup_20261124/h2_e1e3/e1-score",
    )

    def guard(event, args):
        network_deny(event, args)
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        if any(path.is_relative_to(item) for item in blocked):
            raise PermissionError(f"External asset is locked before E4 selection: {path}")

    sys.addaudithook(guard)


def registry():
    result = parquet_lock(V1_ASSETS / "folds.parquet", FOLDS_SHA)
    if len(result) != 1186 or not result.index.is_unique:
        raise ValueError("Changed source scaffold registry")
    return result


def train_hash():
    return read_json(COLD / "receipt.json")["files"]["train.parquet"]


def source_metadata():
    return parquet_lock(TRAIN, train_hash(), columns=META).sort_values("row_id", ignore_index=True)


def source_rows(parents, screens, *, labels):
    filters = [(KEY, "in", parents), ("source_id", "in", screens)]
    columns = META + OUTCOMES if labels else META
    with TRAIN.open("rb") as handle:
        return pd.read_parquet(handle, columns=columns, filters=filters).sort_values("row_id", ignore_index=True)


def lincs_profiles(parents):
    v1 = read_json(V1_ASSETS / "data_receipt.json")
    library = V2 / "assets-lincs"
    v2 = read_json(library / "receipt.json")
    return {
        "modz24": parquet_lock(V1_ASSETS / "lincs_modz24.parquet", v1["files"]["lincs_modz24.parquet"]).reindex(parents),
        "landmark_mean": parquet_lock(library / "landmark_mean.parquet",
                                      v2["artifacts"]["landmark_mean.parquet"]).reindex(parents),
    }


def observed(profiles, recipe="landmark_mean"):
    return profiles[recipe].notna().mean(axis=1).ge(.75)


def source_context(reg, cells, parents):
    target_proof = read_json(V1_TARGETS / "receipt.json")
    target_file = V1_TARGETS / "targets.json"
    if sha256(target_file) != target_proof["targets_sha256"]:
        raise ValueError("Changed target registry")
    source = read_json(COLD / "receipt.json")
    rna = parquet_lock(COLD / "rna.parquet", source["files"]["rna.parquet"]).reindex(cells)
    edges, _, _ = molecular_inputs(PRIOR, HALLMARK, TARGET)
    aux, _ = load_auxiliary(AUXILIARY)
    return Inputs(V1_ASSETS, molecular_features(reg.loc[parents]), lincs_profiles(parents), None,
                  rna, edges, read_json(target_file), aux, CPU)
