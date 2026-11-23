"""Exercise the one-pass scorer on registered row keys and wholly simulated responses."""
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from h3v2 import score
from h3v2.data import ASSETS, ROOT, read_data
from h3v2.final_fit import external_query
from h3v2.report import previous_metrics
from nextgen_h2.assets import COLD, KEY, canonical_registry, linked_groups, parquet_lock
from nextgen_h2.run import META, read_json
from vc_build.io import publish_json, sha256


def simulated_frames():
    rng = np.random.default_rng(37129)
    receipt = read_json(ASSETS / "data_receipt.json")
    coverage = parquet_lock(ASSETS / "coverage.parquet", receipt["files"]["coverage.parquet"])
    frames, truth = {}, {}
    external = external_query()
    for split in score.ALL_PARTITIONS:
        rows = external if split == "external" else read_data(split, "query")
        data = rows[META].assign(lincs=rows[KEY].map(coverage.lincs_modz24).eq(True))
        truth[split] = data[["row_id"]].assign(truth=rng.standard_normal(len(data)))
        arms = ["C1", "H2", "drug_mean", "drug_ridge"]
        if any(split.startswith(name + "-") for name in score.SCREENS):
            arms += ["structured_v1", "drug_ridge_v1", "drug_mean_v1"]
            arms.extend(f"H3v1__{seed}" for seed in score.SEEDS)
        arms.extend(f"{name}__{seed}" for name in (*score.FAMILIES, "Champion") for seed in score.SEEDS)
        for arm in arms:
            data[arm] = rng.standard_normal(len(data)).astype(np.float32)
        data["Champion-ensemble"] = data[[f"Champion__{seed}" for seed in score.SEEDS]].mean(axis=1)
        frames[split] = data
    licob_stamp = read_json(COLD / "receipt.json")["files"]["query.parquet"]
    licence = parquet_lock(COLD / "query.parquet", licob_stamp)
    ext_blocks = linked_groups(canonical_registry(external))
    return frames, truth, licence, ext_blocks


def verify_archived_h2_macro():
    """Check the reporting replay from published unit aggregates rather than response rows."""
    from h3v2.data import STUDY

    root = STUDY / "score"
    receipt = read_json(root / "score_receipt.json")
    output = []
    supports = {"all": "all", "champion_observed": "lincs724", "champion_unavailable": "structure462"}
    for file, names in (("unit_drug.parquet", score.DRUG), ("unit_cell.parquet", score.CELL)):
        units = parquet_lock(root / file, receipt["results"][file])
        matches = units.scope.eq("DRP") & units.arm.isin(["C1", "champion"]) & units.support.isin(supports)
        units = units.loc[matches].assign(tissue="", scope="drug")
        units["arm"] = units.arm.replace({"champion":"H2"})
        units["support"] = units.support.map(supports)
        output.append(score.native_metric_points(units, names))
    assert len(previous_metrics(pd.concat(output, ignore_index=True))) == 36


def rehearsal():
    verify_archived_h2_macro()
    frames, truth, licence, ext_blocks = simulated_frames()
    with TemporaryDirectory(dir=ROOT / "tmp", prefix="score-rehearsal-") as workspace:
        path = Path(workspace)
        publish_json(path / "locks/frozen.json", {"simulated": True})
        publish_json(path / "splits/manifest.json", {"split_support": {
            split: {"label_sha256": "simulated"} for split in score.SPLITS
        }})
        seen = []

        def labels(file, digest):
            if file == ASSETS / "cells.parquet":
                return parquet_lock(file, digest)
            assert file.parent == path / "outcomes" and digest == "simulated"
            assert (path / "score/consumed.json").is_file()
            seen.append(file.stem)
            return truth[file.stem]

        def ext_labels(*args):
            assert (path / "score/consumed.json").is_file()
            seen.append("external")
            return truth["external"], licence, ext_blocks

        with patch.object(score, "ROOT", path), patch.object(score, "parquet_lock", labels), \
                patch.object(score, "external_labels", ext_labels), \
                patch.object(score, "preflight", return_value=(frames, {"simulated": True}, {})), \
                patch.object(sys, "argv", ["score-rehearsal"]):
            score.evaluate()
            try:
                score.evaluate()
            except FileExistsError:
                pass
            else:
                raise AssertionError("A second synthetic pass was allowed")
        assert seen == list(score.ALL_PARTITIONS)
        root = path / "score"
        receipt = read_json(root / "score_receipt.json")
        assert receipt["score_passes"] == 1 and len(receipt["tissue_roster"]) == 28
        assert {key: receipt["parts"][key] for key in ("GDSC1", "LIMORE", "LICOB")} == {
            "GDSC1": 39545, "LIMORE": 3400, "LICOB": 2888
        }
        assert sum(receipt["parts"][key] for key in score.SPLITS[:3]) == 670830
        assert sum(receipt["parts"][key] for key in score.SPLITS[3:6]) == 670830
        assert sum(receipt["parts"][key] for key in score.SPLITS[6:]) == 45428
        assert all(sha256(root / file) == digest for file, digest in receipt["tables"].items())
        print("Verified synthetic-only one-pass run for 15 folds, 3 external cohorts and 28 tissues", flush=True)
        print("Archived H2 macro verification matched all 36 metric/support/control entries", flush=True)
        print("Registered response files were never read by the rehearsal", flush=True)


if __name__ == "__main__":
    rehearsal()
