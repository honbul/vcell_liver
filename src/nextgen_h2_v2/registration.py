"""Verify the inherited V1 benchmark query coordinates without opening responses."""
import argparse
from pathlib import Path

import numpy as np
import torch

from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.run import META, response, subfold
from vc_build.io import publish_json, sha256

from .assets import FOLDS_SHA, GDSC_SHA, V1_ASSETS, V1_RUN
from .freeze import V1_SHA
from .kernel import context


def run(root, out):
    _, _, registry, coverage, external = context(root, torch.device("cpu"), external=True)
    heads, counts = [], []
    for i in range(4):
        if i < 3:
            fit, held = subfold(registry, i)
            expected = response(held, False)
            assert not (set(registry.loc[fit].scaffold_block) & set(registry.loc[held].scaffold_block))
            assert not (set(registry.loc[fit].connectivity) & set(registry.loc[held].connectivity))
        else:
            expected = external[META].sort_values("row_id", ignore_index=True)
        prior = parquet_lock(V1_RUN / "forecasts" / str(i) / "predictions.parquet", V1_SHA[i])
        if not expected[META].equals(prior[META]) or not np.isfinite(prior[["C1", "champion"]]).all().all():
            raise ValueError(f"Baseline forecasts differ from the registered query metadata for {i}")
        whole = prior[["cell_id", KEY, "source_id", "endpoint_id"]].copy()
        heads.append({"part": i, "v1_forecast_sha256": V1_SHA[i], "query_observations": len(whole),
                      "query_parents": whole[KEY].nunique(), "identical_keys": True,
                      "rows_by_screen": whole.source_id.value_counts().to_dict(),
                      "LINCS_parents": whole.loc[whole[KEY].map(coverage.lincs_modz24).fillna(False), KEY].nunique()})
        counts.append(len(whole))
    if sum(counts[:3]) != 670830:
        raise ValueError("The pinned v1 response benchmark has changed")
    publish_json(out, {"folds_sha256": FOLDS_SHA, "gdsc_sha256": GDSC_SHA, "parts": heads,
                       "source_outcome_columns_read": [], "registration_sha256": sha256(
                           V1_ASSETS / "data_receipt.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    run(a.root, a.out)
