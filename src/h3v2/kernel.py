"""Fit the published H2 PSD kernel on strictly cross-fitted support for stacking."""
import argparse
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from nextgen_h2.model import Baseline, Fit, Fusion, Representation
from nextgen_h2.run import META, load_context, read_json
from vc_build.io import publish_json, sha256

from .data import ASSETS, ROOT, SPLITS, TARGETS, grouped_split, nested, read_data, store, table_hash

CONFIG = (Baseline("molecular", 4.0), Representation("lincs", "modz24", "raw", 0, "cos"),
          Fusion("psd", .5))


def kernel_predict(data, registry, fit, query, mode, device):
    groups = set(grouped_split(fit, mode, registry))
    if groups & set(grouped_split(query, mode, registry)):
        raise ValueError("Kernel stack training leaked its assessment scaffold or lineage")
    if len(query) == 0 or not {"endpoint_direction", "endpoint_value"} <= set(fit):
        raise ValueError("Stack kernel needs fit outcomes and a valid query")
    started = perf_counter()
    engine = Fit(data, fit, query[META], registry=registry)
    base, rep, fusion = CONFIG
    c1 = engine.extract(engine.baseline(base))
    h2 = engine.extract(engine.fusion(base, [rep], fusion))
    mu = engine.intercept.cpu().numpy()[engine.selector[1]]
    result = query[["row_id"]].assign(H2=h2, C1=c1, drug_mean=mu)
    if not np.isfinite(result[["H2","C1","drug_mean"]]).all().all():
        raise ValueError("Nonfinite kernel forecast")
    wall = perf_counter() - started
    return result, {"train_rows": len(fit), "query_rows": len(query), "fit_query_group_overlap": 0,
                    "fit_row_key_sha256": table_hash(fit), "query_row_key_sha256": table_hash(query),
                    "seconds": wall, "solver": engine.notes, "device": str(device)}


def crossfit(split):
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("The kernel bank requires its Slurm GPU")
    torch.set_float32_matmul_precision("high")
    data, registry, _, _ = load_context(ASSETS, TARGETS, device)
    full = read_data(split, "fit")
    query = read_data(split, "query")
    mode = split.rsplit("-", 1)[0]
    chunks = [nested(full, split, registry, inner)[1] for inner in range(3)]
    if len(pd.concat(chunks)) != len(full) or pd.concat(chunks).row_id.duplicated().any():
        raise ValueError("Cross-fitting chunks must partition the outer training set")
    root = ROOT / "kernels" / split
    root.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for n in range(3):
        rest = pd.concat([ch for i, ch in enumerate(chunks) if i != n], ignore_index=True)
        for key, f, q in ((f"one-{n}", chunks[n], rest), (f"two-{n}", rest, chunks[n])):
            values, proof = kernel_predict(data, registry, f, q, mode, device)
            file = root / (key + ".parquet")
            store(values, file)
            evidence[key] = proof | {"file_sha256": sha256(file)}
            print(split, key, len(values), round(proof["seconds"], 2), flush=True)
    kernel, proof = kernel_predict(data, registry, full, query, mode, device)
    file = root / "outer.parquet"
    store(kernel, file)
    evidence["outer"] = proof | {"file_sha256": sha256(file)}
    publish_json(root / "receipt.json", {"split":split, "parts":evidence, "outer_query_outcomes_loaded": False})


def read_bank(split, part):
    root = ROOT / "kernels" / split
    path = root / f"{part}.parquet"
    digest = read_json(root / "receipt.json")["parts"][part]["file_sha256"]
    if sha256(path) != digest:
        raise ValueError("Frozen OOF kernel artifact changed")
    return pd.read_parquet(path).set_index("row_id")


def oof_fit(rows, split, registry, inner=None):
    if inner is None:
        return pd.concat([read_bank(split,f"two-{j}") for j in range(3)], verify_integrity=True).loc[rows.row_id]
    mode, idx = split.rsplit("-",1)
    all_rows = read_data(split,"fit")
    check_q = {j: set(nested(all_rows, split, registry, j)[1].row_id)
               for j in range(3) if j != inner}
    columns = []
    for one, other in (tuple(check_q), tuple(reversed(check_q))):
        intended = rows.row_id[rows.row_id.isin(check_q[other])]
        columns.append(read_bank(split, f"one-{one}").loc[intended])
    return pd.concat(columns, verify_integrity=True).loc[rows.row_id]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=SPLITS)
    crossfit(p.parse_args().split)
