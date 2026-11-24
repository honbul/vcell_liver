"""Replay registered inner fits on the allocated G3 A40 and compare CPU evidence."""
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from nextgen_h2.run import read_json
from vc_build.io import publish_json, sha256
from .data import ROOT
from .fit import fold_fit
from .selection import receipt


def main():
    overlay = ROOT / "deps/lightgbm-cuda-4.6.0/runtime"
    library = overlay / "lightgbm/lib/lib_lightgbm.so"
    if (lgb.__version__ != "4.6.0" or Path(lgb.basic._LIB._name).resolve() != library.resolve()
            or sha256(library) != (overlay / "library.sha256").read_text().strip()):
        raise ValueError("Expected the verified isolated CUDA LightGBM runtime")
    if (os.environ.get("SLURMD_NODENAME") != "g3" or torch.cuda.device_count() != 1
            or "A40" not in torch.cuda.get_device_name(0)):
        raise ValueError("Each CUDA check requires one assigned G3 A40")

    cases = read_json(ROOT / "locks/cuda-g3-checks.json")["cases"]
    case = cases[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    stage, split, inner, family, config_id = (case[key] for key in
        ("stage", "split", "inner", "family", "config_id"))
    suffix = Path(stage) / split / f"inner-{inner}" / family / config_id
    cpu_dir = ROOT / "fits" / suffix
    gpu_dir = ROOT / "cuda-validation" / suffix
    cpu_receipt = read_json(cpu_dir / "receipt.json")
    verified = receipt(stage, split, family, cpu_receipt["config"], inner)
    if verified["config_id"] != config_id or verified["receipt_sha256"] != case["cpu_receipt_sha256"]:
        raise ValueError("The locked CPU comparison changed")

    fold_fit(stage, split, family, inner, "gbdt", tree_device="cuda",
             candidate_id=config_id, diagnostic=True)
    cuda_receipt = read_json(gpu_dir / "receipt.json")
    exact_fields = ("stage", "split", "inner", "family", "config", "config_id", "seed", "atlas_sha256",
                    "model_input_sha256", "lincs_full_unclipped", "query_has_outer_labels")
    for key in exact_fields:
        if cpu_receipt[key] != cuda_receipt[key]:
            raise ValueError(f"The CUDA replay changed {key}")
    for artifact, key in (("predictions.parquet", "forecast_sha256"), ("curves.parquet", "curve_sha256")):
        if sha256(gpu_dir / artifact) != cuda_receipt[key]:
            raise ValueError(f"The CUDA replay failed verification for {artifact}")
    if cuda_receipt["training"]["device"] != "cuda":
        raise ValueError("The diagnostic did not use CUDA")

    cpu = pd.read_parquet(cpu_dir / "predictions.parquet").set_index("row_id")
    cuda = pd.read_parquet(gpu_dir / "predictions.parquet").set_index("row_id")
    if not cpu.index.is_unique or not cuda.index.is_unique or not cpu.index.equals(cuda.index):
        raise ValueError("CPU and CUDA prediction rows differ")
    first, second = (frame.predicted.to_numpy(dtype=np.float64) for frame in (cpu, cuda))
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("The device comparison includes a nonfinite prediction")
    cpu_seconds = cpu_receipt["training"]["wall_seconds"]
    cuda_seconds = cuda_receipt["training"]["wall_seconds"]
    difference = second - first
    evidence = {
        **case,
        "cuda_receipt_sha256": sha256(gpu_dir / "receipt.json"),
        "cuda_library_sha256": sha256(library),
        "gpu": torch.cuda.get_device_name(0),
        "rows": len(first),
        "cpu_seconds": cpu_seconds,
        "cuda_seconds": cuda_seconds,
        "speedup": cpu_seconds / cuda_seconds,
        "cpu_best_iteration": cpu_receipt["training"]["best_iteration"],
        "cuda_best_iteration": cuda_receipt["training"]["best_iteration"],
        "prediction_rms_delta_over_cpu_sd": float(np.sqrt(np.mean(difference**2)) / max(first.std(), 1e-12)),
        "prediction_max_absolute_delta": float(np.max(np.abs(difference))),
        "utility_delta": cuda_receipt["utility"] - cpu_receipt["utility"],
        "metric_deltas": {key: cuda_receipt["score"][key] - value for key, value in cpu_receipt["score"].items()},
        "query_has_outer_labels": False,
    }
    publish_json(gpu_dir / "comparison.json", evidence)
    print(f"{split}/{inner}/{config_id} measured speedup {evidence['speedup']:.2f}, "
          f"inner utility change {evidence['utility_delta']:+.6f}", flush=True)


if __name__ == "__main__":
    main()
