"""Replay registered inner fits on the allocated G3 A40 and compare CPU evidence."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nextgen_h2.run import read_json
from vc_build.io import publish_json, sha256
from .data import ROOT
from .fit import fold_fit
from .fit_tree import g3_cuda_library, g3_opencl_library
from .selection import receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "opencl"), default="cuda")
    device = parser.parse_args().device
    library = g3_cuda_library() if device == "cuda" else g3_opencl_library()
    cases = read_json(ROOT / "locks/cuda-g3-checks.json")["cases"]
    case = cases[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    stage, split, inner, family, config_id = (case[key] for key in
        ("stage", "split", "inner", "family", "config_id"))
    suffix = Path(stage) / split / f"inner-{inner}" / family / config_id
    cpu_dir = ROOT / "fits" / suffix
    gpu_dir = ROOT / f"{device}-validation" / suffix
    cpu_receipt = read_json(cpu_dir / "receipt.json")
    verified = receipt(stage, split, family, cpu_receipt["config"], inner)
    if verified["config_id"] != config_id or verified["receipt_sha256"] != case["cpu_receipt_sha256"]:
        raise ValueError("The locked CPU comparison changed")

    fold_fit(stage, split, family, inner, "gbdt", tree_device=device,
             candidate_id=config_id, diagnostic=True)
    gpu_receipt = read_json(gpu_dir / "receipt.json")
    exact_fields = ("stage", "split", "inner", "family", "config", "config_id", "seed", "atlas_sha256",
                    "model_input_sha256", "lincs_full_unclipped", "query_has_outer_labels")
    for key in exact_fields:
        if cpu_receipt[key] != gpu_receipt[key]:
            raise ValueError(f"The GPU replay changed {key}")
    for artifact, key in (("predictions.parquet", "forecast_sha256"), ("curves.parquet", "curve_sha256")):
        if sha256(gpu_dir / artifact) != gpu_receipt[key]:
            raise ValueError(f"The GPU replay failed verification for {artifact}")
    if gpu_receipt["training"]["device"] != device:
        raise ValueError("The diagnostic did not use the registered GPU implementation")

    cpu = pd.read_parquet(cpu_dir / "predictions.parquet").set_index("row_id")
    gpu = pd.read_parquet(gpu_dir / "predictions.parquet").set_index("row_id")
    if not cpu.index.is_unique or not gpu.index.is_unique or not cpu.index.equals(gpu.index):
        raise ValueError("CPU and GPU prediction rows differ")
    first, second = (frame.predicted.to_numpy(dtype=np.float64) for frame in (cpu, gpu))
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("The device comparison includes a nonfinite prediction")
    cpu_seconds = cpu_receipt["training"]["wall_seconds"]
    gpu_seconds = gpu_receipt["training"]["wall_seconds"]
    difference = second - first
    evidence = {
        **case,
        f"{device}_receipt_sha256": sha256(gpu_dir / "receipt.json"),
        f"{device}_library_sha256": sha256(library),
        "gpu": torch.cuda.get_device_name(0),
        "rows": len(first),
        "cpu_seconds": cpu_seconds,
        f"{device}_seconds": gpu_seconds,
        "speedup": cpu_seconds / gpu_seconds,
        "cpu_best_iteration": cpu_receipt["training"]["best_iteration"],
        f"{device}_best_iteration": gpu_receipt["training"]["best_iteration"],
        "prediction_rms_delta_over_cpu_sd": float(np.sqrt(np.mean(difference**2)) / max(first.std(), 1e-12)),
        "prediction_max_absolute_delta": float(np.max(np.abs(difference))),
        "utility_delta": gpu_receipt["utility"] - cpu_receipt["utility"],
        "metric_deltas": {key: gpu_receipt["score"][key] - value for key, value in cpu_receipt["score"].items()},
        "query_has_outer_labels": False,
    }
    publish_json(gpu_dir / "comparison.json", evidence)
    print(f"{split}/{inner}/{config_id} {device} measured speedup {evidence['speedup']:.2f}, "
          f"inner utility change {evidence['utility_delta']:+.6f}", flush=True)


if __name__ == "__main__":
    main()
