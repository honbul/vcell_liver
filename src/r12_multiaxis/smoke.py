"""Exercise real staged Tahoe rows and the G1 training environment."""

import argparse
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.r11_chemical_probe.common import config, write_json
from src.r11_chemical_probe.dataset import GROUP

from .data import Fit, Source
from .network import fit_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    cfg = config(args.config)
    source = Source(cfg, args.assets, 0, "A")
    fit_rows, val_rows = source.inner(0)
    fit_rows = fit_rows[fit_rows[GROUP].isin(fit_rows[GROUP].unique()[:12])].reset_index(drop=True)
    val_rows = val_rows[val_rows[GROUP].isin(val_rows[GROUP].unique()[:12])].reset_index(drop=True)
    fit = Fit(source, fit_rows, val_rows, observe_query=True)
    experiments = {}
    for arm, head, loss in (("A4", "concat", "L0"), ("J1_FiLM_L2", "film", "L2"),
                            ("L6", "concat", "L6")):
        design = fit.inputs("pca", augment=loss == "L6")
        results = [
            fit_network(fit, design, head, loss, 0.0003, 42, args.device, 2, vic=0.01)
            for _ in range(2 if arm == "J1_FiLM_L2" else 1)
        ]
        best, runtime, curve, _ = results[0]
        experiments[arm] = {"abs": best["abs"][0], "contrast": best["contrast"][0],
                            "runtime": runtime, "curve": curve}
        if len(results) == 2:
            difference = float(np.max(np.abs(best["contrast"][2] - results[1][0]["contrast"][2])))
            if difference:
                raise ValueError(f"Seed/checkpoint replay differed by {difference}")
            experiments[arm]["repeat_max_abs"] = difference
    write_json(args.out, dict(
        job=os.environ.get("SLURM_JOB_ID"), host=platform.node(), torch=torch.__version__,
        cuda=torch.version.cuda, fit_rows=len(fit_rows), validation_rows=len(val_rows),
        seconds=time.monotonic() - started, results=experiments,
    ))
    print(experiments, flush=True)


if __name__ == "__main__":
    main()
