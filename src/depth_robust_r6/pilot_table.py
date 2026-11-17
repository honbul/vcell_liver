"""Collect the development runs' records into one table (learning curves included)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+", default=["pilot_"])
    ap.add_argument("--out", default="reports/depth_robust_r6/tables/pilot_runs.tsv")
    ap.add_argument("--curves", default="reports/depth_robust_r6/tables/pilot_curves.tsv")
    a = ap.parse_args()
    root = Path(os.environ["R6_ASSETS"]) / "runs" / "dev"
    rows, curves = [], []
    for run in sorted(root.iterdir()):
        if not any(run.name.startswith(p) for p in a.prefix):
            continue
        rf = run / "record.json"
        if not rf.exists():
            rows.append(dict(run_id=run.name, status="no record (cancelled or failed)"))
            continue
        r = json.loads(rf.read_text())
        dv = r["dev"]
        rows.append(dict(run_id=r["run_id"], status="ok", arm=r["arm"], lr=r["lr"], seed=r["seed"], epochs=r["epochs"],
                         weights=json.dumps(r["weights"]), params=r["params"],
                         dev_p1=dv["p1.0_s0"], dev_p05=float(np.mean([dv[f"p0.5_s{s}"] for s in range(3)])),
                         dev_p025=float(np.mean([dv[f"p0.25_s{s}"] for s in range(3)])),
                         train_seconds=r["train_seconds"], peak_vram_MB=r["peak_vram_MB"], gpu=r["gpu"], job=r["job"],
                         teacher=(r["teacher"] or {}).get("run", ""), commit=r["commit"][:7]))
        for c in r["curve"]:
            if "dev_rmse_p1.0" in c:
                curves.append(dict(run_id=r["run_id"], arm=r["arm"], lr=r["lr"], epoch=c["epoch"], train_supervised=c["train_supervised"],
                                   dev_p1=c["dev_rmse_p1.0"], dev_p025=c["dev_rmse_p0.25"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(a.out, sep="\t", index=False, float_format="%.6f")
    pd.DataFrame(curves).to_csv(a.curves, sep="\t", index=False, float_format="%.6f")
    print(t.drop(columns=["weights", "teacher", "commit", "gpu"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
