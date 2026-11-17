"""RUN_LEDGER.tsv: every run directory of R6, finished or not, plus hand-written notes.

A run without `record.json` did not finish (cancelled or failed); the reason comes from
`reports/depth_robust_r6/run_notes.tsv` (run_id <tab> status <tab> note), which is edited by hand.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def main() -> None:
    root = Path(os.environ["R6_ASSETS"]) / "runs"
    notes = {}
    nf = Path("reports/depth_robust_r6/run_notes.tsv")
    if nf.exists():
        for r in csv.DictReader(nf.open(), delimiter="\t"):
            notes[r["run_id"]] = r
    rows = []
    for setting in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for run in sorted((root / setting).iterdir()):
            rec = json.loads((run / "record.json").read_text()) if (run / "record.json").exists() else {}
            n = notes.get(run.name, {})
            dev = rec.get("dev") or {}
            for ef in sorted(run.glob("eval_final.json")):          # held-out scores written once by final_eval
                dev = json.loads(ef.read_text())["scores"]
            if setting == "final" and "scores" in rec:              # ridge / public reference records
                sc = rec["scores"]
                if all(isinstance(v, dict) for v in sc.values()):
                    ks = list(next(iter(sc.values())))
                    dev = {k: sum(v[k] for v in sc.values()) / len(sc) for k in ks} if "alphas" not in rec else sc.get("Z0", {})
            p25 = [v for k, v in dev.items() if k.startswith("p0.25_")]
            rows.append(dict(
                run_id=run.name, setting=setting, arm=rec.get("arm", rec.get("model", "")),
                scored_on="test drugs (after freeze)" if setting == "final" else "dev fold",
                status=n.get("status") or ("finished" if rec else "not_finished"),
                job=rec.get("job", ""), commit=(rec.get("commit") or "")[:7], lr=rec.get("lr", ""), seed=rec.get("seed", ""),
                epochs=rec.get("epochs", ""), head_mode=(rec.get("hp") or {}).get("head_mode", "joint") if "hp" in rec else "",
                drug_shrink=(rec.get("hp") or {}).get("drug_shrink", 0) if "hp" in rec else "",
                teacher=Path(rec["teacher"]["run"]).name if rec.get("teacher") else "",
                train_seconds=rec.get("train_seconds", ""), gpu=rec.get("gpu", ""),
                rmse_p1=f"{dev['p1.0_s0']:.5f}" if "p1.0_s0" in dev else "",
                rmse_p025=f"{sum(p25) / len(p25):.5f}" if p25 else "",
                ckpt_sha256=rec.get("ckpt_sha256", ""), note=n.get("note", "")))
    out = Path("reports/depth_robust_r6/RUN_LEDGER.tsv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(len(rows), "runs ->", out)


if __name__ == "__main__":
    main()
