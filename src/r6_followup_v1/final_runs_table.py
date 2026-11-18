"""Run ledger of the record-only follow-up evaluation (D7): one row per frozen run, from the run records.

Reads only training records and the scoring manifest (no response values); writes
reports/r6_followup_v1/final_followup_eval_runs.tsv. Written after the scoring job; it changes no score.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import common as C


def main() -> None:
    fm = json.loads(Path("manifests/r6_followup_v1/freeze_manifest.json").read_text())
    inp = json.loads(Path("manifests/r6_followup_v1/final_followup_eval_inputs.json").read_text())
    rows = []
    for fr in fm["final_runs"]:
        rid = fr["run_id"]
        rec = json.loads((C.assets() / "runs" / "final_followup" / rid / "record.json").read_text())
        for k in ("arm", "lr", "seed", "anchor_kappa", "pair_lambda", "epochs"):
            assert rec[k] == fr[k] or (k == "lr" and abs(rec[k] - fr[k]) < 1e-12), (rid, k)
        assert rec["views"]["sha256_counts"] == fr["view_cache_sha256_counts"], rid
        ev = inp["runs"][rid]
        assert ev["ckpt_sha256"] == rec["ckpt_sha256"], rid
        rows.append(dict(label=fr["label"], arm=rec["arm"], run_id=rid, stage=rec["stage"], setting=rec["setting"], seed=rec["seed"],
                         lr=rec["lr"], anchor_kappa="" if rec["anchor_kappa"] is None else rec["anchor_kappa"], pair_lambda=rec["pair_lambda"],
                         epochs=rec["epochs"], n_fit_drugs=rec["n_fit_drugs"], n_params=rec["params"]["total"], updates=rec["updates"],
                         train_seconds=rec["train_seconds"], peak_vram_MB=rec["peak_vram_MB"], gpu=rec["gpu"], job=rec["job"],
                         array=rec.get("array", ""), distinct_views_used=rec["distinct_views_used"], view_cache=Path(rec["views"]["cache"]).name,
                         train_commit=rec["commit"], ckpt_sha256=rec["ckpt_sha256"], pred_test_sha256=ev["pred_sha256"],
                         split_role=inp["split_role"]))
    out = Path("reports/r6_followup_v1/final_followup_eval_runs.tsv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(out, len(rows))


if __name__ == "__main__":
    main()
