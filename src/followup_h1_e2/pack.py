"""Package only aggregate tables and signed provenance from the G1 runs."""
import argparse
from pathlib import Path
from shutil import copyfile

import pandas as pd

from nextgen_h1_anchor.inputs import json_write
from nextgen_h1_v2.data import ROOT as H1
from vc_build.io import sha256

from .curves import CONTRACT, ROOT, SCREENS, checked, read_json
from .score import TASKS

RAW_PATHS = {s: f"curves-{s.lower()}-a" for s in SCREENS}
P0_RUNS = {"LIMORE": "p0-LIMORE-c", "K2": "p0-K2-b", "K4": "p0-K4-b", "K8": "p0-K8-b"}
IN_USE = (36199, 36200, 36214, 36215, 36216, 36238, 36239, 36240, 36241, 36242, 36243, 36252, 36253, 36255)
ATTEMPTS = {
    36205: "original import path missing",
    36206: "original import path missing",
    36207: "original import path missing",
    36208: "original import path missing",
    36213: "same LIMORE comparisons prior to explicit second OOF-baseline check",
    36229: "forecast parent directory missing",
    36230: "forecast parent directory missing",
    36231: "forecast parent directory missing",
    36232: "forecast parent directory missing",
    36233: "forecast parent directory missing",
    36234: "forecast parent directory missing",
}


def fetch_jobs(path):
    records = pd.read_csv(path, sep="|", dtype=str).set_index("JobIDRaw", verify_integrity=True)
    expected = sorted(set(IN_USE) | set(ATTEMPTS))
    if set(records.index) != set(map(str, expected)):
        raise ValueError("Slurm accounting is incomplete")
    rows = []
    for job in expected:
        data = records.loc[str(job)]
        reason = ATTEMPTS.get(job)
        if (not data.JobName.startswith("fu-h1-") or data.Partition != "cpu" or
            any("gpu" in str(data[field]) for field in ("ReqTRES", "AllocTRES")) or
            data.NodeList not in {"c3", "c4", "c5"}):
            raise ValueError(f"Unexpected Slurm resource or job identity for {job}")
        if not reason and (data.State, data.ExitCode) != ("COMPLETED", "0:0"):
            raise ValueError(f"Production job {job} did not finish")
        rows.append({"job_id": job, "name": data.JobName, "partition": data.Partition, "node": data.NodeList,
                     "state": data.State, "exit_code": data.ExitCode, "elapsed": data.Elapsed,
                     "submitted_kst": data.Submit, "start_kst": data.Start, "end_kst": data.End,
                     "requested_tres": data.ReqTRES, "allocated_tres": data.AllocTRES, "gpu_count": 0,
                     "scope": reason or "published evidence"})
    return rows


def pack(out, accounting):
    out.mkdir()
    gate_path = ROOT / "e2_seal_a.json"
    gate = read_json(gate_path)
    e2 = ROOT / "e2-score-a"
    e2stamp = read_json(e2 / "receipt.json")
    e2claim = ROOT / "e2_score_claim.json"
    if (sha256(e2claim) != e2stamp["score_claim_sha256"] or sha256(gate_path) != e2stamp["sealed_forecast_gate_sha256"]
        or gate["contract_sha256"] != sha256(CONTRACT) or e2stamp["score_passes"] != 1 or
        e2stamp["h1_model_fit_count"] or read_json(e2claim)["slurm_job"] != "36253"):
        raise ValueError("New E2 finish does not match the signed forecast, model and score contract")
    for filename, h in e2stamp["files"].items():
        copyfile(checked(e2 / filename, h), out / filename)
    copyfile(gate_path, out / "e2_forecast_seal_receipt.json")

    raw_ledger, forecast_ledger = {}, {}
    for screen, subpath in RAW_PATHS.items():
        path = checked(ROOT / subpath / "receipt.json", gate["sources"][screen])
        raw_ledger[screen] = read_json(path)
    for task in TASKS:
        path = checked(ROOT / "forecast-b" / task / "receipt.json", gate["folds"][task])
        forecast_ledger[task] = read_json(path)
    json_write(out / "curve_support_receipt.json", {"contract_sha256": sha256(CONTRACT),
        "vc_build_commit": "5ec8ed2",
        "vc_build_document_sha256": {
            "LICOB_DOSE_ENDPOINT_PROTOCOL.md": "061a5bc5947b6c9340475f7d0887c3607886d959c6cf430dc9a7842624f66a60",
            "LICOB_CURVE_TRANSFER_RESULTS.md": "c501b0919b8e57881e7d485bb00dd157770bb382b2b39ddcfc8aa3f2e5151d6b"},
        "screens": raw_ledger, "folds": forecast_ledger,
        "licob_3d_per_dose_source_registered": False,
        "query_support_only_before_scoring": True})

    p0_ledger, contrasts = {}, []
    for task, run in P0_RUNS.items():
        path = ROOT / run
        stamp = read_json(path / "receipt.json")
        verify_file = checked(path / "original_validation.json", stamp["original_validation_sha256"])
        pairs = pd.read_csv(checked(path / "comparisons.tsv", stamp["fixed_table_sha256"]), sep="\t")
        if stamp["fit_count"] or stamp["predictions_regenerated"] or stamp["task"] != task:
            raise ValueError("P0 fit count or predictions changed")
        contrasts.append(pairs)
        p0_ledger[task] = {"resample": stamp, "original_validation": read_json(verify_file),
                           "source_receipt_sha256": sha256(path / "receipt.json")}
    if len({p0_ledger[task]["resample"]["weight_sha256"] for task in ("K2", "K4", "K8")}) != 1:
        raise ValueError("LICOB must reuse cohort weights across all budgets and repeats")
    pd.concat(contrasts, ignore_index=True).to_csv(out / "FIXED_PANEL_CI_SENSITIVITY.tsv", sep="\t",
                                                  index=False, float_format="%.12g")
    json_write(out / "p0_replay_receipt.json", {"original_branch_commit": "1912b5d",
        "analyses": p0_ledger, "new_fits": 0, "original_contract_preserved": True})
    pd.DataFrame(fetch_jobs(accounting)).to_csv(out / "SLURM_JOBS.tsv", sep="\t", index=False)
    original_score = H1 / "score-a/score_receipt.json"
    if read_json(original_score)["score_passes"] != 1:
        raise ValueError("Original H1-v2 one-pass receipt changed")
    json_write(out / "score_receipt.json", {"base_h1_v2_commit": read_json(CONTRACT)["base_h1_v2"],
        "run_root": str(ROOT), "contract_sha256": sha256(CONTRACT),
        "e2": {"g1_receipt_sha256": sha256(e2 / "receipt.json"), **e2stamp},
        "p0": {"model_fits": 0, "draws": 2000, "frozen_forecasts_reused": True,
               "original_joint_decisions_preserved": True},
        "original_h1_v2_score_receipt_sha256": sha256(original_score), "original_h1_v2_score_passes": 1,
        "gpu_allocations": 0, "slurm_accounting_sha256": sha256(accounting),
        "evidence_files": {f.name: sha256(f) for f in sorted(out.iterdir())}})
    print("Packaged aggregate evidence only", len(list(out.iterdir())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("accounting", type=Path)
    args = parser.parse_args()
    pack(args.out, args.accounting)


if __name__ == "__main__":
    main()
