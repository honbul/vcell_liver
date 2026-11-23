"""Assemble and validate the small H1-v2 release on G1 after final CPU replay."""
import argparse
import csv
import json
import os
from pathlib import Path
from shutil import copy2, copytree

from render_reports import inner, results
from verify_report import digest, verify


AGGREGATES = {
    "inner-aggregates": (
        "H1V2_INNER_ANCHORS.tsv", "H1V2_INNER_GRID.tsv", "H1V2_INNER_WINNERS.tsv", "H1V2_INNER_BUDGET.tsv",
        "H1V2_INNER_BOUNDARIES.tsv", "H1V2_INNER_ERRORS.tsv", "H1V2_LEARNING_CURVES.tsv", "inner_receipt.json",
    ),
    "size-probe-a": ("H1V2_LICOB_SAMPLE_SIZE.tsv", "size_probe_receipt.json"),
    "score-a": ("H1V2_METRICS.tsv", "H1V2_PAIRED_95CI.tsv", "H1V2_FAMILY_BENCHMARK.tsv",
                "H1V2_PANCANCER_BY_TISSUE.tsv", "H1V2_HCC_TKI.tsv", "score_receipt.json"),
    ".": ("frozen_receipt.json", "prediction_receipt.json", "primary_replay_receipt.json",
          "prelabel_receipt.json", "outer_score_claim.json", "replay_receipt.json"),
}

# These jobs are ancestors of the successful CPU replay and also produce the indicated evidence.
COMPLETION_PROOFS = {
    "inner_neural_search": "round23-neural/LIMORE_to_LICOB-0/C/result.json",
    "inner_boundary_c": "round23-closed/LIMORE_to_LICOB-0/C/result.json",
    "inner_evidence": "inner-aggregates/inner_receipt.json",
    "fit_frozen_models": "frozen_receipt.json",
    "seal_models": "frozen_receipt.json",
    "predict_primary": "prediction_receipt.json",
    "seal_predictions": "prediction_receipt.json",
    "replay_primary": "primary_replay_receipt.json",
    "inner_licob_sample_sensitivity": "size-probe-a/size_probe_receipt.json",
    "verify_scoring_paths": "prelabel_receipt.json",
    "outer_score": "score-a/score_receipt.json",
    "replay_final": "replay_receipt.json",
}


def stage(root, templates, v1, out):
    copytree(templates, out)
    for directory, files in AGGREGATES.items():
        for name in files:
            copy2(root / directory / name, out / name)
    copy2(root / "cost-registry-receipt.json", out / "cost_receipt.json")
    ledger = out / "SLURM_JOBS.tsv"
    with ledger.open() as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows = list(reader)
        fields = reader.fieldnames
    smoke = root / "score-shape-31a23ce.json"
    if smoke.is_file():
        copy2(smoke, out / "blind_bootstrap_receipt.json")
    for row in rows:
        stage_name = row["stage"]
        proof = COMPLETION_PROOFS.get(stage_name)
        if proof and row["status"] in {"DEPENDENT", "RUNNING"}:
            if not (root / proof).is_file():
                raise ValueError(f"Missing successful Slurm output for {row['job_id']}: {proof}")
        elif stage_name == "blind_bootstrap_and_tissue_integration" and smoke.is_file():
            proof = "blind_bootstrap_receipt.json"
        else:
            continue
        row["status"] = "COMPLETED"
        row["evidence"] = f"{row['evidence']}; {proof}"
    if job_id := os.environ.get("SLURM_JOB_ID"):
        rows.append(dict(job_id=job_id, job_name=os.environ["SLURM_JOB_NAME"], stage="verify_release",
                         partition=os.environ["SLURM_JOB_PARTITION"], gpu_each=0, gpu_concurrent_cap=0,
                         status="COMPLETED", evidence="Verified aggregate files and release_manifest.json"))
    with ledger.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    inner(out)
    results(out)
    verify(out, v1)
    manifest = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    with (out / "release_manifest.json").open("x") as file:
        json.dump({"files": manifest}, file, sort_keys=True, indent=2)
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "templates", "v1", "out"):
        parser.add_argument(name, type=Path)
    stage(**vars(parser.parse_args()))
