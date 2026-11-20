"""Collect R12 small result tables, per-job runs, and server artifact receipts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil

import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, table



def replica_fits(assets, names):
    for name in names:
        path = assets / name
        for file in sorted(path.glob("outer*-*.json")):
            yield name, read_json(file)


def runtime_rows(assets, config_path):
    config_sha = sha256(config_path)
    seen = set()
    for stage, result in replica_fits(assets, ("inner-core-v2", "inner-expanded", "development", "dev-scaffold")):
        variant = "scaffold" if stage == "dev-scaffold" else "primary"
        code = result.get("code")
        digest = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest() if code else "see log and first-stage commit"
        for record in result["ledger"]:
            candidate = record.get("candidate", {})
            key = (variant, result["role"], result["outer"], record.get("inner"), record["phase"],
                   json.dumps(candidate, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            yield ledger_row(result["job"], result["host"], config_sha, digest, variant, result["outer"],
                             result["role"], record)
    for name in ("final-predictions", "pred-scaffold"):
        stage = "primary" if name == "final-predictions" else "scaffold"
        for file in sorted((assets / name).glob("outer*/[AB]/receipt.json")):
            receipt = read_json(file)
            code = hashlib.sha256(json.dumps(receipt["code"], sort_keys=True).encode()).hexdigest()
            runtime = read_json(file.parent / "runtime.json")
            for record in [*read_json(file.parent / "joint_dev.json"), *runtime["ledger"]]:
                yield ledger_row(receipt["job"], receipt["host"], config_sha, code, stage, receipt["outer"],
                                 receipt["role"], record)


def ledger_row(job, host, config_sha, code, variant, outer, role, record):
    candidate = record.get("candidate", {})
    runtime = record.get("runtime", {})
    crossfit = record.get("calibration_runs", [])
    return dict(job=job, host=host, variant=variant, outer=outer, role=role, inner=record.get("inner"),
                stage=record["phase"], arm=record.get("arm", candidate.get("arm", "")),
                seed=runtime.get("seed", record.get("seed", "")), device=runtime.get("device", ""),
                seconds=runtime.get("seconds", 0), calibration_seconds=sum(x.get("seconds", 0) for x in crossfit),
                epochs=runtime.get("epochs", 0), updates=runtime.get("updates", 0),
                params=runtime.get("params", 0), dense_macs=runtime.get("dense_macs", 0),
                max_gpu_bytes=runtime.get("peak_gpu_bytes", 0), state="completed", config_sha256=config_sha,
                pipeline_code_hash=code, settings=json.dumps(candidate, sort_keys=True),
                calibration_coefficients=record.get("coefficients", ""))


def jobs(log_dir):
    device_job = re.compile(r"^(inner-|primary-|scaffold-|uce-|scgpt-|smoke-|secondary-)")
    for path in sorted(log_dir.glob("*.out")):
        content = path.read_text()
        found = re.search(r"job=(\d+) host=(\S+) start=(\S+)", content)
        if found is None:
            continue
        job, host, start = found.groups()
        start_at = datetime.fromisoformat(start)
        end_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        memory = [int(n) for n in re.findall(r"Maximum resident set size \(kbytes\): (\d+)", content)]
        exits = [int(n) for n in re.findall(r"Exit status: (\d+)", content)]
        failure = "Traceback" in content or any(exits)
        cpus = 2 if path.name.startswith(("split-", "preflight-")) else (
            4 if path.name.startswith(("depmap-", "chemmean-", "feasibility-")) else 8)
        yield dict(job=job, host=host, output=path.name, alloc_gpu=int(bool(device_job.search(path.name))),
                   cpu_slots=cpus, started=start_at.isoformat(), last_log=end_at.isoformat(),
                   elapsed_seconds=max(0.0, (end_at - start_at).total_seconds()),
                   max_rss_kib=max(memory, default=0),
                   state="failed_attempt_preserved" if failure else "log_finished",
                   terminal_detail=next((line for line in reversed(content.splitlines())
                                         if "Error:" in line or "Exception:" in line), "") if failure else "")


def data_receipts(cfg, assets):
    r11 = Path(cfg["r11_assets"])
    paths = [
        ("Tahoe metadata", r11 / "metadata/receipt.json", "R11 reusable pinned mirror descriptors"),
        ("Tahoe source", r11 / "source/receipt.json", "R11 reusable normalized inputs"),
        ("R11 final", r11 / "final/receipt.json", "R11 source-fixed panel, project-seen replay only"),
        ("R11 ECFP4 and CLS", r11 / "features/receipt.json", "reuse entity alignment and frozen vectors"),
        ("DMSO original cells", assets / "cells-v2/receipt.json", "C4, C7, C8, L4, L5, L6"),
        ("ChemBERTa mean", assets / "chemmean/receipt.json", "E3 validated against R11 CLS"),
        ("scGPT whole-human", assets / "scgpt/receipt.json", "C7 same-cell comparison"),
        ("UCE 4-layer", assets / "uce/receipt.json", "C8 same-cell comparison"),
        ("DepMap 25Q3", assets / "depmap/receipt.json", "C9 RRID-matched annotations"),
        ("Additional doses", assets / "doses/source/receipt.json", "D1 0.05/5 µM, fit-group only"),
        ("Primary stage", assets / "staged/receipt.json", "held-out response arrays isolated"),
        ("Scaffold stage", assets / "staged-scaffold/receipt.json", "separate scaffold-linked folds"),
    ]
    return paths


def artifact_paths(assets, out):
    paths = []
    for folder in ("final-predictions", "pred-scaffold", "scored", "scored-scaffold", "replay", "transfer"):
        root = assets / folder
        receipts = list(root.glob("outer*/[AB]/receipt.json"))
        if (root / "receipt.json").exists():
            receipts.append(root / "receipt.json")
        if (root / "freeze.json").exists():
            receipts.append(root / "freeze.json")
        if (root / "score_receipt.json").exists():
            receipts.append(root / "score_receipt.json")
        for receipt in receipts:
            index = read_json(receipt)
            paths.append(dict(artifact=str(receipt.relative_to(assets)), path=str(receipt.resolve()),
                              sha256=sha256(receipt), bytes=receipt.stat().st_size))
            files = index.get("files", index.get("truth_sha256", {}))
            for filename, expected in files.items():
                path = receipt.parent / filename
                if not path.exists():
                    raise ValueError(f"Artifact path in receipt is missing: {path}")
                paths.append(dict(artifact=str(path.relative_to(assets)), path=str(path.resolve()),
                                  sha256=expected, bytes=path.stat().st_size))
    pd.DataFrame(paths).drop_duplicates().to_csv(out / "artifacts.tsv", sep="\t", index=False)


def collect(cfg, args):
    assets = args.assets
    args.out.mkdir(parents=True, exist_ok=False)
    report = args.out
    round_jobs = pd.DataFrame(jobs(args.logs))
    round_jobs.to_csv(report / "jobs.tsv", sep="\t", index=False)
    fits = pd.DataFrame(runtime_rows(assets, args.config))
    fits.to_csv(report / "run_ledger.tsv", sep="\t", index=False)
    data = []
    receipts = report / "asset_receipts"
    receipts.mkdir()
    for i, (name, path, use) in enumerate(data_receipts(cfg, assets)):
        if path.exists():
            relative = f"asset_receipts/{i:02d}.json"
            shutil.copyfile(path, report / relative)
            data.append(dict(asset=name, state="verified", use=use, location=str(path.resolve()),
                             receipt=relative, sha256=sha256(path)))
        else:
            data.append(dict(asset=name, state="missing", use=use, location=str(path.resolve())))
    pd.DataFrame(data).to_csv(report / "asset_inventory.tsv", sep="\t", index=False)
    copies = [
        ("freeze.json", "freeze.json"), ("audit.json", "audit.json"),
        ("scaffold-freeze.json", "scaffold_freeze.json"), ("audit-scaffold.json", "scaffold_audit.json"),
        ("scaffold/strict_cell_support.tsv", "strict_cell_support.tsv"),
        ("scaffold/group_scaffolds.tsv", "group_scaffolds.tsv"),
        ("scored-scaffold/paired_effects.tsv", "scaffold_paired_effects.tsv"),
        ("scored-scaffold/arm_metrics.tsv", "scaffold_metrics.tsv"),
        ("replay/arm_metrics.tsv", "replay_metrics.tsv"),
        ("transfer/arm_metrics.tsv", "cell_transfer_metrics.tsv"),
        ("scored/paired_effects.tsv", "paired_effects.tsv"),
        ("scored/response_pca_diagnostics.tsv", "response_pca_diagnostics.tsv"),
        ("scored/arm_metrics_by_role_fold.tsv", "arm_metrics_by_role_fold.tsv"),
    ]
    for source, name in copies:
        shutil.copyfile(assets / source, report / name)

    provenance, diag = {}, []
    for k in range(3):
        for role in ("A", "B"):
            sel_path = assets / f"final-predictions/outer{k}/{role}"
            record = read_json(sel_path / "selection.json")
            dev = read_json(assets / f"development/outer{k}-{role}.json")
            provenance[f"outer{k}/{role}"] = dict(
                **record, inner_grid=dev["records"], inner_shift=dev["shift"],
                inference_receipt=sha256(sel_path / "receipt.json"),
                final_calibrations=[row for row in read_json(sel_path / "runtime.json")["ledger"]
                                    if row["phase"] == "outer_calibration"],
            )
            rows = table(sel_path / "rows.tsv")
            for name in rows.columns:
                if name.startswith("C6_") and name.endswith("_changed"):
                    mask = rows[name].astype(bool)
                    diag.append(dict(outer=k, role=role, diagnostic=name.removesuffix("_changed"),
                                     changed_rows=int(mask.sum()), total_rows=len(mask),
                                     group_coverage=rows.loc[mask, GROUP].nunique()))
    write_json(report / "selection_by_role_fold.json", provenance)
    pd.DataFrame(diag).to_csv(report / "context_replacement_coverage.tsv", sep="\t", index=False)
    metrics = table(assets / "scored/arm_metrics.tsv")
    per_arm = fits.loc[fits.stage.eq("outer_fit") & fits.variant.eq("primary")].groupby("arm").agg(
        fitted_seed_models=("job", "size"), train_seconds=("seconds", "sum"),
        largest_parameter_count=("params", "max"), max_allocated_gpu_bytes=("max_gpu_bytes", "max"),
    )
    metrics.join(per_arm, on="arm").to_csv(report / "arm_metrics.tsv", sep="\t", index=False)
    shutil.copyfile(assets / "scored/context_diagnostics.tsv", report / "context_diagnostics.tsv")
    interaction = table(assets / "scored/loss_structure_interaction.tsv")
    paired = table(assets / "scored/paired_effects.tsv")
    effect = paired.loc[paired.contrast.eq("J1_interaction")].iloc[0]
    interaction = pd.concat([interaction, pd.DataFrame([dict(
        arm="J1_interaction", scope="all", groups=effect.groups, contrast=effect.effect,
        lower=effect.lower, upper=effect.upper, confidence=effect.confidence,
    )])], ignore_index=True)
    interaction.to_csv(report / "loss_structure_interaction.tsv", sep="\t", index=False)
    artifact_paths(assets, report)
    write_json(report / "budget_usage.json", dict(
        accounting="Slurm log start to last-written log time; total does not include queue waits.",
        observed_gpu_hours=float((round_jobs.alloc_gpu * round_jobs.elapsed_seconds).sum() / 3600),
        observed_cpu_slot_hours=float((round_jobs.cpu_slots * round_jobs.elapsed_seconds).sum() / 3600),
        model_max_gpu_bytes=int(fits.max_gpu_bytes.max()),
        largest_logged_job_rss_kib=int(round_jobs.max_rss_kib.max()),
        scratch_gib=sum(f.stat().st_size for f in assets.rglob("*") if f.is_file()) / (1 << 30),
        host=platform.node(),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(config(args.config), args)


if __name__ == "__main__":
    main()
