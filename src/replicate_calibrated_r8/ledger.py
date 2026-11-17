"""Collect every run directory (including failed and cancelled ones) and the hand-kept job notes into RUN_LEDGER.tsv."""
import glob, json, os, re
from datetime import datetime
import pandas as pd

from .common import MAN, runs, public

KEYS = ["arm", "lr", "seed", "fold", "stopped", "steps_run", "max_steps", "lam", "n_parameters", "best_epoch", "val_delta_rmse_best", "n_ft_runs_with_divergence",
        "frozen_code_commit", "eval_fix", "n_rows", "gpu"]


def slurm_jobs():
    """One row per Slurm job from the job logs (the accounting database is not available on this cluster)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(os.environ["R8_ASSETS"], "logs", "r8_*.out"))):
        m = re.match(r"(r8_.+)_(\d+)\.out$", os.path.basename(f))
        if not m:
            continue                                               # per-task logs of a multi-task job
        txt = open(f, errors="replace").read()
        head = re.search(r"^start=(\S+) commit=(\S*) host=(\S+) job=(\S*)(.*)$", txt, re.M)
        end = re.search(r"^end=(\S+)\s*$", txt, re.M)
        gpu = bool(head and " gpu=" in head.group(5))
        multi = bool(head and " tasks=" in head.group(5))
        rc = re.findall(r"^task \d+ rc=(\d+)", txt, re.M)
        secs = (datetime.fromisoformat(end.group(1)) - datetime.fromisoformat(head.group(1))).total_seconds() if head and end else ""
        if not head:
            status = "no start line (failed before the step began)"
        elif not end:
            status = "no end line (failed, cancelled or still running)"
        elif multi and any(r != "0" for r in rc):
            status = f"completed with {sum(r != '0' for r in rc)} failed task(s) of {len(rc)}"
        else:
            status = "completed"
        rows.append(dict(slurm_job=m.group(2), job_name=m.group(1), status=status, start=head.group(1) if head else "", end=end.group(1) if end else "", seconds=secs,
                         host=head.group(3) if head else "", commit=head.group(2) if head else "", partition="gpu" if gpu else "cpu",
                         cores=4 if gpu else (20 if multi else 8), gpus=1 if gpu else 0, n_tasks=len(rc) if multi else 1,
                         args=re.sub(r"^ (gpu=.*? )?(args=|tasks=)", "", head.group(5) + " ").strip() if head else ""))
    df = pd.DataFrame(rows)
    df.to_csv(public("manifests", "replicate_calibrated_r8", "slurm_jobs.tsv"), sep="\t", index=False)
    s = pd.to_numeric(df.seconds, errors="coerce").fillna(0)
    tot = dict(n_jobs=len(df), n_not_completed=int((df.status != "completed").sum()), gpu_hours=float((s * df.gpus).sum() / 3600), cpu_core_hours_reserved=float((s * df.cores).sum() / 3600),
               wall_hours=float(s.sum() / 3600), note="from job-log start/end lines; jobs without an end line count as 0; cores are the reserved cores of the job script")
    print(json.dumps(tot, indent=1))
    return tot


def main():
    tot = slurm_jobs()
    json.dump(tot, open(public("manifests", "replicate_calibrated_r8", "compute_usage.json"), "w"), indent=1)
    rows = []
    for d in sorted(glob.glob(runs("*", "*"))):
        kind_dir, rid = d.split(os.sep)[-2:]
        f = os.path.join(d, "result.json")
        if not os.path.exists(f):
            rows.append(dict(run_id=rid, kind=kind_dir, status="no_result (failed, cancelled or still running)", slurm_job="", commit="", host="", seconds="", detail=""))
            continue
        r = json.load(open(f))
        rows.append(dict(run_id=rid, kind=r.get("kind", kind_dir), status=r.get("stopped", "completed"), slurm_job=r.get("slurm_job", ""), commit=r.get("commit", ""),
                         host=r.get("host", ""), seconds=r.get("seconds", ""), detail=json.dumps({k: r[k] for k in KEYS if k in r and r[k] not in (None, {}, False, "")}, sort_keys=True)))
    man = os.path.join(MAN, "run_ledger_manual.tsv")
    if os.path.exists(man):
        rows += pd.read_csv(man, sep="\t", dtype=str).fillna("").to_dict("records")
    df = pd.DataFrame(rows).fillna("")
    df.to_csv(public("reports", "replicate_calibrated_r8", "RUN_LEDGER.tsv"), sep="\t", index=False)
    cpu = pd.to_numeric(df.seconds, errors="coerce")
    print(df.groupby("kind").size().to_string())
    print("sum of run seconds by kind:\n", cpu.groupby(df.kind).sum().to_string())


if __name__ == "__main__":
    main()
