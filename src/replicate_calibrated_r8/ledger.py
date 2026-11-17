"""Collect every run directory (including failed and cancelled ones) and the hand-kept job notes into RUN_LEDGER.tsv."""
import glob, json, os
import pandas as pd

from .common import MAN, runs, public

KEYS = ["arm", "lr", "seed", "fold", "stopped", "steps_run", "max_steps", "lam", "n_parameters", "best_epoch", "val_delta_rmse_best", "n_ft_runs_with_divergence",
        "frozen_code_commit", "eval_fix", "n_rows", "gpu"]


def main():
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
