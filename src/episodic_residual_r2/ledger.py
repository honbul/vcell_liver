"""Collect every run directory (including diverged and failed ones) into run_ledger.tsv."""
import glob, json, os
import pandas as pd

from .common import MAN, runs, public

KEYS = ["lr", "seed", "variant", "train_shuffle_residual", "stopped", "best_step", "steps_run", "val_delta_rmse", "val_mean_offset", "lam", "selected",
        "n_ft_runs", "n_ft_runs_with_divergence", "n_parameters", "note", "eval_fix"]


def main():
    rows = []
    for d in sorted(glob.glob(runs("*", "*"))):
        kind_dir, rid = d.split(os.sep)[-2:]
        f = os.path.join(d, "result.json")
        if not os.path.exists(f):
            rows.append(dict(run_id=rid, kind=kind_dir, status="no_result (crashed, cancelled or still running)", slurm_job="", commit="", host="", seconds="", detail=""))
            continue
        r = json.load(open(f))
        st = r.get("stopped", "completed")
        rows.append(dict(run_id=rid, kind=r.get("kind", kind_dir), status=st, slurm_job=r.get("slurm_job", ""), commit=r.get("commit", ""), host=r.get("host", ""),
                         seconds=r.get("seconds", ""), detail=json.dumps({k: r[k] for k in KEYS if k in r and r[k] not in (None, {}, False)}, sort_keys=True)))
    man = os.path.join(MAN, "run_ledger_manual.tsv")
    if os.path.exists(man):
        m = pd.read_csv(man, sep="\t", dtype=str).fillna("")
        rows += m.to_dict("records")
    df = pd.DataFrame(rows)
    df.to_csv(public("manifests", "episodic_residual_r2", "run_ledger.tsv"), sep="\t", index=False)
    print(df[["run_id", "kind", "status", "slurm_job", "seconds"]].to_string())


if __name__ == "__main__":
    main()
