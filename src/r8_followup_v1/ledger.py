"""Slurm run ledger of the follow-up from the job logs (sacct is not available on this cluster).
status: completed (an end= line and no traceback) | failed (traceback) | no end line. Jobs cancelled while pending never wrote a
log; they are added from --manual "jobid|name|status|note" entries."""
import argparse, datetime, glob, os, re
import pandas as pd

from .common import pub_man


def parse(f):
    txt = open(f, errors="replace").read()
    m = re.search(r"start=(\S+) commit=(\S+) dirty=(\d+) host=(\S+) job=(\d+)(?: gpu=([^a]*?))? args=(.*)", txt)
    end = re.search(r"end=(\S+)", txt)
    name = os.path.basename(f).rsplit("_", 1)[0]
    if not m:
        return None
    st = "failed (traceback)" if "Traceback" in txt else ("completed" if end else "no end line")
    s = datetime.datetime.fromisoformat(m.group(1))
    sec = (datetime.datetime.fromisoformat(end.group(1)) - s).total_seconds() if end else None
    part = "gpu" if "gpu=" in txt.split("\n")[0] else "cpu"
    return dict(slurm_job=int(m.group(5)), job_name=name, status=st, start=m.group(1), end=end.group(1) if end else "", seconds=sec, host=m.group(4),
                commit=m.group(2), dirty_files=int(m.group(3)), partition=part, cores=8 if part == "cpu" else 4, gpus=1 if part == "gpu" else 0,
                args=m.group(7).strip()[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", nargs="*", default=[])
    a = ap.parse_args()
    rows = [r for r in (parse(f) for f in sorted(glob.glob("assets/logs/r8f_*.out"))) if r]
    for m in a.manual:
        j, n, s, note = m.split("|")
        rows.append(dict(slurm_job=int(j), job_name=n, status=s, args=note, seconds=0, cores=0, gpus=0))
    df = pd.DataFrame(rows).sort_values("slurm_job")
    df.to_csv(pub_man("slurm_jobs.tsv"), sep="\t", index=False)
    done = df.seconds.fillna(0)
    print(df[["slurm_job", "job_name", "status", "seconds"]].to_string())
    print("cpu core-hours", float((done * df.cores.fillna(0)).sum() / 3600), "gpu hours", float((done * df.gpus.fillna(0)).sum() / 3600))


if __name__ == "__main__":
    main()
