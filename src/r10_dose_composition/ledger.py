"""Collect run_ledger rows from R10 Slurm logs (assets/logs/r10_*.out): job, host, git SHA, command,
start/end, wall time, max RSS, exit status. Purely descriptive; no outcome numbers."""
from __future__ import annotations

import csv
import glob
import os
import re
import sys


def parse(path):
    txt = open(path, errors="replace").read()
    g = lambda pat: (re.search(pat, txt, re.M).group(1).strip() if re.search(pat, txt, re.M) else "")
    return dict(log=os.path.basename(path), job_name=os.path.basename(path).split("_")[0] + "_" +
                os.path.basename(path).split("_")[1], host=g(r"host=(\S+)"), slurm_job=g(r"job=(\d+)"),
                git=g(r"git=([0-9a-f]+)"), dirty=g(r"dirty=(\d+)"), start=g(r"start=(\S+)"), end=g(r"^end=(\S+)"),
                cmd=g(r"^cmd: (.*)$")[:400], wall=g(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)"),
                max_rss_gb=(f"{int(g(r'Maximum resident set size \(kbytes\): (\d+)')) / 1e6:.2f}"
                            if g(r"Maximum resident set size \(kbytes\): (\d+)") else ""),
                user_cpu_s=g(r"User time \(seconds\): (\S+)"), exit_status=g(r"Exit status: (\d+)"),
                traceback="yes" if "Traceback" in txt else "")


def main():
    rows = [parse(p) for p in sorted(glob.glob(sys.argv[1] + "/r10_*.out"))]
    w = csv.DictWriter(open(sys.argv[2], "w"), fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
