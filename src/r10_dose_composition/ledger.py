"""Collect run_ledger rows from R10 Slurm logs (assets/logs/r10_*.out): job, host, git SHA, command,
start/end, wall time, CPU seconds (user+sys from /usr/bin/time), max RSS, exit status.
Purely descriptive; no outcome numbers."""
from __future__ import annotations

import csv
import glob
import os
import re
import sys


def _get(txt, pat):
    m = re.search(pat, txt, re.M)
    return m.group(1).strip() if m else ""


def parse(path):
    txt = open(path, errors="replace").read()
    rss = _get(txt, r"Maximum resident set size \(kbytes\): (\d+)")
    u = _get(txt, r"User time \(seconds\): (\S+)")
    s = _get(txt, r"System time \(seconds\): (\S+)")
    name = os.path.basename(path)[:-4]
    return dict(log=os.path.basename(path), job_name=re.sub(r"_\d+(_\d+)?$", "", name),
                slurm_job=_get(txt, r"job=(\d+)"), host=_get(txt, r"host=(\S+)"), git=_get(txt, r"git=([0-9a-f]+)"),
                dirty=_get(txt, r"dirty=(\d+)"), start=_get(txt, r"start=(\S+)"), end=_get(txt, r"^end=(\S+)"),
                cmd=_get(txt, r"^cmd: (.*)$")[:400],
                wall=_get(txt, r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)"),
                cpu_seconds=f"{float(u) + float(s):.0f}" if u and s else "",
                max_rss_gb=f"{int(rss) / 1e6:.2f}" if rss else "",
                exit_status=_get(txt, r"Exit status: (\d+)"), traceback="yes" if "Traceback" in txt else "")


def main():
    rows = [parse(p) for p in sorted(glob.glob(sys.argv[1] + "/r10_*.out"))]
    w = csv.DictWriter(open(sys.argv[2], "w"), fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
