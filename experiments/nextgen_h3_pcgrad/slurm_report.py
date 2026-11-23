"""Reduce H3 Slurm observations to a terminal-state ledger."""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# The controller returned these terminal records before its five-minute retention expired.
EARLY = '''
JobId=34524 JobName=ng-h3-prep JobState=COMPLETED ExitCode=0:0 RunTime=00:00:15 StartTime=2026-11-23T12:39:35 EndTime=2026-11-23T12:39:50
JobId=34527 JobName=ng-h3-invariants JobState=COMPLETED ExitCode=0:0 RunTime=00:00:09 StartTime=2026-11-23T12:42:03 EndTime=2026-11-23T12:42:12
'''
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}


def records(log):
    latest = {}
    for line in (EARLY + log.read_text()).splitlines():
        raw = {word.split("=", 1)[0]: word.split("=", 1)[1] for word in line.split() if "=" in word}
        if not raw.get("JobName", "").startswith("ng-h3-") or raw.get("JobState") not in TERMINAL:
            continue
        job = raw.get("ArrayJobId", raw["JobId"])
        task = raw.get("ArrayTaskId", "-")
        if task == "-" or task.isdecimal() or raw["JobState"] == "CANCELLED":
            latest[job, task] = raw
    return latest


def elapsed(raw):
    text = raw["RunTime"]
    days, hour_text = text.split("-", 1) if "-" in text else ("0", text)
    hours, minutes, seconds = list(map(int, hour_text.split(":")))
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def render(raw_file, out, required):
    known = records(raw_file)
    jobs = defaultdict(list)
    allocations = []
    for (job, task), values in known.items():
        if ":a40:1" in values.get("TresPerNode", "") and values["JobState"] == "COMPLETED":
            allocations.extend([(datetime.fromisoformat(values["StartTime"]), 1),
                                (datetime.fromisoformat(values["EndTime"]), -1)])
        jobs[job].append((task, values))
    concurrent = high = 0
    for _, event in sorted(allocations):
        concurrent += event
        high = max(high, concurrent)
    if high > 2 or concurrent != 0:
        raise ValueError("H3 data contains conflicting GPU allocation records")
    for item in required:
        job, expected = item.split(":")
        terminal = jobs.get(job, ())
        evidence = [row for _, row in terminal if row["JobState"] == "COMPLETED" and row["ExitCode"] == "0:0"]
        if len(evidence) != int(expected):
            raise ValueError(f"Missing terminal Slurm records: {job}, {len(evidence)}/{expected}")
    with out.open("w", newline="") as file:
        names = ("slurm_job", "array_task", "name", "state", "exit_code", "gpu_per_task",
                 "array_gpu_throttle", "started_kst", "ended_kst", "runtime_seconds", "h3_peak_gpu")
        writer = csv.DictWriter(file, fieldnames=names, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for (job, task), row in sorted(known.items(), key=lambda item: (int(item[0][0]), item[0][1])):
            gpu = int(":a40:1" in row.get("TresPerNode", ""))
            writer.writerow(dict(zip(names, (
                job, task, row["JobName"], row["JobState"], row["ExitCode"], gpu,
                row.get("ArrayTaskThrottle", "-") if gpu else "-", row["StartTime"], row["EndTime"],
                elapsed(row), high,
            ))))
    print(f"Terminal H3 records {len(known)}, peak simultaneous completed A40 allocations {high}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scontrol", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--require", nargs="+", required=True, metavar="JOB:N_TASKS")
    args = parser.parse_args()
    render(args.scontrol, args.out, args.require)
