"""Run ledger of the track: every GPU trajectory record (pilot + main) and the CPU job logs -> run_ledger.tsv + totals,
plus disk use of the track's assets."""
import glob, json, os, re, subprocess
import pandas as pd

from .common import runs, pman, dump, stamp


def main():
    rows = []
    for f in sorted(glob.glob(runs("traj", "*", "record.json"))):
        r = json.load(open(f))
        rows.append(dict(run_id=r["run_id"], kind="gpu_trajectory", scope=r["traj"]["scope"], recipe=r["traj"]["recipe"], variant=r["traj"]["variant"],
                         seed=r["traj"]["seed"], epochs=r["epochs_run"], gpu=r["gpu"], slurm_job=r["slurm_job"], commit=r["commit"],
                         gpu_hours=r["seconds_total"] / 3600, sec_train_epoch=r["mean_sec_train_epoch"], sec_select_epoch=r["mean_sec_select_epoch"],
                         max_mem_gb=r["max_mem_gb"], selected_epochs=json.dumps({k: v["epoch"] for k, v in r["selectors"].items()})))
    logs = os.path.join(os.environ["R7V2_ASSETS"], "logs")
    for f in sorted(glob.glob(os.path.join(logs, "r7v2_*.out"))):
        t = open(f).read()
        m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t)
        if not m:
            continue
        p = [float(x) for x in m.group(1).split(":")]
        sec = p[-1] + 60 * p[-2] + (3600 * p[-3] if len(p) > 2 else 0)
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
        cpu = re.search(r"cpu_step.sbatch|-c (\d+)", t)
        rows.append(dict(run_id=os.path.basename(f), kind="cpu_job", wall_hours=sec / 3600, cpu_hours_8c=8 * sec / 3600,
                         max_rss_gb=int(rss.group(1)) / 2 ** 20 if rss else None))
    df = pd.DataFrame(rows)
    df.to_csv(pman("run_ledger.tsv"), sep="\t", index=False)
    du = subprocess.run(["du", "-sb", os.environ["R7V2_ASSETS"]], capture_output=True, text=True).stdout.split()[0]
    g = df[df.kind == "gpu_trajectory"]
    tot = dict(gpu_hours_total=float(g.gpu_hours.sum()), gpu_hours_pilot=float(g[g.run_id.str.contains("_pilot|_tA")].gpu_hours.sum()),
               gpu_hours_main=float(g[~g.run_id.str.contains("_pilot|_tA")].gpu_hours.sum()),
               gpu_hours_by_type=g.groupby("gpu").gpu_hours.sum().to_dict(), n_trajectories_main=int((~g.run_id.str.contains("_pilot|_tA")).sum()),
               cpu_job_hours_8core=float(df[df.kind == "cpu_job"].cpu_hours_8c.sum()), assets_bytes=int(du), assets_gb=int(du) / 1e9, **stamp())
    dump(tot, pman("run_ledger_totals.json"))
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
