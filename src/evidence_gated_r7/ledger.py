"""RUN_LEDGER.tsv from the run records (completed runs) plus the recorded incomplete / refused jobs."""
import glob, json, os
import pandas as pd

from .common import pman

INCOMPLETE = [
    dict(run_id="p_* (15 runs)", kind="pilot (first)", status="cancelled after ~37 s (jobs 25732-25746); KG Phase-0 correction D-2; early validation lines printed by fast arms, not used"),
    dict(run_id="smoke_{chem,concat,gated,fixed}", kind="smoke", status="300 updates, timing / memory only (jobs 25727-25730)"),
    dict(run_id="xp_timing", kind="xpert", status="2 epochs timing run (job 25788) used for the epoch cap D-3 and the wrapper check (job 25792)"),
    dict(run_id="xp_e100_s{42,43,44}", kind="xpert", status="25790 failed (timestamp folder collision); 25789 / 25791 cancelled at ~2 epochs (shared work dir), D-4; relaunched as xpr_e100_*"),
    dict(run_id="xpe_e100_s{42,43,44}", kind="xpert (addendum A-2)", status="input-matched XPert (ECFP folded to 1024, HG zero), jobs 26037-26039; addendum freeze 28cca70; eval job 26569"),
    dict(run_id="-", kind="setup", status="refused / failed setup jobs: 25718 (mamba binary), 25766 (huggingface_hub missing), 25768 (upstream tqdm bug), 25769-25772 (NFS stale clean-tree check), 25773 (gene-id alignment); none trained a model"),
]


def main():
    rows = []
    for f in sorted(glob.glob(pman("runs", "*.json"))):
        r = json.load(open(f))
        if "config" not in r:
            continue
        c = r["config"]; b = r.get("best", {})
        rows.append(dict(run_id=r["run_id"], kind=r["kind"], arm=c.get("arm"), aug=c.get("aug"), graph=c.get("graph"), evidence=c.get("evidence"),
                         lr=c.get("lr"), seed=c.get("seed"), updates=c.get("updates"), best_val_source_rmse=b.get("score"), best_step=b.get("step"),
                         seconds=r.get("seconds"), gpu_mem_gb=r.get("gpu_max_mem_gb"), slurm_job=r.get("slurm_job"), commit=r.get("commit"), status="completed"))
    for f in sorted(glob.glob(pman("runs", "xpert_*_pred_*.json"))):
        r = json.load(open(f))
        rows.append(dict(run_id=r["run"], kind="xpert_predict", status=f"split={r['split']} treated_slot_max_abs_diff={r['treated_slot_max_abs_diff']}",
                         slurm_job=r.get("job")))
    t = pd.concat([pd.DataFrame(rows), pd.DataFrame(INCOMPLETE)], ignore_index=True)
    t.to_csv(pman("RUN_LEDGER.tsv"), sep="\t", index=False)
    print(t.to_string(max_rows=200))


if __name__ == "__main__":
    main()
