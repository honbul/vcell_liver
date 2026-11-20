"""Freeze manifest of the follow-up configuration (Phase E). Reads only run records / checkpoints of the `orig` context
(original train / validation; no test delta) and the existence of the original R7 final-evaluation directory
(its contents are not read). Writes manifests/r7_followup_v1/freeze_manifest.json (to be committed before final_eval).

keys: C (original chem_only checkpoints, read only), M, G0, GR, GS142/143/144, plus the branch keys listed in
--branch_keys (e.g. TD TDS142 TDS143 TDS144) and --extra_pairs (branch contrasts, descriptive / as in DECISIONS.md)."""
import argparse, json, os, subprocess, time

from .common import MAN, fruns, orig_runs, sha256, dump, pub, NULL_SEEDS, MODEL_SEEDS

ORIG_CHEM = {42: "pilot_chem_lr1e-4_s42", 43: "sel_chem_lr1e-4_s43", 44: "sel_chem_lr1e-4_s44"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch_keys", nargs="*", default=[])
    ap.add_argument("--extra_pairs", nargs="*", default=[])
    ap.add_argument("--branch_core", nargs="*", default=[])
    ap.add_argument("--branch", default="none")
    a = ap.parse_args()
    keys = ["C", "M", "G0", "GR"] + [f"GS{n}" for n in NULL_SEEDS] + list(a.branch_keys)
    shuf = json.load(open(os.path.join(MAN, "target_shuffle_manifest.json")))["mappings"]
    runs = {}
    for s in MODEL_SEEDS:
        for key in keys:
            if key == "C":
                d = orig_runs("chem", ORIG_CHEM[s]); arm = "chem"; rid = ORIG_CHEM[s]; null = None
            else:
                arm = "GS" if key.startswith("GS") else ("TDS" if key.startswith("TDS") else key)
                null = int(key[len(arm):]) if arm in ("GS", "TDS") else None
                rid = f"orig_s{s}_{key}"; d = fruns(arm, rid)
            rec = json.load(open(os.path.join(d, "record.json")))
            assert rec["config"]["seed"] == s and rec["config"]["arm"] == arm, rid
            ck = os.path.join(d, "best.pt")
            r = dict(run_id=rid, arm=arm, checkpoint=ck, checkpoint_sha256=sha256(ck), best_step=rec["best"]["step"], sel_rmse_best=rec["best"]["score"],
                     n_params_correction=rec["config"].get("n_params_correction", 0), train_seconds=rec.get("train_seconds", rec.get("seconds")),
                     gpu_max_mem_gb=rec.get("gpu_max_mem_gb"), slurm_job=rec.get("slurm_job"), host=rec.get("host"), commit=rec.get("commit"),
                     source="original R7 run (read only)" if key == "C" else "follow-up run, ctx orig")
            if key != "C":
                assert rec["config"]["ctx"] == "orig"
                if "checkpoint_sha256" in rec:
                    assert rec["checkpoint_sha256"] == r["checkpoint_sha256"], rid
            if null is not None:
                m = shuf[f"orig_null{null}"]
                assert sha256(m["path"]) == m["sha256"] == rec["config"]["null_mapping_sha256"]
                r.update(null_seed=null, null_mapping_sha256=m["sha256"])
            runs[f"{key}_s{s}"] = r
    orig_eval = os.path.join(os.environ["R7_ORIG_RUNS"], "test_eval", "test_eval_v1")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    fr = dict(track="R7", version="followup-v1", frozen_commit=head, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), branch=a.branch,
              keys=keys, seeds=list(MODEL_SEEDS), extra_pairs=list(a.extra_pairs), branch_core=list(a.branch_core), n_boot=5000, runs=runs,
              original_final_eval_observed=dict(path=orig_eval, exists_at_freeze=os.path.exists(orig_eval),
                                                note="existence of the directory only; its contents were not read by the follow-up"),
              note="final_eval opens the test deltas once, only with R7F_FINAL_EVAL=1, this file committed and a clean worktree")
    dump(fr, pub(MAN, "freeze_manifest.json"))
    print(json.dumps(dict(n_runs=len(runs), keys=keys, frozen_commit=head, orig_eval_exists=fr["original_final_eval_observed"]["exists_at_freeze"])))


if __name__ == "__main__":
    main()
