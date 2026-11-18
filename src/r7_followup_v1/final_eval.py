"""Existing-benchmark follow-up evaluation (one time). NOT a blind test: the test drugs of this benchmark were already
scored by earlier studies and by the original R7 evaluation.

The test deltas are opened only here, and only if
  * R7F_FINAL_EVAL=1,
  * manifests/r7_followup_v1/freeze_manifest.json is committed and the worktree is clean,
  * the original R7 one-time evaluation is observed complete (the path recorded in the freeze manifest exists),
  * the output directory does not exist yet (immutable).
Every model of the freeze manifest was trained in the `orig` context (original train / validation) by this study's
code on the frozen original chemistry backbones; checkpoints are verified by sha256.
With --mock the identical code runs on the VALIDATION units (wiring check; the test array is not opened)."""
import argparse, json, os, subprocess, time
import numpy as np
import torch

from src.evidence_gated_r7.data import Store, Knowledge
from .common import MAN, fruns, orig_runs, pman, dump, sha256, NULL_SEEDS
from .score_oof import tables
from .train import FoldStats, tensors, build, predict

FREEZE = os.path.join(MAN, "freeze_manifest.json")


def sealed_test_delta(S, idx):
    assert os.environ.get("R7F_FINAL_EVAL") == "1", "test deltas only inside the follow-up final evaluation job"
    assert subprocess.run(["git", "ls-files", "--error-unmatch", FREEZE], capture_output=True).returncode == 0, "follow-up freeze manifest not committed"
    assert subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() == "", "worktree not clean"
    idx = np.asarray(idx)
    assert (S.role[idx] == "test").all()
    t = np.load(os.path.join(os.environ["R4_PROC"], "unit_delta_test.npy"), mmap_mode="r")
    return np.asarray(t[S.units.test_idx.to_numpy()[idx]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    fr = json.load(open(FREEZE))
    S = Store(); st = FoldStats(S, S.train); K = Knowledge(len(S.groups))
    assert len(st.cells_without_train) == 0
    if a.mock:
        assert os.environ.get("R7F_FINAL_EVAL") != "1"
        te = S.val; y = S.delta(te); tag = "final_mock"; out = fruns("final", f"mock_{int(t0)}")
        role = "development(validation units; wiring check of the final evaluation code)"
    else:
        assert os.path.exists(fr["original_final_eval_observed"]["path"]), "the original R7 final evaluation is not observed complete"
        te = S.test; y = sealed_test_delta(S, te); tag = "final"; out = fruns("final", "followup_test_eval_v1")
        role = "existing-benchmark follow-up evaluation(original test drugs; previously exposed benchmark)"
    os.makedirs(out, exist_ok=False)
    dev = torch.device("cuda"); T = tensors(S, st, dev)
    man = json.load(open(os.path.join(MAN, "target_shuffle_manifest.json")))["mappings"]
    cache, hashes = {}, {}
    seeds = fr["seeds"]; keys = fr["keys"]

    def load(key, s):
        if (key, s) in cache:
            return cache[(key, s)]
        r = fr["runs"][f"{key}_s{s}"]
        assert sha256(r["checkpoint"]) == r["checkpoint_sha256"], r["run_id"]
        arm = r["arm"]; null_path = ""
        if arm in ("GS", "TDS"):
            m_ = man[f"orig_null{r['null_seed']}"]; null_path = m_["path"]
            assert sha256(null_path) == m_["sha256"] == r["null_mapping_sha256"]
        model, bank = build(arm, K, dev, null_path)
        sd = torch.load(r["checkpoint"], map_location=dev)
        if arm == "chem":           # original chem_only checkpoint: backbone weights only
            model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
        else:
            model.load_state_dict(sd)
        p = predict(model, T, bank, te, dev)
        f = os.path.join(out, f"{key}_s{s}.npy"); np.save(f, p.astype(np.float32)); hashes[f"{key}_s{s}.npy"] = sha256(f)
        rec = dict(best=dict(step=r.get("best_step", -1), score=r.get("sel_rmse_best", np.nan)), config=dict(n_params_correction=r.get("n_params_correction", 0)),
                   train_seconds=r.get("train_seconds"), gpu_max_mem_gb=r.get("gpu_max_mem_gb"), slurm_job=r.get("slurm_job"), host=r.get("host"))
        cache[(key, s)] = (p.astype(np.float64), rec, None, r["run_id"])
        return cache[(key, s)]

    blocks = [dict(name="test" if not a.mock else "validation", k=0, held=te, y=y, load=load)]
    tables(S, blocks, keys, seeds, tag, role, official=False, extra_pairs=fr.get("extra_pairs", []), n_boot=fr.get("n_boot", 5000), halves=False,
           unit_file=f"{tag}_unit_rmse.tsv.gz")
    dump(dict(tag=tag, job=os.environ.get("SLURM_JOB_ID"), commit=os.environ.get("R7F_COMMIT"), frozen_commit=fr.get("frozen_commit"), seconds=round(time.time() - t0),
              n_units=int(len(te)), prediction_dir=out, prediction_sha256=hashes), pman(tag, "final_eval_record.json"))


if __name__ == "__main__":
    main()
