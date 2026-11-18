"""Post-freeze evaluation of the frozen follow-up candidate and its baselines on the ALREADY-SEEN panels, one pass:
  confirmation   the 9 confirmation families (support seeds 1-3, K = 8/16/32; original confirmation rule)
  matched        the confirmation families' matched-plate E1 / E2 episodes (original confirmation rule: up to 6 pairs, seeds 1-3)
  hcc            HUH7 / JHH5 / JHH7 with the original frozen support / query lists (split seeds 1-3); zero target-validation labels
This is an existing-benchmark follow-up evaluation, not a blind test. No constant, scale, gate or learning rate is chosen here.
The guarded loader refuses to run unless the follow-up freeze manifest is committed in HEAD and the code is unchanged since.
"""
import argparse, json, os, subprocess, time
import numpy as np
import pandas as pd
import torch

from src.replicate_calibrated_r8.data import Episode, overlap, Features
from src.replicate_calibrated_r8.build_store import row_weights
from src.replicate_calibrated_r8 import diag as D
from src.episodic_residual_r2.build_units import rows_to_units
from src.episodic_residual_r2.metrics import per_unit
from .common import MAN, TARGETS, orig_man, sha256, dump, new_run_dir, run_record
from .partitions import Bank
from . import engine as Eng
from . import fmethods as F
from .matched import matched_episodes, load_rows

torch.set_grad_enabled(False)
FREEZE = os.path.join(MAN, "followup_freeze.json")
B_MAN = "manifests/transigen_adaptation_b1/build_manifest.json"


def assert_frozen():
    assert os.environ.get("R8F_FINAL_EVAL") == "1", "already-seen panels are only read by the post-freeze evaluation job"
    r = subprocess.run(["git", "show", f"HEAD:{FREEZE}"], capture_output=True)
    assert r.returncode == 0, "follow-up freeze manifest is not committed"
    assert r.stdout == open(FREEZE, "rb").read(), "freeze manifest differs from the committed version"
    fz = json.loads(r.stdout)
    assert subprocess.run(["git", "merge-base", "--is-ancestor", fz["code_commit"], "HEAD"]).returncode == 0
    changed = subprocess.run(["git", "diff", "--stat", fz["code_commit"], "HEAD", "--", "src/r8_followup_v1", "configs/r8_followup_v1"],
                             capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], capture_output=True, text=True).stdout.strip()
    fix = os.environ.get("R8F_EVAL_FIX", "")
    assert (changed == "" and dirty == "") or fix, f"code changed after the freeze without a recorded reason:\n{changed}\n{dirty}"
    for k in ("constants", "pairs", "jconstants"):
        if k in fz:
            assert sha256(fz[k]["path"]) == fz[k]["sha256"], f"{k} file changed after the freeze"
    return fz, fix


def load_target(t):
    d = os.environ["TB_PROC"]
    man = json.load(open(B_MAN))
    for f in ("rows.tsv", "x0_all.npy", "x1_target.npy"):
        assert sha256(os.path.join(d, f)) == man["files"][f]["sha256"]
    rows = pd.read_csv(os.path.join(d, "rows.tsv"), sep="\t")
    n_src = man["n_source_rows"]
    r = rows[(rows.role == "target") & (rows.cell == t)]
    assert len(r) and (r.row_id >= n_src).all()
    x0 = np.asarray(np.load(os.path.join(d, "x0_all.npy"), mmap_mode="r")[r.row_id.to_numpy()])
    x1 = np.asarray(np.load(os.path.join(d, "x1_target.npy"), mmap_mode="r")[r.row_id.to_numpy() - n_src])
    return r.reset_index(drop=True), x0, x1


def hcc_episode(st, t, split, lists, rows, tab, ud, ux):
    """hcc_eval.build_episode (regime 'hcc', every kept source family behind f0) plus the planned query-support plate overlap."""
    pos = {g: i for i, g in enumerate(tab.drug_group)}
    gi = tab.drug_group.map(st.groups.set_index("drug_group").group_idx).to_numpy()
    sup = [pos[g] for g in lists["support_order"]]
    qn = list(lists["query_test"]) + list(lists["query_val_secondary"])
    qry = [pos[g] for g in qn]
    assert not (set(sup) & set(qry)) and len(set(qry)) == len(qry)
    f0s, n0s = st.f0("hcc", gi[sup], allow_missing=True)
    f0q, n0q = st.f0("hcc", gi[qry], allow_missing=True)
    w = row_weights(rows)
    by = {}
    for dg, pl, wi in zip(rows.drug_group, rows.det_plate, w):
        by.setdefault(dg, ([], []))
        by[dg][0].append(pl)
        by[dg][1].append(wi)
    o, q = overlap([by[g][0] for g in lists["support_order"]], [by[g][1] for g in lists["support_order"]])
    plates = sorted(set(rows.det_plate))
    pp = {p: i for i, p in enumerate(plates)}

    def Wm(names):
        M_ = np.zeros((len(names), len(plates)))
        for i, g in enumerate(names):
            for p, x in zip(*by[g]):
                M_[i, pp[p]] += x
        return M_
    o_qs = Wm(qn) @ Wm(list(lists["support_order"])).T
    t_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    ep = Episode(name=t, family=t, seed=int(split), g_s=gi[sup], g_q=gi[qry], x0_s=t_(ux[sup]), x0_q=t_(ux[qry]), f0_s=f0s, f0_q=f0q, n0_s=n0s, n0_q=n0q,
                 y_s=t_(ud[sup]), y_q=t_(ud[qry]), o_s=t_(o), q_s=t_(q), plates_s=list(tab.plates.to_numpy()[sup]) if "plates" in tab else [],
                 plates_q=list(tab.plates.to_numpy()[qry]) if "plates" in tab else [])
    ep.o_qs = t_(o_qs)
    is_test = np.array([True] * len(lists["query_test"]) + [False] * len(lists["query_val_secondary"]))
    return ep, qn, is_test


def unit_table(pred, truth, names, is_test, extra):
    out = []
    for qset, m in (("test", is_test), ("test_plus_val", np.ones(len(is_test), bool))):
        pu = per_unit(pred[m], truth[m])
        rr, leg, _ = F.retrieval(pred[m].astype(np.float64), truth[m].astype(np.float64))
        pc, tc = pred[m] - pred[m].mean(0, keepdims=True), truth[m] - truth[m].mean(0, keepdims=True)
        rrc, legc, _ = F.retrieval(pc.astype(np.float64), tc.astype(np.float64))
        pu["mrr_tie"], pu["mrr_tie_centred"], pu["mrr_legacy_centred"] = rr, rrc, legc
        pu.insert(0, "drug_group", np.array(names)[m])
        pu.insert(0, "is_test", is_test[m])
        pu.insert(0, "query_set", qset)
        for k, v in extra.items():
            pu.insert(0, k, v)
        out.append(pu)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--parts", default="confirmation,matched,hcc")
    a = ap.parse_args()
    t0 = time.time()
    fz, fix = assert_frozen()
    out = new_run_dir("final_eval", a.run_id)
    const = json.load(open(fz["constants"]["path"]))
    assert const["partition"] == "final_inner"
    jc = json.load(open(fz["jconstants"]["path"])) if "jconstants" in fz else None
    assert jc is None or jc["partition"] == "final_inner"
    pairs = pd.read_csv(fz["pairs"]["path"], sep="\t")
    fx = Eng.load_fixed(orig_man("anchor_selection.json"))
    bank = Bank()
    ctx = Eng.Ctx(bank, "global", fmap=Features.load(os.path.join(os.environ["R8_PROC"], "feature_basis.npz"), "cpu"))
    nui = Eng.fit_outer(ctx, pairs, const)
    dump(dict(cov=nui, scales=ctx.scales), os.path.join(out, "global_nuisance.json"))
    st = ctx.st
    parts = a.parts.split(",")
    res = dict(frozen_code_commit=fz["code_commit"], eval_fix=fix, candidate=fz["candidate"], baselines=fz["baselines"])
    if "confirmation" in parts:
        rows = []
        for ep in ctx.episodes("val", [1, 2, 3], None):
            for K in (8, 16, 32):
                inp = Eng.inputs(ep, K)
                for m, p, ex in Eng.method_preds(ctx, inp, K, const, fx, jc=jc):
                    rows.append(dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, method=m, **F.score(p.numpy(), ep.y_q.numpy()), **ex))
        pd.DataFrame(rows).to_csv(os.path.join(out, "confirmation_episodes.tsv.gz"), sep="\t", index=False, float_format="%.7g")
        print("confirmation", f"{time.time() - t0:.0f}s", flush=True)
    if "matched" in parts:
        X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
        mrows, mplates, MRD = load_rows(os.environ["R8_PROC"], D.row_delta)
        rec = []
        for meta, e1, e2, pc in matched_episodes(st, mrows, mplates, MRD, X0, cell_role="val", max_pairs=6, seeds=(1, 2, 3)):
            b = dict(fold="global", **meta)
            for name in ("same_drug_other_plate", "other_drug_other_plate"):
                rec.append(dict(b, block="replicate", K=0, method=f"PC_{name}", ctrl_shift_rms=pc["ctrl_shift_rms"], **F.score(pc[name], pc["truth"])))
            for block, ep in (("in_block", e1), ("other_block", e2)):
                for K in (8, 16, 32):
                    if K > meta["kmax"]:
                        continue
                    for m, p, ex in Eng.method_preds(ctx, Eng.inputs(ep, K), K, const, fx, jc=jc):
                        rec.append(dict(b, block=block, K=K, method=m, ctrl_shift_rms=pc["ctrl_shift_rms"], **F.score(p.numpy(), pc["truth"]), **ex))
        pd.DataFrame(rec).to_csv(os.path.join(out, "matched_episodes.tsv.gz"), sep="\t", index=False, float_format="%.7g")
        print("matched", f"{time.time() - t0:.0f}s", flush=True)
    if "hcc" in parts:
        ofz = json.load(open(orig_man("freeze_manifest.json")))
        units, preds_files, missing = [], {}, []
        for t in TARGETS:
            rows, x0, x1 = load_target(t)
            assert set(rows.cell) == {t}
            tab, ud, ux = rows_to_units(rows, x1 - x0, x0)
            for split, lists in ofz["targets"][t].items():
                ep, qn, is_test = hcc_episode(st, t, split, lists, rows, tab, ud, ux)
                missing.append(dict(target=t, split_seed=int(split), support_without_source_family=int((ep.n0_s == 0).sum()), query_without_source_family=int((ep.n0_q == 0).sum())))
                truth = ep.y_q.numpy()
                store = {}
                for K in (8, 16, 32):
                    if K > len(ep.g_s):
                        continue
                    inp = Eng.inputs(ep, K)
                    for m, p, _ in Eng.method_preds(ctx, inp, K, const, fx, jc=jc):
                        pn = p.numpy()
                        store[f"{m}|{K}"] = pn.astype(np.float32)
                        units += unit_table(pn, truth, qn, is_test, dict(method=m, K=K, split_seed=int(split), target=t))
                f = os.path.join(out, f"pred_{t}_split{split}.npz")
                np.savez_compressed(f, query_drug_groups=np.array(qn), is_test=is_test, **store)
                preds_files[os.path.basename(f)] = dict(path=f, sha256=sha256(f), bytes=os.path.getsize(f))
                print(t, split, f"{time.time() - t0:.0f}s", flush=True)
        u = pd.concat(units, ignore_index=True)
        p = os.path.join(out, "hcc_per_unit.tsv.gz")
        u.to_csv(p, sep="\t", index=False, float_format="%.6g")
        res.update(hcc_per_unit=p, hcc_per_unit_sha256=sha256(p), prediction_files=preds_files, groups_without_source_family=missing)
    dump(run_record(a.run_id, "final_eval", t0, **res), os.path.join(out, "result.json"))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
