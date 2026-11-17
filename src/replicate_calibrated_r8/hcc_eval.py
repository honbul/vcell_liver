"""Post-freeze evaluation on the three HCC targets (already-seen public benchmark). One pass; no selection happens here.

--mock runs the identical code path on three confirmation-family source cells posing as targets (wiring check before the
freeze; reads no HCC value). Without --mock the guarded loader requires the committed freeze manifest.
"""
import argparse, json, os, subprocess, time
import numpy as np
import pandas as pd
import torch

from .common import TARGETS, sha256, dump, new_run_dir, run_record
from .data import Store, Episode, overlap
from .build_store import row_weights
from .suite import Suite, current_selection
from .contrasts import ADAPTERS
from src.episodic_residual_r2.build_units import rows_to_units
from src.episodic_residual_r2.metrics import per_unit
from src.episodic_residual_r2 import tsg as G
from src.transigen_adaptation_b1.data import Data

MOCK = ("TC32", "MINO", "22RV1")
KEEP = ["delta_rmse", "delta_pearson", "centred_rmse", "centred_pearson", "top20_direction", "retrieval_rank", "retrieval_rank_centred", "truth_rms", "pred_rms"]


def mock_bundle(B, st, cell, split_seeds):
    """A confirmation cell dressed as a target: same row layout, splits drawn with study B's 60/20/20 rule."""
    ra = G.source_row_access(B, st, cell)
    rows = ra.rows.copy()
    gidx = rows.drug_group.map(st.groups.set_index("drug_group").group_idx).to_numpy()
    ok = (st.n_train[torch.tensor(gidx, device=st.dev)] >= 1).cpu().numpy()
    g = sorted(rows.drug_group[ok].unique())
    splits = {}
    for s in split_seeds:
        rs = np.random.RandomState(100000 + s)
        p = [g[i] for i in rs.permutation(len(g))]
        nt = int(round(0.2 * len(g)))
        test, val, pool = p[:nt], p[nt:2 * nt], p[2 * nt:]
        pool = [pool[i] for i in rs.permutation(len(pool))]
        splits[str(s)] = dict(support_order=pool[:32], query_test=test, query_val_secondary=val)
    keep = np.where(ok)[0]
    return rows.iloc[keep].reset_index(drop=True), ra.x0[keep], ra._x1[keep], ra.fp[keep], splits


def build_episode(st, target, split, lists, rows, tab, ud, ux, regime):
    pos = {g: i for i, g in enumerate(tab.drug_group)}
    gi = tab.drug_group.map(st.groups.set_index("drug_group").group_idx).to_numpy()
    sup = [pos[g] for g in lists["support_order"]]
    qry_names = list(lists["query_test"]) + list(lists["query_val_secondary"])
    qry = [pos[g] for g in qry_names]
    assert not (set(sup) & set(qry)) and len(set(qry)) == len(qry)
    f0s, n0s = st.f0(regime, gi[sup], allow_missing=True)
    f0q, n0q = st.f0(regime, gi[qry], allow_missing=True)
    w = row_weights(rows)                                    # metadata: weight of each plate row in its unit mean
    by = {}
    for dg, pl, wi in zip(rows.drug_group, rows.det_plate, w):
        by.setdefault(dg, ([], []))
        by[dg][0].append(pl)
        by[dg][1].append(wi)
    o, q = overlap([by[g][0] for g in lists["support_order"]], [by[g][1] for g in lists["support_order"]])
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=st.dev)
    ep = Episode(name=target, family=target, seed=int(split), g_s=gi[sup], g_q=gi[qry], x0_s=t(ux[sup]), x0_q=t(ux[qry]), f0_s=f0s, f0_q=f0q, n0_s=n0s, n0_q=n0q,
                 y_s=t(ud[sup]), y_q=t(ud[qry]), o_s=t(o), q_s=t(q), plates_s=list(tab.plates.to_numpy()[sup]), plates_q=list(tab.plates.to_numpy()[qry]))
    is_test = np.array([True] * len(lists["query_test"]) + [False] * len(lists["query_val_secondary"]))
    return ep, qry_names, is_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda")
    torch.set_num_threads(1)
    st = Store(dev)
    B = Data(os.environ["TB_PROC"])
    if a.mock:
        assert os.environ.get("R8_FINAL_EVAL") != "1"
        sel = current_selection()
        fz = dict(k_values=[8, 16, 32], split_seeds=[1, 2, 3], code_commit="mock")
        targets, eval_fix, regime = MOCK, "", "train"
        out = new_run_dir("hcc_mock", a.run_id)
    else:
        from .target import assert_frozen, load_target
        fz = assert_frozen()
        changed = subprocess.run(["git", "diff", "--stat", fz["code_commit"], "HEAD", "--", "src/replicate_calibrated_r8", "configs/replicate_calibrated_r8"],
                                 capture_output=True, text=True).stdout.strip()
        eval_fix = os.environ.get("R8_EVAL_FIX", "")
        assert changed == "" or eval_fix, f"code changed after the freeze without a recorded reason:\n{changed}"
        assert sha256(fz["config"]["path"]) == fz["config"]["sha256"] and sha256(fz["feature_basis"]["path"]) == fz["feature_basis"]["sha256"]
        sel, targets, regime = fz["selection"], TARGETS, "hcc"
        out = new_run_dir("hcc", a.run_id)
    suite = Suite(sel, st, dev, shuffle_modes=ADAPTERS)
    names = st.groups.drug_group.to_numpy()
    rows_out, files, fails, missing = [], {}, [], []
    for target in targets:
        if a.mock:
            rows, x0, x1, fp, splits = mock_bundle(B, st, target, fz["split_seeds"])
        else:
            rows, x0, x1, _ = load_target(target)
            fp = B.fp[rows.drug_idx.to_numpy()].astype(np.float32)
            splits = fz["targets"][target]
            assert set(rows.cell) == {target}                       # a target never sees another target's rows
        tab, ud, ux = rows_to_units(rows, x1 - x0, x0)
        ra = G.RowAccess(rows, x0, x1, fp, names)
        for split, lists in splits.items():
            ep, qry_names, is_test = build_episode(st, target, split, lists, rows, tab, ud, ux, regime)
            missing.append(dict(target=target, split_seed=int(split), support_without_source_family=int((ep.n0_s == 0).sum().item()),
                                query_without_source_family=int((ep.n0_q == 0).sum().item()), n_query=len(qry_names)))
            preds = {}
            truth = ep.y_q.cpu().numpy()
            for K in fz["k_values"]:
                if K > len(ep.g_s):
                    continue
                for method, seed, pred in suite.run(ep, K, ra):
                    if isinstance(pred, str):
                        fails.append(dict(target=target, split_seed=int(split), K=K, method=method, model_seed=seed, status=pred))
                        continue
                    p = pred.detach().cpu().numpy()
                    preds[f"{method}|{seed}|{K}"] = p.astype(np.float32)
                    for qset, m in (("test", is_test), ("test_plus_val", np.ones(len(is_test), bool))):
                        pu = per_unit(p[m], truth[m])[KEEP]
                        pu.insert(0, "drug_group", np.array(qry_names)[m])
                        pu.insert(0, "is_test", is_test[m])
                        for c, v in (("query_set", qset), ("model_seed", seed), ("method", method), ("K", K), ("split_seed", int(split)), ("target", target)):
                            pu.insert(0, c, v)
                        rows_out.append(pu)
            f = os.path.join(out, f"pred_{target}_split{split}.npz")
            np.savez_compressed(f, query_drug_groups=np.array(qry_names), is_test=is_test, support_order=np.array(lists["support_order"]), **preds)
            files[os.path.basename(f)] = dict(path=f, sha256=sha256(f), bytes=os.path.getsize(f), n_query=len(qry_names), n_arrays=len(preds))
            print(target, split, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.concat(rows_out, ignore_index=True)
    df.to_csv(os.path.join(out, "per_unit_all_seeds.tsv"), sep="\t", index=False, float_format="%.6g")
    res = run_record(a.run_id, "hcc_mock" if a.mock else "hcc_eval", t0, frozen_code_commit=fz["code_commit"], eval_fix=eval_fix, targets=list(targets),
                     failures=fails, groups_without_source_family=missing, prediction_files=files, per_unit_table=os.path.join(out, "per_unit_all_seeds.tsv"),
                     per_unit_sha256=sha256(os.path.join(out, "per_unit_all_seeds.tsv")))
    dump(res, os.path.join(out, "result.json"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("prediction_files",)}, indent=1))
    q = df[(df.query_set == "test")].groupby(["target", "K", "method"]).delta_rmse.mean().unstack("K")
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(q.to_string())


if __name__ == "__main__":
    main()
