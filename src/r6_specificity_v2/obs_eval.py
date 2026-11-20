"""R6-O evaluation: O0 (N64-trained) vs O1 (N-sampled) M models on the source held folds (plan section 3.4).

Spec: configs/r6_specificity_v2/r6o_spec.json. All models and baselines receive the same observed cells
and the same stored thinning draws:
    N in {64, 32, 16} (training range of O1) and 8 (outside both arms' training range, auxiliary)
    = the first N of the 64 fixed eval_input cells of the pool (nested subsets),
    p in {1, 0.5, 0.25} = the stored thinning draws of those cells (p<1: thinning seeds 0, 1, 2).
Metrics per condition are averaged over training seeds 42/43/44 and thinning seeds before the
drug-structure-group bootstrap (Scorer of the v1 follow-up, 2,000 replicates, seed 0).

Also written: the observed UMI (panel + rest counts) and usable-cell distribution of every (N, p) input,
and the auxiliary UMI-matched comparison (per pool: 16 cells at p = 1 vs 64 cells thinned to the same
expected total UMI, keep rate p* clipped to the trained range [0.25, 1]).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.depth_robust_r6 import data as D
from src.depth_robust_r6.count_view import set_features
from src.r6_followup_v1 import common as C
from src.r6_followup_v1 import metrics as M
from src.r6_followup_v1.oof_tables import mean_baselines, oof_conditions
from src.r6_followup_v1.scoring import DEF_TOTAL, DEF_WITHIN, Scorer
from src.r6_followup_v1.train_fixed import build_model

NS = (64, 32, 16, 8)
PS = (1.0, 0.5, 0.25)
REP = Path("reports/r6_specificity_v2")


@torch.no_grad()
def predict_sets(model, d, L, Dg, pid, cnt_sets, rest_sets, y_scale, dev, chunk=256):
    """cnt_sets [n_pool, N, G], rest_sets [n_pool, N] -> prediction [n_cond, G] for conditions (L, Dg) in pools pid."""
    model.eval()
    fp = torch.as_tensor(d.fp, device=dev)
    zs = []
    for i in range(0, len(cnt_sets), 32):
        cc = torch.as_tensor(cnt_sets[i:i + 32], device=dev).float(); rr = torch.as_tensor(rest_sets[i:i + 32], device=dev).float()
        zs.append(model.state(set_features(cc, rr))[0])
    z = torch.cat(zs)
    out = np.zeros((len(L), d.Y.shape[2]), np.float32)
    for i in range(0, len(L), chunk):
        g, h = model.delta_parts(z[torch.as_tensor(pid[i:i + chunk], device=dev)], fp[torch.as_tensor(Dg[i:i + chunk], device=dev)].unsqueeze(1))
        out[i:i + chunk] = ((h[:, 0] if g is None else g[:, 0] + h[:, 0]) * y_scale).float().cpu().numpy()
    return out


def eval_sets(d):
    """{(p, seed): (cnt [n_pool, 64, G], rest [n_pool, 64])} of the fixed eval_input cells."""
    rows = np.concatenate([np.arange(r.eval_start, r.eval_end) for r in d.pools.itertuples()])
    n_pool, G = len(d.pools), d.cnt.shape[1]
    out = {(1.0, 0): (d.cnt[rows].reshape(n_pool, D.EVAL_N, G), d.crest[rows].reshape(n_pool, D.EVAL_N))}
    for k, (c, r) in d.eval_views.items():
        out[k] = (c.reshape(n_pool, D.EVAL_N, G), r.reshape(n_pool, D.EVAL_N))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/r6_specificity_v2/r6o_spec.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    dev = "cuda"
    d = C.load()
    L, Dg, fold = oof_conditions(d)
    pool = C.pool_of(d, L, Dg)
    G = d.Y.shape[2]
    sc = Scorer(d.Y[L, Dg], L, Dg, pool, fold, n_boot=a.n_boot, seed=0)
    kappa = json.loads(Path("manifests/r6_followup_v1/kappa_selection.json").read_text())["chosen_kappa"]
    sc.add("poolmean", [mean_baselines(d, L, Dg, fold, pool, kappa)["poolmean"]], retrieval=False)
    ev = eval_sets(d)

    # ---- observed UMI / usable cells of every input ---------------------------------------------------
    umi_rows = []
    for N in NS:
        for p in PS:
            keys = [k for k in ev if k[0] == p]
            tot, per_cell, usable = [], [], []
            for k in keys:
                c, r = ev[k]
                lib = c[:, :N].astype(np.int64).sum(-1) + r[:, :N].astype(np.int64)
                tot.append(lib.sum(1)); per_cell.append(lib.ravel()); usable.append((lib > 0).sum(1))
            tot, per_cell, usable = np.concatenate(tot), np.concatenate(per_cell), np.concatenate(usable)
            umi_rows.append(dict(n_cells=N, p=p, n_times_p=N * p, n_pool_views=len(tot), set_umi_mean=tot.mean(), set_umi_median=np.median(tot),
                                 set_umi_q10=np.quantile(tot, 0.1), set_umi_q90=np.quantile(tot, 0.9), cell_umi_mean=per_cell.mean(),
                                 cell_umi_median=np.median(per_cell), usable_cells_mean=usable.mean(), usable_cells_min=int(usable.min()),
                                 frac_sets_all_cells_usable=float((usable == N).mean())))
    umi = pd.DataFrame(umi_rows)

    # ---- UMI-matched auxiliary inputs (per pool) ----------------------------------------------------------
    c1, r1 = ev[(1.0, 0)]
    libA = c1[:, :16].astype(np.int64).sum((1, 2)) + r1[:, :16].astype(np.int64).sum(1)
    lib64 = c1.astype(np.int64).sum((1, 2)) + r1.astype(np.int64).sum(1)
    p_star = np.clip(libA / np.maximum(lib64, 1), 0.25, 1.0)
    aux = {}
    for s in range(3):
        rng = np.random.Generator(np.random.PCG64([20261119, s]))
        cb = rng.binomial(c1.astype(np.int64), p_star[:, None, None]).astype(np.int16)
        rb = rng.binomial(r1.astype(np.int64), p_star[:, None]).astype(np.int32)
        aux[s] = (cb, rb)
    libB = np.mean([aux[s][0].astype(np.int64).sum((1, 2)) + aux[s][1].astype(np.int64).sum(1) for s in range(3)], 0)
    umi_aux = dict(p_star_median=float(np.median(p_star)), p_star_q10=float(np.quantile(p_star, 0.1)), p_star_q90=float(np.quantile(p_star, 0.9)),
                   frac_p_star_clipped=float(((libA / np.maximum(lib64, 1)) < 0.25).mean() + ((libA / np.maximum(lib64, 1)) > 1).mean()),
                   set_umi_A_mean=float(libA.mean()), set_umi_B_mean=float(libB.mean()), rel_umi_B_minus_A_median=float(np.median(libB / libA - 1)))

    # ---- models -------------------------------------------------------------------------------------------
    recs = []
    for arm, item in spec["arms"].items():
        models = {}
        for k in C.FOLDS:
            for s in C.SEEDS:
                run = C.assets() / "runs" / item["stage"] / item["pattern"].format(fold=k, seed=s)
                ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
                assert ck["fold"] == k and ck["obs_arm"] == arm and ck["seed"] == s
                assert not (set(ck["fit_drugs"]) & set(C.source_split(d, k)[1]))
                m = build_model(d, "M", ck["hp"]).to(dev); m.load_state_dict(ck["state_dict"])
                models[(k, s)] = (m, ck["y_scale"])
                r = json.loads((run / "record.json").read_text())
                recs.append(dict(arm=arm, fold=k, seed=s, run_id=r["run_id"], updates=r["updates"], train_seconds=r["train_seconds"],
                                 cells_processed=r["cells_processed"], n_update_histogram=json.dumps(r["n_update_histogram"]),
                                 peak_vram_MB=r["peak_vram_MB"], gpu=r["gpu"], ckpt_sha256=r["ckpt_sha256"], commit=r["commit"]))
        for N in NS:
            preds = {p: [] for p in PS}
            for s in C.SEEDS:
                for (p, ts), (c, r) in ev.items():
                    A = np.zeros((len(L), G), np.float32)
                    for k in C.FOLDS:
                        q = fold == k
                        m, ys = models[(k, s)]
                        A[q] = predict_sets(m, d, L[q], Dg[q], pool[q], c[:, :N], r[:, :N], ys, dev)
                    preds[p].append(A)
            for p in PS:
                sc.add(f"{arm}|{N}|{p}", preds[p], retrieval=False)
            print("scored", arm, N, flush=True)
        # auxiliary UMI-matched inputs
        pa, pb = [], []
        for s in C.SEEDS:
            A = np.zeros((len(L), G), np.float32); Bp = [np.zeros((len(L), G), np.float32) for _ in range(3)]
            for k in C.FOLDS:
                q = fold == k
                m, ys = models[(k, s)]
                A[q] = predict_sets(m, d, L[q], Dg[q], pool[q], c1[:, :16], r1[:, :16], ys, dev)
                for t in range(3):
                    Bp[t][q] = predict_sets(m, d, L[q], Dg[q], pool[q], aux[t][0], aux[t][1], ys, dev)
            pa.append(A); pb.extend(Bp)
        sc.add(f"{arm}|umiA_N16_p1", pa, retrieval=False); sc.add(f"{arm}|umiB_N64_pstar", pb, retrieval=False)
        del models
        torch.cuda.empty_cache()

    # ---- tables ---------------------------------------------------------------------------------------------
    def bp(ca, cb):
        x, y = sc.arms[ca], sc.arms[cb]
        bs = x["bs"]["all"]["total"] - y["bs"]["all"]["total"]
        bs = bs[~np.isnan(bs)]
        return float(max(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), 1.0 / len(bs)))

    rows = []
    std = C.std_fields(split_role="source_oof", fold="0,1,2", model_seed="42,43,44")
    std["version"] = "r6_specificity_v2"
    for arm in spec["arms"]:
        for N in NS:
            for p in PS:
                s = sc.summary(f"{arm}|{N}|{p}")
                pr = sc.pair(f"{arm}|{N}|{p}", "poolmean")
                rows.append(dict(std, row_type="arm", arm=arm, n_cells=N, p=p, n_times_p=N * p, n_drugs=s["n_drugs"], n_units=s["n_units"],
                                 total_rmse=s["total_rmse"], total_lo=s["total_lo"], total_hi=s["total_hi"], within_rmse=s["within_rmse"],
                                 within_lo=s["within_lo"], within_hi=s["within_hi"], centred_magnitude_ratio=s["centred_magnitude_ratio"],
                                 total_minus_poolmean=pr["total_diff"], total_minus_poolmean_lo=pr["total_lo"], total_minus_poolmean_hi=pr["total_hi"]))
        for nm in ("umiA_N16_p1", "umiB_N64_pstar"):
            s = sc.summary(f"{arm}|{nm}")
            rows.append(dict(std, row_type="arm_umi_matched_aux", arm=arm, setting=nm, total_rmse=s["total_rmse"], total_lo=s["total_lo"], total_hi=s["total_hi"],
                             within_rmse=s["within_rmse"], n_drugs=s["n_drugs"], n_units=s["n_units"]))
    s = sc.summary("poolmean")
    rows.append(dict(std, row_type="baseline", arm="poolmean (shrunk, kappa from v1)", total_rmse=s["total_rmse"], total_lo=s["total_lo"], total_hi=s["total_hi"],
                     within_rmse=s["within_rmse"]))

    def pair_row(kind, ca, cb, family=None, **kw):
        pr = sc.pair(ca, cb)
        rows.append(dict(std, row_type=kind, family=family or "", candidate=ca, comparator=cb, sign="candidate - comparator (negative = candidate better)",
                         n_drugs=pr["n_drugs_total"], n_units=pr["n_units_total"], total_diff=pr["total_diff"], total_diff_lo=pr["total_lo"],
                         total_diff_hi=pr["total_hi"], total_rel=pr["total_rel"], total_category=M.decision_category(pr["total_diff"], pr["total_lo"], pr["total_hi"], pr["total_comparator"], 0.005),
                         p_boot_total=bp(ca, cb), within_diff=pr["within_diff"], within_diff_lo=pr["within_lo"], within_diff_hi=pr["within_hi"],
                         within_rel=pr["within_rel"], within_category=M.decision_category(pr["within_diff"], pr["within_lo"], pr["within_hi"], pr["within_comparator"], 0.005), **kw))

    for N in NS:
        for p in PS:
            fam = {16: "primary_N16", 64: "secondary_N64", 32: "secondary_N32", 8: "auxiliary_N8_outside_training"}[N]
            pair_row("O1_minus_O0", f"O1|{N}|{p}", f"O0|{N}|{p}", family=fam, n_cells=N, p=p)
    for arm in spec["arms"]:
        for (N1, p1), (N2, p2) in (((64, 0.25), (16, 1.0)), ((64, 0.5), (32, 1.0))):
            pair_row("same_model_iso_n_times_p", f"{arm}|{N1}|{p1}", f"{arm}|{N2}|{p2}", arm=arm, candidate_setting=f"N={N1},p={p1}", comparator_setting=f"N={N2},p={p2}")
        for N in (32, 16):
            for p in PS:
                pair_row("same_model_cells_effect", f"{arm}|{N}|{p}", f"{arm}|64|{p}", arm=arm, candidate_setting=f"N={N},p={p}", comparator_setting=f"N=64,p={p}")
        pair_row("same_model_umi_matched_aux", f"{arm}|umiB_N64_pstar", f"{arm}|umiA_N16_p1", arm=arm,
                 candidate_setting="N=64, p* (same expected UMI as A)", comparator_setting="N=16, p=1")
    pair_row("O1_minus_O0_umi_matched_aux", "O1|umiA_N16_p1", "O0|umiA_N16_p1")
    pair_row("O1_minus_O0_umi_matched_aux", "O1|umiB_N64_pstar", "O0|umiB_N64_pstar")
    T = pd.DataFrame(rows)
    # Holm within each O1-O0 family (total RMSE)
    T["holm_reject_total"] = np.nan
    for fam, g in T[T.row_type == "O1_minus_O0"].groupby("family"):
        order = g.p_boot_total.sort_values().index
        ok = True
        for i, ix in enumerate(order):
            ok = ok and T.at[ix, "p_boot_total"] <= 0.05 / (len(order) - i)
            T.at[ix, "holm_reject_total"] = ok
    # share of the N16 loss of O0 that O1 removes (point estimates)
    share = []
    for p in PS:
        g0 = sc.summary(f"O0|16|{p}")["total_rmse"] - sc.summary(f"O0|64|{p}")["total_rmse"]
        g1 = sc.summary(f"O1|16|{p}")["total_rmse"] - sc.summary(f"O1|64|{p}")["total_rmse"]
        share.append(dict(p=p, O0_gap_N16_minus_N64=g0, O1_gap_N16_minus_N64=g1, share_removed=(g0 - g1) / g0 if g0 else np.nan))
    T["metric_definition"] = f"total: {DEF_TOTAL} | within: {DEF_WITHIN}"
    T.to_csv(REP / "observation_matched.tsv", sep="\t", index=False, float_format="%.6f")
    umi.to_csv(REP / "observation_umi_inputs.tsv", sep="\t", index=False, float_format="%.3f")
    pd.DataFrame(recs).to_csv(REP / "observation_matched_runs.tsv", sep="\t", index=False)
    C.dump_json(dict(umi_matched_aux=umi_aux, n16_gap_share=share, spec=a.spec, commit=C.git_commit(), kappa=kappa,
                     n_conditions=int(len(L)), n_drugs=int(len(np.unique(Dg)))), "manifests/r6_specificity_v2/observation_matched_receipt.json")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    cols = ["row_type", "family", "arm", "candidate", "comparator", "n_cells", "p", "total_rmse", "total_diff", "total_diff_lo", "total_diff_hi", "total_rel",
            "total_category", "p_boot_total", "holm_reject_total", "within_rmse", "within_diff", "within_diff_lo", "within_diff_hi"]
    print(T[[c for c in cols if c in T]].to_string())
    print(umi.to_string()); print(json.dumps(umi_aux)); print(json.dumps(share))


if __name__ == "__main__":
    main()
