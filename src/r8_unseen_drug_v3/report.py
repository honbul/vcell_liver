"""Re-aggregation of the saved v3 outer predictions (preds.npz + pred_meta.tsv.gz of every scored block) with the frozen statistics.

Two-way family x drug paired bootstrap (run_spec.yaml statistics.primary.ci): per draw one multinomial count per scored family and one
per outer query drug group; the episode centring mean uses the drug weights (recomputed every draw) and identical weights are applied
to every arm. Centred RMSE only depends on d = pred - truth: ||d_q - dbar_w||^2 = G_qq - 2 (G w)_q / W + w'G w / W^2 with G = D D'.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import yaml

from .core import ARMS, SPEC, dump, runs, sha256, log

N_GENES = 978


def load(dirs):
    eps, meta_all = [], []
    for d in dirs:
        r = json.load(open(os.path.join(d, "result.json")))
        assert r["scored"], d
        for f, e in r["files"].items():
            assert sha256(os.path.join(d, f)) == e["sha256"], f"{d}/{f} hash differs"
        m = pd.read_csv(os.path.join(d, "pred_meta.tsv.gz"), sep="\t")
        z = np.load(os.path.join(d, "preds.npz"))
        t = z["truth"].astype(np.float64)
        P = {a: z[a].astype(np.float64) for a in ARMS}
        for e, s in m.groupby("episode", sort=True):
            ix = s.index.to_numpy()
            rec = dict(partition=s.partition.iloc[0], drug_fold=int(s.drug_fold.iloc[0]), cell=s.cell.iloc[0], family=s.family.iloc[0],
                       seed=int(s.seed.iloc[0]), groups=s.drug_group.to_numpy(), gram={}, diag={})
            for a in ARMS:
                D = P[a][ix] - t[ix]
                g = D @ D.T
                rec["gram"][a], rec["diag"][a] = g, np.diag(g).copy()
            eps.append(rec)
        meta_all.append(m)
    return eps, pd.concat(meta_all, ignore_index=True)


def episode_scores(rec, Wq):
    """Wq: (B, Q) drug weights. Returns {arm: (centred (B,), delta (B,))} with NaN where the episode has zero weight."""
    W = Wq.sum(1)
    ok = W > 0
    Ws = np.where(ok, W, 1.0)
    out = {}
    for a in ARMS:
        G, dg = rec["gram"][a], rec["diag"][a]
        GW = Wq @ G
        dbar2 = (GW * Wq).sum(1) / Ws ** 2
        r2 = dg[None, :] - 2.0 * GW / Ws[:, None] + dbar2[:, None]
        rc = np.sqrt(np.clip(r2, 0.0, None) / N_GENES)
        rd = np.sqrt(np.clip(dg, 0.0, None) / N_GENES)
        c = (Wq * rc).sum(1) / Ws
        dl = (Wq * rd[None, :]).sum(1) / Ws
        c[~ok], dl[~ok] = np.nan, np.nan
        out[a] = (c, dl)
    return out


def aggregate(eps, fam_idx, CF, CD, drug_pos, sel=None):
    """Hierarchical means: episodes -> cell -> family -> overall weighted by CF. sel: optional family mask (outer fold)."""
    B = CF.shape[0]
    cells = {}
    for rec in eps:
        q = np.array([drug_pos[g] for g in rec["groups"]])
        sc = episode_scores(rec, CD[:, q])
        key = (rec["family"], rec["cell"])
        acc = cells.setdefault(key, {a: [np.zeros(B), np.zeros(B), np.zeros(B)] for a in ARMS})   # sum c, sum d, count
        for a in ARMS:
            c, dl = sc[a]
            ok = np.isfinite(c)
            acc[a][0] += np.where(ok, c, 0.0)
            acc[a][1] += np.where(ok, dl, 0.0)
            acc[a][2] += ok
    nf = len(fam_idx)
    fam = {a: [np.zeros((B, nf)), np.zeros((B, nf)), np.zeros((B, nf))] for a in ARMS}
    for (f, _), acc in cells.items():
        k = fam_idx[f]
        for a in ARMS:
            s_c, s_d, n = acc[a]
            ok = n > 0
            fam[a][0][:, k] += np.where(ok, s_c / np.maximum(n, 1), 0.0)
            fam[a][1][:, k] += np.where(ok, s_d / np.maximum(n, 1), 0.0)
            fam[a][2][:, k] += ok
    res = {}
    mask = np.ones(nf, bool) if sel is None else sel
    for a in ARMS:
        s_c, s_d, n = fam[a]
        ok = (n > 0) & mask[None, :]
        w = CF * ok
        fc = np.where(ok, s_c / np.maximum(n, 1), 0.0)
        fd = np.where(ok, s_d / np.maximum(n, 1), 0.0)
        res[a] = dict(centred_rmse=(w * fc).sum(1) / w.sum(1), delta_rmse=(w * fd).sum(1) / w.sum(1))
    return res


def ci(x):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), int(len(x))


def main(args):
    sp = yaml.safe_load(open(SPEC))
    out = runs("report", args.run_id)
    os.makedirs(out, exist_ok=False)
    eps, meta = load(args.blocks)
    fams = sorted({e["family"] for e in eps})
    fam_idx = {f: i for i, f in enumerate(fams)}
    drugs = sorted({g for e in eps for g in e["groups"]})
    drug_pos = {g: i for i, g in enumerate(drugs)}
    fam_fold = {e["family"]: e["partition"] for e in eps}
    log("episodes", len(eps), "families", len(fams), "drugs", len(drugs))
    B, seed = 2000, 20261120
    rs = np.random.RandomState(seed)
    CF2 = rs.multinomial(len(fams), np.full(len(fams), 1.0 / len(fams)), size=B).astype(np.float64)
    CD2 = rs.multinomial(len(drugs), np.full(len(drugs), 1.0 / len(drugs)), size=B).astype(np.float64)
    one_f, one_d = np.ones((1, len(fams))), np.ones((1, len(drugs)))
    point = aggregate(eps, fam_idx, one_f, one_d, drug_pos)
    two = aggregate(eps, fam_idx, CF2, CD2, drug_pos)
    famonly = aggregate(eps, fam_idx, CF2, np.ones((B, len(drugs))), drug_pos)
    folds = sorted(set(fam_fold.values()))
    fold_point = {P: aggregate(eps, fam_idx, one_f, one_d, drug_pos, sel=np.array([fam_fold[f] == P for f in fams])) for P in folds}
    # ---- arm means
    arm_rows = [dict(method=m, centred_rmse=float(point[m]["centred_rmse"][0]), delta_rmse=float(point[m]["delta_rmse"][0]), n_families=len(fams))
                for m in ARMS]
    pd.DataFrame(arm_rows).to_csv(os.path.join(out, "arm_means.tsv"), sep="\t", index=False, float_format="%.6f")

    def contrast(cand, base, metric, role):
        d = float(point[cand][metric][0] - point[base][metric][0])
        lo, hi, n2 = ci(two[cand][metric] - two[base][metric])
        flo, fhi, nf_ = ci(famonly[cand][metric] - famonly[base][metric])
        fd = {P: float(fold_point[P][cand][metric][0] - fold_point[P][base][metric][0]) for P in folds}
        return dict(role=role, contrast=f"{cand} - {base}", metric=metric, base_mean=float(point[base][metric][0]), cand_mean=float(point[cand][metric][0]),
                    diff=d, relative=d / float(point[base][metric][0]), ci_lo_two_way=lo, ci_hi_two_way=hi, n_draws_two_way=n2,
                    ci_lo_family_only=flo, ci_hi_family_only=fhi, n_folds_negative=int(sum(v < 0 for v in fd.values())), n_folds=len(folds),
                    **{f"diff_{P}": v for P, v in fd.items()}, n_families=len(fams), n_drugs=len(drugs))
    prim = [contrast("ridge_C1chem", "fb_C0", "centred_rmse", "primary"), contrast("ridge_C1chem", "fb_C0", "delta_rmse", "primary_overall_delta_rmse")]
    sec = []
    for cand, base in (("ridge_C0", "fb_C0"), ("ridge_C1chem", "ridge_C0"), ("fb_C1chem", "fb_C0")):
        for metric in ("centred_rmse", "delta_rmse"):
            sec.append(contrast(cand, base, metric, "secondary"))
    pd.DataFrame(prim).to_csv(os.path.join(out, "primary_contrast.tsv"), sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(sec).to_csv(os.path.join(out, "secondary_contrasts.tsv"), sep="\t", index=False, float_format="%.6g")
    # ---- decision (literal application of run_spec.yaml decision)
    p = prim[0]
    nfold_eval = len(folds)
    if nfold_eval == 0:
        state = "평가 불가"
    elif p["ci_lo_two_way"] > 0:
        state = "열세 (fb_C0 유지)"
    elif p["ci_hi_two_way"] >= 0:
        state = "미관측 약물 경로의 추가 이득 미확립 (채택 보류)"
    elif nfold_eval < 3:
        state = "채택 보류 (평가 fold 부족)"
    elif p["relative"] <= -0.005 and p["n_folds_negative"] >= 2:
        state = "채택 조건 충족" if prim[1]["diff"] <= 0 else "연구 가설 일부 지지, 운영 후보 교체 보류 (채택 보류)"
    else:
        state = "작은 개선 / 방향 불일치, 채택 보류"
    dump(dict(state=state, primary=p, overall_delta_rmse=prim[1], n_folds_evaluable=nfold_eval, bootstrap=dict(draws=B, seed=seed)),
         os.path.join(out, "decision.json"))
    # ---- per family x drug metrics (point centring inside each episode)
    rows = []
    for rec in eps:
        for a in ARMS:
            G, dg = rec["gram"][a], rec["diag"][a]
            Q = len(dg)
            r2 = dg - 2.0 * G.sum(1) / Q + G.sum() / Q ** 2
            for g, rc, rd in zip(rec["groups"], np.sqrt(np.clip(r2, 0, None) / N_GENES), np.sqrt(dg / N_GENES)):
                rows.append((rec["partition"], rec["family"], rec["cell"], rec["seed"], g, a, rc, rd))
    q = pd.DataFrame(rows, columns=["partition", "family", "cell", "seed", "drug_group", "method", "centred_rmse", "delta_rmse"])
    fd = q.groupby(["partition", "family", "drug_group", "method"]).agg(centred_rmse=("centred_rmse", "mean"), delta_rmse=("delta_rmse", "mean"),
                                                                       n_queries=("centred_rmse", "size")).reset_index()
    w = fd.pivot_table(index=["partition", "family", "drug_group", "n_queries"], columns="method", values=["centred_rmse", "delta_rmse"])
    w.columns = [f"{m}__{v}" for v, m in w.columns]
    w.reset_index().to_csv(os.path.join(out, "per_family_drug_metrics.tsv.gz"), sep="\t", index=False, float_format="%.5f")
    # ---- consistency: per-episode centred RMSE from the Gram path vs the scores written at run time
    ep_tab = pd.concat([pd.read_csv(os.path.join(d, "episodes.tsv.gz"), sep="\t") for d in args.blocks])
    ep_tab["seed"] = ep_tab.support_seed.str.split("|").str[0].astype(int)
    ep_tab["drug_fold"] = ep_tab.support_seed.str.split("|").str[1]
    # episodes are unique per (partition, drug fold, cell, seed); rebuild that key on the Gram side
    qe = []
    for rec in eps:
        for a in ARMS:
            G, dg = rec["gram"][a], rec["diag"][a]
            Q = len(dg)
            r2 = dg - 2.0 * G.sum(1) / Q + G.sum() / Q ** 2
            qe.append(dict(partition=rec["partition"], drug_fold=f"d{rec['drug_fold']}", cell=rec["cell"], seed=rec["seed"], method=a,
                           c=float(np.sqrt(np.clip(r2, 0, None) / N_GENES).mean()), d=float(np.sqrt(dg / N_GENES).mean())))
    mm = ep_tab.merge(pd.DataFrame(qe), on=["partition", "drug_fold", "cell", "seed", "method"], how="outer", indicator=True)
    cons = dict(n_rows=len(mm), n_unmatched=int((mm._merge != "both").sum()), max_abs_centred=float((mm.centred_rmse - mm.c).abs().max()),
                max_abs_delta=float((mm.delta_rmse - mm.d).abs().max()))
    # family-mean point estimates from the run-time episode table (independent path) for the 4 arms
    from src.episodic_residual_r2.evalcore import family_mean
    fm = {}
    for metric in ("centred_rmse", "delta_rmse"):
        agg, _ = family_mean(ep_tab.rename(columns={metric: "v"}), value="v", by=["method"])
        fm[metric] = {r.method: float(r.v) for r in agg.itertuples()}
    cons["max_abs_family_mean_vs_gram"] = max(abs(fm[m_][a_] - float(point[a_][m_][0])) for m_ in fm for a_ in ARMS)
    dump(cons, os.path.join(out, "consistency.json"))
    assets = [dict(block=os.path.basename(d), path=os.path.join(d, f), sha256=e["sha256"], bytes=e["bytes"])
              for d in args.blocks for f, e in json.load(open(os.path.join(d, "result.json")))["files"].items()]
    pd.DataFrame(assets).to_csv(os.path.join(out, "prediction_assets.tsv"), sep="\t", index=False)
    log("state", state, json.dumps(cons))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("blocks", nargs="+")
    main(ap.parse_args())
