"""R7-C (nested OOF affine calibration) and R7-L (raw objective x selection 2x2) on the outer folds of the R4 training
drugs (private plan 2026-11-19 sections 4.2-4.4). CPU. Spec: configs/r7_xpert_followup_v2/spec.json (frozen).

Inputs (server runs, never committed): outer trajectories outer_{L0,L1}_k{k}_s{seed} -> predictions of the U/R/P
checkpoints on T_k; inner trajectories inner_L0_k{k}_j{j}_s{seed} -> predictions on inner fold j of A_k.
For every seed and outer fold k:
  calibration fit data = inner OOF predictions on A_k (+ A_k truths); lambda chosen per arm by drug-group calibration CV
  over the inner folds of A_k (source units, family-equal unit RMSE; tie -> strongest shrinkage); coefficients refit on
  all of A_k; applied to the outer model's T_k predictions. T_k truths are read only by the scorer.
Scoring: per-unit RMSE / Pearson / centred Pearson / tie-corrected centred-cosine MRR with candidate pool = cell line x
outer fold; metrics per seed, unit values averaged over seeds, family-equal source statistic, paired drug-group bootstrap
(2000 draws, seed 20261117, all cell lines of a group move together). HCC lines reported per line and as their mean.
Mode --pilot: wiring only (pilot runs, outer fold 0, seed 42); T truths are replaced by synthetic noise so no outer
fold is scored before the freeze."""
import argparse, json, os, time
import numpy as np
import pandas as pd

from .common import runs, pman, public, dump, load_json, sha_array, stamp, CFG, HCC, SEEDS
from .folds import load_folds
from . import fit_calibrator as fc
from .metrics import retrieval_by_pool
from .stats import FamBoot, interval, boot_p, holm, category, THR
from src.evidence_gated_r7.data import Store

ARMS = fc.ARMS
METRICS = ["rmse", "pearson", "centred_pearson", "mrr_tie", "mrr_legacy", "rms_pred", "rms_true"]


class Ctx:
    def __init__(self, S, folds, pilot=False):
        self.S = S
        u = S.units
        self.idx = S.train
        self.pos = {int(i): p for p, i in enumerate(self.idx)}
        self.cells = u.cell.to_numpy()[self.idx]
        self.fam = u.family.to_numpy()[self.idx]
        self.hcc = u.is_hcc.to_numpy()[self.idx] == 1
        self.group = u.drug_group.to_numpy()[self.idx]
        fo = folds.set_index("group_idx")
        g = S.group_idx[self.idx]
        self.outer = fo.outer.loc[g].to_numpy()
        self.inner = np.stack([fo[f"inner_k{k}"].loc[g].to_numpy() for k in range(3)])
        self.pool = np.array([f"{c}|{o}" for c, o in zip(self.cells, self.outer)])
        self.Y = S.delta(self.idx).astype(np.float64)
        self.pilot = pilot

    def positions(self, unit_idx):
        return np.array([self.pos[int(i)] for i in unit_idx])


def traj_dir(name, pilot):
    return runs("traj", name + ("_pilot" if pilot else ""))


def load_outer(C, recipe, sel, seed, folds=(0, 1, 2)):
    P = np.full(C.Y.shape, np.nan)
    for k in folds:
        d = traj_dir(f"outer_{recipe}_k{k}_s{seed}", C.pilot)
        pos = C.positions(np.load(os.path.join(d, "unit_idx_T.npy")))
        assert (C.outer[pos] == k).all() and len(pos) == (C.outer == k).sum()
        P[pos] = np.load(os.path.join(d, f"pred_{sel}_T.npy"))
    return P


def load_inner(C, sel, seed, k, inner=(0, 1, 2)):
    P = np.full(C.Y.shape, np.nan)
    for j in inner:
        d = traj_dir(f"inner_L0_k{k}_j{j}_s{seed}", C.pilot)
        pos = C.positions(np.load(os.path.join(d, "unit_idx_Ain.npy")))
        assert (C.outer[pos] != k).all() and (C.inner[k][pos] == j).all()
        P[pos] = np.load(os.path.join(d, f"pred_{sel}_Ain.npy"))
    return P


def unit_metrics(Z, Y, pool):
    Z = np.asarray(Z, np.float64); Y = np.asarray(Y, np.float64)
    rmse = np.sqrt(((Z - Y) ** 2).mean(1))
    zc = Z - Z.mean(1, keepdims=True); yc = Y - Y.mean(1, keepdims=True)
    pear = (zc * yc).sum(1) / np.sqrt((zc ** 2).sum(1) * (yc ** 2).sum(1) + 1e-12)
    Zp = Z.copy(); Yp = Y.copy()
    for p in np.unique(pool):
        s = pool == p
        Zp[s] -= Z[s].mean(0); Yp[s] -= Y[s].mean(0)
    a = Zp - Zp.mean(1, keepdims=True); b = Yp - Yp.mean(1, keepdims=True)
    cp = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)
    r = retrieval_by_pool(Z, Y, pool)
    return pd.DataFrame(dict(rmse=rmse, pearson=pear, centred_pearson=cp, mrr_tie=r["mrr_tie_expected"], mrr_legacy=r["legacy_midrank_mrr"],
                             n_tied=r["n_tied"], n_candidates=r["n_candidates"], rms_pred=np.sqrt((Z ** 2).mean(1)), rms_true=np.sqrt((Y ** 2).mean(1))))


def panel_summary(m, C, mask=None):
    mask = np.ones(len(m), bool) if mask is None else mask
    src = mask & ~C.hcc
    out = {f"src_{k}": fc.fam_mean(m[k].to_numpy()[src], C.fam[src]) for k in METRICS}
    out["src_rms_ratio"] = out["src_rms_pred"] / out["src_rms_true"]
    for h in HCC:
        s = mask & (C.cells == h)
        for k in ["rmse", "centred_pearson", "mrr_tie", "rms_pred"]:
            out[f"{h}_{k}"] = float(m[k].to_numpy()[s].mean()) if s.any() else np.nan
        out[f"{h}_n"] = int(s.sum())
    for k in ["rmse", "centred_pearson", "mrr_tie"]:
        out[f"hcc_mean_{k}"] = float(np.nanmean([out[f"{h}_{k}"] for h in HCC]))
    return out


def calibrate_fold(C, Ytrain, Pin, Pout, k, arms=ARMS):
    """Fit every arm on the inner OOF of A_k (truth from Ytrain rows of A_k only) and apply to T_k predictions."""
    A = np.where(C.outer != k)[0]; T = np.where(C.outer == k)[0]
    if C.pilot:
        A = A[~np.isnan(Pin[A]).any(1)]
    assert not np.isnan(Pin[A]).any() and not np.isnan(Pout[T]).any()
    res = {}
    for arm in arms:
        lam, cv = fc.select_lambda(arm, Pin[A], Ytrain[A], C.cells[A], C.fam[A], C.hcc[A], C.inner[k][A])
        coef = fc.fit(arm, lam, Pin[A], Ytrain[A], C.cells[A], C.fam[A], C.hcc[A])
        Z, fb = fc.apply(coef, Pout[T], C.cells[T])
        Zg, _ = fc.apply(coef, Pout[T], C.cells[T], hcc_global_only=C.hcc[T])
        res[arm] = dict(lam=lam, cv=cv, coef=coef, Z=Z, Zhccglob=Zg, fallback=fb, T=T, A=A)
    return res


def shift_record(C, Pin, Pout, k):
    A = np.where((C.outer != k) & ~np.isnan(Pin).any(1))[0]; T = np.where(C.outer == k)[0]
    f = lambda P, I: dict(rms=fc.fam_mean(np.sqrt((P[I] ** 2).mean(1))[~C.hcc[I]], C.fam[I][~C.hcc[I]]),
                          mean=fc.fam_mean(P[I].mean(1)[~C.hcc[I]], C.fam[I][~C.hcc[I]]))
    return dict(inner_oof_on_A=f(Pin, A), outer_on_T=f(Pout, T))


def contrasts(fb, C, table, defs, primary=None, core_m=1):
    """table: dict name -> seed-mean unit metric frame. defs: list of (name, expr dict {model: coef})."""
    rows = []
    src = ~C.hcc
    for name, ex in defs:
        for metric in ["rmse", "centred_pearson", "mrr_tie", "rms_pred"]:
            d = sum(w * table[m][metric].to_numpy() for m, w in ex.items())
            point, bs = fb.stat(d, src)
            lo, hi = interval(bs)
            row = dict(contrast=name, metric=metric, panel="source_family_equal", diff=point, lo95=lo, hi95=hi, boot_p=boot_p(bs),
                       n_units=int(src.sum()), n_groups=int(len(np.unique(C.group[src]))), n_families=int(len(np.unique(C.fam[src]))))
            if metric == "rmse":
                row["category_ci95"] = category(point, lo, hi)
                if core_m > 1 and primary and name in primary:
                    slo, shi = interval(bs, 1 - 0.05 / core_m)
                    row.update(lo_simultaneous=slo, hi_simultaneous=shi, category_simultaneous=category(point, slo, shi))
            rows.append(row)
            if metric == "rmse":
                hb = []
                for h in HCC:
                    p, b = fb.unit_mean(d, C.cells == h)
                    lo, hi = interval(b)
                    rows.append(dict(contrast=name, metric=metric, panel=h, diff=p, lo95=lo, hi95=hi, n_units=int((C.cells == h).sum())))
                    hb.append((p, b))
                p = float(np.mean([x[0] for x in hb])); b = np.mean([x[1] for x in hb], 0)
                lo, hi = interval(b)
                rows.append(dict(contrast=name, metric=metric, panel="hcc_mean", diff=p, lo95=lo, hi95=hi))
    df = pd.DataFrame(rows)
    if primary:
        m = (df.metric == "rmse") & (df.panel == "source_family_equal") & df.contrast.isin(primary)
        df.loc[m, "holm_p"] = holm(df.loc[m, "boot_p"].to_numpy())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    spec = load_json(os.path.join(CFG, "spec.json")) if not a.pilot else None
    S = Store(); folds = load_folds()
    C = Ctx(S, folds, pilot=a.pilot)
    seeds = [42] if a.pilot else list(SEEDS)
    kk = [0] if a.pilot else [0, 1, 2]
    out = pman("pilot" if a.pilot else "oof")
    os.makedirs(out, exist_ok=True)
    srv = runs("eval", "pilot_oof" if a.pilot else "oof_v1")
    os.makedirs(srv, exist_ok=a.pilot)
    Yscore = C.Y.copy()
    if a.pilot:                         # never score an outer fold before the freeze: synthetic truths on T_0
        T0 = C.outer == 0
        Yscore[T0] = np.random.RandomState(0).normal(0, 0.5, (T0.sum(), C.Y.shape[1]))
    inner_sel = ["U", "R", "P"]
    tables = {}        # (model) -> list of per-seed unit frames (T units only, S.train order)
    per_seed, tuning, coefs, inv, shifts, fixtures = [], [], [], [], [], {}
    for seed in seeds:
        for base in ["L0_U", "L0_R", "L0_P", "L1_U", "L1_R", "L1_P"]:
            rec, sel = base.split("_")
            Pout = load_outer(C, rec, sel, seed, kk)
            Tm = ~np.isnan(Pout).any(1)
            m = unit_metrics(Pout[Tm], Yscore[Tm], C.pool[Tm])
            tables.setdefault(f"{base}|raw", []).append(m)
            per_seed.append(dict(seed=seed, model=f"{base}|raw", **panel_summary(m, _sub(C, Tm))))
            if rec != "L0" or sel not in inner_sel:
                continue
            Z = {arm: np.full(C.Y.shape, np.nan) for arm in ARMS}
            Zg = {arm: np.full(C.Y.shape, np.nan) for arm in ARMS}
            FB = {arm: np.zeros(len(C.Y), bool) for arm in ARMS}
            for k in kk:
                Pin = load_inner(C, sel, seed, k, [0] if a.pilot else (0, 1, 2))
                if a.pilot:
                    Pin_valid = ~np.isnan(Pin).any(1)
                    C.inner[k][Pin_valid] = np.arange(Pin_valid.sum()) % 3     # pilot only: pseudo CV folds inside inner fold 0
                res = calibrate_fold(C, C.Y, Pin, Pout, k)
                shifts.append(dict(seed=seed, base=base, outer=k, **{f"{w}_{q}": v for w, d in shift_record(C, Pin, Pout, k).items() for q, v in d.items()}))
                for arm, r in res.items():
                    Z[arm][r["T"]] = r["Z"]; Zg[arm][r["T"]] = r["Zhccglob"]; FB[arm][r["T"]] = r["fallback"]
                    tuning.append(dict(seed=seed, base=base, outer=k, arm=arm, chosen_lambda=("identity" if not np.isfinite(r["lam"]) else r["lam"]),
                                       n_fit_units=int(len(r["A"])), n_fallback_units_T=int(r["fallback"].sum()),
                                       fallback_cells=sorted(set(C.cells[r["T"]][r["fallback"]])) if arm not in ("C0", "Cg") else [],
                                       **{f"cv_rmse_{lk}": v for lk, v in r["cv"].items() if lk is not None}))
                    coefs.append(dict(seed=seed, base=base, outer=k, coef_sha=fc.coef_hash(r["coef"]), **fc.coef_summary(r["coef"])))
                    if r["coef"]["cells"]:
                        for c, d in r["coef"]["cells"].items():
                            coefs.append(dict(seed=seed, base=base, outer=k, arm=arm, cell=c, a=d["a"], u_rms=d["u_rms"], n_units=d["n_units"], at_bound=d["at_bound"]))
                if seed == seeds[0] and base == "L0_U" and k == kk[0]:
                    fixtures.update(tamper_check(C, Pin, Pout, k, res))
            for arm in ARMS:
                Tm2 = ~np.isnan(Z[arm]).any(1)
                assert (Tm2 == Tm).all()
                mm = unit_metrics(Z[arm][Tm], Yscore[Tm], C.pool[Tm])
                tables.setdefault(f"{base}|{arm}", []).append(mm)
                ps = panel_summary(mm, _sub(C, Tm)); ps["n_fallback_units"] = int(FB[arm][Tm].sum())
                per_seed.append(dict(seed=seed, model=f"{base}|{arm}", **ps))
                mg = unit_metrics(Zg[arm][Tm], Yscore[Tm], C.pool[Tm])
                per_seed.append(dict(seed=seed, model=f"{base}|{arm}|hcc_global_only", **panel_summary(mg, _sub(C, Tm))))
                if arm != "C0":
                    c0 = tables[f"{base}|C0"][-1]
                    inv.append(dict(seed=seed, base=base, arm=arm, max_abs_diff_centred_pearson=float(np.abs(mm.centred_pearson - c0.centred_pearson).max()),
                                    max_abs_diff_mrr_tie=float(np.abs(mm.mrr_tie - c0.mrr_tie).max()),
                                    max_abs_diff_mrr_legacy=float(np.abs(mm.mrr_legacy - c0.mrr_legacy).max()),
                                    n_tied_identical=bool((mm.n_tied.to_numpy() == c0.n_tied.to_numpy()).all()),
                                    candidate_sets_identical=bool((mm.n_candidates.to_numpy() == c0.n_candidates.to_numpy()).all()),
                                    raw_pearson_max_abs_diff=float(np.abs(mm.pearson - c0.pearson).max())))
            ident = tables[f"{base}|C0"][-1]
            raw = tables[f"{base}|raw"][-1]
            fixtures.setdefault("identity_reproduces_raw", []).append(dict(seed=seed, base=base, max_abs_rmse_diff=float(np.abs(ident.rmse - raw.rmse).max())))
            fixtures.setdefault("positive_scales", []).append(dict(seed=seed, base=base, all_positive=bool(all(r.get("all_a_positive", True) for r in coefs if r.get("base") == base and "all_a_positive" in r))))
    Tm = ~np.isnan(load_outer(C, "L0", "U", seeds[0], kk)).any(1)
    Csub = _sub(C, Tm)
    mean_tab = {k: sum(v) / len(v) for k, v in tables.items()}
    pd.DataFrame(per_seed).to_csv(os.path.join(out, "per_seed_summary.tsv"), sep="\t", index=False)
    pd.DataFrame(tuning).to_csv(os.path.join(out, "calibration_tuning.tsv"), sep="\t", index=False)
    pd.DataFrame(coefs).to_csv(os.path.join(out, "calibration_coefficients.tsv"), sep="\t", index=False)
    pd.DataFrame(inv).to_csv(os.path.join(out, "invariance_audit.tsv"), sep="\t", index=False)
    pd.DataFrame(shifts).to_csv(os.path.join(out, "inner_outer_shift.tsv"), sep="\t", index=False)
    seedmean = pd.DataFrame([dict(model=k, **panel_summary(v, Csub)) for k, v in mean_tab.items()])
    seedmean.to_csv(os.path.join(out, "seedmean_summary.tsv"), sep="\t", index=False)
    fb = FamBoot(Csub.group, Csub.fam, n_boot=2000, seed=20261117)
    rc_primary = ["Csb-C0"]
    rc = [("Csb-C0", {"L0_U|Csb": 1, "L0_U|C0": -1}), ("Csb-Cs", {"L0_U|Csb": 1, "L0_U|Cs": -1}), ("Csb-Cb", {"L0_U|Csb": 1, "L0_U|Cb": -1}),
          ("Cg-C0", {"L0_U|Cg": 1, "L0_U|C0": -1}), ("Cs-C0", {"L0_U|Cs": 1, "L0_U|C0": -1}), ("Cb-C0", {"L0_U|Cb": 1, "L0_U|C0": -1}),
          ("L0R: Csb-C0", {"L0_R|Csb": 1, "L0_R|C0": -1}), ("L0P: Csb-C0", {"L0_P|Csb": 1, "L0_P|C0": -1}),
          ("calibrated R-P (L0, Csb)", {"L0_R|Csb": 1, "L0_P|Csb": -1})]
    rl_primary = ["L0: R-P", "L1: R-P", "R: L1-L0", "P: L1-L0", "interaction (L1R-L1P)-(L0R-L0P)"]
    rl = [("L0: R-P", {"L0_R|raw": 1, "L0_P|raw": -1}), ("L1: R-P", {"L1_R|raw": 1, "L1_P|raw": -1}),
          ("R: L1-L0", {"L1_R|raw": 1, "L0_R|raw": -1}), ("P: L1-L0", {"L1_P|raw": 1, "L0_P|raw": -1}),
          ("interaction (L1R-L1P)-(L0R-L0P)", {"L1_R|raw": 1, "L1_P|raw": -1, "L0_R|raw": -1, "L0_P|raw": 1}),
          ("U: L1-L0 (report)", {"L1_U|raw": 1, "L0_U|raw": -1}), ("L0: R-U (report)", {"L0_R|raw": 1, "L0_U|raw": -1}),
          ("L0: P-U (report)", {"L0_P|raw": 1, "L0_U|raw": -1})]
    crc = contrasts(fb, Csub, mean_tab, rc, primary=rc_primary)
    crl = contrasts(fb, Csub, mean_tab, rl, primary=rl_primary, core_m=len(rl_primary))
    crc.to_csv(os.path.join(out, "r7c_contrasts.tsv"), sep="\t", index=False)
    crl.to_csv(os.path.join(out, "r7l_contrasts.tsv"), sep="\t", index=False)
    # per-seed paired values of the primary contrasts
    ps = pd.DataFrame(per_seed)
    rows = []
    for name, ex in rc[:1] + rl[:5]:
        for seed in seeds:
            v = sum(w * ps[(ps.seed == seed) & (ps.model == m)].src_rmse.iloc[0] for m, w in ex.items())
            rows.append(dict(contrast=name, seed=seed, src_rmse_diff=v))
    pd.DataFrame(rows).to_csv(os.path.join(out, "primary_by_seed.tsv"), sep="\t", index=False)
    # epoch boundary check
    eb = []
    for seed in seeds:
        for rec in ["L0", "L1"]:
            for k in kk:
                r = load_json(os.path.join(traj_dir(f"outer_{rec}_k{k}_s{seed}", a.pilot), "record.json"))
                for s, b in r["selectors"].items():
                    eb.append(dict(seed=seed, recipe=rec, outer=k, selector=s, best_epoch=b["epoch"], epochs_run=r["epochs_run"], stopped_at=b.get("stopped_at_epoch")))
    pd.DataFrame(eb).to_csv(os.path.join(out, "best_epochs.tsv"), sep="\t", index=False)
    np.savez_compressed(os.path.join(srv, "seedmean_unit_metrics.npz"), **{k.replace("|", "__"): v[["rmse", "centred_pearson", "mrr_tie", "rms_pred"]].to_numpy() for k, v in mean_tab.items()},
                        train_idx=C.idx[Tm])
    rec = dict(mode="pilot (synthetic T truths; wiring only)" if a.pilot else "oof_v1", spec_sha=None if a.pilot else spec.get("spec_version"),
               seeds=seeds, outer_folds=kk, fixtures=fixtures, invariance_max=dict(
                   centred_pearson=float(pd.DataFrame(inv).max_abs_diff_centred_pearson.max()) if inv else None,
                   mrr_tie=float(pd.DataFrame(inv).max_abs_diff_mrr_tie.max()) if inv else None,
                   n_tied_identical_all=bool(pd.DataFrame(inv).n_tied_identical.all()) if inv else None),
               seconds=round(time.time() - t0, 1), **stamp())
    dump(rec, os.path.join(out, "oof_eval_record.json"))
    print(json.dumps(rec, indent=1, default=str))
    print(crc[(crc.metric == "rmse") & (crc.panel == "source_family_equal")].to_string())
    print(crl[(crl.metric == "rmse") & (crl.panel == "source_family_equal")].to_string())


def _sub(C, mask):
    class V:
        pass
    v = V()
    for k in ["cells", "fam", "hcc", "group", "outer", "pool"]:
        setattr(v, k, getattr(C, k)[mask])
    return v


def tamper_check(C, Pin, Pout, k, res):
    """Outer-fold labels must not reach the calibration fit: replace T_k truths with noise, refit, compare."""
    Y2 = C.Y.copy()
    T = C.outer == k
    Y2[T] = np.random.RandomState(1).normal(0, 1, (T.sum(), Y2.shape[1]))
    r2 = calibrate_fold(C, Y2, Pin, Pout, k, arms=("Csb", "Cg"))
    return dict(outer_label_tamper=[dict(arm=arm, coef_identical=fc.coef_hash(r2[arm]["coef"]) == fc.coef_hash(res[arm]["coef"]),
                                         lambda_identical=r2[arm]["lam"] == res[arm]["lam"],
                                         calibrated_T_identical=bool(np.array_equal(r2[arm]["Z"], res[arm]["Z"]))) for arm in ("Csb", "Cg")])


if __name__ == "__main__":
    main()
