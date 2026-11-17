#!/usr/bin/env python3
"""Frozen final evaluation (R1-5). The only script that reads outer-test labels and LCCL labels.

Internal: for every training set of the nested runs, each named model picks its configuration from that
set's inner-fold losses (rule in configs: `models`), the held-out predictions of the picked
configuration become the out-of-fold table, and Track A's primary metric with the family bootstrap is
computed. Permuted-prior versions go through the identical path with the permuted gene sets.
External: LIMORE-trained predictions for LCCL models, Spearman against -AUC / -GI50 per matched drug
(Track A's definition and bootstrap), cross-study leave-family-out and new lineages kept apart.

Usage: evaluate.py --internal-runs A,B,.. --perm-runs C,D,.. --external-runs E,F,.. --cmp-run G
                   --tracka-run H --tag final --out-dir reports/softprior_multitask_r1/tables
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_eval as E  # noqa: E402
import r1_models as M  # noqa: E402
import r1_select as SEL  # noqa: E402
import metrics as A1M  # noqa: E402


# ------------------------------------------------------------------------------------------------
# loading nested runs
# ------------------------------------------------------------------------------------------------

def load_units(run_dirs: list, prior: str, prefix: str) -> dict:
    """{set id: keys, loss [config, drug], pred [config, row, drug], models} merged over run directories."""
    out = {}
    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "inner", f"{prefix}*__{prior}.npz"))):
            sid = os.path.basename(f).split("__")[0]
            z = np.load(f, allow_pickle=False)
            p = np.load(os.path.join(rd, "pred", os.path.basename(f)), allow_pickle=False)
            assert list(z["configs"]) == list(p["configs"])
            keys = [SEL.parse_key(s) for s in z["configs"]]
            rec = json.loads(str(z["records"]))
            item = dict(keys=keys, loss=z["loss"], pred=p["pred"], models=[str(m) for m in p["models"]], records=rec,
                        meta=[json.loads(str(z["meta"]))])
            if sid not in out:
                out[sid] = item
                continue
            cur = out[sid]
            assert not set(cur["keys"]) & set(keys) and cur["models"] == item["models"]
            cur["keys"] += keys
            cur["loss"] = np.vstack([cur["loss"], item["loss"]])
            cur["pred"] = np.concatenate([cur["pred"], item["pred"]], axis=0)
            cur["records"] += rec
            cur["meta"] += item["meta"]
    return out


def pick(unit: dict, spec: dict) -> tuple:
    idx = SEL.choose(spec, unit["keys"], unit["loss"])
    D = unit["loss"].shape[1]
    return unit["pred"][idx, :, np.arange(D)].T, [unit["keys"][i] for i in idx]      # [row, drug]


def internal_oof(data: R.Data, units: dict, spec: dict) -> tuple:
    row = {m: i for i, m in enumerate(data.models)}
    reps = sorted(data.splits.repeat.unique())
    pred = np.full((len(reps),) + data.Y.shape, np.nan)
    chosen = collections.Counter()
    assert len(units) == 15
    for sid, u in units.items():
        rep = int(sid[1:].split("f")[0])
        P, keys = pick(u, spec)
        pred[rep, [row[m] for m in u["models"]]] = P
        chosen.update(M.key_str(k) for k in keys)
    assert np.isfinite(pred).all()
    return pred, chosen


def cmp_internal(data: R.Data, run_dir: str, name: str):
    files = sorted(glob.glob(os.path.join(run_dir, "cmp", name, "r*.npz")))
    if len(files) != 15:
        return None, f"{len(files)} of 15 training sets present"
    row = {m: i for i, m in enumerate(data.models)}
    pred = np.full((3,) + data.Y.shape, np.nan)
    for f in files:
        z = np.load(f, allow_pickle=False)
        rep = int(os.path.basename(f)[1:].split("f")[0])
        pred[rep, [row[str(m)] for m in z["models"]]] = z["pred"]
    if not np.isfinite(pred).all():
        return None, "non-finite predictions"
    return pred, ""


# ------------------------------------------------------------------------------------------------
# external (LCCL)
# ------------------------------------------------------------------------------------------------

class ExternalScorer:
    def __init__(self, cfg: dict, data: R.Data, ext: R.External):
        self.ext, self.data = ext, data
        self.drugs = sorted(ext.drug_match.limore_name)
        self.dcol = np.array([data.drugs.index(d) for d in self.drugs])
        l2c = dict(zip(ext.drug_match.limore_name, ext.drug_match.lccl_name))
        self.lccl_names = [l2c[d] for d in self.drugs]
        obs = ext.obs[ext.obs.model_id.isin(set(ext.models))]
        self.sens = {}
        for ep in ("AUC", "GI50"):
            w = obs.pivot(index="model_id", columns="lccl_name", values=ep)
            self.sens[ep] = -w.reindex(index=ext.models, columns=self.lccl_names).values.astype(float)
        self.min_n = cfg["metric"]["min_families_for_spearman"]
        self.n_boot, self.seed = cfg["metric"]["bootstrap"]["n"], cfg["metric"]["external_bootstrap_seed"]
        self.types = ["cross_study_leave_family_out", "new_lineage"]

    def fam_mean(self, A: np.ndarray, rows: np.ndarray) -> tuple:
        fams = np.array(sorted(set(self.ext.fam[rows])))
        out = np.full((len(fams), A.shape[1]), np.nan)
        for i, f in enumerate(fams):
            sub = A[rows & (self.ext.fam == f)]
            ok = ~np.isnan(sub)
            out[i] = np.where(ok.any(axis=0), np.nansum(sub, axis=0) / np.maximum(ok.sum(axis=0), 1), np.nan)
        return fams, out

    def _mean_rho(self, P: np.ndarray, O: np.ndarray, pickrows=None) -> tuple:
        rho = np.full(P.shape[1], np.nan)
        for d in range(P.shape[1]):
            p, o = (P[:, d], O[:, d]) if pickrows is None else (P[pickrows, d], O[pickrows, d])
            ok = ~np.isnan(p) & ~np.isnan(o)
            if ok.sum() >= self.min_n:
                rho[d] = A1M.safe_spearman(p[ok], o[ok])
        return (float(np.nanmean(rho)) if np.isfinite(rho).any() else float("nan")), rho

    def score(self, pred: np.ndarray) -> dict:
        """pred: [LCCL model, matched drug] predicted AA. Returns per (eval type, endpoint): mean rho,
        per-drug rho, bootstrap replicates (Track A resampling: families with replacement, same seed)."""
        out = {}
        for et in self.types:
            rows = self.ext.eval_type == et
            fams, Pf = self.fam_mean(pred, rows)
            for ep in ("AUC", "GI50"):
                _, Of = self.fam_mean(self.sens[ep], rows)
                mean, rho = self._mean_rho(Pf, Of)
                rs = np.random.RandomState(self.seed)
                boot = np.empty(self.n_boot)
                for b in range(self.n_boot):
                    pk = rs.choice(len(fams), size=len(fams), replace=True)
                    boot[b] = self._mean_rho(Pf, Of, pk)[0]
                out[(et, ep)] = dict(mean=mean, rho=rho, boot=boot, n_families=len(fams), n_models=int(rows.sum()))
        return out

    def check_against_track_a(self, pred: np.ndarray, result: dict) -> float:
        """Recompute one bootstrap interval with Track A's own functions; returns the largest deviation."""
        import run_external as A1X
        et, ep = "cross_study_leave_family_out", "AUC"
        rows = self.ext.eval_type == et
        df = pd.DataFrame(pred[rows], index=np.array(self.ext.models)[rows], columns=self.drugs).rename_axis("model_id")
        df = df.reset_index().melt(id_vars="model_id", var_name="drug_id", value_name="pred")
        s = pd.DataFrame(self.sens[ep][rows], index=np.array(self.ext.models)[rows], columns=self.drugs).rename_axis("model_id")
        df = df.merge(s.reset_index().melt(id_vars="model_id", var_name="drug_id", value_name="sens_auc"))
        df["family_id"] = df.model_id.map(dict(zip(self.ext.models, self.ext.fam)))
        fam = df.groupby(["drug_id", "family_id"])[["pred", "sens_auc"]].mean().reset_index()
        tab = A1X.spearman_table(fam, "sens_auc", self.min_n)
        lo, hi = A1X.boot_mean_spearman(fam, "sens_auc", self.min_n, self.n_boot, self.seed)
        mine = A1M.ci(result[(et, ep)]["boot"])
        return max(abs(tab.spearman.mean() - result[(et, ep)]["mean"]), abs(lo - mine[0]), abs(hi - mine[1]))


def external_pred(ext: R.External, units: dict, spec: dict | None, dcol: np.ndarray) -> np.ndarray:
    row = {m: i for i, m in enumerate(ext.models)}
    out = np.full((len(ext.models), len(dcol)), np.nan)
    for sid, u in units.items():
        P = pick(u, spec)[0] if spec is not None else u["pred"]
        out[[row[m] for m in u["models"]]] = P[:, dcol]
    assert np.isfinite(out).all()
    return out


def cmp_external(ext: R.External, run_dir: str, name: str, dcol: np.ndarray):
    files = sorted(glob.glob(os.path.join(run_dir, "cmp", name, "x_*.npz")))
    n_sets = 1 + len(set(ext.fam[ext.eval_type != "new_lineage"]))
    if len(files) != n_sets:
        return None, f"{len(files)} of {n_sets} training sets present"
    units = {os.path.basename(f)[:-4]: dict(pred=(z := np.load(f, allow_pickle=False))["pred"],
                                            models=[str(m) for m in z["models"]]) for f in files}
    return external_pred(ext, units, None, dcol), ""


# ------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal-runs", required=True)
    ap.add_argument("--perm-runs", default="")
    ap.add_argument("--external-runs", default="")
    ap.add_argument("--cmp-run", default="")
    ap.add_argument("--tracka-run", required=True)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pred-dir", default=R.PREDICTIONS)
    args = ap.parse_args()
    cfg = R.load_config()
    fin = cfg["final"]
    rd = lambda s: [os.path.join(R.runs_dir(cfg), r) for r in s.split(",") if r]
    data = R.load_data(cfg)
    sc = E.Scorer(data)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)
    T = lambda name: os.path.join(args.out_dir, f"{args.tag}_{name}")
    missing = []

    # ---------------- internal ----------------
    units = load_units(rd(args.internal_runs), "real", "r")
    rec = [r for u in units.values() for r in u["records"]]
    solver = dict(n_solver_records=len(rec), n_not_converged=int(sum(not r["converged"] for r in rec)),
                  max_iterations=int(max(r["iterations"] for r in rec)),
                  cpu_seconds=float(sum(m["seconds"] for u in units.values() for m in u["meta"])))
    preds, chosen_rows = {}, []
    for name in fin["models"]:
        preds[name], cnt = internal_oof(data, units, SEL.model_spec(cfg, name))
        tot = sum(cnt.values())
        chosen_rows += [dict(model=name, config=k, n_chosen=n, fraction=n / tot) for k, n in cnt.most_common()]
    tracka = os.path.join(R.runs_dir(cfg), args.tracka_run)
    for name in fin["comparators"]:
        if name in ("A0", "A1", "A1r", "A3"):
            preds[name] = E.load_oof(tracka, name)["pred"]
            continue
        p, why = cmp_internal(data, rd(args.cmp_run)[0], name) if args.cmp_run else (None, "no comparator run given")
        if p is None:
            missing.append(dict(model=name, part="internal", reason=why))
        else:
            preds[name] = p
    assert np.abs(preds["A0"] - sc.mu_train).max() < 1e-9          # training mean = family-weighted mean of the fold
    cubes = {k: sc.cube(v) for k, v in preds.items()}
    order = ["A0", "A1", "A1r", "A3"] + [m for m in fin["comparators"] if m in preds and m[0] != "A"] + fin["models"]
    summ = pd.DataFrame([sc.summary_row(k, preds[k]) for k in order])
    summ.to_csv(T("internal_summary.tsv"), sep="\t", index=False, float_format="%.6g")
    diffs = pd.DataFrame([sc.difference_row(a, b, cubes[a], cubes[b]) for a, b in fin["pairs"] if a in cubes and b in cubes])
    diffs.to_csv(T("internal_differences.tsv"), sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(chosen_rows).to_csv(T("internal_chosen.tsv"), sep="\t", index=False, float_format="%.4g")
    per_drug = pd.concat([sc.per_drug(k, preds[k]) for k in order], ignore_index=True)
    per_drug.to_csv(T("internal_per_drug.tsv"), sep="\t", index=False, float_format="%.5g")
    pref = pd.DataFrame([sc.preference_rows(k, preds[k]) for k in order])
    pref.to_csv(T("internal_broad_vs_preference.tsv"), sep="\t", index=False, float_format="%.5g")
    print(summ.to_string(), diffs.to_string(), pref.to_string(), sep="\n")

    # light out-of-fold table: ids and predictions only
    idx = pd.MultiIndex.from_product([sorted(data.splits.repeat.unique()), data.models, data.drugs],
                                     names=["repeat", "model_id", "drug_id"])
    oof = pd.DataFrame({k: preds[k].reshape(-1) for k in order}, index=idx).reset_index()
    oof.to_csv(os.path.join(args.pred_dir, "internal_oof_predictions.tsv.gz"), sep="\t", index=False,
               float_format="%.5g", compression="gzip")

    # ---------------- permuted prior ----------------
    perm_summary = []
    if args.perm_runs:
        boots = {m: [] for m in fin["permuted_prior_models"]}
        rows = []
        for seed in cfg["hallmark"]["permutation_seeds"]:
            pu = load_units(rd(args.perm_runs), f"perm{seed:03d}", "r")
            solver["n_not_converged"] += int(sum(not r["converged"] for u in pu.values() for r in u["records"]))
            solver["n_solver_records"] += int(sum(len(u["records"]) for u in pu.values()))
            for m in fin["permuted_prior_models"]:
                p, _ = internal_oof(data, pu, SEL.model_spec(cfg, m))
                F = sc.cube(p)
                rows.append(dict(model=m, prior=f"perm{seed:03d}", macro_drug_normalized_rmse=sc.macro(F)))
                boots[m].append(A1M.boot_macro(F, sc.W))
        pt = pd.DataFrame(rows)
        pt.to_csv(T("internal_permuted_prior_scores.tsv"), sep="\t", index=False, float_format="%.6g")
        for m in fin["permuted_prior_models"]:
            v = pt[pt.model == m].macro_drug_normalized_rmse.values
            real = sc.macro(cubes[m])
            d = A1M.boot_macro(cubes[m], sc.W) - np.mean(boots[m], axis=0)
            lo, hi = A1M.ci(d, *sc.ci)
            perm_summary.append(dict(model=m, real=real, perm_mean=v.mean(), perm_sd=v.std(ddof=1), perm_min=v.min(),
                                     perm_max=v.max(), n_perm=len(v), n_perm_at_or_below_real=int((v <= real).sum()),
                                     real_minus_perm_mean=real - v.mean(), ci_low=lo, ci_high=hi))
        pd.DataFrame(perm_summary).to_csv(T("internal_real_vs_permuted_prior.tsv"), sep="\t", index=False, float_format="%.6g")
        print(pd.DataFrame(perm_summary).to_string())

    # ---------------- external ----------------
    ext_rows, ext_diff = [], []
    if args.external_runs:
        ext = R.load_external(cfg, data)
        xs = ExternalScorer(cfg, data, ext)
        xu = load_units(rd(args.external_runs), "real", "x_")
        assert len(xu) == 1 + len(set(ext.fam[ext.eval_type != "new_lineage"]))
        xrec = [r for u in xu.values() for r in u["records"]]
        solver["external_n_solver_records"] = len(xrec)
        solver["external_n_not_converged"] = int(sum(not r["converged"] for r in xrec))
        xp = {}
        for name in fin["external_models"]:
            if name in cfg["models"]:
                xp[name] = external_pred(ext, xu, SEL.model_spec(cfg, name), xs.dcol)
            else:
                p, why = cmp_external(ext, rd(args.cmp_run)[0], name, xs.dcol) if args.cmp_run else (None, "no comparator run given")
                if p is None:
                    missing.append(dict(model=name, part="external", reason=why))
                else:
                    xp[name] = p
        res = {k: xs.score(v) for k, v in xp.items()}
        dev = xs.check_against_track_a(xp["A1r"], res["A1r"])
        assert dev < 1e-10, dev
        pdr = []
        for k, r in res.items():
            for (et, ep), v in r.items():
                lo, hi = A1M.ci(v["boot"])
                ext_rows.append(dict(model=k, eval_type=et, endpoint=ep, n_families=v["n_families"], n_models=v["n_models"],
                                     n_drugs_scored=int(np.isfinite(v["rho"]).sum()), mean_spearman=v["mean"],
                                     median_spearman=float(np.nanmedian(v["rho"])) if np.isfinite(v["rho"]).any() else np.nan,
                                     ci_low=lo, ci_high=hi, n_drugs_positive=int((v["rho"] > 0).sum())))
                pdr += [dict(model=k, eval_type=et, endpoint=ep, drug_id=d, spearman=x) for d, x in zip(xs.drugs, v["rho"])]
        for a, b in fin["external_pairs"]:
            if a in res and b in res:
                for key in res[a]:
                    d = res[a][key]["boot"] - res[b][key]["boot"]
                    lo, hi = A1M.ci(d)
                    ext_diff.append(dict(model=a, reference=b, eval_type=key[0], endpoint=key[1],
                                         difference=res[a][key]["mean"] - res[b][key]["mean"], ci_low=lo, ci_high=hi))
        pd.DataFrame(ext_rows).to_csv(T("external_summary.tsv"), sep="\t", index=False, float_format="%.4g")
        pd.DataFrame(ext_diff).to_csv(T("external_differences.tsv"), sep="\t", index=False, float_format="%.4g")
        pd.DataFrame(pdr).to_csv(T("external_per_drug.tsv"), sep="\t", index=False, float_format="%.4g")
        ix = pd.MultiIndex.from_product([ext.models, xs.drugs], names=["model_id", "drug_id"])
        xt = pd.DataFrame({k: v.reshape(-1) for k, v in xp.items()}, index=ix).reset_index()
        xt.insert(1, "family_id", xt.model_id.map(dict(zip(ext.models, ext.fam))))
        xt.insert(2, "eval_type", xt.model_id.map(dict(zip(ext.models, ext.eval_type))))
        xt.insert(4, "lccl_name", xt.drug_id.map(dict(zip(xs.drugs, xs.lccl_names))))
        xt.to_csv(os.path.join(args.pred_dir, "external_lccl_predictions.tsv.gz"), sep="\t", index=False,
                  float_format="%.5g", compression="gzip")
        print(pd.DataFrame(ext_rows).to_string(), pd.DataFrame(ext_diff).to_string(), sep="\n")
        solver["external_bootstrap_check_vs_track_a_max_dev"] = dev

    pd.DataFrame(missing, columns=["model", "part", "reason"]).to_csv(T("not_run.tsv"), sep="\t", index=False)
    with open(T("results.json"), "w") as fh:
        json.dump(dict(runs=vars(args), solver=solver, internal_summary=summ.to_dict(orient="records"),
                       internal_differences=diffs.to_dict(orient="records"), permuted_prior=perm_summary,
                       external_summary=ext_rows, external_differences=ext_diff, not_run=missing,
                       prior_exposure="post-freeze evaluation on a reused public benchmark (LIMORE and LCCL results were seen in Track A)"),
                  fh, indent=1, default=float)
    print(json.dumps(solver, indent=1), missing)


if __name__ == "__main__":
    main()
