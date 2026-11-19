"""Frozen evaluations on the existing R4 benchmark after the spec freeze (private plan 2026-11-19 sections 4.1, 4.2 step 6).

Exposure: the R4 test split was opened by R4 and again by R7; these are correction / frozen follow-up evaluations on an
exposed benchmark, not blind tests. Runs only with R7V2_FINAL_EVAL=1 and a committed freeze record
(configs/r7_xpert_followup_v2/freeze.json) whose commit is an ancestor of HEAD.

(1) Correction impact (section 4.1), official XPert and the A-2 XPert-ECFP comparator, 3 seeds each:
    old original = v1 checkpoint (HCC-included selection, upstream script, original run),
    old replay   = new trajectory, checkpoint chosen by the original rule (all validation units, HCC included),
    corrected    = same new trajectory, checkpoint chosen by the source-only rule (validation is_hcc == 0).
    Test source panel (family-equal ΔRMSE) and HCC lines; paired drug-group bootstrap (2000, seed 20261117).
    Validation source panel = the selection population (descriptive only).
(2) Section 4.2 step 6: calibration coefficients refit on the full-train cross-fit OOF (outer L0-U predictions,
    calibration CV over the outer folds), applied to the corrected full-train XPert (L0-U) test predictions."""
import json, os, subprocess, time
import numpy as np
import pandas as pd

from .common import runs, orig_xpert, pman, dump, load_json, stamp, sha256, CFG, HCC, SEEDS, ORIG_XPERT_RUNS
from .folds import load_folds
from . import fit_calibrator as fc
from .evaluate_oof import Ctx, load_outer, unit_metrics
from .stats import FamBoot, interval, boot_p, category
from src.evidence_gated_r7.data import Store
from src.evidence_gated_r7.evaluate import unit_metrics as legacy_unit_metrics


def gate():
    assert os.environ.get("R7V2_FINAL_EVAL") == "1", "final evaluation only inside the frozen final job"
    fr = load_json(os.path.join(CFG, "freeze.json"))
    ok = subprocess.run(["git", "merge-base", "--is-ancestor", fr["freeze_commit"], "HEAD"]).returncode == 0
    assert ok, "freeze commit is not an ancestor of HEAD"
    assert subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() == ""
    return fr


def summary(m, cells, fam, hcc):
    out = {}
    src = ~hcc
    for k in ["rmse", "pearson", "centred_pearson", "mrr_tie", "mrr_legacy", "rms_pred", "rms_true"]:
        out[f"src_{k}"] = fc.fam_mean(m[k].to_numpy()[src], fam[src])
    out["src_rms_ratio"] = out["src_rms_pred"] / out["src_rms_true"]
    for h in HCC:
        s = cells == h
        if s.any():
            for k in ["rmse", "centred_pearson", "mrr_tie"]:
                out[f"{h}_{k}"] = float(m[k].to_numpy()[s].mean())
            out[f"{h}_n"] = int(s.sum())
    if all(f"{h}_rmse" in out for h in HCC):
        out["hcc_mean_rmse"] = float(np.mean([out[f"{h}_rmse"] for h in HCC]))
    return out


def contrast(fb, d, cells, hcc, name, extra):
    rows = []
    p, bs = fb.stat(d, ~hcc); lo, hi = interval(bs)
    rows.append(dict(contrast=name, panel="source_family_equal", diff=p, lo95=lo, hi95=hi, boot_p=boot_p(bs), category_ci95=category(p, lo, hi), **extra))
    hb = []
    for h in HCC:
        s = cells == h
        if s.any():
            p, b = fb.unit_mean(d, s); lo, hi = interval(b)
            rows.append(dict(contrast=name, panel=h, diff=p, lo95=lo, hi95=hi, n_units=int(s.sum()), **extra)); hb.append((p, b))
    if len(hb) == 3:
        p = float(np.mean([x[0] for x in hb])); lo, hi = interval(np.mean([x[1] for x in hb], 0))
        rows.append(dict(contrast=name, panel="hcc_mean", diff=p, lo95=lo, hi95=hi, **extra))
    return rows


def main():
    t0 = time.time()
    fr = gate()
    S = Store()
    u = S.units
    part1 = load_json(os.path.join("manifests", "r7_xpert_followup_v2", "selection_population_audit_part1.json"))
    # ---- (1) correction impact --------------------------------------------------------------------------------
    test_idx = S.test
    Yt = S.test_delta(test_idx).astype(np.float64)
    ct, ft, ht, gt = u.cell.to_numpy()[test_idx], u.family.to_numpy()[test_idx], u.is_hcc.to_numpy()[test_idx] == 1, u.drug_group.to_numpy()[test_idx]
    vsrc = S.val[u.is_hcc.to_numpy()[S.val] == 0]
    Yv = S.delta(vsrc).astype(np.float64)
    cv_, fv = u.cell.to_numpy()[vsrc], u.family.to_numpy()[vsrc]
    vpos = np.searchsorted(S.val, vsrc) if (np.diff(S.val) > 0).all() else np.array([list(S.val).index(i) for i in vsrc])
    fbt = FamBoot(gt, ft, n_boot=2000, seed=20261117)
    sel_rows, per_seed, crow, epochs = [], [], [], []
    tabs = {}
    for var, olds in ORIG_XPERT_RUNS.items():
        for seed, old in zip(SEEDS, olds):
            d = runs("traj", f"full_{var}_L0_s{seed}")
            r = load_json(os.path.join(d, "record.json"))
            assert (np.load(os.path.join(d, "unit_idx_test.npy")) == test_idx).all() and (np.load(os.path.join(d, "unit_idx_val.npy")) == S.val).all()
            ck_old = part1["runs"][old]["checkpoint_epoch"]
            epochs.append(dict(variant=var, seed=seed, old_original_epoch=ck_old, replay_epoch=r["selectors"]["U_all"]["epoch"],
                               corrected_epoch=r["selectors"]["U"]["epoch"], replay_score_all_valid=r["selectors"]["U_all"]["score"],
                               corrected_score_src_valid=r["selectors"]["U"]["score"], same_checkpoint_replay_vs_corrected=r["selectors"]["U_all"]["epoch"] == r["selectors"]["U"]["epoch"],
                               old_logged_loss4_all_valid=part1["runs"][old]["checkpoint_logged_loss4"]))
            preds = dict(old_original=(np.load(orig_xpert(old, "pred_test.npy")), np.load(orig_xpert(old, "pred_validation.npy"))),
                         old_replay=(np.load(os.path.join(d, "pred_U_all_test.npy")), np.load(os.path.join(d, "pred_U_all_val.npy"))),
                         corrected=(np.load(os.path.join(d, "pred_U_test.npy")), np.load(os.path.join(d, "pred_U_val.npy"))))
            for name, (pt, pv) in preds.items():
                m = unit_metrics(pt, Yt, ct)
                leg = legacy_unit_metrics(pt, Yt, ct)
                m["mrr_legacy_r7"] = leg.mrr.to_numpy()
                tabs.setdefault((var, name), []).append(m)
                s = summary(m, ct, ft, ht); s["src_mrr_legacy_r7"] = fc.fam_mean(leg.mrr.to_numpy()[~ht], ft[~ht])
                per_seed.append(dict(variant=var, seed=seed, model=name, panel="test", **s))
                mv = unit_metrics(pv[vpos], Yv, cv_)
                sel_rows.append(dict(variant=var, seed=seed, model=name, panel="validation_source (selection population)",
                                     **{k: v for k, v in summary(mv, cv_, fv, np.zeros(len(vsrc), bool)).items() if k.startswith("src_")}))
    for var in ORIG_XPERT_RUNS:
        mt = {n: sum(tabs[(var, n)]) / 3 for n in ["old_original", "old_replay", "corrected"]}
        for name, a_, b_ in [("corrected - old_original", "corrected", "old_original"), ("corrected - old_replay", "corrected", "old_replay"),
                             ("old_replay - old_original", "old_replay", "old_original")]:
            for metric in ["rmse", "centred_pearson", "mrr_tie"]:
                d = mt[a_][metric].to_numpy() - mt[b_][metric].to_numpy()
                rows = contrast(fbt, d, ct, ht, name, dict(variant=var, metric=metric))
                crow += rows if metric == "rmse" else rows[:1]
    out = pman("final")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(per_seed).to_csv(os.path.join(out, "correction_test_by_seed.tsv"), sep="\t", index=False)
    pd.DataFrame(sel_rows).to_csv(os.path.join(out, "correction_selection_population_scores.tsv"), sep="\t", index=False)
    pd.DataFrame(crow).to_csv(os.path.join(out, "correction_impact_contrasts.tsv"), sep="\t", index=False)
    pd.DataFrame(epochs).to_csv(os.path.join(out, "correction_selected_epochs.tsv"), sep="\t", index=False)
    ps = pd.DataFrame(per_seed)
    seedmean = ps.groupby(["variant", "model"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    seedmean.to_csv(os.path.join(out, "correction_test_seedmean.tsv"), sep="\t", index=False)

    audit = dict(part1)
    audit["correction"] = dict(
        method="retrain (no per-epoch snapshots): same L0 recipe/config/seeds/100 epochs/patience 50 emulated; one trajectory per seed records "
               "both the original rule (all validation units incl. HCC; replay) and the corrected rule (validation is_hcc == 0)",
        selected_epochs=epochs,
        test_impact_rmse_source=[r for r in crow if r["panel"] == "source_family_equal" and r["metric"] == "rmse"],
        test_seedmean=seedmean.to_dict("records"),
        exposure="R4 test is an exposed benchmark; correction re-evaluation, not a blind test", freeze=fr)
    dump(audit, pman("selection_population_audit.json"))

    # ---- (2) section 4.2 step 6: full-train cross-fit coefficients applied to the corrected final base -----------
    C = Ctx(S, load_folds())
    rows, crows, coefs = [], [], []
    tab6 = {}
    for seed in SEEDS:
        Poof = load_outer(C, "L0", "U", seed)
        pt = np.load(os.path.join(runs("traj", f"full_official_L0_s{seed}"), "pred_U_test.npy")).astype(np.float64)
        for arm in fc.ARMS:
            lam, cvs = fc.select_lambda(arm, Poof, C.Y, C.cells, C.fam, C.hcc, C.outer)
            coef = fc.fit(arm, lam, Poof, C.Y, C.cells, C.fam, C.hcc)
            Z, fbk = fc.apply(coef, pt, ct)
            m = unit_metrics(Z, Yt, ct)
            tab6.setdefault(arm, []).append(m)
            rows.append(dict(seed=seed, arm=arm, chosen_lambda="identity" if not np.isfinite(lam) else lam, n_fallback=int(fbk.sum()), **summary(m, ct, ft, ht)))
            coefs.append(dict(seed=seed, **fc.coef_summary(coef), coef_sha=fc.coef_hash(coef), **{f"cv_{k}": v for k, v in cvs.items() if k is not None}))
            if arm != "C0":
                c0 = tab6["C0"][-1]
                rows[-1]["inv_max_abs_centred_pearson"] = float(np.abs(m.centred_pearson - c0.centred_pearson).max())
                rows[-1]["inv_max_abs_mrr_tie"] = float(np.abs(m.mrr_tie - c0.mrr_tie).max())
    mt = {k: sum(v) / 3 for k, v in tab6.items()}
    for arm in ["Csb", "Cs", "Cb", "Cg"]:
        for metric in ["rmse", "centred_pearson", "mrr_tie"]:
            r_ = contrast(fbt, mt[arm][metric].to_numpy() - mt["C0"][metric].to_numpy(), ct, ht, f"{arm}-C0", dict(metric=metric))
            crows += r_ if metric == "rmse" else r_[:1]
    pd.DataFrame(rows).to_csv(os.path.join(out, "step6_test_by_seed.tsv"), sep="\t", index=False)
    pd.DataFrame(crows).to_csv(os.path.join(out, "step6_test_contrasts.tsv"), sep="\t", index=False)
    pd.DataFrame(coefs).to_csv(os.path.join(out, "step6_coefficients.tsv"), sep="\t", index=False)
    rec = dict(freeze=fr, exposure="frozen follow-up evaluation on the exposed R4 test (not blind)", n_test_units=int(len(test_idx)),
               n_test_source_families=int(len(np.unique(ft[~ht]))), seconds=round(time.time() - t0, 1), **stamp())
    dump(rec, os.path.join(out, "final_eval_record.json"))
    print(pd.DataFrame(crow).query("panel == 'source_family_equal'").to_string())
    print(pd.DataFrame(crows).query("panel == 'source_family_equal'").to_string())
    print(pd.DataFrame(epochs).to_string())


if __name__ == "__main__":
    main()
