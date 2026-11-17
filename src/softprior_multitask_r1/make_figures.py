#!/usr/bin/env python3
"""Figures for the R1 report, drawn from the committed result tables only.

Usage: make_figures.py --tables reports/softprior_multitask_r1/tables --out reports/softprior_multitask_r1/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
GROUP = {"baseline": "#8a8983", "hard prior": "#eda100", "soft prior": "#1baf7a", "multi-drug": "#2a78d6",
         "integrated": "#eb6834", "public method": "#e87ba4"}
KIND = {"A0": "baseline", "A1": "baseline", "A1r": "baseline", "enet": "baseline", "ridge_pd": "baseline",
        "ridge_gl": "baseline", "A3": "hard prior", "par_pd": "hard prior", "par_gl": "hard prior",
        "soft_pd": "soft prior", "soft_gl": "soft prior", "mt_noprior": "multi-drug", "mt_resid": "multi-drug",
        "full": "integrated", "full_resid": "integrated", "cwkbmf": "public method", "tabpfn": "public method"}
LABEL = {"A0": "training mean", "A1": "log2 gene Ridge (Track A)", "A1r": "gene-rank Ridge (Track A grid)",
         "A3": "Hallmark-score Ridge (Track A)", "enet": "raw-RNA ElasticNet", "ridge_pd": "gene-rank Ridge, per-drug lambda",
         "ridge_gl": "gene-rank Ridge, one lambda", "par_pd": "pathway subspace only, per-drug",
         "par_gl": "pathway subspace only, one lambda", "soft_pd": "soft prior, per-drug", "soft_gl": "soft prior, one setting",
         "mt_noprior": "multi-drug low-rank, no prior", "full": "soft prior + low-rank (initial design)",
         "mt_resid": "low-rank + per-drug residual, no prior", "full_resid": "soft prior + low-rank + residual (final)",
         "cwkbmf": "cwKBMF (authors' code, Octave)", "tabpfn": "TabPFN v2 on 50 PCs"}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def dot_ci(ax, tab, value, lo, hi, names, ref=None):
    y = np.arange(len(tab))[::-1]
    for yi, (_, r) in zip(y, tab.iterrows()):
        c = GROUP[KIND.get(r[names], "baseline")]
        ax.plot([r[lo], r[hi]], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(r[value], yi, "o", color=c, ms=6, mec=SURFACE, mew=1)
        ax.annotate(f"{r[value]:.3f}", (r[hi], yi), textcoords="offset points", xytext=(5, -3), fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(m, m) for m in tab[names]])
    ax.grid(axis="y", visible=False)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=0.8, ls=(0, (3, 3)))


def legend(ax, kinds):
    h = [plt.Line2D([], [], marker="o", ls="", color=GROUP[k], label=k) for k in GROUP if k in kinds]
    ax.legend(handles=h, frameon=False, fontsize=8, loc="lower right")


def fig_internal(T, out):
    s = pd.read_csv(os.path.join(T, "final_internal_summary.tsv"), sep="\t").sort_values("macro_drug_normalized_rmse")
    fig, ax = plt.subplots(figsize=(7.4, 0.33 * len(s) + 1.2))
    dot_ci(ax, s, "macro_drug_normalized_rmse", "ci_low", "ci_high", "model", ref=float(s[s.model == "A0"].macro_drug_normalized_rmse.iloc[0]))
    legend(ax, {KIND[m] for m in s.model})
    ax.set_xlabel("macro-drug normalised RMSE, nested family CV (lower is better; bar = 95% family bootstrap)")
    ax.set_title("LIMORE, unseen HCC families: every model and its interval", loc="left", fontsize=10.5)
    save(fig, out, "fig1_internal")


def fig_differences(T, out):
    d = pd.read_csv(os.path.join(T, "final_internal_differences.tsv"), sep="\t")
    d = d[d.model == "full_resid"].copy()
    d["name"] = d.reference
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(d) + 1.2))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        c = GROUP[KIND.get(r.reference, "baseline")]
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(r.difference, yi, "o", color=c, ms=6, mec=SURFACE, mew=1)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(m, m) for m in d.reference])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("final integrated model minus reference (negative = integrated model better)")
    ax.set_title("Paired differences on the same families", loc="left", fontsize=10.5)
    save(fig, out, "fig2_paired_differences")


def fig_perm(T, out):
    rp = pd.read_csv(os.path.join(T, "final_internal_real_vs_permuted_prior.tsv"), sep="\t")
    ps = pd.read_csv(os.path.join(T, "final_internal_permuted_prior_scores.tsv"), sep="\t")
    fig, axes = plt.subplots(1, len(rp), figsize=(3.4 * len(rp), 2.9))
    for ax, (_, r) in zip(np.atleast_1d(axes), rp.iterrows()):
        v = ps[ps.model == r.model].macro_drug_normalized_rmse.values
        ax.plot(v, np.zeros_like(v) + np.random.RandomState(1).uniform(-0.25, 0.25, len(v)), "o", color=GROUP["baseline"],
                ms=5, mec=SURFACE, mew=0.8, label="20 permuted gene-label priors")
        ax.axvline(r.real, color=GROUP["integrated"], lw=2, label="real Hallmark prior")
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        pad = max(np.ptp(np.append(v, r.real)) * 0.25, 2e-4)
        ax.set_xlim(min(v.min(), r.real) - pad, max(v.max(), r.real) + pad)
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(LABEL.get(r.model, r.model), fontsize=8.5, loc="left")
        ax.set_xlabel("macro nRMSE")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, out, "fig3_real_vs_permuted_prior")


def fig_development(T, out):
    names = [("d1", ["ridge_gl", "ridge_pd", "par_gl"]), ("r2", ["soft_gl", "mt_noprior", "full"]), ("r3", ["mt_resid", "full_resid"])]
    rows = []
    for tag, ms in names:
        t = pd.read_csv(os.path.join(T, f"{tag}_dev_summary.tsv"), sep="\t").set_index("model")
        rows += [dict(model=m, score=t.loc[m, "dev_xsel_macro_nrmse"], stage={"d1": "R1-1", "r2": "R1-2", "r3": "R1-3"}[tag]) for m in ms]
    d = pd.DataFrame(rows)
    fin = pd.read_csv(os.path.join(T, "final_internal_summary.tsv"), sep="\t").set_index("model").macro_drug_normalized_rmse
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(d))[::-1]
    ax.plot(d.score, y, "o", color=INK2, ms=6, label="development (inner folds, selection-honest)")
    ax.plot([fin[m] for m in d.model], y, "D", color=GROUP["integrated"], ms=5, label="frozen nested evaluation (outer folds)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  {LABEL[m]}" for s, m in zip(d.stage, d.model)])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("macro nRMSE (the two scores use different training-set sizes; compare within a marker type)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Design steps: development score and final score", loc="left", fontsize=10.5)
    save(fig, out, "fig4_development_and_final")


def fig_per_drug(T, out):
    p = pd.read_csv(os.path.join(T, "final_internal_per_drug.tsv"), sep="\t").pivot(index="drug_id", columns="model", values="nrmse")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
    for ax, ref in zip(axes, ["ridge_pd", "A3"]):
        lo, hi = min(p[ref].min(), p.full_resid.min()) - 0.01, max(p[ref].max(), p.full_resid.max()) + 0.01
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        ax.plot(p[ref], p.full_resid, "o", color=GROUP["integrated"], ms=4, alpha=0.8, mec=SURFACE, mew=0.5)
        n = int((p.full_resid < p[ref]).sum())
        ax.set_xlabel(f"{LABEL[ref]}\nper-drug nRMSE")
        ax.set_title(f"final model lower in {n} of {len(p)} drugs", fontsize=9, loc="left")
    axes[0].set_ylabel("final integrated model, per-drug nRMSE")
    save(fig, out, "fig5_per_drug")


def fig_external(T, out):
    e = pd.read_csv(os.path.join(T, "final_external_summary.tsv"), sep="\t")
    e = e[e.endpoint == "AUC"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.3 * e.model.nunique() + 1.4), sharey=True)
    order = e[e.eval_type == "cross_study_leave_family_out"].sort_values("mean_spearman", ascending=False).model.tolist()
    for ax, (et, title) in zip(axes, [("cross_study_leave_family_out", "families also in LIMORE (26), family left out of training"),
                                      ("new_lineage", "families not in LIMORE (4): too few for a conclusion")]):
        t = e[e.eval_type == et].set_index("model").loc[order].reset_index()
        dot_ci(ax, t, "mean_spearman", "ci_low", "ci_high", "model", ref=0.0)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("mean Spearman(predicted AA, -AUC) over 18 matched drugs")
    save(fig, out, "fig6_external_lccl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (fig_internal, fig_differences, fig_perm, fig_development, fig_per_drug, fig_external):
        f(a.tables, a.out)
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
