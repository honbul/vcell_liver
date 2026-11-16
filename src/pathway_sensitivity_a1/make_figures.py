#!/usr/bin/env python3
"""Regenerate the manuscript figures from committed result tables (and, for the case figure, the
out-of-fold tables joined to observed responses).

Usage: make_figures.py --tables reports/pathway_sensitivity_a1/tables --out reports/pathway_sensitivity_a1/figures
                       [--oof-dir reports/pathway_sensitivity_a1/oof]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COLOR = {"A0": "#8a8983", "A1": "#2a78d6", "A1r": "#eb6834", "A2": "#1baf7a", "A3": "#eda100", "A4": "#e87ba4"}
LABEL = {"A0": "A0 mean", "A1": "A1 genes", "A1r": "A1r gene ranks", "A2": "A2 rank PCA", "A3": "A3 Hallmark",
         "A4": "A4 permuted sets"}
ORDER = ["A0", "A1", "A1r", "A2", "A3"]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_design(out, res):
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc="#ffffff"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=INK2, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    box(0.0, 0.35, 0.17, 0.3, "Baseline RNA\n(LIMORE, %d HCC models\n%d families)" % (res["n_models"], res["n_families"]))
    reps = ["A1 gene log2 FPKM", "A1r within-sample ranks", "A2 PCA of ranks", "A3 Hallmark singscore"]
    for i, r in enumerate(reps):
        box(0.25, 0.78 - i * 0.22, 0.22, 0.16, r)
        ax.annotate("", xy=(0.25, 0.86 - i * 0.22), xytext=(0.17, 0.5), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.annotate("", xy=(0.55, 0.5), xytext=(0.47, 0.86 - i * 0.22), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    box(0.55, 0.35, 0.18, 0.3, "Same Ridge,\nsame alpha grid,\nnested family-group CV")
    box(0.81, 0.35, 0.19, 0.3, "Predicted Activity Area\nper drug (%d drugs)\nvs A0 training mean" % res["n_drugs"])
    ax.annotate("", xy=(0.81, 0.5), xytext=(0.73, 0.5), arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.set_title("Study design: one predictor, four representations of the same RNA", loc="left", fontsize=11)
    save(fig, out, "fig1_design")


def fig_internal(out, tables):
    s = pd.read_csv(os.path.join(tables, "internal_summary.tsv"), sep="\t").set_index("model").loc[ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(ORDER))[::-1]
    for yi, m in zip(y, ORDER):
        r = s.loc[m]
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=COLOR[m], lw=2, solid_capstyle="round")
        ax.plot(r.macro_drug_normalized_rmse, yi, "o", color=COLOR[m], ms=8, mec=SURFACE, mew=2)
        ax.text(r.ci_high + 0.004, yi, f"{r.macro_drug_normalized_rmse:.3f}", va="center", fontsize=9, color=INK)
    ax.axvline(s.loc["A0"].macro_drug_normalized_rmse, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in ORDER])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("macro drug-normalised RMSE (lower is better); bar = 95% family bootstrap interval")
    ax.set_title("Internal out-of-fold error, mean of 3 repeats", loc="left", fontsize=11)
    save(fig, out, "fig2_internal")


def fig_learning(out, tables):
    lc = pd.read_csv(os.path.join(tables, "learning_curve.tsv"), sep="\t")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    off = dict(zip(ORDER, np.linspace(-0.012, 0.012, len(ORDER))))
    for m in ORDER:
        t = lc[lc.model == m].sort_values("train_fraction")
        x = t.train_fraction.values + off[m]
        ax.plot(x, t.macro_drug_normalized_rmse, "-o", color=COLOR[m], lw=2, ms=7, mec=SURFACE, mew=1.5, label=LABEL[m])
        ax.vlines(x, t.ci_low, t.ci_high, color=COLOR[m], lw=1.2, alpha=0.7)
    ax.set_xticks([0.5, 1.0])
    ax.set_xticklabels(["50% of training families", "100%"])
    ax.set_xlim(0.4, 1.1)
    ax.set_ylabel("macro drug-normalised RMSE")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Learning curve (same test families)", loc="left", fontsize=11)
    save(fig, out, "fig3_learning_curve")


def fig_a4(out, tables):
    a4 = pd.read_csv(os.path.join(tables, "a4_permutations.tsv"), sep="\t")
    res = json.load(open(os.path.join(tables, "controls_results.json")))["a4"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    rs = np.random.RandomState(0)
    ax.plot(a4.macro_drug_normalized_rmse, rs.uniform(-0.12, 0.12, len(a4)), "o", color=COLOR["A4"], ms=8,
            mec=SURFACE, mew=1.5, label="A4: 20 permuted gene-label sets")
    ax.plot(res["A3"], 0, "D", color=COLOR["A3"], ms=10, mec=INK, mew=1, label="A3: real Hallmark sets")
    ax.axvline(res["A0"], color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(res["A0"], 0.42, " A0 mean", fontsize=8.5, color=INK2, va="top")
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("macro drug-normalised RMSE (lower is better)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Real pathways vs size- and overlap-matched permuted pathways", loc="left", fontsize=11)
    save(fig, out, "fig4_permuted_pathways")


def fig_external(out, tables):
    w = pd.read_csv(os.path.join(tables, "external_per_drug.tsv"), sep="\t")
    pre = "cross_study_leave_family_out__"
    w = w.sort_values("measured_AA_vs_AUC_spearman").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(w))
    ax.plot(w.measured_AA_vs_AUC_spearman, y, "|", color=INK, ms=14, mew=2, label="measured AA vs measured AUC (no model)")
    for m in ["A1", "A1r", "A2", "A3"]:
        ax.plot(w[pre + m], y, "o", color=COLOR[m], ms=7, mec=SURFACE, mew=1.2, label=LABEL[m])
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(w.drug_id)
    ax.set_xlabel("Spearman with LCCL sensitivity (-AUC) across shared families\n(LIMORE family left out of training)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("External rank agreement per matched drug (same lineage, other study)", loc="left", fontsize=11)
    save(fig, out, "fig5_external")


def fig_cases(out, tables, oof_dir):
    import core
    cfg = core.load_config()
    d = pd.read_csv(os.path.join(tables, "internal_per_drug.tsv"), sep="\t")
    d = d[~d.constant_response]
    picks = [("lowest A3 error", d.sort_values("nrmse_A3").drug_id.iloc[0]),
             ("highest A3 error", d.sort_values("nrmse_A3").drug_id.iloc[-1])]
    long = core.load_limore_response(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, (why, drug) in zip(axes, picks):
        for m in ("A1", "A3"):
            p = pd.read_csv(os.path.join(oof_dir, f"{m}__rep0.tsv.gz"), sep="\t")
            p = p[p.drug_id == drug].merge(long[long.drug_id == drug][["model_id", "y"]], on="model_id", validate="one_to_one")
            ax.plot(p.y, p.pred, "o", color=COLOR[m], ms=6, mec=SURFACE, mew=1, label=LABEL[m], alpha=0.9)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls=(0, (3, 3)))
        r = d.set_index("drug_id").loc[drug]
        ax.set_title(f"{drug} ({why})\nnRMSE A0 {r.nrmse_A0:.2f}, A1 {r.nrmse_A1:.2f}, A3 {r.nrmse_A3:.2f}", loc="left", fontsize=9.5)
        ax.set_xlabel("observed Activity Area")
        ax.set_ylabel("out-of-fold prediction (repeat 0)")
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig6_cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oof-dir")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = json.load(open(os.path.join(a.tables, "internal_results.json")))
    fig_design(a.out, res)
    fig_internal(a.out, a.tables)
    if os.path.exists(os.path.join(a.tables, "learning_curve.tsv")):
        fig_learning(a.out, a.tables)
        fig_a4(a.out, a.tables)
    if os.path.exists(os.path.join(a.tables, "external_per_drug.tsv")):
        fig_external(a.out, a.tables)
    if a.oof_dir:
        fig_cases(a.out, a.tables, a.oof_dir)
