"""Figures of plan section 13.2 from the committed-to-be follow-up tables (no recomputation).

  1 covariance V0 -> V3: v_r raw / floored per partition, and the performance change of the observation model (source OOF, E1, E2)
  2 E1 vs E2 paired contrasts: common correction, per-drug correction, covariance (family bootstrap 95% CI)
  3 C1 kernel composition (block-normalised, own lambda / eta) and the direct residual shuffle
  4 chosen branches: J models and the support budget
  5 HCC per target (post-freeze) with the support cost
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import pub_man, public, TRACK

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]     # fixed categorical order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def fig_path(n):
    return public("reports", TRACK, "figures", n)


def interval_panel(ax, labels, d, lo, hi, colors, title, xlabel="difference in delta RMSE (negative = first method better)"):
    y = np.arange(len(labels))[::-1]
    for yi, di, l, h, c in zip(y, d, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot([di], [yi], "o", color=c, ms=6, mec="white", mew=1.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel(xlabel)


def fig1():
    v = pd.read_csv(pub_man("covariance_repair_versions.tsv"), sep="\t")
    v = v[v.partition.isin(["global", "outer0", "outer1", "outer2"]) & v.version.isin(["V0", "V1", "V2", "V3"])]
    fh = pd.read_csv(pub_man("fixed_hyper_vs_retuned.tsv"), sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    parts = ["global", "outer0", "outer1", "outer2"]
    w = 0.2
    for i, ver in enumerate(["V0", "V1", "V2", "V3"]):
        t = v[v.version == ver].set_index("partition").reindex(parts)
        x = np.arange(len(parts)) + (i - 1.5) * w
        ax.bar(x, t.v_r_raw, width=w - 0.02, color=C[i], label=ver)
        ax.scatter(x, t.v_r_floor, marker="_", s=120, color=INK, zorder=3, label="floor 0.05 v_tot" if i == 0 else None)
    ax.set_xticks(np.arange(len(parts)))
    ax.set_xticklabels(parts)
    ax.set_ylabel("v_r before floor (gene-averaged variance)")
    ax.set_title("1a  residual signal variance by version (bars) and the floor (ticks)", loc="left", fontsize=10)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax = axes[1]
    rows = []
    for s in ("source_oof", "E1", "E2"):
        for a_, b_, lab in (("C0_V3", "C0_iid", "C0 V3 - iid"), ("C1_V3", "C1_iid", "C1 V3 - iid"), ("C0_V3", "C0_V0", "C0 V3 - V0")):
            r = fh[(fh.set == s) & (fh.K == 16) & (fh.metric == "delta_rmse") & (fh.a == a_) & (fh.b == b_)]
            if len(r):
                rows.append((f"{s}: {lab}", r["diff"].iloc[0], r.lo.iloc[0], r.hi.iloc[0], C[["source_oof", "E1", "E2"].index(s)]))
    interval_panel(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows],
                   "1b  observation model vs iid (K=16, nested source-retuned)")
    fig.tight_layout()
    fig.savefig(fig_path("fig1_covariance_versions.png"), dpi=150)


def fig2():
    m = pd.read_csv(pub_man("matched_plate_repaired.tsv"), sep="\t")
    m = m[(m.K == 16) & (m.metric == "delta_rmse")]
    rows = []
    for name, lab in (("C0_iid-shrunk_mean", "common correction (C0 - shrunk mean)"), ("C1_iid-C0_iid", "per-drug correction (C1 - C0)"),
                      ("C0_V3-C0_iid", "covariance V3 (C0)"), ("C1_V3-C1_iid", "covariance V3 (C1)")):
        for j, s in enumerate(("E1", "E2")):
            r = m[(m.row == f"contrast_{s}") & (m.name == name)]
            if len(r):
                rows.append((f"{lab}  [{s}]", r["diff"].iloc[0], r.lo.iloc[0], r.hi.iloc[0], C[j]))
    fig, ax = plt.subplots(figsize=(8, 4))
    interval_panel(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows],
                   "2  same plate (E1, blue) vs other X-tag replicate plate (E2, orange); source-train families, out-of-fold, K=16")
    fig.tight_layout()
    fig.savefig(fig_path("fig2_e1_e2_paired.png"), dpi=150)


def fig3():
    k = pd.read_csv(pub_man("kernel_components.tsv"), sep="\t")
    r = pd.read_csv(pub_man("residual_components.tsv"), sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = []
    for j, s in enumerate(("source_oof", "E1", "E2")):
        for cfg in ("chem", "f0", "ctrl", "chem+f0", "all"):
            t = k[(k.set == s) & (k.K == 16) & (k.metric == "delta_rmse") & (k.a == f"kn_{cfg}") & (k.b == "C0_iid")]
            if len(t):
                rows.append((f"{s}: {cfg}", t["diff"].iloc[0], t.lo.iloc[0], t.hi.iloc[0], C[j]))
    interval_panel(axes[0], [x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows], [x[3] for x in rows], [x[4] for x in rows],
                   "3a  kernel block (normalised scale, own lambda/eta) vs C0")
    rows = []
    for j, s in enumerate(("source_oof", "E1", "E2")):
        for name, lab in (("C1_iid-C0_iid", "correct residuals - C0"), ("C1_iid_shuf1-C0_iid", "shuffled residuals - C0")):
            t = r[(r.set == s) & (r.K == 16) & (r.row == "contrast") & (r.name == name) & (r.metric == "delta_rmse")]
            if len(t):
                rows.append((f"{s}: {lab}", t["value"].iloc[0], t.lo.iloc[0], t.hi.iloc[0], C[j]))
    interval_panel(axes[1], [x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows], [x[3] for x in rows], [x[4] for x in rows],
                   "3b  C1 drug-residual correspondence: correct vs shuffled")
    fig.tight_layout()
    fig.savefig(fig_path("fig3_kernel_and_shuffle.png"), dpi=150)


def fig4():
    j = pd.read_csv(pub_man("joint_model_results.tsv"), sep="\t")
    j = j[(j.row == "contrast") & (j.K == 16) & (j.metric == "delta_rmse")]
    rows = []
    for k_, s in enumerate(("source_oof", "E1", "E2")):
        for a_, b_ in (("J1", "C0_iid"), ("J2", "C0_iid"), ("J2", "J1"), ("J3", "J2"), ("J3", "C1_iid")):
            t = j[(j.set == s) & (j.a == a_) & (j.b == b_)]
            if len(t):
                rows.append((f"{s}: {a_} - {b_}", t["diff"].iloc[0], t.lo.iloc[0], t.hi.iloc[0], C[k_]))
    has_b = os.path.exists(pub_man("support_budget.tsv"))
    fig, axes = plt.subplots(1, 2 if has_b else 1, figsize=(11 if has_b else 7, 4.5))
    axes = np.atleast_1d(axes)
    interval_panel(axes[0], [x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows], [x[3] for x in rows], [x[4] for x in rows],
                   "4a  observation-environment models (K=16, out-of-fold)")
    if has_b:
        b = pd.read_csv(pub_man("support_budget.tsv"), sep="\t")
        b = b[(b.row == "D16_minus_REP8")]
        rows = []
        for i, met in enumerate(("delta_rmse", "centred_rmse")):
            for m in ("C0_common", "C1_common", "C0_reselected", "C1_reselected"):
                t = b[(b.method == m) & (b.metric == met)]
                if len(t):
                    rows.append((f"{met}: {m}", t["diff"].iloc[0], t.lo.iloc[0], t.hi.iloc[0], C[i]))
        interval_panel(axes[1], [x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows], [x[3] for x in rows], [x[4] for x in rows],
                       "4b  16 drugs x 1 row  minus  8 drugs x 2 rows", xlabel="D16 - REP8 (negative = more drugs better)")
    fig.tight_layout()
    fig.savefig(fig_path("fig4_branches.png"), dpi=150)


def fig5():
    h = pd.read_csv(pub_man("hcc_followup_results.tsv"), sep="\t")
    c = pd.read_csv(pub_man("hcc_followup_contrasts.tsv"), sep="\t")
    c = c[(c.K == 16) & (c.query_set == "test") & (c.metric == "delta_rmse")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rows = []
    names = list(dict.fromkeys(c.contrast))[:6]
    for t_i, t in enumerate(("HUH7", "JHH5", "JHH7", "EQUAL_MEAN_3")):
        for n in names:
            r = c[(c.target == t) & (c.contrast == n)]
            if len(r):
                rows.append((f"{t}: {n}", r["diff"].iloc[0], r.lo.iloc[0], r.hi.iloc[0], C[t_i]))
    interval_panel(ax, [x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows], [x[3] for x in rows], [x[4] for x in rows],
                   "5  HCC per target, K=16 test queries (existing-benchmark follow-up evaluation)")
    fig.tight_layout()
    fig.savefig(fig_path("fig5_hcc_by_target.png"), dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="+", type=int)
    a = ap.parse_args()
    for f in a.figs:
        {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}[f]()
        print("figure", f, "written")
