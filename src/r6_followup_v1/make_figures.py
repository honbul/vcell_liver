"""Figures of the follow-up report, drawn only from the committed tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("reports/r6_followup_v1")
F = R / "figures"
COL = {"poolmean": "#2a78d6", "linemean": "#eb6834", "S": "#1baf7a", "SD": "#eda100", "M": "#e87ba4", "R": "#008300", "F1": "#6250d6", "oracle": "#e34948"}
LAB = {"poolmean": "pool mean (no model)", "linemean": "cell-line mean", "S": "S: control set, no drug input", "SD": "SD: control set + drug (R6 B00)",
       "M": "M: set mean + drug", "R": "R: ridge on set mean + drug", "F1": "F1: pool-mean anchor + pair loss"}
INK, MUTED = "#1f1f1e", "#6b6a65"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color="#ecebe6", lw=0.8)
    ax.set_axisbelow(True)


def fig_total_vs_within():
    t = pd.read_csv(R / "input_capacity_controls.tsv", sep="\t")
    a = t[(t.row_type == "arm") & (t.subset == "all") & (t.p == 0.25)].set_index("arm")
    extra = R / "anchor_pairloss.tsv"
    if extra.exists():
        e = pd.read_csv(extra, sep="\t")
        e = e[(e.row_type == "arm") & (e.subset == "all") & (e.p == 0.25) & (e.arm == "F1")].set_index("arm")
        a = pd.concat([a, e])
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    style(ax)
    for n in [k for k in LAB if k in a.index]:
        r = a.loc[n]
        ax.errorbar(r.total_rmse, r.within_rmse, xerr=[[r.total_rmse - r.total_lo], [r.total_hi - r.total_rmse]],
                    yerr=[[r.within_rmse - r.within_lo], [r.within_hi - r.within_rmse]], fmt="o", ms=8, color=COL[n], ecolor=COL[n], elinewidth=1, alpha=0.95,
                    mec="white", mew=1.5, label=LAB[n])
    pc = pd.read_csv(R / "positive_control_calibration.tsv", sep="\t")
    o = pc[(pc.row_type == "arm") & (pc.target == "full_well_truth_R6") & (pc.arm == "oracle_halfA_disjoint")]
    if len(o):
        ax.axhline(float(o.within_rmse.iloc[0]), color=COL["oracle"], lw=1.5, ls="--")
        ax.text(ax.get_xlim()[1], float(o.within_rmse.iloc[0]), " half-well oracle (technical split)", color=INK, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("total RMSE (lower is better)", color=INK); ax.set_ylabel("within-pool contrast RMSE (lower is better)", color=INK)
    ax.set_title("Source out-of-fold, p = 0.25: total error moves, drug contrast does not", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(F / "total_vs_within_contrast.png"); plt.close(fig)


def fig_pool_improvement():
    pc = pd.read_csv("predictions/r6_followup_v1/confirm_per_condition.tsv.gz", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=160, sharey=False)
    for ax, key, name in ((axes[0], "pool", "control pools"), (axes[1], "line", "cell lines")):
        style(ax)
        for comp, c in (("poolmean", COL["poolmean"]), ("linemean", COL["linemean"])):
            dd = (pc["rmse_SD_p0.25"] - pc[f"rmse_{comp}_p0.25"]).groupby(pc[key]).mean()
            ax.hist(dd, bins=30, histtype="stepfilled", alpha=0.55, color=c, edgecolor="white", label=f"SD - {LAB[comp]}")
        ax.axvline(0, color=INK, lw=1)
        ax.set_xlabel(f"mean RMSE difference per {name[:-1]} (negative = SD better)", color=INK, fontsize=9)
        ax.set_ylabel(f"number of {name}", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Where SD beats the cell-line mean it does not beat the pool mean (source out-of-fold, p = 0.25)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(F / "pool_line_improvement_distribution.png"); plt.close(fig)


def fig_budget():
    p_ = R / "observation_budget.tsv"
    if not p_.exists():
        return
    t = pd.read_csv(p_, sep="\t")
    a = t[t.row_type == "arm"]
    arms = [k for k in ("S", "SD", "M", "R") if k in set(a.arm)]
    fig, axes = plt.subplots(1, len(arms), figsize=(3.1 * len(arms), 3.6), dpi=160, sharey=True)
    axes = np.atleast_1d(axes)
    ls = {1.0: "-", 0.5: "--", 0.25: ":"}
    for ax, arm in zip(axes, arms):
        style(ax)
        for p, g in a[a.arm == arm].groupby("p"):
            g = g.sort_values("n_cells")
            ax.plot(g.n_cells * g.p, g.total_rmse, ls[p], color=COL[arm], lw=2, marker="o", ms=5, mec="white", label=f"p = {p}")
        pm = float((a.total_rmse - a.total_minus_poolmean).iloc[0])
        ax.axhline(pm, color=COL["poolmean"], lw=1.5)
        ax.set_xscale("log", base=2); ax.set_title(LAB[arm].split(":")[0] + " (" + LAB[arm].split(": ")[1] + ")", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("budget N x p (cell equivalents)", fontsize=9, color=INK)
    axes[0].set_ylabel("total RMSE", color=INK); axes[0].legend(frameon=False, fontsize=8)
    axes[-1].text(0.98, pm, "pool mean", transform=axes[-1].get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=INK)
    fig.suptitle("Inference-only observation budget (models trained at N = 64)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(F / "observation_budget_curve.png"); plt.close(fig)


def fig_teacher():
    t = pd.read_csv(R / "teacher_depth_interaction.tsv", sep="\t")
    t["pair"] = t.candidate + " vs " + t.base
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=160)
    for ax, metric in zip(axes, ("total", "within")):
        style(ax)
        g = t[t.metric == metric]
        roles = list(dict.fromkeys(g.split_role))
        y = np.arange(len(g))[::-1]
        c = [COL["poolmean"] if r == roles[0] else COL["linemean"] for r in g.split_role]
        ax.errorbar(g.I_depth, y, xerr=[g.I_depth - g.I_lo, g.I_hi - g.I_depth], fmt="none", ecolor=MUTED, elinewidth=1)
        ax.scatter(g.I_depth, y, c=c, s=60, edgecolors="white", linewidths=1.5, zorder=3)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y); ax.set_yticklabels([f"{p_}  [{r.replace('_', ' ')}]" for p_, r in zip(g.pair, g.split_role)], fontsize=7.5)
        ax.set_title(f"{metric} RMSE: gain at p = 0.25 minus gain at p = 1", fontsize=9, color=INK, loc="left")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("Depth interaction of the teacher (existing R6 predictions; positive = helps more at low depth)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(F / "teacher_depth_interaction.png"); plt.close(fig)


if __name__ == "__main__":
    F.mkdir(parents=True, exist_ok=True)
    for f in (fig_total_vs_within, fig_pool_improvement, fig_budget, fig_teacher):
        f(); print("ok", f.__name__)
