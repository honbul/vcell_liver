"""Small report figures from the committed result tables (no model, no cell data).

    <tag>_error_by_p.png   delta RMSE of each arm at keep-rate 0.25 / 0.5 / 1.0
    <tag>_pairs.png        paired differences (a - b) with drug-group bootstrap intervals
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COL = dict(Z0="#52514e", B00="#2a78d6", B10="#eb6834", B01="#1baf7a", B11="#eda100", DEN="#e87ba4",
           NOSTATE="#008300", TEACH0="#6250d6", linemean="#9a9992")
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="reports/depth_robust_r6/tables")
    ap.add_argument("--out", default="reports/depth_robust_r6/figures")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", nargs="+", default=["linemean", "Z0", "B00", "B10", "B01", "B11", "DEN", "TEACH0"])
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    T, O = Path(a.tables), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(T / f"{a.tag}_main.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150); fig.patch.set_facecolor(SURF); style(ax)
    xs = {0.25: 0, 0.5: 1, 1.0: 2}
    ends = []
    for arm in a.arms:
        g = m[m.arm == arm].sort_values("p")
        if g.empty:
            continue
        x = [xs[p] for p in g.p]
        ax.plot(x, g.delta_rmse, color=COL.get(arm, MUTED), lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=arm,
                ls="--" if arm in ("linemean", "TEACH0") else "-")
        ends.append((float(g.delta_rmse.iloc[0]), arm))
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["p = 0.25", "p = 0.5", "p = 1.0"])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("delta RMSE (drug-group mean; lower is better)", color=MUTED, fontsize=9)
    ax.set_xlabel("fraction of control-cell UMIs kept (N = 64 cells)", color=MUTED, fontsize=9)
    ax.set_title(a.title or f"{a.tag}: error by observation level", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(); fig.savefig(O / f"{a.tag}_error_by_p.png", facecolor=SURF); plt.close(fig)

    pr = pd.read_csv(T / f"{a.tag}_pairs.tsv", sep="\t")
    pr = pr[pr.p.isin([0.25, 1.0])].copy()
    pr["label"] = pr.a + " − " + pr.b
    labs = list(dict.fromkeys(pr.label))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 0.34 * len(labs) + 1.6), dpi=150, sharey=True); fig.patch.set_facecolor(SURF)
    for ax, p in zip(axes, (0.25, 1.0)):
        style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color="#e8e7e2", lw=0.8)
        g = pr[pr.p == p].set_index("label").loc[labs]
        y = range(len(labs))[::-1]
        ax.hlines(y, g.lo, g.hi, color="#2a78d6", lw=2)
        ax.plot(g["diff"], y, "o", color="#2a78d6", ms=6, mec=SURF, mew=1.5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(list(y)); ax.set_yticklabels(labs, fontsize=8, color=INK)
        ax.set_title(f"p = {p}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("paired difference in delta RMSE (negative: first is better)", color=MUTED, fontsize=8)
    fig.tight_layout(); fig.savefig(O / f"{a.tag}_pairs.png", facecolor=SURF); plt.close(fig)
    print("figures ->", O)


if __name__ == "__main__":
    main()
