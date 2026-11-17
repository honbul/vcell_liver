"""Figures for the R3 report, drawn only from the small committed tables (no cell arrays needed)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

T = Path("reports/anchored_distribution_r3/tables")
F = Path("reports/anchored_distribution_r3/figures")
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"pre": "#2a78d6", "anchored": "#eb6834", "anchored2": "#1baf7a"}
VLAB = {"pre": "before restoration", "anchored": "restoration v1", "anchored2": "restoration v2 (frozen)"}
MLAB = {"control": "control as is (native)", "shifted": "shifted control", "shifted@zero": "shifted control, no-change mean",
        "shifted@delta_linemean": "shifted control, line-mean", "shifted_pool": "shifted treated pool",
        "gauss_diag": "Gaussian, diagonal", "gauss_lr50": "Gaussian, low-rank k=50"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, lw=0.8)


def dots(ax, labels, est, lo, hi, color, off=0.0, label=None):
    y = [i + off for i in range(len(labels))]
    ax.hlines(y, lo, hi, color=color, lw=2)
    ax.plot(est, y, "o", color=color, ms=6, mec=SURF, mew=1.2, label=label)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()


def fig_gate():
    d = json.loads((T / "d1_summary.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.9), gridspec_kw=dict(wspace=0.9))
    k = [("all.c_raw", "same well, split in half"), ("all.a", "control vs treated"), ("all.e_raw", "another drug's well vs treated")]
    dots(ax[0], [x[1] for x in k], [d[x[0]]["est"] for x in k], [d[x[0]]["lo"] for x in k], [d[x[0]]["hi"] for x in k], C["pre"])
    style(ax[0], "Metric check: raw energy distance (208 train drugs)")
    k = [("multi.G1_H_b_minus_d_mm", "G1: mean-shifted control minus replicate well"),
         ("multi.G2_e_mm_minus_d_mm", "G2: other drug's well minus replicate well"),
         ("all.b_half_minus_c_mm", "mean-shifted control minus split-half (all wells)")]
    dots(ax[1], [x[1] for x in k], [d[x[0]]["est"] for x in k], [d[x[0]]["lo"] for x in k], [d[x[0]]["hi"] for x in k], C["pre"])
    style(ax[1], "Gate: differences after matching the mean (95% drug-block CI)")
    for a in ax:
        a.set_xlabel("energy distance")
    fig.savefig(F / "fig1_gate_diagnostics.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_methods(table, title, out, variants=("pre", "anchored", "anchored2"), xlim=None):
    R = pd.read_csv(T / table, sep="\t")
    R[["m", "v"]] = R.method.str.split(":", expand=True)
    meths = [m for m in MLAB if m in set(R.m) and m != "control"]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(meths) + 1.4))
    for j, v in enumerate(variants):
        S = R[R.v == v].set_index("m").reindex(meths)
        dots(ax, [MLAB[m] for m in meths], S.ed, S.ed_lo, S.ed_hi, C[v], off=(j - 1) * 0.22, label=VLAB[v])
    style(ax, title)
    ax.set_xlabel("energy distance to the observed treated cells (lower is closer)")
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(F / out, dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_secondary(table, out):
    R = pd.read_csv(T / table, sep="\t")
    R[["m", "v"]] = R.method.str.split(":", expand=True)
    meths = [m for m in MLAB if m in set(R.m) and m not in ("control", "shifted@zero", "shifted@delta_linemean")]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 0.5 * len(meths) + 1.4), gridspec_kw=dict(wspace=0.08), sharey=True)
    for j, v in enumerate(("pre", "anchored", "anchored2")):
        S = R[R.v == v].set_index("m").reindex(meths)
        for a, col in zip(ax, ["zero_frac_mae", "var_ratio"]):
            y = [i + (j - 1) * 0.22 for i in range(len(meths))]
            a.plot(S[col], y, "o", color=C[v], ms=6, mec=SURF, mew=1.2, label=VLAB[v])
    ax[0].set_yticks(range(len(meths))); ax[0].set_yticklabels([MLAB[m] for m in meths]); ax[0].invert_yaxis()
    style(ax[0], "Zero-fraction error per gene (0 = matches observed)")
    style(ax[1], "Total variance, generated / observed (1 = matches)")
    ax[1].axvline(1, color=MUTED, lw=0.8)
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(F / out, dpi=160, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    F.mkdir(parents=True, exist_ok=True)
    fig_gate()
    fig_methods("dev_c1_restoration.tsv", "Development (8 cell lines, 79 train-fold drugs)", "fig2_dev_restoration.png")
    if (T / "final_all_lines.tsv").exists():
        fig_methods("final_all_lines.tsv", "Held-out drugs, all eligible cell lines (post-freeze)", "fig3_final_all_lines.png")
        fig_methods("final_all_lines.tsv", "Held-out drugs, all lines: frozen restoration and before, zoomed",
                    "fig3b_final_zoom.png", variants=("pre", "anchored2"))
        fig_methods("final_snu423.tsv", "Held-out drugs, SNU-423 only (post-freeze)", "fig4_final_snu423.png")
        fig_secondary("final_all_lines.tsv", "fig5_final_zero_variance.png")
    print(sorted(p.name for p in F.glob("*.png")))


if __name__ == "__main__":
    main()
