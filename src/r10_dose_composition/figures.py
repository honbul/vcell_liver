"""Section 11 figures (static PNG, small). Colours: fixed categorical order for the three cell
lines (blue, orange, aqua); reference/neutral marks in grey. Text stays in ink colours."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LINE_COL = {"A549": "#2a78d6", "K562": "#eb6834", "MCF7": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_decomposition(dec: pd.DataFrame, path):
    """Fig 1: mean change vs control split into composition and within-state terms (symmetric ref.)."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    ax = axs[0]
    for line, s in dec.groupby("line"):
        ax.scatter(s.comp_term_sym_rms, s.within_term_sym_rms, s=8, alpha=0.5, color=LINE_COL[line], label=line,
                   edgecolors="none")
    m = float(np.nanmax(dec[["comp_term_sym_rms", "within_term_sym_rms"]].values))
    ax.plot([0, m], [0, m], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("composition term (RMS over PCs)", color=INK, fontsize=9)
    ax.set_ylabel("within-state term (RMS over PCs)", color=INK, fontsize=9)
    ax.set_title("train conditions vs vehicle", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    ax = axs[1]
    d = dec[dec.dmean_rms > dec.dmean_rms.quantile(0.5)]
    for line, s in d.groupby("line"):
        ax.hist(s.comp_share_sym.clip(-0.5, 1.5), bins=40, histtype="step", lw=2, color=LINE_COL[line], label=line)
    ax.set_xlabel("composition share of the mean change (symmetric, clipped)", color=INK, fontsize=9)
    ax.set_ylabel("conditions (upper half of |mean change|)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_failure_map(cond: pd.DataFrame, path):
    """Fig 2: where direct interpolation (mixture) fails, relative to the replicate limit.
    x: response strength (ED of vehicle to the query, descriptive, source drugs only)."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, met, lab in ((axs[0], "comp_tv", "composition TV"), (axs[1], "within_mean_rmse", "within-state mean RMSE")):
        for line, s in cond.groupby("line"):
            ax.scatter(s["strength"].clip(lower=0.01), s[met + "_excess"], s=8, alpha=0.5, color=LINE_COL[line], label=line,
                       edgecolors="none")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("response strength: ED(vehicle, query), clipped at 0.01", color=INK, fontsize=9)
        ax.set_ylabel(f"mixture {lab} minus replicate-oracle", color=INK, fontsize=9)
        _style(ax)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("source (train) drugs, cross-replicate, both episodes", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_forest(pairs: pd.DataFrame, path, title):
    """Fig 3: paired drug-level differences vs D00 (negative = better), 95% bootstrap CI."""
    mets = [m for m in ("comp_tv", "within_mean_rmse", "ed", "dmean_rmse_gene") if m in set(pairs.metric)]
    eps = sorted(pairs.episode.unique(), reverse=True)
    fig, axs = plt.subplots(len(eps), len(mets), figsize=(3.0 * len(mets), 2.4 * len(eps)), squeeze=False)
    for i, ep in enumerate(eps):
        for j, met in enumerate(mets):
            ax = axs[i, j]
            s = pairs[(pairs.episode == ep) & (pairs.metric == met)].reset_index(drop=True)
            y = np.arange(len(s))[::-1]
            ax.errorbar(s["diff"], y, xerr=[s["diff"] - s.lo, s.hi - s["diff"]], fmt="o", ms=4, color=INK,
                        ecolor=MUTED, elinewidth=1.5, capsize=0)
            ax.axvline(0, color=MUTED, lw=1, ls="--")
            ax.set_yticks(y)
            ax.set_yticklabels(s.method, fontsize=8)
            ax.set_title(f"{ep}: {met}", color=INK, fontsize=9)
            _style(ax)
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_budget(bud: pd.DataFrame, path):
    """Fig 4: endpoint cell budget vs error (development drugs), per method."""
    mets = ["comp_tv", "within_mean_rmse", "ed"]
    fig, axs = plt.subplots(1, len(mets), figsize=(10, 3.4))
    styles = {"mixture": ("#2a78d6", "o"), "state_moment": ("#eb6834", "s"), "nearest_low": ("#1baf7a", "^")}
    for ax, met in zip(axs, mets):
        for i, (m, s) in enumerate(bud.groupby("method")):
            col, mk = styles.get(m, (MUTED, "d"))
            s = s.sort_values("budget")
            x = s.budget * (1.06 ** (i - 1))                     # small horizontal offset per method
            ax.errorbar(x, s[met], yerr=[s[met] - s[met + "_lo"], s[met + "_hi"] - s[met]], color=col,
                        marker=mk, ms=5, lw=2, label=m, capsize=0, elinewidth=1, alpha=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 1024])
        ax.set_xticklabels(["32", "64", "all"])
        ax.set_xlabel("endpoint cells per well (cap)", color=INK, fontsize=9)
        ax.set_title(met, color=INK, fontsize=9)
        _style(ax)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
