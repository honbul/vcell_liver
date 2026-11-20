"""Figures of the R7 follow-up report. Reads only the small result tables (no expression array, no checkpoint).

fig1  annotation coverage by K group (label-free)
fig2  paired differences of the real-target arm against its controls (pooled source out-of-fold and K groups)
fig3  size of the GR correction against the error difference (label-free tertiles)
fig4  K-group-wise error difference and how much correction each arm applies ("usage")
figB  branch contrasts (only with --branch <tag>)
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import MAN, pman, prep

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"M": "#2a78d6", "GS": "#eb6834", "G0": "#1baf7a", "C": "#eda100", "GR": "#6250d6", "TDS": "#eb6834", "TD": "#6250d6"}
KCOL = {"K0": "#c3c2b7", "K1": "#2a78d6", "K2": "#1baf7a", "K3": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def tab(*p):
    f = pman(*p)
    return pd.read_csv(f if os.path.exists(f) else os.path.join(MAN, *p), sep="\t")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(prep("figures", name), dpi=160)
    plt.close(fig)
    print("wrote", name)


def fig1():
    k = tab("k_group_counts.tsv")
    k = k[k.split.isin(["train", "validation", "test"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, sp in enumerate(["train", "validation", "test"]):
        d = k[k.split == sp].set_index("k_group").n_drug_groups
        left = 0.0; tot = d.sum()
        for g in ["K0", "K1", "K2", "K3"]:
            w = d[g] / tot
            ax.barh(i, w, left=left, color=KCOL[g], edgecolor=SURF, linewidth=2, height=0.6, label=g if i == 0 else None)
            ax.text(left + w / 2, i, f"{int(d[g])}\n({w:.0%})", ha="center", va="center", fontsize=7.5, color=INK if g == "K0" else "white")
            left += w
    ax.set_yticks(range(3)); ax.set_yticklabels(["train", "validation", "test"]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("share of drug groups")
    ax.set_title("Drug groups by pre-specified target reach group (no response data used)", loc="left", fontsize=9)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              labels=["K0 no annotation", "K1 landmark target", "K2 reaches landmark <=2 hops", "K3 cannot reach"])
    save(fig, "fig1_annotation_coverage_by_k_group.png")


def forest(ax, d, pairs, groups, title):
    y = 0; ticks = []
    for g in groups:
        for x, b in pairs:
            r = d[(d.k_group == g) & (d.candidate == x) & (d.comparator == b)]
            if not len(r):
                continue
            r = r.iloc[0]
            lo, hi = (r.lo_simultaneous, r.hi_simultaneous) if ("lo_simultaneous" in r and r.core == 1 and not np.isnan(r.lo_simultaneous)) else (r.lo95, r.hi95)
            c = COL.get(b, INK2)
            ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
            ax.plot([r["diff"]], [y], "o", color=c, ms=6, markeredgecolor=SURF, markeredgewidth=1.5)
            ticks.append((y, f"{g}: {x} - {b}")); y += 1
        y += 0.6
    ax.axvline(0, color=INK2, lw=1); ax.axvline(-0.001, color=INK2, lw=1, ls=":")
    ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([s for _, s in ticks]); ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("delta-RMSE difference (candidate - comparator); negative = candidate better")
    ax.set_title(title, loc="left", fontsize=9)


def fig2():
    c = tab("capacity_attribution_contrasts.tsv"); c = c[c.scope == "pooled"]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    forest(ax, c, [("GR", "M"), ("GR", "GS"), ("GR", "G0"), ("GR", "C")], ["all", "annotated", "K1", "K2", "K3"],
           "Real-target graph arm (GR) against its controls, pooled source out-of-fold\n"
           "(line: paired drug-group bootstrap interval, 97.5 % for the two core contrasts on M and GS, else 95 %; dotted: -0.001)")
    save(fig, "fig2_real_vs_controls_paired_differences.png")


def fig3():
    s = tab("correction_size_vs_error.tsv")
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    order = ["small", "middle", "large"]; comps = [b for b in ["C", "M", "GS", "G0"] if (s.comparator == b).any()]
    for j, b in enumerate(comps):
        d = s[s.comparator == b].set_index("gr_correction_tertile").loc[order]
        x = np.arange(3) + (j - (len(comps) - 1) / 2) * 0.17
        ax.vlines(x, d.lo95, d.hi95, color=COL[b], lw=2)
        ax.plot(x, d["diff"], "o", color=COL[b], ms=6, markeredgecolor=SURF, markeredgewidth=1.5, label=f"GR - {b}")
    d0 = s[s.comparator == comps[0]].set_index("gr_correction_tertile").loc[order]
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{o}\n[{a:.3f}, {b_:.3f}]" for o, a, b_ in zip(order, d0.corr_rms_min, d0.corr_rms_max)])
    ax.axhline(0, color=INK2, lw=1); ax.axhline(-0.001, color=INK2, lw=1, ls=":")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("tertile of the per-unit RMS of the GR correction (annotated source units)"); ax.set_ylabel("delta-RMSE difference")
    ax.set_title("Size of the GR correction against the error difference (95 % intervals)", loc="left", fontsize=9)
    ax.legend(frameon=False, ncol=len(comps), loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "fig3_correction_size_vs_error.png")


def fig4():
    c = tab("capacity_attribution_contrasts.tsv"); c = c[c.scope == "pooled"]
    a = tab("capacity_attribution.tsv"); a = a[a.scope == "pooled"]
    groups = ["K1", "K2", "K3"]; arms = [x for x in ["M", "G0", "GS", "GR"] if (a.arm == x).any()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for j, arm in enumerate(arms):
        d = c[(c.candidate == arm) & (c.comparator == "C")].set_index("k_group").reindex(groups)
        x = np.arange(3) + (j - (len(arms) - 1) / 2) * 0.17
        ax1.vlines(x, d.lo95, d.hi95, color=COL[arm], lw=2)
        ax1.plot(x, d["diff"], "o", color=COL[arm], ms=6, markeredgecolor=SURF, markeredgewidth=1.5, label=arm)
        u = a[a.arm == arm].set_index("k_group").reindex(groups).corr_rms_mean_unit
        ax2.bar(x, u, width=0.15, color=COL[arm], label=arm)
    ax1.axhline(0, color=INK2, lw=1); ax1.axhline(-0.001, color=INK2, lw=1, ls=":")
    for ax, yl, tt in [(ax1, "delta-RMSE difference to C (arm - C)", "Error difference to the chemistry backbone"),
                       (ax2, "mean per-unit RMS of the correction", "How much correction each arm applies")]:
        ax.set_xticks(range(3)); ax.set_xticklabels(groups); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_ylabel(yl); ax.set_title(tt, loc="left", fontsize=9)
    ax1.legend(frameon=False, ncol=len(arms), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "fig4_k_group_improvement_and_usage.png")


def figb(tag, pairs):
    c = tab(tag, "capacity_attribution_contrasts.tsv"); c = c[c.scope == "pooled"]
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.3 * len(pairs) * 5))
    forest(ax, c, pairs, ["all", "annotated", "K1", "K2", "K3"], f"Branch contrasts ({tag}), pooled source out-of-fold, 95 % intervals; dotted: -0.001")
    save(fig, f"figB_{tag}_contrasts.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="")
    ap.add_argument("--pairs", nargs="*", default=[])
    a = ap.parse_args()
    fig1(); fig2(); fig3(); fig4()
    if a.branch:
        figb(a.branch, [tuple(p.split(":")) for p in a.pairs])


if __name__ == "__main__":
    main()
