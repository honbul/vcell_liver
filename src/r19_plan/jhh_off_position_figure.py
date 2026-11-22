"""Render the paired off-position adult HCC results as a vector figure."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from src.r19_plan.manuscript_review import score_rows


def render(root, output):
    cohorts = (("JHH-5", 305, "#156C80", "o", -.13), ("JHH-7", 88, "#B35421", "s", .13))
    fair = list(score_rows(root))
    comparisons = ("Adaptive vs. constant gain on donor mean", "Shrinkage vs. raw mean, both adaptive",
                   "Best public adaptation")
    metrics = {
        "rmse": ("relative_rmse_gain_pct", "rmse_low99_pct", "rmse_high99_pct"),
        "r2_pooled": ("delta_r2_points", "r2_low99_points", "r2_high99_points"),
    }
    studies = []
    for cohort, _, *_ in cohorts:
        rows = {r["comparator"]: r for r in fair if r["evaluation"] == f"{cohort}, 10 µM"}
        studies.append({metric: np.asarray([[rows[comparison][field] for field in fields]
                                            for comparison in comparisons]) for metric, fields in metrics.items()})
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42, "text.color":"#273442",
                         "axes.labelcolor":"#273442", "ytick.color":"#273442", "xtick.color":"#273442"}):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
        fig.subplots_adjust(left=.29, right=.98, top=.79, bottom=.22, wspace=.23)
        names = ("Adaptive vs. constant gain\non raw donor mean", "Shrinkage vs. raw mean\nwith matched adaptive gain",
                 "Earlier fixed-gain benchmark\nvs. best matched adaptation")
        y = np.arange(len(names))
        labels = (("rmse", "a  Parent-macro RMSE", "Relative reduction (%)"),
                  ("r2_pooled", "b  Pooled R²", "Improvement (percentage points)"))
        handles = []
        for ax, (metric, title, label) in zip(axes, labels, strict=True):
            for (cohort, n, color, marker, offset), values in zip(cohorts, studies, strict=True):
                point, lower, upper = values[metric].T
                series = ax.errorbar(point, y + offset, xerr=(point - lower, upper - point), color=color, label=f"{cohort} ({n})",
                            marker=marker, ls="none", markersize=4.5, elinewidth=1.3, capsize=2.2)
                if metric == "rmse":
                    handles.append(series)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#9AA4AB")
            ax.tick_params(axis="y", length=0)
            ax.set_title(title, loc="left", fontweight="bold", pad=9)
            ax.set_xlabel(label)
            ax.axvline(0, color="#7C848A", lw=.9, ls=":")
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#DDE3E8", lw=.65)
        axes[0].set_yticks(y, names)
        axes[0].set_ylim(len(names)-.45, -.45)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.59, .978), frameon=False, ncol=2)
        fig.text(.24, .055, "99% paired parent intervals. Both cohorts were B1-exposed. JHH-7 used one evaluation RNA plate.",
                 fontsize=8, color="#465967")
        fig.savefig(output, format="pdf", metadata={"Title":"Adult HCC off-position donor comparisons", "CreationDate":None})
        plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    render(args.data, args.output)
