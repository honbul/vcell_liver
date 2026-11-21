"""Render the paired off-position adult HCC results as a vector figure."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from src.r11_chemical_probe.common import read_json, sha256


REFERENCES = {
    "scalar": "Fitted scalar",
    "eight_token_State": "Eight-token State",
    "wrong_calibrated_shrink": "Different drug",
    "B0": "No response",
}


def data(root, directory, n):
    folder = root / directory / "score"
    path = folder / "paired.tsv"
    if sha256(path) != read_json(folder / "receipt.json")["files"][path.name]:
        raise ValueError(f"Plot source differs from its score receipt: {path}")
    with path.open(newline="") as handle:
        rows = {(r["metric"], r["reference"]): r for r in csv.DictReader(handle, delimiter="\t")
                if r["scope"] == "all" and r["arm"] == "calibrated_shrink"}
    out = {}
    for name, fields in (("rmse", ("relative_gain", "relative_low99", "relative_high99")),
                         ("r2_pooled", ("difference", "low99", "high99"))):
        selected = [rows[name, reference] for reference in REFERENCES]
        if any(int(row["parents"]) != n for row in selected):
            raise ValueError("A comparison no longer uses the registered paired support")
        out[name] = 100 * np.asarray([[float(r[f]) for f in fields] for r in selected])
    return out


def render(root, output):
    cohorts = (("jhh5_off_position", "JHH-5 (305)", 305, "#156C80", "o", -.13),
               ("jhh7_off_position", "JHH-7 (88)", 88, "#B35421", "s", .13))
    studies = [data(root, item[0], item[2]) for item in cohorts]
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42, "text.color":"#273442",
                         "axes.labelcolor":"#273442", "ytick.color":"#273442", "xtick.color":"#273442"}):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
        fig.subplots_adjust(left=.175, right=.98, top=.79, bottom=.22, wspace=.23)
        y = np.arange(len(REFERENCES))
        labels = (("rmse", "a  Parent-macro RMSE", "Relative reduction (%)"),
                  ("r2_pooled", "b  Pooled R²", "Improvement (percentage points)"))
        handles = []
        for ax, (metric, title, label) in zip(axes, labels, strict=True):
            for (_, cohort, _, color, marker, offset), values in zip(cohorts, studies, strict=True):
                point, lower, upper = values[metric].T
                series = ax.errorbar(point, y + offset, xerr=(point - lower, upper - point), color=color, label=cohort,
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
        axes[0].set_yticks(y, list(REFERENCES.values()))
        axes[0].set_ylim(len(REFERENCES)-.45, -.45)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.59, .978), frameon=False, ncol=2)
        fig.text(.175, .055, "99% paired parent intervals. Both cohorts were B1-exposed. JHH-7 used one evaluation RNA plate.",
                 fontsize=8, color="#465967")
        fig.savefig(output, format="pdf", metadata={"Title":"Adult HCC off-position donor comparisons", "CreationDate":None})
        plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    render(args.data, args.output)
