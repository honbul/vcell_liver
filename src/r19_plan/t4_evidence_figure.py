"""Render the manuscript's temporal and cross-assay comparison from checked tables."""

import argparse
import csv
from pathlib import Path

import matplotlib

from src.r11_chemical_probe.common import read_json, sha256
from src.r19_plan.manuscript_review import score_rows

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(root, name):
    receipt = read_json(root/"receipt.json")
    path = root/name
    if sha256(path) != receipt["files"][name]:
        raise ValueError(f"Frozen figure input changed: {path}")
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def temporal(ax, data):
    choices = (
        ("t4_mean_shift", "parent", "purged_operator", "purged_mean_scalar", "Centered scalar, n=132"),
        ("t4_mean_shift", "parent", "purged_operator", "purged_diagonal", "Diagonal ridge, n=132"),
        ("t4_mean_shift", "parent", "purged_operator", "purged_scalar", "Origin scalar, n=132"),
        ("t4", "ALL", "operator6", "identity6", "Direct 6 h, n=132"),
        ("t4", "ALL", "operator6", "direct24", "Direct 24 h replicate, n=132"),
        ("t4_repeat/HEPG2", "ALL", "operator6", "direct24_scalar", "Scaled 24 h replicate, n=132"),
        ("t4_attribution", "ALL", "operator6", "mean24", "Fit mean 24 h, n=132"),
        ("t4_attribution", "same_dose", "operator6", "same_dose_swap6", "Wrong-parent 6 h, n=89"),
    )
    receipts = {study: table(data/study/"score", "paired.tsv") for study in {choice[0] for choice in choices}}
    positions = range(len(choices)-1, -1, -1)
    ax.axvline(0, lw=1, color="#8C929A", ls="--", zorder=0)
    for y, (study, scope, arm, baseline, _) in zip(positions, choices, strict=True):
        rows = [r for r in receipts[study] if (r.get("scope", "ALL"), r["metric"], r["arm"], r["reference"])
                == (scope, "rmse", arm, baseline)]
        if len(rows) != 1:
            raise ValueError(f"The figure requires one paired effect for {study} {scope} {baseline}")
        mean, lo, hi = (100*float(rows[0][k]) for k in ("relative_gain", "relative_low99", "relative_high99"))
        ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt="o", ms=5, lw=1.4, capsize=2.5,
                    color="#1F597D" if study=="t4_mean_shift" else "#5A596B", zorder=2)
    ax.set_yticks(positions, [choice[-1] for choice in choices])
    ax.set_ylim(-.55, len(choices)-.45)
    ax.set_xlim(-9, 22)
    ax.set_xlabel("Parent-macro RMSE reduction (%)")
    ax.set_title("a  Temporal prediction in HEPG2 L1000", loc="left", weight="bold", pad=12)


def cross_assay(ax, data):
    report = table(data/"t4_heparg"/"score", "cosine.tsv")
    model = (("identity6", "Direct 6 h"), ("operator6", "6 h → 24 h model"), ("direct24", "Direct 24 h"))
    studies = (("GSE284321", "5 parents", (5.3,4.3,3.3)), ("GSE296254", "3 parents", (1.4,.4,-.6)))
    assays = (("L1000_24h", "HEPG2 L1000", "o", "#1F597D", .11),
              ("HepaRG_24h", "HepaRG TempO-Seq", "s", "#A65518", -.11))
    ax.axvline(0,lw=1,color="#8C929A",ls="--",zorder=0)
    levels, labels = [], []
    for index, (study, n, ys) in enumerate(studies):
        ax.text(-.127,ys[0]+.65,f"{study} · {n}",weight="bold",fontsize=9)
        for (arm, title), y in zip(model, ys, strict=True):
            levels.append(y)
            labels.append(title)
            for assay, name, marker, color, offset in assays:
                exact = [r for r in report if (r["dataset"],r["score"],r["arm"]) == (study,assay,arm)]
                if len(exact) != 1:
                    raise ValueError(f"The figure requires one cosine mean per arm and assay for {study}")
                mean = float(exact[0]["mean_cosine"])
                ax.scatter(mean, y+offset, color=color, marker=marker, s=23,
                           label=name if index==0 and arm==model[0][0] else None)
    ax.set_yticks(levels, labels)
    ax.set_xlim(-.135,.47)
    ax.set_xticks([-.1,0,.1,.2,.3,.4])
    ax.set_ylim(-1.25,6.45)
    ax.set_xlabel("Parent-macro cosine")
    ax.set_title("b  Matched cross-assay transfer",loc="left",weight="bold",pad=12)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,-.38),ncols=2,frameon=False,fontsize=9,
              columnspacing=1.1,handletextpad=.35)


def render(data, output):
    if output.suffix != ".pdf":
        raise ValueError("The manuscript figure is rendered as a vector PDF")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, (left, right, bottom) = plt.subplots(1, 3, figsize=(14.1, 4.8),
                                               gridspec_kw={"width_ratios": [1.18, 1, 1]})
    fig.subplots_adjust(left=.17, right=.985, top=.88, bottom=.24, wspace=.83)
    temporal(left,data)
    cross_assay(right,data)
    r18 = [row for row in score_rows(data) if row["joint_gate"]]
    bottom.axvline(0, lw=1, color="#8C929A", ls="--")
    for y, row in enumerate(r18):
        point, lo, hi = (row[key] for key in ("delta_r2_points", "r2_low99_points", "r2_high99_points"))
        bottom.errorbar(point, y, xerr=[[point-lo], [hi-point]], fmt="o", color="#1F597D", capsize=2.5)
    bottom.set(yticks=range(len(r18)),
               yticklabels=[row["evaluation"][-1] + " · " + row["joint_gate"] for row in r18],
               ylim=(1.5, -.5), xlabel="Pooled R² gain vs R15 (points)")
    bottom.set_title("c  R18 joint criterion", loc="left", weight="bold", pad=12)
    for ax in (left,right,bottom):
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#8C929A")
        ax.tick_params(axis="y",length=0,pad=8)
        ax.tick_params(axis="x",length=3,color="#8C929A")
        ax.grid(axis="x",color="#E8E9EB",lw=.6)
        ax.set_axisbelow(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,metadata={"Title":"Temporal and cross-assay HEPG2 transfer",
                                "CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    config = parser.parse_args()
    render(config.data,config.output)
