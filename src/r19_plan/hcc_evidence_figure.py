"""Render the adult HCC comparisons from verified paired-score summaries."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from src.r11_chemical_probe.common import read_json, sha256


COLORS = ("#156C80", "#B35421")
CELLS = ("JHH5", "JHH7")


def read_scores(root, study, name="paired.tsv"):
    folder = root/study/"score"
    file = folder/name
    if sha256(file) != read_json(folder/"receipt.json")["files"][name]:
        raise ValueError(f"The plotted score is different from its receipt: {file}")
    with file.open(newline="") as stream:
        return [row for row in csv.DictReader(stream, delimiter="\t")
                if row["scope"] in ("absolute", "all") and row["metric"] == "rmse"]


def gain(rows, **key_values):
    selected = [row for row in rows if all(row[key] == value for key, value in key_values.items())]
    if len(selected) != 1:
        raise ValueError(f"Cannot uniquely identify a plotted HCC comparison: {key_values}")
    row = selected[0]
    return 100*np.asarray([row[key] for key in ("relative_gain", "relative_low99", "relative_high99")], dtype=float)


def forest(ax, data, labels, cell_labels):
    y = np.arange(len(labels))
    for column, (series, color, name) in enumerate(zip(data, COLORS, cell_labels, strict=True)):
        points, low, high = np.asarray(series).T
        ax.errorbar(points, y+(.15 if column else -.15), xerr=(points-low, high-points), c=color, label=name,
            ecolor=color, marker="s" if column else "o", linestyle="none", markersize=5, elinewidth=1.35, capsize=2.2)
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels)-.48,-.48)
    ax.axvline(0, color="#7C848A", lw=.9, ls=":")
    ax.grid(axis="x", color="#DDE3E8", lw=.65)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_xlabel("Parent-macro RMSE reduction (%)")


def panels(ax, rows):
    for cell, color in zip(CELLS, COLORS, strict=True):
        for method, linestyle, marker in (("mean", "-", "o"), ("shrink", "--", "s")):
            count = (2,4,8,16)
            point, low, high = np.asarray([gain(rows,cell=cell,arm=f"{method}_{n}",relative_to="B0") for n in count]).T
            label = ("JHH-5" if cell=="JHH5" else "JHH-7")+" "+("mean" if method=="mean" else "shrinkage")
            ax.plot(count,point,c=color,ls=linestyle,marker=marker,ms=4,label=label,lw=1.25)
            ax.fill_between(count,low,high,color=color,alpha=.08)
    ax.set_xscale("log", base=2)
    ax.set_xticks((2,4,8,16),("2","4","8","16"))
    ax.set_xlabel("Number of measured donor cell lines")
    ax.set_ylabel("RMSE reduction vs. zero (%)")
    ax.grid(axis="y",color="#DDE3E8",lw=.65)
    ax.axhline(0,color="#7C848A",ls=":",lw=.9)
    ax.legend(loc="upper left",frameon=False,fontsize=8)


def render(root, output):
    jhh5 = read_scores(root,"jhh5_donor_attribution")
    jhh7 = read_scores(root,"jhh7_donor_transfer")
    counts = read_scores(root,"hcc_donor_panel_size")
    matched = read_scores(root,"hcc_donor8_state")
    common = ["No response","State adaptation","Different compound"]
    main = ("B0","State_24h_donor","other_parent_same_pool")
    known = [[gain(table,arm="donor_mean",reference=ref) for ref in main] for table in (jhh5,jhh7)]
    references = ("mean_8","scalar_8","State_8","B0")
    reduced = [[gain(matched,cell=cell,arm="shrink_8",reference=ref) for ref in references] for cell in CELLS]
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"axes.labelcolor":"#273442",
                         "text.color":"#273442","xtick.color":"#273442","ytick.color":"#273442"}):
        fig = plt.figure(figsize=(10.3,7.1),layout="constrained")
        axes = fig.subplot_mosaic([["matched","counts"],["reduced","reduced"]],height_ratios=[1.25,1])
        titles = ("a  Same-compound donor mean","b  Donor panel size","c  Matched eight-donor transfer")
        for ax,title in zip(axes.values(),titles,strict=True):
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["bottom","left"]].set_color("#8C929A")
            ax.set_axisbelow(True)
            ax.set_title(title,loc="left",fontweight="bold",pad=12)
        forest(axes["matched"],known,common,("JHH-5 (609)","JHH-7 (99)"))
        panels(axes["counts"],counts)
        forest(axes["reduced"],reduced,("Donor mean","Fitted scalar","State adaptation","No response"),("JHH-5 (605)","JHH-7 (97)"))
        fig.savefig(output,format="pdf",metadata={"Title":"Matched nonhepatic donor transfer in adult HCC","CreationDate":None})
        plt.close(fig)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args = cli.parse_args()
    if args.output.suffix!=".pdf":
        cli.error("The manuscript figure is a vector PDF")
    render(args.data,args.output)
