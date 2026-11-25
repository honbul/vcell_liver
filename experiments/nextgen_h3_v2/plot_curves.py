"""Plot measured nested learning and saturation for the prespecified scaffold partition."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS={"gbdt":"LightGBM","film":"FiLM-ResNet","attention":"Cross-Attention","gnn":"Molecular GNN",
        "hybrid":"H2 + nonlinear hybrid"}


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle,delimiter="\t"))


def plot_learning(root):
    learning=read(root/"H3V2_FAMILY_LEARNING_CURVES.tsv")
    fig, axes = plt.subplots(2,3,figsize=(12,7),layout="constrained")
    panels = []
    for family, label in LABELS.items():
        part=[row for row in learning if row["split"]=="drug-0" and row["family"]==family and row["frozen"]=="True"]
        branches = list(dict.fromkeys(r["branch"] for r in part if r.get("branch") and r["branch"] != "selected"))
        for branch in branches or [""]:
            title = f"{label} · {branch}" if len(branches) > 1 else label
            panels.append((title, branch, part))
    for (label, branch, part), ax in zip(panels, axes.ravel()):
        train,valid=("train_huber","val_huber") if not branch else ("train_objective","validation_objective")
        for color,axis in (("#2a658a",train),("#bd632c",valid)):
            for fold in range(3):
                rows=[r for r in part if int(r["inner"])==fold and r.get(axis) and (not branch or r["branch"]==branch)]
                rows.sort(key=lambda r:int(r["iteration"]))
                ax.plot([int(x["iteration"]) for x in rows],[float(x[axis]) for x in rows],
                        color=color,alpha=.85-fold*.20,label=("Train" if axis==train else "Inner validation") if fold==0 else None)
        ax.set_title(label,loc="left",fontweight="bold")
        ax.set_xlabel("Epoch" if not branch else "Tree")
        ax.set_ylabel("Huber loss" if not branch else next(r["metric"] for r in part if r.get("metric") and r["branch"] == branch))
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    fig.suptitle("Drug-Cold fold 0 · Selected settings on three inner splits", fontsize=15,fontweight="bold")
    fig.savefig(root/"H3V2_FAMILY_LEARNING_CURVES.png",dpi=170)
    plt.close(fig)


def capacity_slice(rows):
    winner, = [row for row in rows if row["frozen"] == "True"]
    chosen = json.loads(winner["configuration"])
    axis = "depth" if chosen["backend"] == "gbdt" else "width"
    selected_utility = float(winner["utility"])
    ignored = {axis}
    if axis == "width" and chosen["arch"] == "attention":
        ignored.update(("batch", "eval_batch"))
    lock = {name: value for name, value in chosen.items() if name not in ignored}
    matched = []
    measured = {}
    for row in rows:
        config = json.loads(row["configuration"])
        if (config["backend"], config.get("arch")) != (chosen["backend"], chosen.get("arch")):
            continue
        capacity = config[axis]
        difference = float(row["utility"]) - selected_utility
        measured[capacity] = max(measured.get(capacity, difference), difference)
        if {name: value for name, value in config.items() if name not in ignored} == lock:
            matched.append((capacity, difference))
    return axis, chosen[axis], sorted(matched), sorted(measured.items())


def plot_saturation(root):
    grid = read(root / "H3V2_GRID_SATURATION.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for (family, label), ax in zip(LABELS.items(), axes.ravel()):
        rows = [row for row in grid if row["split"] == "drug-0" and row["family"] == family]
        axis, selected, matched, measured = capacity_slice(rows)
        x, delta = zip(*measured)
        best, = ax.plot(x, delta, "o--", color="#777f88", label="Best measured score at each capacity")
        matched_x, matched_delta = zip(*matched)
        fixed, = ax.plot(matched_x, matched_delta, "o-", color="#2a658a", label="Matched grid slice")
        winner = ax.scatter(selected, 0, color="#bd632c", s=55, zorder=3, label="Frozen operating point")
        ax.axhline(0, color="#a5aab0", lw=.75)
        if axis == "width":
            ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(set(x)), labels=sorted(set(x)))
        ax.set_xlabel("Tree depth" if axis == "depth" else "Hidden width")
        ax.set_ylabel("Inner utility minus selected")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=.2)
    legend_axis = axes.ravel()[-1]
    legend_axis.set_axis_off()
    legend_axis.legend(handles=[best, fixed, winner], loc="center left", frameon=False)
    fig.suptitle("Drug-Cold fold 0 · Observed inner utility by capacity", fontsize=15, fontweight="bold")
    fig.savefig(root / "H3V2_GRID_SATURATION.png", dpi=170)
    plt.close(fig)


if __name__=="__main__":
    root = Path(sys.argv[1])
    plot_learning(root)
    plot_saturation(root)
