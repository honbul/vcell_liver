"""Plot actual nested learning curves for one prespecified Drug-Cold scaffold partition."""
import csv
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


def main(root):
    learning=read(root/"H3V2_FAMILY_LEARNING_CURVES.tsv")
    fig, axes = plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for (family,label),ax in zip(LABELS.items(),axes.ravel()):
        part=[row for row in learning if row["split"]=="drug-0" and row["family"]==family and row["frozen"]=="True"]
        branch=next((r["branch"] for r in part if r.get("branch") and r["branch"]!="selected"),"")
        train,valid=("train_huber","val_huber") if not branch else ("train_objective","validation_objective")
        for color,axis in (("#2a658a",train),("#bd632c",valid)):
            for fold in range(3):
                rows=[r for r in part if int(r["inner"])==fold and r.get(axis) and (not branch or r["branch"]==branch)]
                rows.sort(key=lambda r:int(r["iteration"]))
                ax.plot([int(x["iteration"]) for x in rows],[float(x[axis]) for x in rows],
                        color=color,alpha=.85-fold*.20,label=("Train" if axis==train else "Inner validation") if fold==0 else None)
        ax.set_title(label,loc="left",fontweight="bold")
        ax.set_xlabel("Epoch" if not branch else "Tree")
        ax.set_ylabel("Huber loss" if not branch else next(r["metric"] for r in part if r.get("metric")))
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    axes.ravel()[-1].set_axis_off()
    fig.suptitle("Drug-Cold fold 0 · Selected settings on three inner splits", fontsize=15,fontweight="bold")
    fig.savefig(root/"H3V2_FAMILY_LEARNING_CURVES.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main(Path(sys.argv[1]))
