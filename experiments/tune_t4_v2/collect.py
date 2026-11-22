"""Collect published aggregate evidence without copying patient or well predictions."""

import argparse
from collections import Counter
from pathlib import Path
import shutil

import json
import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def main(run, release, out, pan):
    out.mkdir(parents=True, exist_ok=False)
    for name, path in {
        "release_freeze.json": release/"freeze.json", "final_score_receipt.json": release/"score/receipt.json",
        "pretrain_receipt.json": run/"pretrain-01/receipt.json", "source_pairs_receipt.json": pan/"receipt.json",
        "offline_receipt.json": release/"offline-02/receipt.json",
    }.items():
        shutil.copy2(path, out/name)
    by_cell, transfer, selections = [], [], []
    for cell in ("HEPG2", "PHH"):
        shutil.copy2(release/"score"/f"{cell}.tsv", out/f"paired_{cell}.tsv")
        order, comparing = {}, {}
        for fold in range(3):
            for role in "ABC":
                group = f"f{fold}-{role}"
                a, b, c = (read(run/tree/cell/group/"receipt.json") for tree in
                           ("round1-01","round2-01","rounds34-02"))
                ar, br, cr = [{m["slice"]:m for m in receipt["slices"]} for receipt in (a,b,c)]
                for key, report in cr.items():
                    v1 = ar[key]["rank16_rmse"]
                    r1 = ar[key]["selected_inner_rmse"]
                    r2 = br[key]["metrics"][br[key]["champion"]]
                    r3 = report["inner_rmse"][report["round3_pan_complete"]]
                    r4 = report["inner_rmse"][report["round4_champion"]]
                    gains = dict(cell=cell, fold=fold, role=role, slice=key, rank16=v1, round1=r1, round2=r2,
                                 round3=r3, round4=r4, round2_model=br[key]["champion"],
                                 round3_model=report["round3_pan_complete"], round4_model=report["round4_champion"])
                    selections.append(gains)
                    for method, metric in report["inner_rmse"].items():
                        if method.startswith(("pan_", "linear_", "scalar_")) and "+" not in method:
                            transfer.append(dict(cell=cell, method=method, rmse=metric))
                for method, v in b["paired_fixed_half_scale"].items():
                    comparing.setdefault(method, []).append(v["rmse_gain_percent"])
        values = pd.DataFrame(selections)
        metrics = values.loc[values.cell.eq(cell)]
        by_cell.append(dict(cell=cell, slices=len(metrics),
                            **{k:metrics[k].mean() for k in ("rank16","round1","round2","round3","round4")}))
        for method, items in comparing.items():
            order[method] = float(np.mean(items))
        (out/f"nonlinear_{cell}.json").write_text(json.dumps(order,indent=2)+"\n")
    pd.DataFrame(selections).to_csv(out/"internal_by_slice.tsv",sep="\t",index=False)
    pd.DataFrame(by_cell).to_csv(out/"internal_summary.tsv",sep="\t",index=False)
    pd.DataFrame(transfer).groupby(["cell","method"]).rmse.agg(["mean","size"]).to_csv(
        out/"internal_transfer.tsv",sep="\t")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,3.9),layout="constrained")
    x = ["Rank 16","R1","R2","R3","R4"]
    for a, cell_summary in zip(axes,by_cell):
        y = [cell_summary[k] for k in ("rank16","round1","round2","round3","round4")]
        a.plot(x,y,marker="o",color="#23615a")
        a.set_ylabel("Mean inner-fold RMSE")
        a.set_title(cell_summary["cell"])
        a.grid(axis="y",alpha=.25)
        a.margins(y=.3)
    fig.savefig(out/"internal_trajectory.pdf")
    fig.savefig(out/"internal_trajectory.png",dpi=170)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","release","out","pan"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    main(a.run,a.release,a.out,a.pan)
