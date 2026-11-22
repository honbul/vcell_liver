"""Write compact V0+ v2 tables from registered development and one-shot score artifacts."""

import argparse
import shutil
from pathlib import Path

import json
import numpy as np
import pandas as pd

from vc_build.v0plus_cv import training_inputs
from vc_build.v0plus_score import compound_strata

AXES = ("spearman","pearson","macro_ndcg_at_5","macro_top5_overlap","macro_spearman")


def read(path):
    return json.loads(path.read_text())


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    package = args.result/"package"
    hist = read(package/"release.json")
    stages = {
        "round1":("round1-02","evaluation"), "round2":("round2-02","outer"),
        "round3":("round3-01","outer"), "round4":("round4-01","outer"),
    }
    metrics, hcc, cvmeta = [],[],[]
    for stage,(folder,field) in stages.items():
        for f in range(3):
            run = args.runs/folder/f"fold{f}"
            result = read(run/"receipt.json")
            metrics.extend(dict(stage=stage,fold=f, **d) for d in result[field])
            if stage=="round3":
                cvmeta.append({"fold":f,"verified_fm_sha256":result["verified_fm_sha256"],
                               "coverage":result["fm_train_coverage"],"hcc_cell_count":result["liver_held_cells"],
                               "inner_selected":result["selected"]})
                hcc.extend(dict(fold=f,**d) for d in result["liver_held"])
    result = pd.DataFrame(metrics)
    result.to_csv(args.out/"cv_by_source_fold.tsv",sep="\t",index=False)
    result.groupby(["stage","method"])[list(AXES)].mean().to_csv(args.out/"cv_macro.tsv",sep="\t")
    if hcc:
        pd.DataFrame(hcc).to_csv(args.out/"neural_held_liver.tsv",sep="\t",index=False)
    (args.out/"neural_fm_receipt.json").write_text(json.dumps(cvmeta,indent=2)+"\n")
    train,_,_ = training_inputs(args.bundle)
    register = train[["compound_id","source_compound_label","drug_target_source_labels"]].drop_duplicates()
    classes = compound_strata(register)
    drugs = {key for group in ("hcc_tki","egfr","fgfr","mtor","hdac") for key in classes[group]}
    registered = train.loc[train.compound_id.isin(drugs)]
    coverage = registered.groupby(["compound_id","source_id"]).agg(fit_measurements=("cell_id","size"),
        fit_cells=("cell_id","nunique"), dose_min_uM=("dose_min_uM","min"), dose_max_uM=("dose_max_uM","max")).reset_index()
    fields = ["source_id","compound_id","n","response_sd","drug_spearman","drug_pearson"]
    out = pd.concat([pd.read_csv(args.runs/"round1-02"/f"fold{f}"/"diagnostics.tsv",sep="\t",usecols=fields)
                     .assign(fold=f) for f in range(3)])
    lineage = (coverage.merge(out.groupby(["compound_id","source_id"],as_index=False)
                            .mean(numeric_only=True),how="left",validate="one_to_one")
               .merge(register,on="compound_id",validate="many_to_one"))
    lineage["strata"] = lineage.compound_id.map(lambda d:",".join(k for k,i in classes.items() if d in i))
    lineage.to_csv(args.out/"target_failure_diagnostics.tsv",sep="\t",index=False)
    for name,path in {
        "release_manifest.json":package/"release.json", "limore_score.json":args.result/"limore_score.json",
        "limore_forecast_freeze.json":args.result/"limore.parquet.freeze.json", "audit_score.json":args.result/"audits/audit_score.json",
        "offline_receipt.json":args.result/"offline.json", "cmap_receipt.json":args.cmap/"receipt.json",
        "tcga_panels.tsv":args.result/"audits/tcga_panels.tsv", "tcga_driver_subtype.tsv":args.result/"audits/tcga_driver_subtype.tsv",
        "tcga_purity.tsv":args.result/"audits/tcga_purity.tsv",
    }.items():
        shutil.copy2(path,args.out/name)
    import matplotlib.pyplot as plt
    means = result.groupby(["stage","method"])[list(AXES)].mean()
    choices = ["adaptive", "adaptive_shrunk", "two_tower_cmap", hist["default"]]
    rows = np.asarray([means.loc[(stage, choice)].to_numpy() for stage,choice in zip(stages,choices)])
    fig, axes = plt.subplots(1,2,figsize=(9.3,3.7),layout="constrained")
    stages_label = ["R1","R2","R3","R4"]
    for i,label in enumerate(("Within-drug Spearman","Within-drug Pearson",
                              "Within-cell NDCG@5","Within-cell Precision@5","Within-cell Spearman")):
        k = 0 if i<2 else 1
        axes[k].plot(stages_label,rows[:,i],marker="o",label=label)
    for axis in axes:
        axis.grid(axis="y",alpha=.25)
        axis.legend(frameon=False,loc="best",fontsize=8)
    fig.savefig(args.out/"internal_axes.pdf")
    fig.savefig(args.out/"internal_axes.png",dpi=165)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("runs","result","cmap","bundle","out"):
        p.add_argument("--"+key,type=Path,required=True)
    run(p.parse_args())
