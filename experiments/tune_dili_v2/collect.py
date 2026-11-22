"""Collect small held-fold metric tables and verify the paired DILI release receipts."""

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


def read(path):
    return json.loads(path.read_text())


def run(args):
    args.out.mkdir(exist_ok=False,parents=True)
    metrics, selections, calibration, errors, cases = [],[],[],[],[]
    sources = ("round1-03-cpu","round1-03-cat","round1-04-kernels","round2-01","round3-02","round4-01")
    for folder in sources:
        for split in ("murcko","butina"):
            for fold in range(5):
                p=args.runs/folder/f"{split}-{fold}"/"receipt.json"
                observed = read(p)
                if "outer_metrics" in observed:
                    records = observed["outer_metrics"]
                else:
                    key = "round3_results" if folder=="round3-02" else "evaluations"
                    field = "outer" if folder=="round3-02" else "metrics"
                    records = {arm:value[field] for arm,value in observed[key].items()}
                    for arm,value in observed[key].items():
                        trials = value.get("trials",value.get("calibrator_grid"))
                        calibration.extend({"stage":folder,"split":split,"fold":fold,"model":arm,
                                            "calibrator":choice,**point} for choice,point in trials.items())
                metrics.extend({"stage":folder,"split":split,"fold":fold,"method":name,**data}
                               for name,data in records.items())
                if folder=="round3-02":
                    sample=pd.read_csv(p.parent/"error_roster.tsv",sep="\t")
                    cases.append(sample.assign(split=split,fold=fold))
                    errors.extend({"split":split,"fold":fold,**r} for r in observed["errors"])
                if folder.startswith("round1"):
                    selections.extend({"stage":folder,"split":split,"fold":fold,"method":method,**cfg}
                                      for method,cfg in observed["selected"].items())
    measure = pd.DataFrame(metrics)
    measure.to_csv(args.out/"cv_fold_metrics.tsv",sep="\t",index=False)
    measure.drop(columns="fold").groupby(["stage","split","method"]).mean(numeric_only=True).to_csv(
        args.out/"cv_macro.tsv",sep="\t")
    pd.DataFrame(selections).to_csv(args.out/"chemistry_selected.tsv",sep="\t",index=False)
    pd.DataFrame(calibration).to_csv(args.out/"calibrator_cv.tsv",sep="\t",index=False)
    pd.DataFrame(errors).to_csv(args.out/"subgroup_error.tsv",sep="\t",index=False)
    if args.fda_registry:
        roster = pd.read_csv(args.fda_registry,sep="\t",usecols=["fda_id","name"],dtype=str).drop_duplicates("fda_id")
        types = pd.concat(cases,ignore_index=True)
        types["error_type"] = types.label.map({0:"false_positive",1:"false_negative"})
        worst = (types.sort_values("error",ascending=False).groupby(["split","error_type"]).head(8)
                 .merge(roster,on="fda_id",how="left",validate="many_to_one"))
        worst.drop(columns=["error","label","probability"]).to_csv(args.out/"largest_oof_errors.tsv",sep="\t",index=False)
    score = args.release/"score.json"
    for name,path in {
        "final_score.json":score,"forecast_freeze.json":args.release/"freeze.json",
        "historical_fit_manifest.json":args.release/"models/release-0/release.json",
        "internal_policy.json":args.runs/"round4-01/policy.json","offline_receipt.json":args.release/"offline-smoke.json",
        "public_targets_receipt.json":args.input/"assays-01/receipt.json",
        "epa_target_receipt.json":args.input/"assays-04/receipt.json",
        "mechanism_fit_replication.json":args.runs/"round2-01/murcko-0/receipt.json",
    }.items():
        shutil.copy2(path,args.out/name)
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(10,3.45),sharey=True,layout="constrained")
    for axis,(kind, record) in zip(axes,read(score)["splits"].items()):
        raw = pd.DataFrame(record["risk_coverage_v2"]).sort_values("coverage")
        axis.plot(raw.coverage,raw.auroc,marker="o",color="#184e77",label="v2")
        axis.axhline(record["v1"]["auroc"],linestyle="--",color="#8d342d",label="v1 all compounds")
        axis.set_title(kind)
        axis.set_xlabel("Coverage")
        axis.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("AUROC")
    axes[-1].legend(frameon=False,fontsize=8)
    fig.savefig(args.out/"risk_coverage.pdf")
    fig.savefig(args.out/"risk_coverage.png",dpi=165)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("runs","release","input","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fda-registry",type=Path)
    run(p.parse_args())
