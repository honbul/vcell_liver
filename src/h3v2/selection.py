"""Register each round and freeze operating points from the corresponding nested results."""
import argparse
from collections import Counter
from statistics import mean, median

import pandas as pd

from vc_build.io import publish_json, sha256
from nextgen_h2.run import read_json
from .data import ROOT, SPLITS
from .grid import FAMILIES, SEEDS, boundary_neighbors, identify, initial, nearby


def stage_options(split, family, stage):
    if stage == "r1":
        return initial(family)
    return read_json(ROOT/"locks"/f"{stage}.json")["splits"][split][family]


def receipt(stage, split, family, config, inner):
    identity=identify(config)
    path=ROOT/"fits"/stage/split/f"inner-{inner}"/family/identity
    data=read_json(path/"receipt.json")
    if (data["stage"],data["split"],data["family"],data["inner"],data["config_id"],data["seed"]) != (
        stage,split,family,inner,identity,SEEDS[0]
    ) or data["query_has_outer_labels"]:
        raise ValueError("A submission did not match the registered inner arm")
    if (sha256(path/"curves.parquet"),sha256(path/"predictions.parquet")) != (
        data["curve_sha256"],data["forecast_sha256"]
    ):
        raise ValueError("Altered learning curve or inner forecasts")
    return data | {"receipt_sha256":sha256(path/"receipt.json")}


def evaluated(split, family, stages):
    sweep=[]
    for stage in stages:
        for cfg in stage_options(split,family,stage):
            rows=[receipt(stage,split,family,cfg,inner) for inner in range(3)]
            scores={key:mean([x["score"][key] for x in rows]) for key in rows[0]["score"]}
            sweep.append({"stage":stage,"id":identify(cfg),"config":cfg,"point":scores,
                          "utility":mean([x["utility"] for x in rows]),"folds":rows,
                          "selected_budget":int(median([x["training"]["best_iteration"] for x in rows])),
                          "max_best_iteration":max(x["training"]["best_iteration"] for x in rows)})
    return sweep


def best(rows):
    return min(rows,key=lambda r:(-r["utility"],-r["point"]["ndcg_at_5"],-r["point"]["spearman"],r["id"]))


def compact(row):
    return {k:v for k,v in row.items() if k!="folds"} | {
        "receipt_sha256":[x["receipt_sha256"] for x in row["folds"]]}


def fit_gap(split, family, winner):
    gaps=[]
    if winner["config"]["backend"]!="torch":
        return {"diagnosis":"tree_grid","validation_train_relative_gap":None}
    for part,d in enumerate(winner["folds"]):
        path=ROOT/"fits"/winner["stage"]/split/f"inner-{part}"/family/winner["id"]/"curves.parquet"
        frame=pd.read_parquet(path)
        record=frame[frame.iteration.eq(d["training"]["best_iteration"])].iloc[0]
        gaps.append(float((record.val_huber-record.train_huber)/max(1e-6,record.train_huber)))
    gap=median(gaps)
    return {"diagnosis":"overfit" if gap>.15 else "capacity_or_objective",
            "validation_train_relative_gap":gap,"individual_inner_gaps":gaps}


def select(stages):
    scores={}
    for split in SPLITS:
        scores[split]={family:evaluated(split,family,stages) for family in FAMILIES}
    label=stages[-1]
    winners={split:{name:best(rows) for name,rows in results.items()} for split,results in scores.items()}
    if label=="r1":
        next_round="r2"
    elif label=="r2":
        next_round="boundary-1"
    else:
        next_round="boundary-"+str(int(label.split("-")[-1])+1)
    pending={}
    boundary_status={}
    diagnostics={}
    for split,results in scores.items():
        pending[split],boundary_status[split],diagnostics[split]={},{},{}
        top=sorted(winners[split],key=lambda f:-winners[split][f]["utility"])[:3]
        for family, variants in results.items():
            chosen=winners[split][family]
            explored=[row["config"] for row in variants]
            if label=="r1":
                diagnosis=fit_gap(split,family,chosen)
                diagnostics[split][family]=diagnosis
                candidates=nearby(chosen["config"],intensive=family in top,diagnosis=diagnosis["diagnosis"])
            else:
                candidates=boundary_neighbors(chosen["config"],explored,chosen["max_best_iteration"])
            pending[split][family]=list({identify(c):c for c in candidates
                                         if identify(c) not in {r["id"] for r in variants}}.values())
            boundary_status[split][family]=[identify(c) for c in pending[split][family]]
    has_work=any(configs for families in pending.values() for configs in families.values())
    ledger=ROOT/"locks"/f"{label}-selection.json"
    publish_json(ledger,{
        "stages":stages,"no_outer_scores_loaded":True,
        "points":{split:{f:[compact(row) for row in rows] for f,rows in res.items()} for split,res in scores.items()},
        "winners":{split:{f:compact(row) for f,row in res.items()} for split,res in winners.items()},
        "open_neighbors":boundary_status, "curve_diagnoses":diagnostics,
        "next":next_round if has_work else None,
    })
    if has_work:
        publish_json(ROOT/"locks"/f"{next_round}.json",{"splits":pending,"based_on_sha256":sha256(ledger)})
        print(next_round, sum(len(cfg) for fs in pending.values() for cfg in fs.values()),flush=True)
        return
    local={split:{"families":{family:compact(row) for family,row in ws.items()},"champion":best(ws.values())["config"]["family"]}
           for split,ws in winners.items()}
    drug=[local[tag] for tag in ("drug-0","drug-1","drug-2")]
    families=Counter([x["champion"] for x in drug])
    chosen=min(families,key=lambda f:(-families[f],-mean([x["families"][f]["utility"] for x in drug]),f))
    options=[x["families"][chosen] for x in drug]
    counts=Counter([x["id"] for x in options])
    exterior=min(options,key=lambda x:(-counts[x["id"]],-x["utility"],x["id"]))
    publish_json(ROOT/"locks/frozen.json",{
        "splits":local,"external":{"champion":chosen,"operating_point":exterior,
          "rule":"most frequent primary inner-selected family, then mean inner utility; then modal setting"},
        "selection_ledger_sha256":sha256(ledger),"seeds":list(SEEDS),"outer_labels_opened":False,
    })
    print("FROZEN",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stages",nargs="+")
    select(p.parse_args().stages)
