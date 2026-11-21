"""Bind all R15 predictions, then score held parent outcomes in a separate job."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE

from .data import STUDIES
from .design import Bank
from .estimators import unit
from .forecast import LEGACY, PROPOSED, split


CI = (.005,.995)
METRICS = ("rmse","r2_pooled")


def load_slice(args, bank, scenario, fold, role, legacy):
    _,_,train,expected = split(bank,fold)
    root = args.out/"predict"/scenario/f"f{fold}-{role}"
    result, receipts = {}, {}
    for name in (PROPOSED,*legacy):
        path = root/name
        stamp = read_json(path/"receipt.json")
        if (stamp["protocol_sha256"] != sha256(args.protocol) or stamp["cache"] != bank.receipts
                or (stamp["scenario"],stamp["fold"],stamp["role"],stamp["selection"]["model"]) != (scenario,fold,role,name)
                or stamp["selection"]["fit_groups"] != sorted(set(train.parent))):
            raise ValueError(f"Prediction scope changed: {path}")
        for item,digest in stamp["files"].items():
            if sha256(path/item)!=digest:
                raise ValueError(f"Changed prediction file: {path/item}")
        rows=pd.read_csv(path/"rows.tsv",sep="\t",keep_default_na=False)
        axis=["study","index",*KEY,GROUP]
        if not rows[axis].equals(expected[axis]):
            raise ValueError(f"Prediction row universe changed: {path}")
        with np.load(path/"forecast.npz",allow_pickle=False) as arrays:
            if any(x.shape!=(len(rows),978) or not np.isfinite(x).all() for x in arrays.values()):
                raise ValueError(f"Invalid forecast: {path}")
            if set(arrays)&set(result):
                raise ValueError("Forecast arm duplication")
            result.update(arrays)
        receipts[name]=sha256(path/"receipt.json")
    result["B_abs"]=result["B0"]
    return expected,result,receipts


def scopes(rows):
    good=rows.quality.to_numpy()
    intended=rows[LINE].eq("HEPG2") & rows.study.eq("GSE92742") & rows.hours.eq(6)
    primary=intended & good & ~rows.prior_exact
    yield "HEPG2_6h_primary",primary
    yield "HEPG2_6h_exemplar",primary & rows.is_exemplar.astype(str).isin(["1","1.0","True"])
    yield "HEPG2_6h_low_quality",intended & ~good & ~rows.prior_exact
    yield "HEPG2_6h_parent_prior",primary & rows.prior_parent
    yield "HEPG2_6h_no_parent_prior",primary & ~rows.prior_parent
    for study in STUDIES:
        h24=rows[LINE].eq("HEPG2") & rows.study.eq(study) & rows.hours.eq(24) & good
        yield f"HEPG2_24h_{study}",h24
        yield f"HEPG2_24h_{study}_no_exact_prior",h24 & ~rows.prior_exact
    for cell in ("HUH7","PHH"):
        for hours in (6,24):
            yield f"{cell}_{hours}h",rows[LINE].eq(cell)&rows.hours.eq(hours)&good


def per_parent(rows,truth,pred,delta):
    rows=rows.reset_index(drop=True)
    sample=rows[GROUP].to_numpy()
    groups,codes=np.unique(sample,return_inverse=True)
    count=np.bincount(codes)
    sse=np.square((truth-pred).astype(np.float64)).sum(axis=1)
    sst=np.square(truth.astype(np.float64)).sum(axis=1)
    pp=np.zeros((len(groups),truth.shape[1]),dtype=np.float64)
    tt=np.zeros_like(pp)
    np.add.at(pp,codes,pred)
    np.add.at(tt,codes,truth)
    sim=unit(pp)@unit(tt).T
    draws=np.isclose(sim,sim.max(axis=1,keepdims=True),rtol=1e-9,atol=1e-11)
    top=np.diag(draws)/draws.sum(axis=1)
    within=np.einsum("ij,ij->i",unit(truth),unit(pred))
    return pd.DataFrame({"parent":groups,"rows":count,
        "rmse_sum":np.bincount(codes,np.sqrt(sse/truth.shape[1])),
        "sse":np.bincount(codes,sse),"sst":np.bincount(codes,sst),
        "absolute_sum":np.bincount(codes,np.sqrt(np.square((truth-pred+delta).astype(np.float64)).mean(1))),
        "cosine_sum":np.bincount(codes,within),"top1_sum":top,"retrieval_cases":1})


def aggregate(blocks):
    units=pd.concat(blocks,ignore_index=True).groupby("parent",sort=True).sum()
    units["rmse"]=units.rmse_sum/units.rows
    units["abs_rmse"]=units.absolute_sum/units.rows
    units["cosine"]=units.cosine_sum/units.rows
    units["top1"]=units.top1_sum/units.retrieval_cases
    return units


def summarize(units):
    return dict(rmse=units.rmse.mean(),r2_pooled=1-units.sse.sum()/units.sst.sum(),
                absolute_rmse=units.abs_rmse.mean(),cosine=units.cosine.mean(),top1=units.top1.mean())


def summarize_scope(records,scope,scenario,legacy):
    units={a:aggregate(x) for a,x in records.items()}
    names=list(units)
    key=units["B0"].index
    if any(not z.index.equals(key) for z in units.values()):
        raise ValueError("R15 comparisons do not have a common parent support")
    n=len(key)
    b0=units["B0"]
    points={a:summarize(z) for a,z in units.items()}
    point_rows=[dict(scenario=scenario,scope=scope,arm=a,parents=n,rows=z.rows.sum(),**points[a]) for a,z in units.items()]
    rng=np.random.default_rng(150104)
    rmse=np.column_stack([units[a].rmse for a in names])
    sse=np.column_stack([units[a].sse for a in names])
    sst=b0.sst.to_numpy()
    boots=np.empty((2,10000,len(names)),dtype=np.float64)
    for at in range(0,10000,128):
        end=min(at+128,10000)
        picked=rng.integers(n,size=(end-at,n))
        boots[0,at:end]=rmse[picked].mean(axis=1)
        boots[1,at:end]=1-sse[picked].sum(axis=1)/sst[picked].sum(axis=1)[:,None]
    for row in point_rows:
        for i,metric in enumerate(METRICS):
            low,high=np.quantile(boots[i,:,names.index(row["arm"])],CI)
            row.update({f"{metric}_low99":low,f"{metric}_high99":high})
    refs=["B0","B_pos"]
    if scenario=="B":
        refs+=["B_cell","B_pos_cell"]
    rows=[]
    for i,label in enumerate(METRICS):
        metric={a:boots[i,:,j] for j,a in enumerate(names)}
        outer={k:(k,metric[k]) for k in refs}
        winner=(min if i==0 else max)(legacy,key=lambda a:points[a][label])
        best=(np.minimum.reduce if i==0 else np.maximum.reduce)([metric[a] for a in legacy])
        outer["best-VC"]=(winner,best)
        for r,(name,expected) in outer.items():
            diff=metric[PROPOSED]-expected
            low,high=np.quantile(diff,CI)
            rows.append(dict(scenario=scenario,scope=scope,reference=r,metric=label,arm=PROPOSED,best_at_point=name,
                             difference=points[PROPOSED][label]-points[name][label],low99=low,high99=high,
                             passes=bool(high<0 if i==0 else low>0)))
    effects=units[PROPOSED].copy()
    effects["savings"]=b0.sse-effects.sse
    effects=effects.sort_values("savings",ascending=False)
    positive=effects.loc[effects.savings.gt(0)]
    exclusion=[]
    for i in (0,5,10,20):
        removed=positive.head(i)
        retained=effects.drop(removed.index)
        if retained.empty:
            continue
        p,q=retained.savings.sum(),removed.savings.sum()
        exclusion.append(dict(scenario=scenario,scope=scope,removed_top=i,retained_parents=len(retained),
                  positive_share_pct=100*q/positive.savings.sum() if len(positive) else np.nan,
                  net_share_pct=100*q/(p+q) if p+q else np.nan,b0_rmse=b0.loc[retained.index,"rmse"].mean(),
                  **summarize(retained)))
    return point_rows,rows,effects.reset_index().assign(scenario=scenario,scope=scope),exclusion


def run(args):
    manifest=args.out/"internal/freeze.json"
    banks={name:Bank(args.cache,args.protocol,name) for name in "AB"}
    if args.action=="score":
        frozen=read_json(manifest)
        if (frozen["protocol_sha256"]!=sha256(args.protocol) or frozen["code_sha256"]!=sha256(__file__)
                or tuple(frozen["legacy"])!=tuple(args.legacy)):
            raise ValueError("Scoring contract changed")
    result={}
    summary=[]
    for scenario,bank in banks.items():
        cases={}
        for fold in range(3):
            for role in "AB":
                name=f"{scenario}-f{fold}-{role}"
                rows,arms,hist=load_slice(args,bank,scenario,fold,role,args.legacy)
                result[name]=hist
                if args.action=="freeze":
                    continue
                if hist!=frozen["prediction_receipts"][name]:
                    raise ValueError("Changed forecast after freeze")
                other="B" if role=="A" else "A"
                obs=bank.response(rows,other)
                delta=bank.control(rows,other)-bank.control(rows,role)
                for label,mask in scopes(rows):
                    if mask.sum()==0:
                        continue
                    selected=cases.setdefault(label,{a:[] for a in arms})
                    for model,pred in arms.items():
                        selected[model].append(per_parent(rows.loc[mask],obs[mask],pred[mask],delta[mask]))
                print(f"Scored {name} ({len(rows)} wells)",flush=True)
        if args.action=="score":
            for label,records in cases.items():
                summary.append(summarize_scope(records,label,scenario,args.legacy))
    if args.action=="freeze":
        manifest.parent.mkdir(parents=True,exist_ok=True)
        write_json(manifest,dict(protocol_sha256=sha256(args.protocol),code_sha256=sha256(__file__),legacy=args.legacy,
                                 prediction_receipts=result))
        return
    path=manifest.parent/"scored"
    with TemporaryDirectory(dir=manifest.parent,prefix=".score-") as t:
        out=Path(t)
        for idx,table in enumerate(("points","paired","contributions","top_group_exclusion")):
            data=pd.concat([pd.DataFrame(x[idx]) for x in summary],ignore_index=True)
            data.to_csv(out/f"{table}.tsv",sep="\t",index=False)
        write_json(out/"receipt.json",dict(freeze_sha256=sha256(manifest),files={f.name:sha256(f) for f in out.iterdir()}))
        if path.exists():
            raise FileExistsError(path)
        out.rename(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","score"))
    for key in ("cache","protocol","assets","out"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    parser.add_argument("--legacy",nargs="+",choices=LEGACY,default=LEGACY)
    run(parser.parse_args())


if __name__=="__main__":
    main()
