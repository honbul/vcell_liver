"""Zero-shot LINCS-to-Tahoe prediction, manifest freeze, and external scoring."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import MIN_CELLS, eligibility, slice_name
from src.goal_tracks.t1_io import load_task, score_truth
from src.goal_tracks.t1_score import registry
from src.p123.p1_score import ENDPOINTS, StudyBook
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY
from src.r13_optimization.lincs_transfer import clean, structural_keys

from .data import table
from .design import Bank, chemical_features
from .estimators import chemical_grid, cross_position
from .forecast import choose_proposed, setup, split


DOSE = 5.


def published(args):
    stage = args.tahoe/"t1-cache"
    t1 = args.tahoe/"t1/first/d5"
    t2 = args.tahoe/"t2/d5"
    stamp1, stamp2 = (read_json(p/"freeze.json") for p in (t1,t2))
    if (sha256(t2/"freeze.json") != sha256(args.t2_reference)
            or stamp2["t1_freeze_sha256"] != sha256(t1/"freeze.json")
            or stamp2["stage_sha256"] != sha256(stage/"receipt.json")):
        raise ValueError("The historical Tahoe 5 µM benchmark differs from its committed freeze")
    if len(stamp2["eligible_known"]) != 10:
        raise ValueError("T2 comparator registry changed")
    return stage,t1,stamp1,stamp2


def common_genes(args, bank, stage):
    lincs = pd.read_csv(args.assets/"metadata/genes.tsv",sep="\t")
    tahoe = pd.read_csv(args.tahoe_assets/"metadata/genes.tsv",sep="\t")
    usable = tahoe.loc[tahoe.ensembl_id.notna() & ~tahoe.ensembl_id.duplicated(keep=False)]
    source = lincs.loc[lincs.ensembl_id.notna() & ~lincs.ensembl_id.duplicated(keep=False)]
    joined = source.merge(usable,on="ensembl_id",validate="one_to_one",suffixes=("_lincs","_tahoe"))
    shared = set(joined.token_id_tahoe)
    for fold in range(3):
        options = read_json(stage/f"f{fold}/panels.json")
        for n in MIN_CELLS:
            shared &= set(options[slice_name(DOSE,n,fold,"panel") + "_-1"])
    use = joined.loc[joined.token_id_tahoe.isin(shared)].sort_values("token_id_lincs")
    ids = pd.Index(bank.landmarks.pr_gene_id).get_indexer(use.token_id_lincs)
    if len(ids) < 30 or np.any(ids < 0):
        raise ValueError("Too few confirmed common Ensembl landmarks")
    return use.reset_index(drop=True), ids


def load_published(args, context, n, fold, role, include_vc):
    stage,t1,stamp1,stamp2 = context
    part = slice_name(DOSE,n,fold,role)
    t1path = t1/"predictions"/part
    evidence = {"T1":sha256(t1path/"receipt.json")}
    if evidence["T1"] != stamp1["tasks"][part]:
        raise ValueError("T1 forecast changed")
    protocol = args.tahoe/"t1-code/reports/goal_tracks/PROTOCOL.md"
    _, rows, panel, t1arms = load_task(t1path,stage,protocol,DOSE,n,fold,role)
    arms = {"B_pos-all":t1arms["B_pos-all"],"B0":t1arms["B0"]}
    if include_vc:
        for model,path in stamp2["eligible_known"].items():
            folder = Path(path)/part
            evidence[model] = sha256(folder/"receipt.json")
            if evidence[model] != stamp2["tasks"][part][model]:
                raise ValueError(f"Stored T2 prediction changed: {model}")
            _, other, genes, pred = load_task(folder,stage,protocol,DOSE,n,fold,role)
            if not rows[KEY].equals(other[KEY]) or not np.array_equal(panel,genes) or set(pred)!={model}:
                raise ValueError("T2 forecast doesn't share the common slice")
            arms.update(pred)
    return rows,panel,arms,evidence


def predict(args):
    bank=Bank(args.cache,args.protocol,args.scenario)
    context=published(args)
    genes,loci=common_genes(args,bank,context[0])
    inner, val, train, _ = split(bank, 3, external=True)
    if set(train.parent)&bank.blocked:
        raise ValueError("Tahoe parent reached the LINCS fit")
    labels,truth,fit_pos,val_pos,position_gain=setup(bank,inner,val,args.role)
    (rule,gain), objective=choose_proposed(bank, inner,val,labels,truth,fit_pos,val_pos,position_gain)
    final_y=bank.response(train,args.role)
    tr_pos,_=cross_position(train,train.iloc[:0],final_y)
    signatures=bank.transfer(train)
    final=final_y-position_gain*tr_pos-signatures

    entities=table(args.tahoe_assets/"features/entities.tsv").set_index(ENTITY,verify_integrity=True)
    prepared=[]
    evidence={}
    for n in MIN_CELLS:
        for fold in range(3):
            rows,_,_,proof=load_published(args,context,n,fold,args.role,False)
            part=slice_name(DOSE,n,fold,args.role)
            evidence[part]=proof
            prepared.append((part,rows))
    entities_needed=sorted({a for _,rows in prepared for a in rows[ENTITY]})
    records=entities.loc[entities_needed].reset_index()
    records["parent"]=records.smiles.map(lambda v: structural_keys(clean(v))[1])
    if records.parent.eq("").any():
        raise ValueError("External Tahoe structure could not be standardized")
    if set(train.parent)&set(records.parent):
        raise ValueError("Direct structural intersection was not closed by the published identity ledger")
    records[GROUP]=records[GROUP].astype(int)
    zeros=np.zeros((len(records),len(bank.landmarks)),dtype=np.float32)
    supported=np.ones(len(records),dtype=bool)
    inputs=zeros.copy()
    if args.scenario == "B":
        for i,g in enumerate(records[GROUP]):
            vector=bank.external_inputs.get((str(g),6))
            if vector is None:
                supported[i]=False
            else:
                inputs[i]=vector
    fitted_rows=records.loc[supported].reset_index(drop=True)
    out=args.out/"external"/args.scenario/args.role
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent,prefix=".external-") as temp:
        p=Path(temp)
        estimated=chemical_grid(train,fitted_rows,bank.chemistry(train),chemical_features(fitted_rows.parent),
                                 final,args.scenario,signatures,inputs[supported],[rule],p/"model.npz")[rule]
        predictions=zeros.copy()
        predictions[supported]=gain*estimated+inputs[supported]
        predictions*=np.float32(np.log(2))
        inputs*=np.float32(np.log(2))
        records["supported"]=supported
        records[[ENTITY,GROUP,"supported"]].to_csv(p/"entities.tsv",sep="\t",index=False)
        genes.to_csv(p/"genes.tsv",sep="\t",index=False)
        np.savez_compressed(p/"forecasts.npz",proposal=predictions[:,loci],B_cell=inputs[:,loci])
        write_json(p/"receipt.json",dict(scenario=args.scenario,role=args.role,protocol_sha256=sha256(args.protocol),
                   code_sha256=sha256(__file__),cache=bank.receipts,history=evidence,train_groups=sorted(set(train.parent)),
                   blocked_parents=len(bank.blocked),rule=rule,gain=gain,position_gain=position_gain,inner_loss=objective,
                   files={f.name:sha256(f) for f in sorted(p.iterdir())}))
        if out.exists():
            raise FileExistsError(out)
        p.rename(out)


def read_bundle(args, context):
    result, hashes = {}, {}
    for scenario in "AB":
        for role in "AB":
            p=args.out/"external"/scenario/role
            receipt=read_json(p/"receipt.json")
            if (receipt["protocol_sha256"]!=sha256(args.protocol) or receipt["code_sha256"]!=sha256(__file__)
                    or (receipt["scenario"],receipt["role"])!=(scenario,role)):
                raise ValueError("External predictions have unexpected provenance")
            hashes[scenario+role]=sha256(p/"receipt.json")
            for name,digest in receipt["files"].items():
                if sha256(p/name)!=digest:
                    raise ValueError(f"External artifact changed: {name}")
            pred=np.load(p/"forecasts.npz",allow_pickle=False)
            rows=pd.read_csv(p/"entities.tsv",sep="\t").set_index(ENTITY,verify_integrity=True)
            genes=pd.read_csv(p/"genes.tsv",sep="\t")
            result[scenario,role]=(receipt,rows,genes,dict(pred))
    return result,hashes


def interval_rows(points,boot,known):
    result=[]
    for (scope,cohort), rows in points.groupby(["stratum","cohort"]):
        values=rows.set_index("arm")
        for metric in ENDPOINTS:
            rmse=metric=="contrast_rmse"
            ref=(min if rmse else max)(sorted(known),key=lambda name:values.at[name,metric])
            best=(np.minimum.reduce if rmse else np.maximum.reduce)([boot[scope,k,metric] for k in sorted(known)])
            for arm in ("R15-A","R15-B"):
                if arm not in values.index:
                    continue
                for baseline in ("best-T2","B0","B_pos-all","B_cell"):
                    if baseline not in values.index and baseline!="best-T2":
                        continue
                    point=ref if baseline=="best-T2" else baseline
                    distribution=best if baseline=="best-T2" else boot[scope,baseline,metric]
                    low,high=np.quantile(boot[scope,arm,metric]-distribution,[.005,.995])
                    result.append(dict(stratum=scope,cohort=cohort,minimum_cells=values.at[arm,"minimum_cells"],
                        arm=arm,metric=metric,reference=baseline,best_at_point=point,
                        difference=values.at[arm,metric]-values.at[point,metric],low99=low,high99=high,
                        passes=bool(high<0 if rmse else low>0)))
    return pd.DataFrame(result)


def verify_inputs(args,context,bundles):
    bound={}
    for n in MIN_CELLS:
        for fold in range(3):
            for role in "AB":
                part=slice_name(DOSE,n,fold,role)
                rows,panel,_,proof=load_published(args,context,n,fold,role,True)
                for scenario in "AB":
                    stamp,entities,genes,pred=bundles[scenario,role]
                    if (stamp["history"][part]["T1"]!=proof["T1"] or not set(rows[ENTITY]) <= set(entities.index)
                            or not set(genes.token_id_tahoe) <= set(panel)
                            or any(x.shape!=(len(entities),len(genes)) for x in pred.values())):
                        raise ValueError("External forecast differs from the registered prediction axes")
                bound[part]=proof
    return bound


def official(args):
    context=published(args)
    bundles,hashes=read_bundle(args,context)
    manifest=args.out/"external/freeze.json"
    guard=dict(protocol_sha256=sha256(args.protocol),code_sha256=sha256(__file__),bundles=hashes,
               published=verify_inputs(args,context,bundles))
    if args.action=="freeze":
        write_json(manifest,guard)
        return
    if read_json(manifest)!=guard:
        raise ValueError("External score requires its unchanged freeze")
    stage=context[0]
    records,pairs=[],[]
    for n in MIN_CELLS:
        books={name:StudyBook(registry(stage),DOSE,n,name) for name in ("all-A","LINCS-supported")}
        for fold in range(3):
            for role in "AB":
                rows,panel,old,_=load_published(args,context,n,fold,role,True)
                _,ma,ga,pa=bundles["A",role]
                _,mb,gb,pb=bundles["B",role]
                if not ga.equals(gb):
                    raise ValueError("The external arms require a common gene axis")
                axis=pd.Index(panel).get_indexer(ga.token_id_tahoe)
                observed=score_truth(stage,DOSE,n,fold,role,rows,panel)[:,axis]
                arrays={k:v[:,axis] for k,v in old.items()}
                arrays["R15-A"]=pa["proposal"][ma.index.get_indexer(rows[ENTITY])]
                books["all-A"].add(rows,observed,arrays)
                ix=mb.index.get_indexer(rows[ENTITY])
                subset=eligibility(rows.loc[mb.supported.to_numpy()[ix]].assign(_index=np.flatnonzero(
                    mb.supported.to_numpy()[ix])),0)
                if not subset.empty:
                    active=subset.pop("_index").to_numpy(dtype=int)
                    both={k:v[active] for k,v in arrays.items()}
                    both["R15-B"]=pb["proposal"][ix[active]]
                    both["B_cell"]=pb["B_cell"][ix[active]]
                    books["LINCS-supported"].add(subset,observed[active],both)
                print(f"Scored external d5 n{n} fold{fold} {role}",flush=True)
        for name,book in books.items():
            points,boot=book.results()
            records.append(points)
            pairs.append(interval_rows(points,boot,set(context[-1]["eligible_known"])))
    path=args.out/"external/scored"
    path.mkdir(exist_ok=False)
    pd.concat(records).to_csv(path/"points.tsv",sep="\t",index=False)
    pd.concat(pairs).to_csv(path/"paired.tsv",sep="\t",index=False)
    write_json(path/"receipt.json",dict(freeze_sha256=sha256(manifest),
                                       files={p.name:sha256(p) for p in path.iterdir()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("predict","freeze","score"))
    for key in ("cache","protocol","assets","tahoe-assets","tahoe","t2-reference","out"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    parser.add_argument("--scenario",choices="AB")
    parser.add_argument("--role",choices="AB")
    args=parser.parse_args()
    (predict if args.action=="predict" else official)(args)


if __name__=="__main__":
    main()
