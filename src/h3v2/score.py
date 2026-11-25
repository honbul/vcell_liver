"""Consume all frozen primary, secondary and external outcomes in one scored run."""
import argparse
from datetime import datetime,timezone

import numpy as np
import pandas as pd

from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.metrics import CELL, DRUG, Evaluator, contrast, macro, stack
from nextgen_h2.run import META, read_json
from nextgen_h2.score import adult_licob, external_labels
from vc_build.io import _publish, publish_json, sha256

from . import core_score
from .atlas import model_input_digest
from .data import ASSETS, ROOT, SCREENS, SPLITS, register, read_data
from .error_components import decompose
from .final_fit import exact_join, external_query, external_operating_point
from .grid import SEEDS, FAMILIES
from .selection import PRE_UNSEALING_FREEZE

ARMS = (*FAMILIES,"Champion","Champion-ensemble")
AXES = ("spearman","interaction_spearman","ndcg_at_5")
MARGINS = dict(zip(AXES,(.01,.02,.015)))
ALL_PARTITIONS = (*SPLITS,"external")
BOOTSTRAPS = 2000


def preflight():
    lock_path=ROOT/"locks/frozen.json"
    lock=read_json(lock_path)
    if lock["outer_labels_opened"] or lock["seeds"]!=list(SEEDS):
        raise ValueError("Incorrect freeze protocol or seed set")
    hashes={}
    frames={}
    for split in ALL_PARTITIONS:
        ext=split=="external"
        expected=external_query() if ext else read_data(split,"query")
        root=ROOT/"final"/split
        meta=read_json(root/"references/receipt.json")
        if (meta["lock_sha256"],meta["outer_labels_loaded"],meta["rows"])!=(sha256(lock_path),False,len(expected)):
            raise ValueError("Reference forecast scope changed")
        path=root/"references/forecast.parquet"
        reference=parquet_lock(path,meta["forecast_sha256"])
        exact_join(expected,reference.set_index("row_id"))
        frame=reference.copy()
        atlas_dir = ROOT / "features" / split
        if not ext:
            atlas_dir /= "outer"
        atlas = read_json(atlas_dir / "receipt.json")
        if sha256(atlas_dir / "schema.pkl") != atlas["schema_sha256"]:
            raise ValueError(f"Final feature schema changed for {split}")
        input_stamp = model_input_digest(atlas["atlas_sha256"])
        hashes[split]={"reference":meta["forecast_sha256"],"forecasts":{},"checkpoints":{}}
        for family in FAMILIES:
            chosen=(external_operating_point(lock,family) if ext else lock["splits"][split]["families"][family])
            for seed in SEEDS:
                part=root/family/str(seed)
                receipt=read_json(part/"receipt.json")
                if (receipt["split"],receipt["family"],receipt["seed"],receipt["operating_point"],
                    receipt["query_rows"],receipt["outer_labels_loaded"],receipt["lock_sha256"])!=(
                      split,family,seed,chosen["id"],len(expected),False,sha256(lock_path)) or not receipt.get("lincs_full_unclipped"):
                    raise ValueError("Frozen model evidence disagrees with the winning inner operating point")
                if (receipt["schema_sha256"], receipt["model_input_sha256"], receipt["atlas_sha256"]) != (
                    atlas["schema_sha256"], input_stamp, atlas["atlas_sha256"]
                ):
                    raise ValueError(f"Final fitting inputs changed for {split}/{family}/{seed}")
                if receipt["selected_budget"] != (chosen["selected_branch_budgets"] or chosen["selected_budget"]):
                    raise ValueError(f"Final tree or epoch budget changed for {split}/{family}/{seed}")
                frame_seed=parquet_lock(part/"forecast.parquet",receipt["forecast_sha256"]).set_index("row_id")
                exact_join(expected,frame_seed)
                name=f"{family}__{seed}"
                frame[name]=frame_seed.reindex(frame.row_id).predicted.to_numpy()
                if sha256(part/"state.pkl")!=receipt["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint seal changed {split}/{name}")
                hashes[split]["forecasts"][name]=receipt["forecast_sha256"]
                hashes[split]["checkpoints"][name]=receipt["checkpoint_sha256"]
        winner=lock["external"]["champion"] if ext else lock["splits"][split]["champion"]
        if winner not in FAMILIES:
            raise ValueError("Frozen single champion was not registered")
        for seed in SEEDS:
            frame[f"Champion__{seed}"]=frame[f"{winner}__{seed}"]
        frame["Champion-ensemble"]=frame[[f"Champion__{seed}" for seed in SEEDS]].mean(axis=1)
        hits=[k for k in frame if k not in META]
        if not np.isfinite(frame[hits].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite model predictions on the registered native support: {split}")
        frames[split]=frame
    return frames,hashes,lock


def join_truth(reference,observed):
    if set(observed.row_id)!=set(reference.row_id) or observed.row_id.duplicated().any():
        raise ValueError("Held evaluation rows differ from its frozen outcomes")
    joint=reference.merge(observed[["row_id","truth"]],on="row_id",validate="one_to_one")
    if not np.isfinite(joint.truth).all():
        raise ValueError("Expected measured native endpoint support")
    return joint


def tagged(columns):
    table={}
    for column in columns:
        pair=column.rsplit("__",1)
        table[column]=(pair[0],pair[1]) if len(pair)>1 else (column,"fixed")
    return table


def units(frame, clusters, fold, scope, supports, columns, *, tissue=""):
    score=Evaluator(frame,clusters,fold,scope=scope,interaction=True)
    pairs=[]
    for name,selector in supports.items():
        if selector is not None and not np.asarray(selector).any():
            continue
        for method in columns:
            pair=score.score(method,frame[method].to_numpy(),selector,name)
            pairs.extend(pair)
    if not pairs:
        return None
    d,c=stack(pairs)
    return d.assign(tissue=tissue), c.assign(tissue=tissue)


def native_metric_points(table, metrics):
    rows=[]
    for (scope,support,arm,tissue), frame in table.groupby(["scope","support","arm","tissue"],sort=True):
        model,seed = tagged([arm])[arm]
        by={"scope":scope,"support":support,"model":model,"seed":seed,"tissue":tissue}
        for screen, block in [("macro",frame),*frame.groupby("screen")]:
            rows.extend(by|{"screen":screen,"metric":m,"value":macro(block,m),
                            "unit_count":int(block.unit.nunique()),"cluster_count":int(block.cluster.nunique())}
                        for m in metrics)
    return pd.DataFrame(rows)


def mean_and_sd(points):
    key=[x for x in points if x not in ("value","seed")]
    parts=points[points.seed.ne("fixed")]
    if len(parts) and not parts.groupby(key,dropna=False).seed.nunique().eq(3).all():
        raise ValueError("There must be exactly three score-level replicates")
    result=parts.groupby(key,dropna=False).value.agg(["mean","std"]).reset_index().rename(columns={"mean":"value","std":"seed_sd"})
    return pd.concat([points,result.assign(seed="mean")],ignore_index=True)


def mean_units(table):
    data=table.copy()
    data[["arm","seed"]]=pd.DataFrame([tagged([k])[k] for k in data.arm],index=data.index)
    keys=[k for k in ("scope","fold","screen","support","tissue","arm","unit","cluster") if k in data]
    return data.drop(columns="seed").groupby(keys,as_index=False).mean(numeric_only=True)


def paired(d,c):
    out=[]
    for table,names in ((d,DRUG),(c,CELL)):
        aggregate=mean_units(table)
        for (scope,support), unit in aggregate.groupby(["scope","support"]):
            references={"H2","C1","drug_ridge"} & set(unit.arm)
            for candidate in (*FAMILIES,"Champion"):
                if candidate not in set(unit.arm):
                    continue
                for baseline in sorted(references):
                    for screen,g in [("macro",unit),*unit.groupby("screen")]:
                        for name in names:
                            delta,lo,hi,n=contrast(g,candidate,baseline,name,draws=BOOTSTRAPS,seed=318743)
                            out.append({"scope":scope,"support":support,"screen":screen,"candidate":candidate,
                                        "baseline":baseline,"metric":name,"delta":delta,"lower":lo,"upper":hi,
                                        "cluster_axis":"scaffold" if name in DRUG else "global_lineage",
                                        "clusters":n,"draws":BOOTSTRAPS})
    return pd.DataFrame(out)


def held_points(frames):
    joint=pd.concat(frames,ignore_index=True)
    id_cols=tuple(META)+("truth","fold","lincs")
    cols=[k for k in joint if k not in id_cols]
    scopes=core_score.screen_scores(joint.rename(columns={"truth":"observed"}),cols,
                    core_score.shared_draws(joint,1200,14639),minimum=5)
    axes={m:m for m in core_score.ALL_METRICS}|{
        "within_drug_spearman":"spearman","within_drug_pearson":"pearson","precision_at_k":"precision_at_5",
        "residual_spearman":"interaction_spearman",
    }
    scientific={}
    for name in scopes:
        scientific[name]={}
        for raw, (point, sampling) in scopes[name].items():
            scientific[name][raw]=({axes[key]:val for key,val in point.items()},
                                    {axes[key]:val for key,val in sampling.items()})
    screen=list(scopes)
    scientific["macro"]={}
    for raw in cols:
        pieces=[scientific[s][raw] for s in screen]
        view={m:mean_strict([p[0][m] for p in pieces]) for m in pieces[0][0]}
        boot={m:np.mean([p[1][m] for p in pieces],axis=0) for m in pieces[0][1]}
        scientific["macro"][raw]=view,boot
    points=[]
    ranges=[]
    support=joint.lineage_group.nunique()
    models=tagged(cols)
    for screen,shelf in scientific.items():
        for arm,(point,boot) in shelf.items():
            model,seed=models[arm]
            points.extend({"scope":"held","support":"core68","screen":screen,"tissue":"",
                           "model":model,"seed":seed,"metric":m,"value":p,"unit_count":support,
                           "cluster_count":support} for m,p in point.items())
        for candidate in (*FAMILIES,"Champion"):
            fitted=[shelf[f"{candidate}__{seed}"] for seed in SEEDS]
            for baseline in ("structured_v1","drug_ridge_v1","H3v1"):
                bx=[shelf[baseline]] if baseline in shelf else [shelf[f"{baseline}__{s}"] for s in SEEDS]
                for m in bx[0][1]:
                    delta=np.mean([v[0][m] for v in fitted])-np.mean([v[0][m] for v in bx])
                    differences=np.mean([v[1][m] for v in fitted],axis=0)-np.mean([v[1][m] for v in bx],axis=0)
                    lo,hi=np.quantile(differences,[.025,.975])
                    ranges.append({"scope":"held","support":"core68","screen":screen,"candidate":candidate,
                                   "baseline":baseline,"metric":m,"delta":delta,"lower":lo,"upper":hi,
                                   "cluster_axis":"global_lineage","clusters":support,"draws":1200})
    return pd.DataFrame(points),pd.DataFrame(ranges)


def mean_strict(values):
    return float(np.mean(values))


def rules(ranges):
    out={}
    for model in (*FAMILIES,"Champion"):
        matrix=ranges[ranges.candidate.eq(model)&ranges.baseline.eq("H2")]
        main=matrix[matrix.scope.eq("drug")&matrix.support.eq("all")]
        outer=main[main.screen.eq("macro")].set_index("metric")
        external=matrix[matrix.scope.eq("GDSC1")&matrix.screen.eq("macro")].set_index("metric")
        gates={}
        for axis in AXES:
            others=[x for x in AXES if x!=axis]
            gain=outer.loc[axis]
            gates[axis]={
                "minimum_improvement_and_ci":bool(gain.delta>=MARGINS[axis] and gain.lower>0),
                "other_two_axes_protected":bool(outer.loc[others,"lower"].ge(-.005).all()),
                "two_screens_improved":int(main.loc[main.metric.eq(axis)&main.screen.ne("macro"),"delta"].gt(0).sum())>=2,
                "gdsc1_three_axes_protected":bool(external.loc[list(AXES),"lower"].ge(-.005).all()),
            }
        out[model]={"by_axis":gates,"verdict":"GO-MODEL" if any(all(x.values()) for x in gates.values()) else "NO-GO"}
    return out


def evaluate():
    args=argparse.ArgumentParser()
    args.add_argument("--preflight",action="store_true")
    flags=args.parse_args()
    root=ROOT/"score"
    if (root/"consumed.json").exists():
        raise FileExistsError("Outer outcomes were already released for this experiment")
    frames,hashes,lock=preflight()
    if flags.preflight:
        print("All held forecasts, feature schemas, checkpoints and 16 row rosters verified without outcomes",flush=True)
        return
    claim={"score_passes":1,"forecasts":hashes,"released_utc":datetime.now(timezone.utc).isoformat(),
           "locked_operating_points_sha256":sha256(ROOT/"locks/frozen.json")}
    publish_json(root/"consumed.json",claim)
    manifest=read_json(ROOT/"splits/manifest.json")
    registry=register()
    asset=read_json(ASSETS/"data_receipt.json")
    cell=parquet_lock(ASSETS/"cells.parquet",asset["files"]["cells.parquet"])
    if not isinstance(cell.index,pd.RangeIndex):
        cell=cell.reset_index()
    tissue=cell.set_index("cell_id").tissue
    support={"drug":{"all":None},"cell":{"all":None}}
    results,tissues,held,handled,errors=[],[],[],{},[]
    for name in SPLITS:
        track,fold=name.rsplit("-",1)
        file=ROOT/"outcomes"/f"{name}.parquet"
        archive=parquet_lock(file,manifest["split_support"][name]["label_sha256"])
        frame=join_truth(frames[name],archive).assign(fold=int(fold))
        handled[name]=len(frame)
        if track in SCREENS:
            held.append(frame)
            continue
        cols=[key for key in frames[name] if key not in (*META,"lincs")]
        sets={"all":None,"lincs724":frame.lincs,"structure462":~frame.lincs} if track=="drug" else support["cell"]
        results.append(units(frame,registry.scaffold_block,int(fold),track,sets,cols))
        if track=="drug":
            diagnosed=["C1","H2",*(f"{name}__{SEEDS[0]}" for name in (*FAMILIES,"Champion"))]
            errors.extend(decompose(frame,diagnosed,track,int(fold),sets))
            for label in sorted(tissue.unique()):
                sub=frame.loc[frame.cell_id.map(tissue).eq(label)].reset_index(drop=True)
                if sub.empty:
                    continue
                tissues.append(units(sub,registry.scaffold_block,int(fold),track,{"all":None},cols,tissue=label))
    label,licob,ext_blocks=external_labels(ASSETS,asset)
    external=join_truth(frames["external"],label)
    cols=[k for k in frames["external"] if k not in (*META,"lincs")]
    for name, part in external.groupby("source_id"):
        if name=="LICOB":
            part=part.rename(columns={"lincs":"lincs_observed"})
            part=adult_licob(part,licob)
        handled[name]=len(part)
        results.append(units(part.reset_index(drop=True),ext_blocks,0,name,{"all":None},cols))
    d,c=map(pd.concat,zip(*results))
    td,tc=map(pd.concat,zip(*tissues))
    hscore,hci=held_points(held)
    out=pd.concat([native_metric_points(d,DRUG),native_metric_points(c,CELL),hscore],ignore_index=True)
    out=mean_and_sd(out)
    groups=pd.concat([native_metric_points(td,DRUG),native_metric_points(tc,CELL)],ignore_index=True)
    groups=mean_and_sd(groups)
    groups["status"]="estimated"
    blank=set(tissue.unique())-set(groups.tissue.unique())
    if blank:
        gaps=pd.DataFrame({"scope":"drug","tissue":sorted(blank),"status":"insufficient_native_support"})
        groups=pd.concat([groups,gaps],ignore_index=True)
    CI=pd.concat([paired(d,c),hci],ignore_index=True)
    tables={"H3V2_METRICS.tsv":out,"H3V2_PAIRED_95CI.tsv":CI,"H3V2_PANCANCER_BY_TISSUE.tsv":groups,
            "H3V2_ERROR_COMPONENTS.tsv":pd.DataFrame(errors)}
    for name,table in tables.items():
        _publish(root/name,lambda p:table.to_csv(p,sep="\t",index=False))
    for label,table in (("drug_units",d),("cell_units",c)):
        _publish(root/f"{label}.parquet",lambda p:table.to_parquet(p,index=False))
    verdicts=rules(CI)
    ceiling_scope = "The frozen candidate grids, input modalities, and registered data. This is an empirical joint-gate outcome."
    if lock.get("freeze_reason") == PRE_UNSEALING_FREEZE:
        ceiling_scope += " Selection covers r1, r2, and boundary-1. The ledger records the open neighboring settings."
    publish_json(root/"score_receipt.json",claim|{
        "finished_utc":datetime.now(timezone.utc).isoformat(),"parts":handled,
        "claim_sha256":sha256(root/"consumed.json"),"verdicts":verdicts,
        "overall":"GO-MODEL" if any(r["verdict"]=="GO-MODEL" for r in verdicts.values()) else "CEILING-CONFIRMED",
        "ceiling_scope":ceiling_scope,
        "source_manifest_sha256":sha256(ROOT/"splits/manifest.json"),
        "tables":{name:sha256(root/name) for name in tables},"tissue_roster":sorted(tissue.unique()),
        "seed_accounting":"point mean and sample SD of seed-level scores; explicit Champion-ensemble separate",
    })


if __name__=="__main__":
    evaluate()
