"""Evaluate a locked JHH-5 donor rule on a single adult JHH-7 plate."""

import argparse
from hashlib import sha256 as hash_text
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_access import CONTEXT, assigned_controls
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.i_preflight import read_pinned, write_plan
from src.r19_plan.i_score import checked_array, checked_rows, parent_summaries, squared_rows
from src.r19_plan.i_stage import parse_landmarks, pinned_bytes
from src.r19_plan.jhh5_donor_attribution import MEAN, FIX, SHUFFLE, STATE, shrink_consensus, wrong_parent
from src.r19_plan.r18_input_bundle import indices
from src.r19_plan.r18_matrix_index import metadata_index
from src.r19_plan.t4_position import OP, SC, ST, compare, contrast, retrieval
from src.r19_plan.t4_scaffold_transfer import cluster_effects
from src.r19_plan.t4_state import state_forecast
from src.r19_plan.t4_workflow import scalar
from src.r19_plan.x_inputs import check_unique, control_axis


SCALED = "JHH5_origin_scalar"
ARMS = ("B0", MEAN, FIX, SCALED, STATE, SHUFFLE)
PAIRS = ((FIX, MEAN), (FIX, SCALED), (MEAN, STATE), (MEAN, "B0"), (MEAN, SHUFFLE))


def controls_for(controls, rows, role):
    selected = controls.loc[controls.role.eq(role)].merge(rows[CONTEXT].drop_duplicates(), on=CONTEXT, validate="many_to_one")
    control_axis(selected, rows, role)
    return selected


def lock(args):
    audited = read_json(args.audit_pin)
    if sha256(args.audit/"receipt.json") != sha256(args.audit_pin) or not audited["metadata_only"] or audited["expression_read"]:
        raise ValueError("The source CMap support audit changed")
    plans = {name: read_pinned(args.audit, name, h) for name,h in audited["files"].items()}
    target = plans["targets.tsv"].loc[lambda rows: rows.cell_id.eq("JHH7")]
    all_fit = plans["targets.tsv"].loc[lambda rows: rows.cell_id.eq("JHH5")]
    fit = all_fit.loc[~all_fit.scaffold.isin(target.scaffold) & ~all_fit.parent.isin(target.parent)]
    if (target.parent.nunique(), len(target), target.rna_plate.nunique(), fit.parent.nunique()) != (99, 99, 1, 475):
        raise ValueError("The registered JHH-7 and purged JHH-5 supports differ")
    active = set(target.parent) | set(fit.parent)
    source = plans["donors.tsv"].loc[lambda rows: rows.view.eq("nonhepatic") & rows.parent.isin(active)]
    support = source.groupby("parent").agg(cells=("cell_id", "nunique"), plates=("rna_plate", "nunique"))
    if (set(support.index) != active or not (support.ge(2).all().all())
            or set(source.rna_plate) & (set(target.rna_plate) | set(fit.rna_plate))
            or not source.actual_dose_uM.eq("10").all()):
        raise ValueError("The same-dose donor support or independent plate separation differs")
    target = target.assign(in_JHH5=target.parent.isin(all_fit.parent))
    treatments = pd.concat((target, fit, source), ignore_index=True)
    used = treatments[CONTEXT].drop_duplicates()
    controls = assigned_controls(plans["controls.tsv"].merge(used, on=CONTEXT, validate="many_to_one"))
    if int(controls.cell_id.eq("JHH7").sum()) != 13:
        raise ValueError("The fixed JHH-7 DMSO support differs")
    check_unique(pd.concat((treatments, controls), ignore_index=True))

    standard = read_json(args.cmap_pin)
    landmarks = parse_landmarks(pinned_bytes(args.landmarks, standard["landmark_axis_sha256"]))
    genes = landmarks.pr_gene_id.to_numpy()
    matrices = standard["matrix_records"]
    t, trt_genes, trt = metadata_index(Path(matrices["treated"]["path"]), treatments, genes)
    controls, ctl_genes, ctl = metadata_index(Path(matrices["controls"]["path"]), controls, genes)
    if matrices != dict(treated=trt, controls=ctl):
        raise ValueError("The physical Level 3 metadata does not match CMap")
    target, fit = (t.loc[t.cell_id.eq(cell)].copy() for cell in ("JHH7", "JHH5"))
    source = t.loc[~t.cell_id.isin(("JHH5", "JHH7"))].copy()
    locked = args.run/"access"
    if locked.exists():
        raise FileExistsError(locked)
    with TemporaryDirectory(dir=args.run, prefix=".jhh7-input-") as folder:
        out = Path(folder)
        splits, stats = {}, []
        for i, role in enumerate("ABC"):
            outcome_role = "ABC"[(i+2)%3]
            fit_dmsos = controls_for(controls, fit, role)
            donor_dmsos = controls_for(controls, source, role)
            outcome_dmsos = controls_for(controls, target, outcome_role)
            if set(target.inst_id) & set(pd.concat((fit, source, donor_dmsos, fit_dmsos)).inst_id):
                raise ValueError("JHH-7 outcome crossed the predictor boundary")
            query = target.drop(columns="level3_row_index")
            splits[role] = dict(predict=write_plan(out/"predict"/role, {"fit.tsv":fit,"fit_controls.tsv":fit_dmsos,
                "source.tsv":source,"source_controls.tsv":donor_dmsos,"query.tsv":query}),
                score=write_plan(out/"score"/role, {"query.tsv":target,"controls.tsv":outcome_dmsos}))
            stats.append(dict(role=role, score_role=outcome_role, score_wells=len(target),
                score_dmsos=len(outcome_dmsos), fit_wells=len(fit), fit_parents=fit.parent.nunique(), fit_dmsos=len(fit_dmsos),
                donor_wells=len(source), donor_cells=source.cell_id.nunique(), donor_plates=source.rna_plate.nunique()))
        landmarks.to_csv(out/"landmarks.tsv",sep="\t",index=False)
        trt_genes.to_csv(out/"treated_genes.tsv",sep="\t",index=False)
        ctl_genes.to_csv(out/"controls_genes.tsv",sep="\t",index=False)
        pd.DataFrame(stats).to_csv(out/"support.tsv",sep="\t",index=False)
        write_json(out/"receipt.json", dict(protocol_sha256=sha256(args.protocol), audit_sha256=sha256(args.audit_pin),
            code_sha256=sha256(__file__), metadata_only=True, expression_read=False,
            roles=splits, matrices=matrices, files={p.name:sha256(p) for p in out.glob("*.tsv")}))
        out.rename(locked)


def provenance(args):
    access = args.run/"access"
    ref = read_json(access/"receipt.json")
    stamp = dict(access_sha256=sha256(access/"receipt.json"), code_sha256=sha256(__file__), protocol_sha256=sha256(args.protocol))
    if (stamp["access_sha256"] != sha256(args.access_pin) or not ref["metadata_only"] or ref["expression_read"]
            or any(ref[k] != v for k,v in stamp.items() if k!="access_sha256")):
        raise ValueError("The metadata-only JHH-7 authority changed")
    genes = parse_landmarks(pinned_bytes(access/"landmarks.tsv",ref["files"]["landmarks.tsv"])).pr_gene_id.to_numpy()
    cols = {}
    for kind in ("treated","controls"):
        file = kind+"_genes.tsv"
        table = read_pinned(access,file,ref["files"][file])
        if not np.array_equal(indices(table.gene_id),genes):
            raise ValueError("JHH-7 and donor landmark axes disagree")
        cols[kind] = indices(table.level3_gene_index)
    return ref, genes, cols, stamp


def read_plan(args, ref, phase, role):
    return {name.removesuffix(".tsv"):read_pinned(args.run/"access"/phase/role,name,h)
            for name,h in ref["roles"][role][phase].items()}


def predict(args, ref, genes, cols, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    block = read_plan(args,ref,"predict",args.role)
    fit, query, source = (block[n] for n in ("fit","query","source"))
    if (set(fit.scaffold) & set(query.scaffold) or set(fit.parent) & set(query.parent) or "level3_row_index" in query
            or not query.cell_id.eq("JHH7").all() or not fit.cell_id.eq("JHH5").all()
            or not source.view.eq("nonhepatic").all()):
        raise ValueError("JHH-7 target, its chemotypes, or target Level 3 indices intersect fitting")
    values = effect(source, block["source_controls"], ref, genes, cols, args.role)
    plates, pvals = group_means(values,source,["parent","cell_id","rna_plate"])
    cells, cvals = group_means(pvals,plates,["parent","cell_id"])
    registry, dmean = group_means(cvals,cells,["parent"])
    profiles = [cvals[ix] for ix in cells.groupby("parent",sort=True).indices.values()]
    shrunk = shrink_consensus(profiles,[len(a) for a in profiles],dmean)
    registry = pd.Index(registry.parent)
    fit_rows, truth = group_means(effect(fit,block["fit_controls"],ref,genes,cols,args.role),fit,["parent"])
    fit_index, test = (registry.get_indexer(a) for a in (fit_rows.parent,query.parent))
    wrong = np.empty(len(query),dtype=object)
    for (fold, plate), ix in query.groupby(["fold","rna_plate"]).indices.items():
        wrong[ix] = wrong_parent(query.iloc[ix],fold,plate)
    wrong_index = registry.get_indexer(wrong)
    if any((ix<0).any() for ix in (fit_index,test,wrong_index)) or (wrong_index==test).any():
        raise ValueError("Same input parent identifiers changed")
    x = dmean[fit_index]
    full = dmean[test]
    gain = scalar(fit_rows,x,truth)
    predicted = {"B0":np.zeros_like(full),MEAN:full,FIX:shrunk[test],SCALED:gain*full,SHUFFLE:dmean[wrong_index],
        STATE:state_forecast(load_network(args.state),(fit_rows,x,truth),full)}
    dest = args.run/"forecasts"/args.role
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=dest.parent,prefix=".forecast-") as temp:
        stage = Path(temp)
        matrix = stage/"predictions.npz"
        np.savez_compressed(matrix,**{k:checked_array(v,(len(query),len(genes)),k).astype("float32") for k,v in predicted.items()})
        write_json(stage/"receipt.json",dict(stamp,role=args.role,file_sha256=sha256(matrix),fit_parents=len(fit_rows),
            scalar_gain=gain, state_upstream=UPSTREAM,
            state_model_sha256=sha256(args.state/"src/state/tx/models/state_transition.py"),
            wrong_parent_mapping_sha256=hash_text("\n".join(wrong).encode()).hexdigest()))
        stage.rename(dest)


def finish(args, ref, genes, cols, stamp):
    score = args.phase=="score"
    freeze = args.run/"freeze.json"
    if score and (sha256(freeze)!=args.digest or (args.run/"score").exists()):
        raise ValueError("JHH-7 scoring requires its independent freeze")
    if not score and freeze.exists():
        raise FileExistsError(freeze)
    registry, queries, predictions, mappings = {}, {}, {}, set()
    for role in "ABC":
        folder = args.run/"forecasts"/role
        path = folder/"receipt.json"
        receipt = read_json(path)
        matrix = folder/"predictions.npz"
        query = read_pinned(args.run/"access"/"predict"/role,"query.tsv",ref["roles"][role]["predict"]["query.tsv"])
        if (any(receipt[k]!=v for k,v in stamp.items()) or receipt["role"]!=role or sha256(matrix)!=receipt["file_sha256"]
                or receipt["fit_parents"]!=475):
            raise ValueError("The JHH-7 frozen forecast differs from its metadata and fitting scope")
        registry[role] = sha256(path)
        queries[role] = query
        mappings.add(receipt["wrong_parent_mapping_sha256"])
        with np.load(matrix,allow_pickle=False) as data:
            values = {arm:checked_array(data[arm],(len(query),len(genes)),arm) for arm in ARMS}
        if score:
            predictions[role] = values
    rows = checked_rows(pd.concat([q.assign(role=role) for role,q in queries.items()],ignore_index=True),None)
    if (len(mappings)!=1 or rows.inst_id.nunique()!=99 or rows.parent.nunique()!=99
            or not rows.groupby("inst_id").role.nunique().eq(3).all() or rows.rna_plate.nunique()!=1):
        raise ValueError("JHH-7 physical support or wrong-parent role matching changed")
    locked = dict(stamp,roles=registry,parents=99,wells=99,plates=1)
    if not score:
        write_json(freeze,locked)
        return
    if read_json(freeze)!=locked:
        raise ValueError("JHH-7 predictions changed after freezing")
    outcomes = []
    for role,query in queries.items():
        plan = read_plan(args,ref,"score",role)
        target, dmso = plan["query"],plan["controls"]
        if not target.drop(columns="level3_row_index").equals(query):
            raise ValueError("JHH-7 scorer received a different physical row order")
        outcomes.append(effect(target,dmso,ref,genes,cols,"ABC"[("ABC".index(role)+2)%3]))
    y = np.concatenate(outcomes)
    arrays = {arm:np.concatenate([a[arm] for a in predictions.values()]) for arm in ARMS}
    truth, centered, hits = contrast(rows,y,{OP:arrays[FIX],SC:arrays[MEAN],ST:arrays[STATE]})
    centered = {FIX:centered[OP],MEAN:centered[SC],STATE:centered[ST]}
    mapping = rows[["parent","scaffold"]].drop_duplicates().set_index("parent",verify_integrity=True).scaffold.to_dict()
    clusters = []
    for arm,other in PAIRS:
        moments = parent_summaries(rows,y.shape[1],squared_rows(y),squared_rows(y,arrays[arm]),squared_rows(y,arrays[other]))
        bounds,relative = cluster_effects(sorted(mapping),mapping,*moments)
        for metric,(lo,hi) in zip(("rmse","r2_pooled"),bounds,strict=True):
            clusters.append(dict(arm=arm,reference=other,metric=metric,low99=lo,high99=hi,
                relative_low99=relative[0] if metric=="rmse" else np.nan,relative_high99=relative[1] if metric=="rmse" else np.nan))
    with TemporaryDirectory(dir=args.run,prefix=".jhh7-score-") as temp:
        out = Path(temp)
        tables = dict(paired=pd.DataFrame(compare(rows,y,arrays,PAIRS,"absolute")),
            centered=pd.DataFrame(compare(rows,truth,centered,((FIX,MEAN),(MEAN,STATE)),"centered")),
            retrieval=retrieval(hits).replace({OP:FIX,SC:MEAN,ST:STATE}),scaffold=pd.DataFrame(clusters))
        for name,table in tables.items():
            table.to_csv(out/(name+".tsv"),sep="\t",index=False,na_rep="NA")
        write_json(out/"receipt.json",dict(locked,freeze_sha256=args.digest,
            original_JHH5_parent_overlap=int(rows.loc[rows.in_JHH5.eq("True"),"parent"].nunique()),
            files={file.name:sha256(file) for file in out.iterdir()}))
        out.rename(args.run/"score")


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=("lock","predict","freeze","score"))
    for field in ("audit","audit-pin","cmap-pin","landmarks","protocol","run","access-pin","state"):
        cli.add_argument("--"+field,type=Path,required=True)
    cli.add_argument("--role",choices=list("ABC"))
    cli.add_argument("--digest")
    args = cli.parse_args()
    if args.phase=="lock":
        lock(args)
    else:
        (predict if args.phase=="predict" else finish)(args,*provenance(args))
