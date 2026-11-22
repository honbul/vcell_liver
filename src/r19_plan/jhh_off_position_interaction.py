"""Replay the matched adult HCC forecasts as a paired between-context contrast."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r19_plan.hcc_lincs_24h_access import CONTEXT
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.jhh5_donor_attribution import wrong_parent
from src.r19_plan.jhh5_off_position import CALIBRATED, SCALAR
from src.r19_plan.jhh5_off_position_native import references as five_ref
from src.r19_plan.jhh7_donor_transfer import read_plan
from src.r19_plan.jhh7_off_position_native import REFERENCES, best_comparison, references as seven_ref
from src.r19_plan.jhh_off_position_robustness import frozen_vectors
from src.r19_plan.t4_position import compare


PLATE = "ASG002_JHH5_24H_X1"
LINEUP = (CALIBRATED, SCALAR, *REFERENCES)


def binding(args, cell):
    lower = cell.lower()
    by_cell = {"JHH5": ("jhh5-off-position.0ldmuIkJ", "jhh5-off-native.xTAKElO6"),
               "JHH7": ("jhh7-off-position.GMpuNJHy", "jhh7-off-native.HAX62Mi9")}
    previous, current = (args.server/name for name in by_cell[cell])
    data = args.reports/"data"/(lower+"_off_position")
    native = args.reports/"data"/(lower+"_off_position_native")
    return SimpleNamespace(cell=cell, run=current, previous=previous, protocol=args.protocol,
        previous_protocol=args.reports/(cell+"_OFF_POSITION_PROTOCOL.md"), previous_pin=data/"freeze.json",
        native_protocol=args.reports/(cell+"_OFF_POSITION_NATIVE_PROTOCOL.md"), native_pin=native/"freeze.json",
        score_pin=native/"score/receipt.json", access_pin=data/"access/receipt.json")


def matching(five, seven):
    _, ref5, genes5, cols5, _, stamp = five_ref(SimpleNamespace(**(vars(five)|{"protocol":five.native_protocol})))
    _, ref7, genes7, cols7, _, stamp7 = seven_ref(SimpleNamespace(**(vars(seven)|{"protocol":seven.native_protocol})))
    if (ref5["matrices"] != ref7["matrices"] or not np.array_equal(genes5, genes7)
            or any(not np.array_equal(cols5[k], cols7[k]) for k in cols5)):
        raise ValueError("The adult HCC measurements require the same assay and aligned landmark order")
    rows5, y5, predictions5 = frozen_vectors(five)
    rows7, y7, predictions7 = frozen_vectors(seven)
    search5 = rows5.loc[rows5.rna_plate.eq(PLATE)]
    common = set(search5.parent) & set(rows7.parent)
    chosen = rows7.loc[rows7.parent.isin(common)].sort_values(["parent","role"])
    a = search5.set_index(["parent","role"], verify_integrity=True)
    ids = pd.MultiIndex.from_frame(chosen[["parent","role"]])
    indexes = pd.Series(search5.index, index=a.index).loc[ids].to_numpy()
    matched = rows5.loc[indexes]
    if (len(common)!=86 or len(indexes)!=3*len(common) or not chosen.parent.eq(matched.parent.to_numpy()).all()
            or not chosen.fold.eq(matched.fold.to_numpy()).all() or not chosen.scaffold.eq(matched.scaffold.to_numpy()).all()
            or not chosen.groupby("parent").role.nunique().eq(3).all()):
        raise ValueError("The frozen physical JHH-5 and JHH-7 pairs changed")
    index7 = chosen.index.to_numpy()
    difference = {arm: predictions7[arm][index7]-predictions5[arm][indexes] for arm in LINEUP}
    chosen = chosen.assign(jhh5_well=matched.rna_well.to_numpy()).reset_index(drop=True)
    return chosen, y5[indexes], y7[index7], difference, stamp, stamp7


def source_support(five, seven, common):
    five = SimpleNamespace(**(vars(five)|{"protocol":five.native_protocol}))
    seven = SimpleNamespace(**(vars(seven)|{"protocol":seven.native_protocol}))
    old5, ref5, _, _, _, _ = five_ref(five)
    old7, ref7, _, _, _, _ = seven_ref(seven)
    reports = []
    for role in "ABC":
        directory = old5.access/"donor-plan"/role
        recipe = ref5["donors"][role]
        d5,c5 = (read_pinned(directory,name,recipe[name]) for name in ("source.tsv","controls.tsv"))
        plan = read_plan(old7,ref7,"predict",role)
        d7,c7 = plan["source"],plan["source_controls"]
        left,right = ({p:d for p,d in df.loc[df.parent.isin(common)].groupby("parent")} for df in (d5,d7))
        for p in sorted(common):
            da,db = left[p],right[p]
            same = set(da.inst_id)==set(db.inst_id)
            reports.append(dict(parent=p,role=role,common_source_cells=len(set(da.cell_id)&set(db.cell_id)),
                identical_treated_wells=same, identical_source_vehicles=(
                    set(c5.merge(da[CONTEXT].drop_duplicates(),on=CONTEXT,validate="many_to_one").inst_id)
                    == set(c7.merge(db[CONTEXT].drop_duplicates(),on=CONTEXT,validate="many_to_one").inst_id))))
    return pd.DataFrame(reports)


def centered(rows, vector):
    out = vector.astype("float64")
    for ids in rows.groupby(["fold","role"]).indices.values():
        if len(ids)<5:
            raise ValueError("A registered paired contrast lacks five compounds")
        out[ids] -= out[ids].mean(axis=0)
    return out


def wrong_contrast(rows, forecast):
    out = np.empty_like(forecast)
    for (fold,_), ids in rows.groupby(["fold","role"]).indices.items():
        block = rows.iloc[ids]
        swapped = wrong_parent(block,fold,PLATE)
        index = pd.Series(ids,index=block.parent).loc[swapped].to_numpy()
        out[ids]=forecast[index]
    return out


def evaluate(rows,y5,y7,predictions,support):
    scores,best,similarity = [],[],[]
    all_sizes = []
    sets = [("all",np.ones(len(rows),dtype=bool))]
    keep=rows.parent.isin(support)
    eligible = keep.sum()//3 >= 20 and rows.loc[keep].groupby("fold").parent.nunique().reindex([0,1,2],fill_value=0).min()>=5
    all_sizes.append(dict(scope="identical_source",parents=int(keep.sum()//3),roles=3,scored=bool(eligible)))
    if eligible:
        sets.append(("identical_source",keep.to_numpy()))
    for scope,active in sets:
        block = rows.loc[active].reset_index(drop=True)
        for co in ("raw", "centered"):
            x,y = (centered(block,v[active]) if co=="centered" else v[active].astype(float) for v in (y5,y7))
            truth=y-x
            arms = {arm:centered(block,v[active]) if co=="centered" else v[active] for arm,v in predictions.items()}
            arms.update(B0=np.zeros_like(truth),reversed_context=-arms[CALIBRATED],
                        wrong_parent=wrong_contrast(block,arms[CALIBRATED]))
            name=scope+"_"+co
            pairs=tuple((arm,"B0") for arm in LINEUP) + tuple((CALIBRATED,r) for r in (SCALAR,"reversed_context","wrong_parent"))
            scores.extend(compare(block,truth,arms,pairs,name))
            result,_ = best_comparison(block,truth,arms,name)
            best.extend(result)
            norms=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
            cosine=np.divide(np.sum(x*y,axis=1),norms,out=np.full(len(x),np.nan),where=norms>0)
            def power(m): return np.square(m).mean()
            similarity.append(dict(scope=name,parents=block.parent.nunique(),
                observed_between_cell_cosine=pd.Series(cosine).groupby(block.parent).mean().mean(),
                observed_between_cell_delta_rms=np.sqrt(power(truth)),
                candidate_between_cell_delta_rms=np.sqrt(power(arms[CALIBRATED])),
                observed_jhh5_rms=np.sqrt(power(x)),observed_jhh7_rms=np.sqrt(power(y))))
    return dict(paired=scores,best=best,similarity=similarity,restricted_support=all_sizes)


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    five,seven = (binding(args,c) for c in ("JHH5","JHH7"))
    rows,y5,y7,delta,p5,p7 = matching(five,seven)
    report=source_support(five,seven,set(rows.parent))
    identical=(report.identical_treated_wells & report.identical_source_vehicles).groupby(report.parent).all()
    same_target=rows.drop_duplicates("parent")
    audit=dict(parents=len(same_target),same_target_well_index=int(same_target.rna_well.eq(same_target.jhh5_well).sum()),
        exact_both_source_and_vehicle_parents=int(identical.sum()),
        minimum_common_donor_cells=int(report.common_source_cells.min()),
        median_common_donor_cells=float(report.common_source_cells.median()),
        maximum_common_donor_cells=int(report.common_source_cells.max()),jhh5_plates=1,jhh7_plates=1)
    report_tables=evaluate(rows,y5,y7,delta,set(identical.index[identical]))
    report_tables["audit"]=[audit]
    report_tables["fold_support"]=[dict(fold=int(fold),parents=group.parent.nunique(),
         exact_source_and_vehicle_parents=group.parent.isin(identical.index[identical]).sum()//3)
         for fold,group in rows.groupby("fold")]
    with TemporaryDirectory(dir=args.out.parent,prefix=".joint-cell-") as temporary:
        stage=Path(temporary)
        for name,record in report_tables.items():
            pd.DataFrame(record).to_csv(stage/(name+".tsv"),sep="\t",index=False,na_rep="NA")
        write_json(stage/"receipt.json",dict(previous_score_replayed=True,protocol_sha256=sha256(args.protocol),
            code_sha256=sha256(__file__),JHH5=dict(p5,native_freeze=sha256(five.native_pin)),
            JHH7=dict(p7,native_freeze=sha256(seven.native_pin)),
            files={p.name:sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("server","reports","protocol","out"):
        cli.add_argument("--"+name,type=Path,required=True)
    run(cli.parse_args())
