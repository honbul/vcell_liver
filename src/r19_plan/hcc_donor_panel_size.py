"""Score frozen, nested donor-cell reductions in the two CMap adult HCC development cohorts."""

import argparse
from hashlib import sha256 as hash_text
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_access import CONTEXT
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import BOOTSTRAPS, METRICS, checked_array, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.jhh5_donor_attribution import FIX, MEAN, setup, shrink_consensus
from src.r19_plan.jhh7_donor_transfer import provenance, read_plan
from src.r19_plan.t4_workflow import axes


COUNTS = (2, 4, 8, 16)
SEEDS = tuple(range(8))
CELLS = ("JHH5", "JHH7")
METHODS = ("mean", "shrink")
FULL = {"mean": MEAN, "shrink": FIX}


def contexts(args):
    pins = args.reports/"data"
    prior5 = SimpleNamespace(previous=args.jhh5, access_pin=pins/"hcc_lincs24/access/receipt.json",
        previous_protocol=args.reports/"HCC_LINCS_24H_TRANSFER_PROTOCOL.md",
        previous_pin=pins/"hcc_lincs24/transfer/freeze.json", protocol=args.reports/"JHH5_DONOR_ATTRIBUTION_PROTOCOL.md")
    five = setup(prior5)
    prior7 = SimpleNamespace(run=args.jhh7, access_pin=pins/"jhh7_donor_transfer/access/receipt.json",
        protocol=args.reports/"JHH7_DONOR_TRANSFER_PROTOCOL.md")
    ref, genes, cols, stamp = provenance(prior7)
    seven = SimpleNamespace(ref=ref, genes=genes, cols=cols, stamp=stamp)
    if five.ref["matrices"] != ref["matrices"] or not np.array_equal(five.genes, genes) or any(
            not np.array_equal(five.cols[k], cols[k]) for k in cols):
        raise ValueError("The two adult HCC experiments have different donor matrix or gene support")
    sources = {
        "JHH5": (args.attribution, five, pins/"jhh5_donor_attribution/freeze.json"),
        "JHH7": (args.jhh7, seven, pins/"jhh7_donor_transfer/freeze.json"),
    }
    proven = {}
    for cell, (run, scope, pin) in sources.items():
        actual = run/"freeze.json"
        if sha256(actual) != sha256(pin) or any(read_json(pin)[k] != v for k,v in scope.stamp.items()):
            raise ValueError(f"The frozen {cell} comparison differs")
        proven[cell] = sha256(actual)
    new = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__), previous_freezes=proven)
    return five, seven, new


def average_by_cell(args, five, seven, role):
    donor_dir = five.previous.access/"donor-plan"/role
    recipe = five.ref["donors"][role]
    d5, c5 = (read_pinned(donor_dir,name,recipe[name]) for name in ("source.tsv","controls.tsv"))
    held = read_plan(SimpleNamespace(run=args.jhh7), seven.ref, "predict", role)
    wanted = set(held["query"].parent)
    d7 = held["source"].loc[held["source"].parent.isin(wanted)]
    c7 = held["source_controls"].merge(d7[CONTEXT].drop_duplicates(), on=CONTEXT, validate="many_to_one")
    fields = [*CONTEXT, "inst_id", "rna_well", "level3_row_index"]
    donor_fields, control_fields = [*fields, "parent"], [*fields, "role"]
    treatment = pd.concat((d5[donor_fields], d7[donor_fields]), ignore_index=True).drop_duplicates()
    controls = pd.concat((c5[control_fields], c7[control_fields]), ignore_index=True).drop_duplicates()
    values = effect(treatment, controls, five.ref, five.genes, five.cols, role)
    plate, pvals = group_means(values, treatment, ["parent","cell_id","rna_plate"])
    return group_means(pvals, plate, ["parent","cell_id"])


def choices(subject, cells, seed):
    ranks = [hash_text(f"HCC-donor-panel-v1|{seed}|{subject}|{name}".encode()).digest() for name in cells]
    return sorted(range(len(cells)), key=ranks.__getitem__)[:COUNTS[-1]]


def panel_profiles(rows, vectors):
    sets, manifests = {}, {}
    for subject, ix in rows.groupby("parent", sort=True).indices.items():
        cells = rows.cell_id.iloc[ix].tolist()
        values = vectors[ix]
        if len(cells) < 16:
            continue
        result = {kind:{n:[] for n in COUNTS} for kind in METHODS}
        picked = []
        for seed in SEEDS:
            order = choices(subject,cells,seed)
            picked.append(tuple(cells[i] for i in order))
            for count in COUNTS:
                batch = values[order[:count]]
                mean = batch.mean(axis=0)
                shrink = shrink_consensus([batch],[count],mean[None,:])[0]
                result["mean"][count].append(mean)
                result["shrink"][count].append(shrink)
        sets[subject] = {f"{kind}_{count}":np.stack(profiles).astype("float32")
            for kind in METHODS for count,profiles in result[kind].items()}
        manifests[subject] = picked
    return sets, hash_text(repr(manifests).encode()).hexdigest()


def original_slices(args, five, seven, role):
    path = args.attribution/"forecasts"/role
    pin = read_json(args.attribution/"freeze.json")
    receipt = read_json(path/"receipt.json")
    if sha256(path/"receipt.json") != pin["roles"][role]:
        raise ValueError("The frozen JHH-5 role receipt changed")
    for key,h in sorted(receipt["slices"].items()):
        name = "query.tsv"
        q = read_pinned(five.previous.access/"predict-plan"/key, name, five.ref["slices"][key]["predict"][name])
        yield "JHH5", key, q, path/(key+".npz"), h
    path = args.jhh7/"forecasts"/role
    pin = read_json(args.jhh7/"freeze.json")
    receipt = read_json(path/"receipt.json")
    if sha256(path/"receipt.json") != pin["roles"][role]:
        raise ValueError("The frozen JHH-7 role receipt changed")
    name = "query.tsv"
    q = read_pinned(args.jhh7/"access"/"predict"/role, name, seven.ref["roles"][role]["predict"][name])
    yield "JHH7", role, q, path/"predictions.npz", receipt["file_sha256"]


def predict(args, five, seven, stamp):
    cells, vectors = average_by_cell(args,five,seven,args.role)
    profiles, selections = panel_profiles(cells,vectors)
    groups = cells.groupby("parent",sort=True).indices
    out = args.run/"forecasts"/args.role
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent,prefix=".panel-forecast-") as tmp:
        stage = Path(tmp)
        slices = {}
        for cell,key,query,original,digest in original_slices(args,five,seven,args.role):
            if sha256(original) != digest:
                raise ValueError("The prior adult HCC full-source array changed")
            with np.load(original,allow_pickle=False) as archive:
                full = {arm:checked_array(archive[arm],(len(query),len(five.genes)),arm) for arm in FULL.values()}
            eligible = query.parent.isin(profiles).to_numpy()
            blocks = [vectors[groups[p]] for p in query.parent]
            mean = np.stack([b.mean(axis=0) for b in blocks])
            shrink = shrink_consensus(blocks,[len(b) for b in blocks],mean)
            if not np.array_equal(mean.astype("float32"),full[MEAN]) or not np.array_equal(shrink.astype("float32"),full[FIX]):
                raise ValueError("Original full donor forecasts were not reproduced")
            selected = query.parent.loc[eligible]
            panels = {f"{kind}_{count}":np.stack([profiles[p][f"{kind}_{count}"] for p in selected])
                for kind in METHODS for count in COUNTS}
            dest = stage/f"{cell}-{key}.npz"
            np.savez_compressed(dest,eligible=eligible,**{k:v[eligible] for k,v in full.items()},**panels)
            slices[f"{cell}-{key}"] = sha256(dest)
        write_json(stage/"receipt.json",dict(stamp,role=args.role,subset_sha256=selections,slices=slices))
        stage.rename(out)


def read_truth(args, five, seven, cell, key, role, query):
    if cell=="JHH5":
        plan = five.ref["slices"][key]["score"]
        directory = five.previous.access/"score-plan"/key
        target, control = (read_pinned(directory,n,plan[n]) for n in ("query.tsv","score_controls.tsv"))
        ref, cols, label = five.ref, five.cols, axes(key)[3]
    else:
        plan = read_plan(SimpleNamespace(run=args.jhh7),seven.ref,"score",role)
        target,control = plan["query"],plan["controls"]
        ref,cols,label = seven.ref,seven.cols,"ABC"[("ABC".index(role)+2)%3]
    if not target.drop(columns="level3_row_index").equals(query):
        raise ValueError("The scored physical samples differ from the forecast support")
    return effect(target,control,ref,five.genes,cols,label)


def moments(y, forecast):
    if forecast.ndim==2:
        error = squared_rows(y,forecast)
        return np.sqrt(error/y.shape[1]), error
    by_seed = np.stack([squared_rows(y,forecast[:,seed]) for seed in SEEDS])
    return np.sqrt(by_seed/y.shape[1]).mean(axis=0), by_seed.mean(axis=0)


def contrast_table(rows, y, predictions, cell, scope="all"):
    choices_to_score = []
    for kind in METHODS:
        for n in COUNTS:
            arm = f"{kind}_{n}"
            choices_to_score.extend((arm, ref) for ref in ("B0",FULL[kind]))
            if n!=16:
                choices_to_score.append((arm,f"{kind}_16"))
            if kind=="shrink":
                choices_to_score.append((arm,f"mean_{n}"))
    energy = squared_rows(y)
    errors = {arm:moments(y,value) for arm,value in predictions.items()}
    results = []
    for arm, ref in choices_to_score:
        a,b = (errors[k] for k in (arm,ref))
        _,sse,energy_sums = parent_summaries(rows, y.shape[1], energy, a[1], b[1])
        # A whole panel is sampled for a prediction. Average sqrt(SSE) over panels before parent aggregation.
        by_parent = pd.DataFrame({"parent":rows.parent,"arm":a[0],"reference":b[0]}).groupby("parent",sort=True).mean()
        rmse = by_parent.to_numpy()
        points, bounds, n = paired_effects(rmse,sse,energy_sums)
        relative_lo,relative_hi,_ = relative_interval(rmse)
        for metric,(a_point,b_point),(lo,hi),draws in zip(METRICS,points,bounds,n,strict=True):
            is_rmse = metric=="rmse"
            results.append(dict(cell=cell,scope=scope,parents=rows.parent.nunique(),wells=rows.inst_id.nunique(),
                arm=arm,relative_to=ref,metric=metric,arm_value=a_point,reference_value=b_point,difference=a_point-b_point,
                low99=lo,high99=hi,relative_gain=1-a_point/b_point if is_rmse else np.nan,
                relative_low99=relative_lo if is_rmse else np.nan,relative_high99=relative_hi if is_rmse else np.nan,draws=draws))
    return results


def seed_points(rows,y,predictions,cell):
    energy = squared_rows(y)
    result = []
    for kind in METHODS:
        for count in COUNTS:
            profiles = predictions[f"{kind}_{count}"]
            for seed in SEEDS:
                error = squared_rows(y,profiles[:,seed])
                by_parent = pd.DataFrame({"parent":rows.parent,"error":np.sqrt(error/y.shape[1])})
                result.append(dict(cell=cell,method=kind,donor_cells=count,seed=seed,
                    rmse=by_parent.groupby("parent").error.mean().mean(),r2_pooled=1-error.sum()/energy.sum()))
    return result


def finish(args, five, seven, stamp):
    freeze = args.run/"freeze.json"
    scoring = args.phase=="score"
    if scoring and (sha256(freeze)!=args.digest or (args.run/"score").exists()):
        raise ValueError("Donor panel scoring needs the committed freeze")
    if not scoring and freeze.exists():
        raise FileExistsError(freeze)
    roles, selections = {}, set()
    data = {cell:[] for cell in CELLS}
    arms = (*FULL.values(),*(f"{kind}_{n}" for kind in METHODS for n in COUNTS))
    for role in "ABC":
        root = args.run/"forecasts"/role
        receipt = read_json(root/"receipt.json")
        if receipt["role"]!=role or any(receipt[k]!=v for k,v in stamp.items()):
            raise ValueError("A panel forecast receipt differs from the fixed source or role")
        roles[role] = sha256(root/"receipt.json")
        selections.add(receipt["subset_sha256"])
        expected = set()
        for cell,key,query,_,_ in original_slices(args,five,seven,role):
            label = f"{cell}-{key}"
            expected.add(label)
            path = root/f"{label}.npz"
            if sha256(path)!=receipt["slices"][label]:
                raise ValueError("A fixed donor panel prediction changed")
            with np.load(path,allow_pickle=False) as file:
                keep = file["eligible"]
                if len(keep)!=len(query) or keep.dtype.kind!="b" or not keep.any():
                    raise ValueError("Changed matched donor panel availability")
                shape = (int(keep.sum()), len(five.genes))
                predictions = {a:checked_array(file[a],shape if a in FULL.values() else (shape[0],len(SEEDS),shape[1]),a)
                    for a in arms}
            data[cell].append((role,key,query,keep,predictions if scoring else None))
        if set(receipt["slices"])!=expected:
            raise ValueError("Fixed donor panel slices differ")
    if len(selections)!=1:
        raise ValueError("Selected donor cell identities differ across DMSO roles")

    rows_by_cell = {}
    for cell,pieces in data.items():
        rows = checked_rows(pd.concat([q.loc[keep].assign(role=role) for role,_,q,keep,_ in pieces],ignore_index=True),None)
        if not rows.groupby("inst_id").role.nunique().eq(3).all() or rows.parent.nunique()<50:
            raise ValueError("Insufficient common donor panel support")
        rows_by_cell[cell] = rows
    support = {cell:dict(parents=rows.parent.nunique(),wells=rows.inst_id.nunique(),plates=rows.rna_plate.nunique())
        for cell,rows in rows_by_cell.items()}
    manifest = dict(stamp,roles=roles,selection_sha256=next(iter(selections)),support=support)
    if not scoring:
        write_json(freeze,manifest)
        return
    if read_json(freeze)!=manifest:
        raise ValueError("Donor panel compare changed after freeze")

    paired,draws,leave = [],[],[]
    for cell,pieces in data.items():
        rows = rows_by_cell[cell]
        y = np.concatenate([read_truth(args,five,seven,cell,key,role,q)[keep] for role,key,q,keep,_ in pieces])
        predictions = {a:np.concatenate([item[-1][a] for item in pieces]) for a in arms}
        predictions["B0"] = np.zeros_like(y)
        paired.extend(contrast_table(rows,y,predictions,cell))
        draws.extend(seed_points(rows,y,predictions,cell))
        if cell=="JHH5":
            for plate in sorted(rows.rna_plate.unique()):
                keep = rows.rna_plate.ne(plate)
                leave.extend(contrast_table(rows.loc[keep].reset_index(drop=True),y[keep],
                    {a:arr[keep] for a,arr in predictions.items()},cell,scope=plate))
    with TemporaryDirectory(dir=args.run,prefix=".panel-score-") as tmp:
        stage = Path(tmp)
        for name,table in (("paired",paired),("panel_seeds",draws),("leave_plate",leave)):
            pd.DataFrame(table).to_csv(stage/f"{name}.tsv",sep="\t",index=False,na_rep="NA")
        write_json(stage/"receipt.json",dict(manifest,freeze_sha256=args.digest,bootstrap_draws=BOOTSTRAPS,
            files={file.name:sha256(file) for file in stage.iterdir()}))
        stage.rename(args.run/"score")


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=("predict","freeze","score"))
    for field in ("reports","jhh5","attribution","jhh7","protocol","run"):
        cli.add_argument("--"+field,type=Path,required=True)
    cli.add_argument("--role",choices=list("ABC"))
    cli.add_argument("--digest")
    args = cli.parse_args()
    (predict if args.phase=="predict" else finish)(args,*contexts(args))
