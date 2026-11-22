"""Compare equal-information State and fit-free shrinkage on locked eight-cell donor panels."""

import argparse
from hashlib import sha256 as hash_text
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_donor_panel_size import SEEDS, contexts, moments, original_slices, read_truth
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import METRICS, checked_array, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.jhh7_donor_transfer import read_plan
from src.r19_plan.t4_workflow import axes, scalar


STATE, SCALAR = "State_8", "scalar_8"
SHRINK, MEAN = "shrink_8", "mean_8"
COMPARISONS = ((SHRINK, STATE), (SHRINK, SCALAR), (SHRINK, MEAN), (SHRINK, "B0"), (STATE, MEAN))


def lineage(args):
    old_protocol = args.reports/"HCC_DONOR_PANEL_SIZE_PROTOCOL.md"
    earlier = SimpleNamespace(reports=args.reports,jhh5=args.jhh5,jhh7=args.jhh7,attribution=args.attribution,protocol=old_protocol)
    five, seven, earlier_stamp = contexts(earlier)
    pin = args.reports/"data/hcc_donor_panel_size/freeze.json"
    frozen = read_json(pin)
    if sha256(args.panel/"freeze.json") != sha256(pin) or any(frozen[k]!=v for k,v in earlier_stamp.items()):
        raise ValueError("Frozen eight-line panels have a different lineage")
    return five, seven, frozen, dict(panel_freeze_sha256=sha256(pin),protocol_sha256=sha256(args.protocol),
        code_sha256=sha256(__file__))


def panels(args, five, seven, frozen, role):
    folder = args.panel/"forecasts"/role
    receipt = read_json(folder/"receipt.json")
    if sha256(folder/"receipt.json") != frozen["roles"][role]:
        raise ValueError("The fixed panel arrays have a different role receipt")
    found = set()
    for cell,key,query,_,_ in original_slices(args,five,seven,role):
        label = f"{cell}-{key}"
        found.add(label)
        file = folder/f"{label}.npz"
        h = receipt["slices"][label]
        if sha256(file)!=h:
            raise ValueError("The fixed eight-source forecast differs from its freeze")
        with np.load(file,allow_pickle=False) as block:
            eligible = block["eligible"]
            if eligible.shape!=(len(query),) or eligible.dtype.kind!="b":
                raise ValueError("Changed eligibility on an eight-line query slice")
            shape = (int(eligible.sum()),len(SEEDS),len(five.genes))
            predictions = {name:checked_array(block[name],shape,name) for name in (MEAN,SHRINK)}
        yield cell,key,query,eligible,predictions,h
    if found!=set(receipt["slices"]):
        raise ValueError("The frozen panel slices changed")


def fit_plan(args, five, seven, cell, key, role):
    if cell=="JHH5":
        recipe = five.ref["slices"][key]["predict"]
        root = five.previous.access/"predict-plan"/key
        return tuple(read_pinned(root,name,recipe[name]) for name in ("fit.tsv","fit_controls.tsv"))
    plan = read_plan(SimpleNamespace(run=args.jhh7),seven.ref,"predict",role)
    return plan["fit"],plan["fit_controls"]


def panel_offsets(parents):
    return np.asarray([int.from_bytes(hash_text(p.encode()).digest()[:4],"big")%len(SEEDS) for p in parents])


def state_panel_forecast(network, fit, panels, y, query):
    import torch
    from src.goal_tracks.t2_state import build_network, forward

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = build_network(network,y.shape[1],1,cell_set_len=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    predictor = torch.as_tensor(panels,dtype=torch.float32,device=device)
    target = torch.as_tensor(y,dtype=torch.float32,device=device)
    offsets = torch.as_tensor(panel_offsets(fit.parent),dtype=torch.long,device=device)
    sample = torch.arange(len(fit),device=device)
    ones = torch.ones((len(fit),1),device=device)
    for epoch in range(40):
        model.train()
        x = predictor[sample,(offsets+epoch)%len(SEEDS),None,:]
        for index in torch.randperm(len(fit),device=device).split(128):
            predicted = forward(model,x[index],ones[index])
            loss = (predicted-target[index]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    q = torch.as_tensor(query.reshape(-1,y.shape[1]),dtype=torch.float32,device=device)
    with torch.inference_mode():
        blocks = [forward(model,b[:,None,:],torch.ones((len(b),1),device=device)).cpu().numpy() for b in q.split(128)]
    return np.concatenate(blocks).reshape(query.shape)


def fit_predictions(args, five, seven, frozen, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    group = args.group
    evaluation_cell, fold, role = group.split("-")
    available = list(panels(args,five,seven,frozen,role))
    # Only donor input means are written to this registry. Target-fit RNA is loaded by slice below.
    donors = {}
    for cell,_,query,eligible,predictions,_ in available:
        if cell!="JHH5":
            continue
        for p,value in zip(query.loc[eligible,"parent"],predictions[MEAN],strict=True):
            if not np.array_equal(donors.setdefault(p,value),value):
                raise ValueError("A donor's input changed across its physical target wells")
    if len(donors)!=frozen["support"]["JHH5"]["parents"]:
        raise ValueError("The matched JHH-5 fit vector population changed")

    dest = args.run/"forecasts"/group
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    network = load_network(args.state)
    with TemporaryDirectory(dir=dest.parent,prefix=".donor8-state-") as folder:
        out = Path(folder)
        items, fit_stats = {}, []
        for cell,key,query,eligible,predictions,old_h in available:
            if cell!=evaluation_cell or (cell=="JHH5" and axes(key)[0]!=int(fold)):
                continue
            block,controls = fit_plan(args,five,seven,cell,key,role)
            block = block.loc[block.parent.isin(donors)]
            controls = controls.loc[controls.rna_plate.isin(block.rna_plate)]
            if (set(block.parent)&set(query.parent) or set(block.scaffold)&set(query.scaffold)
                    or set(block.rna_plate)&set(query.rna_plate) or not block.cell_id.eq("JHH5").all()
                    or not query.cell_id.eq(cell).all() or "level3_row_index" in query or block.parent.nunique()<100):
                raise ValueError("The matched State fit intersects the scored cell, scaffold, plate, or parent")
            labels,y = group_means(effect(block,controls,five.ref,five.genes,five.cols,role),block,["parent"])
            x = np.stack([donors[p] for p in labels.parent])
            repeated = labels.iloc[np.repeat(np.arange(len(labels)),len(SEEDS))].reset_index(drop=True)
            gain = scalar(repeated,x.reshape(-1,len(five.genes)),np.repeat(y,len(SEEDS),axis=0))
            query_x = predictions[MEAN]
            state = state_panel_forecast(network,labels,x,y,query_x)
            new = out/f"{key}.npz"
            np.savez_compressed(new,**{STATE:checked_array(state,query_x.shape,STATE),
                SCALAR:checked_array((gain*query_x).astype("float32"),query_x.shape,SCALAR)})
            items[key] = dict(array_sha256=sha256(new),input_panel_sha256=old_h)
            fit_stats.append(dict(slice=key,cell=cell,role=role,fit_parents=len(labels),fit_wells=len(block),scalar_gain=gain,
                query_wells=int(eligible.sum())))
            print(f"{cell} {key}: {len(labels)} fit parents; {eligible.sum()} query wells",flush=True)
        if not items:
            raise ValueError("No matched forecast slices were found")
        network_source = args.state/"src/state/tx/models/state_transition.py"
        write_json(out/"receipt.json",dict(stamp,group=group,slices=items,fit=fit_stats,
            state_upstream=UPSTREAM,state_model_sha256=sha256(network_source)))
        out.rename(dest)


def paired(rows, y, predictions, cell, scope):
    errors = {name:moments(y,values) for name,values in predictions.items()}
    energy = squared_rows(y)
    result = []
    for arm, reference in COMPARISONS:
        a,b = errors[arm],errors[reference]
        _,sse,e = parent_summaries(rows,y.shape[1],energy,a[1],b[1])
        rmse = pd.DataFrame(dict(parent=rows.parent,arm=a[0],reference=b[0])).groupby("parent",sort=True).mean().to_numpy()
        points,intervals,counts = paired_effects(rmse,sse,e)
        relative_lo,relative_hi,_ = relative_interval(rmse)
        for metric,(p,q),(lo,hi),n in zip(METRICS,points,intervals,counts,strict=True):
            rms = metric=="rmse"
            result.append(dict(cell=cell,scope=scope,parents=rows.parent.nunique(),wells=rows.inst_id.nunique(),
                arm=arm,reference=reference,metric=metric,arm_value=p,reference_value=q,difference=p-q,low99=lo,high99=hi,
                relative_gain=1-p/q if rms else np.nan,relative_low99=relative_lo if rms else np.nan,
                relative_high99=relative_hi if rms else np.nan,draws=n))
    return result


def finish(args, five, seven, frozen, stamp):
    freeze = args.run/"freeze.json"
    score = args.phase=="score"
    if score and (sha256(freeze)!=args.digest or (args.run/"score").exists()):
        raise ValueError("Matched eight-line State scoring requires the independent freeze")
    if not score and freeze.exists():
        raise FileExistsError(freeze)
    hashes, fits, parsed = {}, {}, {"JHH5":[],"JHH7":[]}
    for role in "ABC":
        for cell,key,q,keep,panel,old_h in panels(args,five,seven,frozen,role):
            group = f"{cell}-{axes(key)[0] if cell=='JHH5' else 'all'}-{role}"
            folder = args.run/"forecasts"/group
            receipt_path = folder/"receipt.json"
            receipt = read_json(receipt_path)
            actual = folder/f"{key}.npz"
            expected = receipt["slices"][key]
            if (receipt["group"]!=group or any(receipt[k]!=v for k,v in stamp.items())
                    or expected["input_panel_sha256"]!=old_h or expected["array_sha256"]!=sha256(actual)):
                raise ValueError("The same-information State predictions or inputs changed")
            hashes[group] = sha256(receipt_path)
            fits[group] = receipt["fit"]
            with np.load(actual,allow_pickle=False) as file:
                new = {name:checked_array(file[name],panel[MEAN].shape,name) for name in (STATE,SCALAR)}
            parsed[cell].append((key,role,q,keep,{**panel,**new} if score else None))
    scoped = {}
    for cell,pieces in parsed.items():
        rows = checked_rows(pd.concat([q.loc[keep].assign(role=role) for _,role,q,keep,_ in pieces],ignore_index=True),None)
        support = dict(parents=rows.parent.nunique(),wells=rows.inst_id.nunique(),plates=rows.rna_plate.nunique())
        if support!=frozen["support"][cell] or not rows.groupby("inst_id").role.nunique().eq(3).all():
            raise ValueError("The forecast does not cover the same three roles and physical wells")
        scoped[cell] = rows
    manifest = dict(stamp,groups=hashes,support=frozen["support"])
    if not score:
        write_json(freeze,manifest)
        return
    if read_json(freeze)!=manifest:
        raise ValueError("Matched State predictions changed after freeze")

    result = {"paired":[],"folds":[],"leave_plate":[]}
    for cell,pieces in parsed.items():
        rows = scoped[cell]
        truth = np.concatenate([read_truth(args,five,seven,cell,key,role,q)[keep] for key,role,q,keep,_ in pieces])
        arrays = {name:np.concatenate([piece[-1][name] for piece in pieces]) for name in (STATE,SCALAR,MEAN,SHRINK)}
        arrays["B0"] = np.zeros_like(truth)
        result["paired"].extend(paired(rows,truth,arrays,cell,"all"))
        if cell=="JHH5":
            groups = [("folds",f"fold_{f}",rows.fold.eq(f)) for f in range(3)]
            groups.extend(("leave_plate",plate,rows.rna_plate.ne(plate)) for plate in sorted(rows.rna_plate.unique()))
            for table,label,mask in groups:
                result[table].extend(paired(rows.loc[mask].reset_index(drop=True),truth[mask],{a:x[mask] for a,x in arrays.items()},cell,label))
    result["fit"] = [item for records in fits.values() for item in records]
    with TemporaryDirectory(dir=args.run,prefix=".donor8-state-score-") as folder:
        out = Path(folder)
        for name,values in result.items():
            pd.DataFrame(values).to_csv(out/f"{name}.tsv",sep="\t",index=False,na_rep="NA")
        write_json(out/"receipt.json",dict(manifest,freeze_sha256=args.digest,
            files={file.name:sha256(file) for file in out.iterdir()}))
        out.rename(args.run/"score")


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=("predict","freeze","score"))
    for field in ("reports","jhh5","jhh7","attribution","panel","state","protocol","run"):
        cli.add_argument("--"+field,type=Path,required=True)
    cli.add_argument("--group")
    cli.add_argument("--digest")
    args = cli.parse_args()
    (fit_predictions if args.phase=="predict" else finish)(args,*lineage(args))
