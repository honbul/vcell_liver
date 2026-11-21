"""Verify frozen R16 forecasts and calculate paired held-parent results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE
from src.r15_lincs.design import Bank
from src.r15_lincs.forecast import LEGACY, LEGACY_CPU, LEGACY_GPU, PROPOSED, split
from src.r15_lincs.score import CI, METRICS, load_slice

from .predict import CELL_GAIN, GAIN_SUFFIX, HEADS

LEGACY_SCALED = tuple(model+GAIN_SUFFIX for model in LEGACY)
REFERENCES = {
    "best-VC": LEGACY_SCALED,
    "best-published": tuple(model+GAIN_SUFFIX for model in LEGACY_GPU),
    "best-project": tuple(model+GAIN_SUFFIX for model in LEGACY_CPU),
}
R15_FREEZE = "02dbc1a8d9258b267863283009f8265d8db324f8d465080ab4fc47aa32aae9a7"


def manifest_rows(args, bank, phase):
    prior = None
    if not args.confirm:
        path = args.prior/"internal/freeze.json"
        if sha256(path) != R15_FREEZE:
            raise ValueError("R15 source is not the registered forecast freeze")
        prior = read_json(path)
    for fold in range(3):
        for role in "AB":
            scenario = bank.scenario
            key = f"{scenario}-f{fold}-{role}"
            if args.confirm:
                from .confirmation import confirm_split

                _,_,training,expected = confirm_split(bank, fold)
            else:
                old = argparse.Namespace(protocol=args.r15_protocol, out=args.prior)
                expected,_,history = load_slice(old, bank, scenario, fold, role, LEGACY)
                if history != prior["prediction_receipts"][key]:
                    raise ValueError("An R15 prediction, row ledger or receipt changed")
                _,_,training,_ = split(bank,fold)
            path = args.out/"predict"/scenario/f"f{fold}-{role}"
            arms, settings, receipts = {}, {}, {}
            models = (PROPOSED,*LEGACY) if phase == "fair" else (PROPOSED,*LEGACY,*HEADS)
            for model in models:
                directory = path/model
                stamp = read_json(directory/"receipt.json")
                selected = stamp["selection"]
                if (stamp["protocol_sha256"] != sha256(args.protocol) or stamp["cache"] != bank.receipts
                    or (stamp["scenario"],stamp["fold"],stamp["role"],selected["model"]) != (scenario,fold,role,model)
                    or selected["fit_groups"] != sorted(set(training.parent))):
                    raise ValueError(f"Out-of-contract R16 prediction: {directory}")
                for name, digest in stamp["files"].items():
                    if sha256(directory/name) != digest:
                        raise ValueError(f"Changed R16 forecast artifact: {directory/name}")
                rows = pd.read_csv(directory/"rows.tsv",sep="\t",keep_default_na=False)
                axis = ["study","index",*KEY,GROUP]
                if not rows[axis].equals(expected[axis]):
                    raise ValueError(f"Changed prediction axis: {directory}")
                with np.load(directory/"forecast.npz",allow_pickle=False) as f:
                    if set(arms)&set(f.files) or any(
                            f[a].shape!=(len(rows),len(bank.landmarks)) or not np.isfinite(f[a]).all()
                            for a in f.files):
                        raise ValueError(f"Invalid forecast bundle: {directory}")
                    arms.update(f)
                settings[model] = selected
                receipts[model] = sha256(directory/"receipt.json")
            anticipated = {PROPOSED,"B0","B_pos",*LEGACY_SCALED}
            if scenario == "B":
                anticipated.add(CELL_GAIN)
            if phase == "blend":
                anticipated.update(HEADS)
            if anticipated != set(arms):
                raise ValueError("R16 arm manifest changed")
            yield key,role,expected,arms,settings,receipts


def moment(rows, truth, prediction):
    error = np.square((truth-prediction).astype(np.float64)).sum(axis=1)
    energy = np.square(truth.astype(np.float64)).sum(axis=1)
    return pd.DataFrame(dict(parent=rows.parent.to_numpy(),rows=1,rmse_sum=np.sqrt(error/truth.shape[1]),
                             sse=error,sst=energy)).groupby("parent").sum()


def pool(blocks):
    result = pd.concat(blocks).groupby(level=0).sum()
    result["rmse"] = result.rmse_sum/result.rows
    return result


def summarize(records, scope, scenario, candidates, references, minimum):
    models = list(records)
    units = [pool(records[a]) for a in models]
    base = units[models.index("B0")]
    if any(not u.index.equals(base.index) or not u.rows.equals(base.rows) for u in units):
        raise ValueError("R16 does not compare common parent support")
    n = len(base)
    rmse = np.column_stack([u.rmse for u in units])
    sse = np.column_stack([u.sse for u in units])
    sst = base.sst.to_numpy()
    points = np.stack((rmse.mean(axis=0),1-sse.sum(axis=0)/sst.sum()))
    boots = np.empty((2,10000,len(models)),dtype=np.float64)
    rng = np.random.default_rng(150104)
    for start in range(0,10000,128):
        stop = min(start+128,10000)
        ix = rng.integers(n,size=(stop-start,n))
        boots[0,start:stop] = rmse[ix].mean(axis=1)
        boots[1,start:stop] = 1-sse[ix].sum(axis=1)/sst[ix].sum(axis=1)[:,None]
    support = dict(scope=scope,scenario=scenario,parents=n,rows=int(base.rows.sum()),eligible=n>=minimum)
    levels = np.quantile(boots,CI,axis=1)
    point_rows = [dict(support,arm=model,**{
        label:points[i,j] for i,label in enumerate(METRICS)}, **{
        f"{label}_{level}99":levels[e,i,j] for e,level in enumerate(("low","high"))
        for i,label in enumerate(METRICS)}) for j,model in enumerate(models)]
    paired = []
    for candidate in candidates:
        c = models.index(candidate)
        for reference, options in references.items():
            idx = [models.index(a) for a in options]
            for i,label in enumerate(METRICS):
                best = np.argmin if i == 0 else np.argmax
                reducer = np.min if i == 0 else np.max
                chosen = idx[best(points[i,idx])]
                paired_draws = boots[i,:,c]-reducer(boots[i][:,idx],axis=1)
                low,high = np.quantile(paired_draws,CI)
                paired.append(dict(support,arm=candidate,reference=reference,metric=label,
                                   best_at_point=models[chosen],
                                   difference=points[i,c]-points[i,chosen],low99=low,high99=high,
                                   passes=bool(n>=minimum and (high<0 if i==0 else low>0))))
    return point_rows,paired


def repeats(rows, donor, different_location):
    """A donor plate contributes one average per parent, dose, time and cell."""
    rows = rows.reset_index(drop=True)
    plate, coordinate = rows.plate_id.to_numpy(), rows.rna_well.to_numpy()
    result = np.zeros_like(donor)
    valid = np.zeros(len(rows),dtype=bool)
    for indices in rows.groupby([LINE,"parent","hours","dose"]).indices.values():
        if len(set(plate[indices])) < 2:
            continue
        for i in indices:
            other = indices[plate[indices]!=plate[i]]
            if different_location:
                other = other[coordinate[other]!=coordinate[i]]
            if len(other):
                by_plate = pd.Series(plate[other]).groupby(plate[other]).indices
                result[i] = np.stack([donor[other[ix]].mean(axis=0) for ix in by_plate.values()]).mean(axis=0)
                valid[i] = True
    return valid,result


def masks(rows, args):
    if not args.confirm:
        yield "HEPG2_6h_primary",(rows[LINE].eq("HEPG2") & rows.study.eq("GSE92742") &
              rows.hours.eq(6) & rows.quality & ~rows.prior_exact).to_numpy()
        return
    for label in ("all","B1_same_well","B1_other_well","B1_unseen_parent",
                  "R8_same_well","R8_other_well","R8_unseen_parent"):
        match = np.ones(len(rows),dtype=bool)
        if label != "all":
            round_, category = label.split("_",1)
            exact, parent = rows[f"{round_}_exact"],rows[f"{round_}_parent"]
            match = {"same_well":exact,"other_well":parent&~exact,"unseen_parent":~parent}[category].to_numpy()
        yield f"{args.cell}_24h_{label}",match


def run(args):
    freezing = args.action.startswith("freeze_")
    phase = "fair" if args.action in ("freeze_fair","fair") else "blend"
    contract = args.out/f"freeze_{phase}.json"
    expected = read_json(contract) if not freezing else None
    manifest = dict(protocol_sha256=sha256(args.protocol),score_code_sha256=sha256(__file__),
                    r15_freeze_sha256=None if args.confirm else R15_FREEZE,prediction_receipts={})
    if expected and any(manifest[k]!=expected[k] for k in manifest if k!="prediction_receipts"):
        raise ValueError("Frozen R16 scoring contract changed")
    scores, selections, supports, reached = [],[],[],[]
    for scenario in "AB":
        if args.confirm:
            from .confirmation import ConfirmBank

            bank = ConfirmBank(args.cache,args.protocol,scenario,args.cell)
        else:
            bank = Bank(args.cache,args.r15_protocol,scenario)
        recorded = {}
        requested = [PROPOSED] if phase == "fair" else list(HEADS)
        references = dict(REFERENCES)
        if phase == "blend":
            references.update(B0=("B0",),B_pos=("B_pos",))
            if not args.confirm:
                references[PROPOSED] = (PROPOSED,)
        if scenario == "B":
            references[CELL_GAIN] = (CELL_GAIN,)
        if args.action == "repeat":
            requested = ["repeat",PROPOSED,*HEADS]
            references = {"B0":("B0",)}
        for key,role,rows,arms,settings,receipts in manifest_rows(args,bank,phase):
            manifest["prediction_receipts"][key] = receipts
            if freezing:
                continue
            if receipts != expected["prediction_receipts"][key]:
                raise ValueError(f"R16 forecast changed since freeze: {key}")
            for arm, selected in settings.items():
                selections.append(dict(slice=key,arm=arm,**{
                    k:str(v) if isinstance(v,(list,tuple)) else v for k,v in selected.items()
                    if k not in ("model","fit_groups")}))
            other = "B" if role == "A" else "A"
            truth = bank.response(rows,other)
            for scope, keep in masks(rows,args):
                if args.action == "repeat":
                    if not keep.any() or (args.confirm and not scope.endswith("_all")):
                        continue
                    r = rows.loc[keep].reset_index(drop=True)
                    observed, donor = truth[keep], bank.response(r,role)
                    for suffix in ("other_plate","other_plate_and_position"):
                        supported, pred = repeats(r,donor,suffix.endswith("_position"))
                        values = {a:arms[a][keep][supported] for a in ("B0",PROPOSED,*HEADS)}
                        values["repeat"] = pred[supported]
                        collect(recorded,scope+"_"+suffix,r.loc[supported],observed[supported],values)
                else:
                    collect(recorded,scope,rows.loc[keep],truth[keep],{a:p[keep] for a,p in arms.items()})
        minimum = 100 if args.confirm else 1
        for scope,records in recorded.items():
            points, paired = summarize(records,scope,scenario,requested,references,minimum)
            scores.extend(points)
            supports.extend(paired)
            if args.action == "repeat":
                summary = {point["arm"]:point for point in points}
                baseline, repeat = summary["B0"],summary["repeat"]
                for arm in (PROPOSED,*HEADS):
                    rmse_gap = baseline["rmse"]-repeat["rmse"]
                    r2_gap = repeat["r2_pooled"]-baseline["r2_pooled"]
                    reached.append(dict(scope=scope,scenario=scenario,arm=arm,parents=repeat["parents"],
                                        rmse_fraction=(baseline["rmse"]-summary[arm]["rmse"])/rmse_gap if rmse_gap>0 else np.nan,
                                        r2_fraction=(summary[arm]["r2_pooled"]-baseline["r2_pooled"])/r2_gap if r2_gap>0 else np.nan))
    if freezing:
        with contract.open("x") as stream:
            import json

            json.dump(manifest,stream,sort_keys=True,indent=2)
            stream.write("\n")
        return
    if manifest!=expected:
        raise ValueError("R16 scored slice inventory differs from freeze")
    out = args.out/args.action
    out.mkdir(exist_ok=False)
    for name, rows in (("points",scores),("paired",supports),("selection",selections),("repeat_fraction",reached)):
        if rows:
            pd.DataFrame(rows).to_csv(out/f"{name}.tsv",sep="\t",index=False)
    write_json(out/"receipt.json",dict(freeze_sha256=sha256(contract),code_sha256=sha256(__file__),
                                      files={p.name:sha256(p) for p in out.iterdir()}))


def collect(target, scope, rows, truth, arrays):
    if len(rows):
        group = target.setdefault(scope,{arm:[] for arm in arrays})
        for arm,prediction in arrays.items():
            group[arm].append(moment(rows,truth,prediction))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze_fair","fair","freeze_blend","blend","repeat"))
    for name in ("cache","protocol","out","prior","r15-protocol"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--confirm",action="store_true")
    parser.add_argument("--cell")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
