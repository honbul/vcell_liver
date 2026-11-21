"""Freeze the registered R17 forecast axis and separately score two endpoints."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY
from src.r15_lincs.forecast import LEGACY_GPU, PROPOSED
from src.r16_lincs.predict import CELL_GAIN, HEADS
from src.r16_lincs.score import LEGACY_SCALED, REFERENCES, collect, moment, pool, summarize

from .predict import RoleBank, parser, rotation, split
from .vehicles import ROLES


ARMS = (PROPOSED,*HEADS)


def slices(args, bank):
    for fold in range(3):
        _, select, train, rows = split(bank,fold)
        for role in ROLES:
            sel_role, score_role = rotation(role)
            key = f"{bank.scenario}-f{fold}-{role}"
            root = args.out/"predict"/bank.scenario/f"f{fold}-{role}"
            receipts, arrays, settings = {}, {}, []
            for model in ("cpu",*LEGACY_GPU):
                path = root/model
                receipt = read_json(path/"receipt.json")
                flags = receipt["selection"]
                if (receipt["protocol_sha256"]!=sha256(args.protocol) or receipt["cache"]!=bank.receipts
                    or (receipt["scenario"],receipt["fold"],receipt["role"])!=(bank.scenario,fold,role)
                    or (flags["model"],flags["fit"],flags["select"],flags["score"])!=(model,role,sel_role,score_role)
                    or flags["fit_groups"]!=sorted(set(train.parent))
                    or flags["selection_groups"]!=sorted(set(select.parent))):
                    raise ValueError(f"Out-of-contract R17 forecast: {path}")
                for name,digest in receipt["files"].items():
                    if sha256(path/name)!=digest:
                        raise ValueError(f"R17 forecast artifact changed: {path/name}")
                metadata = pd.read_csv(path/"rows.tsv",sep="\t",keep_default_na=False)
                axis = ["study","index",*KEY,GROUP]
                if not rows[axis].equals(metadata[axis]):
                    raise ValueError(f"R17 forecast axis changed: {path}")
                with np.load(path/"forecast.npz",allow_pickle=False) as x:
                    if set(x.files)&set(arrays) or any(
                            x[n].shape!=(len(rows),len(bank.landmarks)) or not np.isfinite(x[n]).all() for n in x.files):
                        raise ValueError(f"Invalid R17 matrices: {path}")
                    arrays.update(x)
                for name,value in flags["settings"].items():
                    settings.append(dict(slice=key,arm=name,fit=role,select=sel_role,score=score_role,
                                         position_gain=flags["position_gain"],**value))
                receipts[model] = sha256(path/"receipt.json")
            expected = {"B0","B_pos","anchor","blend_anchor",*LEGACY_SCALED,*ARMS}
            if bank.scenario=="B":
                expected.add(CELL_GAIN)
            if expected!=set(arrays):
                raise ValueError(f"R17 arm registry changed: {key}")
            yield key, role, score_role, rows, arrays, receipts, settings


def project(residual, vehicle):
    energy = np.einsum("ij,ij->i",vehicle,vehicle)
    dot = np.einsum("ij,ij->i",residual,vehicle)
    coefficients = np.divide(dot,energy,out=np.zeros_like(dot),where=energy>0)
    return residual-coefficients[:,None]*vehicle


def inner(x,y):
    return float(np.einsum("ij,ij->",x,y))


def moments(x,y):
    return np.array([inner(x,y),inner(x,x),inner(y,y)],dtype=np.float64)


def cosine(stats):
    dot, xx, yy = stats
    return float(dot/np.sqrt(xx*yy)) if min(xx,yy)>0 else np.nan


class Projection:
    def __init__(self):
        self.blocks = defaultdict(list)
        self.cos = defaultdict(lambda:np.zeros((5,3),dtype=np.float64))

    def record(self, bank, rows, arms, fit, score, truth):
        direction = (bank.control(rows,fit)-bank.control(rows,score)).astype(np.float64)
        truth = truth.astype(np.float64)
        for arm in (*ARMS,*LEGACY_SCALED,*([CELL_GAIN] if bank.scenario=="B" else [])):
            if arm==CELL_GAIN:
                anchor = np.zeros_like(truth)
            else:
                anchor = arms["blend_anchor" if arm in HEADS else "anchor"].astype(np.float64)
            raw = arms[arm].astype(np.float64)
            residual, target = raw-anchor,truth-anchor
            projected = project(residual,direction)
            denoised = anchor+projected
            self.blocks[arm].append((moment(rows,truth,raw),moment(rows,truth,denoised)))
            self.cos[arm] += np.stack((moments(raw,truth),moments(denoised,truth),
                  moments(residual,target),moments(projected,target),moments(residual,-direction)))

    def results(self, scenario):
        for arm, blocks in self.blocks.items():
            before, after = (pool(part) for part in zip(*blocks,strict=True))
            metrics = dict(zip(("cos_before","cos_projected","residual_cos_before","residual_cos_projected",
                                "cos_vs_same_role_vehicle_noise"),map(cosine,self.cos[arm]),strict=True))
            yield dict(scenario=scenario,arm=arm,parents=len(before),rows=int(before.rows.sum()),
                  rmse_before=before.rmse.mean(),rmse_projected=after.rmse.mean(),
                  removed_residual_energy_fraction=(self.cos[arm][2,1]-self.cos[arm][3,1])/self.cos[arm][2,1]
                  if self.cos[arm][2,1]>0 else np.nan,**metrics)


def run(args):
    freeze_path = args.out/"freeze.json"
    is_freeze = args.action=="freeze"
    existing = None if is_freeze else read_json(freeze_path)
    manifest = dict(protocol_sha256=sha256(args.protocol),code_sha256=sha256(__file__),slices={})
    if existing and any(existing[k]!=manifest[k] for k in manifest if k!="slices"):
        raise ValueError("R17 score contract changed")
    selections, points, pairs, projections = [],[],[],[]
    for scenario in "AB":
        bank = RoleBank(args,scenario)
        blocks = {}
        diagnostic = Projection()
        for key,fit,score,rows,arms,receipts,settings in slices(args,bank):
            manifest["slices"][key] = receipts
            if is_freeze:
                continue
            if receipts!=existing["slices"][key]:
                raise ValueError(f"R17 prediction changed after freeze: {key}")
            truth = bank.response(rows,score)
            if args.action=="score":
                collect(blocks,"HEPG2_6h_primary",rows,truth,
                        {k:v for k,v in arms.items() if k not in ("anchor","blend_anchor")})
                selections.extend(settings)
            else:
                diagnostic.record(bank,rows,arms,fit,score,truth)
            print(f"{args.action} {key}, {len(rows)} wells",flush=True)
        if args.action=="score":
            refs = dict(REFERENCES,B0=("B0",),B_pos=("B_pos",),R15=(PROPOSED,))
            if scenario=="B":
                refs[CELL_GAIN]=(CELL_GAIN,)
            p,d = summarize(blocks["HEPG2_6h_primary"],"HEPG2_6h_primary",scenario,ARMS,refs,1000)
            points.extend(p)
            pairs.extend(x for x in d if x["arm"]!=x["reference"])
        elif args.action=="projection":
            projections.extend(diagnostic.results(scenario))
    if is_freeze:
        with freeze_path.open("x") as stream:
            json.dump(manifest,stream,sort_keys=True,indent=2)
            stream.write("\n")
        return
    if manifest!=existing:
        raise ValueError("R17 scored manifest differs from freeze")
    out = args.out/args.action
    out.mkdir(exist_ok=False)
    output = {"points":points,"paired":pairs,"selection":selections,"projection":projections}
    for name, values in output.items():
        if values:
            pd.DataFrame(values).to_csv(out/f"{name}.tsv",sep="\t",index=False,na_rep="NA")
    write_json(out/"receipt.json",dict(freeze_sha256=sha256(freeze_path),action=args.action,
                     files={p.name:sha256(p) for p in out.iterdir()}))


if __name__=="__main__":
    p = parser()
    p.add_argument("action",choices=("freeze","score","projection"))
    run(p.parse_args())
