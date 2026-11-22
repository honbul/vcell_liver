"""Fit R15/R16 candidates with disjoint vehicle roles and persist held forecasts."""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256
from src.r15_lincs.design import Bank, fold_id
from src.r15_lincs.estimators import GAINS, loss
from src.r15_lincs.forecast import (LEGACY_CPU, LEGACY_GPU, PROPOSED, choose_proposed,
                                   fit_known, grid_chemical, pick, position, write_forecast)
from src.r16_lincs.predict import CELL_GAIN, GAIN_SUFFIX, HEADS, gain_choice, inner_known

from .vehicles import GEO, ROLES


class RoleBank(Bank):
    def __init__(self, args, scenario):
        super().__init__(args.cache,args.r15_protocol,scenario)
        receipt = read_json(args.vehicles/"receipt.json")
        if receipt["protocol_sha256"]!=sha256(args.protocol):
            raise ValueError("Registered vehicle stage or protocol changed")
        for name,digest in receipt["files"].items():
            if sha256(args.vehicles/name)!=digest:
                raise ValueError(f"Staged vehicle asset changed: {name}")
        self.receipts["R17"] = sha256(args.vehicles/"receipt.json")
        self.axis = pd.Index(pd.read_csv(args.vehicles/"vehicles.tsv",sep="\t").rna_plate)
        self.dmso = np.load(args.vehicles/"vehicles.npy",mmap_mode="r",allow_pickle=False)
        if self.dmso.shape!=(len(self.axis),3,len(self.landmarks)) or not self.axis.is_unique:
            raise ValueError("Vehicle roles or axes changed")
        self.rows = self.rows.loc[self.rows.study.eq(GEO) & self.rows.rna_plate.isin(self.axis)].copy()

    def control(self, rows, role):
        indices = self.axis.get_indexer(rows.rna_plate)
        if (indices<0).any():
            raise ValueError("A forecast row lacks a registered role")
        return self.dmso[indices,ROLES.index(role)]


def rotation(fit):
    i = ROLES.index(fit)
    return ROLES[(i+1)%3],ROLES[(i+2)%3]


def split(bank, fold):
    rows = bank.target()
    held = rows.parent.map(lambda p:fold_id(p,"R15-outer",3)).eq(fold)
    training, outer = rows.loc[~held],rows.loc[held]
    selection = training.parent.map(lambda p:fold_id(p,f"R15-inner-{fold}",5)).eq(0)
    return tuple(frame.reset_index(drop=True) for frame in
                 (training.loc[~selection],training.loc[selection],training,outer))


def fit_slice(args, bank, inner, validation, training, outer, get_position=position):
    """Run the R17 fit and selection on the supplied physical role and parent axes."""
    if set(inner.parent)&set(validation.parent) or set(training.parent)&set(outer.parent):
        raise ValueError("R17 parent leakage into fit")
    select, score = rotation(args.role)
    truth = bank.response(validation,select)
    y, inner_pos, val_pos = get_position(bank,inner,validation,args.role)
    pos_gain = gain_choice(validation,truth,val_pos,np.zeros_like(truth))
    raw, final_pos, outer_pos = get_position(bank,training,outer,args.role)
    x_inner, x_valid = bank.transfer(inner),bank.transfer(validation)
    x_outer = bank.transfer(outer)
    inner_anchor = pos_gain*val_pos+x_valid
    outer_anchor = pos_gain*outer_pos+x_outer
    inner_labels = y-pos_gain*inner_pos-x_inner
    outer_labels = raw-pos_gain*final_pos-bank.transfer(training)
    audit = dict(fit=args.role,select=select,score=score,position_gain=pos_gain,
                 fit_groups=sorted(set(training.parent)),selection_groups=sorted(set(validation.parent)))
    arrays, choices = {}, {}

    def learn(name, work):
        params, val_pred, _ = inner_known(args,bank,inner,validation,inner_labels,truth,inner_anchor,name,work)
        gain = gain_choice(validation,truth,val_pred,inner_anchor)
        current = fit_known(args,bank,training,outer,outer_labels,None,outer_anchor,name,work,params)
        choices[name] = dict(parameters=params,gain=gain)
        arrays[name+GAIN_SUFFIX] = outer_anchor+gain*current
        return val_pred,current

    with TemporaryDirectory(prefix=f"r17-{args.model}-") as work:
        work = Path(work)
        if args.model in LEGACY_GPU:
            learn(args.model,work)
        else:
            settings, objective = choose_proposed(bank,inner,validation,y,truth,inner_pos,val_pos,pos_gain)
            rule, gain = settings
            choices[PROPOSED] = dict(rule=rule,gain=gain,objective=objective)
            val_kernel = grid_chemical(bank,inner,validation,inner_labels,[rule])[rule]
            outer_kernel = grid_chemical(bank,training,outer,outer_labels,[rule])[rule]
            arrays.update({PROPOSED:outer_anchor+gain*outer_kernel,"B0":np.zeros_like(outer_kernel),
                           "B_pos":pos_gain*outer_pos,"anchor":outer_anchor})
            cell_gain = gain_choice(validation,truth,x_valid,np.zeros_like(truth)) if args.scenario=="B" else 0.
            blend_anchor = pos_gain*outer_pos+cell_gain*x_outer
            val_blend_anchor = pos_gain*val_pos+cell_gain*x_valid
            arrays["blend_anchor"] = blend_anchor
            if args.scenario=="B":
                arrays[CELL_GAIN] = cell_gain*x_outer
            choices[CELL_GAIN] = dict(gain=cell_gain)
            neural = {name:learn(name,work) for name in LEGACY_CPU}
            for blend,name in HEADS.items():
                val_neural,outer_neural = neural[name]
                grid = {(a,b):loss(validation,truth,val_blend_anchor+a*val_kernel+b*val_neural)
                        for a,b in product(GAINS,repeat=2)}
                a,b = pick(grid)
                choices[blend] = dict(a=a,b=b,objective=grid[a,b])
                arrays[blend] = blend_anchor+a*outer_kernel+b*outer_neural
    return arrays, dict(audit,model=args.model,settings=choices)


def forecast(args):
    bank = RoleBank(args,args.scenario)
    inner, validation, training, outer = split(bank,args.fold)
    arrays, selection = fit_slice(args,bank,inner,validation,training,outer)
    path = args.out/"predict"/args.scenario/f"f{args.fold}-{args.role}"/args.model
    write_forecast(args,bank,path,outer,arrays,dict(selection,
        predictor_sha256=sha256(__file__),vehicles_sha256=bank.receipts["R17"]))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","vehicles","protocol","r15-protocol","out"):
        p.add_argument("--"+name,type=Path,required=True)
    return p


if __name__=="__main__":
    p = parser()
    for name in ("config","assets","prnet"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--model",choices=("cpu",*LEGACY_GPU),required=True)
    p.add_argument("--scenario",choices="AB",required=True)
    p.add_argument("--fold",type=int,choices=range(3),required=True)
    p.add_argument("--role",choices=ROLES,required=True)
    forecast(p.parse_args())
