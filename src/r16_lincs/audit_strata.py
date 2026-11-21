"""Quantify history and technical JHH support using only the completed audit."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.design import fold_id
from src.r15_lincs.position_audit import position_scores, repeat_layout

from .audit import CELLS


def run(args):
    receipt = read_json(args.audit/"receipt.json")
    for name in ("target_rows.tsv","source_wells.tsv","target_wells.npy","vehicle.npy"):
        if sha256(args.audit/name)!=receipt["files"][name]:
            raise ValueError(f"Changed audited artifact: {name}")
    rows = pd.read_csv(args.audit/"target_rows.tsv",sep="\t",keep_default_na=False)
    source = pd.read_csv(args.audit/"source_wells.tsv",sep="\t",usecols=["parent"])
    y = np.load(args.audit/"target_wells.npy",allow_pickle=False)
    ctrl = np.load(args.audit/"vehicle.npy",allow_pickle=False)
    target_counts, position, repeats = [], [], []
    for cell in CELLS:
        secondary = cell.startswith("JHH")
        target = rows.loc[rows.cell_id.eq(cell)]
        target = target if secondary else target.loc[target.quality]
        for scenario in "AB":
            cohort = target if scenario == "A" else target.loc[target.parent.isin(source.parent)]
            for label in ("all","B1_same_well","B1_other_well","B1_unseen_parent",
                          "R8_same_well","R8_other_well","R8_unseen_parent"):
                if label == "all":
                    group = cohort
                else:
                    rnd,tag = label.split("_",1)
                    known, same = cohort[rnd+"_parent"],cohort[rnd+"_exact"]
                    group = cohort.loc[{"same_well":same,"other_well":known&~same,"unseen_parent":~known}[tag]]
                groups = group.parent.unique()
                target_counts.append(dict(cell=cell,scenario=scenario,stratum=label,technical=secondary,
                                          wells=len(group),parents=len(groups),plates=group.rna_plate.nunique()))
            assigned = cohort.parent.map(lambda p:fold_id(p,"R15-outer",3))
            for fold in range(3):
                train = cohort.loc[assigned.ne(fold),"parent"]
                flags = train.map(lambda p:fold_id(p,f"R15-inner-{fold}",5)).eq(0)
                if len(cohort) and (not train.loc[flags].nunique() or not cohort.loc[assigned.eq(fold),"parent"].nunique()):
                    raise ValueError(f"Missing target validation or evaluation: {cell}, {scenario}, {fold}")
        if secondary:
            use = target.loc[target.groupby("rna_plate").parent.transform("nunique").ge(3)]
            repeats.append(dict(cell=cell,**repeat_layout(use)))
            if len(use) and use.rna_plate.nunique()>1:
                response = y[use["index"]]-ctrl[use.ctrl_index].mean(axis=1)
                position.extend(dict(cell=cell,**p) for p in position_scores(use.reset_index(drop=True),response))
    args.out.mkdir(parents=True,exist_ok=False)
    for name, data in (("strata",target_counts),("technical-position",position),("technical-repeat-layout",repeats)):
        pd.DataFrame(data).to_csv(args.out/f"{name}.tsv",sep="\t",index=False)
    write_json(args.out/"receipt.json",dict(audit_sha256=sha256(args.audit/"receipt.json"),
        code_sha256=sha256(__file__),files={p.name:sha256(p) for p in args.out.iterdir()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args())
