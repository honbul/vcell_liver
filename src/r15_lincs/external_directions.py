"""Compute descriptive direction and scale diagnostics on the existing external freeze."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import MIN_CELLS, MIN_GROUPS, eligibility, slice_name
from src.goal_tracks.t1_io import score_truth
from src.goal_tracks.t1_score import STRATA, STRATUM_LINES, search
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, LINE, group_weights
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.objectives import center, row_products

from . import external


def record(book, n, cohort, rows, observed, arrays):
    metric = Contrast(rows, MIN_GROUPS)
    truth = center(metric, observed)
    reference = row_products(truth, truth)
    for name, prediction in arrays.items():
        value = center(metric, prediction)
        energy = row_products(value, value)
        inner = row_products(value, truth)
        scale = np.sqrt(energy*reference)
        cos = np.divide(inner,scale,out=np.zeros_like(scale),where=scale>1e-14)
        top = search(metric,value,truth)
        for scope in STRATA:
            keep = (np.ones(len(rows),dtype=bool) if scope=="all" else
                    rows[LINE].isin(STRATUM_LINES[scope]).to_numpy())
            if not keep.any():
                continue
            weight = group_weights(rows.loc[keep])
            book[n,cohort,scope,name] += (np.dot(weight,cos[keep]),np.dot(weight,top[keep]),weight.sum(),
                energy[keep].sum(),reference[keep].sum(),inner[keep].sum())


def run(args):
    context=external.published(args)
    stage=context[0]
    bundles,hashes=external.read_bundle(args,context)
    seal=args.out/"external/freeze.json"
    frozen=read_json(seal)
    if frozen["bundles"]!=hashes or frozen["protocol_sha256"]!=sha256(args.protocol):
        raise ValueError("Descriptive statistics require the exact external freeze")
    book=defaultdict(lambda:np.zeros(6,dtype=np.float64))
    for n in MIN_CELLS:
        for fold in range(3):
            for role in "AB":
                rows,panel,old,proof=external.load_published(args,context,n,fold,role,True)
                if frozen["published"][slice_name(5.,n,fold,role)]!=proof:
                    raise ValueError("Historical forecasts changed after freeze")
                _,ma,ga,pa=bundles["A",role]
                _,mb,gb,pb=bundles["B",role]
                if not ga.equals(gb):
                    raise ValueError("External gene axis changed")
                axis=pd.Index(panel).get_indexer(ga.token_id_tahoe)
                obs=score_truth(stage,5.,n,fold,role,rows,panel)[:,axis]
                arrays={k:v[:,axis] for k,v in old.items()}
                arrays["R15-A"]=pa["proposal"][ma.index.get_indexer(rows[ENTITY])]
                record(book,n,"all-A",rows,obs,arrays)
                ib=mb.index.get_indexer(rows[ENTITY])
                eligible=mb.supported.to_numpy()[ib]
                subset=eligibility(rows.loc[eligible].assign(_row=np.flatnonzero(eligible)),0)
                active=subset.pop("_row").to_numpy(dtype=int)
                if len(active):
                    projected={k:v[active] for k,v in arrays.items()}
                    projected["R15-B"]=pb["proposal"][ib[active]]
                    projected["B_cell"]=pb["B_cell"][ib[active]]
                    record(book,n,"LINCS-supported",subset,obs[active],projected)
    entries=[]
    for (n,cohort,scope,arm),(cos,top,mass,energy,observed,inner) in sorted(book.items()):
        joint=np.sqrt(energy*observed)
        entries.append(dict(minimum_cells=n,cohort=cohort,stratum=scope,arm=arm,cosine=cos/mass,top1=top/mass,
            prediction_to_observed_rms=np.sqrt(energy/observed),pooled_cosine=inner/joint if joint>0 else 0.))
    path=args.out/"external/descriptive"
    path.mkdir(exist_ok=False)
    output=path/"direction_scale.tsv"
    pd.DataFrame(entries).to_csv(output,sep="\t",index=False)
    write_json(path/"receipt.json",dict(freeze_sha256=sha256(seal),files={output.name:sha256(output)},
                                        descriptive_only=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","tahoe","t2-reference","out"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    run(parser.parse_args())
