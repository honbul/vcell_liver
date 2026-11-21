from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

from src.r19_plan.hcc_fixed_panel_refit import cohort, eligible_slice, observed, projection, provenance
from src.r11_chemical_probe.dataset import GROUP

run = Path(sys.argv[1])
root = run.parent
args = SimpleNamespace(reports=run/'repo/reports', previous=root/'hcc-scaffold-refit.lWd6KCZt',
    panel_lock=root/'hcc-vehicle-panel.MvfmPZOH/lock', protocol=run/'repo/reports/r19_plan/HCC_FIXED_PANEL_REFIT_PROTOCOL.md')
parts,panel,_,blocks,_ = provenance(args)
records = []
for fold in range(3):
    fit, held = (projection(args, parts, fold, phase) for phase in ('fit', 'held'))
    data, _, _ = cohort(fit)
    query, _, _ = cohort(held)
    values = [observed(args,parts,panel,fold,role,'visible',fit,data) for role in 'AB']
    masks = {'full': np.ones(len(data),dtype=bool), 'purged': ~data[GROUP].map(blocks).isin(query[GROUP].map(blocks)).to_numpy()}
    for scheme, active in masks.items():
        for part, keep in [('outer',active), *((f'inner{j}',active & data.inner_fold.eq(j).to_numpy()) for j in (0,1))]:
            a, b = (eligible_slice(data,y,keep) for y in values)
            assert a[0].equals(b[0])
            gap = float(np.max(np.abs(a[1]-b[1])))
            assert gap <= 1e-6, gap
            records.append(dict(fold=fold,scheme=scheme,split=part,parents=a[0][GROUP].nunique(),max_abs_gap=gap))
pd.DataFrame(records).to_csv(run/'training_role_check.tsv',sep='\t',index=False)
