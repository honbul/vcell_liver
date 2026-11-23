"""Separate descriptive additive and residual errors on identical observed rank support."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from nextgen_h2.assets import KEY
from nextgen_h2.metrics import center_interactions


def decompose(frame, columns, scope, fold, supports):
    """Percentile-rank error removes unrelated differences in assay response units."""
    points=[]
    for name, mask in supports.items():
        section=frame if mask is None else frame.loc[mask]
        for screen, rows in section.groupby("source_id"):
            cells=pd.Index(sorted(rows.cell_id.unique()))
            drugs=pd.Index(sorted(rows[KEY].unique()))
            if len(cells)<2 or len(drugs)<2:
                continue
            i,j=cells.get_indexer(rows.cell_id),drugs.get_indexer(rows[KEY])
            target=rankdata(rows.truth.to_numpy())/len(rows)
            for model in columns:
                err=rankdata(rows[model].to_numpy())/len(rows)-target
                table=np.full((len(cells),len(drugs)),np.nan)
                table[i,j]=err
                delta=center_interactions(table)[i,j]
                additive=err-delta
                bias=err.mean()
                total=np.mean(err**2)
                broad=np.mean((additive-bias)**2)
                interaction=np.mean(delta**2)
                if not np.isclose(total,bias**2+broad+interaction,atol=2e-8,rtol=1e-5):
                    raise ValueError("Descriptive error projection failed its sum-of-squares check")
                points.append({"scope":scope,"fold":fold,"screen":screen,"support":name,"arm":model,"rows":len(rows),
                               "rank_mse":total,"bias_sq":bias**2,"additive_mse":broad,"interaction_mse":interaction})
    return points
