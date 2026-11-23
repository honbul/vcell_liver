"""Shared native-screen metrics for tuning and exact H2 replay."""
import numpy as np

from nextgen_h2.metrics import CELL, DRUG, Evaluator, macro, stack

AXES = ("spearman","interaction_spearman","ndcg_at_5")
WEIGHT = (.45,.30,.25)


def quality(rows,pred,registry,*,interaction=True):
    frame=rows.reset_index(drop=True)
    if not hasattr(frame,"truth"):
        from vc_build.external import native_sensitivity
        frame=frame.assign(truth=native_sensitivity(frame))
    evaluator=Evaluator(frame,registry.scaffold_block,0,scope="inner",interaction=interaction)
    d,c=stack(evaluator.score("arm",np.asarray(pred)))
    record={key:macro(tab,key) for tab, names in ((d,DRUG),(c,CELL)) for key in names if key in tab}
    return record


def utility(metrics):
    return sum(w*metrics[k] for k,w in zip(AXES,WEIGHT))
