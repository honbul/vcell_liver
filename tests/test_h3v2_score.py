import numpy as np
import pandas as pd

from h3v2.error_components import decompose
from h3v2.score import AXES, FAMILIES, SEEDS, held_points, join_truth, mean_and_sd, rules


def test_gate_requires_external_and_two_protected_axes_even_when_ndcg_improves():
    rows=[]
    for model in (*FAMILIES,"Champion"):
        for cohort,screens in (("drug",["macro","CTRPv2","GDSC2","PRISM20Q2"]),("GDSC1",["macro"])):
            for screen in screens:
                for axis in AXES:
                    target=axis=="ndcg_at_5"
                    rows.append({"candidate":model,"baseline":"H2","scope":cohort,"support":"all","screen":screen,
                                 "metric":axis,"delta":.02 if target else 0, "lower":.01 if target else -.001})
    frame=pd.DataFrame(rows)
    assert rules(frame)["Champion"]["verdict"]=="GO-MODEL"
    bad=(frame.candidate.eq("Champion")&frame.scope.eq("GDSC1")&frame.metric.eq("spearman"))
    frame.loc[bad,"lower"]=-.006
    assert rules(frame)["Champion"]["verdict"]=="NO-GO"
    assert rules(frame)["film"]["verdict"]=="GO-MODEL"


def test_exact_same_native_positions_and_percentile_error_projection():
    rng=np.random.default_rng(123)
    rows=[]
    for fold in range(3):
        for screen in ("CTRPv2","GDSC2","PRISM20Q2"):
            for i in range(10):
                for j in range(7):
                    if (i+j)%7==0:
                        continue
                    truth=rng.normal()
                    vals={f"{model}__{s}":truth for model in (*FAMILIES,"Champion","H3v1") for s in SEEDS}
                    rows.append({"row_id":f"{screen}/{fold}/{i}/{j}","fold":fold,"source_id":screen,
                                 "endpoint_id":screen,"lineage_group":f"{fold}-lineage{i}","cell_id":f"{fold}-cell{i}",
                                 "compound_id":f"drug{j}","truth":truth,"lincs":j%2==0,
                                 "structured_v1":truth,"drug_ridge_v1":truth,**vals})
    truth=pd.DataFrame(rows)
    data=join_truth(truth.drop(columns="truth"),truth[["row_id","truth"]])
    score,ci=held_points([data])
    assert ci[["delta","lower","upper"]].to_numpy().max()==0
    point=mean_and_sd(score)
    assert point.loc[point.seed.eq("mean")&point.metric.eq("ndcg_at_5"),"value"].eq(1.).all()
    ret=decompose(data.loc[data.fold.eq(0)],["structured_v1"],"toy",0,{"all":None})
    assert all(r["rank_mse"]==0 for r in ret)


if __name__=="__main__":
    test_gate_requires_external_and_two_protected_axes_even_when_ndcg_improves()
    test_exact_same_native_positions_and_percentile_error_projection()
    print("Score gates, paired seed scores and observed error projection passed")
