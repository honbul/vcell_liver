"""Synthetic checks for exactly the new fold and score contracts."""
import argparse
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from nextgen_h2.model import Prediction
from vc_build.io import publish_json

from .e1 import arms, v1_parts, rank_invariance
from .e3 import ARMS, MODES, REFERENCES, column
from .inference import weighted_midrank
from .register import balanced_labels
from .score import e1_scopes, e3_policy, e3_scopes


def mock(seed):
    rng = np.random.default_rng(seed)
    p = pd.MultiIndex.from_product([[f"c{i}" for i in range(8)],list("abcdefghij")],
                                    names=["cell_id","compound_id"]).to_frame(index=False)
    p["lineage_group"] = p.cell_id
    p["source_id"], p["endpoint_id"] = "Screen", "AUC"
    p["truth"] = rng.normal(size=len(p))
    return p, rng


class Contracts(unittest.TestCase):
    def test_frozen_fallback_and_exact_within_drug_invariance(self):
        raw = torch.eye(2)
        b = Prediction(torch.tensor([1.,2.]), torch.eye(2), False)
        p = Prediction(torch.tensor([3.,4.]), 2 * torch.eye(2), False,
                       present=torch.tensor([True,False]),baseline=b)
        i,r = v1_parts(p,raw,raw.mean(0))
        torch.testing.assert_close(i+r,p.values(raw,raw.mean(0)),rtol=0,atol=0)
        frame,rng = mock(19)
        frame["i0"],frame["i1"] = frame.compound_id.map(dict(zip("abcdefghij", range(10)))),5.
        frame["r0"],frame["rc"],frame["r1"] = rng.normal(size=(3,len(frame)))
        frame["lincs_observed"] = frame.compound_id.le("f")
        frame = frame.assign(**arms(frame), **{"Cchem-int":frame.rc})
        groups = pd.Series({k:k for k in frame.compound_id.unique()})
        self.assertEqual(rank_invariance(frame,arms(frame)),20)
        results,_,_ = e1_scopes([(0,frame)],groups,"DRP")
        measurements=[d for _, rows in results for d in rows if d["contrast"]=="F01 - F00" and d["metric"]=="spearman"]
        self.assertTrue(measurements)
        self.assertTrue(all(d["delta"] == d["lo95"] == d["hi95"] == 0 for d in measurements))

    def test_both_split_axes_hold_groups(self):
        left = pd.Series(["p","p","q","r","s","t"])
        right = pd.Series(["C","D","D","E","F","G"])
        for groups in (left,right):
            values=balanced_labels(groups,24)
            self.assertEqual(values.groupby(groups).nunique().max(),1)

    def test_weighted_midranks_include_ties_under_real_weights(self):
        x = np.array([1.,4.,1.,2.,2.])
        unit = np.ones((2,len(x)))
        got = weighted_midrank(x,unit)
        np.testing.assert_array_equal(got[0],rankdata(x)-3.)
        heavy = np.array([[1.,3.,1.,4.,2.]])
        result = weighted_midrank(x,heavy)[0]
        self.assertAlmostEqual(float(heavy @ result),0.)
        self.assertEqual(result[0],result[2])
        self.assertEqual(result[3],result[4])

    def test_e3_three_resampling_axes_and_prespecified_gate(self):
        data,rng = mock(111)
        data["REF-EXCLUDED/observed"] = data.compound_id.le("f")
        for mode in MODES:
            for ref in REFERENCES:
                for a in ARMS:
                    data[column(mode,ref,a)] = data.truth if a == "H2-v2" else -data.truth
        # Each fold has its own native screens and independent sample/chemical groups.
        panels=[]
        groups={}
        for fold in range(3):
            tables = []
            for s in range(3):
                z=data.assign(source_id=f"Screen{s}",lineage_group=data.lineage_group+str(fold),
                              compound_id=data.compound_id+str(fold))
                groups.update({k:k for k in z.compound_id.unique()})
                tables.append(z)
            panels.append((fold,pd.concat(tables)))
        stats,notes=e3_scopes(panels,pd.Series(groups),"DRP")
        ci=pd.DataFrame([d for _, rows in stats for d in rows])
        self.assertEqual(e3_policy(ci)["decision"],"GO-DOUBLE-COLD")
        self.assertEqual(set(ci[ci.metric.eq("spearman")].resampling),{"scaffold","lineage","scaffold_x_lineage"})
        self.assertEqual(notes["all"]["rows"],720)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out",type=Path)
    args=cli.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    publish_json(args.out,{"tests":result.testsRun,"outcomes_read":0,"status":"PASS"})
