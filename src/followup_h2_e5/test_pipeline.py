"""Run E5 numerical and split invariants against the measured outcome-blind forecasts."""
import argparse
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from followup_h2.common import panel_hash, sha256
from followup_h2_e4.data import cpu_threads, registry
from followup_h2_e4.forecast import frozen_components, outcomes_locked
from followup_h2_e4.forecast import licob18
from followup_h2_e4.selection import role_data
from followup_h2_e4.split import overlaps
from followup_h2_e4.test_pipeline import licence
from nextgen_h2.assets import KEY
from nextgen_h2.run import META
from nextgen_h2.score import adult_licob
from nextgen_h2_v2.freeze import gram_receipt
from vc_build.structured_kernel import solve_residuals
from vc_build.io import publish_json

from .forecast import (
    E4_ARM, P1_OFFSET, P1_PERSONAL, P2_OFFSET, V2, active_task, choice_lock, pinned_e4, read_frozen,
)
from .kernel import DEBIASED, ScreenKernel, System, pcg, solve_shifts
from .select import P1,P2
from followup_h2_e4.data import utc_now


def aggregates(frame, lic):
    # Average the common personal prediction once, then restore each constant anchor.
    block=adult_licob(frame.drop(columns=[P1,P2]),lic)
    m=frame.groupby(KEY)[[P1_OFFSET,P2_OFFSET]].first()
    for key in m:
        assert frame.groupby(KEY)[key].nunique().max()==1
        block[key]=block[KEY].map(m[key])
    return block.assign(**{P1:block[P1_PERSONAL]+block[P1_OFFSET],P2:block[P1_PERSONAL]+block[P2_OFFSET]})


def same_ranks(panel):
    x,y=panel[P1].to_numpy(),panel[P2].to_numpy()
    counts={"all":0,"lincs_observed":0,"lincs_unobserved":0}
    for idx in panel.groupby(["source_id","endpoint_id",KEY],sort=False).indices.values():
        assert np.array_equal(rankdata(x[idx]),rankdata(y[idx]))
        present=panel.lincs_observed.iloc[idx]
        assert present.nunique()==1
        counts["all"]+=1
        counts["lincs_observed" if present.iloc[0] else "lincs_unobserved"]+=1
    return counts


def tau0(path):
    """Compare independent existing and task solvers on a pinned finite fit subgrid."""
    with np.load(path,allow_pickle=False) as f:
        x={k:torch.from_numpy(f[k]).double() for k in ("cell","drug","task_mass","task_y","orig_mass","orig_response")}
    mass,y=x["task_mass"],x["task_y"]
    response=torch.where(mass>0,y/mass.clamp_min(1e-12),0.)
    system=System(x["cell"],x["drug"],mass,response,tau=0.,stacked=True)
    out={}
    for penalty in (.25,.5,1.,2.,4.,8.,16.,32.):
        new,_,_=pcg(system,penalty,1e-9)
        prev,_,_=solve_residuals(
            x["cell"],x["drug"],x["orig_mass"],x["orig_response"],penalty,tolerance=1e-10,max_steps=4000)
        actual=(x["cell"]@new.sum(0))@x["drug"]
        original=(x["cell"]@prev)@x["drug"]
        out[str(penalty)]=float((actual-original).abs().max())
    return out


def synthetic_numerics():
    torch.manual_seed(42791)
    a,b=torch.randn(7,7,dtype=torch.float64),torch.randn(8,8,dtype=torch.float64)
    c,d=a@a.T+.3*torch.eye(7),b@b.T+.4*torch.eye(8)
    w=torch.rand((3,7,8),dtype=torch.float64)
    w[w<.31]=0
    y=torch.randn_like(w)
    system=System(c,d,w,y,tau=.5,stacked=True)
    k=torch.kron(torch.ones((3,3),dtype=torch.float64)+.5*torch.eye(3),torch.kron(c,d))
    r=w.sqrt().reshape(-1)
    exact=r[:,None]*k*r[None,:]
    assert torch.linalg.eigvalsh(k)[0]>0
    errors={}
    for p,(coef,_,res,_) in solve_shifts(system,(.25,1.,4.,32.),tol=1e-9,max_steps=500).items():
        true=r*torch.linalg.solve(exact+p*torch.eye(len(exact)),(w.sqrt()*y).reshape(-1))
        errors[str(p)]=float((true-coef.reshape(-1)).abs().max())
        assert errors[str(p)]<1e-7 and res<=1e-9
    return errors


def verify(preflight,selected,forecasts,out):
    cpu_threads()
    split,choice,choice_hash=choice_lock(preflight,selected)
    outcomes_locked()
    reg=registry()
    packed=[read_frozen(forecasts/str(i),choice_hash) for i in range(7)]
    counts,records,checks,zeros,transforms={},{},{},{},{}

    class Contracts(unittest.TestCase):
        def test_zero_overlaps_for_every_role(self):
            for role,item in split["roles"].items():
                self.assertEqual(set(overlaps(reg,item["fit_parents"],item["query_parents"]).values()),{0})
                if item["kind"]=="loso":
                    self.assertFalse(set(item["fit_screens"])&set(item["query_screens"]))
                if item["outer_overlap"] is not None:
                    parent="/".join(role.split("/")[:2])+"/outer"
                    outer=split["roles"][parent]["query_parents"]
                    self.assertEqual(set(overlaps(reg,item["fit_parents"]+item["query_parents"],outer).values()),{0})

        def test_frozen_baselines(self):
            for i,(panel,note) in enumerate(packed):
                old,_=pinned_e4(i)
                self.assertTrue(panel[META].equals(old[META]))
                for arm in ("C1","H2-v1",E4_ARM):
                    self.assertLess(float(np.max(np.abs(panel[arm]-old[arm]))),1e-6)
                if i>=3:
                    previous,replay=frozen_components(note["fold"])
                    self.assertTrue(panel[META].equals(previous[META]))
                    for arm,key in (("H2-v1","F00_max_abs_diff"),(V2,"F11_max_abs_diff")):
                        self.assertLess(replay[key],1e-6)
                        self.assertLess(float(np.max(np.abs(panel[arm]-previous[arm]))),1e-6)
                records[str(i)]=note["replays"]

        def test_intercept_only_perturbation(self):
            for i,(panel,note) in enumerate(packed):
                for screen,data in panel.groupby("source_id"):
                    if screen=="LICOB":
                        data=aggregates(data,licence())
                        novel=licob18(data)
                        self.assertEqual((len(novel),novel[KEY].nunique()),(684,18))
                        counts["6/LICOB18"]=same_ranks(novel)
                    counts[f"{i}/{screen}"]=same_ranks(data)

        def test_selected_grams_and_tau_zero_equivalence(self):
            for i,(panel,note) in enumerate(packed):
                f=forecasts/str(i)
                for arm,rec in note["selected_grams"].items():
                    gram=f/rec["file"]
                    self.assertEqual(sha256(gram),rec["sha256"])
                    eig=gram_receipt(torch.from_numpy(np.load(gram,allow_pickle=False)))
                    self.assertGreater(eig["minimum_eigenvalue"],0.)
                    self.assertAlmostEqual(eig["minimum_eigenvalue"],rec["minimum_eigenvalue"],places=8)
                    task=note["details"][arm]["task_gram"]
                    if task:
                        self.assertEqual(active_task(task["tasks"],task["tau"]),task)
                path=f/"tau0-inputs.npz"
                self.assertEqual(sha256(path),note["tau0_input_sha256"])
                zeros[str(i)]=tau0(path)
                self.assertLess(max(zeros[str(i)].values()),1e-4)
                checks[str(i)]={"drug":note["selected_grams"],
                                "task":{k:v["task_gram"] for k,v in note["details"].items() if v["task_gram"]}}
            zeros["synthetic_hierarchical_dense"]=synthetic_numerics()

        def test_fit_only_screen_centering_and_scale(self):
            for i,(_,note) in enumerate(packed):
                p=note["details"][DEBIASED]["transform"]
                self.assertTrue(p["fit_only"])
                self.assertEqual(p["fit_rows"],note["fit_rows"])
                self.assertGreater(p["scale"],0.)
                self.assertEqual(sum(x["rows"] for x in p["calibration"]),note["fit_rows"])
                for measure in p["calibration"]:
                    self.assertLess(measure["max_cell_mean_error"],1e-8)
                    self.assertLess(measure["max_drug_mean_error"],1e-8)
                    self.assertLess(measure["max_unit_variance_error"],1e-8)
                transforms[str(i)]=p
            role="T1/2/outer"
            d,fit,q=role_data(split["roles"][role],reg,inner=False)
            a=ScreenKernel(d,fit,q,reg,role)
            poison=q.assign(endpoint_value=np.arange(len(q))*1234567.,endpoint_direction="invalid")
            b=ScreenKernel(d,fit,poison,reg,role)
            u,v=a.targets_for(DEBIASED),b.targets_for(DEBIASED)
            self.assertEqual(list(b.query.columns),META)
            self.assertTrue(torch.equal(u[0],v[0]) and torch.equal(u[1],v[1]))
            self.assertEqual(u[2:],v[2:])
            pd.testing.assert_series_equal(a.anchor,b.anchor)
            self.assertEqual(panel_hash(fit),split["roles"][role]["fit_roster_sha256"])

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    out.mkdir(exist_ok=False,parents=True)
    publish_json(out/"test_receipt.json",{
        "status":"PASS","tests":result.testsRun,"passed_at_utc":utc_now(),"queried_outcomes":0,
        "selection_sha256":choice_hash,"split_sha256":choice["split_sha256"],
        "forecast_receipt_sha256":{str(i):sha256(forecasts/str(i)/"receipt.json") for i in range(7)},
        "replays":records,"rank_groups":counts,"intercept_within_drug_delta":0.,
        "selected_grams":checks,"tau0_max_abs_diffs":zeros,"fit_only_transforms":transforms,
        "audited_splits":len(split["roles"]),
    })


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("preflight","selected","forecasts","out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    verify(a.preflight,a.selected,a.forecasts,a.out)
