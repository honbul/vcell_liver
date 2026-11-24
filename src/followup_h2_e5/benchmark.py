"""Measure two exact numerical paths on a registered source-only role."""
import argparse
from pathlib import Path
from time import monotonic

from followup_h2_e4.data import cpu_threads,registry
from followup_h2_e4.model import Candidate,settings
from followup_h2_e4.selection import role_data
from nextgen_h2.run import META

from .kernel import DEBIASED,TASK,ScreenKernel,System,solve_shifts
from .registration import read_split,source_only
from .spectral import SpectralBasis,solve_block,solve_hybrid


def benchmark(preflight,hybrid=False,block=False):
    source_only()
    split=read_split(preflight)
    role="T3/3/Sel-Pooled/0"
    reg=registry()
    data,fit,query=role_data(split["roles"][role],reg,inner=True)
    model=ScreenKernel(data,fit,query[META],reg,role)
    g,_=model.geometry(settings(role,Candidate("v2_base","cos",.125,.25)))
    for family,tau in ((DEBIASED,0.),(TASK,.25)):
        w,y,_,_=model.targets_for(family)
        system=System(model.cell,g[:model.stop,:model.stop].contiguous(),w,y,tau,family==TASK)
        start=monotonic()
        if hybrid or block:
            ans=(solve_block if block else solve_hybrid)(system,(.25,.5,1.,2.,4.,8.,16.,32.))
            print(family,"block" if block else "hybrid",[(p,a[1:]) for p,a in ans.items()],
                  "time",round(monotonic()-start,3),flush=True)
            continue
        spectral=SpectralBasis(system)
        for p in (.25,1.,4.,32.):
            begin=monotonic()
            _,n,e=spectral.solve(p)
            print(family,tau,"spectral",p,n,e,"time",round(monotonic()-begin,3),flush=True)
        print(family,"spectral setup + solves",round(monotonic()-start,3),flush=True)
        start=monotonic()
        ans=solve_shifts(system,(.25,.5,1.,2.,4.,8.,16.,32.))
        print(family,"shifted",[(p,answer[1:]) for p,answer in ans.items()],
              "time",round(monotonic()-start,3),flush=True)


if __name__=="__main__":
    cpu_threads()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("preflight",type=Path)
    p.add_argument("--hybrid",action="store_true")
    p.add_argument("--block",action="store_true")
    args=p.parse_args()
    benchmark(args.preflight,args.hybrid,args.block)
