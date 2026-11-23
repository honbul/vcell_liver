"""Run numerical scoring routines on the registered key density with simulated outcomes."""
from collections import Counter

import numpy as np
from nextgen_h2.run import META
from h3v2.data import read_data, register
from h3v2.error_components import decompose
from h3v2.final_fit import exact_join, old_h2
from h3v2.score import native_metric_points, units, DRUG, CELL


def main():
    expected=read_data("drug-0","query")
    ref,h=old_h2(0)
    ref=exact_join(expected,ref)
    rng=np.random.default_rng(338901)
    simulated=expected[META].assign(
        truth=rng.standard_normal(len(expected)),
        H2=ref.champion.to_numpy(),C1=ref.C1.to_numpy(),
        lincs=ref.champion_observed.to_numpy(),
    )
    scopes={"whole":None,"mode_observed":simulated.lincs,"mode_unobserved":~simulated.lincs}
    d,c=units(simulated,register().scaffold_block,0,"synthetic-rehearsal",scopes,["H2","C1"])
    decomposition=decompose(simulated,["H2","C1"],"synthetic-rehearsal",0,scopes)
    assert len(simulated)==218445 and len(scopes)==3
    assert len(native_metric_points(d,DRUG))>0 and len(native_metric_points(c,CELL))>0
    assert all(n["rank_mse"]>0 for n in decomposition)
    print("Simulated outcome rehearsal on the registered key density",len(simulated),
          dict(Counter(x["support"] for x in decomposition)),h)
    print("Registered held outcomes were not sourced by this rehearsal.")


if __name__=="__main__":
    main()
