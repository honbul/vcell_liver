"""Revision R-2, adoption step: turn the adopted closed-form variant into the second observation model 'cov2'.
kappa_f as selected under equal weights, kappa_b per K as selected for the variant on the dev episodes, and the residual
regression of C1 keeps the lambda / eta selected under equal weights (the variant is close to equal weights)."""
import json, os
import pandas as pd

from .common import public, dump, man_file


def main():
    dec = json.load(open(man_file("cov_strength_decision.json")))
    assert dec["adopt"], "revision R-2 was not adopted by its gate"
    sel = json.load(open(man_file("anchor_selection.json")))
    tab = pd.read_csv(man_file("cov_strength_dev.tsv"), sep="\t")
    v = dec["best_k16"]["variant"]
    structure, scale = v.split("|x")
    t = tab[tab.method == v].set_index("K")
    anchor = {str(K): dict(kappa_f=sel["anchor"]["iid"][str(K)]["kappa_f"], kappa_b=float(t.kappa_b[K]), dev=float(t.delta_rmse[K])) for K in (8, 16, 32)}
    cov = dict(sel["covariance_model"], v_r_scale=float(scale), structure=structure)
    dump(dict(anchor=anchor, c1=sel["c1"]["iid"], covariance_model=cov, source=dict(decision=dec, variant=v, commit=os.environ.get("R8_COMMIT", ""))),
         public("manifests", "replicate_calibrated_r8", "cov2_selection.json"))
    print(json.dumps(anchor), cov["v_r_scale"], cov["structure"])


if __name__ == "__main__":
    main()
