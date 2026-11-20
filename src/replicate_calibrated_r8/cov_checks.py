"""Assemble covariance_checks.json from the committed source-side tables: estimation basis, numerical checks, behaviour of the
observation model in the closed-form methods and the outcome of revision R-2. Recomputes nothing."""
import json, os
import pandas as pd

from .common import public, dump, man_file


def main():
    cov = json.load(open(man_file("covariance_model.json")))
    fx = json.load(open(man_file("fixtures.json")))
    ke = pd.read_csv(man_file("anchor_k_eff.tsv"), sep="\t")
    cf = pd.read_csv(man_file("closed_form_contrasts.tsv"), sep="\t")
    d1 = pd.read_csv(man_file("d1_plate_pair_summary.tsv"), sep="\t")
    dec = json.load(open(man_file("cov_strength_decision.json")))
    pick = lambda a, b: cf[(cf.a == a) & (cf.b == b)][["set", "K", "diff", "lo", "hi", "n_families"]].to_dict("records")
    out = dict(
        model=dict(formula="Omega_ij = [delta_ij (v_r + v_e q_i) + v_s o_ij (x0.9 off the diagonal)] / (v_r + v_e + v_s) + 0.05 delta_ij",
                   constants={k: cov[k] for k in ("v_r", "v_e", "v_s")}, provenance=cov["provenance"]),
        replicate_data_sufficient=cov["provenance"]["replicate_data_sufficient"],
        plate_pair_decomposition=d1.to_dict("records"),
        numerical=dict(solver=fx["anchor"], one_shared_plate=fx["one_plate"],
                       condition_number_max={f"{r.set}|K{r.K}": r.cond_max for r in ke[ke.obs == "cov"].itertuples()},
                       k_eff={f"{r.set}|K{r.K}": dict(mean=r.k_eff_mean, min=r.k_eff_min, max=r.k_eff_max, max_relative_weight=r.max_rel_weight) for r in ke[ke.obs == "cov"].itertuples()}),
        closed_form_effect=dict(C0_cov_minus_C0_iid=pick("C0_cov", "C0_iid"), C1_cov_minus_C1_iid=pick("C1_cov", "C1_iid")),
        revision_R2=dict(decision=dec, table="cov_strength_dev.tsv",
                         reading="every variant with the shared-plate block is not better than equal weights; the adopted diagonal-only variant differs from equal weights by about 1e-4"),
        not_verified=["a gene-by-gene covariance (only a K x K observation covariance scaled by gene-averaged variances is modelled)",
                      "the covariance of an HCC target's own plates (source-estimated constants are applied to the target's plate metadata)",
                      "whether X-tag replicate plates are independent cultures"])
    dump(out, public("manifests", "replicate_calibrated_r8", "covariance_checks.json"))
    print(json.dumps(out["closed_form_effect"], indent=1)[:1500])


if __name__ == "__main__":
    main()
