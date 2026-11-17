"""Pre-specified paired contrasts (linear combinations of methods), shared by source, matched-plate and HCC summaries."""
import numpy as np
import pandas as pd

ADAPTERS = ("B00", "B01", "B10", "B11")       # arms that also get the shuffled-residual control


def _d(a, b):
    return {a: 1.0, b: -1.0}


CONTRASTS = {
    # residual value beyond the common correction
    "C1_iid-C0_iid": _d("C1_iid", "C0_iid"), "B00-C0_iid": _d("B00", "C0_iid"), "B00-C1_iid": _d("B00", "C1_iid"),
    "C1_cov-C0_cov": _d("C1_cov", "C0_cov"), "B10-C0_cov": _d("B10", "C0_cov"), "B10-C1_cov": _d("B10", "C1_cov"),
    # 2 x 2: state preservation x observation weighting
    "B01-B00": _d("B01", "B00"), "B10-B00": _d("B10", "B00"), "B11-B00": _d("B11", "B00"), "B11-B01": _d("B11", "B01"), "B11-B10": _d("B11", "B10"),
    "interaction(B11-B10-B01+B00)": {"B11": 1.0, "B10": -1.0, "B01": -1.0, "B00": 1.0},
    "C0_cov-C0_iid": _d("C0_cov", "C0_iid"), "C1_cov-C1_iid": _d("C1_cov", "C1_iid"),
    # global-state ablations
    "G_concat-B00": _d("G_concat", "B00"), "G_dummy-B00": _d("G_dummy", "B00"), "B01-G_dummy": _d("B01", "G_dummy"), "B01-G_concat": _d("B01", "G_concat"),
    # shuffled support residuals (does the per-drug residual assignment matter?)
    "B00__shuffle_residual-B00": _d("B00__shuffle_residual", "B00"), "B01__shuffle_residual-B01": _d("B01__shuffle_residual", "B01"),
    "B10__shuffle_residual-B10": _d("B10__shuffle_residual", "B10"), "B11__shuffle_residual-B11": _d("B11__shuffle_residual", "B11"),
    # revision R-1: repeat-informed output basis
    "B00r-B00": _d("B00r", "B00"), "C1r_iid-C1_iid": _d("C1r_iid", "C1_iid"), "B00r-C0_iid": _d("B00r", "C0_iid"),
    # revision R-2: second observation model (diagonal only, signal variance x64)
    "B10v2-B00": _d("B10v2", "B00"), "B11v2-B00": _d("B11v2", "B00"), "B11v2-B01": _d("B11v2", "B01"), "B10v2-B10": _d("B10v2", "B10"),
    "interaction_v2(B11v2-B10v2-B01+B00)": {"B11v2": 1.0, "B10v2": -1.0, "B01": -1.0, "B00": 1.0},
    "C0_cov2-C0_iid": _d("C0_cov2", "C0_iid"), "C1_cov2-C1_iid": _d("C1_cov2", "C1_iid"), "B10v2-C0_cov2": _d("B10v2", "C0_cov2"), "B10v2-C1_cov2": _d("B10v2", "C1_cov2"),
    # simple and public references
    "C0_iid-no_change": _d("C0_iid", "no_change"), "C0_iid-same_drug_mean": _d("C0_iid", "same_drug_mean"), "C0_iid-shrunk_same_drug_mean": _d("C0_iid", "shrunk_same_drug_mean"),
    "C0_iid-mean_offset": _d("C0_iid", "mean_offset"), "mean_offset-same_drug_mean": _d("mean_offset", "same_drug_mean"),
    "B00-mean_offset": _d("B00", "mean_offset"), "B00-same_drug_mean": _d("B00", "same_drug_mean"),
    "cnp-C0_iid": _d("cnp", "C0_iid"), "B00-cnp": _d("B00", "cnp"),
    "transigen_full_ft-C0_iid": _d("transigen_full_ft", "C0_iid"), "B00-transigen_full_ft": _d("B00", "transigen_full_ft"),
    "transigen_offset-C0_iid": _d("transigen_offset", "C0_iid"), "transigen_full_ft-transigen_offset": _d("transigen_full_ft", "transigen_offset"),
    "transigen_as_is-same_drug_mean": _d("transigen_as_is", "same_drug_mean"),
}


def family_contrasts(fam_tab, value, by=("K",), n_boot=2000, seed=20261117, contrasts=None):
    """fam_tab rows: by..., method, family, value. Paired family bootstrap of every available pre-specified contrast."""
    out = []
    by = list(by)
    for key, g in fam_tab.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        p = g.pivot(index="family", columns="method", values=value)
        for name, comb in (contrasts or CONTRASTS).items():
            if not all(m in p for m in comb):
                continue
            d = sum(c * p[m] for m, c in comb.items()).dropna().to_numpy()
            if len(d) < 2:
                continue
            rs = np.random.RandomState(seed)
            bs = d[rs.randint(len(d), size=(n_boot, len(d)))].mean(1)
            out.append(dict(zip(by, key), contrast=name, metric=value, diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                            n_families=len(d), n_families_negative=int((d < 0).sum())))
    return pd.DataFrame(out)
