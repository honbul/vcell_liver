# Provenance: copied unchanged (below this header) from commit 43485ea4824c98a2fef93ebe4ad043c98c5a0a41,
# path src/r7_followup_v1/stats.py (branch research/r7-followup-target-attribution-v1); not merged.
"""Family-balanced statistic with a paired drug-group bootstrap (all cell lines of a drug group are resampled together).

statistic(d) = mean over families of [ sum_u w_g(u) d_u / sum_u w_g(u) ]   (families without a resampled unit are dropped)
This is the aggregation of the original evaluator (family_mean / paired_bootstrap); here every contrast uses the SAME
resampling weights, so intervals of several contrasts are jointly valid for Bonferroni / Holm statements.
Differences are candidate - comparator; negative = the candidate has the lower error."""
import numpy as np

THR = 0.001      # operating criterion of the original study (absolute delta-RMSE); not a biological threshold


class FamBoot:
    def __init__(self, drug_group, family, n_boot=5000, seed=20261118):
        self.ug, self.gi = np.unique(np.asarray(drug_group), return_inverse=True)
        self.uf, self.fi = np.unique(np.asarray(family), return_inverse=True)
        rng = np.random.RandomState(seed)
        G = len(self.ug)
        self.W = np.stack([np.bincount(rng.randint(0, G, G), minlength=G) for _ in range(n_boot)]).astype(np.float32)
        self.n = len(self.gi)

    def _mats(self, d, mask):
        F, G = len(self.uf), len(self.ug)
        D = np.zeros((F, G)); C = np.zeros((F, G))
        k = np.where(mask)[0] if mask is not None else np.arange(self.n)
        np.add.at(D, (self.fi[k], self.gi[k]), np.asarray(d, np.float64)[k]); np.add.at(C, (self.fi[k], self.gi[k]), 1.0)
        return D, C

    def stat(self, d, mask=None, boot=True):
        """Point value and bootstrap draws of the family-balanced mean of d over the masked units."""
        D, C = self._mats(d, mask)
        den = C.sum(1); ok = den > 0
        point = float((D.sum(1)[ok] / den[ok]).mean()) if ok.any() else np.nan
        if not boot:
            return point, None
        num = self.W @ D.T.astype(np.float32); dn = self.W @ C.T.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(dn > 0, num / dn, np.nan)
        return point, np.nanmean(r, 1)

    def unit_mean(self, d, mask=None, boot=True):
        """Plain unit mean (HCC single-line panels of the original evaluator), same resampling."""
        k = mask if mask is not None else np.ones(self.n, bool)
        G = len(self.ug)
        num = np.bincount(self.gi[k], weights=np.asarray(d, np.float64)[k], minlength=G); den = np.bincount(self.gi[k], minlength=G).astype(float)
        point = float(num.sum() / den.sum()) if den.sum() else np.nan
        if not boot:
            return point, None
        with np.errstate(invalid="ignore", divide="ignore"):
            return point, (self.W @ num.astype(np.float32)) / (self.W @ den.astype(np.float32))


def interval(bs, level=0.95):
    a = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(bs, [a, 100 - a])
    return float(lo), float(hi)


def boot_p(bs):
    """Two-sided bootstrap p for 'difference = 0' (percentile inversion with the +1 correction)."""
    bs = bs[~np.isnan(bs)]
    n = len(bs)
    return float(min(1.0, 2 * min(((bs >= 0).sum() + 1) / (n + 1), ((bs <= 0).sum() + 1) / (n + 1))))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def category(diff, lo, hi, thr=THR):
    """Decision categories of plan section 3.4 (candidate - comparator, negative = improvement)."""
    if np.isnan(diff):
        return "not_available"
    if hi < 0:
        return "improvement_detected_operationally_meaningful" if diff <= -thr else "small_improvement_detected"
    if lo > 0:
        return "worse"
    return "uncertain_meaningful_improvement_not_excluded" if lo <= -thr else "no_meaningful_improvement_supported"


def contrast_row(fb, d, mask=None, core_m=1, family_balanced=True):
    point, bs = (fb.stat if family_balanced else fb.unit_mean)(d, mask)
    lo, hi = interval(bs, 0.95)
    row = dict(diff=point, lo95=lo, hi95=hi, boot_p=boot_p(bs), category_ci95=category(point, lo, hi))
    if core_m > 1:
        slo, shi = interval(bs, 1 - 0.05 / core_m)
        row.update(lo_simultaneous=slo, hi_simultaneous=shi, category_simultaneous=category(point, slo, shi))
    return row
