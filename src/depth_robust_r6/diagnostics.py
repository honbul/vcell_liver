"""R6-0: what count thinning does to the observed control state (controls only; no response read).

For the fixed eval_input sets (N=64) at p in {1, 0.5, 0.25}, plus N=16 subsets, per pool:
    molecules, detected panel genes, tokens available to a 256-token encoder
    set-mean log-normalised profile: shift against the full-observation set mean, in units of the
        between-pool spread of that gene (how large the depth effect is relative to what
        distinguishes cell states), and its correlation after removing the across-pool mean
    per-cell rank/token stability: overlap of a cell's top-256 genes with its full-observation top-256
    nearest-pool identification: is a thinned set's profile still closest to its own full-observation
        pool (before and after removing the mean depth shift)
The effect of these changes on delta prediction is measured by ridge.py --diagnostic (Z0_p1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from .count_view import lognorm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/depth_robust_r6/tables/r6_0_observation.json")
    a = ap.parse_args()
    d = D.load()
    P = len(d.pools); N = D.EVAL_N
    rows = np.concatenate([np.arange(r.eval_start, r.eval_end) for r in d.pools.itertuples()])
    full_c, full_r = d.cnt[rows], d.crest[rows]

    def prof(c, r, take=None):
        x = lognorm(c.astype(np.float64), (c.sum(1) + r).astype(np.float64)).reshape(P, N, -1)
        return (x[:, :take] if take else x)

    xf = prof(full_c, full_r)
    mf = xf.mean(1)
    between_sd = np.maximum(mf.std(0), 1e-6)
    tb = d.tiebreak

    def top(c):
        return np.argsort(-(c + tb), axis=1)[:, :256]

    top_full = top(full_c)
    out = []
    views = [("p1.0_N64", full_c, full_r, None), ("p1.0_N16", full_c, full_r, 16)]
    for (p, s), (c, r) in sorted(d.eval_views.items(), reverse=True):
        views.append((f"p{p}_s{s}_N64", c, r, None))
        if s == 0:
            views.append((f"p{p}_s{s}_N16", c, r, 16))
    for name, c, r, take in views:
        x = prof(c, r, take)
        m = x.mean(1)
        shift = m - mf
        mc, mfc = m - m.mean(0), mf - mf.mean(0)
        corr = [float(np.corrcoef(mc[i], mfc[i])[0, 1]) for i in range(P)]
        S = (mc / np.linalg.norm(mc, axis=1, keepdims=True)) @ (mfc / np.linalg.norm(mfc, axis=1, keepdims=True)).T
        S_raw = -((m[:, None, :] - mf[None]) ** 2).sum(-1)
        tp = top(c)
        det = (c > 0).sum(1)
        ov = np.array([len(np.intersect1d(tp[i][: min(256, det[i])], top_full[i])) / 256 for i in range(0, len(c), 8)])
        out.append(dict(
            view=name, n_cells_per_set=take or N,
            median_library=float(np.median(c.sum(1) + r)), median_panel_umi=float(np.median(c.sum(1))),
            median_detected_panel_genes=float(np.median(det)), frac_cells_with_fewer_than_256_tokens=float((det < 256).mean()),
            mean_lognorm=float(m.mean()), mean_within_set_var=float(x.var(1).mean()),
            setmean_shift_rms=float(np.sqrt((shift ** 2).mean())),
            setmean_shift_rms_in_between_pool_sd=float(np.sqrt(((shift / between_sd) ** 2).mean())),
            setmean_shift_after_removing_mean_depth_shift_rms=float(np.sqrt(((shift - shift.mean(0)) ** 2).mean())),
            centred_profile_corr_with_full_median=float(np.median(corr)),
            nearest_pool_correct_centred=float((S.argmax(1) == np.arange(P)).mean()),
            nearest_pool_correct_raw_euclid=float((S_raw.argmax(1) == np.arange(P)).mean()),
            top256_token_overlap_with_full_mean=float(ov.mean())))
        print(json.dumps(out[-1]), flush=True)
    truth_rms = float(np.sqrt((d.Y[d.valid].astype(np.float64) ** 2).mean()))
    D.dump_json(dict(n_pools=P, truth_delta_rms=truth_rms, between_pool_sd_median=float(np.median(between_sd)), views=out,
                     note="controls only; the full-observation N=64 eval_input set is the comparison point of every row"), a.out)


if __name__ == "__main__":
    main()
