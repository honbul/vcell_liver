"""Print markdown tables from reports/feedback/R3/r3_summary.tsv + r3_bootstrap.tsv and write r3_b_vs_c.tsv
(cross-line (b) vs within-SNU-423 (c) paired drug-clustered bootstrap). Usage: r3_report_tables.py <R3 dir> tr49,tr48"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/feedback/R3")
S = pd.read_csv(R / "r3_summary.tsv", sep="\t")
B = pd.read_csv(R / "r3_bootstrap.tsv", sep="\t")
ts_list = sys.argv[2].split(",") if len(sys.argv) > 2 else ["tr49"]
f = lambda v, d=4: "–" if pd.isna(v) else f"{v:.{d}f}"

for ts in ts_list:
    for cn in ["aW", "aP", "aP_repsubset", "b", "c"]:
        for st in ["all", "detectable", "strong", "not_detectable"]:
            x = S[(S.condition == cn) & (S.train_set == ts) & (S.stratum == st)]
            if x.empty:
                continue
            nc = x[x.predictor == "no_change"].rmse
            base = float(nc.iloc[0]) if len(nc) else float("nan")
            print(f"\n#### {cn} / {ts} / {st} (units={int(x.n_units.max())}, drugs={int(x.n_drugs.max())})\n")
            print("| predictor | info | RMSE | ΔRMSE vs no-change [95% CI] | Pearson | pert-centred r | retrieval | DEG AUPRC | DEG prec@k |")
            print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in x.sort_values("rmse").iterrows():
                b = B[(B.condition == cn) & (B.train_set == ts) & (B.stratum == st) & (B.metric == "rmse") &
                      (B.predictor == r.predictor) & (B.baseline == "no_change")]
                ci = f"{b.mean_diff.iloc[0]:+.4f} [{b.ci_lo.iloc[0]:+.4f}, {b.ci_hi.iloc[0]:+.4f}]" if len(b) else "–"
                print(f"| {r.predictor} | {r['info']} | {f(r.rmse)} | {ci} | {f(r.pearson, 3)} | {f(r.pearson_pertcentered, 3)} | "
                      f"{f(r.retrieval_rank, 3)} | {f(r.deg_auprc, 3)} | {f(r.deg_prec_at_k, 3)} |")
            ref = B[(B.condition == cn) & (B.train_set == ts) & (B.stratum == st) & B.ref_is_best_nonsharing]
            if len(ref):
                print(f"\nvs best non-sharing ref `{ref.baseline.iloc[0]}`:")
                for _, r in ref[ref.metric.isin(["rmse", "pearson_pertcentered"])].iterrows():
                    print(f"- {r.predictor} {r.metric}: {r.mean_diff:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] (n={r.n_units})")

# ---- cross-line (b) vs within-SNU-423 (c) on the same test units: drug-clustered paired bootstrap
import numpy as np

U = pd.read_csv(R / "r3_unit_metrics.tsv.gz", sep="\t")
rows = []
rng = np.random.default_rng(20261113)
for ts in ts_list:
    u = U[U.train_set == ts]
    for pb, pc in [("dose_mean", "dose_mean_snu"), ("fp_kernel_ridge", "fp_kernel_ridge_snu"), ("global_mean", "global_mean_snu"),
                   ("moa_dose_mean", "moa_dose_mean_snu")]:
        for st, fl in [("all", lambda d: d), ("detectable", lambda d: d[d.detectable]), ("strong", lambda d: d[d.strong])]:
            for m in ["rmse", "pearson", "pearson_pertcentered"]:
                a = fl(u[(u.condition == "b") & (u.predictor == pb)]).set_index("unit")
                c = fl(u[(u.condition == "c") & (u.predictor == pc)]).set_index("unit")
                j = a[[m, "drug"]].join(c[[m]], rsuffix="_c", how="inner").dropna()
                if len(j) < 5:
                    continue
                d = (j[m] - j[m + "_c"]).to_numpy()
                ud = pd.Index(sorted(j.drug.unique())); code = ud.get_indexer(j.drug)
                sd = np.bincount(code, weights=d, minlength=len(ud)); nd = np.bincount(code, minlength=len(ud))
                W = np.stack([np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud)) for _ in range(2000)])
                bs = (W @ sd) / np.maximum(W @ nd, 1)
                rows.append(dict(train_set=ts, stratum=st, metric=m, cross_line=pb, within_line=pc, n_units=len(j),
                                 cross_mean=j[m].mean(), within_mean=j[m + "_c"].mean(), diff_cross_minus_within=d.mean(),
                                 ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
bc = pd.DataFrame(rows)
bc.to_csv(R / "r3_b_vs_c.tsv", sep="\t", index=False, float_format="%.5g")
print("\n#### b (cross-line) vs c (within SNU-423), same units\n")
print(bc.round(4).to_string(index=False))
