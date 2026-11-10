#!/usr/bin/env python
"""G3 supplement: PIE vs best PIE-split baseline per fold/condition (reads pie_split_summary_<fold>.tsv).
Usage: $PIE_PY src/feedback/g3_pie_allfolds.py reports/feedback/G3"""
import pandas as pd, sys
from pathlib import Path
R = Path(sys.argv[1])
best = {"unseen_ctx": ["same_target_cal"], "unseen_pert": ["context_mean_kd", "ridge_inline_kd"],
        "unseen_ctx_pert": ["global_mean_kd", "ridge_stacked_kd"]}
rows = []
for f in ["hepg2", "jurkat", "k562", "rpe1"]:
    p = R / f"pie_split_summary_{f}.tsv"
    if not p.is_file():
        continue
    s = pd.read_csv(p, sep="\t")
    for st, bl in best.items():
        x = s[s.setting == st].set_index("predictor")
        b = x.loc[bl, "binary_auprc"].idxmax()
        pie = x.loc["PIE_repro"]
        bx = x.loc[b]
        rows.append(dict(fold=f, setting=st, best_baseline=b,
                         auprc_pie=pie.binary_auprc, auprc_base=bx.binary_auprc,
                         auprc_x_pie=pie.auprc_x, auprc_x_base=bx.auprc_x,
                         dir_x_pie=pie.direction_x, dir_x_base=bx.direction_x,
                         tgt_pct_pie=pie.target_pde_pct, pertc_pie=pie.dp_pertc_r,
                         pertc_ridge=x.loc["ridge_stacked_kd", "dp_pertc_r"],
                         rmse_pie=pie.dp_rmse, rmse_nochange=x.loc["no_change", "dp_rmse"],
                         auprc_pie_kd=x.loc["PIE_repro_kd", "binary_auprc"]))
df = pd.DataFrame(rows)
df.to_csv(R / "pie_split_allfolds_summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
