"""Protocol amendment 1: list source rows whose control or treated profile is degenerate (mean over the 978 landmarks < 1).

Only source rows are examined and removed. Target rows are never filtered; for them only the X0 row means are printed.
"""
import os
import numpy as np
import pandas as pd

from .data import MAN_DIR

P = os.path.join(os.environ["TB_PROC"], "cmap2020_v1")
rows = pd.read_csv(os.path.join(P, "rows.tsv"), sep="\t")
x1 = np.load(os.path.join(P, "x1_source.npy"), mmap_mode="r")
x0 = np.load(os.path.join(P, "x0_all.npy"), mmap_mode="r")
n = len(x1)
m1, m0 = np.asarray(x1).mean(1), np.asarray(x0[:n]).mean(1)
bad = (m1 < 1) | (m0 < 1)
out = rows.iloc[:n][bad].assign(x0_mean=m0[bad], x1_mean=m1[bad])
out.to_csv(os.path.join(MAN_DIR, "source_rows_excluded_degenerate.tsv"), sep="\t", index=False, float_format="%.4f")
print("excluded source rows:", int(bad.sum()), "| plates:", sorted(out.det_plate.unique()), "| cells:", sorted(out.cell.unique()))
print("remaining source row means: x0", m0[~bad].min(), m0[~bad].max(), "x1", m1[~bad].min(), m1[~bad].max())
mt = np.asarray(x0[n:]).mean(1)
print("target X0 row means (no target row is filtered):", mt.min(), mt.max())
