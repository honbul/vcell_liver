#!/usr/bin/env python
"""F3 QC: compare our pseudobulk deltas / Welch-BH DEGs with the independently processed
HKU CRISPRi12 deposit (figshare 10.25442/hku.34002315.v1, CC BY-NC-SA 4.0; crispyx v0.1.4).
Used ONLY as a consistency check of our truth construction, never as model input or truth.
Checks file md5 against the deposit MANIFEST first.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HKU = Path("/mnt/S1/sdata/pubdata/Virtual_Cell/HKU_CRISPRi12_2026")
PREFIX = {"K562": "Replogle-E-k562", "RPE1": "Replogle-E-rpe1", "HepG2": "Nadig-HEPG2", "Jurkat": "Nadig-JURKAT"}


def md5(p, bs=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def idx(f, grp):
    key = f[grp].attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return strs(f[f"{grp}/{key}"][:])


def strs(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets/f3")
    ap.add_argument("--out", default="reports/feedback/F3")
    a = ap.parse_args()
    manifest = {r["file"]: r["md5"] for r in csv.DictReader(open(HKU / "MANIFEST.csv"))}
    rows, prov = [], {}
    for L, pre in PREFIX.items():
        fe = HKU / f"effect_{pre}_moments.h5ad"; fd = HKU / f"DE_{pre}_wilcoxon_batch_corrected.h5ad"
        prov[L] = {f.name: dict(manifest=manifest.get(f.name), local=md5(f)) for f in (fe, fd)}
        for v in prov[L].values():
            v["match"] = v["manifest"] == v["local"]
        z = np.load(Path(a.assets) / "matrices" / f"{L}_delta.npz", allow_pickle=False)
        ours = {k: z[k] for k in ("genes", "targets", "delta", "deg")}
        pbz = np.load(Path(a.assets) / "pseudobulk" / f"{L}.npz", allow_pickle=False)
        id2sym = dict(zip(pbz["var_ids"], pbz["var_names"]))
        osym = np.array([id2sym[g] for g in ours["genes"]])
        with h5py.File(fe, "r") as f:
            ep = idx(f, "obs"); eg = idx(f, "var"); E = f["X"][:]
        with h5py.File(fd, "r") as f:
            dp = idx(f, "obs"); dg = idx(f, "var"); Q = f["layers/pvalue_adj"][:]
        print(L, "labels", ep[:3], dp[:3], eg[:3], flush=True)
        dppos = {p: i for i, p in enumerate(dp)}; dgpos = {g: i for i, g in enumerate(dg)}
        gpos = {g: i for i, g in enumerate(eg)}
        gk = np.array([i for i, s in enumerate(osym) if s in gpos]); hk = np.array([gpos[osym[i]] for i in gk])
        ppos = {p: i for i, p in enumerate(ep)}
        r_list, jac, n_ours, n_hku = [], [], [], []
        for i, t in enumerate(ours["targets"]):
            if t not in ppos:
                continue
            x = ours["delta"][i, gk]; y = E[ppos[t], hk]
            r_list.append(np.corrcoef(x, y)[0, 1])
            if t not in dppos:
                continue
            A = ours["deg"][i, gk]; B = Q[dppos[t], [dgpos[eg[j]] for j in hk]] < 0.05
            n_ours.append(A.sum()); n_hku.append(B.sum())
            u = (A | B).sum()
            jac.append((A & B).sum() / u if u else np.nan)
        rows.append(dict(line=L, n_targets_common=len(r_list), n_genes_common=len(gk),
                         median_pearson_delta_vs_hku_effect=float(np.median(r_list)),
                         q05_pearson=float(np.percentile(r_list, 5)),
                         median_deg_ours=float(np.median(n_ours)), median_deg_hku_vanElteren=float(np.median(n_hku)),
                         median_deg_jaccard=float(np.nanmedian(jac)),
                         total_deg_ratio_ours_over_hku=float(np.sum(n_ours) / max(1, np.sum(n_hku)))))
        print(rows[-1], flush=True)
    out = Path(a.out)
    pd.DataFrame(rows).to_csv(out / "crosscheck_hku.tsv", sep="\t", index=False, float_format="%.4f")
    (out / "crosscheck_hku_provenance.json").write_text(json.dumps(dict(
        deposit="figshare 10.25442/hku.34002315.v1 (2026-10-13), CC BY-NC-SA 4.0, Liu, Zhang, Chen, Du, Zhao, Wang et al.; crispyx v0.1.4",
        use="QC cross-check only; not used as truth or model input", files=prov), indent=1))


if __name__ == "__main__":
    main()
