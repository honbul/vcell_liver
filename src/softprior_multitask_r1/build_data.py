#!/usr/bin/env python3
"""Stage R1-0 (part 1): rebuild the gene axis with a fixed-ID duplicate-symbol rule and cache inputs.

Track A kept, among Ensembl IDs that share a gene symbol, the one with the highest mean LIMORE
expression. That rule looks at expression values. R1 keeps the lexicographically smallest Ensembl ID
instead; nothing else in the Track A build is changed (same universe, scales, rank transform,
Hallmark release and inclusion rule, permutation scheme). Registry, splits, drug matching and the
response tables are reused from Track A unchanged.

Writes matrices to assets/cache/data_<key>/ and small tables to manifests/softprior_multitask_r1/.
No prediction is made here; no response value is read.

Usage: build_data.py [--rscript PATH]   (Rscript with singscore, used once to cross-check the scores)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
from r1_data import a1core  # noqa: E402


def build_axis(cfg: dict):
    g = pd.read_csv(cfg["paths"]["limore_rna"], sep="\t")
    assert g.tracking_id.is_unique
    samples_l = list(g.columns[3:])
    lccl = a1core.read_cr_table(cfg["paths"]["lccl_rna"])
    lccl["ens"] = lccl["Gene_ID"].str.replace(r"\.\d+$", "", regex=True)
    assert not lccl.ens.duplicated().any()
    samples_c = list(lccl.columns[2:-1])
    pc = g[g.geneType == "protein_coding"]
    pc = pc.assign(ens=pc.tracking_id.str.replace(r"\.\d+$", "", regex=True))
    assert pc.ens.is_unique
    axis = pc[pc.ens.isin(set(lccl.ens))].copy()
    n_before = len(axis)
    dup = axis[axis.geneSymbol.duplicated(keep=False)].sort_values(["geneSymbol", "ens"])
    # fixed-ID rule: smallest Ensembl ID per symbol; expression is not consulted
    axis = axis.sort_values(["geneSymbol", "ens"]).drop_duplicates("geneSymbol", keep="first")
    axis = axis.sort_values("ens").reset_index(drop=True)
    lc = lccl.set_index("ens").loc[axis.ens]
    X_l = axis[samples_l].values.T.astype(float)
    raw_c = lc[samples_c].values.T.astype(float)
    assert raw_c.min() >= 0 and raw_c.max() > 100
    X_c = np.log2(raw_c + 1.0)
    assert 0 <= X_l.min() and X_l.max() < 25 and X_c.max() < 25
    gene_axis = pd.DataFrame(dict(ensembl_id=axis.ens.values, symbol=axis.geneSymbol.values,
                                  lccl_symbol=lc["Gene_name"].values))
    dup_tab = dup[["geneSymbol", "ens"]].rename(columns={"geneSymbol": "symbol", "ens": "ensembl_id"})
    dup_tab["kept_r1"] = dup_tab.ensembl_id.isin(set(gene_axis.ensembl_id))
    return samples_l, X_l, samples_c, X_c, gene_axis, dup_tab, n_before


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rscript", default=None)
    args = ap.parse_args()
    cfg = R.load_config()
    for k in ("limore_rna", "lccl_rna", "hallmark_gmt"):
        got = R.sha256_file(cfg["paths"][k])
        assert got == cfg["expected_sha256"][k], f"{k}: {got}"
    out = R.cache_dir(cfg)
    os.makedirs(out, exist_ok=True)
    os.makedirs(R.MANIFESTS, exist_ok=True)

    samples_l, X_l, samples_c, X_c, gene_axis, dup_tab, n_before = build_axis(cfg)
    ids_l = [a1core.model_id("LIMORE", s) for s in samples_l]
    ids_c = [a1core.model_id("LCCL", s) for s in samples_c]
    reg = R.load_registry()
    assert set(ids_l) == set(reg.model_id[reg.study == "LIMORE"])
    assert set(ids_c) == set(reg.model_id[reg.study == "LCCL"])

    # comparison with the Track A axis
    old = pd.read_csv(os.path.join(R.A1_MANIFESTS, "gene_axis.tsv"), sep="\t")
    dup_tab["kept_track_a"] = dup_tab.ensembl_id.isin(set(old.ensembl_id))
    changed = dup_tab.groupby("symbol").apply(lambda t: bool((t.kept_r1 != t.kept_track_a).any()))
    assert set(gene_axis.symbol) == set(old.symbol) and len(gene_axis) == len(old)

    sym = list(gene_axis.symbol)
    assert len(set(sym)) == len(sym)
    sym_idx = {s: i for i, s in enumerate(sym)}
    gmt = a1core.read_gmt(cfg["paths"]["hallmark_gmt"])
    kept, rows = {}, []
    for name, genes in gmt.items():
        genes = list(dict.fromkeys(genes))
        on = [x for x in genes if x in sym_idx]
        inc = len(on) >= cfg["hallmark"]["min_genes"] and len(on) / len(genes) >= cfg["hallmark"]["min_fraction"]
        rows.append(dict(set=name, n_original=len(genes), n_on_axis=len(on), included=inc))
        if inc:
            kept[name] = on
    with open(os.path.join(out, "hallmark_on_axis.gmt"), "w") as fh:
        for k, v in kept.items():
            fh.write("\t".join([k, "na"] + v) + "\n")
    gene_axis.to_csv(os.path.join(out, "gene_axis.tsv"), sep="\t", index=False)
    names, sets = R.load_sets(out, gene_axis)
    G = len(sym)

    stats = {}
    old_cache = cfg["paths"]["a1_cache"]
    for study, X, ids in (("limore", X_l, ids_l), ("lccl", X_c, ids_c)):
        np.save(os.path.join(out, f"{study}__expr_log2.npy"), X)
        ranks = a1core.min_rank(X)
        rel = ranks / G
        np.save(os.path.join(out, f"{study}__rank.npy"), rel)
        with open(os.path.join(out, f"{study}__models.txt"), "w") as fh:
            fh.write("\n".join(ids) + "\n")
        sc = np.column_stack([a1core.singscore_up(ranks, idx) for idx in sets])
        pd.DataFrame(sc, index=ids, columns=names).rename_axis("model_id").to_csv(
            os.path.join(out, f"{study}__hallmark_real.tsv"), sep="\t", float_format="%.17g")
        if study == "limore":
            for seed in cfg["hallmark"]["permutation_seeds"]:
                ps = R.permuted_sets(sets, G, seed)
                assert [len(a) for a in ps] == [len(a) for a in sets]
                scp = np.column_stack([a1core.singscore_up(ranks, idx) for idx in ps])
                pd.DataFrame(scp, index=ids, columns=names).rename_axis("model_id").to_csv(
                    os.path.join(out, f"{study}__hallmark_perm{seed:03d}.tsv"), sep="\t", float_format="%.17g")
        # effect of the rule change, against the Track A cache (same model order is asserted)
        old_ids = R._read_ids(os.path.join(old_cache, f"{study}__models.txt"))
        assert old_ids == ids
        old_rank = np.load(os.path.join(old_cache, f"{study}__rank.npy"))
        old_hm = pd.read_csv(os.path.join(old_cache, f"{study}__hallmark_real.tsv"), sep="\t", index_col=0)
        pos_old = {e: i for i, e in enumerate(old.ensembl_id)}
        common = [(i, pos_old[e]) for i, e in enumerate(gene_axis.ensembl_id) if e in pos_old]
        ci, co = np.array(common).T
        stats[study] = dict(
            n_models=len(ids), genes_shared_with_track_a=int(len(common)),
            max_abs_relative_rank_change_on_shared_genes=float(np.abs(rel[:, ci] - old_rank[:, co]).max()),
            max_abs_hallmark_score_change=float(np.abs(sc - old_hm.loc[ids, names].values).max()),
            rank_sha256=R.sha256_array(rel), expr_sha256=R.sha256_array(X))
        if args.rscript:
            ef = os.path.join(out, f"{study}__expr_for_scoring.tsv")
            pd.DataFrame(X.T, index=sym, columns=ids).rename_axis("symbol").to_csv(ef, sep="\t", float_format="%.17g")
            dest = os.path.join(out, f"{study}__hallmark_real_Rsingscore.tsv")
            res = subprocess.run([args.rscript, os.path.join(R.A1_SRC, "score_pathways.R"), ef,
                                  os.path.join(out, "hallmark_on_axis.gmt"), dest], check=True,
                                 capture_output=True, text=True)
            rs = pd.read_csv(dest, sep="\t", index_col=0)
            dev = float(np.abs(rs.loc[ids, names].values - sc).max())
            assert dev < 1e-6, dev
            stats[study]["max_abs_dev_numpy_vs_R_singscore"] = dev
            stats[study]["singscore_version"] = res.stdout.strip().splitlines()[-1]
            os.remove(ef)

    dup_tab.to_csv(os.path.join(R.MANIFESTS, "gene_axis_duplicate_symbols.tsv"), sep="\t", index=False)
    gene_axis.to_csv(os.path.join(R.MANIFESTS, "gene_axis.tsv"), sep="\t", index=False)
    resp = os.path.join(old_cache, "limore__response_long.tsv")
    contract = dict(
        track=R.TRACK, cache_key=R.cache_key(cfg),
        prior_exposure="LIMORE and LCCL results were reported in Track A (b442fd8); post-freeze evaluation on a reused public benchmark",
        unit_of_independence="family_id (76 LIMORE families; 79 models x 90 drugs are not independent samples)",
        inputs=dict(
            limore_rna=dict(sha256=cfg["expected_sha256"]["limore_rna"], source="GEO GSE97098 (Track A download, read-only)"),
            lccl_rna=dict(sha256=cfg["expected_sha256"]["lccl_rna"], source="LCCL catalogue (Track A download, read-only)"),
            hallmark_gmt=dict(sha256=cfg["expected_sha256"]["hallmark_gmt"], release=cfg["hallmark"]["release"]),
            limore_response_long=dict(sha256=R.sha256_file(resp), source="Track A cache table built from LIMORE drug.txt; local copy, original-source verification not claimed"),
            lccl_response_long=dict(sha256=R.sha256_file(os.path.join(old_cache, "lccl__response_long.tsv"))),
            model_registry=dict(sha256=R.sha256_file(os.path.join(R.A1_MANIFESTS, "model_registry.tsv"))),
            splits_outer=dict(sha256=R.sha256_file(os.path.join(R.A1_MANIFESTS, "splits_outer.tsv"))),
            drug_match_lccl=dict(sha256=R.sha256_file(os.path.join(R.A1_MANIFESTS, "drug_match_lccl.tsv")))),
        gene_axis=dict(rule=cfg["gene_axis"], n_before_symbol_dedup=int(n_before), n_genes=int(G),
                       n_duplicated_symbols=int(dup_tab.symbol.nunique()),
                       n_symbols_where_kept_id_differs_from_track_a=int(changed.sum()),
                       gene_axis_sha256=R.sha256_file(os.path.join(R.MANIFESTS, "gene_axis.tsv"))),
        hallmark=dict(K=len(kept), genes_in_any_set=int(len({i for s in sets for i in s})),
                      permutation_seeds=cfg["hallmark"]["permutation_seeds"]),
        change_vs_track_a=stats,
        labels=dict(endpoint="LIMORE Activity Area (no clipping, no imputation)", n_models=79, n_families=76, n_drugs=90,
                    external="LCCL AUC (primary) / GI50; 18 matched single agents; never pooled with AA"))
    with open(os.path.join(R.MANIFESTS, "data_contract.json"), "w") as fh:
        json.dump(contract, fh, indent=1, sort_keys=True)
    print(json.dumps(contract, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
