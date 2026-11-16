#!/usr/bin/env python3
"""Stage A0/A1: link raw files into a registry, a fixed gene axis, cached inputs and split tables.

No prediction is made here and no response value is used to choose a feature, a gene or a model.
Writes small tables to manifests/pathway_sensitivity_a1/ and matrices to assets/cache/data_<key>/.

Usage: build_data.py [--rscript PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


def build_registry(cfg: dict) -> pd.DataFrame:
    with open(core.rpath(cfg["paths"]["model_aliases"])) as fh:
        ali = pd.read_csv(fh, sep="\t", comment="#")
    cello = pd.read_csv(core.rpath(cfg["paths"]["cellosaurus"]), sep="\t").fillna("")
    cello = cello.set_index("cellosaurus_id")
    clinic = pd.read_csv(core.rpath(cfg["paths"]["limore_clinic"]), sep="\t")
    limore_dx = dict(zip(clinic["Cell line"], clinic["Cancer Type"]))
    lccl_desc = core.read_cr_table(core.rpath(cfg["paths"]["lccl_description"])).dropna(axis=1, how="all")
    lccl_desc = lccl_desc.dropna(subset=["Cell line name"])
    lccl_dx = dict(zip(lccl_desc["Cell line name"], lccl_desc["Tumor type"]))
    lccl_cvcl = dict(zip(lccl_desc["Cell line name"], lccl_desc["Cellosaurus ID"]))
    dep = pd.read_csv(core.rpath(cfg["paths"]["depmap_model"]), usecols=["ModelID", "PatientID", "RRID",
                                                                        "OncotreeSubtype"])
    dep = dep.dropna(subset=["RRID"]).drop_duplicates("RRID").set_index("RRID")

    rows = []
    for study, name, ac in zip(ali.study, ali.source_name, ali.cellosaurus_id):
        assert ac in cello.index, f"no Cellosaurus record for {ac}"
        if study == "LIMORE":
            assert name in limore_dx, name
            src_dx = {"HCC": "Hepatocellular Carcinoma", "HB": "Hepatoblastoma"}[limore_dx[name]]
        else:
            assert name in lccl_dx, name
            src_dx = lccl_dx[name]
            listed = lccl_cvcl[name]
            if isinstance(listed, str) and listed.startswith("CVCL"):
                assert listed == ac, f"alias table disagrees with LCCL description for {name}"
        c = cello.loc[ac]
        dis = c["ncit_diseases"]
        reasons, flags = [], []
        if src_dx != "Hepatocellular Carcinoma":
            reasons.append(f"source diagnosis: {src_dx}")
        if "hepatoblastoma" in dis.lower():
            reasons.append("Cellosaurus: hepatoblastoma")
        elif "hepatocellular carcinoma" not in dis.lower():
            reasons.append(f"Cellosaurus disease not HCC: {dis}")
        note = c["problematic_note"]
        if note:
            if any(note.startswith(p) for p in cfg["eligibility"]["exclude_prefixes"]):
                reasons.append(f"Cellosaurus problematic: {note}")
            else:
                flags.append(f"Cellosaurus note (kept): {note}")
        rows.append(dict(model_id=core.model_id(study, name), study=study, source_name=name,
                         cellosaurus_id=ac, cellosaurus_name=c["name"], source_diagnosis=src_dx,
                         cellosaurus_diseases=dis, root_ancestor=c["root_ancestor"],
                         same_individual=c["same_individual"],
                         depmap_model_id=dep["ModelID"].get(ac, ""), depmap_patient_id=dep["PatientID"].get(ac, ""),
                         depmap_subtype=dep["OncotreeSubtype"].get(ac, ""),
                         included=len(reasons) == 0, exclusion_reason="; ".join(reasons), flags="; ".join(flags)))
    reg = pd.DataFrame(rows)
    assert reg.model_id.is_unique

    # families: union-find over documented relations only
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    evidence: dict = {m: [] for m in reg.model_id}
    for r in reg.itertuples():
        node = "AC:" + r.cellosaurus_id
        union("M:" + r.model_id, node)
        if r.root_ancestor and r.root_ancestor != r.cellosaurus_id:
            union(node, "AC:" + r.root_ancestor)
        for oi in filter(None, r.same_individual.split(",")):
            union(node, "AC:" + oi)
        if r.depmap_patient_id:
            union(node, "PT:" + r.depmap_patient_id)
    comp: dict = {}
    for r in reg.itertuples():
        comp.setdefault(find("M:" + r.model_id), []).append(r)
    fam_of = {}
    for members in comp.values():
        acs = sorted({m.cellosaurus_id for m in members})
        fid = "FAM_" + acs[0]
        for m in members:
            fam_of[m.model_id] = fid
            others = [o for o in members if o.model_id != m.model_id]
            ev = []
            for o in others:
                if o.cellosaurus_id == m.cellosaurus_id:
                    kind = "same Cellosaurus accession"
                elif o.cellosaurus_id in m.same_individual.split(","):
                    kind = "Cellosaurus same-individual link"
                elif o.root_ancestor == m.root_ancestor:
                    kind = "Cellosaurus common ancestor"
                elif m.depmap_patient_id and o.depmap_patient_id == m.depmap_patient_id:
                    kind = "DepMap PatientID"
                else:
                    kind = "linked through a third record"
                ev.append(f"{o.model_id} ({kind})")
            evidence[m.model_id] = ev
    reg.insert(1, "family_id", reg.model_id.map(fam_of))
    reg["family_evidence"] = reg.model_id.map(lambda m: "; ".join(evidence[m]))
    return reg


def build_expression(cfg: dict):
    g = pd.read_csv(core.rpath(cfg["paths"]["limore_rna"]), sep="\t")
    assert g.tracking_id.is_unique
    samples_l = list(g.columns[3:])
    # cross-check against the server-held protein-coding copy (same values where symbols are unique)
    srv = pd.read_csv(core.rpath(cfg["paths"]["limore_rna_server_copy"]), sep="\t").set_index("gene")
    assert list(srv.columns[2:]) == samples_l
    pc = g[g.geneType == "protein_coding"]
    uniq = pc[~pc.geneSymbol.duplicated(keep=False)].set_index("geneSymbol")
    common = uniq.index.intersection(srv.index)
    diff = np.abs(uniq.loc[common, samples_l].values - srv.loc[common, samples_l].values).max()
    crosscheck = dict(n_unique_symbol_protein_coding=int(len(uniq)), n_compared=int(len(common)),
                      max_abs_diff_vs_server_copy=float(diff))

    lccl = core.read_cr_table(core.rpath(cfg["paths"]["lccl_rna"]))
    lccl["ens"] = lccl["Gene_ID"].str.replace(r"\.\d+$", "", regex=True)
    n_dup_lccl = int(lccl.ens.duplicated().sum())
    assert n_dup_lccl == 0, "LCCL Ensembl IDs not unique after version stripping"
    samples_c = list(lccl.columns[2:-1])

    pc = pc.assign(ens=pc.tracking_id.str.replace(r"\.\d+$", "", regex=True))
    assert pc.ens.is_unique
    axis = pc[pc.ens.isin(set(lccl.ens))].copy()
    axis["limore_mean"] = axis[samples_l].mean(axis=1)
    n_before = len(axis)
    axis = axis.sort_values(["geneSymbol", "limore_mean", "ens"], ascending=[True, False, True])
    axis = axis.drop_duplicates("geneSymbol", keep="first").sort_values("ens").reset_index(drop=True)
    lc = lccl.set_index("ens").loc[axis.ens]
    X_l = axis[samples_l].values.T.astype(float)                     # models x genes, already log2
    raw_c = lc[samples_c].values.T.astype(float)
    assert raw_c.min() >= 0 and raw_c.max() > 100, "LCCL table does not look like raw FPKM"
    X_c = np.log2(raw_c + 1.0)
    # double-log guards
    assert 0 <= X_l.min() and X_l.max() < 25, "LIMORE matrix is not on a log2 scale"
    assert X_c.max() < 25
    info = dict(limore_genes_total=int(len(g)), limore_protein_coding=int(len(pc)), lccl_genes=int(len(lccl)),
                axis_before_symbol_dedup=int(n_before), axis_genes=int(len(axis)),
                limore_log2_min=float(X_l.min()), limore_log2_max=float(X_l.max()),
                lccl_fpkm_max=float(raw_c.max()), lccl_log2_max=float(X_c.max()),
                limore_server_copy_crosscheck=crosscheck)
    gene_axis = pd.DataFrame(dict(ensembl_id=axis.ens.values, symbol=axis.geneSymbol.values,
                                  lccl_symbol=lc["Gene_name"].values))
    return samples_l, X_l, samples_c, X_c, gene_axis, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rscript", default="Rscript")
    args = ap.parse_args()
    cfg = core.load_config()
    for k, want in cfg["expected_sha256"].items():
        got = core.sha256_file(core.rpath(cfg["paths"][k]))
        assert got == want, f"{k}: sha256 {got} != expected {want}"
    out = core.cache_dir(cfg)
    os.makedirs(out, exist_ok=True)
    os.makedirs(core.MANIFESTS, exist_ok=True)
    card: dict = {"cache_key": core.cache_key(cfg)}

    reg = build_registry(cfg)

    # ---------------- expression
    samples_l, X_l, samples_c, X_c, gene_axis, info = build_expression(cfg)
    card["rna"] = info
    ids_l = [core.model_id("LIMORE", s) for s in samples_l]
    ids_c = [core.model_id("LCCL", s) for s in samples_c]
    assert len(set(ids_l)) == len(ids_l) and len(set(ids_c)) == len(ids_c)
    assert set(ids_l) == set(reg.model_id[reg.study == "LIMORE"]), "LIMORE RNA columns != registry"
    assert set(ids_c) == set(reg.model_id[reg.study == "LCCL"]), "LCCL RNA columns != registry"

    # ---------------- responses
    d = pd.read_csv(core.rpath(cfg["paths"]["limore_response"]), sep="\t")
    assert not d.duplicated(["cell", "drug"]).any()
    d["model_id"] = [core.model_id("LIMORE", c) for c in d.cell]
    assert set(d.model_id) <= set(ids_l)
    # one source name per model id (no two spellings collapsing)
    assert d.groupby("model_id").cell.nunique().max() == 1
    lim = d.rename(columns={"drug": "drug_id", cfg["endpoint"]["primary"]: "y"})[
        ["model_id", "drug_id", "y", "IC50", "Emax"]].sort_values(["drug_id", "model_id"])
    rho_ic50 = spearmanr(lim.y, lim.IC50)[0]
    rho_emax = spearmanr(lim.y, lim.Emax)[0]
    assert rho_ic50 < 0 and rho_emax < 0, "AA direction check failed"
    per_drug_rho = lim.groupby("drug_id").apply(lambda t: spearmanr(t.y, t.Emax)[0])
    card["endpoint"] = dict(
        n_pairs=int(len(lim)), n_models=int(lim.model_id.nunique()), n_drugs=int(lim.drug_id.nunique()),
        missing_pairs=int(lim.model_id.nunique() * lim.drug_id.nunique() - len(lim)),
        aa_min=float(lim.y.min()), aa_max=float(lim.y.max()), aa_zero_fraction=float((lim.y == 0).mean()),
        spearman_aa_ic50=float(rho_ic50), spearman_aa_emax=float(rho_emax),
        drugs_with_positive_aa_emax_spearman=int((per_drug_rho > 0).sum()),
        replicate_rows=0)

    auc = core.read_cr_table(core.rpath(cfg["paths"]["lccl_auc"])).dropna(subset=["Cell line"])
    gi = core.read_cr_table(core.rpath(cfg["paths"]["lccl_gi50"])).dropna(subset=["Cell line"])
    assert list(auc.columns) == list(gi.columns) and list(auc["Cell line"]) == list(gi["Cell line"])
    drugs_c = list(auc.columns[2:])
    la = auc.melt(id_vars=["Cell line"], value_vars=drugs_c, var_name="lccl_name", value_name="AUC")
    lg = gi.melt(id_vars=["Cell line"], value_vars=drugs_c, var_name="lccl_name", value_name="GI50")
    lc = la.merge(lg, on=["Cell line", "lccl_name"], validate="one_to_one")
    lc["model_id"] = [core.model_id("LCCL", c) for c in lc["Cell line"]]
    assert set(lc.model_id) == set(ids_c), "LCCL response rows != RNA columns"
    lc = lc.drop(columns="Cell line").dropna(subset=["AUC", "GI50"], how="all")
    ok = lc.dropna(subset=["AUC", "GI50"])
    rho_ext = spearmanr(ok.AUC, ok.GI50)[0]
    assert rho_ext > 0, "LCCL AUC direction check failed"
    card["external_endpoint"] = dict(n_models=int(lc.model_id.nunique()), n_compounds=len(drugs_c),
                                     auc_min=float(lc.AUC.min()), auc_max=float(lc.AUC.max()),
                                     spearman_auc_gi50=float(rho_ext), gi50_max=float(lc.GI50.max()),
                                     gi50_at_ceiling_fraction=float((lc.GI50 >= lc.GI50.max()).mean()))

    # drug registry
    dm = pd.read_csv(core.rpath(cfg["paths"]["drug_match"]), sep="\t", comment="#").fillna("")
    assert set(dm.lccl_name) == set(drugs_c), set(dm.lccl_name) ^ set(drugs_c)
    matched = dm[dm.match_status == "matched"]
    assert set(matched.limore_name) <= set(lim.drug_id) and matched.limore_name.is_unique
    parent_of = dict(zip(matched.limore_name, matched.canonical_parent))
    lccl_of = dict(zip(matched.limore_name, matched.lccl_name))
    drug_reg = pd.DataFrame(dict(drug_id=sorted(lim.drug_id.unique())))
    drug_reg["source_name"] = drug_reg.drug_id
    drug_reg["study"] = "LIMORE"
    drug_reg["canonical_parent"] = drug_reg.drug_id.map(parent_of).fillna("")
    drug_reg["salt_form"] = "not_verified"
    drug_reg["lccl_name"] = drug_reg.drug_id.map(lccl_of).fillna("")
    drug_reg["match_status"] = np.where(drug_reg.lccl_name != "", "matched_to_LCCL", "LIMORE_only")

    reg["has_rna"] = reg.model_id.isin(set(ids_l) | set(ids_c))
    reg["has_response"] = reg.model_id.isin(set(lim.model_id) | set(lc.model_id))
    lim_fams = set(reg.family_id[(reg.study == "LIMORE")])
    reg["family_in_limore"] = reg.family_id.isin(lim_fams)

    # ---------------- hallmark sets on the gene axis
    sym = list(gene_axis.symbol)
    assert len(set(sym)) == len(sym)
    sym_idx = {s: i for i, s in enumerate(sym)}
    gmt = core.read_gmt(core.rpath(cfg["paths"]["hallmark_gmt"]))
    rows, kept = [], {}
    for name, genes in gmt.items():
        genes = list(dict.fromkeys(genes))
        on = [x for x in genes if x in sym_idx]
        inc = len(on) >= cfg["hallmark"]["min_genes"] and len(on) / len(genes) >= cfg["hallmark"]["min_fraction"]
        rows.append(dict(set=name, n_original=len(genes), n_on_axis=len(on),
                         fraction=round(len(on) / len(genes), 4), included=inc))
        if inc:
            kept[name] = on
    sets_tab = pd.DataFrame(rows)
    card["hallmark"] = dict(release=cfg["hallmark"]["release"], n_sets_in_file=len(gmt), K=len(kept),
                            excluded=[r["set"] for r in rows if not r["included"]],
                            min_fraction_retained=float(sets_tab.fraction.min()),
                            genes_in_any_set=int(len({x for v in kept.values() for x in v})))

    def write_gmt(path, sets):
        with open(path + ".tmp", "w") as fh:
            for k, v in sets.items():
                fh.write("\t".join([k, "na"] + v) + "\n")
        os.replace(path + ".tmp", path)

    def write_expr(path, X, ids):
        df = pd.DataFrame(X.T, index=sym, columns=ids)
        df.index.name = "symbol"
        df.to_csv(path + ".tmp", sep="\t", float_format="%.6g")
        os.replace(path + ".tmp", path)

    gmt_real = os.path.join(out, "hallmark_on_axis.gmt")
    write_gmt(gmt_real, kept)
    perm_gmts = {}
    for seed in cfg["a4"]["seeds"]:
        perm = np.random.RandomState(seed).permutation(len(sym))        # gene i takes the label of gene perm[i]
        psets = {k: [sym[perm[sym_idx[x]]] for x in v] for k, v in kept.items()}
        # structure is preserved: sizes, pairwise overlaps
        for k in kept:
            assert len(set(psets[k])) == len(kept[k])
        names = list(kept)
        for a in range(0, len(names), 7):
            for b in range(a + 1, len(names), 11):
                assert len(set(kept[names[a]]) & set(kept[names[b]])) == len(set(psets[names[a]]) & set(psets[names[b]]))
        p = os.path.join(out, f"hallmark_perm{seed:03d}.gmt")
        write_gmt(p, psets)
        perm_gmts[seed] = p

    rscript = os.path.join(core.HERE, "score_pathways.R")
    versions = {}
    for study, X, ids in (("limore", X_l, ids_l), ("lccl", X_c, ids_c)):
        np.save(os.path.join(out, f"{study}__expr_log2.npy"), X)
        R = core.relative_rank(X)
        np.save(os.path.join(out, f"{study}__rank.npy"), R)
        with open(os.path.join(out, f"{study}__models.txt"), "w") as fh:
            fh.write("\n".join(ids) + "\n")
        ef = os.path.join(out, f"{study}__expr_for_scoring.tsv")
        write_expr(ef, X, ids)
        jobs = [(gmt_real, os.path.join(out, f"{study}__hallmark_real.tsv"))]
        if study == "limore":
            jobs += [(p, os.path.join(out, f"{study}__hallmark_perm{s:03d}.tsv")) for s, p in perm_gmts.items()]
        for gm, dest in jobs:
            if not os.path.exists(dest):
                res = subprocess.run([args.rscript, rscript, ef, gm, dest + ".tmp"], check=True,
                                     capture_output=True, text=True)
                versions["singscore"] = res.stdout.strip().splitlines()[-1]
                os.replace(dest + ".tmp", dest)
        # numpy transcription must reproduce the published implementation on the real sets
        sc = pd.read_csv(os.path.join(out, f"{study}__hallmark_real.tsv"), sep="\t", index_col=0)
        assert list(sc.index) == ids and list(sc.columns) == list(kept)
        ranks = core.min_rank(X)
        mine = np.column_stack([core.singscore_up(ranks, np.array([sym_idx[x] for x in kept[k]])) for k in kept])
        dev = float(np.abs(mine - sc.values).max())
        assert dev < 1e-6, f"numpy singscore deviates from R singscore by {dev}"
        card.setdefault("singscore_check", {})[study] = dev
    card["versions"] = versions
    gene_axis.to_csv(os.path.join(out, "gene_axis.tsv"), sep="\t", index=False)
    lim.to_csv(os.path.join(out, "limore__response_long.tsv"), sep="\t", index=False)
    lc.to_csv(os.path.join(out, "lccl__response_long.tsv"), sep="\t", index=False)

    # ---------------- split table (metadata only)
    elig = reg[(reg.study == "LIMORE") & reg.included].sort_values("model_id")
    n_fam = elig.family_id.nunique()
    n_folds = cfg["split"]["outer_folds"]
    assert n_fam >= 30, f"only {n_fam} eligible families"
    srows = []
    for rep, seed in enumerate(cfg["split"]["outer_seeds"]):
        fold = core.group_folds(elig.family_id.values, n_folds, seed)
        for m, f, k in zip(elig.model_id, elig.family_id, fold):
            srows.append(dict(repeat=rep, model_id=m, family_id=f, outer_fold=int(k)))
    splits = pd.DataFrame(srows)
    for rep, t in splits.groupby("repeat"):
        assert t.groupby("family_id").outer_fold.nunique().max() == 1

    # ---------------- per-drug coverage and data card
    lim_e = lim[lim.model_id.isin(set(elig.model_id))].merge(elig[["model_id", "family_id"]], on="model_id")
    cov = lim_e.groupby("drug_id").agg(n_models=("model_id", "nunique"), n_families=("family_id", "nunique"),
                                       aa_mean=("y", "mean"), aa_sd=("y", "std"),
                                       aa_zero_fraction=("y", lambda v: float((v == 0).mean()))).reset_index()
    cov["evaluated_as"] = np.where(cov.n_families >= 20, "main", "exploratory_small_n")
    drug_reg = drug_reg.merge(cov, on="drug_id", how="left")

    lccl_reg = reg[reg.study == "LCCL"]
    card["counts"] = dict(
        limore_models=int((reg.study == "LIMORE").sum()),
        limore_included_models=int(len(elig)), limore_included_families=int(n_fam),
        limore_excluded=reg[(reg.study == "LIMORE") & ~reg.included][["model_id", "exclusion_reason"]].values.tolist(),
        limore_multi_model_families={f: sorted(t.model_id) for f, t in elig.groupby("family_id") if len(t) > 1},
        flagged_kept=reg[reg.flags != ""][["model_id", "flags"]].values.tolist(),
        drugs_total=int(len(cov)), drugs_main=int((cov.evaluated_as == "main").sum()),
        lccl_models=int(len(lccl_reg)), lccl_included_models=int(lccl_reg.included.sum()),
        lccl_excluded=lccl_reg[~lccl_reg.included][["model_id", "exclusion_reason"]].values.tolist(),
        lccl_included_shared_family_models=int((lccl_reg.included & lccl_reg.family_in_limore).sum()),
        lccl_included_shared_families=int(lccl_reg[lccl_reg.included & lccl_reg.family_in_limore].family_id.nunique()),
        lccl_included_new_models=sorted(lccl_reg[lccl_reg.included & ~lccl_reg.family_in_limore].model_id),
        lccl_included_new_families=int(lccl_reg[lccl_reg.included & ~lccl_reg.family_in_limore].family_id.nunique()),
        matched_drugs=int((drug_reg.match_status == "matched_to_LCCL").sum()))
    card["files"] = {k: core.sha256_file(core.rpath(cfg["paths"][k])) for k in
                     ["limore_rna", "limore_rna_server_copy", "limore_response", "limore_clinic", "lccl_rna",
                      "lccl_auc", "lccl_gi50", "lccl_description", "lccl_compounds", "hallmark_gmt", "depmap_model"]}

    reg.to_csv(os.path.join(core.MANIFESTS, "model_registry.tsv"), sep="\t", index=False)
    drug_reg.to_csv(os.path.join(core.MANIFESTS, "drug_registry.tsv"), sep="\t", index=False)
    dm.to_csv(os.path.join(core.MANIFESTS, "drug_match_lccl.tsv"), sep="\t", index=False)
    sets_tab.to_csv(os.path.join(core.MANIFESTS, "hallmark_sets.tsv"), sep="\t", index=False)
    gene_axis.to_csv(os.path.join(core.MANIFESTS, "gene_axis.tsv"), sep="\t", index=False)
    splits.to_csv(os.path.join(core.MANIFESTS, "splits_outer.tsv"), sep="\t", index=False)
    with open(os.path.join(core.MANIFESTS, "data_card.json"), "w") as fh:
        json.dump(card, fh, indent=1, sort_keys=True)
    with open(os.path.join(out, "key.json"), "w") as fh:
        json.dump(card, fh, indent=1, sort_keys=True)
    print(json.dumps(card, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
