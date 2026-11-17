"""Phase 0: drug -> target evidence (ChEMBL) and target -> gene relations (OmniPath) -> graph cache.

Metadata only: no expression value (X0 or delta) is read here. Every drug group of the R4 split is processed with the
same rule regardless of role (the test drugs' annotations are external inputs allowed at prediction time).

Identity: a drug group is linked to ChEMBL only through structure (InChIKey of each pert_id of the group):
  exact      full standard InChIKey match
  connect    first InChIKey block (connectivity) match only; stereo / protonation may differ -> flagged
Evidence is collected over the matched molecule's ChEMBL family (parent + every child of the parent: salts / forms).
Names are never used for linking.

Evidence kinds (kept separate, never merged into one number):
  mech   ChEMBL drug_mechanism (curated); action sign from the action-type parent: negative / positive / unknown
  actB   binding assay activity with pchembl >= 6 (<= 1 uM), '=' relation, no validity flag, confidence >= 7, human target
  actF   functional assay activity, same filters
Missing annotation stays 'unknown' (no node seed); it is never encoded as 'no effect' or as a zero RNA change.

Graph nodes are HGNC symbols. Measured nodes = the 978 landmarks (X0 observed); every other node carries a
missing-expression mask. Relation channels (OmniPath, all directed): signalling +/-/unsigned, transcriptional +/-/unsigned.
Node set: drug targets T, landmarks L, intermediates I = {v: t->v for some t in T and v->l for some l in L};
edges: every OmniPath edge with both ends in T u I u L. The hop limit is 2 (two message-passing blocks).
"""
import json, os, sqlite3, time
import numpy as np
import pandas as pd

from .common import ext, kg, tb, r4proc, pman, dump, sha256, R4_MAN

PCHEMBL_MIN = 6.0
CONF_MIN = 7
CHANNELS = ["sig_pos", "sig_neg", "sig_unk", "tf_pos", "tf_neg", "tf_unk"]
SEED_FEATS = ["mech", "actB", "actF", "sign_pos", "sign_neg", "action_unknown", "map_exact", "map_connect_only", "multi_gene_target"]
NEG_PARENTS = {"NEGATIVE MODULATOR", "INHIBITOR"}
POS_PARENTS = {"POSITIVE MODULATOR", "ACTIVATOR"}


def hgnc_maps():
    path = "/mnt/S1/sdata/pubdata/HGNC/hgnc_complete_set_2026-09-08.txt"
    h = pd.read_csv(path, sep="\t", dtype=str, usecols=["symbol", "status", "entrez_id", "uniprot_ids", "prev_symbol", "alias_symbol"])
    h = h[h.status == "Approved"]
    uni = {}
    for s, u in zip(h.symbol, h.uniprot_ids.fillna("")):
        for a in u.split("|"):
            if a:
                uni.setdefault(a, set()).add(s)
    ent = {e: s for s, e in zip(h.symbol, h.entrez_id) if isinstance(e, str)}
    return uni, ent, dict(path=path, sha256=sha256(path), approved_symbols=int(len(h)))


def landmarks(ent):
    g = pd.read_csv(tb("genes.tsv"), sep="\t", dtype={"gene_id": str})
    sym = [ent.get(e, s) for e, s in zip(g.gene_id, g.gene_symbol)]
    assert len(set(sym)) == 978, "landmark symbols not unique after HGNC mapping"
    return g.assign(hgnc_symbol=sym)


def chembl_db():
    d = [x for x in os.listdir(ext()) if x.startswith("chembl_")]
    assert len(d) == 1, d
    root = ext(d[0])
    dbs = []
    for dp, _, fs in os.walk(root):
        dbs += [os.path.join(dp, f) for f in fs if f.endswith(".db")]
    assert len(dbs) == 1, dbs
    return dbs[0], d[0]


def link_drugs(con, drugs):
    """pert_id -> ChEMBL molregno family. Returns a long table (drug_group, pert_id, molregno_matched, match, family_molregno)."""
    cs = pd.read_sql("select cs.molregno, cs.standard_inchi_key, md.chembl_id from compound_structures cs join molecule_dictionary md using(molregno)", con)
    cs["block1"] = cs.standard_inchi_key.str[:14]
    full = cs.groupby("standard_inchi_key").molregno.apply(list).to_dict()
    blk = cs.groupby("block1").molregno.apply(list).to_dict()
    hier = pd.read_sql("select molregno, parent_molregno from molecule_hierarchy", con)
    parent = dict(zip(hier.molregno, hier.parent_molregno))
    children = hier.groupby("parent_molregno").molregno.apply(set).to_dict()
    rows, amb = [], []
    for g, pid, ik in zip(drugs.drug_group, drugs.pert_id, drugs.inchi_keys.fillna("")):
        ik = ik.strip()
        if not ik:
            amb.append(dict(drug_group=g, pert_id=pid, issue="no_inchikey")); continue
        if ik in full:
            hits, how = full[ik], "exact"
        elif ik[:14] in blk:
            hits, how = blk[ik[:14]], "connect"
        else:
            continue
        fams = {parent.get(m, m) for m in hits}
        if len(hits) > 1 or len(fams) > 1:
            amb.append(dict(drug_group=g, pert_id=pid, issue=f"{how}_match_{len(hits)}_molecules_{len(fams)}_parents"))
        for m in hits:
            p = parent.get(m, m)
            for f in {p} | children.get(p, set()):
                rows.append(dict(drug_group=g, pert_id=pid, inchikey=ik, match=how, molregno=m, family_molregno=f))
    link = pd.DataFrame(rows)
    # a group whose pert_ids link to different parent families is flagged, never merged silently
    pg = link.assign(parent=link.molregno.map(lambda m: parent.get(m, m))).groupby("drug_group").parent.nunique()
    for g in pg[pg > 1].index:
        amb.append(dict(drug_group=g, pert_id="*", issue=f"group_links_{int(pg[g])}_parent_families"))
    return link, pd.DataFrame(amb)


def target_genes(con, uni):
    t = pd.read_sql("""select td.tid, td.chembl_id as target_chembl_id, td.target_type, td.organism, cs.accession
                       from target_dictionary td join target_components tc using(tid) join component_sequences cs using(component_id)""", con)
    t = t[t.organism == "Homo sapiens"]
    t["genes"] = t.accession.map(lambda a: sorted(uni.get(a, [])))
    unmapped = t[t.genes.map(len) == 0]
    t = t.explode("genes").dropna(subset=["genes"]).rename(columns={"genes": "gene"})
    ng = t.groupby("tid").gene.nunique()
    t["multi_gene_target"] = t.tid.map(ng).gt(1).astype(int)
    return t[["tid", "target_chembl_id", "target_type", "accession", "gene", "multi_gene_target"]].drop_duplicates(), unmapped


def evidence(con, link, tg):
    fam = sorted(set(link.family_molregno))
    con.execute("create temp table fam(molregno integer primary key)")
    con.executemany("insert into fam values (?)", [(int(m),) for m in fam])
    at = pd.read_sql("select action_type, parent_type from action_type", con)
    sign = {a: (-1 if p in NEG_PARENTS or a in NEG_PARENTS else 1 if p in POS_PARENTS or a in POS_PARENTS else 0)
            for a, p in zip(at.action_type, at.parent_type.fillna(""))}
    mech = pd.read_sql("""select dm.molregno, dm.tid, dm.action_type, dm.direct_interaction from drug_mechanism dm
                          join fam using(molregno) where dm.tid is not null""", con)
    mech["kind"] = "mech"; mech["sign"] = mech.action_type.map(sign).fillna(0).astype(int)
    act = pd.read_sql(f"""select a.molregno, s.tid, s.assay_type, max(a.pchembl_value) as max_pchembl, count(*) as n_records
                          from activities a join fam using(molregno) join assays s using(assay_id)
                          where a.pchembl_value >= {PCHEMBL_MIN} and a.standard_relation = '=' and a.data_validity_comment is null
                            and s.confidence_score >= {CONF_MIN} and s.assay_type in ('B', 'F')
                          group by a.molregno, s.tid, s.assay_type""", con)
    act["kind"] = "act" + act.assay_type; act["sign"] = 0; act["action_type"] = ""
    ev = pd.concat([mech[["molregno", "tid", "kind", "action_type", "sign"]],
                    act[["molregno", "tid", "kind", "action_type", "sign", "max_pchembl", "n_records"]]], ignore_index=True)
    ev = ev.merge(tg, on="tid")          # human targets with an HGNC gene only
    l = link.rename(columns={"family_molregno": "molregno", "molregno": "matched_molregno"})
    out = l.merge(ev, on="molregno")
    sign_table = pd.DataFrame(dict(action_type=list(sign), sign=list(sign.values()))).merge(at, on="action_type")
    return out, sign_table


def omnipath_graph(uni, T, L):
    p = ext("omnipath", "interactions_omnipath_collectri.tsv")
    d = pd.read_csv(p, sep="\t", dtype=str)
    tobool = lambda s: s.str.lower().eq("true")

    def members(x):
        if x.startswith("COMPLEX:"):
            return x[8:].split("_"), 1
        return [x], 0

    rows, dropped = [], 0
    for r in d.itertuples(index=False):
        sm, sc = members(r.source); tm, tcx = members(r.target)
        st, ih = r.is_stimulation.lower() == "true", r.is_inhibition.lower() == "true"
        tf = r.type == "transcriptional"
        ch = ("tf_" if tf else "sig_") + ("pos" if st and not ih else "neg" if ih and not st else "unk")
        nsrc = len(str(r.sources).split(";")); nref = len([x for x in str(r.references).split(";") if x and x != "nan"])
        for a in sm:
            for b in tm:
                sa, sb = uni.get(a), uni.get(b)
                if not sa or not sb or len(sa) > 1 or len(sb) > 1:
                    dropped += 1; continue
                rows.append((next(iter(sa)), next(iter(sb)), ch, nsrc, nref, float(r.curation_effort or 0), int(sc or tcx)))
    e = pd.DataFrame(rows, columns=["u", "v", "channel", "n_sources", "n_refs", "curation_effort", "from_complex"])
    e = e[e.u != e.v]
    e = e.groupby(["u", "v", "channel"], as_index=False).agg(n_sources=("n_sources", "max"), n_refs=("n_refs", "max"),
                                                             curation_effort=("curation_effort", "max"), from_complex=("from_complex", "min"))
    out_n = e.groupby("u").v.apply(set).to_dict()
    Ls, Ts = set(L), set(T)
    I = {v for t in Ts for v in out_n.get(t, ()) if out_n.get(v, set()) & Ls} - Ts - Ls
    V = Ts | I | Ls
    e = e[e.u.isin(V) & e.v.isin(V)].reset_index(drop=True)
    return e, I, dict(path=p, sha256=sha256(p), rows=int(len(d)), complex_rows=int(d.source.str.startswith("COMPLEX").sum() + d.target.str.startswith("COMPLEX").sum()),
                      endpoint_pairs_dropped_no_unique_hgnc=int(dropped))


def main():
    t0 = time.time()
    uni, ent, hg = hgnc_maps()
    lm = landmarks(ent)
    L = list(lm.hgnc_symbol)
    drugs = pd.read_csv(tb("drugs.tsv"), sep="\t")
    # pert_ids without an InChIKey in the drug table: InChIKey computed from the table's RDKit SMILES (structure, flagged)
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    nokey = drugs.inchi_keys.isna() | (drugs.inchi_keys.astype(str).str.strip() == "")
    comp = {}
    for i in np.where(nokey)[0]:
        m = Chem.MolFromSmiles(str(drugs.rdkit_smiles.iloc[i]))
        comp[i] = Chem.MolToInchiKey(m) if m is not None else ""
    drugs.loc[list(comp), "inchi_keys"] = pd.Series(comp)
    drugs["inchikey_source"] = np.where(nokey, "computed_from_smiles", "drug_table")
    split = pd.read_csv(os.path.join(R4_MAN, "drug_split.tsv"), sep="\t").set_index("drug_group").role
    groups = pd.read_csv(r4proc("groups.tsv"), sep="\t")            # R4 group index order (metadata)
    dbp, rel = chembl_db()
    con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    link, amb = link_drugs(con, drugs)
    link["inchikey_source"] = link.pert_id.map(drugs.set_index("pert_id").inchikey_source)
    tg, unmapped_acc = target_genes(con, uni)
    ev, sign_table = evidence(con, link, tg)
    con.close()

    # drug-group x gene seed features
    ev["sign_pos"] = ((ev.kind == "mech") & (ev.sign > 0)).astype(int)
    ev["sign_neg"] = ((ev.kind == "mech") & (ev.sign < 0)).astype(int)
    ev["map_exact"] = (ev["match"] == "exact").astype(int)
    for k in ["mech", "actB", "actF"]:
        ev[k] = (ev.kind == k).astype(int)
    seed = ev.groupby(["drug_group", "gene"], as_index=False).agg(mech=("mech", "max"), actB=("actB", "max"), actF=("actF", "max"),
                                                                 sign_pos=("sign_pos", "max"), sign_neg=("sign_neg", "max"),
                                                                 map_exact=("map_exact", "max"), multi_gene_target=("multi_gene_target", "min"),
                                                                 tids=("tid", lambda s: "|".join(map(str, sorted(set(s))))))
    seed["action_unknown"] = 1 - ((seed.sign_pos + seed.sign_neg) > 0).astype(int)
    seed["map_connect_only"] = 1 - seed.map_exact
    T = sorted(set(seed.gene))
    edges, I, om = omnipath_graph(uni, T, L)
    nodes = sorted(set(T) | I | set(L))
    nid = {n: i for i, n in enumerate(nodes)}
    lm_idx = np.array([nid[s] for s in L])                             # readout order = landmark (gene axis) order
    seed["node"] = seed.gene.map(nid)
    gidx = groups.set_index("drug_group").group_idx
    seed = seed[seed.drug_group.isin(gidx.index)].copy()
    seed["group_idx"] = seed.drug_group.map(gidx)
    # per-group status
    ev_g = ev.groupby("drug_group")
    st = pd.DataFrame(dict(drug_group=groups.drug_group, group_idx=groups.group_idx))
    st["role"] = st.drug_group.map(split)
    st["chembl_linked"] = st.drug_group.isin(set(link.drug_group)).astype(int)
    st["link_match"] = st.drug_group.map(link.groupby("drug_group")["match"].agg(lambda s: "exact" if (s == "exact").any() else "connect")).fillna("none")
    for k in ["mech", "actB", "actF"]:
        st[f"n_{k}_genes"] = st.drug_group.map(seed[seed[k] == 1].groupby("drug_group").gene.nunique()).fillna(0).astype(int)
    st["n_target_genes"] = st.drug_group.map(seed.groupby("drug_group").gene.nunique()).fillna(0).astype(int)
    st["n_signed_mech_genes"] = st.drug_group.map(seed[(seed.sign_pos + seed.sign_neg) > 0].groupby("drug_group").gene.nunique()).fillna(0).astype(int)
    st["n_targets_in_landmarks"] = st.drug_group.map(seed[seed.gene.isin(set(L))].groupby("drug_group").gene.nunique()).fillna(0).astype(int)
    st["status"] = np.where(st.n_signed_mech_genes > 0, "complete", np.where(st.n_target_genes > 0, "partial", "missing"))
    st["m_available"] = (st.n_target_genes > 0).astype(int)
    st["missing_reason"] = np.where(st.chembl_linked == 0, "not_linked_to_chembl", np.where(st.n_target_genes == 0, "linked_no_human_evidence", ""))

    os.makedirs(kg(), exist_ok=True)
    ch_id = {c: i for i, c in enumerate(CHANNELS)}
    E = edges.assign(src=edges.u.map(nid), dst=edges.v.map(nid), rel=edges.channel.map(ch_id))
    efeat = np.zeros((len(E), len(CHANNELS) + 5), np.float32)
    efeat[np.arange(len(E)), E.rel.to_numpy()] = 1
    efeat[:, 6] = np.minimum(np.log1p(E.n_sources) / np.log(21), 1)
    efeat[:, 7] = np.minimum(np.log1p(E.n_refs) / np.log(101), 1)
    efeat[:, 8] = np.minimum(np.log1p(E.curation_effort) / np.log(101), 1)
    efeat[:, 9] = E.from_complex
    efeat[:, 10] = (~E.channel.str.endswith("unk")).astype(np.float32)        # sign known
    np.save(kg("edge_index.npy"), E[["src", "dst"]].to_numpy().T.astype(np.int64))
    np.save(kg("edge_rel.npy"), E.rel.to_numpy().astype(np.int64))
    np.save(kg("edge_feat.npy"), efeat)
    measured = np.zeros(len(nodes), np.int8); measured[lm_idx] = 1
    np.save(kg("node_measured.npy"), measured)
    np.save(kg("landmark_node_idx.npy"), lm_idx)
    pd.DataFrame(dict(node=nodes, node_idx=range(len(nodes)), is_target_any=[int(n in set(T)) for n in nodes],
                      is_landmark=measured, is_intermediate=[int(n in I) for n in nodes])).to_csv(kg("nodes.tsv"), sep="\t", index=False)
    seed_cols = ["group_idx", "drug_group", "gene", "node", "tids"] + SEED_FEATS
    seed[seed_cols].to_csv(kg("drug_seeds.tsv"), sep="\t", index=False)
    st.to_csv(kg("group_status.tsv"), sep="\t", index=False)
    E.to_csv(kg("edges.tsv"), sep="\t", index=False)

    # publishable metadata tables (small)
    st.to_csv(pman("drug_target_coverage.tsv"), sep="\t", index=False)
    prov = ev[["drug_group", "pert_id", "inchikey", "match", "matched_molregno", "molregno", "tid", "target_chembl_id", "target_type",
               "accession", "gene", "kind", "action_type", "sign", "max_pchembl", "n_records"]].drop_duplicates()
    prov.to_csv(kg("edge_provenance_full.tsv"), sep="\t", index=False)
    prov.groupby(["drug_group", "gene", "kind"], as_index=False).agg(target_chembl_ids=("target_chembl_id", lambda s: "|".join(sorted(set(s)))),
                                                                    action_types=("action_type", lambda s: "|".join(sorted(set(x for x in s if x)))),
                                                                    match=("match", lambda s: "|".join(sorted(set(s)))),
                                                                    max_pchembl=("max_pchembl", "max")).to_csv(pman("edge_provenance.tsv"), sep="\t", index=False)
    amb.to_csv(pman("mapping_ambiguities.tsv"), sep="\t", index=False)
    sign_table.to_csv(pman("action_type_sign_map.tsv"), sep="\t", index=False)
    cov = st.groupby(["role", "status"]).size().unstack(fill_value=0)
    cov["total"] = cov.sum(1)
    cov.reset_index().to_csv(pman("annotation_coverage.tsv"), sep="\t", index=False)
    files = ["edge_index.npy", "edge_rel.npy", "edge_feat.npy", "node_measured.npy", "landmark_node_idx.npy", "nodes.tsv", "drug_seeds.tsv", "group_status.tsv", "edges.tsv"]
    card = dict(chembl=dict(release=rel, db=dbp, db_bytes=os.path.getsize(dbp), pchembl_min=PCHEMBL_MIN, confidence_min=CONF_MIN,
                            linked_pert_ids=int(link.pert_id.nunique()), inchikey_computed_from_smiles=int(nokey.sum()),
                            linked_pert_ids_with_computed_key=int(link[link.inchikey_source == "computed_from_smiles"].pert_id.nunique()), linked_groups=int(link.drug_group.nunique()),
                            evidence_rows=int(len(ev)), human_target_accessions_without_hgnc=int(unmapped_acc.accession.nunique())),
                omnipath=om, hgnc=hg,
                graph=dict(nodes=len(nodes), targets=len(T), intermediates=len(I), landmarks=len(L), edges=int(len(E)),
                           edges_by_channel=E.channel.value_counts().to_dict(),
                           targets_in_landmarks=int(len(set(T) & set(L))), targets_in_graph_with_outgoing_edge=int(E[E.u.isin(set(T))].u.nunique())),
                seeds=dict(rows=int(len(seed)), feats=SEED_FEATS), channels=CHANNELS,
                coverage_by_role=cov.to_dict(orient="index"),
                files={f: dict(sha256=sha256(kg(f)), bytes=os.path.getsize(kg(f))) for f in files},
                seconds=round(time.time() - t0), commit=os.environ.get("R7_COMMIT"))
    dump(card, pman("kg_card.json"))
    print(json.dumps({k: v for k, v in card.items() if k != "files"}, indent=1, default=str))


if __name__ == "__main__":
    main()
