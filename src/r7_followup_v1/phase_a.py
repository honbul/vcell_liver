"""Phase A / D1 (label-free): source folds, reachability along the model's computation graph, K0-K3 groups,
evidence_off scope, realized annotation masking, rewiring audit, and the fixed target-shuffle mappings.
Reads unit METADATA (units.tsv) and the knowledge cache only; no delta array is opened."""
import json, os, time
import numpy as np
import pandas as pd

from .common import pman, cache, dump, sha256, tag_cols, NULL_SEEDS, N_FOLDS
from . import kgraph as kgm
from . import shuffle as sh

MASK_SEED = 20261117      # the seed of the original stress evaluation (final_eval.py)


def reach_table(KG, R, klab, units, folds):
    st = KG["status"]
    seeds = KG["seeds"]
    role = st.role.to_numpy()
    tr_groups = set(st.group_idx[st.role == "train"])
    # how many TRAIN drug groups carry each node as a target
    tr_count = seeds[seeds.group_idx.isin(tr_groups)].groupby("node").group_idx.nunique()
    rc = kgm.reach_class(R)
    rows = []
    by = {g: d for g, d in seeds.groupby("group_idx")}
    for gi in range(len(st)):
        row = dict(drug_group=st.drug_group[gi], group_idx=gi, split=role[gi], annotation_present=int(gi in by), k_group=klab[gi])
        if gi in by:
            d = by[gi]; nodes = d.node.to_numpy()
            others = tr_count.reindex(nodes).fillna(0).to_numpy() - (1 if gi in tr_groups else 0)
            reach_union = R["reach_mat"][nodes].sum(0) > 0
            reach1_union = R["reach1_mat"][nodes].sum(0) > 0
            mh = R["minhop"][nodes]
            row.update(n_targets=len(nodes), n_signed_targets=int(((d.sign_pos + d.sign_neg) > 0).sum()), n_measured_targets=int(R["is_lm"][nodes].sum()),
                       n_reachable_landmarks_0hop=int(R["is_lm"][nodes].sum()), n_reachable_landmarks_1hop=int(reach1_union.sum()),
                       n_reachable_landmarks_2hop=int(reach_union.sum()),
                       fraction_targets_reachable_2hop=float((rc[nodes] < 2).mean()),
                       min_path_length=int(mh[mh >= 0].min()) if (mh >= 0).any() else -1,
                       n_targets_unreachable=int((rc[nodes] == 2).sum()),
                       mapping_ambiguity=float(((d.map_connect_only == 1) | (d.multi_gene_target == 1)).mean()),
                       frac_map_connect_only=float(d.map_connect_only.mean()), frac_multi_gene_target=float(d.multi_gene_target.mean()),
                       target_seen_in_train=float((others > 0).mean()),
                       degree_summary="out_med=%g;out_max=%d;in_med=%g;no_out=%d" % (np.median(R["outdeg"][nodes]), R["outdeg"][nodes].max(), np.median(R["indeg"][nodes]), int((R["outdeg"][nodes] == 0).sum())),
                       n_tid_blocks=len({t for x in d.tids for t in str(x).split("|")}))
        else:
            row.update(n_targets=0, n_signed_targets=0, n_measured_targets=0, n_reachable_landmarks_0hop=0, n_reachable_landmarks_1hop=0, n_reachable_landmarks_2hop=0,
                       fraction_targets_reachable_2hop=np.nan, min_path_length=-1, n_targets_unreachable=0, mapping_ambiguity=np.nan,
                       frac_map_connect_only=np.nan, frac_multi_gene_target=np.nan, target_seen_in_train=np.nan, degree_summary="", n_tid_blocks=0)
        rows.append(row)
    t = pd.DataFrame(rows)
    f = folds.set_index("group_idx")
    t["source_fold"] = f.fold.reindex(t.group_idx).fillna(-1).astype(int).to_numpy()
    t["source_sub"] = f["sub"].reindex(t.group_idx).fillna(-1).astype(int).to_numpy()
    nb = t.n_tid_blocks.to_numpy()
    t["realized_mask_fraction"] = np.where(nb > 1, np.round(0.5 * nb) / np.maximum(nb, 1), 0.0)   # blocks removed by the original 50 % rule
    t.loc[t.annotation_present == 0, "realized_mask_fraction"] = np.nan
    return t


def masking_replay(units):
    """Replay of the original stress stream (predict(mask_frac=0.5, mask_seed=20261117) over the unit order of the split)."""
    from src.evidence_gated_r7.data import Knowledge
    from src.evidence_gated_r7.train import SeedBank
    K = Knowledge(int(units.group_idx.max()) + 1)
    bank = SeedBank(K, "on")
    out = []
    for role in ["validation", "test"]:
        u = units[units.role == role]
        rng = np.random.RandomState(MASK_SEED)
        for gi in u.group_idx.to_numpy():
            if gi in bank.rows:
                n0 = len(bank.rows[gi][0])
                nodes, f = bank.mask_fraction(gi, 0.5, rng)
                out.append((role, gi, n0, len(nodes)))
    r = pd.DataFrame(out, columns=["split", "group_idx", "rows", "rows_kept"])
    r["row_frac_removed"] = 1 - r.rows_kept / r.rows
    r["emptied"] = (r.rows_kept == 0).astype(int)
    g = r.groupby(["split", "group_idx"]).agg(replay_units=("rows", "size"), replay_mean_row_fraction_removed=("row_frac_removed", "mean"),
                                              replay_frac_units_emptied=("emptied", "mean")).reset_index()
    return g, r


def rewire_audit(KG, klab):
    ei, new, rel, n = KG["edge_index"], KG["rewired"], KG["edge_rel"], KG["n"]
    rows = []
    for r in list(np.unique(rel)) + ["all"]:
        k = np.arange(len(rel)) if r == "all" else np.where(rel == r)[0]
        old = set(zip(ei[0, k].tolist(), ei[1, k].tolist(), rel[k].tolist())); nw = list(zip(new[0, k].tolist(), new[1, k].tolist(), rel[k].tolist()))
        untyped_old = set(zip(ei[0].tolist(), ei[1].tolist()))
        rows.append(dict(relation=r, n_edges=len(k), self_loops=int((new[0, k] == new[1, k]).sum()),
                         duplicate_typed_edges=len(nw) - len(set(nw)),
                         typed_edges_kept_rate=float(np.mean([e in old for e in nw])),
                         untyped_pair_exists_in_real_rate=float(np.mean([(a, b) in untyped_old for a, b, _ in nw])),
                         out_degree_preserved=bool((np.bincount(ei[0, k], minlength=n) == np.bincount(new[0, k], minlength=n)).all()),
                         in_degree_preserved=bool((np.bincount(ei[1, k], minlength=n) == np.bincount(new[1, k], minlength=n)).all()),
                         source_endpoint_unchanged=bool((ei[0, k] == new[0, k]).all())))
    t = pd.DataFrame(rows)
    krew, Rrew = kgm.k_groups(KG, new)
    tab = pd.crosstab(pd.Series(klab, name="k_real"), pd.Series(krew, name="k_rewired"))
    return t, tab, Rrew


def evidence_audit(KG):
    """What `--evidence off` of the original attribution run changes (read from SeedBank / load_graph, checked on the arrays)."""
    s = KG["seeds"]
    rows = []
    for i, f in enumerate(kgm.SEED_FEATS):
        rows.append(dict(feature=f, where="seed feature of the drug-target stub (input of GraphCorrection.ws)", index=i,
                         evidence_on_mean=float(s[f].mean()), evidence_off_value="1.0" if i == 0 else "0.0",
                         changed_by_evidence_off="yes",
                         note="channel 0 is reused as a presence indicator" if i == 0 else "set to 0 for every stub"))
    for i, (nm, val) in enumerate([("has_mechanism", "0"), ("has_activity", "0"), ("log_target_count", "kept"), ("signed_fraction", "0")]):
        rows.append(dict(feature=f"status[{i}] {nm}", where="drug-level status vector (input of Cond -> q)", index=i, evidence_on_mean=np.nan,
                         evidence_off_value=val, changed_by_evidence_off="no" if val == "kept" else "yes", note="target count survives evidence_off"))
    names = ["rel_onehot_%d" % i for i in range(6)] + ["n_sources_log", "n_refs_log", "curation_effort_log", "from_complex", "sign_known"]
    ef = KG["efeat"]
    for i, nm in enumerate(names):
        rows.append(dict(feature=f"edge_feat[{i}] {nm}", where="graph edge provenance (input of the edge gate ke)", index=i, evidence_on_mean=float(ef[:, i].mean()),
                         evidence_off_value="unchanged", changed_by_evidence_off="no", note="evidence_off never touches the graph tensors"))
    rows.append(dict(feature="edge_rel (6 relation channels)", where="relation-typed weights w_rel / b_rel", index=-1, evidence_on_mean=np.nan,
                     evidence_off_value="unchanged", changed_by_evidence_off="no", note="direction / sign channels kept"))
    rows.append(dict(feature="target identity (seeded node index)", where="which node receives the seed", index=-1, evidence_on_mean=np.nan,
                     evidence_off_value="unchanged", changed_by_evidence_off="no", note="evidence_off keeps the drug-target identity"))
    rows.append(dict(feature="m_available", where="hard availability gate on the correction", index=-1, evidence_on_mean=float(KG["status"].m_available.mean()),
                     evidence_off_value="unchanged", changed_by_evidence_off="no", note=""))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    KG = kgm.load_kg(os.environ["R7_KG"])
    units = pd.read_csv(os.path.join(os.environ["R4_PROC"], "units.tsv"), sep="\t")
    st = KG["status"]
    assert (st.group_idx.to_numpy() == np.arange(len(st))).all()
    klab, R = kgm.k_groups(KG)
    folds = kgm.assign_folds(units, st, klab)
    folds.to_csv(pman("source_folds.tsv"), sep="\t", index=False)
    summary = dict(kg_dir=os.environ["R7_KG"], n_nodes=int(KG["n"]), n_edges=int(KG["edge_index"].shape[1]))

    # ---- reachability / K groups
    t = reach_table(KG, R, klab, units, folds)
    rep, rep_units = masking_replay(units)
    t = t.merge(rep.drop(columns="split"), on="group_idx", how="left")
    tag_cols(t, "label_free_metadata").to_csv(pman("target_reachability.tsv"), sep="\t", index=False)
    u = units.merge(pd.DataFrame(dict(group_idx=np.arange(len(st)), k_group=klab)), on="group_idx")
    u = u.merge(folds[["group_idx", "fold"]], on="group_idx", how="left")
    rows = []
    for name, sel in [("train", u.role == "train"), ("validation", u.role == "validation"), ("test", u.role == "test")] + \
                     [(f"source_fold{k}_held", (u.role == "train") & (u.fold == k)) for k in range(N_FOLDS)]:
        for kg_ in ["K0", "K1", "K2", "K3"]:
            s = sel & (u.k_group == kg_)
            rows.append(dict(split=name, k_group=kg_, n_drug_groups=int(u[s].group_idx.nunique()), n_units=int(s.sum()),
                             n_source_units=int((s & (u.is_hcc == 0)).sum()), n_hcc_units=int((s & (u.is_hcc == 1)).sum())))
    kc = pd.DataFrame(rows)
    kc["frac_of_annotated_groups"] = kc.n_drug_groups / kc.split.map(kc[kc.k_group != "K0"].groupby("split").n_drug_groups.sum())
    kc.loc[kc.k_group == "K0", "frac_of_annotated_groups"] = np.nan
    tag_cols(kc, "label_free_metadata").to_csv(pman("k_group_counts.tsv"), sep="\t", index=False)
    rc = kgm.reach_class(R)
    tnodes = np.unique(KG["seeds"].node)
    summary["nodes"] = dict(targets=int(len(tnodes)), targets_landmark=int((rc[tnodes] == 0).sum()), targets_offpanel_reachable_2hop=int((rc[tnodes] == 1).sum()),
                            targets_offpanel_unreachable=int((rc[tnodes] == 2).sum()),
                            nodes_reaching_a_landmark_within_2hop=int((R["n2"] > 0).sum()),
                            nodes_whose_identity_embedding_can_affect_the_readout=int((R["n2"] > 0).sum()))
    # ---- fold audit (metadata)
    fr = []
    for k in range(N_FOLDS):
        role = kgm.inner_role(folds.fold, folds["sub"], k)
        ug = units[units.role == "train"].merge(pd.DataFrame(dict(group_idx=folds.group_idx, r=role)), on="group_idx")
        for r_ in ["inner_train", "inner_val", "held"]:
            x = ug[ug.r == r_]
            fr.append(dict(fold=k, role=r_, n_drug_groups=int(x.group_idx.nunique()), n_units=int(len(x)), n_source_units=int((x.is_hcc == 0).sum()),
                           n_cells=int(x.cell.nunique()), n_families=int(x.family.nunique()),
                           frac_annotated_groups=float((folds.k_group[role == r_] != "K0").mean())))
    tag_cols(pd.DataFrame(fr), "label_free_metadata").to_csv(pman("source_fold_audit.tsv"), sep="\t", index=False)

    # ---- realized masking
    m = t[t.annotation_present == 1][["drug_group", "group_idx", "split", "k_group", "n_targets", "n_tid_blocks", "realized_mask_fraction",
                                      "replay_units", "replay_mean_row_fraction_removed", "replay_frac_units_emptied"]].copy()
    m["blocks_removed_by_rule"] = np.where(m.n_tid_blocks > 1, np.round(0.5 * m.n_tid_blocks), 0).astype(int)
    m["followup_rule_expected_fraction"] = np.where(m.n_tid_blocks > 1, m.realized_mask_fraction, 0.5)   # single block dropped whole with p = 0.5 (definition only)
    tag_cols(m, "label_free_metadata", metric_definition="original rule: int(round(0.5*n_blocks)) tid blocks removed if n_blocks>1 else none; replay = original stress stream seed 20261117").to_csv(
        pman("realized_masking.tsv"), sep="\t", index=False)
    ms = []
    for sp_, d in m.groupby("split"):
        one = d.n_tid_blocks == 1
        ms.append(dict(split=sp_, n_annotated_groups=len(d), groups_single_block=int(one.sum()), frac_groups_single_block_no_masking=float(one.mean()),
                       mean_block_fraction_removed=float(d.realized_mask_fraction.mean()),
                       mean_block_fraction_removed_multiblock=float(d.realized_mask_fraction[~one].mean()),
                       replay_mean_row_fraction_removed=float(d.replay_mean_row_fraction_removed.mean()) if sp_ != "train" else np.nan,
                       replay_unit_weighted_row_fraction_removed=float(rep_units[rep_units.split == sp_].row_frac_removed.mean()) if sp_ != "train" else np.nan,
                       replay_frac_units_emptied=float(rep_units[rep_units.split == sp_].emptied.mean()) if sp_ != "train" else np.nan))
    pd.DataFrame(ms).to_csv(pman("realized_masking_summary.tsv"), sep="\t", index=False)

    # ---- evidence_off scope, rewiring
    tag_cols(evidence_audit(KG), "implementation_audit").to_csv(pman("evidence_feature_audit.tsv"), sep="\t", index=False)
    ra, ktab, Rrew = rewire_audit(KG, klab)
    tag_cols(ra, "implementation_audit").to_csv(pman("rewiring_audit.tsv"), sep="\t", index=False)
    ktab.to_csv(pman("rewiring_k_group_crosstab.tsv"), sep="\t")
    summary["rewired"] = dict(targets_reaching_landmark_real=int((R["n2"][tnodes] > 0).sum()), targets_reaching_landmark_rewired=int((Rrew["n2"][tnodes] > 0).sum()),
                              mean_reachable_landmarks_per_target_real=float(R["n2"][tnodes].mean()), mean_reachable_landmarks_per_target_rewired=float(Rrew["n2"][tnodes].mean()))

    # ---- target-shuffle mappings: contexts fold0..2 (inner_train / inner_val / held) and orig (train / validation / test)
    seeds = KG["seeds"]
    contexts = {}
    for k in range(N_FOLDS):
        role = kgm.inner_role(folds.fold, folds["sub"], k)
        contexts[f"fold{k}"] = (dict(zip(folds.group_idx, role)), "inner_train")
    contexts["orig"] = (dict(zip(st.group_idx, st.role)), "train")
    chosen, bal_all, maps = None, [], {}
    for level in sorted(sh.LEVELS):
        ok, bal_level, maps_level = True, [], {}
        for ctx, (sog, trs) in contexts.items():
            for ns in NULL_SEEDS:
                mp = sh.make_mapping(seeds, sog, trs, R, level, ns)
                b = sh.balance(mp, R, klab)
                b.insert(0, "context", ctx); b.insert(1, "null_seed", ns); b.insert(2, "level", level); b["valid"] = sh.valid(b)
                ok &= sh.valid(b); bal_level.append(b); maps_level[(ctx, ns)] = mp
        bal_all += bal_level
        if ok and chosen is None:
            chosen, maps = level, maps_level
            break
    assert chosen is not None, "no relaxation level gives valid mappings"
    bal = pd.concat(bal_all)
    bal["used"] = bal.level == chosen
    tag_cols(bal, "label_free_metadata").to_csv(pman("shuffle_balance.tsv"), sep="\t", index=False)
    files = {}
    for (ctx, ns), mp in maps.items():
        p = cache("shuffle", f"{ctx}_null{ns}.tsv")
        mp.to_csv(p, sep="\t", index=False)
        files[f"{ctx}_null{ns}"] = dict(path=p, sha256=sha256(p), n_stubs=int(len(mp)), n_changed=int((mp.node_real != mp.node_null).sum()))
    dump(dict(rule="restricted double-edge swap inside each split; drug stub keeps evidence/sign/mapping features and tid blocks",
              null_seeds=list(NULL_SEEDS), levels=sh.LEVELS, relaxation_order=[0, 1, 2], level_used=chosen,
              validity=dict(min_stub_changed_rate=sh.MIN_CHANGED, max_drugs_fully_unchanged_rate=sh.MAX_UNCHANGED_DRUGS),
              attempts_per_stub=sh.ATTEMPTS_PER_STUB, fold_seed=kgm.FOLD_SEED, contexts={k: dict(splits=sorted(set(v[0].values())), train_split=v[1]) for k, v in contexts.items()},
              mappings=files, drug_seeds_sha256=sha256(os.path.join(os.environ["R7_KG"], "drug_seeds.tsv")),
              note="mappings are fixed before any run; the same drug group gets the same null targets in every cell line; no response array was read",
              commit=os.environ.get("R7F_COMMIT")), pman("target_shuffle_manifest.json"))
    summary.update(k_groups={sp_: kc[kc.split == sp_].set_index("k_group").n_drug_groups.to_dict() for sp_ in kc.split.unique()},
                   shuffle_level_used=chosen, seconds=round(time.time() - t0), commit=os.environ.get("R7F_COMMIT"), job=os.environ.get("SLURM_JOB_ID"))
    dump(summary, pman("d1_summary.json"))
    print(json.dumps(summary, indent=1))
    print(kc.to_string()); print(pd.DataFrame(ms).to_string()); print(ra.to_string()); print(ktab.to_string())
    print(bal[bal.used].to_string())


if __name__ == "__main__":
    main()
