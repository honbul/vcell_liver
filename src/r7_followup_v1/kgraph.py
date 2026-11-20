"""Label-free structure of the knowledge input along the model's ACTUAL computation graph.

GraphCorrection (src/evidence_gated_r7/models.py) adds the seed features to the hidden state of the seeded node and
then runs two message-passing blocks in which a node receives messages only from the sources of its incoming edges
(edge_index[0] -> edge_index[1], every relation channel, gates are sigmoids and never exactly zero). The readout of a
landmark l uses H2[l] and q. Hence, with q / status fixed, a seed at node t can change the output of landmark l only if
t == l (0 hop) or a directed path t -> l of length 1 or 2 exists in edge_index. Nothing else connects nodes
(LayerNorm is per node, the in-degree scaling is per node). This module computes exactly that reachability.

No expression array is imported here.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

FOLD_SEED = 20261118
SEED_FEATS = ["mech", "actB", "actF", "sign_pos", "sign_neg", "action_unknown", "map_exact", "map_connect_only", "multi_gene_target"]


def adjacency(edge_index, n):
    """Boolean adjacency A[u, v] = 1 if an edge u -> v exists in any channel."""
    a = sp.csr_matrix((np.ones(edge_index.shape[1], np.int32), (edge_index[0], edge_index[1])), shape=(n, n))
    a.data[:] = 1
    return a


def node_reach(edge_index, n, lm_idx):
    """Per node: number of landmarks reachable in exactly-min 0 / <=1 / <=2 directed hops, and the minimum hop (-1: none)."""
    A = adjacency(edge_index, n).astype(np.int32)
    is_lm = np.zeros(n, bool); is_lm[lm_idx] = True
    L = sp.csr_matrix((np.ones(len(lm_idx), np.int32), (lm_idx, np.arange(len(lm_idx)))), shape=(n, len(lm_idx)))
    r0 = L                                   # node -> itself if landmark
    r1 = A @ L                               # one hop
    r2 = A @ (A @ L)                         # two hops
    b0 = (r0 > 0); b1 = ((r0 + r1) > 0); b2 = ((r0 + r1 + r2) > 0)
    n0 = np.asarray(b0.sum(1)).ravel(); n1 = np.asarray(b1.sum(1)).ravel(); n2 = np.asarray(b2.sum(1)).ravel()
    minhop = np.where(n0 > 0, 0, np.where(np.asarray((r1 > 0).sum(1)).ravel() > 0, 1, np.where(n2 > 0, 2, -1)))
    outdeg = np.asarray(adjacency(edge_index, n).sum(1)).ravel(); indeg = np.asarray(adjacency(edge_index, n).sum(0)).ravel()
    return dict(n0=n0, n1=n1, n2=n2, minhop=minhop, reach_mat=b2.tocsr(), reach1_mat=b1.tocsr(), is_lm=is_lm, outdeg=outdeg, indeg=indeg)


def reach_class(R):
    """0: landmark (read directly), 1: off-panel but reaches a landmark within 2 hops, 2: off-panel and unreachable."""
    return np.where(R["is_lm"], 0, np.where(R["n2"] > 0, 1, 2))


def degree_bin(outdeg):
    """Out-degree bins fixed in advance: 0 / 1-4 / 5-19 / 20+."""
    return np.digitize(outdeg, [1, 5, 20])


def k_group_of(nodes, R):
    """K0 none / K1 >=1 landmark target / K2 all off-panel, >=1 target reaches a landmark within 2 hops / K3 otherwise."""
    if len(nodes) == 0:
        return "K0"
    rc = reach_class(R)[np.asarray(nodes)]
    return "K1" if (rc == 0).any() else ("K2" if (rc == 1).any() else "K3")


def load_kg(kgdir):
    import os
    g = lambda f: os.path.join(kgdir, f)
    ei = np.load(g("edge_index.npy")); rel = np.load(g("edge_rel.npy")); meas = np.load(g("node_measured.npy")); lm = np.load(g("landmark_node_idx.npy"))
    seeds = pd.read_csv(g("drug_seeds.tsv"), sep="\t")
    status = pd.read_csv(g("group_status.tsv"), sep="\t").sort_values("group_idx").reset_index(drop=True)
    return dict(edge_index=ei, edge_rel=rel, measured=meas, lm_idx=lm, seeds=seeds, status=status, n=len(meas),
                efeat=np.load(g("edge_feat.npy")), rewired=np.load(g("edge_index_rewired.npy")))


def k_groups(KG, edge_index=None):
    """Array of K labels indexed by group_idx (real annotation, no augmentation)."""
    R = node_reach(KG["edge_index"] if edge_index is None else edge_index, KG["n"], KG["lm_idx"])
    lab = np.array(["K0"] * len(KG["status"]), object)
    for g, d in KG["seeds"].groupby("group_idx"):
        lab[g] = k_group_of(d.node.to_numpy(), R)
    return lab, R


def assign_folds(units, status, klab):
    """3 outer folds over the ORIGINAL TRAIN drug groups (all cell lines of a drug group move together).
    Stratified on metadata only: K group, presence of an HCC unit, number of cell lines (1-3 / 4-9 / 10+).
    `sub` in 0..19 defines the inner selection split: for held fold k the inner-validation groups are the groups of the
    other two folds with sub in {3k, 3k+1, 3k+2} (15 % of the outer-train groups); the rest is inner-train."""
    tr = units[units.role == "train"]
    g = tr.groupby("group_idx").agg(n_cells=("cell", "nunique"), has_hcc=("is_hcc", "max")).reset_index()
    g["drug_group"] = status.drug_group.to_numpy()[g.group_idx.to_numpy()]
    g["k_group"] = klab[g.group_idx.to_numpy()]
    g["ncell_bin"] = np.digitize(g.n_cells, [4, 10])
    rng = np.random.RandomState(FOLD_SEED)
    g["fold"] = -1; g["sub"] = -1
    off = 0
    for key, d in sorted(g.groupby(["k_group", "has_hcc", "ncell_bin"]), key=lambda x: x[0]):
        idx = d.sort_values("group_idx").index.to_numpy()
        idx = idx[rng.permutation(len(idx))]
        pos = np.arange(len(idx)) + off
        g.loc[idx, "fold"] = pos % 3
        g.loc[idx, "sub"] = (pos // 3) % 20
        off += len(idx)
    return g.sort_values("group_idx").reset_index(drop=True)


def inner_role(fold, sub, k):
    """Role of a train drug group in the context of held fold k."""
    fold = np.asarray(fold); sub = np.asarray(sub)
    return np.where(fold == k, "held", np.where((sub >= 3 * k) & (sub < 3 * k + 3), "inner_val", "inner_train"))
