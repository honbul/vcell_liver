"""Target-identity null: restricted double-edge swap of the drug-target bipartite graph (plan section 9.3).

(drug A, target u), (drug B, target v) -> (drug A, target v), (drug B, target u), separately inside each split.
The drug stub keeps its evidence / sign / mapping features and its tid blocks, so per-drug target count, status vector,
m_available and every augmentation draw are unchanged; the per-split occurrence count of every target is unchanged.
A swap is allowed only between targets of the same stratum:
  level 0: (reachability class, out-degree bin, seen-in-train)      <- strict
  level 1: (reachability class, seen-in-train)                       <- degree condition relaxed first
  level 2: (seen-in-train)                                           <- reachability condition relaxed next
The relaxation order and the validity rule are fixed before any response is read:
  a mapping is valid if in every split >= 90 % of the stubs changed target and <= 5 % of the annotated drugs kept
  their complete original target set. The study uses the strictest level at which ALL mappings are valid.
No response array is imported here.
"""
import numpy as np
import pandas as pd

from .kgraph import reach_class, degree_bin

LEVELS = {0: "reach_class+outdeg_bin+seen_in_train", 1: "reach_class+seen_in_train", 2: "seen_in_train"}
MIN_CHANGED = 0.90
MAX_UNCHANGED_DRUGS = 0.05
ATTEMPTS_PER_STUB = 40


def strata(nodes, R, seen_nodes, level):
    rc = reach_class(R)[nodes]; db = degree_bin(R["outdeg"])[nodes]; seen = np.isin(nodes, seen_nodes).astype(int)
    if level == 0:
        return rc * 100 + db * 10 + seen
    if level == 1:
        return rc * 100 + seen
    return seen


def swap_split(drug, node, strat, rng):
    """drug, node: stub arrays of one split. Returns the null node per stub."""
    node = node.copy()
    n = len(node)
    tset = {}
    for d, t in zip(drug, node):
        tset.setdefault(d, set()).add(t)
    members = {s: np.where(strat == s)[0] for s in np.unique(strat)}
    for _ in range(ATTEMPTS_PER_STUB * n):
        i = rng.randint(n)
        m = members[strat[i]]
        if len(m) < 2:
            continue
        j = m[rng.randint(len(m))]
        di, dj, ti, tj = drug[i], drug[j], node[i], node[j]
        if di == dj or ti == tj or tj in tset[di] or ti in tset[dj]:
            continue
        tset[di].remove(ti); tset[di].add(tj); tset[dj].remove(tj); tset[dj].add(ti)
        node[i], node[j] = tj, ti
    return node


def make_mapping(seeds, split_of_group, train_split, R, level, null_seed):
    """seeds: drug_seeds frame (row order = stub order). split_of_group: dict group_idx -> split name (groups absent are
    left untouched and flagged 'unused'). Returns a frame with node_real / node_null per stub."""
    s = seeds[["group_idx", "node"]].copy().reset_index(drop=True)
    s["stub"] = s.groupby("group_idx").cumcount()
    s["split"] = s.group_idx.map(split_of_group).fillna("unused")
    s["node_real"] = s.node; s["node_null"] = s.node
    seen_nodes = np.unique(s.node[s.split == train_split])
    rng = np.random.RandomState(null_seed)
    for sp_ in sorted(x for x in s.split.unique() if x != "unused"):
        k = np.where(s.split == sp_)[0]
        st = strata(s.node.to_numpy()[k], R, seen_nodes, level)
        s.loc[k, "node_null"] = swap_split(s.group_idx.to_numpy()[k], s.node.to_numpy()[k], st, rng)
    return s.drop(columns="node")


def balance(m, R, kreal):
    """Per-split balance of one mapping (label-free)."""
    rows = []
    rc = reach_class(R)
    for sp_, d in m[m.split != "unused"].groupby("split"):
        ch = (d.node_real != d.node_null).to_numpy()
        g = d.groupby("group_idx")
        jac, unchanged, ksame = [], 0, 0
        for gi, x in g:
            a, b = set(x.node_real), set(x.node_null)
            jac.append(len(a & b) / len(a | b)); unchanged += int(a == b)
            rcn = rc[x.node_null.to_numpy()]
            kn = "K1" if (rcn == 0).any() else ("K2" if (rcn == 1).any() else "K3")
            ksame += int(kn == kreal[gi])
        fr = d.node_real.value_counts().sort_index(); fn = d.node_null.value_counts().sort_index()
        rows.append(dict(split=sp_, n_drugs=int(g.ngroups), n_stubs=int(len(d)), stub_changed_rate=float(ch.mean()),
                         drugs_fully_unchanged_rate=unchanged / g.ngroups, mean_jaccard_real_vs_null=float(np.mean(jac)),
                         target_count_per_drug_preserved=bool((g.node_null.nunique() == g.node_real.nunique()).all() and (g.node_null.nunique() == g.size()).all()),
                         target_frequency_preserved=bool(fr.equals(fn)),
                         k_group_preserved_rate=ksame / g.ngroups,
                         reach_class_preserved_rate=float((rc[d.node_real.to_numpy()] == rc[d.node_null.to_numpy()]).mean()),
                         mean_abs_outdeg_diff=float(np.abs(R["outdeg"][d.node_real.to_numpy()] - R["outdeg"][d.node_null.to_numpy()]).mean()),
                         mean_outdeg_real=float(R["outdeg"][d.node_real.to_numpy()].mean()), mean_outdeg_null=float(R["outdeg"][d.node_null.to_numpy()].mean()),
                         mean_reachable_landmarks_real=float(R["n2"][d.node_real.to_numpy()].mean()), mean_reachable_landmarks_null=float(R["n2"][d.node_null.to_numpy()].mean())))
    return pd.DataFrame(rows)


def valid(bal):
    return bool((bal.stub_changed_rate >= MIN_CHANGED).all() and (bal.drugs_fully_unchanged_rate <= MAX_UNCHANGED_DRUGS).all()
                and bal.target_count_per_drug_preserved.all() and bal.target_frequency_preserved.all())


def load_mapping(path):
    """dict group_idx -> null node array in stub order."""
    m = pd.read_csv(path, sep="\t")
    return {g: d.sort_values("stub").node_null.to_numpy(np.int64) for g, d in m.groupby("group_idx")}
