"""Source cell families and the meta-train / meta-validation split. Reads metadata only (no expression)."""
import argparse, json, os
import pandas as pd

from .common import B_MAN, public, norm_name, EXCLUDED_TARGETS

# Documented derivations among the 161 source cell names (reason kept next to each pair).
DERIVED = [
    ("HEK293", "HEK293T", "HEK293T is the SV40 large-T derivative of HEK293"),
    ("LNCAP", "C42", "C4-2 is a subline derived from LNCaP"),
    ("NPC", "NEU", "release alias: both are NPC-8330-8 (progenitors and the neurons differentiated from them)"),
    ("H1975", "NCIH1975", "same line under two names (same Cellosaurus id CVCL_1511)"),
    ("HIMG001", "HIMG002", "same series and plates (CEGS002); donor relation not documented in the release, merged conservatively"),
    ("SKB", "SKL", "release lists both as skeletal-muscle primary material without donor ids; merged conservatively"),
]
POOL_PREFIX = "XC."      # mixed-cell pools; composition is not in the release metadata on the server


def build(cellinfo_path):
    old = pd.read_csv(os.path.join(B_MAN, "source_split.tsv"), sep="\t")
    cells = sorted(old.cell)
    assert len(cells) == 161 and not any(norm_name(c) in EXCLUDED_TARGETS or "HUH" in norm_name(c) or "JHH" in norm_name(c) for c in cells)
    ci = pd.read_csv(cellinfo_path, sep="\t", dtype=str).fillna("").set_index("cell_iname")
    parent = {c: c for c in cells}
    why = {c: [] for c in cells}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    def union(a, b, reason):
        ra, rb = find(a), find(b)
        why[a].append(reason)
        why[b].append(reason)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_cvcl = {}
    for c in cells:
        cv = ci.cellosaurus_id.get(c, "")
        if cv:
            by_cvcl.setdefault(cv, []).append(c)
    for cv, cs in by_cvcl.items():
        for c in cs[1:]:
            union(cs[0], c, f"same Cellosaurus id {cv}")
    for a, b, r in DERIVED:
        assert a in parent and b in parent, (a, b)
        union(a, b, r)
    pools = [c for c in cells if c.startswith(POOL_PREFIX)]
    assert all(ci.cell_type.get(c, "") == "pool" for c in pools) and not [c for c in cells if ci.cell_type.get(c, "") == "pool" and c not in pools]
    for c in pools[1:]:
        union(pools[0], c, "mixed-cell pool of undocumented composition; all XC.* pools form one family")
    fam = {c: find(c) for c in cells}
    members = {}
    for c, f in fam.items():
        members.setdefault(f, []).append(c)
    name = {f: (("XC_POOLS" if f.startswith(POOL_PREFIX) else "+".join(sorted(m))) if len(m) > 1 else f) for f, m in members.items()}
    tab = old[["cell", "rows", "conditions", "role"]].rename(columns={"role": "old_role"})
    tab["family"] = tab.cell.map(lambda c: name[fam[c]])
    straddle = tab.groupby("family").old_role.nunique()
    straddle = set(straddle[straddle > 1].index)
    tab["meta_role"] = tab.old_role
    tab.loc[tab.family.isin(straddle), "meta_role"] = "train"       # minimal fix: the straddling family goes to meta-train as a whole
    tab["moved"] = (tab.meta_role != tab.old_role).astype(int)
    tab["family_reason"] = tab.cell.map(lambda c: "; ".join(sorted(set(why[c]))))
    tab["cell_type"] = tab.cell.map(ci.cell_type).fillna("")
    assert tab.groupby("family").meta_role.nunique().max() == 1
    return tab, sorted(straddle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellinfo", default=os.environ.get("R2_CELLINFO"))
    a = ap.parse_args()
    tab, straddle = build(a.cellinfo)
    out = public("manifests", "episodic_residual_r2", "source_families.tsv")
    tab.to_csv(out, sep="\t", index=False)
    summ = dict(cells=len(tab), families=int(tab.family.nunique()), multi_cell_families={f: sorted(g.cell) for f, g in tab.groupby("family") if len(g) > 1},
                old_split=dict(train=int((tab.old_role == "train").sum()), val=int((tab.old_role == "val").sum())),
                families_straddling_old_split=straddle, cells_moved_val_to_train=sorted(tab.cell[tab.moved == 1]),
                meta_train=dict(cells=int((tab.meta_role == "train").sum()), families=int(tab.family[tab.meta_role == "train"].nunique())),
                meta_val=dict(cells=sorted(tab.cell[tab.meta_role == "val"]), families=int(tab.family[tab.meta_role == "val"].nunique())))
    json.dump(summ, open(public("manifests", "episodic_residual_r2", "source_families_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
