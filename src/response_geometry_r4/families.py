"""Cell families over the 161 B source cells plus the three HCC lines. Metadata only (no expression).
The derivation list and the pool / Cellosaurus rules are copied from the R2 family rule (metadata, not results)."""
import argparse, json, os
import pandas as pd

from .common import B_MAN, HCC, public

DERIVED = [
    ("HEK293", "HEK293T", "HEK293T is the SV40 large-T derivative of HEK293"),
    ("LNCAP", "C42", "C4-2 is a subline derived from LNCaP"),
    ("NPC", "NEU", "release alias: both are NPC-8330-8 (progenitors and the neurons differentiated from them)"),
    ("H1975", "NCIH1975", "same line under two names (same Cellosaurus id CVCL_1511)"),
    ("HIMG001", "HIMG002", "same series and plates (CEGS002); donor relation not documented, merged conservatively"),
    ("SKB", "SKL", "release lists both as skeletal-muscle primary material without donor ids; merged conservatively"),
]
POOL_PREFIX = "XC."


def build(cellinfo_path):
    src = pd.read_csv(os.path.join(B_MAN, "source_split.tsv"), sep="\t")
    cells = sorted(src.cell) + list(HCC)
    assert len(cells) == 164 and len(set(cells)) == 164
    ci = pd.read_csv(cellinfo_path, sep="\t", dtype=str).fillna("").set_index("cell_iname")
    parent = {c: c for c in cells}
    why = {c: [] for c in cells}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    def union(a, b, reason):
        ra, rb = find(a), find(b)
        why[a].append(reason); why[b].append(reason)
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
        union(a, b, r)
    pools = [c for c in cells if c.startswith(POOL_PREFIX)]
    for c in pools[1:]:
        union(pools[0], c, "mixed-cell pool of undocumented composition; all XC.* pools form one family")
    fam = {c: find(c) for c in cells}
    members = {}
    for c, f in fam.items():
        members.setdefault(f, []).append(c)
    name = {f: (("XC_POOLS" if f.startswith(POOL_PREFIX) else "+".join(sorted(m))) if len(m) > 1 else f) for f, m in members.items()}
    tab = pd.DataFrame(dict(cell=cells))
    tab["family"] = tab.cell.map(lambda c: name[fam[c]])
    tab["is_hcc"] = tab.cell.isin(HCC).astype(int)
    tab["family_reason"] = tab.cell.map(lambda c: "; ".join(sorted(set(why[c]))))
    tab["cell_type"] = tab.cell.map(ci.cell_type).fillna("")
    for h in HCC:   # each HCC line must be its own family
        assert (tab.family == h).sum() == 1, h
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellinfo", default=os.environ.get("R4_CELLINFO"))
    a = ap.parse_args()
    tab = build(a.cellinfo)
    tab.to_csv(public("manifests", "response_geometry_r4", "cell_families.tsv"), sep="\t", index=False)
    print(dict(cells=len(tab), families=tab.family.nunique(), multi={f: sorted(g.cell) for f, g in tab.groupby("family") if len(g) > 1}))


if __name__ == "__main__":
    main()
