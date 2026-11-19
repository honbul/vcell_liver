"""R6-R step 1 (private plan 2026-11-19 section 3.2): eligibility of repeat contrasts from METADATA ONLY.

Reads no response, no expression value: only the well table of the TRAIN-drug wells (sample, plate,
cell line, drug group, cell counts, validity flag), the pool table, the split table and the public
Tahoe sample metadata (plate / drug / concentration / QC means; there is no date, passage or culture
batch column). Writes the eligible-set manifests that the audit (repeat_audit.py) must use unchanged.

Units
    well        one (sample, cell line); a Tahoe sample is one well of one 96-well plate with one drug
    stratum     (cell line, unordered plate pair {P, Q})
    common set  TRAIN drug groups with a usable well on BOTH plates of the stratum
    pair link   (stratum, unordered drug pair {d, e} inside the common set)  -> paired-plate contrast
    drug link   (stratum, drug d in the common set with >= 2 other common drugs) -> common-anchor centring
    same-plate  (cell line, plate, drug) with >= 2 usable wells (different samples on one plate)

A well is usable when the v1 well table marks it valid, its pool (plate, line) exists in the R6
contract, it has >= MIN_FULL cells and each barcode half has >= MIN_HALF cells (the v1 positive-control
rule). Duplicate usable wells of one (line, drug, plate) are averaged in the audit (cell-weighted).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MIN_HALF = 10
MIN_FULL = 2 * MIN_HALF
OUT = Path("manifests/r6_specificity_v2")


def usable_wells(halves_dir: Path, pools: pd.DataFrame, train: set[int]) -> pd.DataFrame:
    w = pd.read_csv(halves_dir / "wells.tsv", sep="\t")
    w = w[(w.kind == "treated") & w.drug.isin(train)].copy()
    pid = {(int(a), int(b)): int(c) for a, b, c in zip(pools.plate_index, pools.line, pools.pool_id)}
    w["pool"] = [pid.get((int(p), int(l)), -1) for p, l in zip(w.plate_index, w.line)]
    w["usable"] = w.valid.astype(bool) & (w.pool >= 0) & (w.n_full >= MIN_FULL) & (w.n_A >= MIN_HALF) & (w.n_B >= MIN_HALF)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halves", default=os.path.join(os.environ.get("R6F_ASSETS", "assets"), "cache", "halves_v1"))
    a = ap.parse_args()
    orig = Path(os.environ["R6_ORIG_ASSETS"]) / "cache" / "contract_v1"
    pools = pd.read_csv(orig / "pools.tsv", sep="\t")
    splits = pd.read_csv("manifests/chemcpa_moe_v2/splits.tsv", sep="\t")
    train = set(int(g) for g in splits.group_index[splits.split == "train"])
    w = usable_wells(Path(a.halves), pools, train)
    u = w[w.usable]
    plate_name = dict(zip(pools.plate_index.astype(int), pools.plate.astype(str)))

    # public sample metadata: what is (not) recorded about culture
    sm = pq.read_table(Path(os.environ["TAHOE_RAW"]) / "metadata" / "sample_metadata.parquet").to_pandas()
    sm = sm[sm["sample"].isin(set(u["sample"]))]
    culture_cols = [c for c in sm.columns if any(k in c.lower() for k in ("date", "batch", "passage", "day", "culture", "replicate"))]

    # ---- per (line, drug): plates and wells ---------------------------------------------------------
    per = u.groupby(["line", "drug", "plate_index"]).agg(n_wells=("sample", "nunique"), n_cells=("n_full", "sum")).reset_index()
    on = per.groupby(["line", "drug"]).plate_index.apply(lambda s: tuple(sorted(s))).rename("plates").reset_index()
    multi = on[on.plates.map(len) >= 2]

    # ---- strata: (line, plate pair) -> common drugs -----------------------------------------------
    strata = {}
    for r in multi.itertuples():
        for P, Q in itertools.combinations(r.plates, 2):
            strata.setdefault((int(r.line), int(P), int(Q)), []).append(int(r.drug))
    pair_rows, drug_rows, strat_rows = [], [], []
    for (l, P, Q), ds in sorted(strata.items()):
        ds = sorted(ds)
        strat_rows.append(dict(line=l, plate_P=P, plate_Q=Q, plate_pair=f"{P}-{Q}", n_common_drugs=len(ds), common_drugs=",".join(map(str, ds))))
        for d, e in itertools.combinations(ds, 2):
            others = [x for x in ds if x not in (d, e)]
            n_disjoint = len(others) * (len(others) - 1) // 2
            pair_rows.append(dict(line=l, plate_P=P, plate_Q=Q, plate_pair=f"{P}-{Q}", drug_d=d, drug_e=e,
                                  n_common_drugs=len(ds), n_disjoint_pairs=n_disjoint))
        if len(ds) >= 3:
            for d in ds:
                drug_rows.append(dict(line=l, plate_P=P, plate_Q=Q, plate_pair=f"{P}-{Q}", drug=d, n_anchor_drugs=len(ds) - 1))
    pairs, drugs_c, strata_df = pd.DataFrame(pair_rows), pd.DataFrame(drug_rows), pd.DataFrame(strat_rows)
    same = per[per.n_wells >= 2].copy()

    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT / "eligible_pair_links.tsv", sep="\t", index=False)
    drugs_c.to_csv(OUT / "eligible_anchor_links.tsv", sep="\t", index=False)
    strata_df.to_csv(OUT / "eligible_strata.tsv", sep="\t", index=False)
    same.to_csv(OUT / "eligible_sameplate_wells.tsv", sep="\t", index=False)
    u[["sample", "plate", "plate_index", "line", "cell_line_id", "drug", "pool", "wid", "n_A", "n_B", "n_full"]].to_csv(
        OUT / "usable_wells.tsv", sep="\t", index=False)

    def eff(x: pd.Series) -> float:
        s = x.value_counts(normalize=True)
        return float(1.0 / (s ** 2).sum()) if len(s) else 0.0

    by_pp = (pairs.groupby("plate_pair").agg(n_lines=("line", "nunique"), n_pair_links=("line", "size"),
                                             n_drugs=("drug_d", lambda s: 0)).reset_index() if len(pairs) else pd.DataFrame())
    if len(pairs):
        by_pp["n_drugs"] = [len(set(pairs.drug_d[pairs.plate_pair == pp]) | set(pairs.drug_e[pairs.plate_pair == pp])) for pp in by_pp.plate_pair]
        by_pp["n_drug_pairs"] = [len(set(zip(pairs.drug_d[pairs.plate_pair == pp], pairs.drug_e[pairs.plate_pair == pp]))) for pp in by_pp.plate_pair]
        by_pp["n_links_with_disjoint_wrongmatch"] = [int((pairs.n_disjoint_pairs[pairs.plate_pair == pp] > 0).sum()) for pp in by_pp.plate_pair]
    pair_drugs = sorted(set(pairs.drug_d) | set(pairs.drug_e)) if len(pairs) else []
    pw = pairs[pairs.n_disjoint_pairs > 0] if len(pairs) else pairs
    rec = dict(
        rule=dict(min_cells_full=MIN_FULL, min_cells_half=MIN_HALF, well_source="v1 halves_v1/wells.tsv (TRAIN-drug wells only)",
                  note="metadata only; no response or expression value was read"),
        n_train_drugs=len(train), n_train_drugs_with_usable_wells=int(u.drug.nunique()), n_usable_wells=int(len(u)),
        n_wells_dropped=int((~w.usable).sum()),
        plates={int(k): plate_name[k] for k in sorted(plate_name)},
        n_line_drug_conditions=int(len(on)), n_line_drug_on_2plus_plates=int(len(multi)),
        n_drugs_on_2plus_plates=int(multi.drug.nunique()), drugs_on_2plus_plates=sorted(int(x) for x in multi.drug.unique()),
        plate_sets_by_drug={int(g): sorted(set(itertools.chain.from_iterable(s.plates))) for g, s in multi.groupby("drug")},
        paired_plate=dict(n_strata=int(len(strata_df)), n_strata_with_2plus_common=int((strata_df.n_common_drugs >= 2).sum()) if len(strata_df) else 0,
                          n_pair_links=int(len(pairs)), n_unique_drug_pairs=int(len(set(zip(pairs.drug_d, pairs.drug_e)))) if len(pairs) else 0,
                          n_unique_drugs=len(pair_drugs), drugs=pair_drugs, n_plate_pairs=int(pairs.plate_pair.nunique()) if len(pairs) else 0,
                          n_lines=int(pairs.line.nunique()) if len(pairs) else 0,
                          n_pair_links_with_disjoint_wrongmatch=int(len(pw)),
                          n_unique_drugs_with_disjoint_wrongmatch=len(set(pw.drug_d) | set(pw.drug_e)) if len(pw) else 0,
                          effective_n_plate_pairs_by_links=eff(pairs.plate_pair) if len(pairs) else 0.0,
                          by_plate_pair=by_pp.to_dict(orient="records") if len(by_pp) else []),
        common_anchor=dict(n_drug_links=int(len(drugs_c)), n_unique_drugs=int(drugs_c.drug.nunique()) if len(drugs_c) else 0,
                           n_plate_pairs=int(drugs_c.plate_pair.nunique()) if len(drugs_c) else 0),
        same_plate_repeat_wells=dict(n_line_drug_plate=int(len(same)), n_drugs=int(same.drug.nunique()),
                                     n_plates=int(same.plate_index.nunique()), drugs=sorted(int(x) for x in same.drug.unique())),
        culture_metadata=dict(sample_metadata_columns=list(sm.columns), culture_like_columns=culture_cols,
                              independent_culture_identifiable=bool(culture_cols),
                              note="different plate = different well and plate; whether day/passage/cell stock differ is not recorded"),
    )
    (OUT / "eligibility_summary.json").write_text(json.dumps(rec, indent=1, default=int))
    print(json.dumps({k: v for k, v in rec.items() if k != "plate_sets_by_drug"}, indent=1, default=int))
    print(json.dumps(rec["plate_sets_by_drug"]))


if __name__ == "__main__":
    main()
