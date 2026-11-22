"""Verify RNA coordinates for GSE70138 using already-held CMap instance aliases."""

import argparse
from pathlib import Path
from zlib import crc32

import pandas as pd

from src.r15_lincs.data import asset, table
from src.r16_lincs.audit import DET, RNA, token
from src.r17_lincs.inventory import ROOT


def find_aliases(path, eligible):
    columns = ["sample_id", "cell_iname", "pert_id", *RNA, *DET]
    identifiers = set(eligible.inst_id)
    detection = set(zip(eligible.det_plate, eligible.det_well))
    plates = set(eligible.det_plate)
    selected = []
    for chunk in pd.read_csv(path, sep="\t", usecols=columns, dtype=str, keep_default_na=False, chunksize=150000):
        chunk = chunk.loc[chunk.det_plate.isin(plates) | chunk.sample_id.isin(identifiers)].copy()
        for col in DET:
            chunk[col] = token(chunk[col])
        matched = pd.MultiIndex.from_frame(chunk[DET]).isin(detection)
        selected.append(chunk.loc[matched | chunk.sample_id.isin(identifiers)])
    return pd.concat(selected, ignore_index=True).drop_duplicates()


def audit(args):
    exposure = table(args.audit / "exposure_manifest.tsv")
    treated = exposure.loc[
        exposure.study.eq("GSE70138") & exposure.cohort.eq("C2")
        & exposure.qc.eq("True") & exposure.all_prior_exact.eq("False")
        & exposure.parent.ne("") & exposure.geo_duplicate.eq("False")
    ].copy()
    treated = treated.drop_duplicates(DET)
    inst_path = asset(ROOT, "GSE70138", "inst_info")
    geo = table(inst_path, usecols=["inst_id", "cell_id", "pert_type", "pert_id", "pert_iname", *DET])
    dmso = geo.loc[geo.pert_type.eq("ctl_vehicle") & (
        geo.pert_id.str.upper().eq("DMSO") | geo.pert_iname.str.upper().eq("DMSO")
    )].copy()
    dmso = dmso.loc[dmso.cell_id.eq("HEPG2") & token(dmso.det_plate).isin(treated.det_plate)]
    used = pd.concat([treated[["inst_id", "cell_id", "pert_id", *DET]], dmso], ignore_index=True)
    used["control"] = used.pert_type.eq("ctl_vehicle")
    for col in DET:
        used[col] = token(used[col])
    used = used.drop(columns=["pert_type", "pert_iname"])
    known = find_aliases(args.cmap, used).rename(columns={"cell_iname": "cell_id"})
    for col in RNA:
        known[col] = token(known[col])
    valid = known.rna_plate.ne("") & known.rna_well.str.fullmatch(r"[A-Z]+\d{2}")
    known = known.loc[valid].copy()
    keys = [*DET, "cell_id", "pert_id"]
    joined = used.merge(known, on=keys, how="left")
    exact = used.merge(known, left_on=["inst_id", "cell_id", "pert_id"],
                       right_on=["sample_id", "cell_id", "pert_id"], suffixes=("", "_alias"))
    exact = exact.drop(columns=[f"{col}_alias" for col in DET])
    hints = pd.concat([joined, exact], ignore_index=True).dropna(subset=RNA).drop_duplicates(["inst_id", *RNA])
    confirmed = hints.loc[~hints.inst_id.duplicated(keep=False)].copy()
    theta = confirmed.loc[confirmed.control].sort_values("inst_id").drop_duplicates(RNA)
    theta["crc"] = theta.inst_id.map(lambda well: crc32(well.encode()))
    theta = theta.sort_values(["rna_plate", "crc", "inst_id"])
    theta["role"] = theta.groupby("rna_plate").cumcount().mod(3).map(dict(enumerate("ABC")))
    n = theta.groupby("rna_plate").size()
    eligible = confirmed.loc[~confirmed.control & confirmed.rna_plate.map(n).fillna(0).ge(3)]
    args.out.mkdir(exist_ok=False)
    confirmed.to_csv(args.out / "rna_crosswalk.tsv", sep="\t", index=False)
    theta.loc[theta.rna_plate.map(n).ge(3)].drop(columns="crc").to_csv(
        args.out / "rna_assignments.tsv", sep="\t", index=False)
    linked = treated.merge(eligible[["inst_id", *RNA]], on="inst_id", suffixes=("", "_verified"))
    linked = linked.drop_duplicates([f"{column}_verified" for column in RNA])
    support = []
    for portion, parent_rows in [("all", treated), *list(treated.groupby("dose"))]:
        good = linked.loc[linked.inst_id.isin(parent_rows.inst_id)]
        support.append({"dose_uM": portion, "DET_clean_parent_count": parent_rows.parent.nunique(),
                        "DET_clean_well_count": len(parent_rows), "RNA_matched_wells": int(
                            parent_rows.inst_id.isin(confirmed.inst_id).sum()),
                        "RNA_3role_parent_count": good.parent.nunique(), "RNA_3role_well_count": len(good),
                        "RNA_3role_plate_count": good.rna_plate_verified.nunique(),
                        "RNA_3role_B_parent_count": good.loc[good.B_eligible.eq("True"), "parent"].nunique()})
    pd.DataFrame(support).to_csv(args.out / "rna_support.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cmap", "audit", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    audit(parser.parse_args())
