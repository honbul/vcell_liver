"""Audit whether local Open TG-GATEs metadata can support a PHH T4 bridge."""

import argparse
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import normalized_name
from src.r15_lincs.data import asset
from src.r17_lincs.inventory import ROOT
from src.r19_plan.audit_r18 import geo_metadata, prepare
from src.r19_plan.i_preflight import provenance, read_r18_domains
from src.r19_plan.t4_support import TIME_PAIR, eligible_links, matched_pairs, measured_dose


STUDY = "GSE92742"
ARCHIVE_ENTRY = "open_tggates_cel_file_attribute.csv"
DOI = "10.18908/lsdba.nbdc00954-01-003"
KEYS = ["parent", "actual_dose_uM"]


def target_metadata(path):
    with ZipFile(path) as archive, archive.open(ARCHIVE_ENTRY) as stream:
        rows = pd.read_csv(stream, encoding="cp932", dtype=str, keep_default_na=False)
    target = rows.loc[rows.SPECIES.eq("Human") & rows.EXP_TEST_TYPE.eq("in vitro")
                      & rows.ORGAN.eq("Liver") & rows.SACRIFICE_PERIOD.eq("24 hr")].copy()
    if target.empty or target.BARCODE.duplicated().any() or not target.ARR_DESIGN.eq("HG-U133_Plus_2").all():
        raise ValueError("Open TG-GATEs human in vitro microarray metadata changed")
    controls = target.loc[target.DOSE_LEVEL.eq("Control")]
    treated = target.loc[target.DOSE_LEVEL.ne("Control")].copy()
    direct = treated.loc[treated.DOSE_UNIT.str.replace("μ", "u").str.replace("µ", "u").str.casefold().eq("um")].copy()
    key = {v: format(Decimal(v).normalize(), "f") for v in direct.DOSE.unique()}
    direct["actual_dose_uM"] = direct.DOSE.map(key)
    direct["name_key"] = direct.COMPOUND_NAME.map(normalized_name)
    return treated, controls, direct


def source_pairs(args):
    pinned = read_json(args.support_reference)
    metadata = pinned["source_metadata_sha256"][STUDY]
    if (not pinned["metadata_only"] or pinned["expression_read"]
            or sha256(args.r18_access/"receipt.json") != pinned["reservation_access_sha256"]
            or any(sha256(k) != h for k, h in metadata.items())):
        raise ValueError("The original nonreserved source audit changed")
    protected = pd.concat([provenance(frame) for frame in read_r18_domains(args)[:4]], ignore_index=True)
    base = args.base/"vcell_liver_r15_lincs/v1"
    inst, sig = geo_metadata(STUDY, base)
    inst = prepare(inst)
    trt = inst.pert_type.eq("trt_cp") & inst.hours.isin(TIME_PAIR)
    dose = measured_dose(asset(ROOT, STUDY, "inst_info"), set(inst.loc[trt, "inst_id"]))
    links, _ = eligible_links(inst, sig, protected, dose)
    pairs = matched_pairs(links)
    keep = pairs.cell_id.eq("PHH") & pairs.cross_time_plate_disjoint & pairs.both_times_2plates
    identity_path = base/"audit"/STUDY/"identities.tsv"
    names = pd.read_csv(identity_path, sep="\t", keep_default_na=False, dtype=str, usecols=["parent","pert_iname"])
    names["name_key"] = names.pert_iname.map(normalized_name)
    names = names.loc[names.name_key.ne("") & names.parent.ne("") & ~names.parent.str.startswith("UNRESOLVED:")]
    unique = names.groupby("name_key").parent.transform("nunique").eq(1)
    return pairs.loc[keep, KEYS], names.loc[unique, ["name_key", "parent"]].drop_duplicates()


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    treated, controls, target = target_metadata(args.catalog)
    pairs, names = source_pairs(args)
    dose_ok = target.loc[target.actual_dose_uM.isin(pairs.actual_dose_uM)]
    same = dose_ok.merge(names, on="name_key", how="inner", validate="many_to_one")
    exact = same.merge(pairs, on=KEYS, how="inner", validate="many_to_one")
    result = []
    for name, frame in (("24h_treated", treated), ("24h_vehicle_group", controls), ("direct_molar", target),
                        ("dose_only_upper_bound", dose_ok), ("plus_unique_normalized_LINCS_name", same),
                        ("plus_same_LINCS_PHH_6h_24h_parent", exact)):
        result.append(dict(level=name, compounds=frame.COMPOUND_NAME.nunique(), microarrays=frame.BARCODE.nunique(),
            study_experiments=frame.EXP_ID.nunique(), registered_physical_plate_role_support=False))
    args.out.mkdir(parents=True)
    pd.DataFrame(result).to_csv(args.out/"support.tsv", sep="\t", index=False)
    exact.groupby(["COMPOUND_NAME", "actual_dose_uM"]).agg(
        array_samples=("BARCODE", "nunique"), experiment_ids=("EXP_ID", "nunique"),
        sample_ids=("INDIVIDUAL_ID", "nunique")).reset_index().to_csv(args.out/"matches.tsv", sep="\t", index=False)
    write_json(args.out/"receipt.json", dict(metadata_only=True, expression_read=False, source_parents=pairs.parent.nunique(),
        source_conditions=len(pairs), tag="Open TG-GATEs CEL attributes", doi=DOI,
        source_reference_sha256=sha256(args.support_reference), catalog_sha256=sha256(args.catalog),
        code_sha256=sha256(__file__), files={p.name: sha256(p) for p in sorted(args.out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index",
                "support-reference", "catalog", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())
