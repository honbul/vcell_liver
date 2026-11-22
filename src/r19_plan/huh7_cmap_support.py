"""Audit CMap HUH7 exact-dose treatment and DMSO support without opening expression."""

import argparse
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r15_lincs.data import table
from src.r16_lincs.audit import geo_exposure
from src.r17_lincs.inventory import CMAP, ROOT, cmap_frames, ids, in_footprint, previous, register_footprint
from src.r19_plan.hcc_lincs_24h_support import forbidden_signatures
from src.r19_plan.hcc_time_lincs import cmap, cmap_measured
from src.r19_plan.i_audit import structure
from src.r19_plan.i_preflight import provenance, read_r18_domains


def run(args):
    protected = pd.concat([provenance(rows) for rows in read_r18_domains(args)], ignore_index=True)
    inst, signatures, paths, _ = cmap(args.base)
    book = ROOT / "LINCS2020/siginfo_beta.txt"
    blocked_members = forbidden_signatures(book, protected)
    blocked = register_footprint(protected)
    _, geo_ids, geo_coords, geo_hashes = geo_exposure(args.base / "vcell_liver_r15_lincs/v1")
    _, earlier, _, _ = cmap_frames(args.base)
    exposures, old_signatures, inputs = previous(args.base, earlier)
    historic = set().union(*(keys for (study, _), keys in old_signatures.items() if study == CMAP))
    old_members = forbidden_signatures(book, pd.DataFrame({"sig_id": sorted(historic)}))

    current = inst.loc[inst.cell_id.eq("HUH7") & inst.hours.eq(24) & inst.qc_well
        & inst.rna_plate.ne("") & inst.rna_well.str.fullmatch(r"[A-Z]+\d{2}")].copy()
    duplicate = current.duplicated("inst_id", keep=False) | current.duplicated(["rna_plate", "rna_well"], keep=False)
    current = current.loc[~duplicate]
    treated = current.loc[current.pert_type.eq("trt_cp")]
    treated = treated.merge(cmap_measured(treated), on="inst_id", validate="one_to_one")
    treated = treated.loc[treated.actual_dose_uM.ne("")]
    control = current.loc[current.pert_type.eq("ctl_vehicle") & (ids(current.pert_id).eq("DMSO") | ids(current.pert_iname).eq("DMSO"))]
    reg = table(ROOT / "LINCS2020/compoundinfo_beta.txt", usecols=["pert_id", "canonical_smiles"]).drop_duplicates()
    reg = reg.loc[reg.pert_id.isin(treated.pert_id)].copy()
    reg["parent"] = reg.canonical_smiles.map(structure)
    reg = reg[["pert_id", "parent"]].drop_duplicates()
    treated = treated.merge(reg.loc[reg.parent.ne("") & ~reg.pert_id.duplicated(keep=False)], on="pert_id", validate="many_to_one")

    def r18_free(rows):
        return rows.loc[~in_footprint(rows, blocked) & ~ids(rows.inst_id).isin(blocked_members)]

    def geo_free(rows):
        return rows.loc[~in_footprint(rows, (geo_ids, geo_coords))]

    retained = geo_free(r18_free(treated))
    exposed = ids(retained.inst_id).isin(old_members)
    for footprints in exposures.values():
        exposed |= in_footprint(retained, footprints)
    available = retained.loc[~exposed]
    control_clean = geo_free(r18_free(control))
    counts = control_clean.groupby("rna_plate").size()
    support = []
    for stage, rows in (("raw_unique_QC", treated), ("R18_excluded", r18_free(treated)),
                        ("R18_and_GEO_excluded", retained), ("previous_exposure_also_excluded", available),
                        ("three_vehicle_roles", available.loc[available.rna_plate.map(counts).ge(3)])):
        for dose in sorted(treated.actual_dose_uM.unique(), key=float):
            group = rows.loc[rows.actual_dose_uM.eq(dose)]
            repeats = group.groupby("parent").agg(plates=("rna_plate", "nunique"), distinct_positions=("rna_well", "nunique"))
            support.append(dict(stage=stage, actual_dose_uM=dose, parents=group.parent.nunique(), wells=len(group),
                plates=group.rna_plate.nunique(), repeated_off_position=int((repeats.plates.ge(2) & repeats.distinct_positions.ge(2)).sum())))
    plate_support = pd.concat((treated.groupby("rna_plate").size().rename("raw_treatment_wells"),
                               control.groupby("rna_plate").size().rename("raw_DMSO"),
                               r18_free(control).groupby("rna_plate").size().rename("DMSO_after_R18"),
                               counts.rename("DMSO_after_R18_and_GEO")), axis=1).fillna(0).astype(int)
    args.out.mkdir()
    pd.DataFrame(support).to_csv(args.out / "support.tsv", sep="\t", index=False)
    plate_support.to_csv(args.out / "plates.tsv", sep="\t")
    write_json(args.out / "receipt.json", dict(metadata_only=True, expression_read=False, matrix_opened=False,
        source_sha256={str(path): sha256(path) for path in paths}, GEO_sha256=geo_hashes,
        prior_metadata_sha256={name: sha256(path) for name, path in inputs.items()},
        reservation_registry_sha256=sha256(args.r18_registered / "receipt.json"),
        protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
        files={file.name: sha256(file) for file in args.out.iterdir()}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index", "protocol", "out"):
        cli.add_argument("--" + field, type=Path, required=True)
    run(cli.parse_args())
