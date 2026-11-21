"""Audit a same-parent, exact-exposure LINCS to existing HepaRG bridge."""

import argparse
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import clean, structural_keys
from src.r15_lincs.data import asset
from src.r17_lincs.inventory import ROOT
from src.r19_plan.audit_r18 import geo_metadata, prepare
from src.r19_plan.i_preflight import provenance, read_pinned, read_r18_domains
from src.r19_plan.t4_support import TIME_PAIR, eligible_links, matched_pairs, measured_dose

DATASETS = {"GSE284321": "heparg_284321", "GSE296254": "heparg"}
BRIDGE_KEY = ["parent", "actual_dose_uM"]
STUDY = "GSE92742"
CELLS = ("HEPG2", "PHH", "HUH7")


def temp_o_seq(args, dataset, hashes):
    root = args.heparg / DATASETS[dataset]
    path = root / "receipt.json"
    if sha256(path) != hashes["heparg"][dataset]:
        raise ValueError("HepaRG source metadata changed since P3")
    receipt = read_json(path)
    rows, controls = [read_pinned(root, name, receipt["files"][name]) for name in ("rows.tsv", "vehicles.tsv")]
    rows = rows.loc[rows.parent_key.ne("") & rows.smiles.ne("") & rows.time.eq("24h")].copy()
    solvents = {"dmso", "dimethyl sulfoxide"}
    controls = controls.loc[controls.chemical.str.strip().str.lower().isin(solvents) & controls.time.eq("24h")]
    fields = ["plate", "position"]
    if (not rows.physical_well_id.eq(rows.plate+"_"+rows.position).all()
            or not controls.physical_well_id.eq(controls.plate+"_"+controls.position).all()
            or pd.concat((rows, controls), ignore_index=True).duplicated(fields).any()
            or any(not pd.to_numeric(frame.mapped_library, errors="raise").ge(100_000).all()
                   for frame in (rows, controls))):
        raise ValueError("HepaRG metadata do not identify disjoint QC physical treatment and control wells")
    raw_dose = rows.concentration.str.extract(r"^([-+\d.eE]+)\s*uM$", expand=False)
    if raw_dose.isna().any():
        raise ValueError("HepaRG rows need explicit measured concentration in µM")
    keys = {text: format(Decimal(text).normalize(), "f") for text in raw_dose.unique()}
    rows["actual_dose_uM"] = raw_dose.map(keys)
    chemistry = {s: structural_keys(clean(s))[1] for s in rows.smiles.unique()}
    rows["parent"] = rows.smiles.map(chemistry)
    rows = rows.loc[rows.parent.ne("")].copy()
    controls_by_plate = controls.groupby("plate").physical_well_id.nunique()
    rows["target_controls"] = rows.plate.map(controls_by_plate).fillna(0).astype(int)
    return rows, sha256(path)


def bridge_parts(source, target, study, cell):
    base = source.loc[source.cell_id.eq(cell)]
    disjoint = base.loc[base.cross_time_plate_disjoint.astype(bool)]
    qualified = disjoint.loc[disjoint.both_times_2plates.astype(bool), [*BRIDGE_KEY, "fold"]]
    context = base[BRIDGE_KEY].drop_duplicates()
    match = target.merge(context, on=BRIDGE_KEY, how="inner", validate="many_to_one")
    paired = target.merge(qualified, on=BRIDGE_KEY, how="inner", validate="many_to_one")
    fit = paired.loc[paired.target_controls.ge(3)].copy()
    if fit.empty:
        proven = fit
    else:
        # An independent culture and a different plate and position must coexist for the same label and exact dose.
        repeats = [*BRIDGE_KEY, "identity", "chemical"]
        active = fit.loc[fit.culture.ne("")]
        combinations = active[repeats + ["physical_well_id", "plate", "position", "culture"]].merge(
            active[repeats + ["physical_well_id", "plate", "position", "culture"]], on=repeats,
            suffixes=("", "_other"), validate="many_to_many")
        keep = (combinations.plate.ne(combinations.plate_other) & combinations.position.ne(combinations.position_other)
                & combinations.culture.ne(combinations.culture_other))
        proven = fit.loc[fit.physical_well_id.isin(combinations.loc[keep, "physical_well_id"])]
    outputs = {}
    for label, rows in (("exact_time_and_dose", match), ("independent_lincs_time_pair", paired),
                        ("plus_3_target_dmso", fit), ("plus_independent_target_repeat", proven)):
        n_fold = rows.groupby("fold").parent.nunique().reindex(range(3), fill_value=0) if "fold" in rows else None
        outputs[label] = dict(target=study, source_cell=cell, level=label,
            parents=rows.parent.nunique(), parent_doses=len(rows[BRIDGE_KEY].drop_duplicates()), wells=len(rows),
            plates=rows.plate.nunique(), cultures=rows.loc[rows.culture.ne(""), "culture"].nunique(),
            **{f"f{f}_parents": str(n_fold[f]) if n_fold is not None else "NA" for f in range(3)})
    return outputs, fit


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    contract = read_json(args.support_reference)
    if (not contract["metadata_only"] or contract["expression_read"]
            or sha256(args.r18_access/"receipt.json") != contract["reservation_access_sha256"]
            or any(sha256(path) != h for path, h in contract["source_metadata_sha256"][STUDY].items())):
        raise ValueError("The metadata inputs differ from the earlier nonreserved T4 source")
    p3 = read_json(args.p3_reference)
    source = pd.concat([provenance(rows) for rows in read_r18_domains(args)[:4]], ignore_index=True)
    inst, sig = geo_metadata(STUDY, args.base/"vcell_liver_r15_lincs/v1")
    inst = prepare(inst)
    treated = inst.pert_type.eq("trt_cp") & inst.hours.isin(TIME_PAIR)
    dose = measured_dose(asset(ROOT, STUDY, "inst_info"), set(inst.loc[treated, "inst_id"]))
    eligible, _ = eligible_links(inst, sig, source, dose)
    paired = matched_pairs(eligible)
    result, inputs, permitted = [], {}, []
    for dataset in DATASETS:
        target, inputs[dataset] = temp_o_seq(args, dataset, p3["input_receipts"])
        for cell in CELLS:
            rows, support = bridge_parts(paired, target, dataset, cell)
            result.extend(rows.values())
            permitted.append(support[["library", "plate", "position", "culture", *BRIDGE_KEY, "fold"]]
                             .assign(dataset=dataset, source_cell=cell))
    args.out.mkdir(parents=True)
    pd.DataFrame(result).to_csv(args.out/"support.tsv", sep="\t", index=False)
    pd.concat(permitted, ignore_index=True).to_csv(args.out/"target_allowlist.tsv", sep="\t", index=False)
    write_json(args.out/"receipt.json", dict(metadata_only=True, expression_read=False, source=STUDY,
        source_audit_sha256=sha256(args.support_reference), target_audit_sha256=sha256(args.p3_reference),
        targets=inputs, code_sha256=sha256(__file__),
        files={f.name: sha256(f) for f in sorted(args.out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reports", "heparg", "r18-registered", "r18-lock", "r18-access", "r18-index",
                 "p3-reference", "support-reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
