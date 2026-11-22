"""Count common exact-dose T4 and archived Tahoe physical supports without expression."""

import argparse
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP
from src.r13_optimization.lincs_transfer import clean, structural_keys
from src.r15_lincs.data import STUDIES, asset
from src.r17_lincs.inventory import ROOT
from src.r19_plan.audit_r18 import geo_metadata, prepare
from src.r19_plan.i_preflight import provenance, read_pinned, read_r18_domains
from src.r19_plan.t2_control_support import LIVER
from src.r19_plan.t4_support import TIME_PAIR, eligible_links, matched_pairs, measured_dose


def target_scope(args):
    cache = args.tahoe / "t1-cache"
    reference = read_json(args.tahoe_reference)
    if sha256(cache/"receipt.json") != reference["tahoe_stage_sha256"]:
        raise ValueError("The Tahoe metadata differ from the retained shift-control audit")
    stamp = read_json(cache/"receipt.json")
    eligible = []
    for fold in range(3):
        name = f"f{fold}/score_rows.tsv"
        rows = read_pinned(cache, name, stamp["files"][name])
        rows = rows.loc[rows.dose_uM.astype(float).eq(5) & rows.time_h.astype(float).eq(24)
                        & rows.n_cells.astype(int).ge(20)].copy()
        pool = ["plate_id", "cell_line_id", "dose_uM", "time_h"]
        rows = rows.loc[rows.groupby(pool)[GROUP].transform("nunique").ge(3)
                        & rows.cell_line_id.isin(LIVER)]
        eligible.append(rows)
    target = pd.concat(eligible, ignore_index=True)
    reg = args.entities.parent / "receipt.json"
    receipt = read_json(reg)
    if receipt["outputs"]["entities.tsv"] != sha256(args.entities):
        raise ValueError("Tahoe compound identity metadata changed after the structure audit")
    entities = pd.read_csv(args.entities, sep="	", dtype=str, keep_default_na=False)
    query = target[[ENTITY, GROUP, "cell_line_id"]].merge(entities[[ENTITY, "smiles"]],
                                                         on=ENTITY, validate="many_to_one")
    parents = {s: structural_keys(clean(s))[1] for s in set(query.smiles)}
    query["parent"] = query.smiles.map(parents)
    if len(query) != len(target) or query.parent.eq("").any():
        raise ValueError("A Tahoe target cannot be connected by canonical structure")
    return query, dict(registry_sha256=sha256(args.entities), registry_receipt_sha256=sha256(reg),
                       stage_sha256=sha256(cache/"receipt.json"))


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    guard = read_json(args.support_reference)
    if (not guard["metadata_only"] or guard["expression_read"]
            or sha256(args.r18_access/"receipt.json") != guard["reservation_access_sha256"]):
        raise ValueError("Current bridges must share the exact T4 nonreserved source contract")
    target, lineage = target_scope(args)
    protected = pd.concat([provenance(r) for r in read_r18_domains(args)[:4]], ignore_index=True)
    parts = []
    prior = args.base/"vcell_liver_r15_lincs/v1"
    for study in STUDIES:
        sources = guard["source_metadata_sha256"][study]
        if any(sha256(p) != h for p, h in sources.items()):
            raise ValueError("A T4 source metadata input changed")
        inst, sig = geo_metadata(study, prior)
        coordinate = "RNA" if {"rna_plate", "rna_well"} <= set(inst) else "DET_ONLY_UNVERIFIED_RNA"
        inst = prepare(inst)
        wanted = inst.pert_type.eq("trt_cp") & inst.hours.isin(TIME_PAIR)
        dose = measured_dose(asset(ROOT, study, "inst_info"), set(inst.loc[wanted, "inst_id"]))
        eligible, _ = eligible_links(inst, sig, protected, dose)
        exact = eligible.loc[eligible.actual_dose_uM.eq("5")]
        pairs = matched_pairs(exact)
        for accession, design in target.groupby("cell_line_id"):
            target_groups = set(design.parent)
            for context in ("HEPG2", "SNU423", "HUH7", "PHH"):
                one = exact.loc[exact.cell_id.str.replace("-", "", regex=False).eq(context)]
                paired = pairs.loc[pairs.cell_id.str.replace("-", "", regex=False).eq(context)]
                disjoint = paired.loc[paired.cross_time_plate_disjoint]
                repeated = disjoint.loc[disjoint.both_times_2plates]
                common6, common24 = (set(one.loc[one.hours.eq(time), "parent"]) for time in TIME_PAIR)
                parts.append(dict(source=study, coordinate_axis=coordinate, source_cell=context,
                    target=LIVER[accession], target_cellosaurus_id=accession,
                    actual_dose_uM="5", target_hours=24, target_groups=design[GROUP].nunique(),
                    target_canonical_parents=len(target_groups),
                    source6_parents=len(common6), source24_parents=len(common24),
                    shared6=len(common6 & target_groups), shared24=len(common24 & target_groups),
                    both_times_disjoint_parents=disjoint.parent.nunique(),
                    shared_disjoint=len(set(disjoint.parent) & target_groups),
                    shared_two_plates_each_time=len(set(repeated.parent) & target_groups)))
    genes = read_json(args.gene_reference)["files"]["genes.tsv"]
    args.out.mkdir(parents=True)
    pd.DataFrame(parts).to_csv(args.out/"support.tsv", sep="\t", index=False)
    write_json(args.out/"receipt.json", dict(metadata_only=True, expression_read=False, dose_uM=5,
        min_target_cells=20, target_pool_min_parents=3, input=lineage, candidate_landmark_axis_sha256=genes,
        upstream_t4_sha256=sha256(args.support_reference), code_sha256=sha256(__file__),
        files={"support.tsv": sha256(args.out/"support.tsv")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index",
                 "support-reference", "tahoe", "tahoe-reference", "entities", "gene-reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
