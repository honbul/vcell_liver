"""Prepare a metadata-only adult HCC pilot slate with same-dose nonhepatic donors."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from rdkit import rdBase

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r17_lincs.inventory import ROOT, ids, in_footprint, liver_cells, register_footprint
from src.r19_plan.hcc_lincs_24h_support import forbidden_signatures, instance_rows, supported_controls
from src.r19_plan.hcc_pilot_chemistry import PHASES
from src.r19_plan.i_audit import base_brd
from src.r19_plan.i_preflight import provenance, read_pinned, read_r18_domains, verify_domain


PROTECTED_CHEMISTRY = {"Tahoe100M", "EmeraldBay", "MIX-seq", "R18_reserved", "R18_donor"}


def identified_registry(path, matching):
    rows = table(path, usecols=["pert_id", "canonical_smiles"]).drop_duplicates()
    rows["brd"] = rows.pert_id.map(base_brd)
    rows = rows.loc[rows.brd.isin(matching.brd)].copy()
    with rdBase.BlockLogs():
        cache = {smiles: canonical_parent(smiles) for smiles in rows.canonical_smiles.unique() if smiles}
    rows["isomeric_parent"] = rows.canonical_smiles.map({smiles: pair[0] for smiles, pair in cache.items() if pair})
    linked = rows.merge(matching[["brd", "isomeric_parent", "parent"]], on=["brd", "isomeric_parent"], validate="many_to_one")
    unique = linked[["pert_id", "parent"]].drop_duplicates()
    if unique.pert_id.duplicated().any():
        raise ValueError("The strict identity mapping assigned two parents to a Broad ID")
    return unique


def run(args):
    paper = args.reports/"r19_plan"/"data"
    old = paper/"hcc_pilot_chemistry"/"audit"/"receipt.json"
    initial = read_json(old)
    if sha256(args.old/"receipt.json") != sha256(old):
        raise ValueError("The original structure-file pilot inventory changed")
    rows = read_pinned(args.old, "candidate_audit.tsv", initial["files"]["candidate_audit.tsv"])
    assigned = read_pinned(args.old, "draft_slots.tsv", initial["files"]["draft_slots.tsv"])
    rows = rows.assign(brd=rows.brd_id.map(base_brd))
    if rows.parent.duplicated().any() or len(assigned)!=96:
        raise ValueError("The original chemical registry does not contain distinct parents")

    known = args.jhh_audit
    pin = paper/"hcc_lincs24"/"audit"/"receipt.json"
    if sha256(known/"receipt.json") != sha256(pin):
        raise ValueError("The prior adult JHH evaluations differ from the pinned audit")
    old_targets = read_pinned(known, "targets.tsv", read_json(pin)["files"]["targets.tsv"])
    protected = read_r18_domains(args)
    reserved = pd.concat([provenance(frame) for frame in protected], ignore_index=True)
    blocked = register_footprint(reserved)
    b1 = args.base/"vcell_liver_transigen_adaptation_b1"/"assets"
    catalog = ROOT/"LINCS2020"
    files = dict(compounds=catalog/"compoundinfo_beta.txt", cells=catalog/"cellinfo_beta.txt",
        signatures=catalog/"siginfo_beta.txt", instances=b1/"raw/cmap2020/instinfo_beta.txt",
        B1=b1/"processed/cmap2020_v1/rows.tsv")
    if sha256(files["B1"]) != read_json(files["B1"].parent/"build_manifest.json")["files"]["rows.tsv"]["sha256"]:
        raise ValueError("B1 adult target exposure inventory changed")
    aliases = identified_registry(files["compounds"], rows)
    prior = table(files["B1"], usecols=["role", "pert_id", "cell"])
    normalized_cells = table(files["cells"], usecols=["cell_iname","cell_lineage"])
    hepatic = set(liver_cells(files["cells"], cmap=True).cell)
    lineage = normalized_cells.cell_lineage.str.strip().str.lower()
    nonhepatic = set(normalized_cells.loc[~lineage.isin(("", "unknown", "na")), "cell_iname"]) - hepatic
    treated, dmsos, _ = instance_rows(files["instances"], nonhepatic)
    old_ids = forbidden_signatures(files["signatures"], reserved)
    treated = treated.loc[~in_footprint(treated, blocked) & ~ids(treated.inst_id).isin(old_ids)]
    dmsos = dmsos.loc[~in_footprint(dmsos, blocked) & ~ids(dmsos.inst_id).isin(old_ids)]
    treated = treated.merge(aliases, on="pert_id", validate="many_to_one")
    treated, dmsos = supported_controls(treated, dmsos)
    verify_domain(treated, dmsos, *protected[:4])
    support = treated.groupby("parent").agg(cells=("cell_id","nunique"),plates=("rna_plate","nunique"),wells=("inst_id","size"))
    prior_adult = set(prior.loc[prior.role.eq("target") & prior.cell.isin(hepatic),"pert_id"])
    exposed = set(aliases.loc[aliases.pert_id.isin(prior_adult),"parent"]) | set(old_targets.parent)
    recorded = rows.exclusions.map(lambda s: bool(set(s.split("|")) & PROTECTED_CHEMISTRY))
    denied = (set(protected[0].parent) | set(protected[1].parent)) - {""}
    frame = rows.assign(prior_adult=rows.parent.isin(exposed), reserved_chemistry=rows.parent.isin(denied) | recorded,
        original_5uM_draft=rows.parent.isin(assigned.parent)).merge(support, on="parent", how="left", validate="one_to_one")
    for col in ("cells","plates","wells"):
        frame[col] = frame[col].fillna(0).astype(int)
    frame["scaffold_unit"] = frame.scaffold.where(frame.scaffold.ne(""), "acyclic:"+frame.parent)
    eligible_phase = frame.phase.isin(PHASES)
    scientific = ~frame.prior_adult & ~frame.reserved_chemistry & eligible_phase
    covered = frame.cells.ge(16) & frame.plates.ge(16)
    eligible = frame.loc[scientific & covered].copy()
    eligible["order"] = eligible.parent.map(lambda s: hashlib.sha256(("HCC-10uM-20261122|"+s).encode()).hexdigest())
    chosen = eligible.sort_values("order").drop_duplicates("scaffold_unit").head(96)
    chosen = chosen.assign(proposed_slot=[f"B{i:03}" for i in range(1,len(chosen)+1)])
    counts = []
    for name, mask in (("original_5uM_draft",frame.original_5uM_draft), ("all_candidates",frame.parent.ne("")),
                      ("known_phase",eligible_phase), ("after_prior_adult_and_R18_exclusion",scientific),
                      ("planned_10uM",scientific & covered), ("selected",frame.parent.isin(chosen.parent))):
        block = frame.loc[mask]
        counts.append(dict(scope=name, parents=len(block), with_any_donor=int(block.cells.gt(0).sum()),
            with_16_cells_16_plates=int((block.cells.ge(16)&block.plates.ge(16)).sum()),
            distinct_scaffolds=block.scaffold_unit.nunique()))

    if args.out.exists():
        raise FileExistsError(args.out)
    with TemporaryDirectory(dir=args.out.parent,prefix=".hcc-10um-") as temp:
        out = Path(temp)
        chosen[["proposed_slot","parent","isomeric_parent","brd_id","drug","phase","cells","plates","wells","scaffold_unit"]].to_csv(
            out/"draft_slots.tsv",sep="\t",index=False)
        pd.DataFrame(counts).to_csv(out/"support.tsv",sep="\t",index=False)
        if args.count_grid:
            grid = []
            for count in (2, 4, 8, 16):
                eligible_at_count = frame.loc[scientific & frame.cells.ge(count) & frame.plates.ge(count)]
                grid.append(dict(minimum_cells=count, minimum_plates=count, parents=len(eligible_at_count),
                    distinct_scaffolds=eligible_at_count.scaffold_unit.nunique()))
            pd.DataFrame(grid).to_csv(out/"count_grid.tsv", sep="\t", index=False)
        write_json(out/"receipt.json",dict(metadata_only=True,expression_read=False,selected=len(chosen),measured=False,
            protocol_sha256=sha256(args.protocol),code_sha256=sha256(__file__),previous_5uM_sha256=sha256(old),
            previous_JHH_sha256=sha256(pin),reserve_access_sha256=sha256(args.r18_access/"receipt.json"),
            source_sha256={name:sha256(path) for name,path in files.items()},
            files={file.name:sha256(file) for file in out.iterdir()}))
        out.rename(args.out)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("base","reports","old","jhh-audit","r18-registered","r18-lock","r18-access","r18-index","protocol","out"):
        cli.add_argument("--"+field,type=Path,required=True)
    cli.add_argument("--count-grid", action="store_true")
    run(cli.parse_args())
