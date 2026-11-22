"""Register the DEV I-track population and audit independent feature-view coverage."""

import argparse
from collections import defaultdict
from pathlib import Path
import re

import pandas as pd
from rdkit import Chem

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import clean, structural_keys
from src.r15_lincs.data import table
from src.r15_lincs.design import fold_id
from src.r17_lincs.inventory import in_footprint, register_footprint
from src.r17_lincs.vehicles import GEO, eligibility
from src.r19_plan.contracts import assert_dev_allowed
from src.r19_plan.r18_preflight import checked_tables

BRD = re.compile(r"(BRD-[A-Z][0-9]{8})(?:-.*)?$", re.IGNORECASE)
CAP_BYTES = 5_000_000_000


def structure(value):
    return structural_keys(clean(value))[1]


def base_brd(value):
    matched = BRD.fullmatch(clean(value))
    return matched.group(1).upper() if matched else ""


def append_to(mapping, key, value):
    if key and value:
        mapping[key].add(value)


def unique_values(mapping):
    return {key: next(iter(values)) for key, values in mapping.items() if len(values) == 1}


def registered_rows(base, registered, registered_reference, role_reference):
    checked_tables(registered, registered_reference)
    reserved = table(registered / "r18_reserved_manifest.tsv")
    exceptions = table(registered / "r18_B_donor_exceptions.tsv")
    blocked = pd.concat([reserved, exceptions], ignore_index=True)
    prior = base / "vcell_liver_r15_lincs/v1/cache"
    controls = base / "vcell_liver_r17_lincs/v1/support/assignments.tsv"
    if sha256(controls) != read_json(role_reference)["files"]["assignments.tsv"]:
        raise ValueError("Changed historical R17 physical role axis")
    eligible, _ = eligibility(prior)
    work = eligible.loc[eligible.rna_plate.isin(table(controls).rna_plate)].copy()
    work[["det_plate", "det_well"]] = ""
    previous = len(work)
    sigs = set(blocked.sig_id.str.strip().str.upper()) - {""}
    overlap = in_footprint(work, register_footprint(blocked)) | work.sig_id.str.strip().str.upper().isin(sigs)
    work = work.loc[~overlap].copy()
    assert_dev_allowed(work, reserved, exceptions)
    if work.inst_id.duplicated().any() or work.duplicated(["rna_plate", "rna_well"]).any():
        raise ValueError("The DEV population double counts a physical measurement")
    work["outer_fold"] = work.parent.map(lambda parent: fold_id(parent, "R15-outer", 3))
    held_parents = set(blocked.parent) - {""}
    work["r18_identity_overlap"] = work.parent.isin(held_parents)
    return work, dict(starting_wells=previous, excluded_assay_wells=int(overlap.sum()),
        retained_wells=len(work), retained_parents=work.parent.nunique(),
        r18_identity_overlap=int(work.loc[work.r18_identity_overlap, "parent"].nunique()),
        expected_counts=[int(work.loc[work.outer_fold.eq(fold), "parent"].nunique()) for fold in range(3)],
        reserved_sha256=sha256(registered / "r18_reserved_manifest.tsv"),
        source_exception_sha256=sha256(registered / "r18_B_donor_exceptions.tsv"),
        development_source_sha256=sha256(prior / GEO / "target_rows.tsv"),
        R17_assignment_sha256=sha256(controls), format="R17 physical A support with R18 assay exclusions")


def read_catalog(path, ids):
    rows = table(path, usecols=["pert_id", "canonical_smiles", "inchi_key"])
    rows["brd"] = rows.pert_id.map(base_brd)
    rows = rows.loc[rows.brd.isin(ids)].copy()
    rows["parent"] = rows.canonical_smiles.map(structure)
    return rows


def build_crosswalk(samples, legacy, catalog, population):
    # Historical BRD aliases are accepted only through unique connectivity.
    by_brd = defaultdict(set)
    by_connectivity = defaultdict(set)
    for row in catalog.itertuples(index=False):
        append_to(by_brd, row.brd, row.parent)
        append_to(by_connectivity, clean(row.inchi_key).split("-")[0].upper(), row.parent)
    for row in population[["pert_id", "parent"]].drop_duplicates().itertuples(index=False):
        append_to(by_brd, base_brd(row.pert_id), row.parent)
    for parent in population.parent.unique():
        molecule = Chem.MolFromSmiles(parent)
        if molecule is None:
            raise ValueError("Invalid canonical parent in frozen LINCS registry")
        append_to(by_connectivity, Chem.MolToInchiKey(molecule).split("-")[0], parent)

    versioned = [name for name in legacy.columns if "broad_id" in name]
    aliases = defaultdict(set)
    for record in legacy[["InChIKey14", *versioned]].itertuples(index=False, name=None):
        for value in record[1:]:
            for token in re.split(r"[;,|\s]+", clean(value)):
                append_to(aliases, base_brd(token), record[0])
    records = []
    for sample in sorted(set(samples)):
        brd = base_brd(sample)
        direct = by_brd[brd]
        via_alias = set().union(*(by_connectivity[key] for key in aliases[brd]))
        if len(direct) == 1 and not (via_alias and direct.isdisjoint(via_alias)):
            parent, method = next(iter(direct)), "BRD_structure"
        elif not direct and len(via_alias) == 1:
            parent, method = next(iter(via_alias)), "unique_historical_BRD_connectivity"
        else:
            parent, method = "", "ambiguous_or_unresolved"
        records.append(dict(source_id=sample, parent=parent, match=method))
    return pd.DataFrame(records)


def checked_sources(root):
    manifest = read_json(root / "source_manifest.json")
    if sum(row["size"] for row in manifest) >= CAP_BYTES:
        raise ValueError("The approved I-track collection would exceed 5 GB")
    for row in manifest:
        path = root / row["path"]
        if path.stat().st_size != row["size"] or sha256(path) != row["sha256"] or not row["license"]:
            raise ValueError(f"Unverified source artifact: {row['path']}")
    return manifest


def write_receipt(out, fields):
    write_json(out / "receipt.json", dict(fields, files={
        path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()
    }))


def audit_views(population, views):
    rows = []
    parents = population.drop_duplicates("parent").copy()
    parents["outer_fold"] = pd.to_numeric(parents.outer_fold)
    parents["r18_identity_overlap"] = parents.r18_identity_overlap.astype(str).eq("True")
    min_total, min_fold = 100, 20
    for name, has_view in views.items():
        for scope, population_scope in (
                ("R17_DEV_assay_excluded", parents),
                ("exclude_shared_R18_parent_sensitivity", parents.loc[~parents.r18_identity_overlap])):
            kept = population_scope.loc[population_scope.parent.isin(has_view)]
            fold_counts = [int(kept.outer_fold.eq(fold).sum()) for fold in range(3)]
            eligible = len(kept) >= min_total and min(fold_counts) >= min_fold
            rows.append(dict(view=name, scope=scope, matched=len(kept), canonical_parents=len(population_scope),
                match_pct=round(100 * len(kept) / len(population_scope), 3) if len(population_scope) else 0.0,
                fold0=fold_counts[0],
                fold1=fold_counts[1], fold2=fold_counts[2], coverage_pass=eligible))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("base", "registered", "registered-reference", "role-reference", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    rows, meta = registered_rows(args.base, args.registered, args.registered_reference, args.role_reference)
    args.out.mkdir(exist_ok=False, parents=True)
    cols = ["inst_id", "sig_id", "parent", "pert_id", "cell_id", "hours", "dose", "rna_plate", "rna_well",
            "det_plate", "det_well", "outer_fold", "r18_identity_overlap"]
    rows[cols].to_csv(args.out / "rows.tsv", sep="\t", index=False)
    write_receipt(args.out, meta)


if __name__ == "__main__":
    main()
