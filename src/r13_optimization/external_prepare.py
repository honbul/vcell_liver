"""Freeze GSE306429 assay, chemistry and gene alignment without reading expression."""

import argparse
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE306nnn/GSE306429/suppl/"
    "GSE306429_combined_pseudobulk.h5ad"
)
SOURCE_SHA256 = "ca113616caae7619c12b9acb0ed47c8abcd0867c3932f02b6316a83db32d2644"
ASSAY_LINES = {
    "A549": {"cellosaurus_id": "CVCL_0023", "tahoe_name": "A549", "relation": "same_named_line"},
    "HepG2": {"cellosaurus_id": "CVCL_0027", "tahoe_name": "HepG2/C3A", "relation": "parent_to_clone"},
}


def digest(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_metadata(path):
    """Read axis labels and inspect array descriptors. The assay arrays stay sealed."""
    with h5py.File(path, "r") as file:
        obs = {}
        for name, node in file["obs"].items():
            if isinstance(node, h5py.Group):
                categories = node["categories"].asstr()[()]
                obs[name] = [categories[code] if code >= 0 else None for code in node["codes"][()]]
            else:
                obs[name] = node.asstr()[()] if h5py.check_string_dtype(node.dtype) else node[()]
        var = file["var/_index"].asstr()[()]
        arrays = {
            key: {"shape": list(file[key].shape), "dtype": str(file[key].dtype)}
            for key in ("X", "layers/counts")
        }
        uns_keys = list(file["uns"])
    observations = pd.DataFrame(obs).rename(columns={"_index": "obs_id"})
    observations["obs_row"] = np.arange(len(observations))
    return observations, var, arrays, uns_keys


def name_key(name):
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def compound_labels(row):
    return {
        name_key(value)
        for value in (row.pert_id, row.cmap_name, row.compound_aliases)
        if value
    }


@lru_cache(maxsize=None)
def parent_key(smiles):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return ""
    try:
        parent = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
    except (RuntimeError, ValueError):
        return ""
    Chem.RemoveStereochemistry(parent)
    return Chem.MolToSmiles(parent, canonical=True)


def indexed_names(names, cmap, tahoe):
    """Require a unique standard parent across exact normalized name and alias hits."""
    keys = {name_key(name): name for name in names}
    hits = defaultdict(set)
    for row in cmap.itertuples(index=False):
        for key in compound_labels(row) & keys.keys():
            hits[key].add((row.canonical_smiles, "cmap2020", row.pert_id))
    for row in tahoe.itertuples(index=False):
        key = name_key(row.drug)
        if key in keys:
            hits[key].add((row.canonical_smiles, "tahoe", row.drug))

    resolved = []
    for name in sorted(names):
        candidate = hits[name_key(name)]
        parents = {parent_key(smiles) for smiles, _, _ in candidate if smiles} - {""}
        status = "resolved" if len(parents) == 1 else "ambiguous" if parents else "unresolved"
        resolved.append({
            "compound_name": name,
            "compound_parent": next(iter(parents)) if status == "resolved" else "",
            "identity_status": status,
            "candidate_parents": "|".join(sorted(parents)),
            "reference_ids": "|".join(sorted({f"{source}/{code}" for _, source, code in candidate})),
        })
    return pd.DataFrame(resolved)


def candidate_parents(identity):
    return {parent for entry in identity.candidate_parents for parent in entry.split("|") if parent}


def expand_reference_aliases(identity, cmap, allowed_parents):
    reference_ids = {
        code.removeprefix("cmap2020/")
        for references in identity.reference_ids.unique()
        for code in references.split("|")
        if code.startswith("cmap2020/")
    }
    aliases = {name_key(name) for name in identity.compound_name}
    candidates = cmap.loc[cmap.pert_id.isin(reference_ids)]
    for row in candidates.itertuples(index=False):
        if parent_key(row.canonical_smiles) in allowed_parents:
            aliases.update(compound_labels(row))
    return aliases


def align_genes(var, tahoe):
    external = pd.DataFrame({"gene_symbol": var, "external_col": np.arange(len(var))})
    external = external.loc[~external.gene_symbol.duplicated(keep=False)]
    unique = tahoe.loc[~tahoe.gene_symbol.duplicated(keep=False) & ~tahoe.ensembl_id.duplicated(keep=False)]
    return external.merge(unique, on="gene_symbol", how="inner", validate="one_to_one")


def held_in_groups(rows, auxiliary, tahoe, test_parents, test_names):
    if tahoe.drug.duplicated().any() or tahoe.drug.eq("").any():
        raise ValueError("Duplicate or blank source compound label")
    groups = rows[["compound_group_id"]].drop_duplicates().sort_values("compound_group_id").copy()
    tahoe = tahoe.assign(parent=tahoe.canonical_smiles.map(parent_key), label=tahoe.drug.map(name_key))
    raw = pd.concat([rows, auxiliary], ignore_index=True)[["compound_entity_id", "compound_group_id", "smiles"]]
    raw = raw.drop_duplicates()
    raw["parent"] = raw.smiles.map(parent_key)
    if raw.parent.eq("").any():
        raise ValueError("An observed Tahoe compound lacks a usable chemical parent")
    raw["label"] = raw.compound_entity_id.map(name_key)

    test_parents, test_names = set(test_parents), set(test_names)
    test_cids, excluded_groups = set(), set()
    while True:
        size = (len(test_parents), len(test_names), len(test_cids), len(excluded_groups))
        found = tahoe.parent.isin(test_parents) | tahoe.label.isin(test_names) | tahoe.pubchem_cid.isin(test_cids)
        test_names.update(tahoe.loc[found, "label"])
        test_parents.update(set(tahoe.loc[found, "parent"]) - {""})
        test_cids.update(set(tahoe.loc[found, "pubchem_cid"]) - {""})

        observed = raw.parent.isin(test_parents) | raw.label.isin(test_names) | raw.compound_group_id.isin(excluded_groups)
        excluded_groups.update(raw.loc[observed, "compound_group_id"])
        test_parents.update(raw.loc[observed, "parent"])
        test_names.update(raw.loc[observed, "label"])
        if size == (len(test_parents), len(test_names), len(test_cids), len(excluded_groups)):
            break

    groups["eligible_for_external_model_fit"] = ~groups.compound_group_id.isin(excluded_groups)
    held_in = raw.loc[raw.compound_group_id.isin(groups.loc[groups.eligible_for_external_model_fit, "compound_group_id"])]
    if set(held_in.parent) & test_parents or set(held_in.label) & test_names:
        raise ValueError("A confirmed or potential external alias remains in Tahoe training metadata")
    public_exclusions = tahoe.loc[found, ["drug", "pubchem_cid", "parent"]]
    raw_exclusions = raw.loc[observed, ["compound_entity_id", "parent"]].rename(columns={"compound_entity_id": "drug"})
    raw_exclusions["pubchem_cid"] = raw_exclusions.drug.map(tahoe.set_index("drug").pubchem_cid).fillna("")
    exclusions = pd.concat([public_exclusions, raw_exclusions], ignore_index=True).drop_duplicates()
    return groups, exclusions.sort_values(["drug", "parent", "pubchem_cid"]), len(raw), len(held_in)


def assay_rows(obs, identity):
    selected = obs.loc[obs.cell_id.isin(ASSAY_LINES) & obs.timepoint_hr.eq(24)].copy()
    selected["control_role"] = selected.replicate.map({"1": "A", "2": "B"})
    selected["physical_well_id"] = selected.plate + "/" + selected.well
    if selected.control_role.isna().any() or not selected.physical_well_id.is_unique:
        raise ValueError("Missing replicate roles or nonunique physical wells")
    library = ["cell_id", "replicate", "plate", "library_id", "timepoint_hr"]
    controls = selected.loc[selected.compound_name.eq("DMSO")].copy()
    vehicle = controls[library + ["obs_row", "physical_well_id"]].rename(
        columns={"obs_row": "control_obs_row", "physical_well_id": "control_well_id"}
    )
    treatment = selected.loc[~selected.compound_name.eq("DMSO")].merge(
        vehicle, on=library, how="left", validate="many_to_one"
    )
    treatment = treatment.merge(identity, on="compound_name", how="left", validate="many_to_one")
    valid = treatment.identity_status.eq("resolved") & treatment.control_obs_row.notna() & treatment.dose_uM.gt(0)
    treatment["library_control_present"] = treatment.control_obs_row.notna()
    treatment["eligible_paired_roles"] = False
    pairs = ["cell_id", "compound_parent", "dose_uM", "timepoint_hr"]
    paired = treatment.loc[valid].groupby(pairs).control_role.transform("nunique").eq(2)
    treatment.loc[paired.index, "eligible_paired_roles"] = paired
    treatment["cellosaurus_id"] = treatment.cell_id.map({key: value["cellosaurus_id"] for key, value in ASSAY_LINES.items()})
    return treatment, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--tahoe-drugs", type=Path, required=True)
    parser.add_argument("--tahoe-genes", type=Path, required=True)
    parser.add_argument("--tahoe-fit-rows", type=Path, nargs="+", required=True)
    parser.add_argument("--tahoe-aux-rows", type=Path, nargs="+", required=True)
    parser.add_argument("--cmap-compounds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_hash = digest(args.h5ad)
    if raw_hash != SOURCE_SHA256:
        raise ValueError("Unexpected GSE306429 download checksum")

    obs, var, arrays, uns_keys = read_metadata(args.h5ad)
    cmap = pd.read_csv(args.cmap_compounds, sep="\t", dtype=str, keep_default_na=False)
    tahoe = pd.read_csv(args.tahoe_drugs, sep="\t", dtype=str, keep_default_na=False)
    genes = pd.read_csv(args.tahoe_genes, sep="\t").dropna(subset=["gene_symbol", "ensembl_id"])
    rows = pd.concat([pd.read_csv(path, sep="\t") for path in args.tahoe_fit_rows], ignore_index=True)
    auxiliary = pd.concat([pd.read_csv(path, sep="\t") for path in args.tahoe_aux_rows], ignore_index=True)
    rows = rows.loc[rows.compound_group_id.ge(0)].assign(smiles=lambda x: x.smiles.fillna(""))
    auxiliary = auxiliary.loc[auxiliary.compound_group_id.ge(0)].assign(smiles=lambda x: x.smiles.fillna(""))
    names = set(obs.loc[obs.cell_id.isin(ASSAY_LINES), "compound_name"]) - {"DMSO"}
    identity = indexed_names(names, cmap, tahoe)
    treatment, controls = assay_rows(obs, identity)
    eligible = treatment.loc[treatment.eligible_paired_roles]
    test_parents = candidate_parents(identity)
    test_names = expand_reference_aliases(identity, cmap, test_parents)
    held_in, excluded, aliases_checked, aliases_remaining = held_in_groups(rows, auxiliary, tahoe, test_parents, test_names)
    mapped_genes = align_genes(var, genes)
    support = (
        treatment.groupby(["cell_id", "control_role", "dose_uM"], dropna=False)
        .agg(treatment_rows=("obs_row", "size"), paired_resolved_rows=("eligible_paired_roles", "sum"))
        .reset_index()
    )
    args.out.mkdir(parents=True, exist_ok=False)
    tables = dict(treatment_rows=treatment, control_rows=controls, compound_identity=identity,
                  held_in_groups=held_in, excluded_tahoe_compounds=excluded, genes=mapped_genes, dose_support=support)
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.tsv", sep="\t", index=False)
    inputs = [args.tahoe_drugs, args.tahoe_genes, args.cmap_compounds, *args.tahoe_fit_rows, *args.tahoe_aux_rows]
    receipt = {
        "dataset": "GSE306429",
        "download": {
            "path": str(args.h5ad.resolve()), "url": SOURCE_URL,
            "sha256": raw_hash, "bytes": args.h5ad.stat().st_size,
        },
        "other_input_sha256": {str(path.resolve()): digest(path) for path in inputs},
        "output_sha256": {name: digest(args.out / f"{name}.tsv") for name in tables},
        "matrix_descriptors_only": arrays,
        "uns_keys": uns_keys,
        "assay_lines": ASSAY_LINES,
        "mappable_genes": len(mapped_genes),
        "mappable_gene_rule": "Same exact unique symbol to unique Tahoe Ensembl ID; fit panel must intersect this map",
        "control_rule": "Distinct physical replicate plates A/B with exact library- and plate-matched DMSO within each role",
        "chemistry_rule": ("Score only uniquely resolved compounds. Exclude all Tahoe training group IDs linked "
                           "to any external compound name, including ambiguous candidate structures. Close the exclusion "
                           "over parent, name, CID and matching LINCS aliases using actual Tahoe primary and auxiliary SMILES."),
        "external_candidate_parents": len(test_parents),
        "tahoe_distinct_group_name_smiles_checked": aliases_checked,
        "tahoe_distinct_group_name_smiles_remaining": aliases_remaining,
        "source_group_exclusions": int((~held_in.eligible_for_external_model_fit).sum()),
        "source_groups_remaining": int(held_in.eligible_for_external_model_fit.sum()),
        "assay_treatment_rows": len(treatment),
        "assay_exact_dose_rows": int(treatment.dose_uM.eq(0.5).sum()),
        "paired_resolved_treatment_rows": len(eligible),
        "paired_resolved_compounds": eligible.compound_parent.nunique(),
        "treated_expression_accessed": False,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
