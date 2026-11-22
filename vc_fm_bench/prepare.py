"""Build label-separated, identity-audited benchmark inputs on G1."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from .artifacts import digest, file_record, new_directory, write_json


def verify_bundle(directory, parts):
    receipt = json.loads((directory / "receipt.json").read_text())
    for relative in parts:
        observed = file_record(directory / relative)
        if observed != receipt["artifacts"][relative]:
            raise ValueError(f"Source changed at {directory / relative}")
    return receipt


def cold_folds(groups, n_folds=5):
    groups = sorted(set(groups), key=lambda group: sha256(f"23019:{group}".encode()).digest())
    if len(groups) < n_folds:
        raise ValueError("Insufficient independent chemical groups")
    return {group: i % n_folds for i, group in enumerate(groups)}


def chemistry(rows):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize

    records = []
    for source_id, smiles in rows.itertuples(index=False, name=None):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Registered structure failed RDKit: {source_id}")
        molecule = rdMolStandardize.ChargeParent(mol)
        key = Chem.MolToInchiKey(molecule)
        if len(key) != 27:
            raise ValueError(f"Could not compute parent identity: {source_id}")
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        if scaffold.GetNumAtoms():
            # Generic scaffold keeps analogues with ring-atom substitutions together.
            group = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold))
        else:
            group = f"acyclic:{key.split('-')[0]}"
        records.append((source_id, key, Chem.MolToSmiles(molecule), group, key.split("-")[0]))
    return pd.DataFrame(records, columns=["source_drug", "drug_id", "smiles", "scaffold", "connectivity"])


def response_frame(rows, registry, split, label=None):
    rows = rows.rename(columns={"compound_id": "source_drug"}).merge(
        registry.drop(columns="smiles"), on="source_drug", how="left", validate="many_to_one",
    )
    columns = ["cell_id", "cell_group", "source_id", "drug_id", "scaffold", "connectivity"]
    if rows[columns].isna().any().any():
        raise ValueError(f"Missing benchmark identity in {split}")
    result = rows[columns].copy()
    result["id"] = [sha256(f"{split}:{cell}:{source}:{drug}".encode()).hexdigest()[:24]
                    for cell, source, drug in zip(result.cell_id, result.source_id, result.drug_id)]
    if result["id"].duplicated().any():
        raise ValueError(f"Duplicate standardized cell-drug-source IDs in {split}")
    if label is not None:
        result["outcome"] = rows[label].astype(float)
    return result


def load_dili(path):
    receipt_path = path.with_name("fda_receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if digest(path) != receipt["sha256"][str(path)]:
        raise ValueError("The FDA structure stage failed its recorded SHA-256")
    cols = ["fda_id", "status", "smiles", "concern", "v1_concern", "release_increment"]
    table = pd.read_csv(path, sep="\t", usecols=cols)
    mapping = table.loc[table["status"] == "resolved", ["fda_id", "smiles"]]
    if mapping.isna().any().any() or table["fda_id"].duplicated().any():
        raise ValueError("FDA structure stage contains ambiguous entries")
    registry = chemistry(mapping)
    labelled = table.rename(columns={"fda_id": "source_drug"}).merge(
        registry.drop(columns="smiles"), validate="one_to_one", on="source_drug",
    )
    return table, registry, labelled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cross", "limore", "fda", "uniprot", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    a = parser.parse_args()
    verified = ["train/rows.parquet", "train/rna.parquet", "test_inputs/rows.parquet",
                "test_inputs/rna.parquet", "audit/holdout_roster.tsv"]
    cross_receipt = verify_bundle(a.cross, verified)
    verify_bundle(a.limore, ["rows.parquet", "rna.parquet"])
    train = pd.read_parquet(a.cross / "train/rows.parquet")
    query = pd.read_parquet(a.cross / "test_inputs/rows.parquet")
    limore = pd.read_parquet(a.limore / "rows.parquet")
    if set(train.cell_group) & set(query.cell_group):
        raise ValueError("Cell-cold groups overlap")
    if set(train.cell_id) & set(query.cell_id):
        raise ValueError("Cell-cold IDs overlap")
    raw_drugs = pd.concat([f[["compound_id", "isomeric_smiles"]] for f in (train, query, limore)])
    raw_drugs = raw_drugs.drop_duplicates()
    if raw_drugs["compound_id"].duplicated().any():
        raise ValueError("Multiple structures for one registered response compound")
    drug_registry = chemistry(raw_drugs)
    raw_fda, fda_registry, dili = load_dili(a.fda)
    drug_query = pd.concat([drug_registry, fda_registry], ignore_index=True)
    if drug_query.groupby("drug_id")["smiles"].nunique().max() != 1:
        raise ValueError("Inconsistent canonical parent")
    drug_query = drug_query[["drug_id", "smiles"]].drop_duplicates().rename(columns={"drug_id": "id"})
    drug_query = drug_query.sort_values("id")

    train = response_frame(train, drug_registry, "training", "endpoint_value")
    limore = response_frame(limore, drug_registry, "limore", "endpoint_value")
    source_keys = query[["record_id"]].copy()
    query = response_frame(query, drug_registry, "cell-cold")
    if not set(query.drug_id) <= set(train.drug_id):
        raise ValueError("Cell-cold evaluation has drugs absent in training")

    proteins = pd.read_csv(a.uniprot, sep="\t", usecols=["Entry", "Gene Names (primary)", "Sequence"])
    proteins = proteins.dropna().rename(columns={"Entry": "id", "Sequence": "sequence",
                                                 "Gene Names (primary)": "gene_symbol"})
    proteins = proteins.sort_values("id")
    if proteins["id"].duplicated().any():
        raise ValueError("Duplicate accessions in UniProt reviewed sequence input")

    # Registration uses identities and scaffolds. The benchmark preparation never
    # opens the cell-cold outcome file. Its expected checksum is passed to scoring.
    with new_directory(a.output) as out:
        drug_query.to_csv(out / "molecules.tsv", sep="\t", index=False)
        proteins.to_csv(out / "proteins.tsv", sep="\t", index=False)
        train.to_parquet(out / "pan_train.parquet", index=False)
        query.to_parquet(out / "limore_cell_cold.parquet", index=False)
        source_keys.assign(id=query.id).to_parquet(out / "limore_cell_cold_keys.parquet", index=False)
        limore.drop(columns="outcome").to_parquet(out / "limore_drug_cold.parquet", index=False)
        limore[["id", "outcome"]].to_parquet(out / "limore_drug_labels.parquet", index=False)
        dili.drop(columns=["concern", "v1_concern"]).to_parquet(out / "dili_inputs.parquet", index=False)
        dili[["source_drug", "concern", "v1_concern"]].to_parquet(out / "dili_labels.parquet", index=False)
        write_json(out / "split_groups.json", {
            "limore_scaffold": cold_folds(limore.scaffold),
            "gdsc2_scaffold": cold_folds(train.loc[train.source_id.str.contains("gdsc", case=False), "scaffold"]),
            "dili_scaffold": cold_folds(dili.scaffold),
        })
        write_json(out / "receipt.json", {
            "cross_receipt": file_record(a.cross / "receipt.json"),
            "limore_receipt": file_record(a.limore / "receipt.json"),
            "fda_structure_file": file_record(a.fda),
            "uniprot": file_record(a.uniprot),
            "cell_cold_outcomes": cross_receipt["artifacts"]["test_labels/outcomes.parquet"],
            "n_cell_cold_queries": len(query), "n_pan_train": len(train),
            "n_limore_drug_cold": len(limore), "n_parent_structures": len(drug_query),
            "n_resolved_fda": len(dili), "n_fda_source": len(raw_fda),
            "n_reviewed_with_gene_proteins": len(proteins),
            "files": {p.name: file_record(p) for p in sorted(out.iterdir())},
        })


if __name__ == "__main__":
    main()
