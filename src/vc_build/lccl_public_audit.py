"""Audit public LCCL identities and drug structures without parsing response values."""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from .cross_source_data import MODEL_KEYS, MODEL_NAMES, parse_cellosaurus_identity, related_cellosaurus
from .function_data import alias_token
from .io import publish_json, sha256
from .v0plus_cv import training_inputs

FILE_HASHES = {
    "public/LCCLdescription.txt": "7bdf1949ef72c63207486e1243cfc16bb925ae1d8376e6bc486d4f82c1d9f549",
    "public/RNAseqFPKM.txt": "bfa155ca86432330e4754514e5978c35d5727544791798853647569609ccc9be",
    "public/Screenedcompounds.txt": "99076e6c940c69b4f0d2a50e6eb59e67a7b6ade3fc950695aaaf177d665d6021",
    "score_only/AUC.txt": "a035053bb976763e36a3b1f05a9e22a8790c9d870d5741fa83afe8240432c71f",
}
# The initial header inspection treated CR-delimited files as LF-delimited.
EXPOSED = {"Huh7", "SNU739", "Hep3B", "Huh6", "HepG2"}


def header(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream, delimiter="\t"))


def manufacturer(name):
    token = alias_token(name)
    return "SELLECK" if token.startswith("SELLECK") else token


def products(path, measured_headers, limore, trained):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))[2:]
    registry = limore[limore.compound_id.isin(trained)].copy()
    lookup = []
    for row in rows:
        label, supplier, product = row[0].strip(), row[-2], row[-1].strip()
        if label not in measured_headers or not product:
            continue
        candidates = registry[
            registry.vendor_code.str.upper().str.strip().eq(product.upper())
            | registry.catalog_code.str.upper().str.strip().eq(product.upper())
        ]
        group = manufacturer(supplier)
        matches = candidates[candidates.vendor.map(manufacturer).eq(group)]
        ids = matches.compound_id.unique()
        if group and len(ids) == 1:
            lookup.append({"drug": label, "supplier": supplier, "product": product, "compound_id": ids[0]})
    return lookup, len(rows)


def audit(args):
    for name, digest in FILE_HASHES.items():
        if sha256(args.lccl / name) != digest:
            raise ValueError(f"Unexpected public LCCL file {name}")
    train, train_rna, receipt = training_inputs(args.bundle)
    files = receipt["inputs"]
    paths = {name: Path(files[name]["path"]) for name in ("depmap_models", "cellosaurus")}
    for name, source in paths.items():
        if sha256(source) != files[name]["sha256"]:
            raise ValueError(f"Training registry identity source changed: {name}")
    model = pd.read_csv(paths["depmap_models"], dtype=str).fillna("")
    db = parse_cellosaurus_identity(paths["cellosaurus"])
    limore = pd.read_csv(args.limore / "cell_registry.tsv", sep="\t", dtype=str).fillna("")
    query = pd.read_csv(args.lccl / "public/LCCLdescription.txt", sep="\t", dtype=str, usecols=[
        "Cell line name", "Synonyms", "Cellosaurus ID", "Tumor type", "Age",
    ]).dropna(subset=["Cell line name"]).fillna("")
    expression_path = args.lccl / "public/RNAseqFPKM.txt"
    expression_ids = {alias_token(name): name for name in header(expression_path)[2:]}
    response_path = args.lccl / "score_only/AUC.txt"
    response_ids = set(pd.read_csv(response_path, sep="\t", usecols=["Cell line"])["Cell line"].map(alias_token))
    known = model[model.ModelID.isin(train.cell_id)]
    names = set(map(alias_token, train.source_cell_label))
    names.update(set().union(*(set(map(alias_token, known[key])) for key in MODEL_NAMES)))
    identities = set().union(*(set(known[key]) for key in MODEL_KEYS)) - {""}
    names.discard("")
    limore_ids = (set(limore.cellosaurus_id) | set(limore.root_ancestor) | set(limore.depmap_model_id)
                  | set(limore.depmap_patient_id) | {j.strip() for text in limore.same_individual
                                                    for j in text.split(",")}) - {""}
    limore_names = (set(map(alias_token, limore.source_name)) | set(map(alias_token, limore.cellosaurus_name))) - {""}
    exposed_names = set(map(alias_token, EXPOSED))
    eligible = []
    for name, alias, accession, histology, age in query.itertuples(index=False, name=None):
        token = alias_token(name)
        if histology != "Hepatocellular Carcinoma" or not age.isnumeric() or int(age) < 18:
            continue
        family, references, aliases = related_cellosaurus({accession}, *db)
        tokens = (set(map(alias_token, aliases)) | {token, alias_token(alias)}) - {""}
        references |= family
        related = model[model.RRID.isin(family) | model.ModelID.isin(references)]
        patients = set(related.PatientID) - {""}
        relatives = model[model.PatientID.isin(patients) | model.ModelID.isin(related.ModelID)]
        references.update(set().union(*(set(relatives[key]) for key in MODEL_KEYS)) - {""})
        eligible.append({
            "cell": name, "accession": accession, "family": min(family),
            "matched_rna_auc_identity": token in expression_ids and token in response_ids,
            "shares_training_lineage": bool(references & identities or tokens & names),
            "shares_limore_lineage": bool(references & limore_ids or tokens & limore_names),
            "initial_preview_exposed": bool(tokens & exposed_names),
        })
    metadata = pd.DataFrame(eligible)
    matched = metadata[metadata.matched_rna_auc_identity]
    train_new = matched[~matched.shares_training_lineage]
    distinct = train_new[~train_new.shares_limore_lineage]
    fresh = distinct[~distinct.initial_preview_exposed]
    symbols = pd.read_csv(expression_path, sep="\t", dtype=str, usecols=["Gene_name"]).Gene_name
    universal = set(symbols.dropna()[~symbols.dropna().duplicated(keep=False)]) & set(train_rna.columns)
    registry = pd.read_csv(args.limore / "compound_registry.tsv", sep="\t", dtype=str).fillna("")
    drugs, declared = products(args.lccl / "public/Screenedcompounds.txt", set(header(response_path)[2:]),
                               registry, set(train.compound_id))
    if len(drugs) != len({item["compound_id"] for item in drugs}):
        raise ValueError("Multiple LCCL products mapped to one training parent")
    publish_json(args.out, {
        "download_sha256": FILE_HASHES, "train_receipt_sha256": sha256(args.bundle / "receipt.json"),
        "limore_registry_sha256": sha256(args.limore / "cell_registry.tsv"),
        "sample_counts": {
            "catalogue": len(query), "known_adult_hcc": len(metadata),
            "matched_adult_hcc": len(matched), "matched_adult_families": matched.family.nunique(),
            "source_new_models": len(train_new), "source_new_families": train_new.family.nunique(),
            "also_limore_new_models": len(distinct), "also_limore_new_families": distinct.family.nunique(),
            "fresh_families_after_preview_exclusion": fresh.family.nunique(),
        },
        "compounds": {"declared_single_agents": declared, "exact_same_supplier_catalogue": drugs},
        "common_unique_gene_symbols": len(universal), "exposed_preview_ids": sorted(EXPOSED),
        "adult_cell_identity_decisions": eligible,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("lccl", "limore", "bundle", "out"):
        parser.add_argument("--" + option, required=True, type=Path)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
