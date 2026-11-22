"""Extract ChEMBL 37 quantitative evidence for locked LINCS parents, read-only."""

import argparse
from collections import defaultdict
import csv
import gzip
from pathlib import Path
import sqlite3

import pandas as pd
from rdkit import Chem

from src.r11_chemical_probe.common import sha256
from src.r15_lincs.data import table
from src.r17_lincs.inventory import ROOT
from src.r19_plan.i_audit import base_brd, structure, unique_values, write_receipt
from src.r19_plan.r18_preflight import checked_tables

QUERY = """
SELECT x.activity_id, x.molregno, m.chembl_id AS molecule_chembl_id,
       a.chembl_id AS assay_chembl_id, a.assay_type, a.confidence_score,
       a.assay_organism, a.assay_tissue, a.assay_cell_type, a.relationship_type, a.bao_format,
       t.chembl_id AS target_chembl_id, t.target_type, t.organism AS target_organism, a.tid,
       x.pchembl_value, x.standard_type, x.standard_relation, x.standard_units, x.standard_value,
       x.action_type, x.potential_duplicate, x.data_validity_comment, x.bao_endpoint,
       d.chembl_id AS document_chembl_id, d.doi, d.pubmed_id,
       x.src_id, s.src_short_name
FROM activities AS x INDEXED BY fk_act_molregno
JOIN assays AS a ON a.assay_id = x.assay_id
JOIN molecule_dictionary AS m ON m.molregno = x.molregno
LEFT JOIN target_dictionary AS t ON t.tid = a.tid
LEFT JOIN docs AS d ON d.doc_id = x.doc_id
LEFT JOIN source AS s ON s.src_id = x.src_id
WHERE x.molregno IN ({placeholders})
  AND x.pchembl_value IS NOT NULL AND typeof(x.pchembl_value) IN ('integer','real')
  AND x.pchembl_value > 0 AND x.standard_relation = '='
  AND COALESCE(x.data_validity_comment, '') IN ('','Manually validated')
  AND lower(COALESCE(s.src_short_name, '')) NOT LIKE '%lincs%'
  AND lower(COALESCE(a.description, '')) NOT LIKE '%l1000%'
ORDER BY x.molregno, x.activity_id
"""


def structure_keys(parents):
    lookup = defaultdict(set)
    for parent in parents:
        mol = Chem.MolFromSmiles(parent)
        if mol is None:
            raise ValueError("Invalid parent on the locked LINCS axis")
        key = Chem.MolToInchiKey(mol)
        if key:
            lookup[key.split("-")[0]].add(parent)
    return lookup


def map_molecules(db, population):
    parents = set(population.parent)
    lookup = structure_keys(parents)
    old = population[["pert_id", "parent"]].drop_duplicates().copy()
    old["brd"] = old.pert_id.map(base_brd)
    old = old.groupby("brd").parent.agg(set)
    old = unique_values(dict(old))
    cmp = table(ROOT / "LINCS2020/compoundinfo_beta.txt", usecols=["pert_id", "canonical_smiles", "inchi_key"])
    cmp["parent"] = cmp.pert_id.map(base_brd).map(old)
    for row in cmp.loc[cmp.parent.notna()].itertuples(index=False):
        if row.inchi_key and structure(row.canonical_smiles) == row.parent:
            lookup[row.inchi_key.split("-")[0].upper()].add(row.parent)

    matched = defaultdict(set)
    for key, expected in sorted(lookup.items()):
        for molregno, smiles in db.execute(
                "SELECT molregno, canonical_smiles FROM compound_structures "
                "WHERE standard_inchi_key >= ? AND standard_inchi_key < ?",
                (f"{key}-", f"{key}.")):
            normalized = structure(smiles)
            if normalized in expected:
                matched[molregno].add(normalized)

    # A ChEMBL salt child can contribute only when the parent relation is explicit
    # and the project canonicalization reproduces the same parent structure.
    roots = defaultdict(set)
    for molregno, parent in unique_values(matched).items():
        result = db.execute("SELECT parent_molregno FROM molecule_hierarchy WHERE molregno=?", (molregno,)).fetchone()
        roots[int(result[0]) if result and result[0] is not None else molregno].add(parent)
    for root, parent in unique_values(roots).items():
        for member, smiles in db.execute(
            "SELECT c.molregno,c.canonical_smiles FROM molecule_hierarchy AS h "
            "JOIN compound_structures AS c ON c.molregno=h.molregno WHERE h.parent_molregno=?", (root,)):
            if structure(smiles) == parent:
                matched[member].add(parent)
    return unique_values(matched)


def curated_actions(db, matched):
    result = defaultdict(set)
    for molregno, tid, action in db.execute(
            "SELECT molregno,tid,action_type FROM drug_mechanism WHERE action_type IS NOT NULL AND tid IS NOT NULL"):
        if molregno in matched:
            result[matched[molregno], tid].add(action)
    return result


def extract(db, matched, path):
    counts = defaultdict(lambda: {"activities": 0, "assay_types": set(), "confidence_ge7": 0})
    actions = curated_actions(db, matched)
    ids = sorted(matched)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        for at in range(0, len(ids), 90):
            batch = ids[at:at + 90]
            cursor = db.execute(QUERY.format(placeholders=",".join("?" for _ in batch)), batch)
            columns = [item[0] for item in cursor.description]
            if at == 0:
                writer.writerow(["parent", *columns, "curated_target_action_annotations"])
            for record in cursor:
                values = dict(zip(columns, record, strict=True))
                parent = matched[values["molregno"]]
                stats = counts[parent]
                stats["activities"] += 1
                stats["assay_types"].add(values["assay_type"] or "")
                stats["confidence_ge7"] += int((values["confidence_score"] or 0) >= 7)
                writer.writerow([parent, *record, "|".join(sorted(actions[parent, values["tid"]]))])
            if path.stat().st_size >= 4_000_000_000:
                raise ValueError("ChEMBL subset would exceed the I-track storage limit")
    return pd.DataFrame([dict(parent=p, activities=s["activities"], confidence_ge7=s["confidence_ge7"],
                              assay_types="|".join(sorted(s["assay_types"]))) for p, s in sorted(counts.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("database", "population", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    checked_tables(args.population, args.population / "receipt.json")
    population = table(args.population / "rows.tsv")
    db = sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        if not db.execute("SELECT 1 FROM version WHERE name='ChEMBL_37'").fetchone():
            raise ValueError("The source SQLite is not ChEMBL 37")
        db.execute("PRAGMA query_only=ON")
        matched = map_molecules(db, population)
        if not matched:
            raise ValueError("ChEMBL exact parent harmonization found no matches")
        args.out.mkdir(exist_ok=False)
        pd.DataFrame(sorted(matched.items()), columns=["molregno", "parent"]).to_csv(
            args.out / "identities.tsv", sep="\t", index=False)
        support = extract(db, matched, args.out / "bioassays.tsv.gz")
        support.to_csv(args.out / "parents.tsv", sep="\t", index=False)
    finally:
        db.close()
    print(f"{len(support)} quantitative parents; hashing preexisting read-only database", flush=True)
    attribution = args.database.parents[2] / "REQUIRED.ATTRIBUTION"
    license_path = args.database.parents[2] / "LICENSE"
    write_receipt(args.out, dict(view="I1", version="ChEMBL_37", license="CC-BY-SA-3.0",
        attribution_sha256=sha256(attribution), license_sha256=sha256(license_path),
        parent_exact_matches=len(set(matched.values())), quantitative_parents=len(support),
        activity_rows=int(support.activities.sum()), database=str(args.database),
        database_bytes=args.database.stat().st_size, database_sha256=sha256(args.database),
        downloaded_database_bytes=0, read_only_SQLite=True,
        standardization_or_imputation_fitted=False))


if __name__ == "__main__":
    main()
