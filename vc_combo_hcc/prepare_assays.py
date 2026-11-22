"""Harmonize public DrugComb, DrugCombDB and Sanger oncology combination outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .drugcombdb import STUDY, oncology_rows
from .identity import CellRegistry, CompoundRegistry
from .splits import KEYS, OUTCOMES, record, validate_assays


ONCOLOGY_STUDIES = {
    "ALMANAC", "ONEIL", "CLOUD", "FRIEDMAN", "FRIEDMAN2",
    "FLOBAK", "FORCINA", "FALLAHI-SICHANI", "NCATS_DIPG",
    "NCATS_ES(FAKI/AURKI)", "NCATS_ES(NAMPT+PARP)",
}
DRUGCOMB_FIELDS = [
    "block_id", "drug_row", "drug_col", "cell_line_name", "study_name", "tissue_name",
    "css_ri", "synergy_zip", "synergy_bliss", "synergy_hsa",
]
SANGER_FIELDS = [
    "BARCODE", "DRUGSET_ID", "cmatrix", "CELL_LINE_NAME", "COSMIC_ID", "SIDM", "TISSUE",
    "lib1_name", "lib2_name", "lib1_ID", "lib2_ID", "lib1_MaxE", "lib2_MaxE", "combo_MaxE",
    "Bliss_matrix", "Delta_MaxE_lib1", "Delta_MaxE_lib2",
]


def map_assay(rows, cells, compounds):
    known = rows.merge(cells.keys[["liver", "adult_hcc", "cell_group"]],
                       left_on="cell_id", right_index=True, how="inner", validate="many_to_one")
    known["liver"] |= known.source_liver
    for arm in ("a", "b"):
        identity = compounds.keys.rename(columns={"connectivity": f"connectivity_{arm}",
                                                  "scaffold": f"scaffold_{arm}"})[[
            f"connectivity_{arm}", f"scaffold_{arm}",
        ]]
        known = known.merge(identity, left_on=f"compound_{arm}", right_index=True, how="inner",
                            validate="many_to_one")
    known = known[known.connectivity_a.ne(known.connectivity_b)].copy()
    # The drug order is canonical across databases and assay orientations.
    reverse = known.connectivity_a.gt(known.connectivity_b)
    for field in ("compound", "connectivity", "scaffold"):
        a, b = field + "_a", field + "_b"
        known.loc[reverse, [a, b]] = known.loc[reverse, [b, a]].to_numpy()
    return known[KEYS + OUTCOMES]


def drugcomb(path, cells, compounds, audit, *, replaced=()):
    studies = ONCOLOGY_STUDIES - set(replaced)
    scans, retained = Counter(), Counter()
    raw_liver = Counter()
    for block in pd.read_csv(path, usecols=DRUGCOMB_FIELDS, chunksize=80_000,
                             na_values=["\\N", "NULL", "NA"],
                             dtype={key: str for key in DRUGCOMB_FIELDS[:6]}):
        eligible = block[["drug_row", "drug_col", "css_ri", "synergy_zip", "synergy_bliss"]].notna().all(axis=1)
        block = block[eligible].copy()
        scans.update(block.study_name.value_counts().to_dict())
        raw_liver.update(block.loc[block.tissue_name.fillna("").str.lower().eq("liver"),
                                   "study_name"].value_counts().to_dict())
        supported = block[block.study_name.isin(studies)].copy()
        if not len(supported):
            continue
        identity = cells.lookup(supported.cell_line_name)
        left, right = [compounds.lookup(supported[name]) for name in ("drug_row", "drug_col")]
        rows = pd.DataFrame({
            "id": "DrugComb1.5:" + supported.study_name + ":" + supported.block_id,
            "source": "DrugComb:" + supported.study_name,
            "source_liver": supported.tissue_name.fillna("").str.lower().eq("liver"),
            "cell_id": identity, "compound_a": left, "compound_b": right,
            "zip": supported.synergy_zip, "bliss": supported.synergy_bliss,
            "css": supported.css_ri, "added_inhibition": supported.synergy_hsa,
        })
        out = map_assay(rows, cells, compounds)
        retained.update(out.source.value_counts().to_dict())
        yield out
    audit.update({"eligible_rows_before_study_exclusions": dict(scans), "liver_tissue_native_studies": dict(raw_liver),
                  "eligible_matched_rows_by_study": dict(retained),
                  "excluded_studies": sorted(set(scans) - ONCOLOGY_STUDIES),
                  "replaced_mirrored_studies": sorted(set(replaced))})


def sanger(files, cells, compounds, audit, zip_scores):
    for path in files:
        source = "Sanger:" + path.stem.removesuffix("_matrix_results")
        data = pd.read_csv(path, usecols=SANGER_FIELDS, dtype={key: str for key in SANGER_FIELDS[:11]})
        identity = cells.lookup(data.CELL_LINE_NAME, data.SIDM, data.COSMIC_ID)
        left, right = [compounds.lookup_sanger(data[f"lib{i}_name"], data[f"lib{i}_ID"]) for i in (1, 2)]
        efficacy = data.combo_MaxE - data[["lib1_MaxE", "lib2_MaxE"]].max(axis=1, skipna=False)
        expected = data[["Delta_MaxE_lib1", "Delta_MaxE_lib2"]].min(axis=1, skipna=False)
        observed = expected.notna() & efficacy.notna()
        if not np.allclose(efficacy[observed], expected[observed], rtol=1e-6, atol=1e-7):
            raise ValueError(f"{source} MaxE gain differs from the publisher formula")
        qp = pd.DataFrame({
            "id": source + ":" + data.BARCODE + ":" + data.DRUGSET_ID + ":" + data.cmatrix,
            "source": source, "cell_id": identity, "compound_a": left, "compound_b": right,
            "source_liver": data.TISSUE.fillna("").str.strip().str.lower().eq("liver"),
            "zip": np.nan, "bliss": data.Bliss_matrix * 100,
            "css": np.nan, "added_inhibition": efficacy * 100,
        })
        qp = qp.dropna(subset=["bliss", "added_inhibition"])
        qp["zip"] = qp.id.map(zip_scores)
        out = map_assay(qp, cells, compounds)
        audit[source] = {"published_matrices": len(data), "resolved_matrices": len(out),
                         "reconstructed_zip_matrices": int(out.zip.notna().sum()),
                         "adult_hcc_zip_matrices": int(out.loc[out.adult_hcc, "zip"].notna().sum()),
                         "adult_hcc_matrices": int(out.adult_hcc.sum()),
                         "adult_hcc_models": out.loc[out.adult_hcc, "cell_id"].nunique()}
        yield out


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cells = CellRegistry(a.depmap, a.cells)
    drugs = CompoundRegistry(a.drugs)
    structures_manifest = a.sanger_structures / "receipt.json"
    claimed = json.loads(structures_manifest.read_text())
    structures = a.sanger_structures / "sanger_structures.tsv"
    if record(structures) != claimed["files"][structures.name]:
        raise ValueError("The dedicated Sanger molecular registries lack a matching receipt")
    if any(record(f) != claimed["inputs"][f.name] for f in a.sanger):
        raise ValueError("Sanger compound registry is frozen against a different summary release")
    drugs.add_registered_sanger(pd.read_csv(structures, sep="\t", dtype=str))
    zip_manifest = a.sanger_zip / "receipt.json"
    reads = json.loads(zip_manifest.read_text())
    zip_path = a.sanger_zip / "sanger_zip.parquet"
    if record(zip_path) != reads["files"]["sanger_zip.parquet"]:
        raise ValueError("Sanger ZIP reconstruction failed its recorded SHA-256")
    reconstructed_zip = pd.read_parquet(zip_path).set_index("id", verify_integrity=True).zip
    audit_comb, audit_sanger, audit_db = {}, {}, {}
    batches, replaced = [], ()
    if a.drugcombdb is not None:
        raw_db, audit_db = oncology_rows(a.drugcombdb, cells, drugs)
        filtered = map_assay(raw_db, cells, drugs)
        if filtered.empty:
            raise ValueError("DrugCombDB O'Neil lacks eligible observed dose matrices")
        audit_db["eligible_matched_rows"] = len(filtered)
        replaced = (STUDY,)
        batches.append(filtered)
    batches.extend(drugcomb(a.drugcomb, cells, drugs, audit_comb, replaced=replaced))
    batches.extend(sanger(a.sanger, cells, drugs, audit_sanger, reconstructed_zip))
    distinct = pd.concat(batches, ignore_index=True).drop_duplicates()
    validate_assays(distinct)
    test = distinct[distinct.adult_hcc]
    if len(test) < 50 or test.cell_group.nunique() < 5:
        raise ValueError("Insufficient real HCC data for the registered split design")
    used = sorted(set(distinct.compound_a) | set(distinct.compound_b))
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as tmp:
        staged = Path(tmp)
        assay = staged / "assays.parquet"
        distinct.to_parquet(assay, index=False)
        drugs.keys.loc[used, ["smiles", "connectivity", "scaffold"]].to_csv(staged / "molecules.tsv", sep="\t")
        cells.keys.loc[sorted(set(distinct.cell_id))].to_csv(staged / "cells.tsv", sep="\t")
        receipt = {
            "input_files": {p.name: record(p) for p in (a.drugcomb, a.depmap, a.drugs, a.cells, *a.sanger)},
            "sanger_compound_snapshot": record(structures_manifest),
            "sanger_zip_reconstruction": record(zip_manifest),
            "drugcomb": audit_comb, "drugcombdb": audit_db, "sanger": audit_sanger, "all_assays": len(distinct),
            "adult_hcc_assays": len(test), "adult_hcc_cell_groups": test.cell_group.nunique(),
            "registered_compounds": len(used), "failed_publisher_compound_records": drugs.rejected,
            "scaffold_aliases_closed_under_equal_parent_connectivity": drugs.scaffold_aliases,
            "endpoint_sources": {
                "DrugComb": "Published ZIP, Bliss, CSS_ri and matched-dose HSA increment",
                "Sanger": "Published Bliss and gain over strongest single agent, and reconstructed ZIP",
                "DrugCombDB": "Published ZIP and Bliss, observed dose-grid gain over the strongest single agent",
            },
            "files": {p.name: record(p) for p in sorted(staged.iterdir()) if p.is_file()},
        }
        (staged / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        staged.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("drugcomb", "depmap", "drugs", "cells", "sanger-structures", "sanger-zip", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--sanger", nargs="+", type=Path, required=True)
    p.add_argument("--drugcombdb", type=Path)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
