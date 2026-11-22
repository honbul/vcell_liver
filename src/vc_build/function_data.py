"""Build the LIMORE observed Activity Area cohort from publisher and GEO tables."""

from contextlib import contextmanager
import hashlib
import json
import re
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


DATA_LICENSE = (
    "Publicly accessible for research. The sources were used in place; "
    "raw source files and derived response matrices are retained in project storage. "
    "Individual data redistribution rights are unverified."
)


def file_sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(specs):
    receipt = {}
    for label, spec in specs.items():
        path = Path(spec["path"])
        sha256 = file_sha256(path)
        if sha256 != spec["sha256"]:
            raise ValueError(f"{label} checksum mismatch for {path}")
        receipt[label] = {**spec, "bytes": path.stat().st_size, "access": DATA_LICENSE}
    return receipt


def read_tsv(path, **options):
    return pd.read_csv(path, sep="\t", **options)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def alias_token(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def clean_cell(value):
    name = str(value).strip()
    return "Huh6" if name == "Huh-6 clone 5" else name


def clean_drug(value):
    # The LI(GEO)/shared legacy spelling is independently verified against the publisher matrix.
    name = str(value).strip()
    return "(+)-JQ1" if name == "JQ1" else name


def canonical_chemistry(smiles):
    """Preserve exact isomeric parents; group connectivity and scaffold for splitting."""
    compound = Chem.MolFromSmiles(smiles)
    if compound is None or not compound.GetNumAtoms():
        raise ValueError("Unparseable compound structure")
    parent = rdMolStandardize.FragmentParent(compound)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    Chem.SanitizeMol(parent)
    key = Chem.MolToInchiKey(parent)
    if not key:
        raise ValueError("Unresolved InChIKey")
    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    return parent_smiles, key, key.split("-")[0], scaffold or f"ACYCLIC:{key.split('-')[0]}"


def raw_sheet(path, sheet, header):
    table = pd.read_excel(path, sheet_name=sheet, header=header, engine="openpyxl").dropna(axis=1, how="all")
    return table


def provenance_registry(path, study):
    table = read_tsv(path, dtype=str).fillna("")
    table = table[table.study.eq(study)].copy()
    if table.source_name.duplicated().any() or table.model_id.duplicated().any():
        raise ValueError("Ambiguous cell identity registry")
    return table.set_index("source_name", drop=False)


def limore_panels(settings, cells):
    cfg = settings["limore"]
    book = cfg["table_s4"]["path"]
    panel = raw_sheet(book, "Drug list", 1).dropna(subset=["Company"]).copy()
    panel = panel.rename(columns={
        "Drug": "source_compound_label", "Company": "vendor", "Catalogue": "vendor_code",
        "Screening max dose (µM)": "dose_max_uM", "Dilution fold": "dilution_text",
        "# Point of doses": "dose_points", "Targets": "drug_target_source_labels",
    })
    panel.source_compound_label = panel.source_compound_label.map(clean_drug)
    panel["dose_dilution_fold"] = panel.dilution_text.str.extract(r"^(\d+)-fold$")[0].astype(int)
    panel["dose_min_uM"] = panel.dose_max_uM / panel.dose_dilution_fold.pow(panel.dose_points - 1)
    panel["dose_points"] = panel.dose_points.astype(int)
    panel["vendor_code"] = panel.vendor_code.astype(str).str.strip()
    panel = panel.drop(columns=["dilution_text"]).set_index("source_compound_label", drop=False)
    if not panel.index.is_unique:
        raise ValueError("Conflicting LIMORE Table S4 compound metadata")

    diagnoses = raw_sheet(cfg["table_s1"]["path"], "Descriptive information", 2)
    diagnoses = diagnoses.rename(columns={diagnoses.columns[0]: "source_name", "Cancer Type": "table_s1_diagnosis"})
    source_identity = {alias_token(name): name for name in cells.index}
    diagnoses["source_name"] = diagnoses.source_name.map(clean_cell).map(alias_token).map(source_identity)
    diagnoses = diagnoses[diagnoses.source_name.notna()].set_index("source_name")
    if set(diagnoses.index) != set(cells.index) or not diagnoses.index.is_unique:
        raise ValueError("LIMORE Table S1 cell identity did not match the audited cell registry")
    accepted = cells.included.eq("True")
    if not diagnoses.loc[accepted, "table_s1_diagnosis"].eq("HCC").all():
        raise ValueError("Included LIMORE cell was not diagnosed as HCC by original Table S1")
    return panel, diagnoses


def limore_endpoint(settings, cells, panel):
    source = settings["limore"]
    aa = raw_sheet(source["table_s4"]["path"], "Activity Area", 1)
    aa = aa.rename(columns={"Cell model": "source_name"})
    aa.source_name = aa.source_name.map(clean_cell)
    aa = aa.set_index("source_name")
    aa.columns = aa.columns.map(clean_drug)
    if set(aa.index) != set(cells.index) or set(aa.columns) != set(panel.index):
        raise ValueError("LIMORE Table S4 response table and registered panel identities disagree")
    if not aa.index.is_unique or not aa.columns.is_unique:
        raise ValueError("Duplicate LIMORE source response keys")
    aa = aa.reset_index().melt("source_name", var_name="source_compound_label", value_name="endpoint_value")

    copy = read_tsv(source["drug_copy"]["path"]).rename(columns={
        "cell": "source_name", "drug": "source_compound_label", "Activity Area": "legacy_aa",
    })
    copy.source_compound_label = copy.source_compound_label.map(clean_drug)
    audit = aa.merge(copy[["source_name", "source_compound_label", "legacy_aa"]],
                     on=["source_name", "source_compound_label"], validate="1:1", how="outer", indicator=True)
    different = audit._merge.ne("both") | ~np.isclose(audit.endpoint_value, audit.legacy_aa, atol=1e-8, rtol=0)
    missing = aa.endpoint_value.isna()
    return aa.loc[~missing], {
        "publisher_observed_activity_area_rows": int((~missing).sum()),
        "publisher_missing_activity_area_rows": int(missing.sum()),
        "legacy_copy_discrepant_rows_at_1e_8": int(different.sum()),
        "legacy_copy_max_abs_difference": float((audit.endpoint_value - audit.legacy_aa).abs().max()),
    }


def limore_rna(path, cells):
    frame = read_tsv(path).rename(columns={"tracking_id": "ensembl_gene_id", "geneSymbol": "gene_symbol"})
    frame = frame[frame.geneType.eq("protein_coding") & frame.gene_symbol.notna()]
    frame = frame[~frame.gene_symbol.duplicated(keep=False) & ~frame.ensembl_gene_id.duplicated(keep=False)]
    axis = frame[["ensembl_gene_id", "gene_symbol"]].sort_values("gene_symbol").reset_index(drop=True)
    if set(cells.index) != (set(frame.columns) - {"gene_symbol", "geneType", "ensembl_gene_id"}):
        raise ValueError("LIMORE GEO RNA cell-axis disagrees with registry")
    included = cells[cells.included.eq("True")]
    values = frame.set_index("gene_symbol").loc[axis.gene_symbol, included.index].T
    values.index = values.index.map(included.model_id).rename("cell_id")
    if not np.isfinite(values).to_numpy().all() or values.min().min() < 0:
        raise ValueError("Invalid deposited LIMORE log2 FPKM matrix")
    return values.astype("float32").sort_index(), axis


def compound_registry(settings, panel):
    evidence = read_tsv(settings["limore"]["chemistry"], dtype=str, keep_default_na=False)
    if set(evidence.source_compound_label) != set(panel.index) or not evidence.source_compound_label.is_unique:
        raise ValueError("LIMORE structure audit must cover the entire published panel")
    evidence = evidence.set_index("source_compound_label", drop=True)
    if not evidence.catalog_code.eq(panel.vendor_code).all():
        raise ValueError("Vendor-evidence catalog identifiers and Table S4 do not match")
    merged = panel.join(evidence).copy()
    columns = ["isomeric_smiles", "compound_id", "connectivity_group", "scaffold_group"]
    identities = []
    for compound in merged.itertuples(index=False):
        if compound.structure_status == "eligible":
            identities.append(canonical_chemistry(compound.pubchem_isomeric_smiles))
        else:
            identities.append((None,) * len(columns))
    merged[columns] = pd.DataFrame(identities, columns=columns, index=merged.index)
    if merged.compound_id.dropna().duplicated().any():
        raise ValueError("Different Table S4 compound labels resolve to the same exact parent identity")
    return merged


def limore_bundle(settings):
    cfg = settings["limore"]
    sources = verify_inputs({name: cfg[name] for name in ("table_s4", "table_s1", "drug_copy", "rna")})
    for key in ("chemistry", "cell_registry"):
        path = Path(cfg[key])
        sources[key] = {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    cells = provenance_registry(cfg["cell_registry"], "LIMORE")
    panel, diagnoses = limore_panels(settings, cells)
    cells["table_s1_age_years"] = pd.to_numeric(diagnoses.Age, errors="coerce")
    excluded_age = cells.included.eq("True") & ~cells.table_s1_age_years.ge(18)
    cells.loc[excluded_age, "included"] = "False"
    cells.loc[excluded_age, "exclusion_reason"] = "Adult donor age unconfirmed by LIMORE Table S1"
    pediatric = excluded_age & cells.table_s1_age_years.lt(18)
    cells.loc[pediatric, "exclusion_reason"] = "Pediatric HCC donor in LIMORE Table S1"
    endpoints, audit = limore_endpoint(settings, cells, panel)
    rna, axis = limore_rna(cfg["rna"]["path"], cells)
    compounds = compound_registry(settings, panel)
    cell_cols = ["model_id", "family_id", "cellosaurus_id", "source_diagnosis", "flags", "included", "exclusion_reason"]
    linked = endpoints.join(cells[cell_cols], on="source_name", validate="m:1")
    compounds_cols = ["compound_id", "isomeric_smiles", "connectivity_group", "scaffold_group",
                      "dose_min_uM", "dose_max_uM", "dose_points", "drug_target_source_labels"]
    linked = linked.join(compounds[compounds_cols], on="source_compound_label", validate="m:1")
    linked = linked[linked.included.eq("True") & linked.compound_id.notna()]
    rows = linked.rename(columns={"model_id": "cell_id", "family_id": "cell_group",
                                  "source_name": "source_cell_label"}).drop(columns=["included", "exclusion_reason"])
    rows = rows.assign(
        source_id="LIMORE", source_protocol=cfg["source_protocol"], assay_id="CellTiterGlo_72h",
        endpoint_id=cfg["endpoint_id"], endpoint_unit=cfg["endpoint_unit"],
        endpoint_contract="curve_summary", endpoint_direction="higher_is_more_sensitive",
        supported_dose_response=False, time_h=cfg["time_h"], dose_uM=None,
        baseline_is_proxy=True, baseline_time_h=None, rna_assay=cfg["rna_assay"],
        cell_type="adult_HCC", lineage_role="adult_hcc", tissue_origin="liver", culture_format="2D", missing_reason=None,
    )
    rows["record_id"] = "LIMORE|" + rows.cell_id + "|" + rows.compound_id + "|" + rows.endpoint_id
    rows = rows.sort_values("record_id", ignore_index=True)
    if rows.record_id.duplicated().any():
        raise ValueError("Duplicate LIMORE observed endpoint records")
    receipt = {
        "source_publication": cfg["publication"], "inputs": sources,
        "biological_replicates": "Aggregate value per observed cell-compound pair, original rows have no replicate IDs",
        "rna_timing": "Independent deposited basal RNA; exact assay-matched 0h RNA unavailable",
        "clinical_provenance": "Read-only Table S1 and public Cellosaurus/DepMap registry. Adult HCC donor age required.",
        "cell_exclusions": cells.loc[cells.included.ne("True"), ["source_name", "exclusion_reason"]].to_dict("records"),
        "prior_exposure": "LIMORE expression and response were used in Pathway A1 and softprior studies.",
        "rna_axis": "Observed protein coding genes, unique symbol and Ensembl ID; no fitted transforms",
        "rna_genes": int(len(axis)), "n_rows": int(len(rows)), "n_cells": int(rows.cell_id.nunique()),
        "n_cell_groups": int(rows.cell_group.nunique()), "n_compounds": int(rows.compound_id.nunique()),
        "adult_age_documented": int(pd.to_numeric(diagnoses.loc[cells.included.eq("True"), "Age"], errors="coerce").ge(18).sum()),
        "all_source_models": int(len(cells)), "all_source_compounds": int(len(compounds)),
        "compound_exclusions": compounds.loc[compounds.compound_id.isna(),
                                             ["source_compound_label", "structure_status"]].to_dict("records"),
        **audit,
    }
    return rows, rna, axis, cells, compounds, receipt


@contextmanager
def new_output(output):
    """Write complete bundles by rename; decline to overwrite an existing run."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        yield tmp
        if output.exists():
            raise FileExistsError(output)
        tmp.rename(output)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def write_limore(settings, output):
    rows, rna, axis, cells, compounds, receipt = limore_bundle(settings)
    with new_output(output) as path:
        rows.to_parquet(path / "rows.parquet", index=False)
        rna.to_parquet(path / "rna.parquet")
        axis.to_csv(path / "genes.tsv", sep="\t", index=False)
        cells.to_csv(path / "cell_registry.tsv", sep="\t", index=False)
        compounds.to_csv(path / "compound_registry.tsv", sep="\t", index=False)
        receipt["artifacts"] = {
            file.name: {"sha256": file_sha256(file), "bytes": file.stat().st_size}
            for file in sorted(path.iterdir())
        }
        write_json(path / "receipt.json", receipt)
    return receipt
