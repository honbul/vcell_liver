"""Register public cell-line and LICOB baseline proteomic matrices."""

from collections import Counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from openpyxl.reader.strings import read_string_table

from .function_data import new_output
from .io import publish_json, sha256
from .measured_drivers import sheet_rows
from .response_component_benchmark import external_inputs
from .v0plus_licob import ASSET_HASHES
from .yang_hcc_external import require_hash

SOURCES = {
    "procan_models": "68ff17c98d3f163992418f66e4ea776d484eb11475b4a8e1d079767eaa703b7e",
    "proteins": "c52b3195020618488b5b8970fe34b202a4514e25a3a583a18f1ec52482e292a9",
    "hgnc": "6df77ac3cbaeb86a6dce5e17d851ebb27b6b3cdcd2471a211a3d8464f00c54c4",
    "models": "9dbb9de8805696c1345816ab07edd23fb4fd95e117739f3c5c3b1cf062c1233b",
}
MEMBER = "xl/worksheets/sheet9.xml"


def unambiguous(table, names):
    table = table.drop_duplicates()
    mask = ~table[list(names)].apply(lambda column: column.duplicated(keep=False)).any(axis=1)
    return table[mask]


def procan(args):
    for name, digest in SOURCES.items():
        require_hash(getattr(args, name), digest)
    author = pd.read_csv(args.procan_models, sep="\t", dtype=str)
    broad = pd.read_csv(args.models, usecols=["ModelID", "SangerModelID"], dtype=str).dropna()
    broad = unambiguous(broad, ["ModelID", "SangerModelID"])
    mapped = unambiguous(author[["SIDM", "Project_Identifier"]].dropna(), ["SIDM", "Project_Identifier"])
    mapped = mapped.merge(broad, left_on="SIDM", right_on="SangerModelID", validate="one_to_one")
    hgnc = pd.read_csv(args.hgnc, sep="\t", usecols=["symbol", "status", "uniprot_ids"], dtype=str)
    approved = hgnc[hgnc.status.eq("Approved")].dropna()
    accession = approved.assign(protein=approved.uniprot_ids.str.split("|")).explode("protein")
    lookup = unambiguous(accession[["protein", "symbol"]], ["protein"]).set_index("protein").symbol
    header = pd.read_csv(args.proteins, sep="\t", nrows=0).columns
    genes = pd.DataFrame({"column": header[1:], "symbol": header[1:].str.split(";").str[0].map(lookup)})
    genes = unambiguous(genes.dropna(), ["symbol"])
    data = pd.read_csv(args.proteins, sep="\t", usecols=["Project_Identifier", *genes.column],
                       dtype=dict.fromkeys(genes.column, "float32")).set_index("Project_Identifier", verify_integrity=True)
    if set(data.index) - set(author.Project_Identifier):
        raise ValueError("ProCan profiles differ from the official identity map")
    data = data.join(mapped.set_index("Project_Identifier").ModelID, how="inner").set_index("ModelID")
    return data.rename(columns=genes.set_index("column").symbol).rename_axis("cell_id")


def licob_proteome(path):
    require_hash(path, ASSET_HASHES["node_archive"])
    with ZipFile(path) as archive:
        info = archive.getinfo(MEMBER)
        with archive.open("xl/sharedStrings.xml") as stream:
            strings = read_string_table(stream)
        rows = sheet_rows(archive, MEMBER, strings)
        _, title = next(rows)
        _, columns = next(rows)
        if not title["A"].startswith("H. Normalized proteomic data") or "Gene symbol" not in columns.pop("A"):
            raise ValueError("Unexpected LICOB H proteomics worksheet")
        if len(set(columns.values())) != len(columns):
            raise ValueError("Ambiguous LICOB proteomic model identifiers")
        parsed = [(row["A"], [float(row[col]) if row.get(col) not in (None, "", "NA") else np.nan
                             for col in columns]) for _, row in rows if "A" in row]
    counts = Counter(gene for gene, _ in parsed)
    frame = pd.DataFrame({gene: values for gene, values in parsed if counts[gene] == 1},
                         index=pd.Index(columns.values(), name="cell_id"), dtype="float32")
    return frame, f"{info.CRC:08x}"


def prepare(args):
    source = procan(args)
    organoids, crc = licob_proteome(args.node)
    train, query, _, registration = external_inputs(args, "LICOB")
    shared = source.columns.intersection(organoids.columns)
    ids = pd.Index(sorted(query.cell_id.unique()))
    if len(set(ids) & set(organoids.index)) != 43 or len(shared) < 100:
        raise ValueError("Unexpected independent LICOB protein support")
    source_support = train[train.cell_id.isin(source.index)]
    organoids = organoids.loc[ids]
    with new_output(args.out) as out:
        files = {"source.parquet": source, "licob.parquet": organoids}
        for name, frame in files.items():
            frame.to_parquet(out / name, index=True)
        publish_json(out / "receipt.json", {
            "input_hashes": SOURCES | {"LICOB_NODE": ASSET_HASHES["node_archive"], "LICOB_registry": registration},
            "licob_protein_member": MEMBER, "licob_protein_crc32": crc,
            "source_models": len(source), "source_proteins": len(source.columns), "licob_models": len(organoids),
            "licob_proteins": len(organoids.columns), "shared_proteins_exact_symbol": len(shared),
            "source_core_models": source_support.cell_id.nunique(),
            "source_core_lineages": source_support.lineage_group.nunique(), "source_core_rows": len(source_support),
            "licob_adult_donors": query.loc[query.adult_age_confirmed, "donor_id"].nunique(),
            "files": {name: sha256(out / name) for name in files},
        })
