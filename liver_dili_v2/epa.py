"""Join EPA invitrodb v4.3 functional endpoints to exact DSSTox structures."""

import argparse
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd
from rdkit import Chem, rdBase

from liver_dili.splits import digest

from .assays import EPA_TARGETS, standardized

CSV_COLUMNS = ("dsstox_substance_id", "aeid", "aenm", "hitc")


def hit_calls(path):
    with ZipFile(path) as z:
        member, = [m for m in z.namelist() if m.startswith("mc5-6") and m.endswith(".csv")]
        annotations, = [m for m in z.namelist() if m.startswith("assay_annotations") and m.endswith(".xlsx")]
        books = pd.read_excel(io.BytesIO(z.read(annotations)))
        names = books.assay_component_endpoint_name.fillna("").astype(str)
        useful = {}
        for target, pattern in EPA_TARGETS.items():
            # Biological assay names and targeted gene-function annotations are distinct endpoints.
            hits = books.loc[names.str.contains(pattern, case=False, regex=True) &
                             ~names.str.contains("viability|ARELUC|^IUF_|^BSK_", case=False, regex=True)]
            for aeid, nm in hits[["aeid", "assay_component_endpoint_name"]].itertuples(index=False):
                useful[int(aeid)] = (target, nm)
        results = []
        for chunk in pd.read_csv(z.open(member), usecols=CSV_COLUMNS, chunksize=260000):
            matches = chunk.loc[chunk.aeid.isin(useful)].dropna(subset=["dsstox_substance_id", "hitc"]).copy()
            matches = matches.loc[matches.hitc.ge(.9) | matches.hitc.le(.5)].copy()
            matches["activity"] = matches.hitc.ge(.9).astype(int)
            matches["task"] = matches.aeid.map(lambda i: f"toxcast__{useful[i][0]}_{i:d}")
            results.append(matches[["dsstox_substance_id", "task", "activity"]])
    return pd.concat(results, ignore_index=True), useful


def normalized(name):
    return re.sub("[^a-z0-9]", "", str(name).lower())


def headings(columns):
    normalized_names = [normalized(n) for n in columns]
    sid = next(i for i, n in enumerate(normalized_names) if n in ("dtxsid", "dsstoxsubstanceid"))
    chemical = next(i for i, n in enumerate(normalized_names)
                    if n in ("inchi", "inchistring", "standardinchi", "smiles", "canonicalsmiles", "msreadysmiles", "strukturinchi"))
    return sid, chemical


def structures(path, which):
    matches = {}
    with ZipFile(path) as z:
        candidates = [p for p in z.namelist() if p.lower().endswith((".xlsx", ".csv", ".txt", ".tsv"))]
        for filename in candidates:
            if filename.lower().endswith(".xlsx"):
                from openpyxl import load_workbook

                workbook = load_workbook(io.BytesIO(z.read(filename)), read_only=True, data_only=True)
                for page in workbook:
                    page.reset_dimensions()
                iterator = (sheet.values for sheet in workbook)
            else:
                reader = pd.read_csv(z.open(filename), sep="\t" if filename.endswith((".txt", ".tsv")) else ",",
                                     chunksize=100_000, dtype=str, encoding_errors="replace")
                iterator = (iter([list(chunk.columns), *chunk.itertuples(index=False, name=None)]) for chunk in reader)
            for sheet in iterator:
                header = next(sheet)
                try:
                    sid, chem = headings(header)
                except StopIteration:
                    continue
                for line in sheet:
                    if len(line) > max(sid, chem) and line[sid] in which and isinstance(line[chem], str):
                        matches.setdefault(line[sid], set()).add(line[chem])
    values = {}
    with rdBase.BlockLogs():
        for key, accepted in matches.items():
            if len(accepted) != 1:
                continue
            text, = accepted
            if text.startswith("InChI="):
                mol = Chem.MolFromInchi(text)
                if mol is None:
                    continue
                value = Chem.MolToSmiles(mol, isomericSmiles=True)
            else:
                value = text
            values[key] = value
    return values


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    outcomes, descriptions = hit_calls(args.epa)
    print("EPA hit call rows", len(outcomes), "endpoints", len(descriptions), flush=True)
    keys = set(outcomes.dsstox_substance_id)
    concordance = structures(args.dsstox, keys)
    outcomes["smiles"] = outcomes.dsstox_substance_id.map(concordance)
    print("EPA DSSTox exact identifiers", len(concordance), "of", len(keys), flush=True)
    toxcast, conflict = standardized(outcomes.dropna(subset="smiles")[["smiles", "task", "activity"]])
    prior_path = args.prior / "mechanistic.parquet"
    earlier = pd.read_parquet(prior_path).set_index("connectivity")
    epa = toxcast.set_index("connectivity")
    ids = ["inchikey", "smiles", "scaffold"]
    column_data = earlier.drop(columns=ids).join(epa.drop(columns=ids), how="outer")
    data = column_data.join(earlier[ids].combine_first(epa[ids])).reset_index()
    data.to_parquet(args.out / "mechanistic.parquet", index=False)
    src_task = {r: {"molecules": int(data[r].count()), "positives": int(data[r].sum())}
                for r in column_data}
    if not len(concordance) or not any(r.startswith("toxcast__") for r in src_task):
        raise RuntimeError("No mapped public ToxCast endpoints")
    (args.out / "receipt.json").write_text(json.dumps({
        "EPA_hitcall_archive_sha256": digest(args.epa), "DSSTox_structure_archive_sha256": digest(args.dsstox),
        "prior_receipt_sha256": digest(args.prior / "receipt.json"), "tasks": src_task,
        "n_toxcast_assay_endpoints": len(descriptions), "toxcast_endpoint_names": descriptions,
        "n_toxcast_dsstox_registry": len(keys), "n_toxcast_exact_dsstox_matched": len(concordance),
        "n_toxcast_conflicting_connectivity_endpoint_removed": conflict,
        "mechanistic_sha256": digest(args.out / "mechanistic.parquet"), "data_n": len(data),
    }, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("epa", "dsstox", "prior", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    run(cli.parse_args())
