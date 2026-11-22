"""Register public Zhu adult-HCC baseline RNA and response identities."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from rdkit import Chem

from .expanded_panel_cv import RECEIPTS
from .external import _validate
from .function_data import canonical_chemistry, new_output
from .io import publish_json, sha256
from .v0plus_cv import training_inputs
from .v0plus_licob import ASSET_HASHES
from .yang_hcc_external import require_hash

ASSETS = {
    "zhu-s1.xlsx": "a81e33f7a8a3a6f86a8c06bd7372f10683144c79a8bdcd1eb67a0474dc4e37c5",
    "zhu-s3.xlsx": "15402cd1122b536fd24f693da31fa1dbeb0f7e4b3c8c8569b1683841f6cbaec5",
    "zhu-s4.xlsx": "f751429f48d7cbbbf97250236766ea06844173ee0f840965d85af44b0b8a1a92",
}
PUBLICATION = "10.1016/j.xcrm.2023.101375"
SHEETS = ("AUC，1-AA", "Primary screen_Inhibition rato")


def row_keys(sheet, *, start, column, model):
    entries = {}
    for n, (value,) in enumerate(sheet.iter_rows(
            min_row=start, min_col=column, max_col=column, values_only=True), start=start):
        if model.fullmatch(str(value).strip()):
            key = str(value).strip()
            if key in entries:
                raise ValueError("Duplicate public screen model")
            entries[key] = n
    return entries


def donors(handle):
    sheet = handle["Descriptive information"]
    records = []
    for name, patient, histology, _, age in sheet.iter_rows(min_row=4, max_col=5, values_only=True):
        if str(histology).strip() == "HCC" and isinstance(age, (int, float)) and age >= 18:
            sample = str(name).strip()
            if not re.fullmatch(r"HCC-\d+", sample) or patient is None:
                raise ValueError("Incomplete identity on an adult HCC model")
            anonymous = hashlib.sha256(f"zhu-101375|{patient}".encode()).hexdigest()[:16]
            records.append({"cell_id": sample, "lineage_group": f"ZHU:{anonymous}", "adult_hcc": True})
    return pd.DataFrame(records).set_index("cell_id", verify_integrity=True)


def expression(handle):
    rows = handle["RNA-seq"].iter_rows(min_row=2, values_only=True)
    label, *samples = next(rows)
    if label != "genes" or len(set(samples)) != 53 or None in samples:
        raise ValueError("Unexpected S3 RNA sample axis")
    frame = pd.DataFrame.from_records(rows, columns=[label, *samples])
    genes = frame[label].astype(str).str.strip()
    keep = frame[label].notna() & ~genes.duplicated(keep=False)
    values = frame.loc[keep, samples].set_axis(genes[keep]).T.astype("float32")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Nonfinite or missing public expression")
    return values.rename_axis("cell_id")


def catalogue(book, vendor):
    sheet = book["Chemical Data-301cpds"]
    rows = list(sheet.values)
    frame = pd.DataFrame(rows[1:], columns=rows[0])[["cat", "name", "SMILES"]]
    frame = frame.dropna().assign(cat=lambda x: x.cat.astype(str).str.strip())
    name = frame.name.map(lambda x: str(x).strip())
    if name.duplicated().any() or frame.cat.duplicated().any():
        raise ValueError("Publication chemistry identifiers are not unique")
    types = pd.read_excel(vendor, sheet_name=1, usecols=["Cat", "SMILES"]).dropna()
    keys = types.groupby("Cat").SMILES.agg(lambda vals: sorted(set(vals)))
    return frame, keys[keys.map(len).eq(1)].map(lambda a: a[0])


def identity(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "no_vendor_structure"
    standardized, key, _, _ = canonical_chemistry(smiles)
    if any(info.specified == Chem.StereoSpecified.Unspecified
           for info in Chem.FindPotentialStereo(Chem.MolFromSmiles(standardized))):
        return None, None, "underspecified_stereo"
    return standardized, key, "exact"


def compound_metadata(book, chemical, vendor):
    shapes = chemical.set_index("cat").SMILES
    by_name = chemical.assign(name=chemical.name.astype(str).str.strip()).set_index("name")
    panels = {key: [] for key in SHEETS}
    sheet = book[SHEETS[0]]
    header = next(sheet.iter_rows(min_row=2,max_row=2,values_only=True))
    if header[1:] != tuple(f"drug{i}" for i in range(1,65)):
        raise ValueError("Unexpected follow-up AUC columns")
    for number, name, _, target, category, pathway, *_, company, code in book["Further drug screen_Drug list"].iter_rows(
            min_row=3, max_col=11, values_only=True):
        if not isinstance(number, int):
            continue
        code = str(code).strip()
        native = str(name).strip()
        shape = None
        if str(company).strip().lower() == "selleck":
            shape = shapes.get(code, vendor.get(code))
        panels[SHEETS[0]].append((native, code, " ".join(str(x) for x in (target,category,pathway) if x),
                                  number, *identity(shape)))
    for n, (name,) in enumerate(book[SHEETS[1]].iter_rows(min_row=3,max_col=1,values_only=True), start=3):
        if name is None:
            continue
        label = str(name).strip()
        row = by_name.loc[label] if label in by_name.index else None
        cat = None if row is None else row["cat"]
        shape = None if row is None else row["SMILES"]
        panels[SHEETS[1]].append((label, cat, "", n, *identity(shape)))
    headers = ["source_compound_label","catalogue","drug_target_source_labels","drug_axis","isomeric_smiles","compound_id","status"]
    data = {name: pd.DataFrame(arms, columns=headers) for name, arms in panels.items()}
    if [len(data[n]) for n in SHEETS] != [64,301]:
        raise ValueError("Published multi- and single-dose compound totals changed")
    for name, frame in data.items():
        duplicates = frame.status.eq("exact") & frame.compound_id.duplicated(keep=False)
        data[name] = frame.assign(status=frame.status.mask(duplicates,"duplicate_parent"))
    return data


def register(args):
    for name, expected in ASSETS.items():
        require_hash(args.data / name, expected)
    require_hash(args.vendor, ASSET_HASHES["catalog"])
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    fit, fit_rna, _ = training_inputs(args.core)
    with closing(load_workbook(args.data/"zhu-s1.xlsx",read_only=True,data_only=True)) as s1:
        patients = donors(s1)
    with closing(load_workbook(args.data/"zhu-s3.xlsx",read_only=True,data_only=True)) as s3:
        rna = expression(s3)
    rna = rna[rna.columns.intersection(fit_rna.columns).sort_values()]
    with closing(load_workbook(args.data/"zhu-s4.xlsx",read_only=True,data_only=True)) as s4:
        m = re.compile(r"(?:HCC|ICC|CHC|GBC|HCCA)-\d+")
        multi = row_keys(s4[SHEETS[0]],start=3,column=1,model=m)
        single_header = next(s4[SHEETS[1]].iter_rows(min_row=2,max_row=2,values_only=True))
        single = {str(k).strip(): i for i,k in enumerate(single_header) if m.fullmatch(str(k).strip())}
        plates = {SHEETS[0]:multi,SHEETS[1]:single}
        chemical, vendor = catalogue(s4,args.vendor)
        drug = compound_metadata(s4,chemical,vendor)
    artifacts, info = {}, {}
    with new_output(args.out) as out:
        all_samples = set()
        for name, label in zip(SHEETS,("auc","single_dose")):
            support = patients.index.intersection(rna.index).intersection(plates[name]).sort_values()
            all_samples.update(support)
            if label == "auc" and (len(support),patients.loc[support].lineage_group.nunique()) != (16,16):
                raise ValueError("Unexpected exact adult AUC-linked patient support")
            fields = ("sheet_row","sheet_column") if label == "auc" else ("sheet_column","sheet_row")
            full = drug[name]
            eligible = full.loc[full.status.eq("exact") & full.compound_id.isin(fit.compound_id)]
            cells = patients.loc[support].assign(**{fields[0]:support.map(plates[name])}).reset_index()
            query = cells.merge(eligible.rename(columns={"drug_axis":fields[1]}),how="cross")
            query = query.assign(row_id=lambda q: "ZHU|"+label+"|"+q.cell_id+"|"+q.compound_id)
            _validate(fit,query)
            for key, data in ((f"{label}_roster.parquet",query),(f"{label}_chemistry.parquet",full)):
                data.to_parquet(out/key,index=False)
                artifacts[key] = sha256(out/key)
            info[label] = {"models":len(support),"donors":patients.loc[support].lineage_group.nunique(),
                           "all_drugs":len(full),"exact_structure_drugs":int(full.status.eq("exact").sum()),
                           "core_drugs":len(eligible),"query_rows":len(query),
                           "chemistry_status":full.status.value_counts().to_dict()}
        target = out/"rna.parquet"
        rna.loc[sorted(all_samples)].to_parquet(target)
        artifacts[target.name] = sha256(target)
        publish_json(out/"receipt.json",{"publication":PUBLICATION,"published_accession":"HRA004348",
                     "supplement_sha256":ASSETS,"drug_catalogue_sha256":ASSET_HASHES["catalog"],
                     "core_receipt_sha256":RECEIPTS["core"],"genes":len(rna.columns),
                     "scored_outcome_values_read":0,"support":info,"files":artifacts})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("data","vendor","core","out"):
        parser.add_argument("--"+field,type=Path,required=True)
    register(parser.parse_args())


if __name__ == "__main__":
    main()
