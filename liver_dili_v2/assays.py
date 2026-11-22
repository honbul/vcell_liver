"""Fetch and harmonize public in-vitro mechanism assays without FDA endpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urljoin
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from liver_dili.chemistry import identity
from liver_dili.splits import digest

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
CHEMBL_TARGETS = {"BSEP": "CHEMBL6020", "CYP3A4": "CHEMBL340", "CYP2C9": "CHEMBL3397", "CYP2D6": "CHEMBL289"}
TOX21 = {"SR-ARE": "ARE", "SR-MMP": "MMP", "SR-HSE": "HSR", "SR-p53": "p53", "SR-ATAD5": "ATAD5"}
EPA_TARGETS = {
    "BSEP": r"BSEP|ABCB11",
    "CYP3A4": r"CYP.?3A4",
    "CYP2C9": r"CYP.?2C9",
    "CYP2D6": r"CYP.?2D6",
    "MMP": r"(?:^|[_ -])MMP(?:$|[_ -])|MITO.*MEMBRANE.*POTENTIAL",
    "ARE": r"(?:^|[_ -])ARE(?:$|[_ -])|NFE2L2|NRF2|ANTIOXIDANT RESPONSE",
    "ERS": r"(?:^|[_ -])ESRE(?:$|[_ -])|ER[_ -]?STRESS|ENDOPLASMIC.*STRESS",
}


def json_request(url):
    last = None
    for attempt in range(4):
        try:
            with urlopen(url, timeout=23) as stream:
                return json.load(stream)
        except (OSError, ValueError) as exc:
            last = exc
            time.sleep(1 + 2 * attempt)
    raise RuntimeError(f"Failed ChEMBL API request {url}") from last


def fetch_target(task, target, cache):
    folder = cache / task
    folder.mkdir(exist_ok=True)
    args = urlencode({
        "target_chembl_id": target, "standard_type__in": "IC50,Ki,Kd",
        "standard_units": "nM", "assay_type__in": "B,F,A", "limit": 1000,
        "only": "activity_id,canonical_smiles,standard_relation,standard_value,standard_type,assay_chembl_id,data_validity_comment",
        "order_by": "activity_id",
    })
    url = f"{CHEMBL}activity.json?{args}"
    pages = []
    while url:
        path = folder / f"{len(pages):04d}.json"
        if not path.exists():
            data = json_request(url)
            temporary = path.with_suffix(".part")
            temporary.write_text(json.dumps(data, sort_keys=True))
            temporary.rename(path)
        data = json.loads(path.read_text())
        pages.append((path, data))
        link = data["page_meta"]["next"]
        url = urljoin(CHEMBL, link) if link else None
    expected = int(pages[0][1]["page_meta"]["total_count"])
    got = [r for _, paged in pages for r in paged["activities"]]
    if len(got) != expected or len({r["activity_id"] for r in got}) != expected:
        raise ValueError(f"Partial or changed ChEMBL result for {task}")
    rows = pd.DataFrame(got).rename(columns={"canonical_smiles": "smiles"})
    rows["value_nm"] = pd.to_numeric(rows.standard_value, errors="coerce")
    quality = rows.data_validity_comment.fillna("").eq("")
    valid = rows.standard_relation.isin(("=", "<", "<=", ">", ">=")) & rows.value_nm.gt(0) & quality
    # Use separated active/inactive thresholds and discard the middle to limit measurement noise.
    strong = rows.value_nm.le(10_000) & ~rows.standard_relation.isin((">", ">="))
    weak = rows.value_nm.ge(30_000) & ~rows.standard_relation.isin(("<", "<="))
    rows = rows.loc[valid & (strong | weak), ["smiles", "assay_chembl_id"]]
    rows["activity"] = strong.loc[rows.index].astype(int)
    rows["task"] = f"chembl__{task}"
    return rows[["smiles", "task", "activity"]], {
        "provider_target": target, "measurement_rows": expected, "binary_rows": len(rows),
        "api_endpoint": f"{CHEMBL}activity.json?{args}",
        "pages": {str(p.relative_to(cache)): digest(p) for p, _ in pages},
    }


def tox21(path):
    rows = []
    with ZipFile(path) as zipfile, rdBase.BlockLogs():
        member, = [name for name in zipfile.namelist() if name.endswith(".sdf")]
        for molecule in Chem.ForwardSDMolSupplier(zipfile.open(member)):
            if molecule is None:
                continue
            smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
            for name, task in TOX21.items():
                if molecule.HasProp(name) and molecule.GetProp(name) in ("0", "1"):
                    rows.append((smiles, f"tox21__{task}", int(molecule.GetProp(name))))
    return pd.DataFrame(rows, columns=["smiles", "task", "activity"])


def standardized(frame):
    output = {}
    with rdBase.BlockLogs():
        for smiles in frame.smiles.dropna().unique():
            try:
                output[smiles] = identity(smiles)
            except (ValueError, RuntimeError):
                continue
    names = ["smiles", "inchikey", "connectivity", "scaffold"]
    values = pd.DataFrame.from_dict(output, orient="index", columns=names).rename_axis("raw_smiles")
    data = frame.rename(columns={"smiles": "raw_smiles"}).merge(values, on="raw_smiles", validate="many_to_one")
    summary = data.groupby(["connectivity", "task"]).activity.agg(["min", "max"])
    usable = summary.loc[summary["min"].eq(summary["max"]), "min"].rename("label")
    identifiers = (data.sort_values("inchikey").drop_duplicates("connectivity")
                   .set_index("connectivity")[["smiles", "scaffold", "inchikey"]])
    wide = identifiers.join(usable.unstack()).dropna(subset=sorted(data.task.unique()), how="all")
    return wide.reset_index(), int(len(summary) - len(usable))


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    cache = args.out / "chembl_pages"
    cache.mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {task: pool.submit(fetch_target, task, target, cache)
                   for task, target in CHEMBL_TARGETS.items()}
        complete = {task: future.result() for task, future in futures.items()}
    table, conflict = standardized(pd.concat([tox21(args.tox21), *(result[0] for result in complete.values())]))
    table.to_parquet(args.out / "mechanistic.parquet", index=False)
    records = []
    for name in [c for c in table if "__" in c]:
        present = table[name].dropna()
        records.append({"endpoint": name, "n": int(len(present)), "positive": int(present.sum())})
    (args.out / "receipt.json").write_text(json.dumps({
        "Tox21_training_sha256": digest(args.tox21), "ChEMBL": {key: val[1] for key, val in complete.items()},
        "task_support": records, "n_structure_connectivities": len(table),
        "conflicting_endpoint_connectivities_removed": conflict,
        "mechanistic_sha256": digest(args.out / "mechanistic.parquet"),
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tox21", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
