"""Snapshot official PubChem structures for named Sanger matrix-screen compounds."""
import argparse
import json
from pathlib import Path
import tempfile
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from .identity import structural_identity
from .splits import record


PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"


def names_and_registry(files):
    compound_lists = []
    for file in files:
        columns = [f"lib{side}_{key}" for side in (1, 2) for key in ("name", "ID")]
        rows = pd.read_csv(file, usecols=columns, dtype=str)
        for side in (1, 2):
            compound_lists.append(rows[[f"lib{side}_ID", f"lib{side}_name"]].set_axis(["drug_id", "name"], axis=1))
    roster = pd.concat(compound_lists).dropna().drop_duplicates()
    if roster.drug_id.duplicated().any() or not len(roster):
        raise ValueError("Sanger assay has conflicting declared compound names")
    return roster.sort_values("drug_id").reset_index(drop=True)


def fetch(name):
    url = PUG + quote(name, safe="") + "/property/InChIKey,IsomericSMILES,Title/JSON"
    requests = []
    for delay in (0.3, 1.0, 3.0):
        sleep(delay)
        try:
            request = Request(url, headers={"User-Agent": "public-hcc-benchmark/1 (oncology metadata audit)"})
            with urlopen(request, timeout=20) as response:
                data = response.read()
                requests.append({"status": response.status, "bytes": len(data), "url": url})
            rows = json.loads(data)["PropertyTable"]["Properties"]
            return requests, rows
        except (HTTPError, URLError, TimeoutError) as error:
            status = error.code if isinstance(error, HTTPError) else str(error)
            requests.append({"status": status, "url": url})
            if status in (400, 403, 404):
                break
    return requests, []


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    roster = names_and_registry(a.sanger)
    receipts, annotations = {}, []
    for row in roster.itertuples(index=False):
        history, response = fetch(row.name)
        receipts[row.drug_id] = {"assay_name": row.name, "requests": history, "response": response}
        valid = []
        for compound in response:
            smiles = compound.get("SMILES", compound.get("IsomericSMILES"))
            key = compound.get("InChIKey")
            identity = structural_identity(smiles, key)
            if identity is not None:
                valid.append((identity, compound["CID"], compound["Title"], key))
        if valid and len({candidate[0][0] for candidate in valid}) == 1:
            chosen = sorted(valid, key=lambda x: x[1])[0]
            key, family, scaffold, smiles = chosen[0]
            annotations.append({"drug_id": row.drug_id, "name": row.name, "id": key,
                                "connectivity": family, "scaffold": scaffold, "smiles": smiles,
                                "pubchem_cid": chosen[1], "pubchem_title": chosen[2],
                                "published_inchikey": chosen[3]})
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        staged = Path(directory)
        pd.DataFrame(annotations).to_csv(staged / "sanger_structures.tsv", sep="\t", index=False)
        (staged / "http.json").write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
        (staged / "receipt.json").write_text(json.dumps({
            "source": PUG, "inputs": {path.name: record(path) for path in a.sanger},
            "n_assay_ids": len(roster), "molecularly_resolved": len(annotations),
            "unresolved": sorted(set(roster.name) - {x["name"] for x in annotations}),
            "identity_rule": "Exact Sanger library drug name queried in official PubChem PUG REST",
            "files": {p.name: record(p) for p in sorted(staged.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        staged.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sanger", nargs="+", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
