"""Freeze FDA release membership and audit exact-name PubChem structure lookups."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

from .chemistry import identity


RELEASE_SIZE = {1: 1036, 2: 1336}
CONCERN = {"mostdiliconcern": "Most", "lessdiliconcern": "Less",
           "nodiliconcern": "No", "ambiguousdiliconcern": "Ambiguous"}


def read_releases(path):
    sheets = {}
    for version, sheet in ((1, 1), (2, 0)):
        table = pd.read_excel(path, sheet_name=sheet, header=1, usecols="A:F")
        table.columns = ["fda_id", "name", "severity", "label_section", "concern", "comment"]
        table = table.loc[table.fda_id.astype(str).str.fullmatch(r"LT\d+")].copy()
        table["concern"] = (table.concern.str.lower().str.replace(r"[^a-z]+", "", regex=True)
                            .str.removeprefix("v").map(CONCERN))
        if len(table) != RELEASE_SIZE[version] or table.fda_id.duplicated().any() or table.concern.isna().any():
            raise ValueError(f"FDA release {version} schema or support changed")
        sheets[version] = table.set_index("fda_id")
    v1, v2 = sheets[1], sheets[2]
    v2 = v2.join(v1[["concern"]].rename(columns={"concern": "v1_concern"}))
    v2["release_increment"] = v2.v1_concern.isna()
    if v2.release_increment.sum() != 300 or not (v2.release_increment == v2.comment.eq("New")).all():
        raise ValueError("FDA New annotations disagree with v1-to-v2 ID membership")
    return v2


class PubChem:
    def __init__(self, cache):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.next_start = 0.0

    def request(self, name):
        key = sha256(name.lower().encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{quote(name, safe='')}/property/InChIKey,SMILES/JSON")
        for trial in range(5):
            with self.lock:
                pause = self.next_start - monotonic()
                if pause > 0:
                    sleep(pause)
                self.next_start = monotonic() + 0.29
            retry_seconds = 2**trial
            try:
                with urlopen(url, timeout=20) as response:
                    data = json.load(response)
                break
            except HTTPError as error:
                if error.code == 404:
                    data = {"PropertyTable": {"Properties": []}}
                    break
                if error.code not in (429, 500, 502, 503, 504):
                    raise
                retry_seconds = max(retry_seconds, int(error.headers.get("Retry-After", "0")))
            except (TimeoutError, URLError, json.JSONDecodeError):
                pass
            sleep(retry_seconds)
        else:
            raise RuntimeError(f"PubChem did not resolve the request for {name!r}; retry with the same cache")
        path.write_text(json.dumps(data, sort_keys=True) + "\n")
        return data

    def structure(self, name):
        data = self.request(name)["PropertyTable"]["Properties"]
        structures = {}
        for record in sorted(data, key=lambda x: x["CID"]):
            smiles = record.get("SMILES") or record.get("IsomericSMILES")
            if not smiles:
                continue
            try:
                parent, key, connectivity, scaffold = identity(smiles)
            except (ValueError, RuntimeError):
                continue
            structures.setdefault(key, dict(pubchem_cid=record["CID"], supplied_smiles=smiles,
                supplied_inchikey=record["InChIKey"], smiles=parent, inchikey=key,
                connectivity=connectivity, scaffold=scaffold))
        if len(structures) == 1:
            return dict(status="resolved", **next(iter(structures.values())))
        return dict(status="multiple_stereo_resolved_parents" if structures else "no_structure")


def acquire(workbook, output, extra_names):
    labels = read_releases(workbook)
    lookup = PubChem(output / "pubchem_cache")
    names = set(labels.name)
    for source in extra_names:
        names.update(line.strip() for line in source.read_text().splitlines() if line.strip())
    names = sorted(names)
    with ThreadPoolExecutor(max_workers=6) as workers:
        structures = pd.DataFrame(workers.map(lookup.structure, names), index=pd.Index(names, name="name"))
    structures.to_csv(output / "structures.tsv", sep="\t")
    rows = labels.join(structures, on="name")
    rows.to_csv(output / "fda_structures.tsv", sep="\t")
    paths = sorted([workbook, *extra_names, *output.glob("*.tsv"), *lookup.cache.glob("*.json")])
    receipt = {
        "sha256": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
        "counts": rows.groupby(["release_increment", "concern", "status"]).size().rename("n").reset_index().to_dict("records"),
        "query": "PubChem PUG REST exact name, no fuzzy fallback",
    }
    (output / "fda_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--names", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    acquire(args.workbook, args.output, args.names)
