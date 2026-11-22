"""Snapshot public ChEMBL drug mechanisms for registered public compounds."""
import argparse
import json
from pathlib import Path
import tempfile
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

from .perturbseq import record
from .predict import write_json


BASE = "https://www.ebi.ac.uk/chembl/api/data/"


def fetch(url, log):
    for retry in range(3):
        try:
            request = Request(url, headers={"Accept": "application/json",
                                           "User-Agent": "public-hcc-benchmark/1 (ChEMBL registry audit)"})
            with urlopen(request, timeout=35) as response:
                data = response.read()
                log.append({"url": url, "status": response.status, "bytes": len(data)})
            return json.loads(data)
        except (TimeoutError, URLError) as error:
            status = error.code if isinstance(error, HTTPError) else str(error)
            log.append({"url": url, "status": status, "attempt": retry + 1})
            if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
                raise
            sleep(2 ** retry)
    raise RuntimeError(f"Unable to read ChEMBL endpoint: {url}")


def paginated(endpoint, key, field, identities, log):
    items = []
    unique = sorted(set(identities))
    for index in range(0, len(unique), 18):
        chunk = unique[index:index + 18]
        params = {field: ",".join(chunk), "limit": 200, "format": "json"}
        url = BASE + endpoint + ".json?" + urlencode(params)
        while url:
            payload = fetch(url, log)
            items.extend(payload[key])
            following = payload["page_meta"]["next"]
            if following:
                parsed = urlsplit(following)
                if parsed.netloc not in ("", "www.ebi.ac.uk") or not parsed.path.startswith("/chembl/api/data/"):
                    raise ValueError("Unregistered ChEMBL pagination URL")
                url = "https://www.ebi.ac.uk" + parsed.path + "?" + parsed.query
            else:
                url = None
            sleep(0.3)
    return items


def snapshot(registry, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compounds = pd.read_csv(registry, sep="\t", dtype=str)
    ids = compounds.compound_id.dropna().unique()
    log = []
    # Preserve the chronological read and the returned release identifier.
    try:
        status = fetch(BASE + "status.json", log)
        molecules = paginated("molecule", "molecules", "molecule_structures__standard_inchi_key__in", ids, log)
        chemical_ids = sorted({(m.get("molecule_hierarchy") or {}).get("parent_chembl_id")
                               or m["molecule_chembl_id"] for m in molecules})
        mechanisms = paginated("mechanism", "mechanisms", "parent_molecule_chembl_id__in", chemical_ids, log)
        targets = paginated("target", "targets", "target_chembl_id__in",
                            [x["target_chembl_id"] for x in mechanisms if x.get("target_chembl_id")], log)
    except Exception:
        write_json(output.with_name(output.name + "-http-failure.json"), log)
        raise
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as directory:
        staged = Path(directory)
        for name, content in (("status", status), ("molecules", molecules), ("mechanisms", mechanisms),
                              ("targets", targets), ("http", log)):
            write_json(staged / (name + ".json"), content)
        write_json(staged / "receipt.json", {
            "source": BASE, "release": status["chembl_db_version"], "registry": record(registry),
            "requested_compounds": len(ids), "exact_structure_matches": len(molecules),
            "mechanisms": len(mechanisms), "target_records": len(targets),
            "files": {p.name: record(p) for p in sorted(staged.iterdir())},
        })
        staged.rename(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    snapshot(a.registry, a.output)


if __name__ == "__main__":
    main()
