"""Fetch the original DrugCombDB public files with byte-range receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

from .splits import record


ROOT = "http://drugcombdb.denglab.org/download/"
FILES = {
    "drugcombs_scored.csv": (32_289_774, 'W/"32289774-1695913610000"'),
    "drugcombs_response.csv": (685_920_301, 'W/"685920301-1695913610000"'),
    "drug_chemical_info.csv": (297_674, 'W/"297674-1695913609000"'),
}
CHUNK_SIZE = 32 << 20


def range_download(url, file, start, stop, size, etag):
    tag = f"bytes {start}-{stop - 1}/{size}"
    evidence = {}
    for attempt in range(4):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{stop - 1}", "Accept-Encoding": "identity"})
            with urlopen(request, timeout=55) as response:
                observed = response.headers.get("Content-Range")
                if response.status != 206 or observed != tag or response.headers.get("ETag") != etag:
                    raise ValueError(f"Source bytes changed: {response.status} {observed}")
                with file.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1 << 20)
                evidence = {"url": response.url, "http_status": response.status, "range": observed,
                            "etag": etag, "last_modified": response.headers.get("Last-Modified")}
            if file.stat().st_size != stop - start:
                raise IOError(f"Incomplete byte range {start}:{stop}")
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return {**evidence, **record(file)}


def ranges(root, name, workers):
    size, etag = FILES[name]
    target = root / name
    if target.is_file():
        if target.stat().st_size != size:
            raise ValueError(f"{target} should have {size:,} bytes")
        return {"url": ROOT + name, "etag": etag, "monolithic_file": record(target)}
    work = root / (name + ".ranges")
    work.mkdir(exist_ok=True)
    order = [(start, min(start + CHUNK_SIZE, size)) for start in range(0, size, CHUNK_SIZE)]

    def fetch(block):
        start, stop = block
        file = work / f"{start}-{stop}.part"
        progress = file.with_suffix(".json")
        if progress.is_file() and file.is_file():
            saved = json.loads(progress.read_text())
            if saved["etag"] == etag and saved["sha256"] == record(file)["sha256"]:
                return file, saved
        data = range_download(ROOT + name, file, start, stop, size, etag)
        progress.write_text(json.dumps(data, sort_keys=True) + "\n")
        print(name, data["range"], flush=True)
        return file, data

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pieces = list(pool.map(fetch, order))
    staged = target.with_suffix(".assembled")
    with staged.open("wb") as out:
        for block, _ in pieces:
            with block.open("rb") as source:
                shutil.copyfileobj(source, out, length=1 << 20)
    if staged.stat().st_size != size:
        raise ValueError("Combined range content has the wrong length")
    if target.exists():
        if record(staged) != record(target):
            raise ValueError("Ranged and monolithic downloads disagree")
        staged.unlink()
    else:
        staged.rename(target)
    return {"url": ROOT + name, "etag": etag, "file": record(target), "ranges": [data for _, data in pieces]}


def prepare(a):
    a.output.mkdir(parents=True, exist_ok=True)
    acquired = {}
    for name in a.files:
        acquired[name] = ranges(a.output, name, a.workers)
    path = a.output / ("receipt-" + "-".join(a.files) + ".json")
    path.write_text(json.dumps(acquired, indent=2, sort_keys=True) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--files", choices=tuple(FILES), nargs="+", default=list(FILES))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 6:
        p.error("Use between one and six HTTP range workers")
    prepare(a)


if __name__ == "__main__":
    main()
