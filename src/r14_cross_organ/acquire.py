"""Stage official perturbation releases and write per-file integrity receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


NCBI = "https://ftp.ncbi.nlm.nih.gov/geo/series"
SCIPLEX_RECEIPT = Path("manifests/r10_dose_composition/raw_data_receipt.json")
MIX_RELEASE = "https://api.figshare.com/v2/articles/10298696/versions/3"


def geo(accession, folder, filename):
    stem = accession[:-3] + "nnn"
    return dict(name=filename, url=f"{NCBI}/{stem}/{accession}/{folder}/{filename}")


def read_json(path):
    return json.loads(path.read_text())


def source_files(source, mix_release):
    if source == "sciplex3":
        return read_json(SCIPLEX_RECEIPT)["files"]
    if source == "heparg_bulk":
        name = "GSE284321"
        names = [f"{name}_metadata.xlsx"] + [f"{name}_heparg_pg{i}_count_data.csv.gz" for i in range(1, 30)]
        return [geo(name, "suppl", x) for x in names]
    if source == "heparg_single_cell":
        name = "GSE299113"
        expression = geo(name, "suppl", f"{name}_sctr-heparg-geo-24h-SRA-PRJNA1256510-submit.h5ad")
        expression["bytes"] = 527518583
        return [geo(name, "soft", f"{name}_family.soft.gz"),
                expression]
    release = read_json(mix_release)
    if release["id"] != 10298696 or release["version"] != 3:
        raise ValueError("Expected Figshare MIX-seq release 3")
    selected = [x for x in release["files"] if x["name"].endswith(("24hr_expt3.zip", "expt10.zip"))
                or x["name"] in {"README.txt", "Supplementary Tables.xlsx"}]
    if len(selected) != 16:
        raise ValueError("MIX-seq experiment files differ from the planned release")
    return [dict(name=x["name"], url=x["download_url"], bytes=x["size"],
                 publisher_md5=x["supplied_md5"]) for x in selected]


def checksums(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def fetch(spec, out, reuse):
    name = spec["name"]
    if Path(name).name != name:
        raise ValueError(f"Invalid released filename {name!r}")
    file = out / name
    partial = out / (name + ".partial")
    reused = None
    if not file.exists():
        reused = next((p / name for p in reuse if (p / name).is_file()), None)
        if reused is not None:
            shutil.copyfile(reused, partial)
        else:
            subprocess.run(["curl", "--location", "--fail", "--silent", "--show-error", "--retry", "3",
                            "--connect-timeout", "15", "--max-time", "1800", "--output", str(partial),
                            spec["url"]], check=True)
    item = file if file.exists() else partial
    actual = item.stat().st_size
    sha, md5 = checksums(item)
    if actual == 0 or ("bytes" in spec and actual != spec["bytes"]):
        raise ValueError(f"Unexpected content length of {name}: {actual}")
    if ("sha256" in spec and sha != spec["sha256"]) or ("publisher_md5" in spec and md5 != spec["publisher_md5"]):
        raise ValueError(f"Published or archived digest mismatch for {name}")
    if name.endswith(".gz") and "sha256" not in spec:
        subprocess.run(["gzip", "-t", str(item)], check=True)
    if item == partial:
        partial.rename(file)
    print(f"verified {name} {actual} bytes", flush=True)
    return dict(spec, bytes=actual, sha256=sha, reused_from=str(reused) if reused else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=("sciplex3", "heparg_bulk", "heparg_single_cell", "mix_seq"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reuse", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / "receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    mix_path = args.out / "figshare_release3.json"
    release_files = [fetch(dict(name=mix_path.name, url=MIX_RELEASE), args.out, [])] if args.source == "mix_seq" else []
    specs = source_files(args.source, mix_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(lambda spec: fetch(spec, args.out, args.reuse), specs))
    with receipt.open("x") as stream:
        json.dump(dict(source=args.source, prepared_utc=datetime.now(timezone.utc).isoformat(),
                       count_values_read=False, files=release_files + files), stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
