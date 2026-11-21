"""Summarize public HCC registration metadata without opening expression data."""

import argparse
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from src.r11_chemical_probe.common import sha256, write_json

GEO = ("GSE151412", "GSE186280", "GSE242333", "GSE242340", "GSE291833")
SDRF = "https://www.ebi.ac.uk/biostudies/files/E-MTAB-7847/E-MTAB-7847.sdrf.txt"
MAX_BYTES = 2_000_000


def metadata(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=20) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Public record exceeds the metadata-only byte limit")
    text = raw.decode("utf-8-sig")
    if any("table_begin" in row.lower() for row in text.splitlines()):
        raise ValueError("Metadata result contains a sample expression table")
    return text, hashlib.sha256(raw).hexdigest()


def geo_samples(accession):
    stem = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    url = f"{stem}?acc={accession}&targ=gsm&form=text&view=brief"
    text, digest = metadata(url)
    sample, rows = None, []
    for line in text.splitlines():
        if line.startswith("^SAMPLE = GSM"):
            sample = dict(study=accession, sample=line.split(" = ", 1)[1], title="", source="")
            rows.append(sample)
        elif sample is not None and line.startswith("!Sample_title = "):
            sample["title"] = line.split(" = ", 1)[1]
        elif sample is not None and line.startswith("!Sample_source_name_ch1 = "):
            sample["source"] = line.split(" = ", 1)[1]
    if not rows or any(not row["title"] for row in rows) or len(rows) != len({row["sample"] for row in rows}):
        raise ValueError(f"{accession} returned incomplete or ambiguous sample metadata")
    series_url = f"{stem}?acc={accession}&targ=self&form=text&view=brief"
    series, series_digest = metadata(series_url)
    listed = {line.split(" = ", 1)[1] for line in series.splitlines() if line.startswith("!Series_sample_id = ")}
    recovered = {row["sample"] for row in rows}
    if not listed or not recovered <= listed:
        raise ValueError(f"{accession} source and sample inventories disagree")
    evidence = dict(sample_url=url, sample_sha256=digest, series_url=series_url, series_sha256=series_digest,
                    declared_samples=len(listed), recovered_samples=len(recovered), missing_ids=sorted(listed-recovered))
    return accession, rows, evidence


def ebi_samples():
    text, digest = metadata(SDRF)
    out, source = {}, {}
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        key = tuple(row[f"Factor Value[{tag}]"].strip() for tag in ("cell line", "compound", "dose"))
        name = row["Source Name"]
        if not name or key[:2] != (row["Characteristics[cell line]"], row["Factor Value[compound]"]):
            raise ValueError("Source and labeled cell contexts differ")
        if name in source and source[name] != key:
            raise ValueError("Sequencing entries assign one source to different interventions")
        source[name] = key
        out.setdefault(key, []).append(name)
    if not out:
        raise ValueError("The E-MTAB-7847 SDRF contains no enrolled conditions")
    records = [dict(study="E-MTAB-7847", cell_line=k[0], compound=k[1], reported_dose=k[2] or "NA",
        metadata_rows=len(names), distinct_sources=len(set(names))) for k, names in sorted(out.items())]
    return records, dict(sdrf_url=SDRF, sdrf_sha256=digest, distinct_sources=len(source))


def write_table(path, rows):
    with path.open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_evidence(accession):
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&targ=self&form=text&view=full"
    text, digest = metadata(url)
    fields = ("!Sample_source_name_ch1", "!Sample_characteristics_ch1", "!Sample_treatment_protocol_ch1",
              "!Sample_growth_protocol_ch1", "!Sample_library_strategy", "!Sample_data_processing")
    if f"^SAMPLE = {accession}" not in text.splitlines():
        raise ValueError(f"{accession} did not return the exact individual record")
    rows = []
    for line in text.splitlines():
        if line.startswith(fields):
            key, value = line.split(" = ", 1)
            rows.append(dict(sample=accession, field=key.removeprefix("!Sample_"), description=value))
    return accession, rows, dict(url=url, sha256=digest)


def run(out, studies=GEO, individual=()):
    if out.exists():
        raise FileExistsError(out)
    with ThreadPoolExecutor(max_workers=3) as pool:
        resources = list(pool.map(geo_samples, studies))
        samples = list(pool.map(sample_evidence, individual))
    ebi, sdrf = ebi_samples()
    out.mkdir(parents=True)
    write_table(out/"geo_sample_metadata.tsv", [row for _, rows, _ in resources for row in rows])
    write_table(out/"ebi_source_counts.tsv", ebi)
    if samples:
        write_table(out/"individual_metadata.tsv", [row for _, rows, _ in samples for row in rows])
    write_json(out/"receipt.json", dict(metadata_only=True, expression_read=False, ebi=sdrf,
        individual={sample: evidence for sample, _, evidence in samples},
        checked_at=datetime.now(timezone.utc).isoformat(), geo={study: body for study, _, body in resources},
        code_sha256=sha256(__file__), files={p.name: sha256(p) for p in out.iterdir()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geo", nargs="+", default=GEO)
    parser.add_argument("--sample", nargs="+", default=())
    args = parser.parse_args()
    run(args.out, args.geo, args.sample)
