"""Audit public LICOB cohort keys and the RNA archive without emitting measurements."""

import argparse
import hashlib
import mmap
from pathlib import Path
import re
from xml.etree import ElementTree
import zipfile
import zlib

from openpyxl import load_workbook

from .io import publish_json, sha256

LICOB_RNA = "xl/worksheets/sheet8.xml"
PAIRED_RNA = "xl/worksheets/sheet3.xml"


def archive_check(path):
    errors = {}
    with zipfile.ZipFile(path) as archive:
        for sheet in (LICOB_RNA, PAIRED_RNA):
            try:
                with archive.open(sheet) as stream:
                    while stream.read(1024 * 1024):
                        pass
            except (zipfile.BadZipFile, zlib.error) as exc:
                errors[sheet] = str(exc)
        with archive.open("xl/sharedStrings.xml") as stream:
            strings = ["".join(item.itertext()) for item in ElementTree.parse(stream).getroot()]
        with archive.open(LICOB_RNA) as stream:
            prefix = stream.read(16384)
        header_rows = re.findall(rb"<row\s.*?</row>", prefix)
        if len(header_rows) < 2:
            raise ValueError("LICOB RNA identity header is unavailable")
        pattern = rb'<c[^>]* t="s"[^>]*>\s*<v>(\d+)</v>'
        title = [strings[int(i)] for i in re.findall(pattern, header_rows[0])]
        ids = [strings[int(i)] for i in re.findall(pattern, header_rows[1])][1:]
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        hole = re.search(rb'\000{1048576,}', data)
        zero_region = {"start": hole.start(), "end_exclusive": hole.end(), "length": len(hole[0])} if hole else None
    return title, ids, errors, zero_region


def receipt(archive, checksum, supplement):
    with archive.open("rb") as file:
        digest = hashlib.md5()
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    actual_md5 = digest.hexdigest()
    expected = checksum.read_text().strip().split()[0].lower()
    title, ids, errors, zero_region = archive_check(archive)

    supplements = load_workbook(supplement, read_only=True, data_only=True)
    cases = {row[0]: row[5] for row in supplements["S1. Clinical information"].iter_rows(
        min_row=4, values_only=True,
    ) if isinstance(row[0], str) and isinstance(row[5], str)}
    hcc = {model for model, histology in cases.items() if histology == "HCC"}
    by_endpoint = {}
    for name in ("S6. AUC", "S7. IC50", "S8. Emax"):
        rows = supplements[name].iter_rows(min_row=2, values_only=True)
        drugs = next(rows)[1:]
        matrix = [row for row in rows if row[0] in hcc]
        models = {row[0] for row in matrix}
        if len(matrix) != len(models):
            raise ValueError(f"Duplicate public organoid labels in {name}")
        by_endpoint[name] = {
            "hcc_model_count": len(models), "drugs": len(drugs), "hcc_missing_ids": sorted(hcc - models),
            "numeric_hcc_measurements": sum(isinstance(number, (float, int)) for row in matrix for number in row[1:]),
            "matching_rna_headers": len(models & set(ids)),
        }
    return {
        "archive": {"size": archive.stat().st_size, "expected_md5": expected, "actual_md5": actual_md5,
                    "sha256": sha256(archive), "verified": expected == actual_md5 and not errors,
                    "rna_integrity_errors": errors, "zero_region": zero_region},
        "supplement": {"size": supplement.stat().st_size, "sha256": sha256(supplement),
                       "accession": "EMBL-EBI BioStudies S-EPMC10949980"},
        "registered_model_count": len(cases), "registered_hcc_models": len(hcc),
        "header": {"total_models": len(ids), "matching_hcc": len(hcc & set(ids)),
                   "rna_description": title},
        "response": by_endpoint,
        "eligible_for_scoring": expected == actual_md5 and not errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "checksum", "supplement", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    publish_json(args.out, receipt(args.archive, args.checksum, args.supplement))


if __name__ == "__main__":
    main()
