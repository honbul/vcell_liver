"""Verify shared NCI donors using the deposited GEO biopsy titles and histology."""
import argparse
import csv
import gzip
from pathlib import Path
import re

import pandas as pd

from .cohorts import normalized_nci_donor, source

HISTOLOGY = {"hepatocellular carcinoma": ("H", "HCC"),
             "intrahepatic cholangiocarcinoma": ("C", "iCCA")}


def document_rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    values = {row[0]: row[1:] for row in rows
              if row and row[0] in {"!Sample_title", "!Sample_geo_accession"}}
    disease = next(row[1:] for row in rows
                   if row and row[0] == "!Sample_characteristics_ch1"
                   and all(v.strip().lower().startswith("cancer type:") for v in row[1:]))
    return zip(values["!Sample_title"], values["!Sample_geo_accession"], disease, strict=True)


def get_crosswalk(current, early):
    later = source(current, ["patient", "histology"])
    later["donor"] = later.patient.map(normalized_nci_donor)
    shared = later[["donor", "histology"]].drop_duplicates()
    if shared.donor.duplicated().any():
        raise ValueError("Conflicting NCI donor histology")
    keys = shared.set_index("donor").histology.to_dict()
    rows = []
    for path in early:
        for title, gsm, condition in document_rows(path):
            match = re.fullmatch(r"S[0-9]+_P[0-9]+_LCP([0-9]+)", title)
            if match is None:
                raise ValueError(f"Unexpected biopsy title: {title}")
            prefix, cancer = HISTOLOGY[condition.split(":", 1)[1].strip().lower()]
            alias = prefix + match[1].zfill(2)
            key = normalized_nci_donor(alias)
            if keys.get(key) != cancer:
                raise ValueError(f"Earlier biopsy fails released donor and histology correspondence: {title}")
            rows.append({"author_sample": title, "patient": alias, "histology": cancer,
                         "geo_sample": gsm, "source_matrix": path.name})
    frame = pd.DataFrame(rows)
    if frame.author_sample.duplicated().any() or frame.geo_sample.duplicated().any():
        raise ValueError("Repeated GEO sample identity in companion atlas")
    return frame.sort_values("author_sample")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--longitudinal-crosswalk", type=Path, required=True)
    p.add_argument("--companion-matrices", type=Path, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = get_crosswalk(a.longitudinal_crosswalk, a.companion_matrices)
    with a.output.open("x") as stream:
        result.to_csv(stream, sep="\t", index=False)


if __name__ == "__main__":
    main()
