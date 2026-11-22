"""Check GSE151530 biopsy timing against the authors' public numeric-ID map."""
import argparse
import json
from pathlib import Path
import re
import tempfile

import pandas as pd

from .cohorts import file_record
from .singlecell_inputs import files_receipt


def register(registry, crosswalk, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    roles = pd.read_parquet(registry)
    biopsies = roles[roles.cohort.eq("GSE151530") & roles.histology.eq("HCC")][
        ["id", "source_sample", "donor"]
    ].copy()
    biopsies["nci_id"] = biopsies.id.str.extract(r"^GSE151530:S([0-9]+):")[0].astype(int)
    biopsies = biopsies.drop(columns="id").drop_duplicates()
    published = pd.read_csv(crosswalk, sep="\t", dtype={"id": int, "sample": str})
    if not published.id.is_unique or not published["sample"].is_unique:
        raise ValueError("Author biopsy map contains duplicate identities")
    rows = []
    labels = biopsies.merge(published, left_on="nci_id", right_on="id", validate="1:1")
    for row in labels.itertuples(index=False):
        match = re.fullmatch(r"([HC][0-9]+)-t([0-9]+)(?:-[LR])?", row.sample)
        if not match or row.donor != f"NCI:{match[1]}":
            raise ValueError(f"Longitudinal biopsy identity differs for {row.source_sample}")
        rows.append({"donor": row.donor, "source_sample": row.source_sample, "biopsy_id": row.nci_id,
                     "public_sample": row.sample, "weeks": int(match[2])})
    frame = pd.DataFrame(rows).sort_values(["donor", "weeks"])
    parts = []
    for _, donor in frame.groupby("donor"):
        if len(donor) >= 2:
            if donor.weeks.iloc[0] != 0 or donor.weeks.duplicated().any():
                raise ValueError("No independent baseline or a repeated biopsy time")
            parts.append(donor.assign(period=["baseline", "first_followup", *(["later_followup"] * (len(donor) - 2))]))
    if len(parts) != 5:
        raise ValueError("Expected five GSE151530 HCC donors with mapped baseline and follow-up")
    together = pd.concat(parts)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as d:
        staged = Path(d)
        together.to_csv(staged / "nci_timepoints.tsv", sep="\t", index=False)
        (staged / "receipt.json").write_text(json.dumps({
            "clinical_metadata": file_record(crosswalk), "registry": file_record(registry),
            "evidence": "Public GSE229772 sample map joins to GSE151530 S_ID by numeric biopsy ID and independent donor identifier",
            "paired_hcc_donors": len(parts), "biopsies": len(together),
            "files": files_receipt(staged),
        }, indent=2, sort_keys=True) + "\n")
        staged.rename(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "crosswalk", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    register(a.registry, a.crosswalk, a.output)


if __name__ == "__main__":
    main()
