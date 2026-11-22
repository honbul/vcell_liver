"""Register public biopsy and donor identities before single-cell analysis."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile

import pandas as pd


NCI_TYPES = {
    "malignant cell": "malignant", "malignant cells": "malignant",
    "t cell": "t_nk", "t cells": "t_nk", "tam": "myeloid", "tams": "myeloid",
    "tec": "endothelial", "tecs": "endothelial", "b cell": "b_cell", "b cells": "b_cell",
    "caf": "fibroblast", "cafs": "fibroblast", "hpc-like": "progenitor_candidate",
    "unclassified": "other",
}
LIVER_TYPES = {
    "t/nk": "t_nk", "hepatocyte": "hepatocyte", "myeloid": "myeloid",
    "b": "b_cell", "endothelial": "endothelial", "fibroblast": "fibroblast",
}


def file_record(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(data)
    return {"bytes": path.stat().st_size, "sha256": checksum.hexdigest()}


def source(path, columns):
    metadata = pd.read_csv(path, sep="\t", dtype=str, usecols=columns)
    for name in columns:
        if metadata[name].isna().any() or metadata[name].str.strip().eq("").any():
            raise ValueError(f"Missing {name} in {path}")
        metadata[name] = metadata[name].str.strip()
    return metadata


def prep_liver(path):
    table = source(path, ["Cell", "sample", "site", "patient", "celltype"])
    result = pd.DataFrame({
        "id": "GSE149614:" + table.Cell, "barcode": table.Cell, "cohort": "GSE149614",
        "source_sample": table["sample"], "donor": "GSE149614:" + table.patient,
        "site": table.site, "histology": "HCC",
        "author_type": table.celltype, "category": table.celltype.str.lower().map(LIVER_TYPES),
    })
    result["malignancy"] = "non_epithelial"
    result.loc[result.category.eq("hepatocyte") & result.site.eq("Normal"), "malignancy"] = "normal_hepatocyte"
    result.loc[result.category.eq("hepatocyte") & ~result.site.eq("Normal"), "malignancy"] = "needs_cnv"
    return result


def normalized_nci_donor(patient):
    # Published author IDs encode a common NCI patient number and prefix across
    # the earlier and longitudinal releases. Biopsies carry optional a/b/c.
    match = re.fullmatch(r"(?:GSE151530_|GSE125449_)?([HC]\d+)[A-Za-z]?", patient)
    if match is None:
        raise ValueError(f"Unregistered NCI donor identifier {patient}")
    return "NCI:" + match[1]


def prep_longitudinal(path, manifest):
    table = source(path, ["Cell", "Sample", "S_ID", "Type"])
    crosswalk = source(manifest, ["S_ID", "author_sample", "patient", "histology"])
    matched = table.merge(crosswalk, on="S_ID", validate="many_to_one", how="left")
    if (matched.patient.isna().any() or (matched.Sample != matched.author_sample).any()):
        raise ValueError("Longitudinal GEO sample metadata disagree with the author crosswalk")
    result = pd.DataFrame({
        "id": "GSE151530:" + matched.S_ID + ":" + matched.Cell,
        "barcode": matched.Cell, "cohort": "GSE151530", "source_sample": matched.Sample,
        "donor": matched.patient.map(normalized_nci_donor), "site": "biopsy",
        "histology": matched.histology, "author_type": matched.Type,
        "category": matched.Type.str.lower().map(NCI_TYPES),
    })
    return nci_malignancy(result)


def prep_companion(path, set_no, manifest):
    table = source(path, ["Cell Barcode", "Sample", "Type"])
    # Explicit crosswalk for the earlier public atlas from its GEO and paper
    # metadata. Shared donor IDs use the same key as the longitudinal atlas.
    crosswalk = source(manifest, ["author_sample", "patient", "histology"])
    lookup = table.merge(crosswalk, left_on="Sample", right_on="author_sample",
                         how="left", validate="many_to_one")
    if lookup.patient.isna().any():
        raise ValueError("No documented author donor ID for an earlier NCI biopsy")
    return nci_malignancy(pd.DataFrame({
        "id": f"GSE125449:set{set_no}:" + lookup["Sample"] + ":" + lookup["Cell Barcode"],
        "barcode": lookup["Cell Barcode"], "cohort": "GSE125449",
        "source_sample": lookup["Sample"], "donor": lookup.patient.map(normalized_nci_donor),
        "site": "biopsy", "histology": lookup.histology, "author_type": lookup.Type,
        "category": lookup.Type.str.lower().map(NCI_TYPES),
    }))


def nci_malignancy(table):
    status = {"malignant": "author_malignant", "progenitor_candidate": "needs_cnv"}
    return table.assign(malignancy=table.category.map(status).fillna("non_epithelial"))


def folds(donors, seed, n_folds):
    unique = sorted(set(donors))
    if len(unique) < n_folds or n_folds < 2:
        raise ValueError("Insufficient independent donors")
    by_study = {}
    for key in unique:
        study = key.split(":", 1)[0]
        by_study.setdefault(study, []).append(key)
    assigned = {}
    for values in by_study.values():
        keys = sorted(values, key=lambda d: sha256(f"track5:{seed}:{d}".encode()).digest())
        assigned.update({key: i % n_folds for i, key in enumerate(keys)})
    return assigned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gse149614", type=Path, required=True)
    p.add_argument("--gse151530", type=Path, required=True)
    p.add_argument("--longitudinal-crosswalk", type=Path, required=True)
    p.add_argument("--gse125449", type=Path, required=True, nargs=2)
    p.add_argument("--companion-crosswalk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=51122)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    liver = prep_liver(a.gse149614)
    later = prep_longitudinal(a.gse151530, a.longitudinal_crosswalk)
    earlier = [prep_companion(f, i + 1, a.companion_crosswalk) for i, f in enumerate(a.gse125449)]
    table = pd.concat([liver, later, *earlier], ignore_index=True)
    if table.id.duplicated().any() or table.category.isna().any():
        raise ValueError("Unregistered lineage category or duplicate assay cell ID")
    table["fold"] = table.donor.map(folds(table.donor, a.seed, a.folds))
    for i in range(a.folds):
        test = set(table.loc[table.fold.eq(i), "donor"])
        train = set(table.loc[~table.fold.eq(i), "donor"])
        if test & train or not test or not train:
            raise ValueError(f"Invalid donor holdout {i}")
    receipt = {
        "inputs": {f.name: file_record(f) for f in [a.gse149614, a.gse151530, a.longitudinal_crosswalk,
                                                    *a.gse125449, a.companion_crosswalk]},
        "donors": int(table.donor.nunique()),
        "shared_nci_donors": len(set(later.donor) & set(pd.concat(earlier).donor)),
        "registered_cells": len(table), "folds": a.folds, "seed": a.seed,
        "cohort_by_malignancy": [
            {"cohort": cohort, "malignancy": role, "cells": int(n)}
            for (cohort, role), n in table.groupby(["cohort", "malignancy"]).size().items()
        ],
    }
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as temporary:
        stage = Path(temporary)
        table.to_parquet(stage / "cohorts.parquet", index=False)
        receipt["registry"] = file_record(stage / "cohorts.parquet")
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


if __name__ == "__main__":
    main()
