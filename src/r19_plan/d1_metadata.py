"""Register the D1 equal-count R17 plate controls without reading expression."""

import argparse
from pathlib import Path
from zlib import crc32

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r19_plan.r18_preflight import GEO, ROLES, check_controls


def balance(controls):
    check_controls(controls.assign(study=GEO))
    counts = controls.groupby(["rna_plate", "role"]).size().unstack().reindex(columns=list(ROLES))
    counts["per_role"] = counts.min(axis=1).astype(int)
    ordered = controls.assign(crc=controls.inst_id.map(lambda key: crc32(key.encode())))
    ordered = ordered.sort_values(["rna_plate", "role", "crc", "inst_id"])
    slot = ordered.groupby(["rna_plate", "role"]).cumcount()
    picked = ordered.loc[slot.lt(ordered.rna_plate.map(counts.per_role))].drop(columns="crc")
    observed = picked.groupby(["rna_plate", "role"]).size().unstack()
    if not observed.eq(counts.per_role, axis=0).all().all():
        raise ValueError("D1 fit/select/score comparison would have unequal control counts")
    return picked, counts


def run(assignments, reference, out):
    if sha256(assignments) != read_json(reference)["files"]["assignments.tsv"]:
        raise ValueError("Original R17 assignment axis differs from its pinned receipt")
    controls = table(assignments)
    picked, counts = balance(controls)
    rotations = [
        {"fit": fit, "shared_select": fit, "independent_select": ROLES[(i + 1) % 3],
         "score": ROLES[(i + 2) % 3]}
        for i, fit in enumerate(ROLES)
    ]
    out.mkdir(exist_ok=False, parents=True)
    picked.to_csv(out / "assignments.tsv", sep="\t", index=False)
    counts.reset_index().to_csv(out / "plate_support.tsv", sep="\t", index=False)
    pd.DataFrame(rotations).to_csv(out / "rotations.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(code_sha256=sha256(__file__), input_sha256=sha256(assignments),
        metadata_only=True, D1_executed=False, original_wells=len(controls), kept_wells=len(picked),
        plates=len(counts), minimum_per_role=int(counts.per_role.min()),
        files={path.name: sha256(path) for path in sorted(out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("assignments", "reference", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(**vars(parser.parse_args()))
