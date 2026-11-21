"""Explain why the fixed Tahoe dose-transfer support gate did not pass."""

import argparse
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, table
from src.r19_plan.hcc_cross_dose_donor import (
    PLATE, POSITION, WELL, eligible_donor_wells, inputs, linked_wells, partition, projected_line,
)
from src.r19_plan.hcc_scaffold_refit import require_hash


def counts(low, high, links):
    cols = [GROUP, ENTITY, WELL, PLATE, POSITION]
    target = projected_line(high)[cols]
    candidates = eligible_donor_wells(low)[cols]
    exact = target.merge(low[cols].drop_duplicates(), on=[GROUP, ENTITY], suffixes=("", "_donor"))
    eligible = exact.loc[exact[WELL + "_donor"].isin(candidates[WELL])]
    different_plate = eligible.loc[eligible[PLATE].ne(eligible[PLATE + "_donor"])
                                    & eligible[WELL].ne(eligible[WELL + "_donor"])]
    different_position = different_plate.loc[different_plate[POSITION].ne(different_plate[POSITION + "_donor"])]
    donor = different_position.sort_values(WELL + "_donor").drop_duplicates(WELL)
    decoys = candidates[[GROUP, WELL, PLATE, POSITION]].rename(columns={
        GROUP: "decoy_parent", WELL: "decoy_id", PLATE: PLATE + "_donor", POSITION: "decoy_position",
    })
    pairs = donor.merge(decoys, on=PLATE + "_donor", validate="many_to_many")
    paired = pairs.loc[pairs.decoy_parent.ne(pairs[GROUP]) & pairs.decoy_position.ne(pairs[POSITION])
                       & pairs.decoy_id.ne(pairs[WELL])]
    if not set(links[WELL]) <= set(paired[WELL]):
        raise ValueError("The independent diagnostic cannot reproduce the registered links")
    steps = [("all_target", target), ("exact_entity", exact), ("donor_eligible", eligible),
             ("different_plate", different_plate), ("different_position", different_position),
             ("same_donor_plate_decoy", paired), ("minimum_parent_pool", links)]
    return [dict(step=name, parent_groups=frame[GROUP].nunique(), target_wells=frame[WELL].nunique())
            for name, frame in steps]


def run(args):
    stamp, proof = inputs(args)
    audit = read_json(args.audit / "receipt.json")
    if any(audit.get(key) != expected for key, expected in proof.items()) or audit["eligible"]:
        raise ValueError("Diagnosis requires the original metadata-only support failure")
    links = table(require_hash(args.audit / "links.tsv", audit["files"]["links.tsv"]))
    result = []
    for fold in range(3):
        for split in ("fit", "score"):
            low, high = partition(args, stamp, fold, split)
            prior = links.loc[links.fold.eq(fold) & links.split.eq(split)].drop(columns=["fold", "split"])
            latest = linked_wells(low, high)
            pd.testing.assert_frame_equal(latest.reset_index(drop=True), prior.reset_index(drop=True), check_dtype=False)
            result.extend(dict(fold=fold, split=split, **row) for row in counts(low, high, latest))
    args.out.mkdir(exist_ok=False)
    output = args.out / "support_funnel.tsv"
    pd.DataFrame(result).to_csv(output, sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(
        proof, diagnosis_code_sha256=sha256(__file__), audit_sha256=sha256(args.audit / "receipt.json"),
        expression_read=False, files={output.name: sha256(output)}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "archive", "run", "protocol", "audit", "out"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    run(cli.parse_args())
