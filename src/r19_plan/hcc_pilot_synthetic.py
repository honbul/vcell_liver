"""Exercise the HCC pilot freeze and scorer without any experimental measurements."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_pilot_layout import ROLES
from src.r19_plan.hcc_pilot_score import aligned_layout, availability, lock, prepare, score, table

ARMS = ("synthetic_candidate", "synthetic_VC_1", "synthetic_VC_2")


def build(root, layout, layout_receipt, protocol):
    root.mkdir(exist_ok=False, parents=True)
    rng = np.random.default_rng(260196)
    planned = pd.read_csv(layout, sep="\t")
    manifest = planned[["plate_slot", "well"]].copy()
    manifest["specimen_id"] = "synth/" + planned.plate_slot + "/" + planned.well
    manifest["culture_id"] = "culture/" + planned.culture_block_slot
    manifest["plate_id"] = "plate/" + planned.plate_slot
    treated = planned.kind.eq("treated")
    manifest["canonical_parent"] = ("synthetic/" + planned.parent_slot).where(treated, "")
    manifest["actual_uM"] = np.where(treated, "5", "")
    manifest["actual_hours"] = 24
    manifest["qc_status"] = "PASS"
    manifest["qc_reason"] = ""
    # Lose two controls from one role. This makes the whole test plate unavailable.
    dropped_plate = planned.plate_slot.eq("HUH7-B01")
    failed_vehicle = planned.index[dropped_plate & planned.vehicle_role.eq("A")][:2]
    example = planned.loc[dropped_plate & treated, "parent_slot"].iloc[0]
    missed_repeat = planned.index[planned.cell.eq("HUH7") & ~dropped_plate & planned.parent_slot.eq(example)][:1]
    failed = failed_vehicle.union(missed_repeat)
    manifest.loc[failed, "qc_status"] = "FAIL"
    manifest.loc[failed, "qc_reason"] = "synthetic_library_failure"
    file = root/"realized.tsv"
    manifest.to_csv(file, sep="\t", index=False)
    config = dict(synthetic=True, arms=list(ARMS), candidate=ARMS[0], existing_vc=list(ARMS[1:]),
                  information="A", min_vehicles_per_role=3, qc_protocol_sha256=sha256(protocol),
                  genes=[f"SYNTH-{i:03d}" for i in range(36)])
    write_json(root/"config.json", config)
    rows = aligned_layout(layout, layout_receipt, table(file))
    accepted, controls, targets, paired, count = availability(rows, 3)
    assert len(targets)==539 and int(paired.sum())==538
    genes = np.array(config["genes"])
    profiles = {key: rng.normal(0,.35,len(genes)) for key in
                accepted.loc[accepted.kind.eq("treated"), "canonical_parent"].unique()}
    offsets = {key: rng.normal(0,.45,len(genes)) for key in accepted.plate_id.unique()}
    def baseline(row):
        return profiles.get(row.canonical_parent, np.zeros(len(genes)))
    values = np.stack([offsets[r.plate_id]+baseline(r)+rng.normal(0,.03,len(genes))
                       for r in accepted.itertuples(index=False)])
    np.savez(root/"measurements.npz", values=values, specimen_id=accepted.specimen_id.to_numpy(dtype=str), genes=genes)
    y = np.stack([baseline(r) for r in targets.itertuples(index=False)])
    z = np.stack([y for _ in ROLES])
    np.savez(root/"forecast.npz", specimen_id=targets.specimen_id.to_numpy(dtype=str), genes=genes,
             rotations=np.array(ROLES), **{name:z*k for name,k in zip(ARMS, (.98,.73,.57), strict=True)})
    declared = []
    for arm in ARMS:
        for i, fit_role in enumerate(ROLES):
            for usage, role in (("fit_vehicle",fit_role), ("select_vehicle",ROLES[(i+1)%3])):
                take = controls.loc[controls.vehicle_role.eq(role) & controls.plate_id.isin(targets.plate_id)]
                declared.extend(dict(arm=arm, rotation=fit_role, specimen_id=s, usage=usage) for s in take.specimen_id)
    pd.DataFrame(declared).to_csv(root/"access.tsv", sep="\t", index=False)
    return SimpleNamespace(layout=layout, layout_receipt=layout_receipt, protocol=protocol, config=root/"config.json",
                           realized=file, access=root/"access.tsv", forecast=root/"forecast.npz",
                           measurements=root/"measurements.npz")


def failures(args):
    failures = []
    root = args.config.parent

    def reject(label, exception, fn):
        try:
            fn()
        except ValueError as error:
            if exception not in str(error):
                raise
            failures.append(dict(probe=label, rejected=True))
        else:
            raise AssertionError(f"{label} was incorrectly accepted")

    copy = table(args.realized)
    duplicate = copy.plate_slot.eq("HUH7-B02")
    copy.loc[duplicate, "culture_id"] = copy.loc[copy.plate_slot.eq("HUH7-B01"), "culture_id"].iloc[0]
    reject("shared_culture", "Reused or fragmented actual culture_id",
           lambda: aligned_layout(args.layout, args.layout_receipt, copy))
    copy = table(args.realized)
    copy.loc[copy.canonical_parent.eq("synthetic/P002"), "canonical_parent"] = "synthetic/P001"
    reject("reused_parent_identity", "wrong identity", lambda: aligned_layout(args.layout,args.layout_receipt,copy))
    ledger = table(args.access)
    select = ledger.loc[ledger.rotation.eq("A") & ledger.usage.eq("select_vehicle")].index[0]
    ledger.loc[select, "rotation"] = "C"
    denied = root/"forbidden_access.tsv"
    ledger.to_csv(denied,sep="\t",index=False)
    reject("score_vehicle_as_select", "wrong-role vehicle", lambda: prepare(SimpleNamespace(**dict(vars(args),access=denied))))
    reject("mutated_freeze_hash", "immutable forecast freeze", lambda: score(
           SimpleNamespace(**dict(vars(args),expected_sha256="invalid",measurements=root/"missing.npz"))))
    corrupted = root/"gene_order_mismatch.npz"
    with np.load(args.measurements, allow_pickle=False) as source:
        np.savez(corrupted, values=source["values"], specimen_id=source["specimen_id"], genes=source["genes"][::-1])
    reject("changed_gene_axis", "specimen and gene axes", lambda: score(SimpleNamespace(**dict(vars(args),measurements=corrupted))))
    return failures


def run(root, layout, layout_receipt, protocol):
    args = build(root, layout, layout_receipt, protocol)
    args.output = root/"freeze.json"
    lock(args)
    freeze = read_json(args.output)
    assert not freeze["expression_read"]
    args.freeze = args.output
    args.expected_sha256 = sha256(args.output)
    args.output = root/"score"
    score(args)
    paper = pd.read_csv(args.output/"paired.tsv", sep="\t")
    better = paper.loc[paper.arm.eq(ARMS[0]) & paper.reference.eq("best_VC")]
    assert len(better)==16 and better.loc[better.metric.eq("RMSE"),"high99"].lt(0).all()
    assert better.loc[better.metric.eq("pooled_R2"),"low99"].gt(0).all()
    pd.DataFrame(failures(args)).to_csv(root/"negative_checks.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "layout", "layout-receipt", "protocol"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args=parser.parse_args()
    run(args.root,args.layout,args.layout_receipt,args.protocol)
