"""Check and evaluate separated 0.5 µM RNA inputs for Tahoe SNU-423."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import MIN_GROUPS, WELL, eligibility, slice_name, verify_one
from src.goal_tracks.t1_io import align_pool, score_truth
from src.p4.common import WellProjection
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.objectives import center
from src.r19_plan.hcc_scaffold import CASES, DOSE, LINE_ID, MINIMUM, TARGET
from src.r19_plan.hcc_scaffold_refit import require_hash


LOW = .5
PLATE, POSITION = "plate_id", "barcode_position"
NON_DONORS = {LINE_ID, "CVCL_1098"}
LINK_COLUMNS = (WELL, GROUP, ENTITY, PLATE, POSITION, "donor_well", "donor_plate", "donor_position",
                "decoy_well", "decoy_position")


def inputs(args):
    original = args.archive / "p4/data/d5"
    p4 = require_hash(original / "freeze.json", read_json(original / "scored/receipt.json")["freeze_sha256"])
    stamp = require_hash(args.stage / "receipt.json", read_json(p4)["stage_sha256"])
    proof = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                 p4_freeze_sha256=sha256(p4), stage_sha256=sha256(stamp))
    return read_json(stamp), proof


def partition(args, receipt, fold, split):
    path = args.stage / f"f{fold}/{split}_rows.tsv"
    verify_one(args.stage, receipt, path)
    rows = table(path)
    if rows.quarantined.any() or rows[PLATE].eq("plate13").any() or rows.duplicated(KEY).any():
        raise ValueError("A protected or duplicated Tahoe physical row entered the study")
    rows = rows.assign(_array_index=np.arange(len(rows)))
    return eligibility(rows, MINIMUM, LOW), eligibility(rows, MINIMUM, DOSE)


def projected_line(rows):
    axis = WellProjection(rows).rows
    return axis.loc[axis[LINE].eq(LINE_ID)].reset_index(drop=True)


def eligible_donor_wells(low):
    others = low.loc[~low[LINE].isin(NON_DONORS)].groupby(WELL)[LINE].nunique()
    eligible = others.index[others.ge(10)]
    return projected_line(low).loc[lambda r: r[WELL].isin(eligible)].sort_values(WELL)


def linked_wells(low, high):
    sources = eligible_donor_wells(low)
    groups = dict(tuple(sources.groupby([GROUP, ENTITY], sort=False)))
    plates = dict(tuple(sources.groupby(PLATE, sort=False)))
    rows = []
    for target in projected_line(high).to_dict("records"):
        group, entity, plate, position, well = (target[k] for k in (GROUP, ENTITY, PLATE, POSITION, WELL))
        donors = groups.get((group, entity))
        if donors is None:
            continue
        allowed = donors.loc[donors[PLATE].ne(plate) & donors[POSITION].ne(position) & donors[WELL].ne(well)]
        if allowed.empty:
            continue
        donor = allowed.iloc[0]
        decoys = plates[donor[PLATE]]
        matched = decoys.loc[decoys[GROUP].ne(group) & decoys[POSITION].ne(position) & decoys[WELL].ne(well)]
        if matched.empty:
            continue
        decoy_id = min(matched[WELL], key=lambda w: (crc32(f"{well}|{w}".encode()), w))
        decoy = matched.loc[matched[WELL].eq(decoy_id)].iloc[0]
        rows.append((well, group, entity, plate, position, donor[WELL], donor[PLATE], donor[POSITION],
                     decoy[WELL], decoy[POSITION]))
    links = pd.DataFrame(rows, columns=LINK_COLUMNS)
    scope = projected_line(high).loc[lambda r: r[WELL].isin(links[WELL])]
    kept = eligibility(scope, MINIMUM, DOSE)
    return links.loc[links[WELL].isin(kept[WELL])].reset_index(drop=True)


def audit(args, contract):
    receipt, proof = contract
    records, supports = [], []
    for fold in range(3):
        parent_sets = {}
        for split in ("fit", "score"):
            low, high = partition(args, receipt, fold, split)
            parent_sets[split] = set(low[GROUP]) | set(high[GROUP])
            links = linked_wells(low, high)
            records.append(links.assign(fold=fold, split=split))
            targets = projected_line(high)
            supports.append(dict(fold=fold, split=split, target_parents=targets[GROUP].nunique(),
                                 target_wells=targets[WELL].nunique(), common_parents=links[GROUP].nunique(),
                                 common_wells=links[WELL].nunique(), donor_plates=links.donor_plate.nunique(),
                                 target_plates=links[PLATE].nunique()))
        if parent_sets["fit"] & parent_sets["score"]:
            raise ValueError("A held parent crossed into training")
    links, support = pd.concat(records, ignore_index=True), pd.DataFrame(supports)
    held = links.loc[links.split.eq("score")]
    gate = (held[GROUP].nunique() >= 50 and support.loc[support.split.eq("score"), "common_parents"].ge(10).all()
            and support.loc[support.split.eq("fit"), "common_parents"].ge(30).all())
    dest = args.run / "audit"
    with TemporaryDirectory(dir=args.run, prefix=".dose-audit-") as temporary:
        root = Path(temporary)
        links.to_csv(root / "links.tsv", sep="\t", index=False)
        support.to_csv(root / "support.tsv", sep="\t", index=False)
        write_json(root / "receipt.json", dict(
            proof, eligible=bool(gate), held_common_parents=held[GROUP].nunique(), expression_read=False,
            files={p.name: sha256(p) for p in root.iterdir()}))
        if dest.exists():
            raise FileExistsError(dest)
        root.rename(dest)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("audit",))
    for name in ("stage", "archive", "run", "protocol"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    audit(args, inputs(args))


if __name__ == "__main__":
    main()
