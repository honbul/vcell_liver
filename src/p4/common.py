"""P4 well projection and immutable access to the published prediction bundles."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import MIN_GROUPS, WELL, eligibility, slice_name
from src.p123.p1_score import historical, load_case
from src.p123.p123_io import load_prediction
from src.r11_chemical_probe.common import read_json, sha256
from src.r11_chemical_probe.dataset import KEY, LINE


PROTOCOL_SHA = "38f7030d273fa719d1cab924d44819630afb72f17fcd0ff2541b70fadf70d993"
MIN_LINES = 10
NEW = ("P4-sibling", "P4-tanimoto", "P4-tanimoto-lincs")
OLD_CORE = ("B0", "B_pos-all", "proposed-v0", "p1-trim", "proposed-v1")
NULL = "well-only-zero"


def check_protocol(path):
    if sha256(path) != PROTOCOL_SHA:
        raise ValueError("P4 protocol differs from the committed contract")


class WellProjection:
    """Use every registered line of a well for the mean, then apply pool support."""

    def __init__(self, rows):
        self.original = rows.reset_index(drop=True)
        if rows.duplicated(KEY).any():
            raise ValueError("A well has duplicate cell-line rows")
        self.wells = self.original.groupby(WELL, sort=False).indices
        counts = {well: len(idx) for well, idx in self.wells.items()}
        supported = self.original[WELL].map(counts).ge(MIN_LINES)
        kept = self.original.loc[supported].assign(_p4_index=np.flatnonzero(supported))
        self.rows = eligibility(kept, 0)
        self.index = self.rows.pop("_p4_index").to_numpy(dtype=int)
        self.support = self.rows[WELL].map(counts).to_numpy(dtype=int)

    def centered(self, values):
        if len(values) != len(self.original):
            raise ValueError("The prediction does not share the well projection axis")
        centered = np.empty_like(values, dtype=np.float64)
        for idx in self.wells.values():
            block = values[idx].astype(np.float64, copy=False)
            centered[idx] = block - block.mean(axis=0)
        return centered[self.index]

    def distinct_lines(self):
        return self.rows[LINE].nunique()


class History:
    """Read only artifacts bound to the committed T1–P3 development freezes."""

    def __init__(self, args, dose):
        tag = f"d{dose:g}"
        self.tag, self.dose, self.stage = tag, dose, args.stage
        self.p123 = args.p123
        self.old_protocol = args.archive / "p123/PROTOCOL.md"
        self.prior = args.prior
        older = self.p123 / "p1-fair-score" / tag / "freeze.json"
        source = self.p123 / "p2" / tag / "freeze.json"
        expected_fair = args.archive / "p123/data/p1/fair" / tag / "freeze.json"
        expected_p2 = args.archive / "p123/data/p2" / tag / "freeze.json"
        self.references = {"p1_fair": sha256(older), "p2": sha256(source)}
        if self.references != {"p1_fair": sha256(expected_fair), "p2": sha256(expected_p2)}:
            raise ValueError("Published P1/P2 freeze differs from its committed copy")
        self.fair_stamp, self.p2_stamp = read_json(older), read_json(source)
        self.p1_root = self.p123 / "p1-predict"
        self.p2_root = self.p123 / "p2/predictions"
        extras = [Path(path) for path in self.fair_stamp["extra_root"]]
        self.fair_args = SimpleNamespace(prior=self.prior, stage=self.stage, dose=dose,
                                         track="P1-fair", p1_root=self.p1_root,
                                         protocol=self.old_protocol, extra_root=extras)
        self.previous = historical(self.fair_args)
        t1, t2, _, _, known = self.previous
        self.references.update(t1=sha256(t1 / "freeze.json"), t2=sha256(t2 / "freeze.json"))
        expected_t2 = args.archive / "goal_tracks/data/known_vc" / tag / "freeze.json"
        if self.references["t2"] != sha256(expected_t2):
            raise ValueError("Published T2 freeze differs from the committed copy")
        if set(extras) != {self.p123 / "p1-fair" / name for name in known}:
            raise ValueError("Historical fairness paths escaped the published arms")
        for stamp in (self.fair_stamp, self.p2_stamp):
            if (stamp["stage_manifest_sha256"] != sha256(self.stage / "receipt.json")
                    or stamp["t1_source_freeze_sha256"] != self.references["t1"]
                    or stamp["t2_source_freeze_sha256"] != self.references["t2"]
                    or stamp["protocol_sha256"] != sha256(self.old_protocol)):
                raise ValueError("Historical arm registries disagree on their source or contract")
        self.known = set(known)

    def source_selection(self, n, fold, role):
        name = slice_name(self.dose, n, fold, role)
        bound = self.p2_stamp["tasks"][name]
        paths = {"P1": self.p1_root / name / "receipt.json",
                 "extra-0": self.p2_root / name / "receipt.json"}
        paths.update({k: v / name / "receipt.json" for k, v in self.previous[-1].items()})
        hashes, choices = {}, {}
        for key, path in paths.items():
            hashes[key] = sha256(path)
            if hashes[key] != bound[key]:
                raise ValueError(f"Published model selection changed: {path}")
            choices[key] = read_json(path)["selection"]
        return choices, hashes

    def forecasts(self, n, fold, role):
        name, records, _, rows, panel, previous = load_case(
            self.fair_args, self.previous, n, fold, role, sealed=self.fair_stamp)
        p2_record = self.p2_stamp["tasks"][name]
        if any(p2_record[key] != records[key] for key in {"T1", "P1", *self.known}):
            raise ValueError("P1 and P2 froze different earlier predictions")
        path = self.p2_root / name
        p2_hash = sha256(path / "receipt.json")
        if p2_hash != p2_record["extra-0"]:
            raise ValueError("P2 prediction receipt changed")
        _, other, other_panel, priors = load_prediction(
            path, self.stage, self.old_protocol, self.dose, n, fold, role)
        if not rows[KEY].equals(other[KEY]) or not np.array_equal(panel, other_panel):
            raise ValueError("P2 does not share the T1 row and gene axes")
        arms = {k: v for k, v in previous.items() if k in OLD_CORE or k in self.known or k.startswith("fair-")}
        arms.update(priors)
        return rows, panel, arms, {"P1_fair": records, "P2": p2_hash}


def common_args(parser):
    for key in ("protocol", "archive", "prior", "p123", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)


def finish_args(args):
    check_protocol(args.protocol)
    args.stage = args.prior / "t1-cache"
    return args
