"""Adapters for the locked T4 source-only, donor-input and score roles."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from zlib import crc32

import numpy as np
import pandas as pd

from src.r19_plan.t4_context_transfer import group_plans, inputs as registered_contexts
from src.r19_plan.t4_scaffold_transfer import fit_effects, scaffold
from src.r19_plan.t4_workflow import axes, average_conditions, lookup


@dataclass
class PairBatch:
    rows: pd.DataFrame
    x: np.ndarray
    y: np.ndarray

    def select(self, mask):
        return PairBatch(self.rows.loc[mask].reset_index(drop=True), self.x[mask], self.y[mask])


def contexts(reports, hepg2, phh):
    cfg = SimpleNamespace(reports=reports, hepg2=hepg2, phh=phh, target="HEPG2",
                          protocol=reports / "T4_CONTEXT_TRANSFER_PROTOCOL.md")
    hep, primary, _ = registered_contexts(cfg)
    return {"HEPG2": hep, "PHH": primary}


def parent_folds(ctx):
    parents = {0: set(), 1: set(), 2: set()}
    for fold in parents:
        for tables in group_plans(ctx, f"f{fold}-A").values():
            parents[fold].update(tables["query"].parent)
    if len(set().union(*parents.values())) != ctx.ref["parents"]:
        raise ValueError("A T4 parent appears in multiple score folds")
    return parents


class SourceWells:
    def __init__(self, ctx, group, protected):
        self.plans = group_plans(ctx, group)
        self.role = axes(next(iter(self.plans)))[1]
        self.protected = protected
        self.index, self.effects = fit_effects(ctx, self.plans, self.role)

    def data(self, key):
        tables = self.plans[key]
        before, after = (tables[name].copy() for name in ("source6", "source24"))
        prepared = []
        for frame in (before, after):
            frame["scaffold"] = frame.parent.map(scaffold)
            frame["dose_uM"] = frame.actual_dose_uM.astype(float)
            selection = ~frame.scaffold.isin(self.protected)
            frame = frame.loc[selection].reset_index(drop=True)
            loc = self.index.get_indexer(frame.inst_id)
            if (loc < 0).any():
                raise ValueError("A source well escaped the locked fit effect index")
            prepared.append((frame, self.effects[loc]))
        if any(set(frame.scaffold) & self.protected for frame, _ in prepared):
            raise ValueError("A protected parent scaffold entered training")
        return prepared


def aligned(before, after, mask_before, mask_after, *, average_late):
    early, x = before
    late, y = after
    e = early.loc[mask_before].reset_index(drop=True)
    l = late.loc[mask_after].reset_index(drop=True)
    match = pd.MultiIndex.from_frame(e[["parent", "actual_dose_uM"]])
    keep = pd.MultiIndex.from_frame(l[["parent", "actual_dose_uM"]]).isin(match)
    e_values = average_conditions(e, x[mask_before])
    l = l.loc[keep].reset_index(drop=True)
    y = y[mask_after][keep]
    if not len(l):
        return PairBatch(l, np.empty((0, x.shape[1])), np.empty((0, x.shape[1])))
    if average_late:
        key, y = average_conditions(l, y)
        l = key.to_frame(index=False).assign(
            scaffold=lambda r: r.parent.map(scaffold),
            dose_uM=lambda r: r.actual_dose_uM.astype(float))
    return PairBatch(l, lookup(e_values, l), y)


def all_pairs(before, after):
    early, late = (np.ones(len(s[0]), dtype=bool) for s in (before, after))
    return aligned(before, after, early, late, average_late=True)


def purged_inner(before, after, salt, count=3):
    """Each inner validation holds out entire scaffold, 6 h plate, and 24 h plate groups.

    Search uses metadata support alone. Each held scaffold belongs to just one
    inner partition and both train and validation empirical controls come from
    mutually exclusive physical plates.
    """
    e, l = before[0], after[0]
    scaffold_groups = np.array(sorted(set(e.scaffold) & set(l.scaffold)))
    earlier, later = (np.array(sorted(rows.rna_plate.unique())) for rows in (e, l))
    rng = np.random.default_rng(crc32(salt.encode()))
    group_sets = [set(g) for g in np.array_split(rng.permutation(scaffold_groups), count)]
    for fold, held_scaffold in enumerate(group_sets):
        best = None
        for _ in range(36):
            p6 = rng.choice(earlier, size=max(1, round(.27 * len(earlier))), replace=False)
            p24 = rng.choice(later, size=max(1, round(.32 * len(later))), replace=False)
            plates = set(p6) | set(p24)
            held_early, held_late = e.rna_plate.isin(plates), l.rna_plate.isin(plates)
            early_core, late_core = e.scaffold.isin(held_scaffold), l.scaffold.isin(held_scaffold)
            a, b = (~(held_early | early_core)).to_numpy(), (~(held_late | late_core)).to_numpy()
            c, d = (held_early & early_core).to_numpy(), (held_late & late_core).to_numpy()
            train = pair_count(e[a], l[b])
            valid = pair_count(e[c], l[d])
            if train < 12 or valid < 3:
                continue
            score = min(valid, 20) + min(train, 75) / 8
            if best is None or score > best[0]:
                best = score, a, b, c, d, sorted(plates), train, valid
        if best is None:
            raise ValueError(f"Inner scaffold/plate partition {salt}/{fold} lacks matched conditions")
        _, a, b, c, d, plates, train, valid = best
        fit = aligned(before, after, a, b, average_late=True)
        validation = aligned(before, after, c, d, average_late=False)
        if set(fit.rows.scaffold) & set(validation.rows.scaffold):
            raise ValueError("Inner scaffold leakage")
        yield fit, validation, {"fold": fold, "fit_pairs": train, "validation_pairs": valid,
                                "held_scaffolds": len(held_scaffold), "held_plates": plates}


def pair_count(early, late):
    columns = ["parent", "actual_dose_uM"]
    x = pd.MultiIndex.from_frame(early[columns].drop_duplicates())
    y = pd.MultiIndex.from_frame(late[columns].drop_duplicates())
    return len(x.intersection(y))


def parent_weights(rows):
    return 1. / rows.groupby("parent").parent.transform("size").to_numpy()
