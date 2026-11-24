"""Verify E4 roles and add matched pooled inner folds for the three DRP fits."""
import argparse
import os
from pathlib import Path
import sys

import pandas as pd

from followup_h2.common import keys_hash, panel_hash, sha256
from followup_h2_e4.data import (
    STORAGE, internal_only, lincs_profiles, observed, registry, source_metadata, train_hash, utc_now,
)
from followup_h2_e4.selection import split_contract
from followup_h2_e4.split import construct, describe, overlaps
from nextgen_h2.assets import KEY
from nextgen_h2.run import read_json
from vc_build.io import publish_json

ROOT = STORAGE / "followup_20261124/h2_e5_upgrade"
E4 = ROOT.with_name("h2_e4_loso")
PLAN = Path(__file__).resolve().parents[2] / "reports/followup_h2_e5_upgrade/PLAN_LOCK.md"
E4_SPLIT = "3bae4c230adf0811edc810449396a67ff06991ddf5f2c1f498c7f1e4899f3143"
E4_SELECTED = "54def01f27e7e5f5673d4d5a7a799c2d1d375868eed9c16b5e46ecfe684495c7"


def source_only():
    internal_only()
    blocked = [E4 / x for x in ("forecasts", "tests", "score", "official-outcome-access.json")]

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            f = Path(os.path.abspath(os.fsdecode(args[0])))
            if any(f.is_relative_to(p) for p in blocked):
                raise PermissionError(f"E4 held evidence is sequestered: {f}")

    sys.addaudithook(guard)


def extended_split(e4, rows, reg, covered):
    roles = dict(e4["roles"])
    selections = dict(e4["selections"])
    for outer in range(3):
        parent = roles[f"T2/{outer}/outer"]
        pool = rows.loc[rows[KEY].isin(parent["fit_parents"])]
        codes = pool[KEY].map(reg[f"inner{outer}"])
        assert set(codes.unique()) == {0, 1, 2}
        prefix = f"T2/{outer}/Sel-Pooled"
        selections[prefix] = []
        for f in range(3):
            name = f"{prefix}/{f}"
            selections[prefix].append(name)
            fit, query = pool.loc[codes.ne(f)], pool.loc[codes.eq(f)]
            a, b = sorted(fit[KEY].unique()), sorted(query[KEY].unique())
            common = overlaps(reg, a, b)
            outer_common = overlaps(reg, set(a)|set(b), parent["query_parents"])
            assert set(common.values()) == set(outer_common.values()) == {0}
            roles[name] = {
                "kind": "pooled", "fit_screens": sorted(fit.source_id.unique()),
                "query_screens": sorted(query.source_id.unique()), "fit_parents": a, "query_parents": b,
                "fit_parents_sha256": keys_hash(a), "query_parents_sha256": keys_hash(b),
                "fit_roster_sha256": panel_hash(fit), "query_roster_sha256": panel_hash(query),
                "fit": describe(fit, reg, covered), "query": describe(query, reg, covered),
                "overlap": common, "outer_overlap": outer_common,
            }
    assert len(roles) == 45 and sum(map(len, selections.values())) == 39
    return roles, selections


def register(out):
    source_only()
    e4_file = E4 / "preflight-projected/e4_split_manifest.json"
    assert sha256(e4_file) == E4_SPLIT
    e4 = split_contract(e4_file.parent)
    reg, rows = registry(), source_metadata()
    profiles = lincs_profiles(reg.index)
    covered = observed(profiles)
    checked, _ = construct(reg, rows, covered, observed(profiles, "modz24"))
    assert checked["roles"] == e4["roles"] and checked["selections"] == e4["selections"]
    assert e4["source_roster_sha256"] == panel_hash(rows)
    roles, selection = extended_split(e4, rows, reg, covered)
    out.mkdir(exist_ok=False, parents=True)
    publish_json(out / "split.json", {
        "E4_split_sha256": E4_SPLIT, "plan_sha256": sha256(PLAN), "source_file_sha256": train_hash(),
        "source_roster_sha256": panel_hash(rows), "source_support": e4["source_support"],
        "roles": roles, "selections": selection, "evaluations": e4["evaluations"],
        "registered_at_utc": utc_now(), "external_cohort_access": 0, "response_columns_read": [],
    })
    print("45 verified roles; 39 inner; original outer rosters preserved.", flush=True)


def read_split(preflight):
    choice = read_json(preflight / "split.json")
    assert choice["E4_split_sha256"] == E4_SPLIT and choice["plan_sha256"] == sha256(PLAN)
    assert choice["source_file_sha256"] == train_hash() and not choice["external_cohort_access"]
    return choice


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    register(cli.parse_args().out)
