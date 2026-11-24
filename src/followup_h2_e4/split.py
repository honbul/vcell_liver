"""Preflight the source-only panel and lock all nested selection roles."""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from followup_h2.common import keys_hash, panel_hash, sha256
from nextgen_h2.assets import KEY, fold_ids, linked_groups
from vc_build.io import publish_json

from .data import PLAN, SCREENS, internal_only, lincs_profiles, observed, registry, source_metadata, train_hash, utc_now


def role_id(track, outer, method):
    return f"{track}/{outer}/{method}"


def identities(reg, parents):
    a = reg.loc[parents]
    return {
        "exact_parent": set(a.index),
        "connectivity": set(a.connectivity),
        "transitive_scaffold": set(a.scaffold_block),
    }


def overlaps(reg, fit, query):
    a, b = identities(reg, fit), identities(reg, query)
    return {k: len(a[k] & b[k]) for k in a}


def held_fit(pool, screen, reg):
    query = pool.loc[pool.source_id.eq(screen)]
    blocked = identities(reg, query[KEY].unique())
    permitted = reg.index[
        ~reg.index.isin(blocked["exact_parent"])
        & ~reg.connectivity.isin(blocked["connectivity"])
        & ~reg.scaffold_block.isin(blocked["transitive_scaffold"])
    ]
    return pool.loc[pool.source_id.ne(screen) & pool[KEY].isin(permitted)], query


def describe(frame, reg, support):
    parents = pd.Index(sorted(frame[KEY].unique()))
    have = parents[support.reindex(parents).to_numpy()]
    return {
        "rows": len(frame), "parents": len(parents),
        "scaffolds": reg.loc[parents].scaffold_block.nunique(),
        "lincs_parents": len(have), "lincs_scaffolds": reg.loc[have].scaffold_block.nunique(),
        "lincs_rows": int(frame[KEY].isin(have).sum()),
    }


def construct(reg, source, support, legacy_support):
    roles, selections, evaluations, checks, summary = {}, {}, [], [], []

    def add(name, fit, query, kind, forbidden=None):
        fit_ids, query_ids = sorted(fit[KEY].unique()), sorted(query[KEY].unique())
        fit_sources, query_sources = sorted(fit.source_id.unique()), sorted(query.source_id.unique())
        assert fit_ids and query_ids
        assert not (set(fit_sources) & set(query_sources)) if kind == "loso" else True
        common = overlaps(reg, fit_ids, query_ids)
        assert all(value == 0 for value in common.values()), (name, common)
        outer = None
        if forbidden is not None:
            outer = overlaps(reg, set(fit_ids) | set(query_ids), forbidden)
            assert all(value == 0 for value in outer.values()), (name, outer)
        for side, frame in (("fit", fit), ("query", query)):
            info = describe(frame, reg, support)
            summary.append({"section": kind, "role": name, "side": side, "screens": ",".join(
                sorted(frame.source_id.unique())), **info,
                "warning": "FIT_LINCS_LT100" if side == "fit" and info["lincs_parents"] < 100 else ""})
        roles[name] = {
            "kind": kind, "fit_screens": fit_sources, "query_screens": query_sources,
            "fit_parents": fit_ids, "query_parents": query_ids,
            "fit_parents_sha256": keys_hash(fit_ids), "query_parents_sha256": keys_hash(query_ids),
            "fit_roster_sha256": panel_hash(fit), "query_roster_sha256": panel_hash(query),
            "fit": describe(fit, reg, support), "query": describe(query, reg, support),
            "overlap": common, "outer_overlap": outer,
        }
        if name not in evaluations and fit[KEY].map(support).groupby(fit[KEY]).first().sum() < 10:
            checks.append(name)

    def loso(prefix, pool, forbidden=None):
        names = []
        for screen in sorted(pool.source_id.unique()):
            name = f"{prefix}/{screen}"
            fit, query = held_fit(pool, screen, reg)
            add(name, fit, query, "loso", forbidden)
            names.append(name)
        return names

    def pooled(prefix, pool, folds, forbidden=None):
        names = []
        codes = pool[KEY].map(folds)
        for fold in range(3):
            name = f"{prefix}/{fold}"
            add(name, pool.loc[codes.ne(fold)], pool.loc[codes.eq(fold)], "pooled", forbidden)
            names.append(name)
        return names

    for screen in SCREENS:
        part = source.loc[source.source_id.eq(screen)]
        summary.append({"section": "screen", "role": screen, "side": "all",
                        "screens": screen, **describe(part, reg, support)})
    for a, b in combinations(SCREENS, 2):
        ids = [sorted(source.loc[source.source_id.eq(s), KEY].unique()) for s in (a, b)]
        common = set(ids[0]) & set(ids[1])
        scaffold = reg.loc[ids[0]].scaffold_block
        others = reg.loc[ids[1]].scaffold_block
        obs_blocks = [set(reg.loc[[key for key in v if support[key]]].scaffold_block) for v in ids]
        summary.append({"section": "screen_overlap", "role": f"{a}/{b}", "side": "intersection",
                        "screens": f"{a},{b}", "parents": len(common),
                        "scaffolds": len(set(scaffold) & set(others)),
                        "lincs_parents": sum(support[key] for key in common),
                        "lincs_scaffolds": len(obs_blocks[0] & obs_blocks[1]),
                        "connectivity_overlap": overlaps(reg, *ids)["connectivity"]})

    for index, screen in enumerate(SCREENS):
        fit, query = held_fit(source, screen, reg)
        outer = role_id("T1", index, "outer")
        evaluations.append(outer)
        add(outer, fit, query, "loso")
        forbidden = roles[outer]["query_parents"]
        prefix = role_id("T1", index, "Sel-LOSO")
        selections[prefix] = loso(prefix, fit, forbidden)
        prefix = role_id("T1", index, "Sel-Pooled")
        folds = fold_ids(reg.loc[roles[outer]["fit_parents"]].scaffold_block)
        selections[prefix] = pooled(prefix, fit, folds, forbidden)

    for fold in range(3):
        mask = source[KEY].map(reg.outer).eq(fold)
        outer = role_id("T2", fold, "outer")
        evaluations.append(outer)
        add(outer, source.loc[~mask], source.loc[mask], "outer_scaffold")
        prefix = role_id("T2", fold, "Sel-LOSO")
        selections[prefix] = loso(prefix, source.loc[~mask], roles[outer]["query_parents"])

    prefix = role_id("T3", 3, "Sel-LOSO")
    selections[prefix] = loso(prefix, source)
    prefix = role_id("T3", 3, "Sel-Pooled")
    selections[prefix] = pooled(prefix, source, reg.outer)

    return {
        "roles": roles, "selections": selections, "evaluations": evaluations,
        "below_minimum_lincs": checks, "source_roster_sha256": panel_hash(source),
        "source_support": describe(source, reg, support),
        "lincs_support_changed_from_v1": sorted(support.index[support.ne(legacy_support)]),
    }, pd.DataFrame(summary)


def register(out):
    internal_only()
    reg, source = registry(), source_metadata()
    assert source.row_id.is_unique and source.source_id.value_counts().to_dict() == SCREENS
    assert set(source[KEY]) == set(reg.index)
    pd.testing.assert_series_equal(linked_groups(reg).rename_axis(KEY).sort_index(), reg.scaffold_block.sort_index())
    profiles = lincs_profiles(reg.index)
    assert profiles["landmark_mean"].shape[1] == 978
    support = observed(profiles)
    legacy_support = observed(profiles, "modz24")
    assert legacy_support.sum() == 724
    manifest, table = construct(reg, source, support, legacy_support)
    out.mkdir(exist_ok=False, parents=True)
    table.to_csv(out / "e4_step0_support.tsv", sep="\t", index=False)
    manifest.update({
        "plan_sha256": sha256(PLAN), "source_file_sha256": train_hash(),
        "support_sha256": sha256(out / "e4_step0_support.tsv"), "registered_at_utc": utc_now(),
        "response_columns_read": [], "external_cohort_access": 0,
    })
    publish_json(out / "e4_split_manifest.json", manifest)
    print(manifest["source_support"], manifest["below_minimum_lincs"], flush=True)
    if manifest["below_minimum_lincs"]:
        raise ValueError("At least one locked selection role has fewer than 10 observed fitting parents")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    register(cli.parse_args().out)
