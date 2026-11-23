"""Audit prior double-cold rosters, align cell families, and register outcome-free roles."""
import argparse
from collections import defaultdict
from hashlib import sha256 as digest
from pathlib import Path

import pandas as pd

from nextgen_h2.assets import KEY, LINCS, MODEL, parquet_lock
from nextgen_h2.run import read_json
from nextgen_h2_v2.assets import LINCS_SIGNATURE_SHA, V1_ASSETS
from vc_build.cross_source_data import MODEL_COLS, MODEL_KEYS, MODEL_NAMES, parse_cellosaurus_identity
from vc_build.external import ANCHOR, SOURCE
from vc_build.function_data import alias_token
from vc_build.io import publish_json, sha256

from .common import ARCHIVE, CELLOSAURUS, keys_hash, native_rows, panel_hash, registered


class Families:
    def __init__(self):
        self.root = {}

    def find(self, key):
        parent = self.root.setdefault(key, key)
        if parent != key:
            self.root[key] = self.find(parent)
        return self.root[key]

    def join(self, a, b):
        a, b = sorted((self.find(a), self.find(b)))
        self.root[b] = a


def family_map(cells, refs):
    profiles, secondary, related = parse_cellosaurus_identity(CELLOSAURUS)
    models = pd.read_csv(MODEL, usecols=MODEL_COLS, dtype=str, keep_default_na=False)
    models = models.set_index("ModelID", verify_integrity=True)
    match_ids, aliases = defaultdict(set), defaultdict(set)
    tree = Families()
    for acc, (names, ids, _) in profiles.items():
        for other in related[acc]:
            tree.join(("CVCL", acc), ("CVCL", other))
        for item in ids:
            match_ids[item].add(acc)
        for name in names:
            if alias_token(name):
                aliases[alias_token(name)].add(("CVCL", acc))
    for model, row in models.iterrows():
        node = ("Model", model)
        if row.PatientID:
            tree.join(node, ("Patient", row.PatientID))
        if row.RRID:
            tree.join(node, ("CVCL", secondary.get(row.RRID, row.RRID)))
        for source_id in (model, row.SangerModelID, row.COSMICID, row.ModelIDAlias):
            found = match_ids.get(source_id, set())
            if len(found) == 1:
                tree.join(node, ("CVCL", next(iter(found))))
        for key in MODEL_NAMES:
            if alias_token(row[key]):
                aliases[alias_token(row[key])].add(node)
    for model, row in cells.iterrows():
        tree.join(("Model", model), ("Registered", row.lineage_group))
        node = ("Model", model)
        rrid = models.loc[model, "RRID"] if model in models.index else ""
        if not rrid and model in models.index:
            for key in MODEL_NAMES:
                matches = {tree.find(x) for x in aliases.get(alias_token(models.loc[model, key]), set()) if x[0] == "CVCL"}
                if len(matches) == 1:
                    tree.join(node, next(iter(matches)))
    aliases = {token: {tree.find(x) for x in ids} for token, ids in aliases.items()}
    by_root = defaultdict(list)
    for model, row in cells.iterrows():
        by_root[tree.find(("Model", model))].append(row.lineage_group)
    named = {key: min(values) for key, values in by_root.items()}
    cell_group = pd.Series({key: named[tree.find(("Model", key))] for key in cells.index}, name="family")
    found = []
    for r in refs.itertuples(index=False):
        acc = secondary.get(r.cellosaurus_id, r.cellosaurus_id)
        exact = tree.find(("CVCL", acc)) if acc else None
        names = [r.cell_iname, r.ccle_name, *r.cell_alias.replace("|", ";").split(";")]
        tokens = {alias_token(x) for x in names} - {""}
        candidates = set().union(*(aliases.get(x, set()) for x in tokens))
        if exact is not None:
            candidates.add(exact)
        families = sorted({named[x] for x in candidates if x in named})
        verified = acc in profiles or bool(candidates)
        found.append({"reference": r.cell_iname, "rrid": acc, "families": families,
                      "identity_evidence": "accession/alias" if verified else "unresolved",
                      "possible_source_families": len(families)})
    return cell_group, found


def balanced_labels(groups, seed, tissue=None):
    levels = groups.value_counts().to_dict()
    strata = {} if tissue is None else tissue.groupby(groups).agg(lambda x: x.mode()[0]).to_dict()
    rank = lambda group: digest(f"{seed}:{group}".encode()).digest()
    general = [0, 0, 0]
    by_stratum = defaultdict(lambda: [0, 0, 0])
    choice = {}
    for group in sorted(levels, key=lambda x: (-levels[x], rank(x))):
        counts = by_stratum[strata.get(group, "all")]
        target = min(range(3), key=lambda k: (counts[k], general[k], k))
        choice[group] = target
        counts[target] += levels[group]
        general[target] += levels[group]
    return groups.map(choice)


def previous_labels(registry, cells, roster):
    assignments, archive_hashes = [], []
    for i in range(3):
        path = ARCHIVE / "parts" / f"fold-{i}.parquet"
        receipt = read_json(path.with_suffix(".parquet.freeze.json"))
        lock = sha256(path)
        expected = receipt.get("sha256", receipt.get("prediction_sha256", receipt.get("table_sha256")))
        if expected is not None and expected != lock:
            raise ValueError(f"Previous joint holdout checksum mismatch: {i}")
        old = pd.read_parquet(path, columns=["row_id", "cell_id", KEY]).drop_duplicates()
        archive_hashes.append({"fold": i, "rows": len(old), "forecast_sha256": lock,
                               "freeze_receipt_sha256": sha256(path.with_suffix(".parquet.freeze.json"))})
        assignments.append((old, i))
    dmap, cmap = {}, {}
    def admit(target, group, fold):
        prior = target.setdefault(group, fold)
        return prior == fold
    if any(not admit(dmap, group, i) for frame, i in assignments for group in frame[KEY].map(registry.scaffold_block)):
        return None, archive_hashes, "Archived withheld scaffold rows assign a linked scaffold/connectivity group to multiple folds"
    if any(not admit(cmap, group, i) for frame, i in assignments for group in frame.cell_id.map(cells.family)):
        return None, archive_hashes, "Archived withheld cells assign an alias/derivation family to multiple folds"
    drug, cell = registry.scaffold_block.map(dmap), cells.family.map(cmap)
    if drug.isna().any() or cell.isna().any():
        return None, archive_hashes, "Archived holdouts do not cover all registered H2 parents and cell families"
    for old, i in assignments:
        hold = roster[roster[KEY].map(drug).eq(i) & roster.cell_id.map(cell).eq(i)]
        if set(hold.row_id) != set(old.row_id):
            return None, archive_hashes, "Archived held rows differ from the complete H2 diagonal roster"
    return (drug.astype(int), cell.astype(int)), archive_hashes, "Compatible registered joint rows"


def pair_rosters(roster):
    sets = {tag: block[ANCHOR].drop_duplicates() for tag, block in roster.groupby(SOURCE, sort=True)}
    ref = next(tag for tag in sets if tag[0] == "GDSC2")
    return {tag[0]: frame.merge(sets[ref], on=ANCHOR, how="inner", validate="one_to_one")
            for tag, frame in sets.items() if tag != ref}


def anchor_support(pairs, dfit, cfit):
    both = {}
    for name, frame in pairs.items():
        counts = frame.loc[frame[KEY].isin(dfit) & frame.cell_id.isin(cfit)].groupby(KEY).size()
        counts = counts[counts.ge(12)]
        both[name] = {"parents": len(counts), "pairs": int(counts.sum())}
    return both


def admissible(pairs, ds, cs):
    return all(count["parents"] >= 3 and count["pairs"] >= 60 for i in range(3)
               for count in anchor_support(pairs, ds.index[ds.ne(i)], cs.index[cs.ne(i)]).values())


def plan_role(name, roster, registry, cells, dfit, cfit, dhold, chold, refmap, pairs, embargo=()):
    train = roster[roster[KEY].isin(dfit) & roster.cell_id.isin(cfit)]
    test = roster[roster[KEY].isin(dhold) & roster.cell_id.isin(chold)]
    fit_scaffolds = set(train[KEY].map(registry.scaffold_block))
    query_scaffolds = set(test[KEY].map(registry.scaffold_block))
    fit_family = set(train.cell_id.map(cells.family))
    query_family = set(test.cell_id.map(cells.family))
    fit_original = set(train.lineage_group)
    query_original = set(test.lineage_group)
    assert not fit_scaffolds & query_scaffolds and not fit_family & query_family
    assert not fit_original & query_original and not set(train.row_id) & set(test.row_id)
    assert set(train.source_id.unique()) == set(test.source_id.unique()) == {"CTRPv2", "GDSC2", "PRISM20Q2"}
    excluded_families = set(cells.loc[chold, "family"]) | set(embargo)
    excluded = [r["reference"] for r in refmap if set(r["families"]) & excluded_families
                or r["identity_evidence"] == "unresolved"]
    liver = test[test.cell_id.map(cells.tissue).eq("Liver")]
    return {
        "name": name, "fit_drugs": sorted(dfit), "query_drugs": sorted(dhold),
        "fit_cells": sorted(cfit), "query_cells": sorted(chold),
        "fit_roster_sha256": panel_hash(train), "query_roster_sha256": panel_hash(test),
        "fit_rows": len(train), "query_rows": len(test),
        "fit_scaffolds_sha256": keys_hash(fit_scaffolds), "query_scaffolds_sha256": keys_hash(query_scaffolds),
        "query_lineages": len(query_family), "zero_overlap_axes": ["scaffold_connectivity", "aliases_derivatives", "registered_lineage", "row_id"],
        "excluded_references": sorted(excluded), "excluded_family_sha256": keys_hash(excluded_families),
        "fit_anchor_support": anchor_support(pairs, dfit, cfit),
        "query_rows_by_screen": test.source_id.value_counts().to_dict(),
        "Liver": {"rows": len(liver), "lineages": liver.cell_id.map(cells.family).nunique(),
                  "cells": liver.cell_id.nunique(), "parents": liver[KEY].nunique()},
    }


def register(out):
    out.mkdir(exist_ok=False, parents=True)
    registry, _, cells, assets = registered()
    roster = native_rows(registry.index)
    assert (len(roster), len(registry), len(cells), cells.lineage_group.nunique()) == (670830, 1186, 1014, 991)
    signatures = parquet_lock(V1_ASSETS / "lincs_selected_signatures.parquet", LINCS_SIGNATURE_SHA)
    metadata = LINCS / "cellinfo_beta.txt"
    if sha256(metadata) != assets["lincs_metadata_sha256"][metadata.name]:
        raise ValueError("Cellinfo differs from H2 registration")
    refs = pd.read_csv(metadata, sep="\t", dtype=str, keep_default_na=False)
    refs = refs[refs.cell_iname.isin(signatures.cell_iname)].sort_values("cell_iname")
    assert refs.cell_iname.is_unique and len(refs) == 147
    family, refmap = family_map(cells, refs)
    cells = cells.assign(family=family)
    cells.to_parquet(out / "cell_roster.parquet")
    previous, provenance, reason = previous_labels(registry, cells, roster)
    pairs = pair_rosters(roster)
    choice = None
    for seed in range(128):
        if seed == 0 and previous is not None:
            ds, cs = previous
        else:
            ds = registry.outer if seed < 32 else balanced_labels(registry.scaffold_block, 11711 + seed)
            cs = balanced_labels(cells.family, 39167 + seed, cells.tissue)
        if not admissible(pairs, ds, cs):
            continue
        inner = []
        for f in range(3):
            drug_fit = registry.loc[ds.ne(f), "scaffold_block"]
            cell_fit = cells.loc[cs.ne(f)]
            for depth in range(128):
                d = balanced_labels(drug_fit, 1691 + depth)
                c = balanced_labels(cell_fit.family, 6203 + depth, cell_fit.tissue)
                if admissible(pairs, d, c):
                    inner.append((d, c, depth))
                    break
        if len(inner) == 3:
            choice = (seed, ds, cs, inner)
            break
    if choice is None:
        raise ValueError("No outcome-free grouped double-cold split has adequate fit-only source anchors")
    seed, ds, cs, subroles = choice
    outer, inner = [], []
    for f, (df, cf, depth) in enumerate(subroles):
        outer.append(plan_role(str(f), roster, registry, cells, ds.index[ds.ne(f)], cs.index[cs.ne(f)],
                               ds.index[ds.eq(f)], cs.index[cs.eq(f)], refmap, pairs))
        groups = set(cells.loc[cs.eq(f), "family"])
        for i in range(3):
            settings = plan_role(f"{f}-{i}", roster, registry, cells, df.index[df.ne(i)], cf.index[cf.ne(i)],
                                 df.index[df.eq(i)], cf.index[cf.eq(i)], refmap, pairs, groups)
            assert set(settings["fit_cells"] + settings["query_cells"]) <= set(outer[f]["fit_cells"])
            assert set(settings["fit_drugs"] + settings["query_drugs"]) <= set(outer[f]["fit_drugs"])
            assert set(outer[f]["excluded_references"]) <= set(settings["excluded_references"])
            settings["label_free_balance_candidate"] = depth
            inner.append(settings)
    publish_json(out / "split_manifest.json", {
        "source_outcome_columns_read": [], "outer": outer, "inner": inner, "source_rows": len(roster),
        "source_parents": len(registry), "scaffold_components": registry.scaffold_block.nunique(),
        "source_cells": len(cells), "registered_lineages": cells.lineage_group.nunique(),
        "alias_augmented_lineages": cells.family.nunique(), "cell_roster_sha256": sha256(out/"cell_roster.parquet"),
        "archive": provenance, "archive_reused": previous is not None and seed == 0,
        "archive_assessment": reason, "outcome_free_outer_candidate": seed,
        "reference_identity": refmap, "cellosaurus_sha256": sha256(CELLOSAURUS), "Model_sha256": sha256(MODEL),
        "signature_registry_sha256": LINCS_SIGNATURE_SHA, "LINCS_cellinfo_sha256": sha256(metadata),
    })
    print([(x["name"], x["fit_rows"], x["query_rows"], len(x["excluded_references"])) for x in outer], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    register(cli.parse_args().out)
