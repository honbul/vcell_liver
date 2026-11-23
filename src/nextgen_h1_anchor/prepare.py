"""Custodian creates separately sealed development packets and held outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vc_build.io import sha256

from .inputs import (CORE, COLD, CROSS, LICOB_LABEL, LIMORE_LABEL, WONG, BASE, NAMES, PAN,
                     DRUG_META, TISSUE, fold_order, json_write, metadata, rosters, source_rows)

TRANSFER = "LIMORE_to_LICOB"
AUXILIARY = "LIMORE_to_Wong"


def panel_matrix(frame, groups, compounds):
    p = frame.groupby(["lineage_group", "compound_id"]).sensitivity.mean().unstack()
    return p.reindex(index=groups, columns=compounds).to_numpy(dtype=float)


def patient_y(limore, licob, wong):
    from vc_build.evaluate import join_observations
    from vc_build.external import KEY
    from vc_build.licob_node_recovery import joined_outcomes
    from vc_build.v0plus_wong import WORKBOOK_HASH
    from vc_build.wong_expanded import observations

    def identity(frame):
        return frame[KEY].assign(predicted_value=0.)

    l = join_observations(identity(limore), pd.read_parquet(LIMORE_LABEL))
    c = joined_outcomes(licob, identity(licob), LICOB_LABEL)
    alt = BASE / "licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx"
    if sha256(alt) != WORKBOOK_HASH:
        raise ValueError("Unexpected Wong source workbook")
    w = observations(wong, alt)
    return l.assign(sensitivity=l.endpoint_value), c.assign(sensitivity=c.endpoint_value), w.assign(
        sensitivity=-w.endpoint_value)


def ranked_samples(rna, roster, groups, genes):
    unique = roster[["cell_id", "lineage_group"]].drop_duplicates()
    if unique.cell_id.duplicated().any():
        raise ValueError("A model has conflicting independent group identities")
    raw = rna.loc[unique.cell_id, genes].to_numpy()
    ranks = ((rankdata(raw, axis=1) - .5) / len(genes) - .5).astype(np.float32)
    return pd.DataFrame(ranks, index=unique.lineage_group).groupby(level=0).mean().loc[groups].to_numpy()


def checked_support(path):
    info = json.loads(path.read_text())
    for asset, expected in info["file_sha256"].items():
        if sha256(Path(asset)) != expected:
            raise ValueError(f"Source changed since H1 plan lock: {asset}")
    return info


def prepare(support, out):
    from vc_build.external import native_sensitivity

    info = checked_support(support)
    out.mkdir()
    sealed = out / "labels"
    sealed.mkdir(mode=0o700)
    sources = source_rows(labels=True)
    sources["sensitivity"] = native_sensitivity(sources)
    sets = {k: v for k, v in sources.groupby("source_id", sort=True)}
    limore, licob, wong = rosters()
    if (limore.lineage_group.nunique(), licob.lineage_group.nunique(), licob.cell_id.nunique()) != (50, 38, 43):
        raise ValueError("Independent HCC roster changed")
    roster = sets | {"LIMORE": limore, "LICOB": licob, "Wong": wong.assign(tissue_origin="Liver")}
    ly, cy, wy = patient_y(limore, licob, wong)
    sets.update(LIMORE=ly, LICOB=cy, Wong=wy)
    target = {TRANSFER: "LICOB", AUXILIARY: "Wong"}
    rna = {name: pd.read_parquet(CORE / "train/rna.parquet") for name in PAN[:1]}
    rna.update({name: rna[PAN[0]] for name in PAN[1:]})
    rna.update(LIMORE=pd.read_parquet(CROSS / "test_inputs/rna.parquet"),
               LICOB=pd.read_parquet(COLD / "rna.parquet"),
               Wong=pd.read_parquet(WONG / "rna.parquet"))
    groups = {name: np.array(sorted(frame.lineage_group.unique())) for name, frame in roster.items()}
    ranked = {name: ranked_samples(rna[name], roster[name], groups[name], rna[name].columns) for name in NAMES}
    global_pan = fold_order(sources.lineage_group, "h1-pan-outer")
    group_map = {g: i % 3 for i, g in enumerate(global_pan)}
    for name in ("LIMORE", "LICOB"):
        group_map.update({g: i % 3 for i, g in enumerate(fold_order(groups[name], f"h1-{name}-outer"))})
    files = {}
    tasks = []
    groups_with_prior = {name: np.array(sorted(sets[name].lineage_group.unique())) for name in PAN}
    for study in (*NAMES, TRANSFER, AUXILIARY):
        cross_study = study in target
        if cross_study:
            cohort = "LIMORE"
            holdout = target[study]
            panel = info["licob_transfer_panels"]["LIMORE"] if study == TRANSFER else info["wong_limore_panel"]
            if len(panel) < 23:
                continue
        else:
            cohort = holdout = study
            panel = info["cohorts"][study]["panel"]
        directory = out / study
        directory.mkdir()
        chem = metadata(roster[cohort]).loc[panel]
        chem.to_csv(directory / "chemistry.tsv", sep="\t")
        files[str((directory / "chemistry.tsv").relative_to(out))] = sha256(directory / "chemistry.tsv")
        fold_ids = np.array([group_map[g] for g in groups[cohort]])
        actual = panel_matrix(sets[cohort], groups[cohort], panel)
        hold = panel_matrix(sets[holdout], groups[holdout], panel) if cross_study else actual
        if study == TRANSFER:
            hold = 1. + hold
        else:
            if study == AUXILIARY:
                hold = 1. + hold
        tissue = (roster[holdout][["lineage_group", TISSUE]].drop_duplicates("lineage_group")
                  .set_index("lineage_group").reindex(groups[holdout])[TISSUE].fillna("Unspecified").to_numpy(str))
        if cross_study and set(groups[cohort]) & set(groups[holdout]):
            raise ValueError("Source and target groups overlap")
        peers = [name for name in PAN if name != study]
        prior = {name: panel_matrix(sets[name], groups_with_prior[name], panel) for name in peers}
        prior_names = np.array(peers)
        for fold in (range(1) if cross_study else range(3)):
            train = np.ones(len(actual), dtype=bool) if cross_study else fold_ids != fold
            test = np.ones(len(hold), dtype=bool) if cross_study else ~train
            fit_groups = groups[cohort][train]
            query_groups = groups[holdout][test]
            if set(fit_groups) & set(query_groups):
                raise ValueError("Fit/outer group overlap")
            name = f"{study}-{fold}"
            tasks.append(name)
            train_path = directory / f"{fold}-develop.npz"
            truth_path = sealed / f"{name}.npz"
            query_path = directory / f"{fold}-query.npz"
            packed = {}
            for source in peers:
                peer_groups = groups_with_prior[source]
                eligible = np.isin(peer_groups, fit_groups) if study in PAN else ~np.isin(peer_groups, query_groups)
                packed[f"prior_{source}"] = prior[source][eligible]
                packed[f"ids_{source}"] = peer_groups[eligible]
            np.savez_compressed(train_path, y=actual[train], groups=fit_groups, panel=np.array(panel),
                                x=ranked[cohort][train] if not cross_study else np.empty((train.sum(), 0)),
                                tissue=(roster[cohort].drop_duplicates("lineage_group").set_index("lineage_group")
                                        .reindex(fit_groups)[TISSUE].fillna("Unspecified").to_numpy(str)),
                                prior_names=prior_names, **packed)
            np.savez_compressed(query_path, groups=query_groups, panel=np.array(panel), tissue=tissue[test],
                                x=ranked[cohort][test] if not cross_study else np.empty((test.sum(), 0)))
            np.savez_compressed(truth_path, y=hold[test], groups=query_groups, panel=np.array(panel))
            for path in (train_path, query_path, truth_path):
                files[str(path.relative_to(out))] = sha256(path)
    json_write(out / "receipt.json", {"support_sha256": sha256(support), "files": files, "tasks": tasks,
                                      "outcome_custodian": "prepare", "Wong_workbook_sha256": sha256(
                                          BASE / "licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx")})
    print(f"Signed {len(tasks)} development packets and held outcome packets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("support", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    prepare(args.support, args.out)
