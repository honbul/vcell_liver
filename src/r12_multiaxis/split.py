"""Pin metadata-only compound assignments and the paired physical-well row universe."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE, POOL, checked_table, table


def common_rows(folder, controls, minimum_groups):
    treated = checked_table(folder, "treated_rows.tsv")
    treated = treated.loc[treated.eligible_cell_count].copy()
    for role in ("A", "B"):
        control = checked_table(controls, f"{role}_rows.tsv")
        control = control.loc[control.eligible_cell_count, ["plate_id", LINE, KEY[0]]]
        treated = treated.merge(control.rename(columns={KEY[0]: f"control_{role}"}),
                                on=["plate_id", LINE], validate="many_to_one")
    if treated.duplicated(KEY).any() or treated.control_A.eq(treated.control_B).any():
        raise ValueError("Duplicate treated keys or shared control well")
    return eligible_pools(treated, minimum_groups)


def eligible_pools(rows, minimum_groups):
    support = rows.groupby(POOL, sort=False)[GROUP].transform("nunique")
    return rows.loc[support.ge(minimum_groups)].reset_index(drop=True)


def balanced_partition(wells, n_folds, seed, blocks=None):
    """Keep blocks intact while balancing compound support across all observed plates."""
    counts = pd.crosstab(wells[GROUP], wells.plate_id).gt(0).astype(int)
    if blocks is None:
        blocks = {g: g for g in counts.index}
    tags = counts.index.map(blocks)
    if tags.hasnans:
        raise ValueError("An identity group lacks a split block")
    support = counts.groupby(tags).sum()
    sizes = counts.groupby(tags).size()
    rng = np.random.default_rng(seed)
    order = support.assign(size=sizes, tie=rng.random(len(support)))
    order = order.sort_values(["size", "tie"], ascending=[False, True]).index
    ideal = support.sum().to_numpy() / n_folds
    ideal_size = len(counts) / n_folds
    loads = np.zeros((n_folds, support.shape[1]))
    totals = np.zeros(n_folds)
    labels = {}
    for block in order:
        v = support.loc[block].to_numpy()
        n = sizes.loc[block]
        plate_cost = ((2 * loads * v + v**2) / np.maximum(ideal, 1)**2).sum(axis=1)
        size_cost = (2 * totals * n + n**2) / ideal_size**2
        k = np.argmin(plate_cost + size_cost + 1e-9 * rng.random(n_folds))
        loads[k] += v
        totals[k] += n
        labels[block] = int(k)
    return {int(g): labels[blocks[g]] for g in counts.index}


def assignments(rows, seed, minimum_groups, blocks=None):
    outer = balanced_partition(rows, 3, seed, blocks)
    rows = rows.assign(outer_fold=rows[GROUP].map(outer))
    recorded = []
    inner = {}
    for k in range(3):
        outer_fit = rows.loc[rows.outer_fold.ne(k)]
        labels = balanced_partition(outer_fit, 2, seed + k + 1, blocks)
        inner[k] = labels
        segment = rows.assign(
            evaluation_outer=k,
            usage=np.where(rows.outer_fold.eq(k), "outer_query", "outer_train"),
            inner_fold=rows[GROUP].map(labels).fillna(-1).astype(int),
        )
        for kind, fit_rows in segment.groupby("inner_fold"):
            usable = eligible_pools(fit_rows, minimum_groups)
            if set(usable.plate_id) != set(rows.plate_id):
                raise ValueError(f"Inadequate plate support in partition {k}/{kind}")
            segment.loc[fit_rows.index, "query_pool_eligible"] = (
                fit_rows.groupby(POOL)[GROUP].transform("nunique").ge(minimum_groups).to_numpy()
            )
        for role in ("A", "B"):
            view = segment.assign(role=role, own_control=segment[f"control_{role}"])
            recorded.append(view)
    return outer, inner, pd.concat(recorded, ignore_index=True)


def exposure(repo, samples, output):
    historical = table(repo / "manifests/chemcpa_moe_v2/splits.tsv").set_index("group_index").split
    relevant = samples.loc[~samples.quarantined & samples.control_role.eq("treated")]
    entries = []
    for split, group in relevant.groupby("split"):
        old = group[GROUP].map(historical)
        if not old.eq(split).all():
            raise ValueError("R11 historical identity IDs disagree with chemCPA roles")
        entries.append({
            "R11_split": split,
            "eligible_metadata_wells": len(group),
            "metadata_groups": group[GROUP].nunique(),
            "chemCPA_split": split,
            "chemCPA_exposure": {"train": "fit", "val": "checkpoint, learning rate and model selection",
                                 "test": "scored/re-scored"}[split],
            "R11_exposure": {"train": "fit and development CV", "val": "unused in R11",
                             "test": "one project-seen evaluation"}[split],
            "R12_route": {"train": "N/source", "val": "excluded: project development exposure",
                          "test": "project-seen replay only"}[split],
            "evidence": "manifests/chemcpa_moe_v2/splits.tsv; reports/chemcpa_moe_v2/RESULTS_FIXED.md; "
                        "reports/r11_chemical_probe/RESULTS.md",
        })
    outside = table(repo / "reports/r11_chemical_probe/manifest/outside_r6.tsv")
    entries.append(dict(R11_split="outside_r6", eligible_metadata_wells=len(outside),
                        metadata_groups=outside.drug.nunique(), chemCPA_split="unassigned",
                        chemCPA_exposure="unconfirmed", R11_exposure="excluded",
                        R12_route="no adequate verified unused holdout", evidence="R11 outside_r6.tsv"))
    pd.DataFrame(entries).to_csv(output, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r11", type=Path)
    args = parser.parse_args()
    cfg = config(args.config)
    if cfg["path"] != "N":
        raise ValueError("This frozen study registers route N")
    r11 = args.r11 or Path(cfg["r11_assets"])
    args.out.mkdir(parents=True, exist_ok=False)
    samples = table(r11 / "metadata/samples.tsv")
    rows = common_rows(r11 / "source", r11 / "source", cfg["minimum_groups"])
    if set(rows.plate_id) != set(cfg["plates"]) or not rows.dose_uM.eq(cfg["dose_uM"]).all():
        raise ValueError("R11 source disagrees with the registered primary condition")
    outer, inner, manifest = assignments(rows, cfg["split_seed"], cfg["minimum_groups"])
    manifest_columns = KEY + ["compound_entity_id", GROUP] + [
        "plate_id", "dose_uM", "time_h", "role", "own_control", "outer_fold",
        "evaluation_outer", "inner_fold", "usage", "query_pool_eligible",
    ]
    manifest.to_csv(args.out / "split_manifest.tsv", columns=manifest_columns, sep="\t", index=False)
    exposure(Path.cwd(), samples, args.out / "exposure_ledger.tsv")
    rows.to_csv(args.out / "source_rows.tsv", sep="\t", index=False)
    write_json(args.out / "split.json", dict(route="N", outer=outer, inner=inner,
               source_rows=len(rows), source_groups=rows[GROUP].nunique(),
               source_wells=rows[KEY[0]].nunique(), config_sha256=sha256(args.config),
               manifest_sha256=sha256(args.out / "split_manifest.tsv")))
    print(pd.crosstab(rows[GROUP].map(outer), rows.plate_id).to_string(), flush=True)


if __name__ == "__main__":
    main()
