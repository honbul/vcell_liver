"""Check scaffold and strict known-drug cell-transfer support from metadata only."""

import argparse
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import config, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, POOL, table

from .split import assignments


def scaffold_blocks(entities):
    parent = {int(g): int(g) for g in entities[GROUP]}

    def root(g):
        while parent[g] != g:
            g = parent[g]
        return g

    unique = entities.loc[~entities.scaffold.isin(["", "<acyclic>"])]
    for _, group in unique.groupby("scaffold"):
        components = {root(g) for g in group[GROUP]}
        if components:
            lower = min(components)
            for upper in components:
                parent[upper] = lower
    return {g: root(g) for g in parent}


def transfer_support(rows):
    # An exact entity needs an eligible treatment response on a different physical
    # well and line before it can enter the known-drug strict new-line question.
    records = []
    for line in sorted(rows[LINE].unique()):
        candidates = rows.loc[rows[LINE].eq(line)].copy()
        source = rows.loc[rows[LINE].ne(line), [ENTITY, KEY[0]]].drop_duplicates()
        paired = candidates.merge(source, on=ENTITY, suffixes=("", "_fit"), validate="many_to_many")
        paired = paired.loc[paired[KEY[0]].ne(paired[KEY[0] + "_fit"])]
        for plate, group in paired.groupby("plate_id"):
            records.append({LINE: line, "plate": plate, "groups": group[GROUP].nunique(),
                            "wells": group[KEY[0]].nunique(), "fit_wells": group[KEY[0] + "_fit"].nunique()})
    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = table(args.partition / "source_rows.tsv")
    entities = table(Path(cfg["r11_assets"]) / "features/entities.tsv")
    groups = scaffold_blocks(entities.loc[entities[GROUP].isin(rows[GROUP])])
    pd.Series(groups, name="scaffold_component").rename_axis(GROUP).to_csv(
        args.out / "group_scaffolds.tsv", sep="\t")
    outer, inner, manifest = assignments(rows, cfg["split_seed"] + 1000, cfg["minimum_groups"], groups)
    manifest = manifest[KEY + [ENTITY, GROUP, *[x for x in POOL if x != LINE],
                               "role", "own_control", "outer_fold", "evaluation_outer", "inner_fold",
                               "usage", "query_pool_eligible"]]
    manifest.to_csv(args.out / "split_manifest.tsv", sep="\t", index=False)
    rows.to_csv(args.out / "source_rows.tsv", sep="\t", index=False)
    by_outer = pd.Series(outer).value_counts().sort_index()
    transfer = transfer_support(rows)
    transfer.to_csv(args.out / "strict_cell_support.tsv", sep="\t", index=False)
    strict_max = int(transfer.groups.max()) if not transfer.empty else 0
    write_json(args.out / "split.json", dict(route="N-scaffold", outer=outer, inner=inner,
               source_rows=len(rows), source_groups=rows[GROUP].nunique(),
               source_wells=rows[KEY[0]].nunique(), config_sha256=sha256(args.config),
               manifest_sha256=sha256(args.out / "split_manifest.tsv")))
    write_json(args.out / "feasibility.json", dict(
        scaffold_components=len(set(groups.values())), largest_component=max(pd.Series(groups).value_counts()),
        scaffold_groups_by_outer={str(k): int(n) for k, n in by_outer.items()},
        scaffold_eligible=bool(by_outer.min() >= 30),
        strict_cell_max_groups_per_line_plate=strict_max,
        strict_cell_eligible=strict_max >= cfg["minimum_groups"],
        min_groups=cfg["minimum_groups"],
        strict_cell_condition="same entity in different physical treatment wells, no same-well references",
    ))


if __name__ == "__main__":
    main()
