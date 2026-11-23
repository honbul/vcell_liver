"""Validate split geometry and source eligibility without reading any outcome column."""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from vc_build.io import publish_json, sha256

from .assets import COLD, KEY, fold_ids, parquet_lock
from .run import read_json, subfold


def run(assets, out):
    receipt = read_json(assets / "data_receipt.json")
    registry = parquet_lock(assets / "folds.parquet", receipt["files"]["folds.parquet"])
    source = pd.read_parquet(COLD / "train.parquet", columns=[KEY, "source_id", "endpoint_id", "cell_id"])
    if not source.groupby("source_id").endpoint_id.nunique().eq(1).all():
        raise ValueError("Multiple endpoint types found in one native source")
    proof = []
    by_source = {screen: group.set_index([KEY, "cell_id"]).index for screen, group in source.groupby("source_id")}
    anchors = {}
    for left, right in combinations(by_source, 2):
        common = by_source[left].intersection(by_source[right])
        count = common.to_frame(index=False).groupby(KEY).size()
        anchors[(left, right)] = set(count[count.ge(12)].index)

    def split_check(fit, query, label):
        sets = [registry.loc[keys] for keys in (fit, query)]
        for key in ("connectivity", "scaffold_block", "scaffold_aliases"):
            selections = [part[key].str.split("|").explode() for part in sets]
            if set(selections[0]) & set(selections[1]):
                raise ValueError(f"Identity or scaffold crosses {label}")
        counts = {"|".join(screens): len(supported & set(fit)) for screens, supported in anchors.items()}
        if any(count < 3 for screens, count in counts.items() if "GDSC2" in screens):
            raise ValueError(f"Insufficient fit-only source bridge in {label}: {counts}")
        proof.append({"fold": label, "fit": len(fit), "query": len(query), "common_anchor_drugs": counts})

    for outer in range(3):
        train, held = subfold(registry, outer)
        split_check(train, held, f"outer{outer}")
        for inner in range(3):
            fit, check = subfold(registry, outer, inner)
            split_check(fit, check, f"outer{outer}/inner{inner}")
            sub = fold_ids(registry.loc[fit, "scaffold_block"])
            for fold in range(3):
                split_check(sub.index[sub.ne(fold)], sub.index[sub.eq(fold)], f"outer{outer}/inner{inner}/oof{fold}")
    publish_json(out, {"comparisons": proof, "split_sha256": sha256(assets / "folds.parquet"),
                       "source_outcome_columns_read": [], "folds": 3, "inner_folds": 3})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.assets, args.out)
