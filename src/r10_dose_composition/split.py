"""Drug-level outer split (train / development / test), fixed before any outcome is looked at.

Groups: drugs with an identical SMILES string in the original hashTable metadata stay together
(no structure toolkit is used, so scaffold-level grouping is NOT done). Stratum: pathway_level_1.
Inside each stratum the groups are shuffled with the fixed seed and dealt in the repeating
pattern train, train, train, development, test (random start per stratum) -> about 60/20/20.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    w = pd.read_csv(f"{cfg['cache_dir']}/wells.tsv", sep="\t")
    d = (w[~w.vehicle].groupby("drug")
         .agg(product_name=("product_name", "first"), pathway_level_1=("pathway_level_1", "first"),
              SMILES=("SMILES", "first")).reset_index())
    smi = d.SMILES.fillna("").astype(str).str.strip()
    d["group"] = np.where(smi.isin(["", "NA", "nan"]), d.drug, smi)
    gid = {g: i for i, g in enumerate(sorted(d.group.unique()))}
    d["group_id"] = d.group.map(gid)
    rng = np.random.default_rng(cfg["split"]["seed"])
    pattern = ["train", "train", "train", "development", "test"]
    role = {}
    for s, sub in d.groupby("pathway_level_1", sort=True):
        groups = np.array(sorted(sub.group_id.unique()))
        rng.shuffle(groups)
        off = int(rng.integers(0, len(pattern)))
        for i, g in enumerate(groups):
            role[g] = pattern[(i + off) % len(pattern)]
    d["role"] = d.group_id.map(role)
    d["shared_smiles_group"] = d.groupby("group_id").drug.transform("size") > 1
    d = d.drop(columns=["group", "SMILES"]).sort_values(["role", "pathway_level_1", "drug"])
    d.to_csv(args.out, sep="\t", index=False)
    print(d.role.value_counts().to_dict(), "shared-SMILES drugs:", int(d.shared_smiles_group.sum()))
    print(pd.crosstab(d.pathway_level_1, d.role).to_string())


if __name__ == "__main__":
    main()
