"""Recompute both H2 24h MODZ aggregations after relation-aware reference removal."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h2.assets import KEY, LINCS, modz_weights, parquet_lock
from nextgen_h2.run import read_json
from nextgen_h2_v2.assets import LINCS_SIGNATURE_SHA, V1_ASSETS, lincs_block, summarized
from vc_build.io import publish_json, sha256

from .common import keys_hash, registered


def contexts(signature, values):
    grouped = signature.groupby([KEY, "cell_iname", "time_h", "dose_um"], sort=True).indices
    index = pd.MultiIndex.from_tuples(grouped, names=[KEY, "cell_iname", "time_h", "dose_um"])
    aggregated = []
    for rows in grouped.values():
        block = values[rows]
        aggregated.append(modz_weights(block) @ block)
    legacy = pd.DataFrame(np.stack(aggregated), index=index)
    return index.to_frame(index=False), legacy


def build(registration, out):
    out.mkdir(exist_ok=False, parents=True)
    registry, _, _, assets = registered()
    split_path = registration / "split_manifest.json"
    split = read_json(split_path)
    source = parquet_lock(V1_ASSETS/"lincs_selected_signatures.parquet", LINCS_SIGNATURE_SHA)
    source = source[source.time_h.eq(24)].reset_index(drop=True)
    gene_file = LINCS/"geneinfo_beta.txt"
    if sha256(gene_file) != assets["lincs_metadata_sha256"][gene_file.name]:
        raise ValueError("Changed L1000 gene metadata")
    genes = pd.read_csv(gene_file, sep="\t", dtype={"gene_id": str}).set_index("gene_id")
    genes = genes.loc[genes.feature_space.eq("landmark")]
    assert len(genes) == 978 and genes.gene_symbol.is_unique
    values = lincs_block(source, genes)
    keys, legacy = contexts(source, values)
    extended = legacy.to_numpy(dtype=np.float32)
    del values
    comparison = {}
    for role in [{"name": "reference-full-check", "excluded_references": []}, *split["outer"], *split["inner"]]:
        tag, excluded = role["name"], set(role["excluded_references"])
        admitted = ~keys.cell_iname.isin(excluded)
        candidate = keys[admitted]
        if set(candidate.cell_iname) & excluded:
            raise ValueError("A held cell entered LINCS consensus")
        allowed = legacy.loc[admitted.to_numpy()]
        current_v1 = allowed.groupby(level=[0, 1], sort=True).mean().groupby(level=0, sort=True).mean()
        current_v1.columns = genes.gene_symbol
        current_v2 = summarized(keys[admitted].reset_index(drop=True), extended[admitted.to_numpy()], "mean")
        current_v2.columns = [f"x{i}" for i in range(len(genes))]
        assert current_v1.index.equals(current_v2.index)
        if tag == "reference-full-check":
            original = parquet_lock(V1_ASSETS/"lincs_modz24.parquet", assets["files"]["lincs_modz24.parquet"])
            assert set(original.index) == set(current_v1.index)
            assert len(registry.index.intersection(current_v1.index)) == 724
            delta = float(np.abs(original.loc[current_v1.index, current_v1.columns].to_numpy() -
                                 current_v1.to_numpy()).max())
            if delta >= 3e-6:
                raise ValueError(f"Re-aggregation at full reference does not match prior L1000 profile: {delta}")
            comparison["full_reference_replay_max_abs_diff"] = delta
            continue
        if not excluded:
            raise ValueError(f"Reference masking is empty for role {tag}")
        frame = pd.concat([current_v1, current_v2], axis=1, keys=["v1", "v2"])
        file = out/f"{tag}.parquet"
        frame.to_parquet(file)
        coverage = set(registry.index) & set(frame.index)
        def counts(keys):
            return len(coverage & set(keys))
        comparison[tag] = {
            "profile_sha256": sha256(file), "excluded_references": sorted(excluded),
            "used_24h_references": sorted(candidate.cell_iname.unique()),
            "retained_signatures": int((~source.cell_iname.isin(excluded)).sum()),
            "retained_contexts": len(candidate), "all_profile_parents": len(frame),
            "source_observed_parents": len(coverage), "fit_observed_parents": counts(role["fit_drugs"]),
            "query_observed_parents": counts(role["query_drugs"]),
            "registered_supported_parents_sha256": keys_hash(coverage), "zero_excluded_references_used": True,
        }
        print(tag, len(coverage), len(candidate.cell_iname.unique()), flush=True)
    publish_json(out/"receipt.json", {
        "split_sha256": sha256(split_path), "signatures_sha256": LINCS_SIGNATURE_SHA,
        "genefield_sha256": sha256(gene_file), "registered_24h_signatures": len(source),
        "registered_reference_cells": 147, "active_24h_reference_cells": source.cell_iname.nunique(),
        "registered_fit_and_query_reaggregated": True, "source_outcome_columns_read": [], "roles": comparison,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    build(args.registration, args.out)
