"""Select a provisional research candidate slate from authorized development inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .functional import (
    available_rna, chemistry_matrix, fit_arm, fit_base, predict_arm, structure_similarity,
)
from .function_data import canonical_chemistry
from .io import publish_json, sha256

SEED = 63012
STRATA = ("strong", "intermediate", "weak")
STAT_COLUMNS = ["cell_id", "cell_group", "compound_id", "isomeric_smiles", "endpoint_value"]


def development_grouped_labels(path: Path, roster: pd.DataFrame, expected: int):
    selected = roster["confirmation_query"].astype(str).str.lower()
    if set(selected) != {"true", "false"}:
        raise ValueError("External roster must carry explicit query statuses.")
    protected = roster.loc[selected.eq("true"), "cell_group"]
    if len(protected) != expected or protected.nunique() != expected:
        raise ValueError("External roster does not match the registered unique-family count.")
    if not roster["cell_id"].is_unique:
        raise ValueError("External roster cell identities are not unique.")
    allowed = roster.loc[~roster["cell_group"].isin(protected), ["cell_id", "cell_group"]]
    observations = pd.read_parquet(path, columns=STAT_COLUMNS,
                                  filters=[("cell_id", "in", sorted(allowed["cell_id"]))])
    if set(observations["cell_group"]) & set(protected):
        raise ValueError("A protected family appeared in development endpoints.")
    assert len(observations[["cell_id", "cell_group"]].drop_duplicates().merge(allowed)) == len(allowed)
    observations = observations.rename(columns={"cell_group": "lineage_group"})
    return observations, sorted(protected), sorted(allowed["cell_group"].unique())


def state_support(path: Path):
    columns = ["cell_id", "context_role", "smiles", "time_h"]
    rows = pd.read_csv(path, sep="\t", usecols=columns).drop_duplicates()
    rows = rows.loc[rows["context_role"].eq("adult_hcc") & rows["time_h"].eq(24)].copy()
    rows["compound_id"] = rows["smiles"].map(lambda smiles: canonical_chemistry(smiles)[1])
    return rows.groupby("compound_id")["cell_id"].agg(lambda values: ",".join(sorted(set(values))))


def ranking_disagreement(observations: pd.DataFrame, rna: pd.DataFrame) -> pd.Series:
    """Cross-fit leave-one-lineage-out RNA-neighbor and drug-mean rankings."""
    differences = []
    with threadpool_limits(limits=1):
        for lineage in sorted(observations["lineage_group"].unique()):
            fit = observations.loc[observations["lineage_group"].ne(lineage)].reset_index(drop=True)
            held = observations.loc[observations["lineage_group"].eq(lineage)].copy()
            query = held[["cell_id", "compound_id", "isomeric_smiles"]]
            base = fit_base(fit, rna)
            for label, method, setting in (("mean", "drug_mean", 0), ("neighbor", "rna_neighbor", 5)):
                model = fit_arm(base, fit, rna, method, setting)
                held[label] = predict_arm(model, query, rna)
            rank = held.groupby("cell_id")[["mean", "neighbor"]].rank(pct=True)
            held["rank_gap"] = (rank["mean"] - rank["neighbor"]).abs()
            differences.append(held[["lineage_group", "compound_id", "rank_gap"]])
    result = pd.concat(differences, ignore_index=True)
    return result.groupby(["lineage_group", "compound_id"])["rank_gap"].mean().groupby("compound_id").mean()


def catalogue(prepared: Path, observations: pd.DataFrame, context: pd.Series, support: pd.Series):
    columns = ["compound_id", "source_compound_label", "vendor_code", "vendor_cas", "pubchem_cid",
               "isomeric_smiles", "drug_target_source_labels", "Category", "Pathway",
               "connectivity_group", "scaffold_group"]
    catalog = pd.read_csv(prepared / "compound_registry.tsv", sep="\t", usecols=columns, dtype=str)
    catalog = catalog.dropna(subset=["compound_id"])
    if not catalog["compound_id"].is_unique:
        raise ValueError("Ambiguous catalog compound identities.")
    aggregate = observations.groupby(["lineage_group", "compound_id"])["endpoint_value"].mean()
    activity = aggregate.groupby("compound_id").agg(["median", "count"])
    activity = activity.sort_values("median", ascending=False, kind="stable")
    tiers = np.array(STRATA)
    activity["activity_stratum"] = tiers[np.arange(len(activity)) * len(tiers) // len(activity)]
    candidates = catalog.set_index("compound_id").join(activity, how="inner", validate="one_to_one")
    candidates = candidates.join(support.rename("eligible_24h_state_contexts"))
    candidates["eligible_24h_state_contexts"] = candidates["eligible_24h_state_contexts"].fillna("")
    candidates["state_context_count"] = candidates["eligible_24h_state_contexts"].map(
        lambda contexts: len(contexts.split(",")) if contexts else 0
    )
    candidates = candidates.join(context.rename("dev_rank_gap"))
    return candidates


def assign(candidates: pd.DataFrame):
    size = len(candidates)
    ids = sorted(candidates.index)
    bits, _ = chemistry_matrix(candidates.loc[ids])
    similarity = pd.DataFrame(structure_similarity(bits, bits), index=ids, columns=ids)
    chosen = {}
    rng = np.random.default_rng(SEED)

    def eligible(tier):
        remaining = candidates.index.difference(chosen)
        used = {candidates.at[item, "connectivity_group"] for item in chosen}
        part = candidates.loc[remaining]
        return part.loc[part["activity_stratum"].eq(tier) & ~part["connectivity_group"].isin(used)]

    def diversity(index):
        if not chosen:
            return np.ones(len(index))
        return 1 - similarity.loc[index, list(chosen)].max(axis=1).to_numpy()

    def choose(tier, role, rna=False):
        part = eligible(tier).copy().sort_index()
        if part.empty:
            raise ValueError(f"Activity tier ran out of eligible distinct chemistry: {tier}")
        represented = Counter(candidates.at[item, "Category"] for item in chosen)
        part["min_chemical_distance"] = diversity(part.index)
        part["category_novelty"] = 1 / (1 + part["Category"].map(represented).fillna(0))
        part["diversity"] = part["min_chemical_distance"] + 0.40 * part["category_novelty"]
        if role == "relative_rank_disagreement":
            order = ["dev_rank_gap", "diversity"]
        else:
            order = ["state_context_count", "diversity"] if rna else ["diversity"]
        ordered = part.sort_values(order, ascending=False, kind="stable")
        if role == "seeded_diversity_audit":
            selected = rng.choice(ordered.head(8).index.to_numpy())
        else:
            selected = ordered.index[0]
        chosen[selected] = role

    # Interleave response strata so the first block contributes four RNA slots per stratum.
    for _ in range(4):
        for tier in STRATA:
            choose(tier, "RNA_coverage_and_diversity", rna=True)
    if len(chosen) != 12:
        raise AssertionError("The matched RNA subset needs twelve compounds.")
    for tier in STRATA:
        choose(tier, "relative_rank_disagreement")
    for tier in ("intermediate", "weak"):
        choose(tier, "seeded_diversity_audit")
    for tier in STRATA:
        already = sum(candidates.at[compound, "activity_stratum"] == tier for compound in chosen)
        for _ in range(8 - already):
            choose(tier, "functional_chemistry_target_diversity")
    if len(chosen) != 24 or size < len(chosen):
        raise AssertionError("The proposed panel needs twenty-four distinct assay compounds.")
    slate = candidates.loc[list(chosen)].copy()
    slate.insert(0, "selection_role", list(chosen.values()))
    slate.insert(0, "RNA_planning_subset", [index < 12 for index in range(len(slate))])
    slate.insert(0, "drug_slot", [f"P{index:02}" for index in range(1, len(slate) + 1)])
    slate["nearest_other_panel_tanimoto"] = [
        float(similarity.loc[compound, [other for other in slate.index if other != compound]].max())
        for compound in slate.index
    ]
    return slate.reset_index()


def run(prepared: Path, roster_path: Path, states: Path, layout: Path, out: Path):
    roster = pd.read_csv(roster_path, sep="\t", dtype=str)
    observations, protected, dev = development_grouped_labels(prepared / "rows.parquet", roster, 50)
    rna = available_rna(pd.read_parquet(prepared / "rna.parquet").loc[sorted(observations["cell_id"].unique())])
    state = state_support(states)
    rank_gap = ranking_disagreement(observations, rna)
    slate = assign(catalogue(prepared, observations, rank_gap, state))
    proposed = set(slate["drug_slot"])
    design = pd.read_csv(layout, sep="\t", dtype=str)
    assert proposed == set(design.loc[design["measurement"].eq("function_signal") & design["condition"].eq("treated"), "drug_slot"])
    rna_slots = set(design.loc[design["measurement"].eq("rna") & design["condition"].eq("treated"), "drug_slot"])
    assert set(slate.loc[slate["RNA_planning_subset"], "drug_slot"]) == rna_slots
    result_file = out / "provisional_compound_slate.tsv"
    out.mkdir(parents=True, exist_ok=False)
    output_columns = ["drug_slot", "RNA_planning_subset", "selection_role", "source_compound_label",
                      "compound_id", "vendor_code", "pubchem_cid", "drug_target_source_labels",
                      "Category", "Pathway", "activity_stratum", "count", "median", "dev_rank_gap",
                      "eligible_24h_state_contexts", "nearest_other_panel_tanimoto"]
    (slate[output_columns].rename(columns={"count": "dev_lineages", "median": "dev_median_AA"})
     .to_csv(result_file, index=False, sep="\t", float_format="%.5f"))
    rna_rows = slate.loc[slate["RNA_planning_subset"]]
    summary = {
        "planning_only": True, "random_seed": SEED, "development_cells": int(observations["cell_id"].nunique()),
        "development_lineages": len(dev), "development_lineage_ids": dev, "development_response_rows": len(observations),
        "protected_lineages": len(protected), "excluded_group_overlap": len(set(protected) & set(dev)),
        "panel_unique_connectivity": int(slate["connectivity_group"].nunique()),
        "panel_distinct_scaffolds": int(slate["scaffold_group"].nunique()),
        "panel_response_strata": {key: int(value) for key, value in slate["activity_stratum"].value_counts().items()},
        "panel_target_categories": int(slate["Category"].nunique()),
        "panel_publisher_pathway_categories": int(slate["Pathway"].nunique()),
        "RNA_supported_in_JHH_24h_metadata": int(rna_rows["state_context_count"].gt(0).sum()),
        "RNA_supported_in_both_JHH_contexts": int(rna_rows["state_context_count"].eq(2).sum()),
        "inputs": {str(path): sha256(path) for path in (
            prepared / "rows.parquet", prepared / "rna.parquet", prepared / "compound_registry.tsv",
            roster_path, states, layout, Path(__file__),
        )},
        "manifest_sha256": sha256(result_file),
    }
    publish_json(out / "pilot_slate_receipt.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--holdout-roster", type=Path, required=True)
    parser.add_argument("--state-metadata", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    run(options.prepared, options.holdout_roster, options.state_metadata, options.layout, options.out)


if __name__ == "__main__":
    main()
