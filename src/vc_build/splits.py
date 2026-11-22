"""Grouped development and outer holdouts for the functional endpoint."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd


def chemical_groups(rows: pd.DataFrame) -> pd.Series:
    """Join chemical families and non-empty scaffolds transitively."""
    chemical = rows[["compound_id", "connectivity_group", "scaffold_group"]].drop_duplicates()
    if chemical["compound_id"].duplicated().any():
        raise ValueError("Each compound must have one family and scaffold.")
    unions = {drug: drug for drug in chemical["compound_id"]}

    def representative(drug):
        while drug != unions[drug]:
            drug = unions[drug]
        return drug

    tokens = {}
    for drug, family, scaffold in chemical.itertuples(index=False, name=None):
        for category, key in (("family", family), ("scaffold", scaffold)):
            if pd.isna(key) or not str(key).strip():
                continue
            token = (category, key)
            previous = tokens.setdefault(token, drug)
            current_root, previous_root = sorted((representative(drug), representative(previous)))
            unions[previous_root] = current_root
    return rows["compound_id"].map({drug: representative(drug) for drug in unions})


def group_folds(groups: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Assign groups by stable hash, balancing observed row counts greedily."""
    sizes = groups.value_counts(dropna=False)
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError("Split grouping identifiers must be present.")
    if len(sizes) < n_folds:
        raise ValueError(f"Need at least {n_folds} distinct groups, got {len(sizes)}.")

    def sort_key(group):
        return -sizes[group], sha256(f"{seed}:{group}".encode()).digest()

    fold_sizes = [0] * n_folds
    assignments = {}
    for group in sorted(sizes.index, key=sort_key):
        fold = min(range(n_folds), key=fold_sizes.__getitem__)
        assignments[group] = fold
        fold_sizes[fold] += sizes[group]
    return groups.map(assignments).to_numpy()


def assignments(rows: pd.DataFrame, track: str, n_folds: int, seed: int) -> pd.DataFrame:
    """Emit fit, withheld, and embargo roles for every row in every fold."""
    if track not in {"C", "D", "CD"}:
        raise ValueError(f"Unknown assessment track {track!r}.")
    cell_folds = group_folds(rows["lineage_group"], n_folds, seed) if "C" in track else None
    drug_folds = group_folds(rows["chemical_group"], n_folds, seed + 1) if "D" in track else None
    folds = []
    for fold in range(n_folds):
        if track == "C":
            held_out = cell_folds == fold
            trained = ~held_out
        elif track == "D":
            held_out = drug_folds == fold
            trained = ~held_out
        else:
            held_out = (cell_folds == fold) & (drug_folds == fold)
            trained = (cell_folds != fold) & (drug_folds != fold)
        if not held_out.any() or not trained.any():
            raise ValueError(f"{track} fold {fold} has no supported training or holdout rows.")
        fit, test = rows.loc[trained], rows.loc[held_out]
        if "C" in track and set(fit.lineage_group) & set(test.lineage_group):
            raise AssertionError("Related cell models crossed the fit/holdout boundary.")
        if "D" in track and set(fit.chemical_group) & set(test.chemical_group):
            raise AssertionError("Related compounds crossed the fit/holdout boundary.")
        frame = rows[["row_id", "cell_id", "lineage_group", "compound_id", "chemical_group"]].copy()
        frame["track"], frame["fold"] = track, fold
        frame["role"] = np.select([trained, held_out], ["fit", "withheld"], default="embargo")
        folds.append(frame)
    return pd.concat(folds, ignore_index=True)
