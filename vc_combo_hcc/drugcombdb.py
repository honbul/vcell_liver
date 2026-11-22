"""Register the DrugCombDB O'Neil release with measured dose-matrix efficacy."""
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from .acquire_drugcombdb import FILES
from .identity import exact_token, structural_identity, unique_map
from .splits import record


STUDY = "ONEIL"
SCORE_FIELDS = ["ID", "Drug1", "Drug2", "Cell line", "ZIP", "Bliss"]
WELL_FIELDS = ["BlockID", "DrugRow", "DrugCol", "ConcRow", "ConcCol", "Response", "source"]


def crosswalk(path, compounds):
    rows = pd.read_csv(path, usecols=["drugName", "drugNameOfficial", "smilesString"],
                       dtype=str, encoding="gb18030")
    registered = set(compounds.keys.index)
    aliases = []
    for row in rows.dropna(subset=["smilesString"]).itertuples(index=False):
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(row.smilesString)
            if mol is None:
                continue
            native = Chem.MolToInchiKey(mol)
        identity = structural_identity(row.smilesString, native)
        if identity is None or identity[0] not in registered:
            continue
        parent = identity[0]
        for name in (row.drugName, row.drugNameOfficial):
            if pd.notna(name):
                aliases.append((exact_token(name), parent))
    return unique_map(aliases)


def block_extrema(well_file, wanted, molecular):
    pieces, mixed, counts = [], set(), Counter()
    for chunk in pd.read_csv(well_file, usecols=WELL_FIELDS, chunksize=200_000,
                             dtype={"BlockID": str, "DrugRow": str, "DrugCol": str, "source": str},
                             na_values=["NA", "NULL", "\\N"]):
        study = chunk.source.str.strip().str.upper()
        counts.update(study.value_counts().to_dict())
        ids = chunk.BlockID.isin(wanted.index)
        mixed.update(chunk.loc[ids & study.ne(STUDY), "BlockID"])
        use = chunk.loc[ids & study.eq(STUDY)].copy()
        if use.empty:
            continue
        a = use.DrugRow.map(exact_token).map(molecular)
        b = use.DrugCol.map(exact_token).map(molecular)
        one = use.BlockID.map(wanted.compound_a)
        two = use.BlockID.map(wanted.compound_b)
        agreement = (a.eq(one) & b.eq(two)) | (a.eq(two) & b.eq(one))
        mixed.update(use.loc[~agreement, "BlockID"])
        z = use[agreement & use[["ConcRow", "ConcCol", "Response"]].notna().all(axis=1)].copy()
        z["forward"] = a.eq(one)
        if z.empty:
            continue
        left = z.ConcRow.gt(0)
        right = z.ConcCol.gt(0)
        zero = z.ConcRow.eq(0) & z.ConcCol.eq(0)
        z["vehicle"] = z.Response.where(zero)
        z["combo"] = z.Response.where(left & right)
        mono_left = left & z.ConcCol.eq(0)
        mono_right = z.ConcRow.eq(0) & right
        z["mono_a"] = z.Response.where((mono_left & z.forward) | (mono_right & ~z.forward))
        z["mono_b"] = z.Response.where((mono_right & z.forward) | (mono_left & ~z.forward))
        by_id = z.groupby("BlockID").agg(
            vehicle_sum=("vehicle", "sum"), vehicle_count=("vehicle", "count"),
            combo=("combo", "min"), mono_a=("mono_a", "min"), mono_b=("mono_b", "min"),
        )
        pieces.append(by_id)
    if not pieces:
        raise ValueError("No DrugCombDB dose matrices match scored oncology records")
    # Handle a block spanning multiple file chunks without assuming sort order.
    grouped = pd.concat(pieces).groupby(level=0).agg({
        "vehicle_sum": "sum", "vehicle_count": "sum", "combo": "min", "mono_a": "min", "mono_b": "min",
    })
    control = grouped.vehicle_sum / grouped.vehicle_count
    keep = (control.between(70, 130) & grouped[["combo", "mono_a", "mono_b"]].apply(np.isfinite).all(axis=1)
            & ~grouped.index.isin(mixed))
    highest_mono = grouped.loc[keep, ["mono_a", "mono_b"]].min(axis=1)
    gain = 100 * (highest_mono - grouped.loc[keep, "combo"]) / control[keep]
    return gain, {"raw_wells_by_study": dict(counts), "applicable_well_blocks": len(grouped),
                  "inconsistent_identity_or_study": len(mixed & set(grouped.index)), "complete_vehicles_and_arms": len(gain)}


def oncology_rows(directory, cells, compounds):
    assets = {name: record(directory / name) for name in FILES}
    for name, fingerprint in assets.items():
        if fingerprint["bytes"] != FILES[name][0]:
            raise ValueError(f"Incomplete original DrugCombDB file: {name}")
    molecular = crosswalk(directory / "drug_chemical_info.csv", compounds)
    raw = pd.read_csv(directory / "drugcombs_scored.csv", usecols=SCORE_FIELDS,
                      dtype=dict.fromkeys(SCORE_FIELDS[:4], str))
    if not raw.ID.is_unique:
        raise ValueError("DrugCombDB distributes one scored row per published block ID")
    data = pd.DataFrame({
        "id": "DrugCombDB:" + STUDY + ":" + raw.ID, "source": "DrugCombDB:" + STUDY,
        "source_liver": False, "cell_id": cells.lookup(raw["Cell line"]),
        "compound_a": raw.Drug1.map(exact_token).map(molecular),
        "compound_b": raw.Drug2.map(exact_token).map(molecular),
        "zip": pd.to_numeric(raw.ZIP, errors="coerce"),
        "bliss": pd.to_numeric(raw.Bliss, errors="coerce"),
        "css": np.nan,
    }, index=raw.index)
    data = data.set_index(raw.ID)
    data = data.dropna(subset=["cell_id", "compound_a", "compound_b", "zip", "bliss"])
    data = data[np.isfinite(data.zip) & np.isfinite(data.bliss)]
    gain, stats = block_extrema(directory / "drugcombs_response.csv", data, molecular)
    data["added_inhibition"] = gain
    matched = data.dropna(subset=["added_inhibition"])
    return matched, {
        "files": assets, "source": STUDY, "published_scores": len(raw), "crosswalk_names": len(molecular),
        "scored_rows_with_exact_identities": len(data), "dose_matched_rows": len(matched), **stats,
        "efficacy_endpoint": "100*(minimum single-agent viability - minimum combination viability)/mean vehicle viability",
        "deduplication": "Use the original DrugCombDB O'Neil release and remove the mirrored O'Neil DrugComb rows.",
        "reference": "https://doi.org/10.1093/nar/gkz1007",
    }
