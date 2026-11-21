"""Metadata-only checks of the two GEO HepaRG plate designs."""

from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.r14_cross_organ.heparg_prepare import source_index


DOSE_LABELS = ("lt_0.01", "0.01_to_0.1", "0.1_to_1", "1_to_10", "10_to_100", "ge_100")
DOSE_CUTS = (0.01, 0.1, 1.0, 10.0, 100.0)


def decade(values):
    return np.searchsorted(DOSE_CUTS, values, side="right")


def chemical_family(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid mapped parent {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if not scaffold.GetNumAtoms():
        return "acyclic", "acyclic"
    hetero = any(atom.GetAtomicNum() != 6 for atom in scaffold.GetAtoms())
    aromatic = any(atom.GetIsAromatic() for atom in scaffold.GetAtoms())
    coarse = ("heterocyclic" if hetero else "carbocyclic") + ("_aromatic" if aromatic else "_nonaromatic")
    generic = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold))
    return coarse, generic


def released_design(index, dataset):
    design = source_index(index, dataset).loc[lambda fr: ~fr.vehicle].copy()
    groups = design.groupby(["identity", "chemical"], sort=False)
    design["max_uM"] = groups.dose_uM.transform("max")
    design["design_rank"] = groups.dose_uM.rank(method="dense").astype(int)
    design["dose_decade"] = decade(design.dose_uM)
    design["plategroup"] = design.batch.str.extract(r"(?:^|\s)plategroup:(\S+)").fillna("")
    design["block"] = design.batch.str.extract(r"(?:^|\s)block:(\S+)").fillna("")
    return design


def enrich(rows, design):
    matched = design.set_index("library").loc[rows.library, ["max_uM", "dose_decade", "plategroup", "block"]]
    for field in matched:
        rows[field] = matched[field].to_numpy()
    known = rows.loc[rows.parent_key.ne(""), ["parent_key", "smiles"]].drop_duplicates("parent_key")
    labels = {key: chemical_family(smiles) for key, smiles in known.itertuples(index=False, name=None)}
    rows["chemical_family"] = [labels[key][0] if key in labels else "" for key in rows.parent_key]
    rows["generic_scaffold"] = [labels[key][1] if key in labels else "" for key in rows.parent_key]
    return rows


def plate_pairs(design):
    records = []
    plates = {p: t.set_index("position") for p, t in design.groupby("plate", sort=True)}
    for (a, left), (b, right) in combinations(plates.items(), 2):
        common = left.index.intersection(right.index)
        if not len(common):
            continue
        l, r = left.loc[common], right.loc[common]
        same = l.identity.eq(r.identity) & l.chemical.eq(r.chemical)
        same_dose = np.isclose(l.dose_uM, r.dose_uM, rtol=1e-6, atol=1e-12)
        same_max = np.isclose(l.max_uM, r.max_uM, rtol=1e-6, atol=1e-12)
        record = {
            "left_plate": a, "right_plate": b, "shared_treated_positions": len(common),
            "layout_union_positions": len(left.index.union(right.index)),
            "same_chemical": int(same.sum()), "same_chemical_and_actual_uM": int((same & same_dose).sum()),
            "same_actual_uM": int(same_dose.sum()), "same_max_uM": int(same_max.sum()),
            "same_design_rank": int(l.design_rank.eq(r.design_rank).sum()),
        }
        for field in ("plategroup", "culture", "block"):
            x, y = left[field].iloc[0], right[field].iloc[0]
            record[f"left_{field}"] = x
            record[f"right_{field}"] = y
            record[f"same_{field}"] = bool(x and y and x == y)
        records.append(record)
    result = pd.DataFrame(records)
    result["copied_layout"] = (
        result.same_chemical_and_actual_uM.eq(result.shared_treated_positions)
        & result.shared_treated_positions.eq(result.layout_union_positions)
    )
    return result


def metadata_out_of_plate(design):
    frame = design.copy()
    frame["actual_log10_uM"] = np.log10(frame.dose_uM)
    frame["max_log10_uM"] = np.log10(frame.max_uM)
    absolute = ("actual_log10_uM", "max_log10_uM")
    categorical = ("design_rank", "dose_decade", "chemical_family", "generic_scaffold")
    records = []
    for field in absolute + categorical:
        part = frame.loc[frame[field].ne("")].copy()
        observed, predicted, baseline, coverage = [], [], [], []
        for plate, held in part.groupby("plate", sort=True):
            fitting = part.loc[part.plate.ne(plate), ["plate", "position", field]]
            if field in absolute:
                per_plate = fitting.groupby(["position", "plate"])[field].mean()
                location = per_plate.groupby("position").mean()
                default = float(fitting.groupby("plate")[field].mean().mean())
            else:
                per_plate = fitting.drop_duplicates()
                counts = per_plate.groupby(["position", field]).size().rename("count").reset_index()
                location = counts.sort_values(["count", field], ascending=[False, True]).drop_duplicates("position").set_index("position")[field]
                default = fitting[["plate", field]].drop_duplicates()[field].mode().iloc[0]
            guess = held.position.map(location)
            coverage.extend(guess.notna())
            predicted.extend(guess.fillna(default))
            baseline.extend([default] * len(held))
            observed.extend(held[field])
        observed, predicted, baseline = map(np.asarray, (observed, predicted, baseline))
        summary = {"field": field, "wells": len(observed), "position_supported": int(np.sum(coverage))}
        if field in absolute:
            pos = np.mean((observed - predicted) ** 2)
            base = np.mean((observed - baseline) ** 2)
            summary.update(metric="mean_square_error", position=float(pos), no_position=float(base), held_plate_r2=float(1 - pos / base))
        else:
            summary.update(metric="top1_accuracy", position=float(np.mean(observed == predicted)), no_position=float(np.mean(observed == baseline)))
        records.append(summary)
    return pd.DataFrame(records)
