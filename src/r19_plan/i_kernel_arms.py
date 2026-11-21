"""Construct paired I-track kernel arms without accessing RNA responses."""

from dataclasses import dataclass
from zlib import crc32

import numpy as np
import pandas as pd

from src.r19_plan.i_kernel import FitProfile, masked_gram

VIEW_NAMES = ("I1", "I2", "I4")
ANNOTATION = "I1_annotation"


def curated_annotation_rows(path):
    """Use curated mechanisms present in the audited ChEMBL assay export."""
    fields = ["parent", "target_chembl_id", "curated_target_action_annotations"]
    rows = pd.read_csv(path, sep="\t", usecols=fields, dtype=str, keep_default_na=False)
    rows = rows.loc[rows[fields].ne("").all(axis=1)].copy()
    rows["action"] = rows[fields[-1]].str.split("|")
    rows = rows.explode("action")
    rows["action"] = rows.action.str.strip()
    rows = rows.loc[rows.action.ne(""), ["parent", "target_chembl_id", "action"]].drop_duplicates()
    rows["feature_key"] = list(rows[["target_chembl_id", "action"]].itertuples(index=False, name=None))
    return rows[["parent", "feature_key"]]


def annotation_matrix(records, parents, fitting):
    """Binary target×action indicators in a vocabulary present in two fit parents."""
    parents = pd.Index(parents)
    fitting = row_mask(fitting, len(parents))
    if not parents.is_unique:
        raise ValueError("The canonical parent axis is not unique")
    training = records.loc[records.parent.isin(parents[fitting])]
    support = training.groupby("feature_key", sort=True).parent.nunique()
    vocabulary = support.loc[support.ge(2)].index
    used = records.loc[records.feature_key.isin(vocabulary) & records.parent.isin(parents)]
    parent_index = parents.get_indexer(used.parent)
    feature_index = vocabulary.get_indexer(used.feature_key)
    result = np.full((len(parents), len(vocabulary)), np.nan, dtype=np.float64)
    result[parent_index] = 0.0
    result[parent_index, feature_index] = 1.0
    return result


def row_mask(values, size):
    mask = np.asarray(values)
    if mask.dtype.kind != "b" or mask.shape != (size,):
        raise ValueError("Expected a Boolean mask on the canonical parent axis")
    return mask


def fit_optional_profiles(profiles, fitting, rank=64):
    """Map each view with the registered fit-only standardization and PCA."""
    fitting = row_mask(fitting, len(fitting))
    if not fitting.any():
        raise ValueError("No fitting parents for optional view transforms")
    result = {}
    for name, data in profiles.items():
        data = np.asarray(data)
        if data.ndim != 2 or len(data) != len(fitting):
            raise ValueError(f"Profile does not match the canonical parent axis: {name}")
        if name == ANNOTATION:
            indicators = np.nan_to_num(data, nan=0.0)
            if not np.isin(indicators, (0.0, 1.0)).all():
                raise ValueError("The curated annotation view must contain binary indicators")
            norm = np.linalg.norm(indicators, axis=1)
            present = norm > 0
            result[name] = (indicators / np.where(present, norm, 1.0)[:, None], present)
            continue
        transformer = FitProfile.learn(data[fitting], rank=rank)
        result[name] = transformer.transform(data)
    return result


@dataclass(frozen=True)
class KernelArm:
    """Gram on active parents ordered as fitting indices then query indices."""

    name: str
    views: tuple[str, ...]
    gram: np.ndarray
    train_indices: np.ndarray
    query_indices: np.ndarray
    query_view_count: np.ndarray

    @property
    def train_gram(self):
        boundary = len(self.train_indices)
        return self.gram[:boundary, :boundary]

    @property
    def query_gram(self):
        boundary = len(self.train_indices)
        return self.gram[boundary:, :boundary]

    def apply_structure_fallback(self, forecast, structure_forecast):
        """Route no-view forecasts to the unchanged I0 result on the same query axis."""
        forecast = np.asarray(forecast)
        structure_forecast = np.asarray(structure_forecast)
        if (forecast.ndim < 1 or forecast.shape != structure_forecast.shape
                or len(forecast) != len(self.query_indices)):
            raise ValueError("Forecasts do not match the active query axis")
        if not self.views:
            return forecast
        result = forecast.copy()
        absent = self.query_view_count == 0
        result[absent] = structure_forecast[absent]
        return result


def shuffle_views(mapped, n_fit, seed):
    """Break feature–parent pairing independently in each active split.

    Shuffle only parents measured in that view and retain the availability mask.
    The input contains active fit and query rows only, so a held-out source
    feature is never copied into a training row.
    """
    shuffled = {}
    for name, (vectors, present) in mapped.items():
        values = vectors.copy()
        measured = np.flatnonzero(present)
        for role, indices in enumerate((measured[measured < n_fit], measured[measured >= n_fit])):
            rng = np.random.default_rng(np.random.SeedSequence([seed, crc32(name.encode()), role]))
            values[indices] = vectors[rng.permutation(indices)]
        shuffled[name] = values, present
    return shuffled


def mask_gram(views):
    # Each view contributes one bit of availability, represented as two states.
    bits = np.column_stack([value for _, present in views for value in (present, ~present)])
    bits = bits.astype(np.float64)
    return (bits @ bits.T) / len(views)


def zero_fill_mask_gram(structure, views):
    """Fixed block weights with zeros for missing values and explicit availability bits."""
    result = structure.copy()
    for values, present in views:
        numeric = np.where(present[:, None], values, 0.0)
        flag = present.astype(np.float64)
        result += (numeric @ numeric.T + np.outer(flag, flag)) / 2
    return result / (len(views) + 1)


def iter_kernel_arms(structure, mapped, fitting, query, shuffle_seed=1501):
    """Yield the registered paired arms without retaining every dense Gram.

    ``mapped`` is the output of ``fit_optional_profiles`` on the same parent
    axis. A valid inner split can exclude rows reserved for outer evaluation.
    The caller applies I0 forecast routing to candidate arms and keeps pure
    profile/mask controls on their diagnostic information contract.
    """
    structure = np.asarray(structure, dtype=np.float64)
    if structure.ndim != 2 or structure.shape[0] != structure.shape[1]:
        raise ValueError("Structure Gram must be square on the canonical parent axis")
    size = len(structure)
    fitting, query = row_mask(fitting, size), row_mask(query, size)
    if not fitting.any() or not query.any() or (fitting & query).any():
        raise ValueError("Fit and active query require disjoint, nonempty parent sets")
    train_indices, query_indices = np.flatnonzero(fitting), np.flatnonzero(query)
    active = np.r_[train_indices, query_indices]
    structure = masked_gram(structure[np.ix_(active, active)], [])

    blocks = {}
    for name, (values, present) in mapped.items():
        values = np.asarray(values, dtype=np.float64)
        present = row_mask(present, size)
        if name not in (*VIEW_NAMES, ANNOTATION) or values.ndim != 2 or len(values) != size:
            raise ValueError(f"Invalid optional view: {name}")
        if not np.isfinite(values[active]).all():
            raise ValueError(f"Nonfinite optional view coordinates: {name}")
        blocks[name] = values[active], present[active]
    if ANNOTATION in blocks and "I1" not in blocks:
        raise ValueError("I1 annotation-only requires its paired I1 assay arm")

    shuffled = shuffle_views(blocks, len(train_indices), shuffle_seed)

    def arm(name, views, gram):
        counts = np.zeros(len(query_indices), dtype=np.int64)
        for view in views:
            counts += blocks[view][1][len(train_indices):]
        return KernelArm(name, views, gram, train_indices, query_indices, counts)

    yield arm("I0", (), structure)
    views = tuple(name for name in VIEW_NAMES if name in blocks)
    for name in views:
        block = blocks[name]
        yield arm(name, (name,), masked_gram(structure, [block]))
        yield arm(f"{name}_mask_only", (name,), mask_gram([block]))
        yield arm(f"{name}_parent_shuffle", (name,), masked_gram(structure, [shuffled[name]]))

    if ANNOTATION in blocks:
        yield arm("I1_annotation_only", (ANNOTATION,), masked_gram(structure, [blocks[ANNOTATION]]))
    if "I2" in blocks:
        values, present = blocks["I2"]
        morphology = np.where(present[:, None], values, 0.0)
        yield arm("I2_morphology_only", ("I2",), morphology @ morphology.T)

    groups = []
    if len(views) >= 2:
        groups.append(("I3", views[:2]))  # I1 + I2 when both are eligible
    if len(views) == 3:
        groups.append(("I3_all", views))
    for name, selected in groups:
        components = [blocks[view] for view in selected]
        yield arm(name, selected, masked_gram(structure, components))
        yield arm(f"{name}_zero_fill_mask", selected, zero_fill_mask_gram(structure, components))
        yield arm(f"{name}_mask_only", selected, mask_gram(components))
        yield arm(f"{name}_parent_shuffle", selected, masked_gram(structure, [shuffled[v] for v in selected]))
        for view in selected:
            kept = tuple(v for v in selected if v != view)
            yield arm(f"{name}_without_{view}", kept, masked_gram(structure, [blocks[v] for v in kept]))
