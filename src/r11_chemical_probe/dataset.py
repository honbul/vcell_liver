"""Assemble aligned rows and expose only the active role's control to a fit."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import read_json, sha256
from .prepare import unique


KEY = ["physical_well_id", "cell_line_id"]
POOL = ["plate_id", "cell_line_id", "dose_uM", "time_h"]
GROUP = "compound_group_id"
ENTITY = "compound_entity_id"
LINE = "cell_line_id"
CONTROL = "control_well_id"


@dataclass
class Block:
    rows: pd.DataFrame
    control: np.ndarray
    chemical: dict
    treated: np.ndarray | None = None

    def take(self, mask):
        return Block(self.rows.loc[mask].reset_index(drop=True), self.control[mask],
                     {name: values[mask] for name, values in self.chemical.items()},
                     None if self.treated is None else self.treated[mask])

    @property
    def delta(self):
        if self.treated is None:
            raise ValueError("Held-out responses are unavailable to this block")
        return self.treated - self.control


def table(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def checked_table(folder, name):
    path = folder / name
    if read_json(folder / "receipt.json")["files"][name] != sha256(path):
        raise ValueError(f"Row receipt hash differs: {path}")
    return table(path)


def normalize(counts, libraries, panel=None):
    if panel is not None:
        counts = counts[:, panel]
    counts = counts.astype(np.float64).tocsr(copy=True)
    if np.any(libraries <= 0):
        raise ValueError("Library normalization requires positive mapped totals")
    if np.any(np.asarray(counts.sum(axis=1)).ravel() > libraries):
        raise ValueError("Mapped library size must include all retained counts")
    counts.data = np.log1p(counts.data * np.repeat(10000 / libraries, np.diff(counts.indptr)))
    return counts


def load_data(folder, role, panel=None, mask=None):
    rows = checked_table(folder, f"{role}_rows.tsv")
    unique(rows, KEY)
    valid = rows.eligible_cell_count.to_numpy(dtype=bool)
    if mask is not None:
        valid &= mask(rows)
    selected = rows[valid]
    file = folder / f"{role}_counts.npz"
    if sha256(file) != read_json(folder / "receipt.json")["files"][file.name]:
        raise ValueError(f"Count receipt hash differs: {file}")
    count = sp.load_npz(file)
    if count.shape[0] != len(rows):
        raise ValueError(f"Rows and matrix disagree for {role}")
    return selected.reset_index(drop=True), normalize(count[valid], selected.mapped_library.to_numpy(), panel)


def weights_for(rows, hierarchy):
    weight = np.ones(len(rows), dtype=np.float64)
    for i, column in enumerate(hierarchy):
        keys = list(hierarchy[:i])
        if not keys:
            continue
        children = rows.groupby(keys, sort=False, dropna=False)[column].transform("nunique")
        weight /= children.to_numpy()
    if hierarchy[-1] != "physical_well_id":
        duplicates = rows.groupby(list(hierarchy), sort=False)[KEY[0]].transform("size")
        weight /= duplicates.to_numpy()
    return weight


def group_weights(rows):
    return weights_for(rows, (GROUP, ENTITY, LINE, KEY[0]))


def select_panel(assets, stage, excluded_fold, size):
    _, treated = load_data(assets / stage, "treated", mask=(
        None if excluded_fold is None else lambda rows: rows.fold.ne(excluded_fold).to_numpy()
    ))
    mean = np.asarray(treated.mean(axis=0)).ravel()
    treated.data **= 2
    variance = np.maximum(0, np.asarray(treated.mean(axis=0)).ravel() - mean ** 2)
    tokens = np.lexsort((np.arange(len(variance)), -variance))
    tokens = tokens[(tokens >= 3) & (variance[tokens] > 0)][:size]
    if len(tokens) != size:
        raise ValueError("Too few variable train-only genes")
    genes = table(assets / "metadata" / "genes.tsv").set_index("token_id").loc[tokens].reset_index()
    genes["train_variance"] = variance[tokens]
    return genes


def assemble(rows, controls, control_vectors, features, treated=None):
    unique(controls, ["plate_id", LINE])
    ids = controls[["plate_id", LINE, KEY[0]]].rename(columns={KEY[0]: CONTROL})
    ids["control_index"] = np.arange(len(ids))
    keyed = rows.copy()
    keyed["treated_index"] = np.arange(len(rows))
    merged = keyed.merge(ids, on=["plate_id", LINE], how="inner", validate="many_to_one", sort=False)
    unique(merged, KEY)
    feature_ids = pd.Index(features["entity"])
    if not feature_ids.is_unique:
        raise ValueError("Duplicate chemical entity")
    chemical_index = feature_ids.get_indexer(merged[ENTITY])
    if (chemical_index < 0).any():
        raise ValueError("Below-comparison missing chemical features")
    cv = control_vectors[merged.pop("control_index").to_numpy()]
    index = merged.pop("treated_index").to_numpy()
    return Block(merged, cv, {name: features[name][chemical_index] for name in ("E", "C")},
                 None if treated is None else treated[index])


def role_blocks(assets, stage, role, panel, final=False):
    path = assets / stage
    controls, control = load_data(path, role, panel)
    rows, treated = load_data(path, "treated", panel)
    with np.load(assets / "features" / "features.npz") as cache:
        features = {key: cache[key] for key in ("E", "C", "entity")}
    source = assemble(rows, controls, control.toarray(), features, treated.toarray())
    if not final:
        return source
    test = checked_table(assets / "final", "treated_rows.tsv")
    test = test[test.eligible_cell_count].reset_index(drop=True)
    return source, assemble(test, controls, control.toarray(), features)


def check_split(fit, query):
    for column in (GROUP, KEY[0]):
        overlap = set(fit.rows[column]) & set(query.rows[column])
        if overlap:
            raise ValueError(f"Leaking {column}: {overlap}")
