"""Tahoe dose-stratified rank panels, controls and isolated outer inputs."""

import argparse
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, ENTITY, GROUP, KEY, LINE, load_data, table
from src.r11_chemical_probe.report import align
from src.r12_multiaxis.data import Source, select_panel
from src.r12_multiaxis.split import eligible_pools


DOSES = (0.05, 0.5, 5.0)
MIN_CELLS = (20, 200)
HIGH_PLATES = {"plate3", "plate6", "plate9", "plate12", "plate14"}
WELL = KEY[0]
MIN_GROUPS = 3
LIVER = ("CVCL_1098", "CVCL_0366")
PROTOCOL_SHA256 = "129af982566410cf860bd494cda4ea526105569a716c171221d332f67fc13dd2"


def validate_protocol(path):
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("Protocol differs from the committed goal-tracks specification")


def slice_name(dose, minimum_cells, fold, role):
    return f"d{dose:g}-n{minimum_cells}-f{fold}-{role}"


def array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def file_hashes(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def verify_one(root, receipt, path):
    name = str(path.relative_to(root))
    if sha256(path) != receipt["files"][name]:
        raise ValueError(f"Staged bytes changed: {path}")


def axes(rows, dose, minimum_cells):
    return rows.dose_uM.eq(dose) & rows.n_cells.ge(minimum_cells)


def eligibility(rows, minimum_cells, dose=None):
    keep = rows.n_cells.ge(minimum_cells)
    if dose is not None:
        keep &= rows.dose_uM.eq(dose)
    return eligible_pools(rows.loc[keep], MIN_GROUPS)


def load_control(folder, role, plate_filter):
    return load_data(folder, role, mask=lambda rows: rows.plate_id.isin(plate_filter).to_numpy())


def control_sources(primary, auxiliary, role, primary_plates, aux_plates):
    a, ac = load_control(primary, role, primary_plates)
    b, bc = load_control(auxiliary, role, aux_plates - primary_plates)
    return pd.concat((a, b), ignore_index=True), vstack((ac, bc), format="csr")


def stage(args):
    validate_protocol(args.protocol)
    cfg = config(args.config)
    partition = args.assets / "partition"
    source = Path(cfg["r11_assets"]) / "source"
    aux = args.assets / "doses/source"
    split = read_json(partition / "split.json")
    if sha256(partition / "split_manifest.tsv") != split["manifest_sha256"]:
        raise ValueError("Partition assignment changed")
    outer = {int(g): int(f) for g, f in split["outer"].items()}
    if len(outer) != 203:
        raise ValueError("R13 parent set is not the canonical 203")
    # Base .5 cohort uses the exact paired R12/R13 source row universe.
    primary_rows = table(partition / "source_rows.tsv")
    expected_keys = pd.MultiIndex.from_frame(primary_rows[KEY])
    a, ac = load_data(source, "treated", mask=lambda x: pd.MultiIndex.from_frame(x[KEY]).isin(expected_keys))
    order = align(primary_rows, a)
    a, ac = a.iloc[order].reset_index(drop=True), ac[order]
    b, bc = load_data(aux, "treated", mask=lambda x: (
        x[GROUP].isin(outer) & x.time_h.eq(24) &
        (x.dose_uM.eq(0.05) | (x.dose_uM.eq(5) & x.plate_id.isin(HIGH_PLATES)))
    ).to_numpy())
    print(f"loaded normalized treated sources: {len(a)} primary, {len(b)} auxiliary", flush=True)
    if b.duplicated(KEY).any() or set(b.loc[b.dose_uM.eq(5), "plate_id"]) != HIGH_PLATES:
        raise ValueError("Invalid auxiliary physical wells or missing registered 5 µM plate")
    rows = pd.concat((a, b), ignore_index=True).assign(matrix_index=lambda x: np.arange(len(x)))
    rows[GROUP] = rows[GROUP].astype(int)
    controls = {}
    for role in "AB":
        control, values = control_sources(source, aux, role, set(a.plate_id), set(b.plate_id))
        idx = control[["plate_id", LINE, WELL]].rename(columns={WELL: f"control_{role}"})
        rows = rows.merge(idx, on=["plate_id", LINE], validate="many_to_one", sort=False)
        controls[role] = control, values
    if rows.duplicated(KEY).any() or rows.control_A.eq(rows.control_B).any():
        raise ValueError("Invalid paired Tahoe condition metadata")
    rows["outer_fold"] = rows[GROUP].map(outer)
    if rows.outer_fold.isna().any():
        raise ValueError("Unmapped parent")
    book = table(args.coordinates / "positions.tsv").set_index("sample", verify_integrity=True)
    if sha256(args.coordinates / "positions.tsv") != read_json(args.coordinates / "receipt.json")["positions_sha256"]:
        raise ValueError("Position index receipt changed")
    if not np.array_equal(book.loc[rows[WELL], "plate"], rows.plate_id):
        raise ValueError("Position index remapped a release plate")
    rows["barcode_position"] = book.loc[rows[WELL], "barcode_position"].to_numpy()

    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "split.json", split)
    for fold in range(3):
        folder = args.out / f"f{fold}"
        folder.mkdir()
        fit = rows.loc[rows.outer_fold.ne(fold)].copy()
        score = rows.loc[rows.outer_fold.eq(fold)].copy()
        p_index = fit[GROUP].map({int(g): int(k) for g, k in split["inner"][str(fold)].items()})
        if p_index.isna().any():
            raise ValueError("Incomplete inner partition")
        fit["inner_fold"] = p_index
        panels = {}
        for dose in DOSES:
            for minimum in MIN_CELLS:
                for inner in (-1, 0, 1):
                    subset = fit if inner == -1 else fit.loc[fit.inner_fold.ne(inner)]
                    use = subset.loc[axes(subset, dose, minimum), "matrix_index"].to_numpy()
                    matrix = ac[use] if dose == 0.5 else bc[use - len(a)]
                    key = slice_name(dose, minimum, fold, "panel") + f"_{inner}"
                    panels[key] = select_panel(matrix, cfg["panel_size"]).tolist()
        write_json(folder / "panels.json", panels)
        union = np.unique(np.concatenate(list(panels.values())))
        np.save(folder / "tokens.npy", union, allow_pickle=False)
        for label, subset in (("fit", fit), ("score", score)):
            ind = subset.matrix_index.to_numpy()
            low, extra = ind < len(a), ind >= len(a)
            values = np.empty((len(subset), len(union)), dtype=np.float32)
            values[low] = ac[ind[low]][:, union].toarray()
            values[extra] = bc[ind[extra] - len(a)][:, union].toarray()
            np.save(folder / f"{label}.npy", values, allow_pickle=False)
            subset.drop(columns="matrix_index").to_csv(folder / f"{label}_rows.tsv", sep="\t", index=False)
        for role, (control, values) in controls.items():
            control.to_csv(folder / f"{role}_rows.tsv", sep="\t", index=False)
            np.save(folder / f"{role}.npy", values[:, union].toarray().astype(np.float32), allow_pickle=False)
        print(f"f{fold}: fit {len(fit)}, sealed {len(score)}, {len(union)} distinct panel tokens", flush=True)
    write_json(args.out / "receipt.json", {
        "protocol_sha256": PROTOCOL_SHA256,
        "config_sha256": sha256(args.config),
        "source_sha256": sha256(source / "receipt.json"),
        "auxiliary_sha256": sha256(aux / "receipt.json"),
        "coordinates_sha256": sha256(args.coordinates / "receipt.json"),
        "files": file_hashes(args.out),
    })


class DoseSource(Source):
    """R12 fit-compatible queryset exposing only same-role controls and outer-fit outcomes."""

    def __init__(self, cfg, assets, staged, dose, minimum_cells, outer_fold, role):
        self.cfg = dict(cfg, dose_uM=dose)
        self.assets, self.staged, self.dose, self.minimum_cells = assets, staged, dose, minimum_cells
        self.outer, self.role = outer_fold, role
        self.r11 = Path(cfg["r11_assets"])
        self.receipt = read_json(staged / "receipt.json")
        if self.receipt["protocol_sha256"] != PROTOCOL_SHA256:
            raise ValueError("Input cache uses another protocol")
        self.split = read_json(staged / "split.json")
        self.folder = staged / f"f{self.outer}"
        for name in ("fit_rows.tsv", "score_rows.tsv", "panels.json", "tokens.npy", f"{role}_rows.tsv"):
            verify_one(staged, self.receipt, self.folder / name)
        self.tokens = pd.Index(array(self.folder / "tokens.npy"))
        self.all_fit_rows = table(self.folder / "fit_rows.tsv")
        self.fit_rows = self.all_fit_rows.loc[axes(self.all_fit_rows, dose, minimum_cells)].reset_index(drop=True)
        self.all_fit_counts = array(self.folder / "fit.npy")
        self.query_rows = eligibility(table(self.folder / "score_rows.tsv"), minimum_cells, dose)
        self.control_rows = table(self.folder / f"{role}_rows.tsv")
        self.control_counts = array(self.folder / f"{role}.npy")
        if set(self.all_fit_rows[GROUP]) & set(self.query_rows[GROUP]):
            raise ValueError("Outer parent leak")
        if set(self.all_fit_rows[WELL]) & set(self.query_rows[WELL]):
            raise ValueError("Outer physical well leak")
        self.panels = read_json(self.folder / "panels.json")
        self.panel = self.inner_panel(-1)
        self.inner_labels = {int(g): k for g, k in self.split["inner"][str(self.outer)].items()}
        with np.load(self.r11 / "features/features.npz", allow_pickle=False) as cache:
            self.chemical = {n: cache[n] for n in ("entity", "E", "C")}
        mean_pool = assets / "chemmean/features.npz"
        if mean_pool.exists():
            with np.load(mean_pool, allow_pickle=False) as cache:
                if not np.array_equal(self.chemical["entity"], cache["entity"]):
                    raise ValueError("Chemical pooling cache uses a different entity order")
                self.chemical["M"] = cache["M"]
        self.dose_cache = AuxiliaryCache(self)

    def inner_panel(self, k):
        label = slice_name(self.dose, self.minimum_cells, self.outer, "panel") + f"_{k}"
        return np.asarray(self.panels[label], dtype=np.int64)

    def columns(self, panel):
        columns = self.tokens.get_indexer(panel)
        if (columns < 0).any():
            raise ValueError("Requested panel outside staged union")
        return columns

    def observed(self, rows, panel):
        return self.all_fit_counts[np.ix_(align(rows, self.all_fit_rows), self.columns(panel))]

    @cached_property
    def fit_counts(self):
        """Panel-indexed sparse adapter for older R12 callers."""
        full = csr_matrix((len(self.fit_rows), int(self.tokens.max()) + 1), dtype=np.float32)
        staged = csr_matrix(self.observed(self.fit_rows, self.tokens.to_numpy()))
        staged.indices = self.tokens.to_numpy()[staged.indices].astype(full.indices.dtype)
        return csr_matrix((staged.data, staged.indices, staged.indptr), shape=full.shape)

    def block(self, fit, query, panel=None, observe_query=False):
        panel = self.panel if panel is None else panel
        cv = self.control_counts[:, self.columns(panel)]
        train = self.assemble(fit, self.control_rows, cv, self.observed(fit, panel))
        prediction = self.assemble(query, self.control_rows, cv, self.observed(query, panel) if observe_query else None)
        self.check_fit_query(train, prediction)
        return train, prediction, panel

    def control_block(self, panel):
        return self.control_rows.rename(columns={WELL: CONTROL}), self.control_counts[:, self.columns(panel)]


class AuxiliaryCache:
    def __init__(self, source):
        self.source = source

    def block(self, fit):
        src = self.source
        rows = src.all_fit_rows
        use = rows[GROUP].isin(set(fit.train.rows[GROUP])) & rows.dose_uM.ne(src.dose)
        aux = eligibility(rows.loc[use], src.minimum_cells)
        if set(aux[GROUP]) & set(fit.query.rows[GROUP]) or set(aux[WELL]) & set(fit.query.rows[WELL]):
            raise ValueError("Held-out rows entered auxiliary fit")
        return src.assemble(aux, src.control_rows, src.control_counts[:, src.columns(fit.panel)],
                            src.observed(aux, fit.panel))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "assets", "coordinates", "protocol", "out"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    stage(parser.parse_args())


if __name__ == "__main__":
    main()
