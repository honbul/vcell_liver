"""Fit-partition data access and pair-aligned drug and context representations."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from src.r11_chemical_probe.common import read_json, sha256
from src.r11_chemical_probe.dataset import (
    CONTROL, ENTITY, GROUP, KEY, LINE, assemble, check_split, group_weights, load_data, table,
)
from src.r11_chemical_probe.models import FullInput, Interaction, PoolMean, Scaling
from src.r11_chemical_probe.report import align

from .split import eligible_pools
from .stage import load_staged


def select_panel(values, size):
    mean = np.asarray(values.mean(axis=0, dtype=np.float64)).ravel()
    variance = np.maximum(0, np.asarray(values.power(2).mean(axis=0, dtype=np.float64)).ravel() - mean**2)
    tokens = np.lexsort((np.arange(len(variance)), -variance))
    panel = tokens[(tokens >= 3) & (variance[tokens] > 0)][:size]
    if len(panel) != size:
        raise ValueError("Too few variable fit-only genes")
    return panel


class Source:
    def __init__(self, cfg, assets, outer, role, split_dir="partition", stage_dir="staged"):
        self.cfg = cfg
        self.assets = assets
        self.role = role
        self.r11 = Path(cfg["r11_assets"])
        self.split = read_json(assets / split_dir / "split.json")
        self.outer = outer
        cache = assets / stage_dir
        staged = read_json(cache / "receipt.json")
        self.fit_rows, self.fit_counts = load_staged(cache / f"outer{outer}", "fit", staged["genes"])
        self.query_rows = table(cache / f"outer{outer}" / "score-rows.tsv")
        self.query_rows = eligible_pools(self.query_rows, cfg["minimum_groups"])
        if set(self.fit_rows[GROUP]) & set(self.query_rows[GROUP]):
            raise ValueError("Outer groups overlap")
        control_rows, control_counts = load_data(self.r11 / "source", role)
        self.control_rows, self.control_counts = control_rows, control_counts
        with np.load(self.r11 / "features/features.npz") as cache:
            self.chemical = {n: cache[n] for n in ("entity", "E", "C")}
        mean_pool = assets / "chemmean" / "features.npz"
        if mean_pool.exists():
            with np.load(mean_pool) as cache:
                if not np.array_equal(self.chemical["entity"], cache["entity"]):
                    raise ValueError("Chemical pooling cache uses a different entity order")
                self.chemical["M"] = cache["M"]
        self.inner_labels = {int(g): k for g, k in self.split["inner"][str(outer)].items()}

    def inner(self, fold):
        labels = self.fit_rows[GROUP].map(self.inner_labels)
        fit = self.fit_rows[labels.ne(fold)].reset_index(drop=True)
        validation = self.fit_rows[labels.eq(fold)].reset_index(drop=True)
        return fit, eligible_pools(validation, self.cfg["minimum_groups"])

    def block(self, fit, query, panel=None, observe_query=False):
        pos = align(fit, self.fit_rows)
        observed = self.fit_counts[pos]
        panel = select_panel(observed, self.cfg["panel_size"]) if panel is None else panel
        controls = self.control_counts[:, panel].toarray().astype(np.float32)
        y_fit = observed[:, panel].toarray()
        y_query = None
        if observe_query:
            y_query = self.fit_counts[align(query, self.fit_rows)][:, panel].toarray()
        t = self.assemble(fit, self.control_rows, controls, y_fit)
        v = self.assemble(query, self.control_rows, controls, y_query)
        self.check_fit_query(t, v)
        return t, v, panel

    def check_fit_query(self, fit, query):
        check_split(fit, query)

    def assemble(self, rows, control_rows, controls, observed):
        own = f"control_{self.role}"
        if own in rows:
            own_id = rows[own].to_numpy()
        else:
            own_id = None
        rows = rows.drop(columns=["control_A", "control_B"], errors="ignore").reset_index(drop=True)
        result = assemble(rows, control_rows, controls, self.chemical, observed)
        if len(result.rows) != len(rows) or not result.rows[KEY].equals(rows[KEY]):
            raise ValueError("Own-control or chemical join changed the common row universe")
        if own_id is not None and not np.array_equal(result.rows[CONTROL], own_id):
            raise ValueError("Wrong role's control reader")
        if "M" in self.chemical:
            pos = pd.Index(self.chemical["entity"]).get_indexer(rows[ENTITY])
            result.chemical["M"] = self.chemical["M"][pos]
        return result

    @cached_property
    def optional_cells(self):
        folder = self.assets / "cells-v2"
        if not (folder / "receipt.json").exists():
            return None
        row_file = folder / f"{self.role}_rows.tsv"
        mat_file = folder / f"{self.role}_cells.npz"
        receipt = read_json(folder / "receipt.json")
        for file in (row_file, mat_file):
            if sha256(file) != receipt["files"][file.name]:
                raise ValueError("DMSO sample changed")
        rows = table(row_file)
        if not rows.role.eq(self.role).all():
            raise ValueError("Control cells crossed roles")
        return CellContexts(rows, sparse.load_npz(mat_file), self.cfg["optional"])

    def control_block(self, panel):
        rows = self.control_rows.rename(columns={KEY[0]: CONTROL})
        vectors = self.control_counts[:, panel].toarray().astype(np.float32)
        return rows, vectors


class CellContexts:
    def __init__(self, rows, counts, settings):
        self.counts = counts
        self.libraries = np.asarray(counts.sum(axis=1)).ravel()
        self.units = rows.rename(columns={KEY[0]: CONTROL}).groupby([CONTROL, LINE], sort=False).indices
        self.view = rows.view.to_numpy()
        self.settings = settings

    def summary(self, panel, needed_units):
        values = {}
        counts = self.counts[:, panel]
        for key in sorted(needed_units):
            pos = self.units[key]
            cell = counts[pos].toarray().astype(np.float32)
            library = self.libraries[pos]
            single = np.log1p(cell * (1e4 / library[:, None]))
            means = [np.log1p(cell[mask].sum(axis=0) * (1e4 / library[mask].sum()))
                     for mask in (np.ones(len(pos), dtype=bool), self.view[pos] == 0, self.view[pos] == 1)]
            view_masks = (self.view[pos] == 0, self.view[pos] == 1)
            moments = np.concatenate((means[0], single.var(axis=0, ddof=1), (cell > 0).mean(axis=0)))
            moment_views = tuple(np.concatenate((means[i + 1], single[mask].var(axis=0, ddof=1),
                                                  (cell[mask] > 0).mean(axis=0)))
                                 for i, mask in enumerate(view_masks))
            values[key] = (means[0], means[1], means[2], moments, *moment_views)
        return values

    def gene_weights(self, panel, fit_units):
        values = self.counts[:, panel]
        rng = np.random.default_rng(self.settings["cell_seed"])
        var = []
        for key in fit_units:
            pos = self.units[key]
            count = values[pos].toarray()
            library = self.libraries[pos]
            draws = self.settings["gene_weight_draws"]
            weights = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=draws)
            sampled = np.log1p((weights @ count) * (1e4 / (weights @ library))[:, None])
            var.append(sampled.var(axis=0, ddof=1))
        v = np.mean(var, axis=0)
        if not np.any(v > 0):
            raise ValueError("DMSO resampling did not identify gene variation")
        weight = 1 / np.maximum(v, 0.1 * np.median(v[v > 0]))
        weight = np.clip(weight / weight.mean(), 0.25, 4)
        return (weight / weight.mean()).astype(np.float32)


def units(rows):
    return list(rows[[CONTROL, LINE]].itertuples(index=False, name=None))


@dataclass
class Design:
    train: tuple
    query: tuple
    views: tuple | None = None
    unit_views: tuple | None = None
    response_head: tuple | None = None


class Fit:
    def __init__(self, source, fit, query, observe_query=False, panel=None):
        self.source = source
        self.train, self.query, self.panel = source.block(fit, query, panel, observe_query)
        self.cfg = source.cfg["training"]
        self.means = PoolMean(self.train.rows, self.train.delta)
        self.offset, self.offset_level, _ = self.means.predict(self.query.rows, self.cfg["offset_shrinkage"])
        baseline = self.means.predict(self.train.rows, self.cfg["offset_shrinkage"], leave_group_out=True)[0]
        self.residual = (self.train.delta - baseline).astype(np.float32)
        self.weight = group_weights(self.train.rows)
        self.scale = max(1e-6, float(np.sqrt(np.dot(self.weight, np.square(self.residual).mean(axis=1))
                                           / self.weight.sum())))
        self.train_units, self.query_units = units(self.train.rows), units(self.query.rows)
        control_rows, control_values = self.source.control_block(self.panel)
        self.ctrl = dict(zip(units(control_rows), control_values))
        self.ctrl_meta = dict(zip(units(control_rows), control_rows.plate_id))
        self.fit_units = sorted(set(self.train_units))
        self.fit_control = np.stack([self.ctrl[key] for key in self.fit_units])
        self.by_line = {}
        for key in self.fit_units:
            self.by_line.setdefault(key[1], []).append(key)
        self.line_mean = {line: np.stack([self.ctrl[key] for key in keys]).mean(axis=0)
                          for line, keys in self.by_line.items()}
        self.nuisance_rank = 0
        self.designs = {}
        self._projection = {}

    @cached_property
    def full(self):
        return FullInput(self.train)

    @cached_property
    def drug_blocks(self):
        unique = ~self.train.rows.duplicated(ENTITY)
        pca, scale = Interaction._fit(self.train.chemical["E"][unique], 32)
        return [scale.transform(pca.transform(block.chemical["E"])).astype(np.float32)
                for block in (self.train, self.query)]

    @cached_property
    def nuisance(self):
        deviation = np.stack([self.ctrl[key] - self.line_mean[key[1]] for key in self.fit_units])
        svd = PCA(n_components=min(2, *deviation.shape), svd_solver="randomized", random_state=42).fit(deviation)
        self.nuisance_rank = int(np.sum(svd.singular_values_ > max(1e-7, svd.singular_values_[0] * 1e-7)))
        return svd.components_[:self.nuisance_rank]

    @cached_property
    def cell_stats(self):
        c = self.source.optional_cells
        return None if c is None else c.summary(self.panel, set(self.train_units) | set(self.query_units))

    @cached_property
    def noise_weights(self):
        return self.source.optional_cells.gene_weights(self.panel, self.fit_units)

    def context_bank(self, kind):
        if kind in {"raw", "pca"}:
            return self.ctrl
        if kind == "stable":
            return {key: self.line_mean.get(key[1], cv) for key, cv in self.ctrl.items()}
        if kind.startswith("nuisance"):
            rank = int(kind[-1])
            if rank > len(self.nuisance):
                raise ValueError(f"Rank {rank} exceeds fit-only DMSO rank")
            basis = self.nuisance[:rank]
            return {key: cv - ((cv - self.line_mean.get(key[1], cv)) @ basis.T) @ basis
                    for key, cv in self.ctrl.items()}
        if kind in {"sample_mean", "moments"}:
            if kind == "sample_mean":
                return {key: stats[0] for key, stats in self.cell_stats.items()}
            return {key: s[3] for key, s in self.cell_stats.items()}
        if kind in {"C7", "C7_mean", "C8", "C8_mean", "C9", "C9_id"}:
            return self.auxiliary_context(kind)
        raise ValueError(kind)

    def auxiliary_context(self, kind):
        if kind in {"C9", "C9_id"}:
            from .assets import depmap_features
            return depmap_features(self, kind)
        root = self.source.assets / {"C7": "scgpt", "C8": "uce"}[kind[:2]]
        feature = "raw" if kind.endswith("mean") else "embedding"
        with np.load(root / f"{self.source.role}.npz") as data:
            keys = list(zip(data["well"], data["line"]))
            values = data[feature]
            return dict(zip(keys, values))

    def project(self, kind):
        if kind not in self._projection:
            bank = self.context_bank(kind)
            values = np.stack([bank[key] for key in self.fit_units])
            if kind in {"raw", "stable", "nuisance0", "nuisance1", "nuisance2"}:
                # Raw-coordinate comparisons share the same dimension.
                scaler = Scaling(values)
                self._projection[kind] = (bank, scaler.transform)
            elif kind in {"C9", "C9_id"}:
                self._projection[kind] = (bank, np.asarray)
            else:
                pca, scaler = Interaction._fit(values, 16)
                self._projection[kind] = (bank, lambda arr: scaler.transform(pca.transform(arr)))
        return self._projection[kind]

    def view_context(self, kind, view, transform, bank):
        if kind in {"C7", "C8"}:
            root = self.source.assets / {"C7": "scgpt", "C8": "uce"}[kind]
            with np.load(root / f"{self.source.role}.npz") as data:
                mapping = dict(zip(zip(data["well"], data["line"]), data["views"][view - 1]))
            return transform(np.stack([mapping[key] for key in self.fit_units]))
        if kind == "moments":
            return transform(np.stack([self.cell_stats[key][3 + view] for key in self.fit_units]))
        values = np.stack([self.cell_stats[key][view] for key in self.fit_units])
        if kind == "stable":
            means = {line: np.stack([self.cell_stats[key][view] for key in keys]).mean(axis=0)
                     for line, keys in self.by_line.items()}
            values = np.stack([means[key[1]] for key in self.fit_units])
        if kind.startswith("nuisance"):
            rank = int(kind[-1])
            basis = self.nuisance[:rank]
            means = np.stack([self.line_mean[key[1]] for key in self.fit_units])
            values -= ((values - means) @ basis.T) @ basis
        if kind == "C9":
            _, rna = self.project("pca")
            rna_values = rna(values)
            annotation = np.stack([bank[key][rna_values.shape[1]:] for key in self.fit_units])
            return np.column_stack((rna_values, annotation))
        return transform(values)

    def inputs(self, kind, augment=False, rank=None, replacement=None):
        key = (kind, augment, rank)
        if replacement is None and key in self.designs:
            return self.designs[key]
        if kind == "full":
            d = Design((self.full.transform(self.train).astype(np.float32),),
                       (self.full.transform(self.query).astype(np.float32),))
        elif kind == "none":
            d = Design((self.drug_blocks[0],), (self.drug_blocks[1],))
        else:
            bank, transform = self.project(kind)
            t = transform(np.stack([bank[key] for key in self.train_units])).astype(np.float32)
            q = transform(np.stack([bank[key] for key in (replacement or self.query_units)])).astype(np.float32)
            d = Design((self.drug_blocks[0], t), (self.drug_blocks[1], q))
            if augment and self.cell_stats is not None:
                views = [self.view_context(kind, v, transform, bank).astype(np.float32) for v in (1, 2)]
                index = pd.Index(self.fit_units).get_indexer(self.train_units)
                d.views = tuple(v[index] for v in views)
                d.unit_views = tuple(views)
        if rank is not None:
            response = self.residual / self.scale
            weight = self.weight / self.weight.sum()
            center = np.average(response, axis=0, weights=weight)
            weighted = (response - center) * np.sqrt(weight[:, None])
            from sklearn.utils.extmath import randomized_svd

            _, _, vectors = randomized_svd(weighted, rank, random_state=42)
            d.response_head = (vectors.astype(np.float32), center.astype(np.float32))
        if replacement is None:
            self.designs[key] = d
        return d
