"""Separable PSD drug and cell kernels with explicit availability fallbacks."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.kernels import mask_kernel, normalized_gram, rbf, shuffled, view_geometry
from nextgen_h2.model import Baseline, Fit, Fusion, Representation
from nextgen_h2.run import load_context, read_json
from vc_build.structured_kernel import solve_residuals

from .assets import ROOT, V1_ASSETS, V1_TARGETS

V1 = Baseline("molecular", 4.)
V1_VIEW = Representation("lincs", "modz24", "raw", 0, "cos")
CHEM_AXES = ("tanimoto", "target", "properties", "identity")
OMICS = ("mut", "cnv", "fusion", "sig", "crispr")
SHAPES = ("cos", "cos2", "cos3", "rbf_half", "rbf", "rbf_wide")


@dataclass(frozen=True)
class View:
    name: str
    recipe: str = "mean"
    transform: str = "raw"
    dimension: int = 0
    kernel: str = "cos"
    weight: float = 1.
    mask: float = .15
    control: str = "real"


@dataclass(frozen=True)
class Design:
    lincs: View = View("lincs", "v1")
    penalty: float = 4.
    tanimoto: float = .35
    target: float = .20
    properties: float = .10
    identity: float = .35
    normalization: str = "sum"
    intercept: str = "chem"
    anchor_penalty: float = 1.
    drug_views: tuple[View, ...] = ()
    cell_views: tuple[View, ...] = ()


def design(spec):
    return Design(**(spec | {"lincs": View(**spec["lincs"]),
                            "drug_views": tuple(View(**v) for v in spec.get("drug_views", [])),
                            "cell_views": tuple(View(**v) for v in spec.get("cell_views", []))}))


def kernel_function(x, kind):
    if kind.startswith("cos"):
        gram = x @ x.T
        return gram if kind == "cos" else gram.pow(int(kind[-1]))
    return rbf(x, {"rbf_half": .5, "rbf": 1., "rbf_wide": 2.}[kind])


def union_masks(masks, n, device):
    return torch.stack(masks).any(dim=0) if masks else torch.zeros(n, dtype=torch.bool, device=device)


def complete_fallback(prediction, baseline, cell_seen=None, drug_seen=None):
    if drug_seen is not None:
        prediction = torch.where(drug_seen[None, :], prediction, baseline)
    if cell_seen is not None:
        prediction = torch.where(cell_seen[:, None], prediction, baseline)
    return prediction


class Assets:
    def __init__(self, root):
        self.root = root
        self.views = {}
        self.sealed = {}

    def read(self, family, name):
        key = (family, name)
        if key not in self.views:
            label = {"omics": "assets-omics-v2", "lincs": "assets-lincs", "tahoe": "assets-tahoe"}[family]
            folder = self.root / label
            if family not in self.sealed:
                self.sealed[family] = read_json(folder / "receipt.json")
            filename = name + ".parquet"
            frame = parquet_lock(folder / filename, self.sealed[family]["artifacts"][filename])
            if not frame.index.is_unique:
                raise ValueError(f"Nonunique {family} profile registry")
            self.views[key] = frame
        return self.views[key]


class Engine(Fit):
    def __init__(self, data, assets, fit, query, registry):
        super().__init__(data, fit, query, registry=registry)
        self.assets = assets
        self.reg = registry
        self.raw_query = data.expression(self.query_cells, self.medians)
        self.all_cells = self.fit_cells.append(self.query_cells)
        n = len(self.fit_cells)
        self._rna_gram = (self.cell, self.cross)
        self._features = {}
        self._geom = {}
        self._predictions = {}
        self._c1 = self.baseline(V1).values(self.cross, self.center)
        self._v1 = None
        self.diagnostics = {}
        prop = self.drug.core - .1
        self.chem_blocks = {"properties": prop / .1, "tanimoto": self.drug.tanimoto,
                            "identity": self.drug.identity, "target": self.drug.target}
        self.nfit_cells = n

    def v1(self):
        if self._v1 is None:
            model = self.fusion(V1, [V1_VIEW], Fusion("psd", .5))
            self._v1 = model.values(self.cross, self.center)
        return self._v1

    def current_chemistry(self, params):
        return .10 + sum(getattr(params, name) * gram for name, gram in self.chem_blocks.items())

    def view_data(self, view):
        if view.name == "lincs":
            return self.data.profiles["modz24"] if view.recipe == "v1" else self.assets.read("lincs", view.recipe)
        if view.name == "painting":
            return self.data.profiles["painting"]
        if view.name == "tahoe":
            return self.assets.read("tahoe", view.recipe)
        return self.assets.read("omics", view.name)

    def prepare(self, view, seed, cell=False):
        role = "cell" if cell else "drug"
        names = self.all_cells if cell else self.drugs
        fit_count = self.nfit_cells if cell else self.stop
        conf = (role, view.name, view.recipe, view.transform, view.dimension)
        if conf not in self._features:
            frame = self.view_data(view).reindex(names)
            if view.name == "fusion":
                frame = np.log1p(frame)
            # V1 preprocessing is kept byte-for-byte for its landmark arm.
            raw = frame.to_numpy(dtype=np.float32)
            present = np.isfinite(raw).mean(axis=1) >= .75
            if present[:fit_count].sum() < 10:
                projected = torch.zeros((len(raw), 1), device=self.device)
                mask = torch.zeros(len(raw), dtype=torch.bool, device=self.device)
            else:
                features = view_geometry(raw, fit_count, view.transform, view.dimension, "cos", self.device)
                projected, mask = features.values, features.present
            self._features[conf] = (projected, mask)
        x, mask = self._features[conf]
        key = conf + (view.kernel, view.control, seed if view.control == "shuffle" else None)
        if key not in self._geom:
            if view.control == "mask":
                term = torch.zeros((len(x), len(x)), device=self.device)
            else:
                a = shuffled(x, mask, fit_count, seed) if view.control == "shuffle" else x
                term = mask_kernel(kernel_function(a, view.kernel), mask)
            self._geom[key] = (term, mask)
        return self._geom[key]

    def new_intercept(self, params, fit_gram, cross, masks):
        if params.intercept == "chem":
            return self.intercept
        y = torch.as_tensor(self.anchor.reindex(self.fit_drugs).to_numpy(np.float32), device=self.device)
        if params.intercept == "krr":
            mean = y.mean()
            a = fit_gram + torch.eye(len(y), device=self.device) * params.anchor_penalty
            return mean + cross @ torch.linalg.solve(a, y - mean)
        if params.intercept == "neighbors":
            candidates = cross.clip(min=0)
            values, indices = torch.topk(candidates, int(params.anchor_penalty), dim=1, sorted=True)
            weights = values.pow(4)
            return torch.where(weights.sum(dim=1) > 0,
                               (weights * y[indices]).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-12),
                               self.intercept)
        raise ValueError(f"Unsupported intercept {params.intercept}")

    def drug_geometry(self, params, seed):
        chem = self.current_chemistry(params)
        views = ([params.lincs] if params.lincs.weight > 0 or params.lincs.mask > 0 else [])
        specs = [*views, *params.drug_views]
        masks = []
        for i, spec in enumerate(specs):
            kernel, avail = self.prepare(spec, seed + i)
            chem = chem + spec.weight * kernel + spec.mask * (avail[:, None] & avail[None, :])
            masks.append(avail)
        if params.normalization == "norm":
            chem = normalized_gram(chem)
        return chem, masks

    def cell_geometry(self, params, seed):
        cell, cross = self._rna_gram
        masks = []
        for i, spec in enumerate(params.cell_views):
            kernel, avail = self.prepare(spec, seed + 71 + i, cell=True)
            full = spec.weight * kernel + spec.mask * (avail[:, None] & avail[None, :])
            n = self.nfit_cells
            cell = cell + full[:n, :n]
            cross = cross + full[n:, :n]
            masks.append(avail[n:])
        return cell, cross, masks

    def solve(self, params, seed=18407, *, retain=False):
        conf = (params, seed)
        if conf in self._predictions and not retain:
            return self._predictions[conf]
        drug, masks = self.drug_geometry(params, seed)
        cell, cross, cells_seen = self.cell_geometry(params, seed)
        fit_gram, drug_cross = drug[:self.stop, :self.stop], drug[self.stop:, :self.stop]
        # Large regularization ratios may require more iterations than v1.
        try:
            coef, steps, error = solve_residuals(cell, fit_gram, self.weights, self.response,
                                                 params.penalty, max_steps=1600)
        except RuntimeError as error32:
            if "Kernel solver residual" not in str(error32):
                raise
            coef, steps, error = solve_residuals(cell.double(), fit_gram.double(), self.weights.double(),
                                                 self.response.double(), params.penalty, max_steps=3000)
            coef = coef.float()
        # Exclude identity from a portable chemical cold anchor. Duplicate parents still
        # use their known, fit-only source mean for technical transfer.
        props = self.new_intercept(params, fit_gram, drug_cross, masks)
        warm = self.drug.identity[self.stop:, :self.stop]
        props = torch.where(warm.any(dim=1), self.intercept, props)
        raw = props + (cross @ coef) @ drug_cross.T
        active = union_masks(masks, len(self.drugs), self.device)[self.stop:]
        chosen = raw
        if masks:
            chosen = complete_fallback(raw, self._c1, drug_seen=active)
        extra = None
        if params.drug_views:
            reference = self.solve(replace(params, drug_views=(), cell_views=()), seed)
            extra = union_masks(masks[-len(params.drug_views):], len(self.drugs), self.device)[self.stop:]
            if not params.cell_views:
                chosen = complete_fallback(chosen, reference, drug_seen=extra)
        if cells_seen:
            fallback = self.solve(replace(params, cell_views=()), seed)
            use = union_masks(cells_seen, len(self.query_cells), self.device)
            chosen = complete_fallback(chosen, fallback, cell_seen=use)
        self.diagnostics[conf] = (steps, error)
        if retain:
            return chosen, {"coef": coef.cpu(), "anchor": props.cpu(), "dfit": fit_gram.cpu(),
                            "dcross": drug_cross.cpu(), "cell_cross": cross.cpu(),
                            "apply_c1": bool(masks), "fit_cells": self.fit_cells, "drugs": self.fit_drugs,
                            "active": active.cpu(),
                            "extra_drug": extra.cpu() if extra is not None and not params.cell_views else None,
                            "cell_coverage": union_masks(cells_seen, len(self.query_cells), self.device).cpu()
                                             if cells_seen else None}
        if len(self._predictions) >= 6:
            self._predictions.pop(next(iter(self._predictions)))
        self._predictions[conf] = chosen
        return chosen

    def extract_rows(self, a):
        return a.cpu().numpy()[self.selector]

    def support(self, name, recipe="v1", cell=False):
        model = View(name, recipe)
        _, available = self.prepare(model, 18407, cell=cell)
        pivot = self.nfit_cells if cell else self.stop
        selector = self.selector[0 if cell else 1]
        return available[pivot:].cpu().numpy()[selector]


def context(root, device, external=False):
    data, registry, coverage, query = load_context(V1_ASSETS, V1_TARGETS, device, external=external)
    return data, Assets(root), registry, coverage, query
