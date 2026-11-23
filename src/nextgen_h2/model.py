"""Fit source-only structured kernels on separable cells and parent compounds."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from vc_build.external import SOURCE, _chemistry, native_sensitivity
from vc_build.response_components import centered_kernels, orthogonal_parts
from vc_build.structured_kernel import (
    fit_cell_kernels, project_shared_features, ranked_expression, solve_residuals, training_grid,
)
from vc_build.v0plus_cv import panel_targets

from .assets import KEY, fold_ids
from .kernels import concat_kernel, guarded, psd_sum, rbf, shuffled, stress_mask, value_kernel, view_geometry

WEIGHTS = (.10, .25, .50)
PENALTIES = (4., 16.)


@dataclass(frozen=True)
class Representation:
    view: str
    recipe: str
    transform: str
    dimension: int
    kernel: str

    @property
    def key(self):
        return "/".join(map(str, (self.view, self.recipe, self.transform, self.dimension, self.kernel)))


@dataclass(frozen=True)
class Baseline:
    style: str
    penalty: float


@dataclass(frozen=True)
class Fusion:
    method: str = "psd"
    weight: float = .25
    match: float = 0.
    stress: float = 0.
    shuffle: bool = False
    protected: bool = True

    @property
    def key(self):
        return f"{self.method}/{self.weight:g}/{self.match:g}/{self.stress:g}/{int(self.shuffle)}"


@dataclass
class Prediction:
    """Sealed coefficient projection on the registered exact-parent query panel."""

    intercept: torch.Tensor
    coefficients: torch.Tensor
    centered: bool
    shared_effect: torch.Tensor | None = None
    residual_coefficients: torch.Tensor | None = None
    present: torch.Tensor | None = None
    baseline: object = None

    def cpu(self):
        return Prediction(
            self.intercept.cpu(), self.coefficients.cpu(), self.centered,
            None if self.shared_effect is None else self.shared_effect.cpu(),
            None if self.residual_coefficients is None else self.residual_coefficients.cpu(),
            None if self.present is None else self.present.cpu(),
            None if self.baseline is None else self.baseline.cpu(),
        )

    def values(self, raw, center, context=None):
        cross = raw - raw.mean(dim=1, keepdim=True) - center + center.mean() if self.centered else raw
        result = self.intercept + cross @ self.coefficients
        if self.shared_effect is not None:
            result = result + context @ self.shared_effect
        if self.residual_coefficients is not None:
            result = result + raw @ self.residual_coefficients
        if self.present is not None:
            result = guarded(self.baseline.values(raw, center), result, self.present)
        return result


@dataclass
class Chemistry:
    core: torch.Tensor
    target: torch.Tensor
    identity: torch.Tensor
    tanimoto: torch.Tensor
    explicit: torch.Tensor


class Inputs:
    def __init__(self, assets, features, profiles, pathway, rna, edges, targets, auxiliary, device):
        self.assets = assets
        self.features = features
        self.profiles = profiles
        self.pathway = pathway
        self.rna = rna
        self.edges = edges
        self.targets = targets
        self.auxiliary = auxiliary
        self.device = device

    def make_chemistry(self, drugs, stop):
        values = torch.as_tensor(self.features.loc[drugs].to_numpy(dtype=np.float32), device=self.device)
        bits, props = values[:, :2048], values[:, 2048:]
        overlap = bits @ bits.T
        mass = bits.sum(dim=1)
        tanimoto = overlap / (mass[:, None] + mass[None, :] - overlap).clamp_min(1)
        props = ((props - props[:stop].mean(dim=0)) / props[:stop].std(dim=0, correction=0).clamp_min(.05)).clamp(-5, 5)
        genes = sorted(set().union(*(self.targets.get(key, ()) for key in drugs[:stop])))
        position = {gene: i for i, gene in enumerate(genes)}
        target = torch.zeros((len(drugs), len(genes)), device=self.device)
        for row, key in enumerate(drugs):
            target[row, [position[g] for g in self.targets.get(key, ()) if g in position]] = 1.
        target = normalize(target, dim=1)
        x = torch.cat([normalize(bits, dim=1), .5 * normalize(props, dim=1), target], dim=1)
        identity = torch.as_tensor(np.equal.outer(drugs.to_numpy(), drugs.to_numpy()), dtype=bits.dtype,
                                   device=self.device)
        return Chemistry(.1 * rbf(.5 * props, 2.) + .1, target @ target.T, identity, tanimoto, x)

    def expression(self, names, medians=None):
        raw = self.rna.reindex(names)
        if raw.isna().all(axis=1).any():
            raise ValueError("Query lineage lacks basal RNA")
        return raw if medians is None else raw.fillna(medians)


class Fit:
    def __init__(self, data, fit, query, *, registry=None):
        self.data, self.fit, self.query = data, fit, query
        self.fit_drugs = pd.Index(sorted(fit[KEY].unique()))
        self.query_drugs = pd.Index(sorted(query[KEY].unique()))
        self.drugs = self.fit_drugs.append(self.query_drugs)
        self.stop = len(self.fit_drugs)
        self.fit_cells = pd.Index(sorted(fit.cell_id.unique()))
        self.query_cells = pd.Index(sorted(query.cell_id.unique()))
        self.device = data.device
        self.drug = data.make_chemistry(self.drugs, self.stop)
        _, self.anchor, self.rows, self.calibration = panel_targets(fit)
        fit_rna = data.expression(self.fit_cells)
        self.medians = fit_rna.median()
        self.query_rna = data.expression(self.query_cells, self.medians)
        self.basis, cells = fit_cell_kernels(fit_rna, data.edges, data.auxiliary, device=self.device)
        self.cell = cells["full"]
        self.cross = self.basis.cross(self.query_rna)["full"]
        self.center = self.cell.mean(dim=0, keepdim=True)
        self.centered_cell = centered_kernels(self.cell, self.cross)[0]
        self.weights, self.response = training_grid(self.rows, self.fit_cells, self.fit_drugs, device=self.device)
        self._interaction_y = None
        self._prepared = {}
        self._base = {}
        self._oof = {}
        self._solutions = {}
        self._context = None
        self.notes = {}
        self.registry = registry
        similarity = self.drug.tanimoto[self.stop:, :self.stop]
        nearest = similarity.argsort(dim=1, descending=True, stable=True)[:, :5]
        taper = torch.zeros_like(similarity)
        taper.scatter_(1, nearest, similarity.gather(1, nearest).pow(4))
        taper /= taper.sum(dim=1, keepdims=True).clamp_min(1e-12)
        same = self.drug.identity[self.stop:, :self.stop]
        taper = torch.where(same.any(dim=1, keepdim=True), same, taper)
        y = torch.as_tensor(self.anchor.reindex(self.fit_drugs).to_numpy(dtype=np.float32), device=self.device)
        self.intercept = torch.where(taper.sum(dim=1) > 0, taper @ y, y.mean())
        self.selector = self.query_cells.get_indexer(query.cell_id), self.query_drugs.get_indexer(query[KEY])

    def chemistry(self, style, targets=True):
        weight = .20 if targets else 0.
        return (self.drug.core + .35 * self.drug.identity + (.35 if targets else .55) * self.drug.tanimoto
                + weight * self.drug.target)

    def fit_predictor(self, gram, base, *, shared=None, notes=None, cache=None):
        fit, cross = gram[:self.stop, :self.stop], gram[self.stop:, :self.stop]
        centered = base.style == "interaction"
        cell, target = self.cell, self.response
        prior = None
        if shared is not None:
            cell_path, drug_path = shared
            prior = (cell_path[:len(self.fit_cells)], drug_path[:self.stop])
        if centered:
            fit, cross = centered_kernels(fit, cross)
            cell = self.centered_cell
            if self._interaction_y is None:
                self._interaction_y = torch.as_tensor(
                    orthogonal_parts(self.response.cpu().numpy(), self.weights.cpu().numpy())[2],
                    dtype=self.weights.dtype, device=self.device)
            target = self._interaction_y
        if cache not in self._solutions:
            solution = solve_residuals(cell, fit, self.weights, target, base.penalty,
                                       shared_prior=prior, max_steps=700)
            if cache is not None:
                self._solutions[cache] = solution
        else:
            solution = self._solutions[cache]
        coeff, steps, error = solution
        if notes is not None:
            self.notes[notes] = {"iterations": steps, "absolute_residual": error}
        effect = None
        if shared is not None:
            x, v = prior
            effect = (((x.T @ coeff) * v.T).sum(dim=1)[:, None] * drug_path[self.stop:].T)
        return Prediction(self.intercept, coeff @ cross.T, centered, effect)

    def baseline(self, base, *, targets=True):
        key = (base, targets)
        if key not in self._base:
            self._base[key] = self.fit_predictor(self.chemistry(base.style, targets), base,
                                                notes=f"C{int(targets)}/{base.style}/{base.penalty:g}")
        return self._base[key]

    def representations(self, config):
        cache_key = (config.view, config.recipe, config.transform, config.dimension)
        if cache_key not in self._prepared:
            key = "painting" if config.view == "painting" else config.recipe
            raw = self.data.profiles[key]
            if config.transform == "pathway":
                raw = self.data.profiles["paths_" + config.recipe]
            self._prepared[cache_key] = view_geometry(raw.reindex(self.drugs).to_numpy(dtype=np.float32),
                                                      self.stop, config.transform, config.dimension, config.kernel, self.device)
        cached = self._prepared[cache_key]
        return cached.values, cached.present, value_kernel(cached.values, config.kernel)

    def pathways(self, recipe, *, shuffle=False, mask=None, seed=18407):
        if self._context is None:
            weights = self.data.pathway
            indices = self.data.rna.columns.get_indexer(weights.index)
            common = indices >= 0
            raw = np.vstack([ranked_expression(self.data.expression(self.fit_cells)), ranked_expression(self.query_rna)])
            scores = raw[:, indices[common]] @ weights.to_numpy(dtype=np.float32)[common]
            fit = scores[:len(self.fit_cells)]
            cell = ((scores - fit.mean(axis=0)) / fit.std(axis=0).clip(.05)).clip(-4, 4)
            self._context = torch.as_tensor(cell, device=self.device)
        raw = self.data.profiles["paths_" + recipe]
        path = raw.reindex(self.drugs).to_numpy(dtype=np.float32)
        present = np.isfinite(path).all(axis=1)
        fitting = path[:self.stop][present[:self.stop]]
        drug = ((path - fitting.mean(axis=0)) / fitting.std(axis=0).clip(.05)).clip(-4, 4)
        drug = torch.as_tensor(np.nan_to_num(drug), device=self.device)
        support = torch.as_tensor(present, device=self.device)
        if mask is not None:
            support = support & mask
        if shuffle:
            drug = shuffled(drug, support, self.stop, seed)
        x = self._context[:len(self.fit_cells)]
        energy = (x.square().mean(dim=0) * drug[:self.stop][support[:self.stop]].square().mean(dim=0)).sum()
        return self._context, drug * support[:, None] / energy.clamp_min(1e-8).sqrt()

    def oof_grid(self, base):
        if base in self._oof:
            return self._oof[base]
        if self.registry is None:
            raise ValueError("Residual gating requires a fitting scaffold roster")
        groups = self.registry.loc[self.fit_drugs, "scaffold_block"]
        parts = []
        roles = fold_ids(groups)
        for fold in range(3):
            hit = roles.index[roles.eq(fold)]
            check = self.fit[self.fit[KEY].isin(hit)]
            fit = self.fit[~self.fit[KEY].isin(hit)]
            query = check.drop(columns=["endpoint_value", "endpoint_direction"])
            sub = Fit(self.data, fit, query)
            prediction = sub.extract(sub.baseline(base))
            mapping = pd.DataFrame(sub.calibration).set_index(SOURCE)
            slope = check.join(mapping[["slope", "intercept"]], on=SOURCE, validate="many_to_one")
            labels = native_sensitivity(check) * slope.slope + slope.intercept
            parts.append(check.assign(residual=labels - prediction))
        residuals = pd.concat(parts, ignore_index=True)
        if set(residuals.row_id) != set(self.fit.row_id) or not residuals.row_id.is_unique:
            raise ValueError("Scaffold-cross-fitted residual roster changed")
        self._oof[base] = training_grid(residuals, self.fit_cells, self.fit_drugs, device=self.device)
        return self._oof[base]

    def fusion(self, base, configs, spec, *, seed=18407):
        raw, masks, grams = zip(*(self.representations(config) for config in configs))
        in_use = tuple(stress_mask(mask, self.stop, spec.stress, seed + i)
                       for i, mask in enumerate(masks))
        if spec.shuffle:
            raw = tuple(shuffled(x, m, self.stop, seed + i) for i, (x, m) in enumerate(zip(raw, in_use)))
            grams = tuple(value_kernel(x, config.kernel) for x, config in zip(raw, configs))
        present = torch.stack(in_use).any(dim=0)
        chem = self.chemistry(base.style)
        masks_and_grams = [(torch.ones_like(chem), m) for m in in_use]
        for gram, available in zip(grams, in_use):
            masks_and_grams.append((gram, available))
        shape = [0.15] * len(configs) + [spec.weight] * len(configs)
        shared = None
        if spec.match:
            [lincs] = [cfg for cfg in configs if cfg.view == "lincs"]
            mask = in_use[configs.index(lincs)]
            cell, drug = self.pathways(lincs.recipe, shuffle=spec.shuffle, mask=mask, seed=seed)
            shared = (cell, drug * np.sqrt(spec.match))
        baseline = self.baseline(base)
        if spec.method == "mask":
            model = self.fit_predictor(psd_sum(chem, masks_and_grams[:len(configs)], shape[:len(configs)]), base)
            if spec.protected:
                model.present, model.baseline = present[self.stop:], baseline
            return model
        if spec.method in {"concat_zero", "concat_mean"}:
            if len(configs) != 1:
                raise ValueError("Simple concat is registered on one view")
            extra = concat_kernel(self.drug.explicit, raw[0], in_use[0], self.stop,
                                  mean_impute=spec.method == "concat_mean")
            full = psd_sum(chem, masks_and_grams[:1], shape[:1]) + spec.weight * extra
        elif spec.method in {"psd", "norm", "gated"}:
            full = psd_sum(chem, masks_and_grams, shape, diagonal_normalize=spec.method == "norm")
        else:
            raise ValueError(f"Unregistered fusion {spec.method}")
        cache = (base, tuple(configs), spec.method, spec.weight, spec.match, spec.shuffle, seed)
        if spec.method == "gated":
            weights, y = self.oof_grid(base)
            weights = weights * present[:self.stop]
            extra = psd_sum(torch.zeros_like(chem), list(zip(grams, in_use)), [spec.weight] * len(configs))
            prior = None if shared is None else (shared[0][:len(self.fit_cells)], shared[1][:self.stop])
            if cache not in self._solutions:
                self._solutions[cache] = solve_residuals(self.cell, extra[:self.stop, :self.stop], weights, y,
                                                        base.penalty, shared_prior=prior, max_steps=700)
            coeff, steps, error = self._solutions[cache]
            effect = None
            if shared is not None:
                x, v = prior
                effect = (((x.T @ coeff) * v.T).sum(dim=1)[:, None] * shared[1][self.stop:].T)
            self.notes[spec.key] = {"iterations": steps, "absolute_residual": error}
            return Prediction(baseline.intercept, baseline.coefficients, baseline.centered, effect,
                              residual_coefficients=coeff @ extra[self.stop:, :self.stop].T,
                              present=present[self.stop:], baseline=baseline)
        model = self.fit_predictor(full, base, shared=shared, notes=spec.key,
                                   cache=cache if not spec.protected else None)
        if spec.protected:
            model.present, model.baseline = present[self.stop:], baseline
        return model

    def extract(self, model):
        context = None if self._context is None else self._context[len(self.fit_cells):]
        return model.values(self.cross, self.center, context).cpu().numpy()[self.selector]


def molecular_features(registry):
    rows = registry.reset_index().rename(columns={"smiles": "isomeric_smiles"})
    return _chemistry(rows, fp_size=2048)
