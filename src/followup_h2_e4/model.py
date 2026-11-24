"""The fixed E4 kernels and fit-only, drug-constant anchor strategies."""
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import torch

from followup_h2.common import ModzEngine
from followup_h2.e1 import v1_parts
from followup_h2.e3 import frozen_policies
from nextgen_h2_v2.kernel import V1, Design, View, design
from vc_build.structured_kernel import solve_residuals

PENALTIES = (.25, .5, 1., 2., 4., 8., 16., 32.)
WEIGHTS = (.125, .25, .5, 1., 2.)
SHAPES = ("cos", "rbf_wide")
BASES = ("v1_base", "v2_base")
GRID_PARTS = tuple(product(BASES, SHAPES))
CONTROL_SEED = 75217
KERNEL_ONLY = "H2-LOSO-Kernel + V2-Int"
FULL = "H2-LOSO-Full"
POOLED, LOSO = "Sel-Pooled", "Sel-LOSO"
CLAS = ("C1", "H2-v1", "H2-v2")
CALIBRATED = (*CLAS, KERNEL_ONLY, FULL)
CONTROLS = ("C2", "C3")
ANCHORS = (("chem_top5", 5.), *(("chem_target_neighbors", k) for k in (5., 10., 20.)),
           *(("chem_target_krr", k) for k in (1., 5., 20.)),
           *(("lincs_neighbors", k) for k in (5., 10.)),
           *(("lincs_krr", k) for k in (1., 5., 20.)))
BLENDS = (.25, .5, .75)
SHRINKS = (.5, .75, 1.)


@dataclass(frozen=True)
class Candidate:
    base: str
    shape: str
    weight: float
    penalty: float

    @property
    def key(self):
        return f"{self.base}/{self.shape}/w={self.weight:g}/lambda={self.penalty:g}"


GRID = tuple(Candidate(b, s, w, p) for (b, s), w, p in product(GRID_PARTS, WEIGHTS, PENALTIES))


def role_fold(role):
    track, fold, _ = role.split("/", 2)
    return int(fold) if track == "T2" else 3


def v2_policy(role):
    return design(frozen_policies(role_fold(role))["v2"])


def settings(role, config, control="real"):
    old = v2_policy(role) if config.base == "v2_base" else Design()
    return replace(old, penalty=config.penalty, intercept="chem", anchor_penalty=1.,
                   lincs=View("lincs", "landmark_mean", kernel=config.shape,
                              weight=config.weight, mask=.6, control=control))


def anchor_key(family, parameter):
    return f"{family}/{parameter:g}"


class Kernel(ModzEngine):
    def __init__(self, data, fit, query, registry, role):
        super().__init__(data, None, fit, query, registry)
        self.role = role
        baseline = self.baseline(V1)
        ci, personal = v1_parts(baseline, self.cross, self.center)
        self.baseline_anchor = ci[0].numpy().astype(float)
        self.baseline_residual = personal.numpy().astype(float)
        self.target_mean = torch.as_tensor(self.anchor.reindex(self.fit_drugs).to_numpy(np.float32))
        self.warm = self.drug.identity[self.stop:, :self.stop].any(dim=1).numpy()
        self.chemical = self.chemistry(V1.style)
        self._gram = {}

    def geometry(self, params):
        # Lambda and anchor choices do not change the drug Gram.
        geometry = replace(params, penalty=1., intercept="chem", anchor_penalty=1.)
        if geometry not in self._gram:
            g, [avail] = self.drug_geometry(geometry, CONTROL_SEED)
            self._gram[geometry] = g, avail[self.stop:].numpy()
            if len(self._gram) > 2:
                self._gram.pop(next(iter(self._gram)))
        return self._gram[geometry]

    def solve_personal(self, params):
        drug, supported = self.geometry(params)
        stop = self.stop
        g, cross = drug[:stop, :stop], drug[stop:, :stop]
        try:
            coef, steps, error = solve_residuals(self.cell, g, self.weights, self.response,
                                                 params.penalty, max_steps=1600)
        except RuntimeError as fp32_error:
            if "Kernel solver residual" not in str(fp32_error):
                raise
            coef, steps, error = solve_residuals(self.cell.double(), g.double(), self.weights.double(),
                                                 self.response.double(), params.penalty, max_steps=3000)
            coef = coef.float()
        personal = ((self.cross @ coef) @ cross.T).numpy().astype(float)
        return np.where(supported[None], personal, self.baseline_residual), drug, supported, steps, error

    def anchor_strategy(self, strategy, drug_gram, supported):
        family, param = strategy
        if family == "chem_top5":
            return self.baseline_anchor
        if family not in {"chem_target_neighbors", "chem_target_krr", "lincs_neighbors", "lincs_krr"}:
            raise ValueError(f"Unknown anchor strategy {family}")
        is_lincs = family.startswith("lincs")
        gram = drug_gram if is_lincs else self.chemical
        matrix, cross = gram[:self.stop, :self.stop], gram[self.stop:, :self.stop]
        y = self.target_mean
        if family.endswith("krr"):
            estimate = y.mean() + cross @ torch.linalg.solve(
                matrix + param * torch.eye(len(matrix)), y - y.mean())
        else:
            scores, indices = torch.topk(cross.clamp_min(0), min(int(param), len(y)), dim=1, sorted=True)
            weights = scores.pow(4)
            estimate = torch.where(weights.sum(1) > 0, (weights * y[indices]).sum(1) /
                                   weights.sum(1).clamp_min(1e-12), self.intercept)
        return np.where(self.warm | (is_lincs & ~supported), self.baseline_anchor, estimate.numpy())

    def selected_anchor(self, record, drug_gram, supported):
        if record["family"] in ("convex_blend", "shrinkage"):
            lincs = self.anchor_strategy(record["lincs"], drug_gram, supported)
            chem = self.anchor_strategy(record["chem"], drug_gram, supported)
            values = (1. - record["alpha"]) * chem + record["alpha"] * lincs
            mean = float(self.target_mean.mean())
            values = mean + record["gamma"] * (values - mean)
            return np.where(self.warm, self.baseline_anchor, values)
        return self.anchor_strategy((record["family"], record["parameter"]), drug_gram, supported)

    def evaluate_anchors(self, drug_gram, supported):
        return {anchor_key(*spec): self.anchor_strategy(spec, drug_gram, supported) for spec in ANCHORS}

    def parts(self, config, anchor, control="real"):
        personal, gram, supported, steps, error = self.solve_personal(settings(self.role, config, control))
        offset = self.selected_anchor(anchor, gram, supported)
        return personal, offset, gram, supported, {"steps": steps, "absolute_residual": error}

    def extract_with(self, personal, intercept):
        return personal[self.selector] + intercept[self.selector[1]]
