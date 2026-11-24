"""Fit-only screen transforms and separable weighted single- and multi-task KRR."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from followup_h2_e4.model import Kernel
from nextgen_h2.assets import KEY
from vc_build.external import SOURCE
from vc_build.structured_kernel import training_grid

TAUS = (0., .25, .5, 1., 2., 4.)
BASELINE, DEBIASED, TASK, BALANCED = ("Base", "P3A-ScreenDebiased", "P3B-HierarchicalMTK", "P3C-EqualScreenWeight")
STRUCTURES = (DEBIASED, TASK, BALANCED)


def debias(rows):
    """Center the observed screen-cell/drug graph, then scale each drug within screen."""
    result = np.empty(len(rows))
    scales, book = [], []
    for (screen, endpoint), positions in rows.groupby(SOURCE, sort=True).indices.items():
        block = rows.iloc[positions]
        y = block.model_value.to_numpy(dtype=float)
        c, cn = pd.factorize(block.cell_id, sort=True)
        d, dn = pd.factorize(block[KEY], sort=True)
        nc, nd = np.bincount(c), np.bincount(d)
        a, b = np.zeros(len(cn)), np.bincount(d, weights=y) / nd
        for step in range(2500):
            a = np.bincount(c, weights=y-b[d]) / nc
            update = np.bincount(d, weights=y-a[c]) / nd
            if np.max(np.abs(update-b)) < 1e-11:
                b = update
                break
            b = update
        r = y-a[c]-b[d]
        cell_error = float(np.max(np.abs(np.bincount(c, weights=r)/nc)))
        drug_error = float(np.max(np.abs(np.bincount(d, weights=r)/nd)))
        if max(cell_error, drug_error) >= 1e-8:
            raise ValueError(f"Screen centering did not converge for {screen}/{endpoint}")
        sd = np.sqrt(np.bincount(d, weights=r*r) / nd)
        good = sd > 1e-10
        assert np.any(good)
        result[positions] = np.divide(r, sd[d], out=np.zeros_like(r), where=good[d])
        scales.extend(sd[good])
        check = np.bincount(d, weights=result[positions]**2) / nd
        book.append({"screen": screen, "endpoint": endpoint, "rows": len(y), "cells": len(cn),
                     "drugs": len(dn), "constant_drugs": int((~good).sum()), "steps": step+1,
                     "max_cell_mean_error": cell_error, "max_drug_mean_error": drug_error,
                     "max_unit_variance_error": float(np.max(np.abs(check[good]-1.)))})
    return result, float(np.median(scales)), book


def row_weights(rows):
    by_assay = rows.groupby([*SOURCE, KEY]).residual
    sizes = by_assay.transform("size")
    largest = rows.assign(sizes=sizes).groupby(KEY).sizes.transform("max")
    assays = rows[[*SOURCE, KEY]].drop_duplicates().groupby(KEY).size()
    return (largest / (sizes * rows[KEY].map(assays))).to_numpy(dtype=float)


def per_screen(rows, cells, drugs, weights):
    ci, di = cells.get_indexer(rows.cell_id), drugs.get_indexer(rows[KEY])
    codes, screens = pd.factorize(rows.source_id, sort=True)
    loc = codes, ci, di
    mass = np.zeros((len(screens), len(cells), len(drugs)), dtype=np.float32)
    sum_y = np.zeros_like(mass)
    np.add.at(mass, loc, weights)
    np.add.at(sum_y, loc, weights * rows.residual.to_numpy())
    return tuple(screens), mass, sum_y


@dataclass
class System:
    cell: torch.Tensor
    drug: torch.Tensor
    mass: torch.Tensor
    values: torch.Tensor
    tau: float = 0.
    stacked: bool = False

    def __post_init__(self):
        self.root = self.mass.sqrt()
        self.y = self.root * self.values

    def product(self, vector):
        data = self.root * vector
        if self.stacked:
            if self.tau:
                specific = (self.cell @ data) @ self.drug
                share = specific.sum(dim=0) + self.tau*specific
            else:
                share = (self.cell @ data.sum(dim=0)) @ self.drug
        else:
            share = (self.cell @ data) @ self.drug
        return self.root * share

    def preconditioner(self, ridge):
        geometry = self.cell.diagonal()[:, None] * self.drug.diagonal()[None, :]
        return ridge + self.mass * geometry * (1. + self.tau)

    def higher_precision(self):
        return System(self.cell.double(), self.drug.double(), self.mass.double(), self.values.double(),
                      self.tau, self.stacked)


def pcg(system, ridge, tol, *, max_steps=4200, initial=None):
    """Independent residual-checked solve used for accurate replay and numerical recovery."""
    y = system.y
    scale = max(1e-12, float(y.norm()))
    if initial is None or not torch.isfinite(initial).all():
        x, r = torch.zeros_like(y), y.clone()
    else:
        x = initial.clone()
        r = y-system.product(x)-ridge*x
    d = system.preconditioner(ridge)
    relative = float(r.norm()) / scale
    if relative <= tol:
        return system.root*x, 0, relative
    p = r / d
    energy = (r*p).sum()
    for step in range(1, max_steps+1):
        a = system.product(p) + ridge*p
        move = energy / (p*a).sum().clamp_min(1e-32)
        x.add_(p*move)
        r.sub_(a*move)
        if float(r.norm()) <= tol*scale:
            check = (y-system.product(x)-ridge*x).norm().item() / scale
            if check <= tol:
                return system.root*x, step, check
            r = y-system.product(x)-ridge*x
            p = r/d
            energy = (r*p).sum()
            continue
        z = r/d
        new = (z*r).sum()
        p = z + (new/energy)*p
        energy = new
    raise RuntimeError(f"PCG failed with λ={ridge:g}, τ={system.tau:g}, residual {float(r.norm())/scale:.5g}")


@dataclass
class Shift:
    penalty: float
    prediction: torch.Tensor
    direction: torch.Tensor | None = None
    coefficient: float = 0.
    diagonal: float = 0.


def solve_shifts(system, penalties, *, tol=5e-5, max_steps=1400):
    """Share one Lanczos sequence across ridge penalties.

    An incremental LDL factorization of each shifted tridiagonal provides a
    solution update and a residual estimate. Verify each result on the actual
    weighted operator and recover with independent double precision PCG if needed.
    """
    y = system.y
    norm = float(y.norm())
    if norm == 0.:
        return {x: (torch.zeros_like(y), 0, 0., "zero") for x in penalties}
    q = y / norm
    prev = torch.zeros_like(q)
    beta = 0.
    live = {p: Shift(p, torch.zeros_like(q), coefficient=norm) for p in penalties}
    output = {}
    for step in range(1, max_steps+1):
        z = system.product(q) - beta*prev
        alpha = float((z*q).sum())
        z.sub_(alpha*q)
        next_beta = float(z.norm())
        for p, state in list(live.items()):
            if state.direction is None:
                state.diagonal = alpha + p
                state.direction = q
            else:
                lower = beta/state.diagonal
                state.diagonal = alpha + p - lower*beta
                state.coefficient *= -lower
                state.direction = q - lower*state.direction
            if state.diagonal <= 0 or not np.isfinite(state.diagonal):
                break
            ratio = state.coefficient/state.diagonal
            state.prediction.add_(state.direction, alpha=ratio)
            if next_beta*abs(ratio) <= (.8*tol)*norm:
                error = (y-system.product(state.prediction)-p*state.prediction).norm().item() / norm
                if error <= tol:
                    output[p] = (system.root*state.prediction, step, error, "shifted")
                    del live[p]
                # The tridiagonal residual has reached its rounding floor.
                elif next_beta*abs(ratio) <= (.1*tol)*norm:
                    alt, n, err = pcg(system.higher_precision(), p, tol*.8, initial=state.prediction.double())
                    output[p] = (alt.to(y.dtype), n, err, "double_pcg")
                    del live[p]
        else:
            if not live:
                break
            if next_beta > 1e-15 and np.isfinite(next_beta):
                prev, q, beta = q, z/next_beta, next_beta
                continue
        break
    if live:
        high = system.higher_precision()
        for p in live:
            alt, steps, residual = pcg(high, p, tol*.8, initial=live[p].prediction.double())
            output[p] = (alt.to(y.dtype), steps, residual, "double_pcg")
    return output


class ScreenKernel(Kernel):
    def __init__(self, data, fit, query, registry, role):
        super().__init__(data, fit, query, registry, role)
        self._transformed = {}
        self._task_grids = None

    def screen_grids(self):
        if self._task_grids is None:
            weights = row_weights(self.rows)
            self._task_grids = per_screen(self.rows, self.fit_cells, self.fit_drugs, weights)
            _, mass, y = self._task_grids
            np.testing.assert_allclose(mass.sum(0), self.weights.numpy(), atol=5e-7, rtol=2e-7)
            np.testing.assert_allclose(y.sum(0), (self.weights*self.response).numpy(), atol=5e-7, rtol=2e-7)
        return self._task_grids

    def targets_for(self, family):
        if family in self._transformed:
            return self._transformed[family]
        if family == DEBIASED:
            y, s, trace = debias(self.rows)
            table = self.rows.assign(residual=y)
            mass, value = training_grid(table, self.fit_cells, self.fit_drugs, device=self.device)
            result = (mass, value, s, {"fit_only": True, "fit_rows": len(table),
                                       "scale": s, "calibration": trace})
        elif family == BALANCED:
            screens, mass, y = self.screen_grids()
            before = mass.sum(axis=(1,2), dtype=np.float64)
            weight = before.mean()/before
            total = np.einsum("sij,s->ij", mass, weight)
            weighted = np.einsum("sij,s->ij", y, weight)
            value = np.divide(weighted, total, out=np.zeros_like(total), where=total>0)
            result = (torch.as_tensor(total, dtype=torch.float32), torch.as_tensor(value, dtype=torch.float32),
                      1., {"screen_mass_before": dict(zip(screens, before)),
                           "screen_mass_after": dict(zip(screens, before*weight))})
        elif family == TASK:
            screens, mass, y = self.screen_grids()
            values = np.divide(y, mass, out=np.zeros_like(y), where=mass>0)
            result = torch.from_numpy(mass), torch.from_numpy(values), 1., {"fit_screens": list(screens)}
        else:
            assert family == BASELINE
            result = self.weights, self.response, 1., {}
        self._transformed[family] = result
        return result

    def project(self, coefficient, cross, supported, tau, family, mode, scale):
        drug = cross.T.contiguous()
        if family != TASK:
            values = ((self.cross @ coefficient) @ drug).numpy()[self.selector] * scale
        else:
            shared = ((self.cross @ coefficient.sum(0)) @ drug).numpy()[self.selector]
            values = shared.copy()
            if tau and mode == "in_screen":
                for i, screen in enumerate(self.screen_grids()[0]):
                    take = self.query.source_id.eq(screen).to_numpy()
                    if take.any():
                        specific = ((self.cross @ coefficient[i]) @ drug).numpy()
                        values[take] += tau * specific[tuple(j[take] for j in self.selector)]
        base = self.baseline_residual[self.selector]
        return np.where(supported[self.selector[1]], values.astype(float), base)

    def solutions(self, design, family, tau, penalties, *, mode="in_screen"):
        full, supported = self.geometry(design)
        g, cross = full[:self.stop, :self.stop].contiguous(), full[self.stop:, :self.stop]
        mass, target, scale, proof = self.targets_for(family)
        if family == TASK and len(mass) == 1:
            # For a one-screen fit K_task is [1+tau]; scale the penalty in the shared solve.
            system = System(self.cell, g, mass[0], target[0])
            keys = {p: p/(1.+tau) for p in penalties}
            solutions = solve_shifts(system, keys.values())
            personal = {}
            for p, key in keys.items():
                coef, iterations, error, solver = solutions[key]
                pred = self.project(coef[None]/(1.+tau), cross, supported, tau, family, mode, scale)
                personal[p] = (pred, iterations, error, solver)
        else:
            system = System(self.cell, g, mass, target, tau=tau, stacked=family == TASK)
            personal = {
                p: (self.project(coef, cross, supported, tau, family, mode, scale), iterations, error, solver)
                for p, (coef, iterations, error, solver) in solve_shifts(system, penalties).items()
            }
        return personal, full, supported, proof
