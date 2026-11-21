"""Registered arm definitions and fit implementations shared by development and inference."""

from dataclasses import dataclass, replace
from functools import cached_property
import time

import numpy as np
from scipy.linalg import eigh

from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.models import Interaction, WeightedRidge

from .network import fit_network


BASELINES = ("B0", "B1", "B2", "B3", "B4", "B5")
CONTEXTS = ("C2", "C3", "C4", "C7", "C8", "C9")
STRUCTURES = ("A2", "A3", "A5", "A6")
LOSSES = ("L1", "L2", "L3", "L4", "L6")
SHIFT_BASES = ("B2", "B3", "B4", "B5", "A4")


@dataclass(frozen=True)
class Candidate:
    arm: str
    method: str
    context: str = "pca"
    head: str = "concat"
    loss: str = "L0"
    lr: float = 0.0003
    horizon: int = 100
    epoch: int = 0
    vic: float = 0.0
    value: float = 0.0
    drug: str = "E"
    dim: int = 32
    response_rank: int = 0

    @property
    def network(self):
        return self.method in {"nn", "dose", "dose_match"}


def registry(fit):
    cfg = fit.cfg
    arms = [
        Candidate("B0", "zero"),
        Candidate("B1", "line"),
        *(Candidate("B2", "pool", value=v) for v in cfg["pool_shrinkage"]),
        *(Candidate(a, m, value=alpha) for a, m in (("B3", "full_ridge"), ("B5", "ridge"))
          for alpha in cfg["legacy_ridge"]),
        Candidate("B4", "nn", "full", "full"),
        Candidate("C1", "nn", "none"),
        Candidate("C2", "nn", "stable"),
        *(Candidate("C3", "nn", f"nuisance{k}") for k in cfg["context_ranks"] if 0 < k <= len(fit.nuisance)),
        Candidate("C3_rank0", "nn", "nuisance0"),
        *(Candidate(f"A{i + 1}", "nn", context, head) for i, (context, head) in
          enumerate((c, h) for c in ("raw", "pca") for h in ("concat", "film", "bilinear"))),
        *(Candidate(a, "nn", loss=a) for a in ("L1", "L2", "L3", "L5")),
        Candidate("L6", "nn", loss="L6", vic=cfg["vicreg"][0]),
        Candidate("J1_FiLM_L2", "nn", head="film", loss="L2"),
        *(Candidate(f"E{a}_{k}", "ridge", value=alpha, drug=drug, dim=k)
          for a, drug in ((1, "E"), (2, "C")) for k in (32, 128) for alpha in cfg["extension_ridge"]),
        *(Candidate(f"E4_{k}", "nn", response_rank=k) for k in (32, 64)),
    ]
    paths = fit.source.assets
    if fit.cell_stats is not None:
        arms += [Candidate("C4", "nn", "moments"), Candidate("C4_mean", "nn", "sample_mean"),
                 Candidate("L4", "nn", loss="L4")]
    for arm, folder in (("C7", "scgpt"), ("C8", "uce")):
        if (paths / folder / "receipt.json").exists():
            arms += [Candidate(arm, "nn", arm), Candidate(f"{arm}_mean", "nn", f"{arm}_mean")]
    if (paths / "depmap/receipt.json").exists():
        arms += [Candidate("C9", "nn", "C9"), Candidate("C9_id", "nn", "C9_id")]
    if "M" in fit.source.chemical:
        arms += [Candidate("E3", "ridge", value=alpha, drug="M") for alpha in cfg["extension_ridge"]]
    if (paths / "doses/source/receipt.json").exists():
        arms += [Candidate("D1", "dose"), Candidate("D1_match", "dose_match")]
    return arms


def expansion(arms, cfg):
    lr = cfg["learning_rates"][1]
    follow = [replace(a, lr=lr) for a in arms if a.network]
    vic = next(a for a in arms if a.arm == "L6")
    return follow + [replace(vic, vic=v) for v in cfg["vicreg"][1:]]


def shift_base(arm, cfg):
    if arm == "B2":
        return Candidate("S_base_B2", "pool", value=cfg["offset_shrinkage"])
    if arm in {"B3", "B5"}:
        return Candidate(f"S_base_{arm}", "full_ridge" if arm == "B3" else "ridge", value=1000)
    if arm == "B4":
        return Candidate("S_base_B4", "nn", "full", "full", epoch=20)
    return Candidate("S_base_A4", "nn", epoch=20)


class RidgePath:
    def __init__(self, x, y, z, rows):
        self.fit = WeightedRidge(x, y, rows)
        self.z = z
        self.design_cols = x.shape[1]
        self.coefs = {}

    @cached_property
    def spectral(self):
        lam, u = eigh(self.fit.gram, check_finite=False, driver="evd")
        return np.maximum(0, lam), u, u.T @ self.fit.cross

    def predict(self, alpha, retain):
        lam, u, cross = self.spectral
        coefficient = u @ (cross / (lam[:, None] + alpha))
        if retain:
            self.coefs[alpha] = coefficient
        return self.fit.predict(self.z, coefficient)

    def checkpoint(self, alpha):
        return dict(coef=self.coefs[alpha], xmean=self.fit.xmean, ymean=self.fit.ymean)


class Evaluator:
    def __init__(self, fit, device):
        self.fit = fit
        self.device = device
        self.ridge = {}
        self.dose_cache = {}

    def ridge_path(self, candidate):
        key = (candidate.method, candidate.drug, candidate.dim)
        if key not in self.ridge:
            t, v = self.fit.train, self.fit.query
            if candidate.method == "full_ridge":
                design = self.fit.inputs("full")
                x, z = design.train[0].astype(np.float64), design.query[0].astype(np.float64)
            else:
                tx = Interaction(t, candidate.drug, candidate.dim, 16)
                x, z = tx.transform(t), tx.transform(v)
            self.ridge[key] = RidgePath(x, self.fit.residual, z, t.rows)
        return self.ridge[key]

    def evaluate(self, candidate, seed=42, retain=False, replacement=None):
        start = time.monotonic()
        f = self.fit
        metrics = {}
        checkpoint, curve, diagnostic = None, [], {}
        if candidate.method == "zero":
            pred = np.zeros_like(f.offset)
        elif candidate.method == "line":
            pred = f.means.predict(f.query.rows, np.inf)[0]
        elif candidate.method == "pool":
            pred = f.means.predict(f.query.rows, candidate.value)[0]
        elif not candidate.network:
            ridge = self.ridge_path(candidate)
            pred = f.offset + ridge.predict(candidate.value, retain)
            metrics["design_cols"] = ridge.design_cols
            if retain:
                checkpoint = ridge.checkpoint(candidate.value)
        if not candidate.network:
            pairs = {key: (None, 0, pred, checkpoint) for key in ("abs", "contrast")}
            metrics.update(device="CPU", seconds=time.monotonic() - start, params=0,
                           seed="deterministic", epochs=0, updates=0)
        else:
            model_fit = f
            design = f.inputs(candidate.context, augment=candidate.loss in {"L5", "L6"},
                              rank=candidate.response_rank or None)
            count = None
            if candidate.method in {"dose", "dose_match"}:
                from .dose import dose_design
                model_fit, design, count = dose_design(f, augmented=candidate.method == "dose")
            pairs, metrics, curve, diagnostic = fit_network(
                model_fit, design, candidate.head, candidate.loss, candidate.lr, seed, self.device,
                candidate.epoch or candidate.horizon, patience=not bool(candidate.epoch),
                vic=candidate.vic, replacement=replacement, pack_count=count,
            )
        return pairs, metrics, curve, diagnostic


def score(fit, prediction):
    return Contrast(fit.query.rows, fit.source.cfg["minimum_groups"]).points(prediction, fit.query.delta)[:2]


def alternatives(fit, seeds):
    controls_by_plate = {}
    for key, plate in fit.ctrl_meta.items():
        controls_by_plate.setdefault(plate, []).append(key)
    designs = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        mapping = {}
        for keys in controls_by_plate.values():
            shuffled = [keys[i] for i in rng.permutation(len(keys))]
            mapping.update(zip(shuffled, shuffled[1:] + shuffled[:1]))
        designs[f"C5_seed{seed}"] = fit.inputs("raw", replacement=[mapping[k] for k in fit.query_units])
    n_alternative = max(len(keys) - 1 for keys in fit.by_line.values())
    for index in range(n_alternative):
        replace_units = []
        actual = 0
        for key in fit.query_units:
            others = [k for k in fit.by_line.get(key[1], []) if fit.ctrl_meta[k] != fit.ctrl_meta[key]]
            if len(others) > index:
                replace_units.append(others[index])
                actual += 1
            else:
                replace_units.append(key)
        label = f"C6_alt{index + 1}"
        designs[label] = fit.inputs("raw", replacement=replace_units)
        fit.query.rows[label + "_source_well"] = [k[0] for k in replace_units]
        fit.query.rows[label + "_changed"] = [a != b for a, b in zip(fit.query_units, replace_units)]
        if actual == 0:
            del designs[label]
    return designs
