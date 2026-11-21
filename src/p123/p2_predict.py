"""Fit one frozen LINCS-prior arm per family using only fit Tahoe parent responses."""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.utils.extmath import randomized_svd

from src.goal_tracks.t1_data import MIN_GROUPS, slice_name
from src.goal_tracks.t1_io import load_task
from src.r11_chemical_probe.common import read_json, sha256
from src.r11_chemical_probe.dataset import ENTITY, GROUP
from src.r11_chemical_probe.metrics import Contrast
from src.r12_multiaxis.data import Fit
from src.r13_optimization.kernels import _tanimoto
from src.r13_optimization.objectives import center
from src.r13_optimization.transfer import mean_units, unit_norm

from .p1_position import MODEL_GAINS, load_context, load_source, source_args
from .p123_io import write_prediction
from .p2_lincs import PROTOCOL_SHA256

PRESETS = ("pan_near10", "pan_anydose", "hepatic_near10")
ALPHAS = (0.1, 1.0, 10.0)
MIX = (0.25, 0.5, 0.75)
BASE = "P2-Tanimoto"
COSINE = "P2-LINCS-cosine"
MIXED = "P2-Tanimoto-LINCS"
DIRECT = "P2-LINCS-direct"
IMPUTED = "P2-LINCS-imputed"
ARMS = (BASE, COSINE, MIXED, DIRECT, IMPUTED)


@dataclass(frozen=True)
class Specification:
    arm: str
    alpha: float
    prior: str = ""
    fraction: float = 0.
    rank: int = 0


def candidate_specs():
    for alpha in ALPHAS:
        yield Specification(BASE, alpha)
    for prior, alpha in product(PRESETS, ALPHAS):
        yield Specification(COSINE, alpha, prior, 1.)
        for part in MIX:
            yield Specification(MIXED, alpha, prior, part)
        for k in (16, 32):
            yield Specification(DIRECT, alpha, prior, rank=k)
        yield Specification(IMPUTED, alpha, prior, .5, 16)


def positive_ridge(gram, rhs, alpha):
    reg = np.array(gram, dtype=np.float64, copy=True)
    np.fill_diagonal(reg, reg.diagonal() + alpha)
    return cho_solve(cho_factor(reg, check_finite=False), rhs, check_finite=False)


def group_order(block):
    rows = block.rows
    groups, codes = np.unique(rows[GROUP], return_inverse=True)
    # Use one verified, deterministic representative per canonical parent.
    first = rows.sort_values([GROUP, ENTITY], kind="stable").reset_index(names="row").drop_duplicates(GROUP)
    if not np.array_equal(groups, first[GROUP]):
        raise ValueError("Canonical fingerprint indexing disagrees with parent ordering")
    return groups, codes, block.chemical["E"][first.row.to_numpy()].astype(np.float64)


class SignatureBank:
    def __init__(self, path, protocol):
        receipt_path = path / "receipt.json"
        receipt = read_json(receipt_path)
        bank_path = path / "lincs_tahoe.npz"
        if (receipt["protocol_sha256"] != sha256(protocol)
                or sha256(bank_path) != receipt["files"][bank_path.name]):
            raise ValueError("LINCS input bank or prespec differs")
        self.receipt_sha256 = sha256(receipt_path)
        with np.load(bank_path, allow_pickle=False) as cached:
            self.keys = pd.Index(cached["keys"].astype(int))
            if not self.keys.is_unique or len(self.keys) != 203 or len(cached["entrez"]) != 978:
                raise ValueError("LINCS bank does not match the canonical 203 parent and landmark axis")
            self.signatures = {name: cached[name].astype(np.float64) for name in PRESETS}
            self.support = {name: cached[f"has_{name}"] for name in PRESETS}

    def lookup(self, name, group):
        i = self.keys.get_indexer(group)
        if np.any(i < 0):
            raise ValueError("A registered parent lacks coverage metadata")
        return self.signatures[name][i], self.support[name][i]


class TahoeRegressor:
    def __init__(self, fit, positions, bank):
        self.fit = fit
        self.bank = bank
        self.train, self.train_ix, xc = group_order(fit.train)
        self.query, self.query_ix, zc = group_order(fit.query)
        if set(self.train) & set(self.query):
            raise ValueError("Held Tahoe treatment parent in the fit")
        within, cross = _tanimoto(xc, xc), _tanimoto(zc, xc)
        self.chem = (within, cross)
        self.chem_kwargs = {"bank": bank.receipt_sha256}
        y = center(Contrast(fit.train.rows, MIN_GROUPS), fit.train.delta - positions[0])
        self.response = mean_units(y, self.train_ix, len(self.train), fit.weight)[0].astype(np.float64)
        self.cache = {}

    @cached_property
    def dimensions(self):
        return self.fit.panel.size

    def profile_pair(self, prior):
        return self.bank.lookup(prior, self.train), self.bank.lookup(prior, self.query)

    def transform(self, spec):
        key = (spec.arm, spec.prior, spec.fraction, spec.rank)
        if key not in self.cache:
            self.cache[key] = self.kernel_pair(spec)
        return self.cache[key]

    def kernel_pair(self, spec):
        if spec.arm == BASE:
            return self.chem
        (train, tr_has), (query, q_has) = self.profile_pair(spec.prior)
        if spec.arm in (DIRECT, IMPUTED) and tr_has.sum() >= 3:
            # Only fit-parent signature rows set the mean and span of the latent coordinates.
            observed = train[tr_has]
            m = observed.mean(axis=0)
            rank = min(spec.rank, len(observed) - 1, observed.shape[1])
            _, _, components = randomized_svd(observed - m, n_components=rank, random_state=42)
            if spec.arm == DIRECT:
                train, query = (train - m) @ components.T, (query - m) @ components.T
                train[~tr_has], query[~q_has] = 0., 0.
            else:
                # Structure-to-signature bridge learns PCA16 scores on fit signatures.
                # Zero held-treatment profile information enters this source decoder.
                encoded = (observed - m) @ components.T
                within, cross = self.chem
                coef = positive_ridge(within[np.ix_(tr_has, tr_has)], encoded, 1.)
                if (~tr_has).any():
                    train[~tr_has] = unit_norm(m + (within[np.ix_(~tr_has, tr_has)] @ coef) @ components)
                if (~q_has).any():
                    query[~q_has] = unit_norm(m + (cross[np.ix_(~q_has, tr_has)] @ coef) @ components)
        elif spec.arm == DIRECT:
            train = np.zeros_like(train)
            query = np.zeros_like(query)
        cosine = (train @ train.T, query @ train.T)
        if spec.arm in (MIXED, IMPUTED):
            # A measured signature is absent as an all-zero vector. In the mixed
            # arm the structural kernel remains available for every compound.
            return tuple((1. - spec.fraction) * a + spec.fraction * b for a, b in zip(self.chem, cosine))
        return cosine

    def predict_groups(self, spec):
        within, cross = self.transform(spec)
        return cross @ positive_ridge(within, self.response, spec.alpha)


class PooledLoss:
    """Use small parent Gram algebra for exact pool-centered SSE/SST."""

    def __init__(self, fit, truth, position, query_ix, n):
        metric = Contrast(fit.query.rows, MIN_GROUPS)
        design = center(metric, np.eye(n, dtype=np.float64)[query_ix])[metric.active]
        e = center(metric, truth - position)[metric.active]
        self.sst = np.square(center(metric, truth)[metric.active]).sum()
        if self.sst <= 0.:
            raise ValueError("No measurable response variation in inner pool")
        self.error_energy = np.square(e).sum()
        self.cross = design.T @ e
        self.gram = design.T @ design

    def evaluate(self, groups):
        cross = np.sum(groups * self.cross)
        energy = np.sum(groups * (self.gram @ groups))
        return (self.error_energy - 2 * MODEL_GAINS * cross + MODEL_GAINS**2 * energy) / self.sst


def select_arm(score, specs):
    options = defaultdict(list)
    for j, spec in enumerate(specs):
        for k, gain in enumerate(MODEL_GAINS):
            # Choose the smaller response scale when observed energy is tied.
            options[spec.arm].append((round(float(score[j, k]), 10), gain, j, k))
    return {arm: min(items)[2:] for arm, items in options.items()}


def fit_predict(args):
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("P1–P3 protocol differs from the committed bytes")
    source = load_source(args)
    bank = SignatureBank(args.lincs, args.protocol)
    specs = tuple(candidate_specs())
    loss = []
    for inner in (0, 1):
        fit = Fit(source, *source.inner(inner), panel=source.inner_panel(inner))
        pos = load_context(args.positions, source, fit, inner, protocol=args.protocol)
        reg = TahoeRegressor(fit, pos, bank)
        truth = source.observed(fit.query.rows, fit.panel) - fit.query.control
        evaluator = PooledLoss(fit, truth, pos[1], reg.query_ix, len(reg.query))
        loss.append(np.stack([evaluator.evaluate(reg.predict_groups(spec)) for spec in specs]))
        print(f"inner {inner}: {len(specs)} preset, alpha and family combinations", flush=True)
    score = np.mean(loss, axis=0)
    selected = select_arm(score, specs)
    final = Fit(source, source.fit_rows, source.query_rows, panel=source.panel)
    pos = load_context(args.positions, source, final, -1, protocol=args.protocol)
    reg = TahoeRegressor(final, pos, bank)
    name = slice_name(source.dose, source.minimum_cells, source.outer, source.role)
    reference = args.prior / "t1/first" / f"d{source.dose:g}/predictions" / name
    prior_protocol = args.t1_protocol
    _, rows, panel, historical = load_task(reference, source.staged, prior_protocol, source.dose,
                                            source.minimum_cells, source.outer, source.role)
    if not np.array_equal(panel, final.panel) or not rows[GROUP].equals(final.query.rows[GROUP]):
        raise ValueError("T1 anchor uses a different outer partition")
    anchor = historical["B_abs"] + pos[1]
    models = {}
    recipe = {}
    for arm, (j, k) in selected.items():
        rule, gain = specs[j], MODEL_GAINS[k]
        raw = reg.predict_groups(rule)[reg.query_ix]
        models[arm] = anchor + float(gain) * raw
        recipe[arm] = {**asdict(rule), "gain": float(gain), "inner_sse_sst": [float(a[j, k]) for a in loss],
                       "energy_loss": float(score[j, k]), "fit_parents": len(reg.train), "held_parents": len(reg.query)}
        if rule.prior:
            recipe[arm].update(fit_measured=int(bank.lookup(rule.prior, reg.train)[1].sum()),
                               query_measured=int(bank.lookup(rule.prior, reg.query)[1].sum()))
    recipe["inputs"] = {"module_sha256": sha256(__file__), "lincs_receipt_sha256": bank.receipt_sha256,
                        "location_receipt_sha256": sha256(args.positions / name / "receipt.json"),
                        "t1_task_sha256": sha256(reference / "receipt.json")}
    write_prediction(args.out / name, source, final, models, recipe, args.protocol)
    print(f"Wrote {name} with {len(models)} fitted prediction arms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_args(parser)
    for key in ("lincs", "positions", "prior", "t1-protocol", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    fit_predict(parser.parse_args())


if __name__ == "__main__":
    main()
