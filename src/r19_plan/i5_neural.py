"""Fit-only I2/I4 inputs for the registered FiLM drug branch."""

from dataclasses import dataclass
from itertools import product
from zlib import crc32

import numpy as np
import pandas as pd

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.goal_tracks.t2_upstream import SEED
from src.r11_chemical_probe.common import config
from src.r12_multiaxis.data import Design, Fit
from src.r12_multiaxis.network import fit_network
from src.r15_lincs.design import FitSource
from src.r15_lincs.estimators import GAINS, loss
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_kernel import FitProfile

VIEWS = ("I2", "I4")
MODES = ("full", "mask_only", "parent_shuffle")
PROFILE_RANK = 64
SHUFFLE_SEED = 1501


@dataclass(frozen=True)
class NeuralSplit:
    fit: Fit
    designs: dict[str, Design]
    fit_presence: np.ndarray
    query_presence: np.ndarray


@dataclass(frozen=True)
class NeuralChoice:
    lr: float
    epoch: int
    gain: float
    loss: float


def transform_role(profile, raw, row_parents, name, role):
    """Transform distinct parents, then expand onto wells on the registered order."""
    chosen, expand = np.unique(row_parents, return_inverse=True)
    values, measured = profile.transform(raw[chosen])
    shuffled = values.copy()
    available = np.flatnonzero(measured)
    rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, crc32(name.encode()), role]))
    shuffled[available] = values[rng.permutation(available)]
    return values[expand], shuffled[expand], measured[expand]


def view_blocks(parents, phenotypes, fit_parents, query_parents, query_domains=None):
    """Build fit-only maps and isolated, presence-preserving shuffle controls."""
    axis = pd.Index(parents)
    if not axis.is_unique or axis.isna().any() or set(phenotypes) != set(VIEWS):
        raise ValueError("I5 needs one canonical parent axis and both registered phenotype views")
    fit = axis.get_indexer(fit_parents)
    query = axis.get_indexer(query_parents)
    if not len(fit) or not len(query) or (fit < 0).any() or (query < 0).any():
        raise ValueError("Every fit and held well must have a parent on the registered I5 axis")
    if np.intersect1d(fit, query).size:
        raise ValueError("An I5 query parent reached the fit partition")
    domains = ("query",) * len(query) if query_domains is None else tuple(query_domains)
    if len(domains) != len(query) or any(not isinstance(x, str) or not x.strip() for x in domains):
        raise ValueError("I5 needs one explicit query domain ID per held row")
    domain_ids = pd.Series(domains)
    role_rows = domain_ids.groupby(domain_ids, sort=True).indices
    fit_views, query_views = [], []
    for name in VIEWS:
        raw = np.asarray(phenotypes[name], dtype=np.float64)
        if raw.ndim != 2 or len(raw) != len(axis):
            raise ValueError(f"{name} does not match the canonical parent axis")
        profile = FitProfile.learn(raw[np.unique(fit)], rank=PROFILE_RANK)
        train, shuffled_train, fit_present = transform_role(profile, raw, fit, name, 0)
        held = np.empty((len(query), train.shape[1]))
        shuffled_held = np.empty_like(held)
        query_present = np.empty(len(query), dtype=bool)
        for role, rows in enumerate(role_rows.values(), start=1):
            held[rows], shuffled_held[rows], query_present[rows] = transform_role(
                profile, raw, query[rows], name, role,
            )
        fit_views.append((train, shuffled_train, fit_present))
        query_views.append((held, shuffled_held, query_present))
    return fit_views, query_views


def augment_designs(base, parents, phenotypes, fit_parents, query_parents, query_domains=None):
    fit, query = view_blocks(parents, phenotypes, fit_parents, query_parents, query_domains)
    fit_presence = np.column_stack([entry[2] for entry in fit])
    query_presence = np.column_stack([entry[2] for entry in query])

    def concatenate(original, blocks, present, mode):
        structure, context = original
        if structure.ndim != 2 or context.ndim != 2:
            raise ValueError("The R12 FiLM structure and context branches must be well-by-feature matrices")
        checked_array(structure, (len(present), structure.shape[1]), "R12 structural branch")
        checked_array(context, (len(present), context.shape[1]), "R12 DMSO context branch")
        column = 1 if mode == "parent_shuffle" else 0
        views = [np.zeros_like(block[0]) if mode == "mask_only" else block[column] for block in blocks]
        features = np.column_stack([structure, *views, present]).astype(np.float32)
        return features, context

    designs = {
        mode: Design(
            train=concatenate(base.train, fit, fit_presence, mode),
            query=concatenate(base.query, query, query_presence, mode),
        )
        for mode in MODES
    }
    return designs, fit_presence, query_presence


def prepare_neural(args, bank, fit_rows, query_rows, fit_residual, parents, phenotypes, *, query_domains=None):
    """Build one registered training split without giving the network query expression."""
    genes = bank.landmarks.pr_gene_id.to_numpy()
    if not len(genes) or len(set(genes)) != len(genes):
        raise ValueError("The I5 response landmark axis is missing or duplicated")
    labels = checked_array(fit_residual, (len(fit_rows), len(genes)), "I5 fit residual")
    source = FitSource(bank, fit_rows, config(args.config), args.assets, args.role, train_delta=labels)
    fit = Fit(source, fit_rows, query_rows, panel=genes)
    designs, fit_presence, query_presence = augment_designs(
        fit.inputs("pca"), parents, phenotypes, fit.train.rows.parent, fit.query.rows.parent, query_domains,
    )
    return NeuralSplit(fit, designs, fit_presence, query_presence)


def checkpoint_prediction(split, mode, lr, epoch):
    """Fit the fixed R12 checkpoint, returning ordered well × gene arrays."""
    import torch

    if mode not in MODES or lr not in R12_LRS or epoch not in R12_EPOCHS:
        raise ValueError("An I5 neural checkpoint differs from the registered grid")
    if split.fit.query.treated is not None:
        raise ValueError("I5 FiLM must receive no held treatment expression")
    torch.set_num_threads(4)
    split.fit.cfg = dict(split.fit.cfg, min_epochs=epoch + 1)
    best, _, _, _ = fit_network(split.fit, split.designs[mode], "film", "L0", lr, SEED, "cpu", epoch,
                                patience=False)
    expected = (len(split.fit.query.rows), len(split.fit.panel))
    return checked_array(best["contrast"][2], expected, "I5 neural checkpoint prediction")


def aggregate_prediction(split, fixed, fallback, neural, gain):
    absent = ~split.query_presence.any(axis=1)
    result = fixed + gain * neural
    result[absent] = fallback[absent]
    return result


def select_neural(split, validation_truth, fixed_prediction, fallback, mode):
    """Select only checkpoint and neural gain around an already fixed kernel+anchor."""
    validation_rows = split.fit.query.rows
    shape = (len(validation_rows), len(split.fit.panel))
    truth = checked_array(validation_truth, shape, "independent I5 selection truth")
    fixed = checked_array(fixed_prediction, shape, "fixed I5 kernel and anchor")
    fallback = checked_array(fallback, shape, "common I5 no-view fallback")
    if not truth.any():
        raise ValueError("The selection target has zero energy")
    winner, winner_prediction, best = None, None, None
    for lr, epoch in product(R12_LRS, R12_EPOCHS):
        neural = checkpoint_prediction(split, mode, lr, epoch)
        for gain in GAINS:
            prediction = aggregate_prediction(split, fixed, fallback, neural, gain)
            objective = loss(validation_rows, truth, prediction)
            if not np.isfinite(objective):
                raise ValueError("Nonfinite I5 checkpoint selection objective")
            key = round(objective, 9), gain, str((lr, epoch))
            if best is None or key < best:
                winner, winner_prediction, best = NeuralChoice(lr, epoch, gain, objective), prediction, key
    return winner, winner_prediction


def predict_neural(split, fixed_prediction, fallback, mode, choice):
    """Infer outer effects using the locked inner settings and no query outcomes."""
    if choice.gain not in GAINS or not np.isfinite(choice.loss):
        raise ValueError("An I5 neural gain differs from the registered grid")
    shape = (len(split.fit.query.rows), len(split.fit.panel))
    fixed = checked_array(fixed_prediction, shape, "fixed I5 kernel and anchor")
    fallback = checked_array(fallback, shape, "common I5 no-view fallback")
    neural = checkpoint_prediction(split, mode, choice.lr, choice.epoch)
    return checked_array(aggregate_prediction(split, fixed, fallback, neural, choice.gain), shape,
                         "I5 outer prediction")
