"""Fit the three fixed P4 chemistry arms and five established VC architectures."""

import argparse
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.goal_tracks.t1_data import DOSES, MIN_CELLS, DoseSource, eligibility, slice_name, verify_one
from src.p123.p1_fair import CPU_MODELS, remove_position, repredict
from src.p123.p1_position import MODEL_GAINS
from src.p123.p1_predict import InnerObjective, Siblings
from src.p123.p2_predict import MIXED, SignatureBank
from src.p123.p123_io import write_prediction
from src.r11_chemical_probe.common import config, sha256
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY
from src.r12_multiaxis.data import Fit

from .common import History, NEW, WellProjection, common_args, finish_args


def fit_target(source, train, query, panel):
    train = eligibility(train, source.minimum_cells, source.dose)
    projection = WellProjection(train)
    controls = source.control_counts[:, source.columns(panel)]
    original = source.assemble(train, source.control_rows, controls, source.observed(train, panel))
    if not original.rows[KEY].equals(train[KEY]):
        raise ValueError("Train projection axis changed after DMSO alignment")
    fit = Fit(source, projection.rows, query, panel=panel)
    target = projection.centered(original.delta).astype(np.float32)
    return remove_position(fit, fit.train.delta - target)


def chemical_predictions(fit, parameters, bank):
    sibling = Siblings(fit, np.zeros_like(fit.train.treated))
    direction = sibling.closest(sibling.k_query, parameters["threshold"])
    kernel = sibling.ridge(parameters["alpha"])
    mixed = copy(sibling)
    ids = fit.query.rows[[ENTITY, GROUP]].drop_duplicates().set_index(ENTITY, verify_integrity=True)
    queried = ids.loc[sibling.entities, GROUP].to_numpy()
    priors = parameters["mixed"]
    train, _ = bank.lookup(priors["prior"], sibling.groups)
    query, _ = bank.lookup(priors["prior"], queried)
    fraction = priors["fraction"]
    mixed.k_fit = (1-fraction) * sibling.k_fit + fraction * (train @ train.T)
    mixed.k_query = (1-fraction) * sibling.k_query + fraction * (query @ train.T)
    return dict(zip(NEW, (direction, kernel, mixed.ridge(priors["alpha"]))))


def vc_predictions(fit, models, prior):
    args = SimpleNamespace(t2_root=prior)
    return {f"P4-refit-{name}": repredict(name, fit, models[name], args, None)
            for name in CPU_MODELS}


def predict(args):
    args = finish_args(args)
    reference = History(args, args.dose)
    prior_selection, source_hashes = reference.source_selection(args.minimum_cells, args.fold, args.role)
    previous_p1 = prior_selection["P1"]["chosen"]
    p2 = prior_selection["extra-0"][MIXED]
    prior_vc = {name: prior_selection[name][name] for name in CPU_MODELS}
    source = DoseSource(config(args.config), args.assets, args.stage, args.dose, args.minimum_cells,
                        args.fold, args.role)
    for name in ("fit.npy", f"{args.role}.npy"):
        verify_one(args.stage, source.receipt, source.folder / name)
    if args.lane == "chemical":
        parameters = {"threshold": previous_p1["p1-closest"][0], "alpha": previous_p1["p1-tanimoto"][0],
                      "mixed": {k: p2[k] for k in ("alpha", "prior", "fraction")}}
        bank = SignatureBank(args.p123 / "lincs", reference.old_protocol)
        if bank.receipt_sha256 != prior_selection["extra-0"]["inputs"]["lincs_receipt_sha256"]:
            raise ValueError("P4 mixed kernel requires the same P2 LINCS bank")

        def generate(fit):
            return chemical_predictions(fit, parameters, bank)

        details = {"inherited": parameters, "lincs_receipt_sha256": bank.receipt_sha256}
    else:
        def generate(fit):
            return vc_predictions(fit, prior_vc, args.prior)

        details = {"inherited": {name: {k: v for k, v in chosen.items()
                                       if k in {"selected_epoch", "selected_lr", "selection"}}
                                  for name, chosen in prior_vc.items()}}

    losses = []
    for internal in (0, 1):
        train, validation = source.inner(internal)
        fit = fit_target(source, train, validation, source.inner_panel(internal))
        projection = WellProjection(fit.query.rows)
        response = source.observed(fit.query.rows, fit.panel) - fit.query.control
        observed = projection.centered(response)
        objective = InnerObjective(projection.rows, observed, np.zeros_like(observed))
        losses.append({name: objective.curve(projection.centered(arm)) for name, arm in generate(fit).items()})
        print(f"inner {internal}, {len(fit.train.rows)} fit rows, {len(projection.rows)} validation rows", flush=True)

    outer = fit_target(source, source.fit_rows, source.query_rows, source.panel)
    forecasts = generate(outer)
    details["gains"] = {}
    for name, prediction in forecasts.items():
        curve = (losses[0][name] + losses[1][name]) / 2
        choice = min(range(len(curve)), key=lambda i: (round(float(curve[i]), 10), i))
        gain = float(MODEL_GAINS[choice])
        forecasts[name] = gain * prediction
        details["gains"][name] = {"gain": gain, "inner_energy_fraction": curve.tolist()}
    details.update(fit_rows=len(outer.train.rows), source_receipts=source_hashes, historical=reference.references,
                   module_sha256=sha256(__file__))
    part = slice_name(args.dose, args.minimum_cells, args.fold, args.role)
    write_prediction(args.out / "predictions" / args.lane / part, source, outer, forecasts, details, args.protocol)
    print(f"Recorded {part} {args.lane} with {len(forecasts)} arms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--dose", type=float, choices=DOSES, required=True)
    parser.add_argument("--minimum-cells", type=int, choices=MIN_CELLS, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--role", choices=("A", "B"), required=True)
    parser.add_argument("--lane", choices=("chemical", "vc"), required=True)
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
