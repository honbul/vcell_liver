"""Select R16 component gains inside the R15 fit and write forecasts."""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS, run_chemberta
from src.goal_tracks.t2_prnet import fit_model as fit_prnet
from src.goal_tracks.t2_upstream import EPOCHS, SEED, upstream
from src.r11_chemical_probe.common import config, read_json
from src.r12_multiaxis.data import Fit
from src.r15_lincs.design import Bank, FitSource
from src.r15_lincs.estimators import GAINS, loss
from src.r15_lincs.forecast import (LEGACY, PROPOSED, StructureMap, choose_proposed, fit_known,
                                   grid_chemical, pick, position, setup, split, write_forecast)


HEADS = {"blend_FiLM": "FiLM_R12", "blend_concat": "Concat_R12"}
GAIN_SUFFIX = "_gain"
CELL_GAIN = "B_cell_gain"


def inner_known(args, bank, inner, validation, observed, truth, anchor, model, work, archived=None):
    if model in HEADS.values():
        # The historic validation helper returns the scalar centered error from its
        # epoch curve. Request actual checkpoint arrays through its inference path.
        choices = [archived] if archived is not None else product(R12_LRS,R12_EPOCHS)
        predictions = {key: fit_known(args,bank,inner,validation,observed,None,anchor,model,work,key)
                       for key in choices}
    else:
        import torch

        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        features = StructureMap(args.assets) if model == "ChemBERTa_R11" else None
        source = FitSource(bank, inner, config(args.config), args.assets, args.role, features, observed)
        fit = Fit(source, inner, validation, panel=bank.landmarks.pr_gene_id.to_numpy())
        if model == "ChemBERTa_R11":
            value, _ = run_chemberta(fit, False)
            predictions = {(): value}
        elif model == "PRnet-well-adaptation":
            value, _ = fit_prnet(fit, 10., args.prnet)
            predictions = {(): value}
        else:
            values, _ = upstream(model)(fit, work, EPOCHS[model], lambda p: p)
            predictions = {(epoch,): value for epoch, value in values.items()}
    if any(np.shape(value)!=truth.shape or not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError(f"{model} validation did not return finite, gene-aligned predictions")
    settings = {key: loss(validation, truth, anchor+value) for key, value in predictions.items()}
    best = pick(settings)
    chosen = best if archived is None else archived
    return chosen, predictions[chosen], best


def gain_choice(rows, truth, residual, anchor):
    return pick({gain: loss(rows, truth, anchor+gain*residual) for gain in GAINS})


def frozen_path(args, model):
    return args.prior/"predict"/args.scenario/f"f{args.fold}-{args.role}"/model


def historic(args, model):
    record = read_json(frozen_path(args, model)/"receipt.json")["selection"]
    if model == PROPOSED:
        return (tuple(record["parameters"][0]), record["parameters"][1]), record["position_gain"]
    return tuple(record["parameters"]), record["position_gain"]


def save(args, bank, rows, model, arrays, settings):
    stage = args.out/"predict"/args.scenario/f"f{args.fold}-{args.role}"/model
    write_forecast(args, bank, stage, rows, arrays, dict(model=model, **settings))


def forecast(args):
    if args.confirm:
        from .confirmation import ConfirmBank, confirm_position, confirm_split

        bank = ConfirmBank(args.cache, args.protocol, args.scenario, args.cell)
        inner, validation, training, outer = confirm_split(bank, args.fold)
        get_position = confirm_position
    else:
        bank = Bank(args.cache, args.r15_protocol, args.scenario)
        inner, validation, training, outer = split(bank, args.fold)
        get_position = position
    if set(training.parent) & set(outer.parent) or set(inner.parent) & set(validation.parent):
        raise ValueError("Parent leakage into fit")
    if args.confirm:
        labels, fit_pos, val_pos = get_position(bank, inner, validation, args.role)
        truth = bank.response(validation, args.role)
        mask = validation.cell_id.eq(args.cell).to_numpy()
        position_gain = gain_choice(validation.loc[mask], truth[mask], val_pos[mask], np.zeros_like(truth[mask]))
    else:
        labels, truth, fit_pos, val_pos, position_gain = setup(bank, inner, validation, args.role)
        mask = np.ones(len(validation), dtype=bool)
    inner_cell, val_cell = bank.transfer(inner), bank.transfer(validation)
    inner_anchor = position_gain*val_pos+val_cell
    observed = labels-position_gain*fit_pos-inner_cell
    query = validation.loc[mask]
    inner_truth, scoring_anchor = truth[mask], inner_anchor[mask]
    raw, outer_fit_pos, query_pos = get_position(bank, training, outer, args.role)
    final_cell = bank.transfer(outer)
    final_pos = position_gain*query_pos
    final_anchor = final_pos+final_cell
    final_labels = raw-position_gain*outer_fit_pos-bank.transfer(training)
    audit = dict(position_gain=position_gain, fit_groups=sorted(set(training.parent)))
    model = args.model

    def known(name, work):
        # Restrict configuration selection to the confirmation target cell.
        archived = None
        if not args.confirm:
            archived, old_position_gain = historic(args, name)
            if old_position_gain != position_gain:
                raise ValueError(f"R15 position gain did not reproduce for {name}")
        chosen, inner_prediction, reproduced = inner_known(args, bank, inner, query, observed,
            inner_truth, scoring_anchor, name, work, archived)
        if reproduced != chosen:
            print(f"Preserving archived R15 settings {chosen} for {name}, reproduced winner {reproduced}",flush=True)
        if args.confirm:
            final_prediction = fit_known(args, bank, training, outer, final_labels, None,
                                         final_anchor, name, work, chosen)
        else:
            with np.load(frozen_path(args,name)/"forecast.npz", allow_pickle=False) as archive:
                final_prediction = archive[name]-final_anchor
        return chosen, inner_prediction, final_prediction, reproduced

    if model in LEGACY:
        with TemporaryDirectory(prefix=f"r16-{model}-") as work:
            settings, val_pred, final_pred, reproduced = known(model, Path(work))
        gain = gain_choice(query, inner_truth, val_pred, scoring_anchor)
        arrays = {model+GAIN_SUFFIX: final_anchor+gain*final_pred}
        save(args, bank, outer, model, arrays, dict(audit, parameters=settings, gain=gain, reproduced_inner_settings=reproduced))
        return

    rule, r15_gain = choose_proposed(bank, inner, query, labels, inner_truth, fit_pos,
                                     val_pos[mask], position_gain)[0] if not args.confirm else choose_r15_confirm(
                                         bank, inner, query, observed, inner_truth, scoring_anchor)
    if not args.confirm and historic(args, PROPOSED) != ((rule, r15_gain), position_gain):
        raise ValueError("R15 proposed inner selection did not reproduce")
    val_kernel = grid_chemical(bank, inner, query, observed, [rule])[rule]
    final_kernel = grid_chemical(bank, training, outer, final_labels, [rule])[rule]
    cell_gain = gain_choice(query, inner_truth, val_cell[mask], np.zeros_like(inner_truth)) if args.scenario == "B" else 0.
    scaled_anchor = final_pos+cell_gain*final_cell
    val_blend_anchor = position_gain*val_pos[mask]+cell_gain*val_cell[mask]

    if model == PROPOSED:
        pred = final_anchor+r15_gain*final_kernel
        if not args.confirm:
            with np.load(frozen_path(args,PROPOSED)/"forecast.npz", allow_pickle=False) as archive:
                if not np.allclose(pred, archive[PROPOSED], atol=2e-6, rtol=1e-6):
                    raise ValueError("R15 selected forecast did not reproduce")
                pred = archive[PROPOSED]
        arrays = {PROPOSED: pred, "B0": np.zeros_like(pred), "B_pos": final_pos}
        if args.scenario == "B":
            arrays[CELL_GAIN] = cell_gain*final_cell
        save(args, bank, outer, model, arrays, dict(audit, rule=rule, r15_gain=r15_gain, cell_gain=cell_gain))
        return

    with TemporaryDirectory(prefix=f"r16-{model}-") as work:
        settings, val_neural, final_neural, reproduced = known(HEADS[model], Path(work))
    a, b = pick({(a,b): loss(query, inner_truth, val_blend_anchor+a*val_kernel+b*val_neural)
                 for a,b in product(GAINS, repeat=2)})
    pred = scaled_anchor+a*final_kernel+b*final_neural
    save(args, bank, outer, model, {model: pred},
         dict(audit, rule=rule, r15_gain=r15_gain, cell_gain=cell_gain, parameters=settings, reproduced_inner_settings=reproduced, a=a, b=b))


def choose_r15_confirm(bank, inner, query, observed, truth, anchor):
    from src.r15_lincs.estimators import candidates

    rules = candidates(bank.scenario)
    predictions = grid_chemical(bank, inner, query, observed, rules)
    return pick({(rule,gain): loss(query, truth, anchor+gain*predictions[rule])
                 for rule in rules for gain in GAINS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "config", "protocol", "assets", "prnet", "out", "prior", "r15-protocol"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--model", choices=(PROPOSED,*LEGACY,*HEADS), required=True)
    parser.add_argument("--scenario", choices="AB", required=True)
    parser.add_argument("--fold", type=int, choices=range(3), required=True)
    parser.add_argument("--role", choices="AB", required=True)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--cell")
    forecast(parser.parse_args())


if __name__ == "__main__":
    main()
