"""Produce matched D1/D2 predictions from an isolated balanced-control bundle."""

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.goal_tracks.t2_upstream import EPOCHS, SEED, upstream
from src.r11_chemical_probe.common import config, sha256, write_json
from src.r12_multiaxis.data import Fit
from src.r15_lincs.design import FitSource
from src.r15_lincs.estimators import loss
from src.r15_lincs.forecast import LEGACY, fit_known, position
from src.r16_lincs.predict import HEADS, gain_choice
from src.r19_plan.d1_access import SLICES
from src.r19_plan.d1_inputs import read_bundle
from src.r19_plan.dev_selection import (
    Checkpoint, CheckpointBank, ResponseAxis, checked_array, objectives, pick_checkpoint,
)
from src.r19_plan.i_run import IModelBank

POLICIES = {
    "D1_shared": "joint",
    "D1_independent": "joint",
    "D2_legacy": "legacy_sequential",
    "D2_joint": "joint",
}


def checkpoint_id(parameters):
    return "fixed" if not parameters else "-".join(map(str, parameters))


def selection_policies(rows, genes, anchor, candidates, shared_truth, independent_truth):
    shape = len(rows), len(genes)
    anchor = checked_array(anchor, shape, "common position anchor")
    shared = checked_array(shared_truth, shape, "fit-role selection truth")
    independent = checked_array(independent_truth, shape, "select-role selection truth")
    axis = ResponseAxis(tuple(rows[["study", "inst_id"]].itertuples(index=False, name=None)), tuple(genes))
    bank = CheckpointBank(axis, anchor, tuple(
        Checkpoint(checkpoint_id(parameters), parameters, checked_array(value, shape, f"checkpoint {parameters}"))
        for parameters, value in candidates.items()
    ))
    if not bank.checkpoints:
        raise ValueError("D1/D2 require actual gene-aligned checkpoint predictions")
    by_config = {item.config_id: item.residual for item in bank.checkpoints}
    shared_grid = objectives(rows, shared, anchor, by_config)
    independent_grid = objectives(rows, independent, anchor, by_config)
    return {arm: pick_checkpoint(bank, shared_grid if arm == "D1_shared" else independent_grid, policy)
            for arm, policy in POLICIES.items()}


def checkpoint_predictions(args, bank, fit_rows, select_rows, labels, anchor, work):
    model = args.model
    if model in HEADS.values():
        parameters = product(R12_LRS, R12_EPOCHS)
    elif model not in EPOCHS:
        parameters = [()]
    else:
        import torch

        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        source = FitSource(bank, fit_rows, config(args.config), args.assets, bank.fit, train_delta=labels)
        fit = Fit(source, fit_rows, select_rows, panel=bank.landmarks.pr_gene_id.to_numpy())
        if fit.query.treated is not None:
            raise ValueError("D2 legacy checkpoint bank has access to a selection label during fit")
        # The callback returns actual same-path checkpoint arrays. It never scores or stops training.
        return {(epoch,): value for epoch, value in upstream(model)(fit, work, EPOCHS[model], lambda p: p).items()}
    return {params: fit_known(args, bank, fit_rows, select_rows, labels, None, anchor, model, work, params)
            for params in parameters}


def fit_policies(args, bank):
    inner, select, train, held = bank.split()
    genes = bank.landmarks.pr_gene_id.to_numpy()
    fit_role, select_role = bank.fit, bank.select
    shared_truth, independent_truth = (bank.response(select, role) for role in (fit_role, select_role))
    inner_raw, inner_pos, select_pos = position(bank, inner, select, fit_role)
    position_gain = gain_choice(select, independent_truth, select_pos, np.zeros_like(select_pos))
    select_anchor = position_gain * select_pos
    inner_labels = inner_raw - position_gain * inner_pos
    raw, train_pos, held_pos = position(bank, train, held, fit_role)
    outer_anchor = position_gain * held_pos
    outer_labels = raw - position_gain * train_pos
    vehicle_difference = shared_truth - independent_truth
    with TemporaryDirectory(prefix="d12-fit-") as temporary:
        work = Path(temporary)
        (work / "select").mkdir()
        candidates = checkpoint_predictions(args, bank, inner, select, inner_labels, select_anchor, work / "select")
        choices = selection_policies(select, genes, select_anchor, candidates, shared_truth, independent_truth)
        outer_checkpoints = {}
        for params in sorted({choice.parameters for choice in choices.values() if choice.gain > 0}):
            folder = work / checkpoint_id(params)
            folder.mkdir()
            outer_checkpoints[params] = fit_known(args, bank, train, held, outer_labels, None, outer_anchor,
                                                  args.model, folder, params)

    shape = len(held), len(genes)
    forecasts, selection = {"B_pos": outer_anchor}, {}
    for arm, choice in choices.items():
        point = candidates[choice.parameters]
        selected = select_anchor + choice.gain * point
        norms = np.linalg.norm(point) * np.linalg.norm(vehicle_difference)
        vehicle_cosine = float(np.sum(point * vehicle_difference) / norms) if norms else None
        selection[arm] = dict(asdict(choice), independent_proxy_loss=loss(select, independent_truth, selected),
                              shared_proxy_loss=loss(select, shared_truth, selected), vehicle_cosine=vehicle_cosine)
        forecast = (outer_anchor + choice.gain * outer_checkpoints[choice.parameters]
                    if choice.gain else outer_anchor.copy())
        forecasts[arm] = checked_array(forecast, shape, arm)
    if not np.array_equal(forecasts["D1_independent"], forecasts["D2_joint"]):
        raise ValueError("D1 independent and D2 joint predictions diverged on their identical inputs")
    selection["position_gain"] = position_gain
    return forecasts, selection


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.model == "PRnet-well-adaptation" and args.prnet is None:
        raise ValueError("The registered PRnet model requires its existing upstream checkout")
    data = read_bundle(args.bundle, args.access_reference, args.slice)
    bank = IModelBank(data, args.slice)
    args.role, args.fold = bank.fit, bank.fold
    forecasts, selection = fit_policies(args, bank)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".d12-forecast-") as temp:
        stage = Path(temp)
        bank.query_axis.to_csv(stage / "rows.tsv", sep="\t", index=False)
        bank.landmarks.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecasts.npz", **forecasts)
        write_json(stage / "selection.json", selection)
        write_json(stage / "receipt.json", dict(slice=args.slice, model=args.model, seed=SEED,
            protocol_sha256=sha256(args.protocol), config_sha256=sha256(args.config), code_sha256=sha256(__file__),
            inputs=data.input_receipts, arms=sorted(forecasts),
            files={p.name: sha256(p) for p in stage.iterdir()},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", choices=SLICES, required=True)
    parser.add_argument("--model", choices=LEGACY, required=True)
    for name in ("bundle", "access-reference", "protocol", "config", "assets", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prnet", type=Path)
    print(run(parser.parse_args()))
