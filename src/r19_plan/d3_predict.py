"""Fit D3 control augmentation and nuisance shrinkage on balanced DEV inputs."""

import argparse
from copy import copy
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.goal_tracks.t2_upstream import SEED
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r12_multiaxis.data import Design, Fit
from src.r12_multiaxis.network import fit_network, packs
from src.r15_lincs.design import FitSource
from src.r15_lincs.forecast import choose_proposed, grid_chemical, position
from src.r16_lincs.predict import gain_choice
from src.r19_plan.d1_inputs import read_bundle
from src.r19_plan.dev_selection import PhysicalControls, checked_array
from src.r19_plan.dev_vehicle import STRENGTHS, augment_fit, learn_nuisance_basis, shrink_neural_residual
from src.r19_plan.i_reference import choose_blend
from src.r19_plan.i_run import IModelBank
from src.r19_plan.i_stage import parse_tsv, pinned_bytes, slice_roles
from src.r19_plan.sl_components import feature_control

ARMS = ("D3a", "D3b", "D3c", "D3a_parent_shuffle", "D3b_parent_shuffle", "D3c_parent_shuffle")
WELL = ["rna_plate", "rna_well"]


def controls_from_inputs(args, staged):
    original = read_json(args.control_reference)
    rows = parse_tsv(pinned_bytes(args.assignments, original["files"]["assignments.tsv"]))
    _, fit, select, score = slice_roles(args.slice)
    roles = {name: tuple(rows.loc[rows.role.eq(role), WELL].itertuples(index=False, name=None))
             for name, role in (("fit", fit), ("select", select), ("score", score))}
    planned = staged.tables["fit_controls"]
    wells = tuple(planned[WELL].itertuples(index=False, name=None))
    package = args.bundle / "expression.npz"
    bundle = read_json(args.bundle / "receipt.json")
    if sha256(package) != bundle["files"][package.name] or set(wells) != set(roles["fit"]):
        raise ValueError("D3 recalled physical control arrays outside its balanced fit role")
    with np.load(package, allow_pickle=False) as arrays:
        values = checked_array(arrays["fit_controls"], (len(wells), len(staged.landmarks)), "physical fit DMSO")
    return PhysicalControls(wells, tuple(staged.landmarks.pr_gene_id), values), roles


def augment_design(fit, design, labeled, fit_anchor, control, roles):
    wells = tuple(fit.train.rows[WELL].itertuples(index=False, name=None))
    # Fit.pool residual = (treatment − original control − location) − fit-only pool offset.
    pool = fit.train.delta - fit.residual
    augmented = augment_fit(wells, tuple(fit.panel), labeled, fit_anchor + pool, fit.weight, control, roles)
    lift = augmented.row_indices
    _, project = fit.project("pca")
    covariates = project(augmented.controls).astype(np.float32)
    expanded = copy(fit)
    expanded.train = copy(fit.train)
    expanded.train.rows = fit.train.rows.iloc[lift].reset_index(drop=True)
    expanded.weight = augmented.weights.astype(np.float32)
    expanded.residual = augmented.residual.astype(np.float32)
    return expanded, Design((design.train[0][lift], covariates), design.query)


def fit_bundle(args, bank, training, query, delta, fit_anchor, control, roles, *, augment, shuffled):
    source = FitSource(bank, training, config(args.config), args.assets, bank.fit, train_delta=delta)
    fit = Fit(source, training, query, panel=bank.landmarks.pr_gene_id.to_numpy())
    design = feature_control(fit.inputs("pca"), fit, "drug_shuffle" if shuffled else None)
    pack_count = len(packs(fit.train.rows, fit.cfg["batch_rows"]))
    if augment:
        labeled = bank.response(training, bank.fit) + bank.control(training, bank.fit)
        fit, design = augment_design(fit, design, labeled, fit_anchor, control, roles)
    return fit, design, pack_count


def nn_choices(fit, design, count):
    predictions = {}
    for rate in R12_LRS:
        _, _, curve, _ = fit_network(fit, design, "film", "L0", rate, SEED, "cpu", max(R12_EPOCHS), patience=False,
                                     pack_count=count, inference_epochs=R12_EPOCHS)
        predictions.update({(rate, row["epoch"]): row["prediction"] for row in curve if "prediction" in row})
    return predictions


def shrink(estimate, offset, basis, rank, strength):
    if strength == 0.:
        return estimate
    residual = estimate - offset
    return shrink_neural_residual(residual, offset, basis.genes, basis, rank, strength)


def paired_forecasts(args, bank, control, roles):
    inner, selected, outer, held = bank.split()
    truth = bank.response(selected, bank.select)
    y, loc_in, loc_v = position(bank, inner, selected, bank.fit)
    gain = gain_choice(selected, truth, loc_v, np.zeros_like(truth))
    y_outer, loc_o, loc_test = position(bank, outer, held, bank.fit)
    anchor = gain * loc_v, gain * loc_test
    train_y = y - gain*loc_in, y_outer - gain*loc_o
    (rule, _), _ = choose_proposed(bank, inner, selected, y, truth, loc_in, loc_v, gain)
    kernel = (grid_chemical(bank, inner, selected, train_y[0], [rule])[rule],
              grid_chemical(bank, outer, held, train_y[1], [rule])[rule])
    basis = learn_nuisance_basis(control, roles, tuple(bank.landmarks.pr_gene_id))
    variants = tuple(product(basis.supported_ranks, STRENGTHS))
    predictions, choices = {}, {}
    for shuffled in (False, True):
        suffix = "_parent_shuffle" if shuffled else ""
        for augment in (False, True):
            fit, design, count = fit_bundle(args, bank, inner, selected, train_y[0], gain*loc_in, control, roles,
                                              augment=augment, shuffled=shuffled)
            predicted = nn_choices(fit, design, count)
            families = {"D3b" if augment else "D3a": predicted}
            if not augment:
                families["D3c"] = {(rate, epoch, rank, strength): shrink(x, fit.offset, basis, rank, strength)
                                   for (rate, epoch), x in predicted.items() for rank, strength in variants}
            fit_final, design_final, count_final = fit_bundle(
                args, bank, outer, held, train_y[1], gain*loc_o, control, roles, augment=augment, shuffled=shuffled)
            refits = {}
            for arm, grid in families.items():
                (params, cg, ng), objective = choose_blend(selected, truth, anchor[0], kernel[0], grid)
                rate, epoch, *reduction = params
                if (rate, epoch) not in refits:
                    best, runtime, _, _ = fit_network(fit_final, design_final, "film", "L0", rate, SEED, "cpu", epoch,
                                                       patience=False, pack_count=count_final)
                    refits[rate, epoch] = best["contrast"][2], runtime
                value, runtime = refits[rate, epoch]
                if reduction:
                    value = shrink(value, fit_final.offset, basis, *reduction)
                predictions[arm + suffix] = anchor[1] + cg * kernel[1] + ng * value
                choices[arm + suffix] = dict(parameters=params, kernel_gain=cg, neural_gain=ng,
                                             objective=objective, runtime=runtime)
    return predictions, dict(arms=choices, position_gain=gain, kernel_rule=rule,
                             fit_controls=basis.physical_wells, degrees_of_freedom=basis.contrast_df,
                             singular_values=basis.singular_values.tolist())


def code_fingerprints():
    path = Path(__file__)
    files = (path, path.with_name("dev_vehicle.py"), path.with_name("d1_inputs.py"),
             path.parent.parent / "r12_multiaxis/network.py", path.with_name("sl_components.py"))
    return {f.name: sha256(f) for f in files}


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    staged = read_bundle(args.bundle, args.pin, args.slice)
    bank = IModelBank(staged, args.slice)
    control, roles = controls_from_inputs(args, staged)
    results, selection = paired_forecasts(args, bank, control, roles)
    if set(results) != set(ARMS):
        raise ValueError("Unexpected D3 forecast registry")
    shape = len(bank.query_axis), len(bank.landmarks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".d3-fit-") as tmp:
        stage = Path(tmp)
        bank.query_axis.to_csv(stage / "rows.tsv", sep="\t", index=False)
        bank.landmarks.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecasts.npz", **{k: checked_array(v, shape, k) for k, v in results.items()})
        write_json(stage / "selection.json", selection)
        write_json(stage / "receipt.json", dict(
            slice=args.slice, protocol_sha256=sha256(args.protocol), access_sha256=sha256(args.pin),
            code_sha256=code_fingerprints(), bundle_sha256=sha256(args.bundle / "receipt.json"),
            control_sha256=sha256(args.control_reference), files={f.name: sha256(f) for f in stage.iterdir()},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", required=True)
    for name in ("bundle", "pin", "assignments", "control-reference", "config", "assets", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
