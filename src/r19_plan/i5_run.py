"""Re-evaluate the pinned phenotype kernel and prepare I5 FiLM predictions."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r15_lincs.design import chemical_features
from src.r15_lincs.estimators import GAINS, _tanimoto, parent_average
from src.r15_lincs.forecast import fit_known, position
from src.r16_lincs.predict import HEADS
from src.r19_plan import i5_neural
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_freeze import verify_fold
from src.r19_plan.i_kernel import masked_gram
from src.r19_plan.i_kernel_arms import KernelArm, fit_optional_profiles
from src.r19_plan.i_kernel_audit import phenotype_matrix
from src.r19_plan.i_predict import ALPHAS, predict_residual, scaled_kernel
from src.r19_plan.i_run import IModelBank
from src.r19_plan.i_stage import read_prediction_bundle
from src.r19_plan.r18_preflight import checked_tables

SOURCE_KEY = "I3_all_without_I1"
SOURCE_HASH = "605f86ec99f862987f72db05fe92455809d0d02be2c32ed80bee2062deb94bf7"
MODE_ARMS = {
    "full": "I5_neural",
    "mask_only": "I5_neural_mask_only",
    "parent_shuffle": "I5_neural_parent_shuffle",
}
VIEWS = i5_neural.VIEWS


def source_slice(root, protocol, slice_id, rows, genes):
    freeze = root / "freeze.json"
    if sha256(freeze) != SOURCE_HASH:
        raise ValueError("I5 source is not the registered I-track freeze")
    path = root / "forecasts" / slice_id
    digest = read_json(freeze)["folds"][slice_id]
    stamp = verify_fold(path, digest, rows, genes, protocol)
    if stamp["views"][SOURCE_KEY] != list(VIEWS):
        raise ValueError("The archived I5 kernel is not the two-phenotype I1 exclusion")
    shape = len(rows), len(genes)
    with np.load(path / "forecasts.npz", allow_pickle=False) as data:
        forecasts = {name: checked_array(data[key], shape, name) for name, key in
                     (("B_R17", "B_R17"), ("I0", "I0"), ("I5_kernel", SOURCE_KEY))}
    with np.load(path / "coverage.npz", allow_pickle=False) as data:
        masks = np.column_stack([data[name] == 1 for name in VIEWS])
        if not np.array_equal(masks.sum(axis=1), data[SOURCE_KEY]):
            raise ValueError("The archived I5 kernel has a different availability axis")
    absent = ~masks.any(axis=1)
    if not np.array_equal(forecasts["I0"][absent], forecasts["I5_kernel"][absent]):
        raise ValueError("The archived I5 kernel changed the structure fallback")
    return path, stamp, digest, forecasts, masks


def kernel_components(parents, structure, profiles, train_rows, query_rows, residual, settings):
    """Rebuild the pinned I0 and two-view residuals without constructing I1 inputs."""
    fitting = parents.isin(train_rows.parent)
    query = parents.isin(query_rows.parent)
    if not fitting.any() or not query.any() or (fitting & query).any():
        raise ValueError("I5 kernel rows cross the canonical parent partitions")
    train_idx, query_idx = np.flatnonzero(fitting), np.flatnonzero(query)
    active = np.r_[train_idx, query_idx]
    structural = masked_gram(structure[np.ix_(active, active)], [])
    maps = fit_optional_profiles(profiles, fitting)
    blocks = [(maps[name][0][active], maps[name][1][active]) for name in VIEWS]
    masks = np.column_stack([present[len(train_idx):] for _, present in blocks])
    counts = masks.sum(axis=1)
    structure_arm = KernelArm("I0", (), structural, train_idx, query_idx, np.zeros_like(counts))
    two_view_arm = KernelArm(SOURCE_KEY, VIEWS, masked_gram(structural, blocks), train_idx, query_idx, counts)
    by_parent = parent_average(train_rows, residual)
    alpha, gain = settings["I0"]
    base = gain * predict_residual(structure_arm, by_parent, alpha)
    alpha, gain = settings[SOURCE_KEY]
    prediction = scaled_kernel(two_view_arm, predict_residual(two_view_arm, by_parent, alpha), gain, base)
    expand = parents[query_idx].get_indexer(query_rows.parent)
    if (expand < 0).any() or len(by_parent) != len(train_idx):
        raise ValueError("I5 parent aggregate or expansion differs from the fit axis")
    return prediction[expand], base[expand], masks[expand]


def approved_profiles(args, parents, source_inputs):
    profiles = {}
    for name in VIEWS:
        reference = args.view_references / f"{name}_receipt.json"
        if sha256(reference) != source_inputs[name]:
            raise ValueError(f"The registered I5 view lineage changed: {name}")
        checked_tables(args.views / name, reference)
        profiles[name] = phenotype_matrix(args.views / name / "features.npz", parents)
    return profiles


def source_kernel_settings(path):
    rows = table(path / "selection.tsv").set_index("arm", verify_integrity=True)
    settings = {name: tuple(map(float, rows.loc[name, ["alpha", "gain"]])) for name in ("I0", SOURCE_KEY)}
    if any(alpha not in ALPHAS or gain not in GAINS for alpha, gain in settings.values()):
        raise ValueError("An archived I5 kernel is outside the registered I-track grid")
    return settings


def run(args):
    out = args.out / args.slice
    if out.exists():
        raise FileExistsError(out)
    staged = read_prediction_bundle(args.bundle, args.split_reference, args.slice)
    bank = IModelBank(staged, args.slice)
    args.fold, args.role = bank.fold, bank.fit
    path, receipt, digest, old, masks = source_slice(args.source, args.source_protocol, args.slice,
                                                     bank.query_axis, bank.landmarks)
    inputs = staged.input_receipts | {"config": sha256(args.config)}
    if any(receipt["inputs"][name] != value for name, value in inputs.items()):
        raise ValueError("I5 predictor inputs differ from the prior frozen axes or configuration")
    profiles = approved_profiles(args, bank.parents, receipt["inputs"])
    fp = chemical_features(bank.parents)
    structure = _tanimoto(fp, fp)
    settings = source_kernel_settings(path)
    ref = receipt["reference_selection"]
    if (ref["fit_role"], ref["select_role"]) != (bank.fit, bank.select):
        raise ValueError("I5 did not inherit the registered independent selection role")
    position_gain = float(ref["position_gain"])
    inner, select, train, held = bank.split()
    inner_raw, inner_pos, select_pos = position(bank, inner, select, bank.fit)
    inner_y = inner_raw - position_gain * inner_pos
    select_anchor = position_gain * select_pos
    inner_k, inner_base, inner_masks = kernel_components(bank.parents, structure, profiles, inner, select,
                                                         inner_y, settings)
    inner_fallback = select_anchor + inner_base
    if (~inner_masks.any(axis=1)).any() and ref["neural_gain"]:
        with TemporaryDirectory(prefix="i5-structural-fallback-") as tmp:
            film = fit_known(args, bank, inner, select, inner_y, None, select_anchor,
                             HEADS["blend_FiLM"], Path(tmp), tuple(ref["checkpoint"]))
        inner_fallback += ref["neural_gain"] * film
    train_raw, train_pos, held_pos = position(bank, train, held, bank.fit)
    train_y = train_raw - position_gain * train_pos
    held_kernel, _, held_masks = kernel_components(bank.parents, structure, profiles, train, held,
                                                   train_y, settings)
    with np.load(path / "kernel_only.npz", allow_pickle=False) as data:
        fixed = checked_array(data[SOURCE_KEY], old["I5_kernel"].shape, "pinned position plus kernel")
    reconstructed = position_gain * held_pos + held_kernel
    max_error = float(np.max(np.abs(reconstructed - fixed)))
    if max_error > 1e-6 or not np.array_equal(masks, held_masks):
        raise ValueError(f"I5 did not reproduce the archived outer kernel: max error {max_error}")
    inner_design = i5_neural.prepare_neural(args, bank, inner, select, inner_y, bank.parents, profiles)
    held_design = i5_neural.prepare_neural(args, bank, train, held, train_y, bank.parents, profiles)
    if (not np.array_equal(inner_design.query_presence, inner_masks)
            or not np.array_equal(held_design.query_presence, masks)):
        raise ValueError("I5 neural and kernel feature support disagree")
    truth = bank.response(select, bank.select)
    choices, predictions = {}, {}
    for mode, arm in MODE_ARMS.items():
        choice, _ = i5_neural.select_neural(inner_design, truth, select_anchor + inner_k, inner_fallback, mode)
        choices[arm] = vars(choice)
        predictions[arm] = i5_neural.predict_neural(held_design, fixed, old["I5_kernel"], mode, choice)

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".i5-predict-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage / "predictions.npz", **predictions)
        np.savez_compressed(stage / "coverage.npz", **dict(zip(VIEWS, masks.T, strict=True)))
        write_json(stage / "receipt.json", dict(
            slice=args.slice, protocol_sha256=sha256(args.protocol), source_slice_sha256=digest,
            predictor_bundle_sha256=sha256(args.bundle / "receipt.json"), reconstruction_max_abs=max_error,
            choices=choices, inputs=inputs | {name: receipt["inputs"][name] for name in VIEWS},
            files={file.name: sha256(file) for file in sorted(stage.iterdir())},
        ))
        stage.rename(out)
    return sha256(out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", required=True)
    for name in ("bundle", "split-reference", "views", "view-references", "config", "assets", "protocol",
                 "source", "source-protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
