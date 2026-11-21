"""Select matched S/L heads around one structure kernel and position anchor."""

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP
from src.r12_multiaxis.data import Fit
from src.r15_lincs.design import FitSource
from src.r15_lincs.estimators import GAINS, loss, parent_average
from src.r15_lincs.forecast import choose_proposed, grid_chemical, position
from src.r16_lincs.predict import gain_choice
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_reference import choose_blend
from src.r19_plan.i_run import IModelBank
from src.r19_plan.i_stage import SLICES, read_prediction_bundle
from src.r19_plan.sl_components import HUBER_SCALES, RANKS, WEIGHTS, exact_pools, fit_variant
from src.r19_plan.t4_support import measured_dose

SETTINGS = {"S0": (None,), "S1": (None,), "S2": (None,), "S3": RANKS,
            "L1": WEIGHTS, "L2": WEIGHTS, "L3": HUBER_SCALES}
CONTROLS = ("S0_drug_shuffle", "S0_context_constant", "S0_context_shuffle", "B_context", "B_mean", "B_resmean",
            "B0", "B_pos", "B_R15")
FULL_ARMS = {*SETTINGS, "B_R17", "L0", *CONTROLS}


@dataclass(frozen=True)
class PairedResult:
    full: np.ndarray
    neural_only: np.ndarray
    selection: dict


def sl_fit(args, bank, fit_rows, query_rows, residual, measured):
    genes = bank.landmarks.pr_gene_id.to_numpy()
    source = FitSource(bank, fit_rows, config(args.config), args.assets, bank.fit, train_delta=residual)
    fit = Fit(source, fit_rows, query_rows, panel=genes)
    return fit, exact_pools(fit.train.rows, measured)


def choose_variant(args, strategy, inner, outer, rows, truth, anchors, kernels, *, control=None):
    training, train_pools = inner
    final, final_pools = outer
    candidates, validation_work = {}, []
    settings = SETTINGS[strategy]
    if strategy == "S3":
        available = min(fit.train.rows[GROUP].nunique() - 1 for fit in (training, final))
        settings = tuple(rank for rank in settings if rank <= min(available, len(training.panel)))
        if not settings:
            raise ValueError("Neither S3 fit partition supports a registered output rank")
    for value, lr in product(settings, R12_LRS):
        result, run = fit_variant(training, strategy, value, lr, R12_EPOCHS, train_pools,
                                  device=args.device, control=control)
        validation_work.append(dict(value=value, **run))
        candidates.update({(value, lr, step): predicted for step, predicted in result.items()})
    (params, kernel_gain, neuro_gain), objective = choose_blend(rows, truth, anchors[0], kernels[0], candidates)
    value, lr, epoch = params
    chosen, run = fit_variant(final, strategy, value, lr, (epoch,), final_pools,
                              device=args.device, control=control)
    predicted = neuro_gain * checked_array(chosen[epoch], anchors[1].shape, strategy)
    return PairedResult(anchors[1] + kernel_gain * kernels[1] + predicted, anchors[1] + predicted,
                        dict(checkpoint=params, kernel_gain=kernel_gain, neural_gain=neuro_gain,
                             objective=objective, feature_control=control,
                             inner_runtime=validation_work, final_runtime=run))


def fit_mean(rows, response, query_truth, query_rows, anchor):
    center = parent_average(rows, response).mean(axis=0).astype(np.float32)
    candidates = {gain: loss(query_rows, query_truth, anchor + gain * center) for gain in GAINS}
    gain = min(candidates, key=lambda key: (round(candidates[key], 9), key))
    return gain, candidates[gain]


def fit_paired_arms(args, bank, measured):
    """Use only independently selected fit/select effects to assemble S/L predictions."""
    if bank.scenario != "A":
        raise ValueError("The S/L A-stage has no query drug response donor")
    inner, select, train, held = bank.split()
    if set(inner.parent) & set(select.parent) or set(train.parent) & set(held.parent):
        raise ValueError("S/L requires disjoint fitting and query canonical parents")
    truth = bank.response(select, bank.select)
    y, fit_pos, select_pos = position(bank, inner, select, bank.fit)
    gain = gain_choice(select, truth, select_pos, np.zeros_like(truth))
    raw, train_pos, held_pos = position(bank, train, held, bank.fit)
    anchors = gain * select_pos, gain * held_pos
    inner_y, outer_y = y - gain * fit_pos, raw - gain * train_pos
    (rule, kernel_gain), kernel_loss = choose_proposed(bank, inner, select, y, truth, fit_pos, select_pos, gain)
    kernels = (grid_chemical(bank, inner, select, inner_y, [rule])[rule],
               grid_chemical(bank, train, held, outer_y, [rule])[rule])
    inner_fit = sl_fit(args, bank, inner, select, inner_y, measured)
    outer_fit = sl_fit(args, bank, train, held, outer_y, measured)
    arms = {strategy: choose_variant(args, strategy, inner_fit, outer_fit, select, truth, anchors, kernels)
            for strategy in SETTINGS}
    arms["L0"] = arms["B_R17"] = arms["S0"]
    # Residual feature controls share the same structural kernel. The context-only baseline drops the kernel.
    for control in ("drug_shuffle", "context_constant", "context_shuffle"):
        arms[f"S0_{control}"] = choose_variant(args, "S0", inner_fit, outer_fit, select, truth, anchors, kernels,
                                               control=control)
    arms["B_context"] = choose_variant(args, "S0", inner_fit, outer_fit, select, truth, anchors,
                                      tuple(np.zeros_like(k) for k in kernels), control="drug_constant")
    zero = np.zeros_like(anchors[1])
    arms["B0"] = PairedResult(zero, zero, {})
    arms["B_pos"] = PairedResult(anchors[1], anchors[1], dict(gain=gain))
    chem = anchors[1] + kernel_gain * kernels[1]
    arms["B_R15"] = PairedResult(chem, chem, dict(rule=rule, gain=kernel_gain, objective=kernel_loss))
    baseline_shape = anchors[1].shape
    for name, base, fitted, bases in (("B_mean", y, raw, tuple(np.zeros_like(p) for p in anchors)),
                                     ("B_resmean", inner_y, outer_y, anchors)):
        choice, objective = fit_mean(inner, base, truth, select, bases[0])
        center = parent_average(train, fitted).mean(axis=0).astype(np.float32)
        pred = checked_array(bases[1] + choice * np.broadcast_to(center, baseline_shape), baseline_shape, name)
        arms[name] = PairedResult(pred, pred, dict(gain=choice, objective=objective))
    return arms, dict(position_gain=gain, kernel_rule=rule, fit_role=bank.fit, select_role=bank.select)


def dose_map(path, receipt, instances):
    lineage = read_json(receipt)
    if not lineage["metadata_only"] or lineage["expression_read"] or sha256(path) != lineage["raw_instance_sha256"]:
        raise ValueError("S/L measured-dose source differs from the L1 metadata audit")
    frame = measured_dose(path, set(instances))
    if set(frame.inst_id) != set(instances) or frame.actual_dose_uM.eq("").any():
        raise ValueError("S/L needs the registered physical instance axis with exact measured dose")
    return frame.set_index("inst_id", verify_integrity=True).actual_dose_uM


def predictor_code():
    module = Path(__file__)
    files = (module, module.with_name("sl_components.py"), module.parent.parent / "r12_multiaxis/network.py")
    return {file.name: sha256(file) for file in files}


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    approved = read_prediction_bundle(args.bundle, args.split_reference, args.slice)
    bank = IModelBank(approved, args.slice)
    measured = dose_map(args.instances, args.dose_reference, bank.training.inst_id)
    arms, structure = fit_paired_arms(args, bank, measured)
    if set(arms) != FULL_ARMS:
        raise ValueError("S/L arm inventory changed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shape = len(bank.query_axis), len(bank.landmarks)
    with TemporaryDirectory(dir=args.out.parent, prefix=".sl-predict-") as temp:
        stage = Path(temp)
        bank.query_axis.to_csv(stage / "rows.tsv", sep="\t", index=False)
        bank.landmarks.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "full.npz", **{name: checked_array(a.full, shape, name) for name, a in arms.items()})
        np.savez_compressed(stage / "neural_only.npz", **{
            name: checked_array(arms[name].neural_only, shape, name) for name in SETTINGS})
        write_json(stage / "selection.json", dict(structure, arms={name: a.selection for name, a in arms.items()}))
        write_json(stage / "receipt.json", dict(
            slice=args.slice, protocol_sha256=sha256(args.protocol), configuration_sha256=sha256(args.config),
            code_sha256=predictor_code(), dose_reference_sha256=sha256(args.dose_reference),
            inputs=approved.input_receipts, files={file.name: sha256(file) for file in sorted(stage.iterdir())},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", choices=sorted(SLICES), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for name in ("bundle", "split-reference", "config", "assets", "instances", "dose-reference", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
