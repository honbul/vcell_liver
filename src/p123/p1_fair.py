"""Give each frozen T2 comparator the same fit-only Tahoe position information."""

import argparse
from copy import copy
from dataclasses import replace
import itertools
from pathlib import Path
import tempfile

import numpy as np

from src.goal_tracks.t1_data import MIN_CELLS, MIN_GROUPS, DoseSource, slice_name
from src.goal_tracks.t1_io import load_task
from src.goal_tracks.t2_predict import run_chemberta, r12_design
from src.goal_tracks.t2_prnet import fit_model as fit_prnet
from src.goal_tracks.t2_state import fit_model as fit_state
from src.goal_tracks.t2_upstream import EPOCHS, SEED, upstream
from src.r11_chemical_probe.common import config, read_json, sha256
from src.r11_chemical_probe.dataset import GROUP, KEY
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.models import PoolMean
from src.r12_multiaxis.data import Fit
from src.r12_multiaxis.network import fit_network
from src.r13_optimization.objectives import center

from .p1_position import MODEL_GAINS, POS_GAINS, energy_curves, load_context
from .p123_io import write_prediction


CPU_MODELS = ("ChemBERTa_R11", "Concat_R12", "FiLM_R12", "scGPT_C7", "UCE_C8")
GPU_MODELS = ("XPert_morgan", "State-Morgan-well-adaptation", "PRnet-well-adaptation",
              "chemCPA_P0", "TranSiGen_Tahoe")
SLICES = tuple(itertools.product((5.0, .5, .05), MIN_CELLS, range(3), "AB"))
PROTOCOL_SHA = "d0680c422b41c63f895b8c1efc0fdedbcbc7965fdc1e35a40e6eb7bde6da8693"


def historical(root, model, source, fit, old_protocol, reports):
    sl = slice_name(source.dose, source.minimum_cells, source.outer, source.role)
    d = f"d{source.dose:g}"
    t2 = root / "t2" / d / "freeze.json"
    if sha256(t2) != sha256(reports / "known_vc" / d / "freeze.json"):
        raise ValueError("Historical T2 freeze differs from the committed receipt")
    receipt = read_json(t2)
    path = Path(receipt["eligible_known"][model]) / sl
    if sha256(path / "receipt.json") != receipt["tasks"][sl][model]:
        raise ValueError("Historical T2 predictions differ from the freeze")
    args = (source.staged, old_protocol, source.dose, source.minimum_cells, source.outer, source.role)
    spec, rows, panel, fore = load_task(path, *args)
    if not rows[KEY].equals(fit.query.rows[KEY]) or not np.array_equal(panel, fit.panel):
        raise ValueError("Historical comparator does not match the current query")
    anchor = root / "t1/first" / d
    afree = anchor / "freeze.json"
    if sha256(afree) != receipt["t1_freeze_sha256"]:
        raise ValueError("Historical T1 freeze differs from the bound T2 receipt")
    apath = anchor / "predictions" / sl
    if sha256(apath / "receipt.json") != read_json(afree)["tasks"][sl]:
        raise ValueError("Historical T1 absolute anchor changed")
    _, _, _, base = load_task(apath, *args)
    return spec["selection"][model], fore[model], base["B_abs"], sha256(t2)


def remove_position(fit, position):
    adjusted = copy(fit)
    adjusted.train = replace(fit.train, treated=fit.train.treated - position)
    adjusted.means = PoolMean(adjusted.train.rows, adjusted.train.delta)
    shrinkage = fit.cfg["offset_shrinkage"]
    adjusted.offset, adjusted.offset_level, _ = adjusted.means.predict(fit.query.rows, shrinkage)
    baseline = adjusted.means.predict(adjusted.train.rows, shrinkage, leave_group_out=True)[0]
    adjusted.residual = (adjusted.train.delta - baseline).astype(np.float32)
    variance = np.dot(fit.weight, np.square(adjusted.residual).mean(axis=1)) / fit.weight.sum()
    adjusted.scale = max(1e-6, float(np.sqrt(variance)))
    return adjusted


def repredict(model, fit, selected, args, work):
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    root = args.t2_root
    extra = root / "t2/context-v2"
    if model == "ChemBERTa_R11":
        return run_chemberta(fit, validation=False)[0]
    if model in CPU_MODELS:
        design, _ = r12_design(fit, model, extra)
        arch = "film" if model == "FiLM_R12" else "concat"
        best, *_ = fit_network(fit, design, arch, "L0", selected["selected_lr"], SEED, "cpu",
                               selected["selected_epoch"], patience=False)
        return best["contrast"][2]
    if model in EPOCHS:
        work.mkdir()
        return upstream(model)(fit, work, (selected["selected_epoch"],), None)[0]
    if model == "PRnet-well-adaptation":
        return fit_prnet(fit, fit.source.dose, root / "t2-public/upstream-prnet")[0]
    original = fit.source.assets / "cells-v2"
    added = extra / "cells-v2"
    folders = {0.05: (original, added), 0.5: (original,), 5.0: (added,)}
    return fit_state(fit, root / "t2-public/upstream-state", folders[fit.source.dose])[0]


def choose(paths):
    loss = np.mean(paths, axis=0)
    position, vc = min(np.ndindex(loss.shape),
                       key=lambda ix: (round(float(loss[ix]), 10), ix[1], ix[0]))
    return float(POS_GAINS[position]), float(MODEL_GAINS[vc]), loss.tolist()


def predict(args, model, dose, n, fold, role):
    if sha256(args.protocol) != PROTOCOL_SHA:
        raise ValueError("Fairness job requires the preregistered specification")
    source = DoseSource(config(args.config), args.assets, args.t2_root / "t1-cache", dose, n, fold, role)
    outer = Fit(source, source.fit_rows, source.query_rows, panel=source.panel)
    selected, frozen, base, historic_sha = historical(
        args.t2_root, model, source, outer, args.old_protocol, args.old_reports)
    pos_receipt = args.positions / slice_name(dose, n, fold, role) / "receipt.json"
    context = read_json(pos_receipt)
    sets = {"fair": []}
    if model in CPU_MODELS:
        sets["fair-refit"] = []
    with tempfile.TemporaryDirectory(prefix="goal-fair-") as directory:
        temp = Path(directory)
        for inner in (0, 1):
            train, query = source.inner(inner)
            fit = Fit(source, train, query, panel=source.inner_panel(inner), observe_query=False)
            if set(fit.train.rows[GROUP]) & set(fit.query.rows[GROUP]):
                raise ValueError("An inner held parent entered training")
            own_truth = source.observed(fit.query.rows, fit.panel) - fit.query.control
            raw_fit, raw_query = load_context(args.positions, source, fit, inner,
                                               scaled=False, protocol=args.protocol)
            fit_position = context["gain"] * raw_fit
            candidates = {"fair": repredict(model, fit, selected, args, temp / f"inner{inner}")}
            if model in CPU_MODELS:
                candidates["fair-refit"] = repredict(model, remove_position(fit, fit_position),
                                                      selected, args, temp / f"inner-refit{inner}")
            for name, value in candidates.items():
                sets[name].append(energy_curves(fit.query.rows, own_truth, raw_query, value))
            print(f"calibration fold {inner} with {len(fit.train.rows)} fit and {len(fit.query.rows)} held wells×line",
                  flush=True)

        raw_fit, raw_query = load_context(args.positions, source, outer, -1,
                                           scaled=False, protocol=args.protocol)
        fit_position = context["gain"] * raw_fit
        components = {"fair": frozen}
        if model in CPU_MODELS:
            components["fair-refit"] = repredict(model, remove_position(outer, fit_position),
                                                 selected, args, temp / "outer-refit")
        metric = Contrast(outer.query.rows, MIN_GROUPS)
        forecasts, receipts = {}, {}
        for family, component in components.items():
            name = f"{family}-{model}"
            a, b, loss = choose(sets[family])
            forecasts[name] = base + a * raw_query + b * center(metric, component)
            receipts[name] = dict(
                positional_gain=a, model_gain=b, inner_sse_fraction=loss, model_settings=selected,
                historical_t2_freeze_sha256=historic_sha, position_receipt_sha256=sha256(pos_receipt),
                trainer_sha256=sha256(__file__), residual_refit=family == "fair-refit",
            )
        destination = args.out / model / slice_name(dose, n, fold, role)
        write_prediction(destination, source, outer, forecasts, receipts, args.protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--task", type=int, required=True)
    for key in ("protocol", "old-protocol", "old-reports", "positions", "config", "assets", "t2-root", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    bank = CPU_MODELS if args.lane == "cpu" else GPU_MODELS
    by_dose = [(model, *slice_) for dose in (5.0, .5, .05) for model in bank for slice_ in SLICES if slice_[0] == dose]
    model, dose, n, fold, role = by_dose[args.task]
    print(f"{model}: dose={dose}, minimum={n}, outer={fold}, role={role}", flush=True)
    predict(args, model, dose, n, fold, role)


if __name__ == "__main__":
    main()
