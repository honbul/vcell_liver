"""Refit the known Tahoe VC comparators while outer response files remain sealed."""

import argparse
from functools import lru_cache
import itertools
from pathlib import Path
import tempfile
import time

import numpy as np

from src.r11_chemical_probe.common import config, read_json, sha256
from src.r11_chemical_probe.dataset import GROUP
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.models import Interaction
from src.r12_multiaxis.candidates import Candidate, Evaluator
from src.r12_multiaxis.data import Design, Fit
from src.r12_multiaxis.network import fit_network

from .t1_data import DOSES, MIN_CELLS, DoseSource, slice_name, validate_protocol
from .t1_io import write_task
from .t2_upstream import EPOCHS, SEED, upstream


SLICES = tuple(itertools.product((5., .5, .05), MIN_CELLS, range(3), ("A", "B")))
MODELS = ("XPert_morgan", "chemCPA_P0", "TranSiGen_Tahoe", "ChemBERTa_R11",
          "FiLM_R12", "Concat_R12", "scGPT_C7", "UCE_C8")
R12_EPOCHS = (1, 5, 10, 20, 40)
R12_LRS = (0.0003, 0.001)


@lru_cache(maxsize=8)
def read_embeddings(root, name, role):
    folder = root / name
    path = folder / f"{role}.npz"
    receipt = read_json(folder / "receipt.json")
    digest = sha256(path)
    if receipt["output_sha256"][path.name] != digest:
        raise ValueError(f"Changed frozen control contexts: {path}")
    with np.load(path, allow_pickle=False) as data:
        keys = list(zip(data["well"], data["line"]))
        bank = dict(zip(keys, data["embedding"]))
        if len(bank) != len(keys):
            raise ValueError("Duplicate physical DMSO embedding")
    return bank, digest


def r12_design(fit, name, extra):
    if name in {"FiLM_R12", "Concat_R12"}:
        return fit.inputs("pca"), {}
    context = {"scGPT_C7": "scgpt", "UCE_C8": "uce"}[name]
    bank, main_sha = read_embeddings(fit.source.assets, context, fit.source.role)
    bank = dict(bank)
    all_keys = set(fit.train_units) | set(fit.query_units)
    receipt = {"R12_context_sha256": main_sha}
    if all_keys.difference(bank):
        saved, more_sha = read_embeddings(extra, context, fit.source.role)
        if bank.keys() & saved.keys():
            raise ValueError("Extra cache overlaps R12 physical wells")
        bank.update(saved)
        receipt["additional_dose_context_sha256"] = more_sha
    if missing := all_keys.difference(bank):
        raise ValueError(f"{name} has no condition vector for {len(missing)} DMSO units: {sorted(missing)[:5]}")
    unique = np.stack([bank[k] for k in fit.fit_units])
    pca, scale = Interaction._fit(unique, 16)

    def project(keys):
        return scale.transform(pca.transform(np.stack([bank[k] for k in keys]))).astype(np.float32)

    return Design((fit.drug_blocks[0], project(fit.train_units)),
                  (fit.drug_blocks[1], project(fit.query_units))), receipt


def run_r12(name, fit, lr, epoch, extra, validation):
    design, context_receipt = r12_design(fit, name, extra)
    head = "film" if name == "FiLM_R12" else "concat"
    if validation:
        # Collect one common validation curve per learning rate without triggering R12's early stop.
        fit.cfg = dict(fit.cfg, min_epochs=epoch + 1)
    best, runtime, curve, _ = fit_network(fit, design, head, "L0", lr, SEED, "cuda", epoch, patience=validation)
    if validation:
        return {r["epoch"]: r["contrast"] for r in curve if r["epoch"] in R12_EPOCHS}
    return best["contrast"][2], dict(runtime, **context_receipt,
        architecture="R12 A5 FiLM / A4 concat / C7 or C8 matched context head",
        input="PCA32 ECFP + PCA16 control context, R12 leave-parent-out population offset and residual MSE")


def run_chemberta(fit, validation):
    candidate = Candidate("ChemBERTa_R11", "ridge", drug="C", value=1000.0)
    pred, *_ = Evaluator(fit, "cpu").evaluate(candidate)
    out = pred["contrast"][2]
    if validation:
        return {1000.0: Contrast(fit.query.rows, 3).point(out, fit.query.delta)}
    feature_receipt = fit.source.r11 / "features/receipt.json"
    return out, dict(architecture="R11 frozen CLS PCA32 × control PCA16 interaction ridge",
                     alpha=1000.0, asset_receipt_sha256=sha256(feature_receipt))


def inner_fits(source):
    for inner in (0, 1):
        train, validation = source.inner(inner)
        fit = Fit(source, train, validation, observe_query=True, panel=source.inner_panel(inner))
        if set(fit.train.rows[GROUP]) & set(fit.query.rows[GROUP]):
            raise ValueError("Validation chemistry crossed the fit boundary")
        yield fit


def fit_model(name, source, extra, scratch):
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    started = time.monotonic()
    held = Fit(source, source.fit_rows, source.query_rows, panel=source.panel)
    if held.query.treated is not None:
        raise ValueError("Outer target became visible during prediction")
    if name == "ChemBERTa_R11":
        pred, final = run_chemberta(held, validation=False)
        return held, pred, dict(final=final, seed=SEED, selection="R11 fixed alpha 1000",
                                seconds=time.monotonic() - started)
    trials = []
    for k, fit in enumerate(inner_fits(source)):
        if name in EPOCHS:
            metric = Contrast(fit.query.rows, 3)
            work = scratch / f"inner{k}"
            work.mkdir()

            def evaluate(prediction):
                return (metric.point(prediction, fit.query.delta) if np.isfinite(prediction).all()
                        else float("inf"))

            torch.manual_seed(SEED)
            values, _ = upstream(name)(fit, work, EPOCHS[name], evaluate)
            trials.append({(0.0, e): v for e, v in values.items()})
        else:
            grid = {}
            for lr in R12_LRS:
                values = run_r12(name, fit, lr, max(R12_EPOCHS), extra, validation=True)
                grid.update({(lr, e): v for e, v in values.items()})
            trials.append(grid)
    scores = {key: float(np.mean([trial[key] for trial in trials])) for key in trials[0]}
    if not any(np.isfinite(v) for v in scores.values()):
        raise ValueError(f"No finite inner score for {name}")
    lr, epoch = min(scores, key=lambda x: (scores[x], x[1], x[0]))
    if name in EPOCHS:
        work = scratch / "outer"
        work.mkdir()
        torch.manual_seed(SEED)
        pred, final = upstream(name)(held, work, (epoch,), None)
    else:
        pred, final = run_r12(name, held, lr, epoch, extra, validation=False)
    receipt = dict(final=final, seed=SEED, selected_epoch=epoch, selected_lr=(lr or final.get("lr")),
                   inner_scores=[dict(lr=a, epoch=b, mean=(score if np.isfinite(score) else None))
                                 for (a, b), score in scores.items()],
                   seconds=time.monotonic() - started)
    return held, pred, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", type=int, choices=range(len(SLICES)), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--extra-context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate_protocol(args.protocol)
    dose, minimum, fold, role = SLICES[args.task]
    if set(DOSES) != {s[0] for s in SLICES}:
        raise ValueError("Tahoe dose contract changed")
    source = DoseSource(config(args.config), args.assets, args.stage, dose, minimum, fold, role)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"fit {args.model} dose={dose} minimum={minimum} fold={fold} role={role}", flush=True)
    with tempfile.TemporaryDirectory(prefix="goal-t2-") as temporary:
        fit, forecast, selection = fit_model(args.model, source, args.extra_context, Path(temporary))
    write_task(args.out / slice_name(dose, minimum, fold, role), source, fit,
               {args.model: forecast}, {args.model: selection})
    print(f"forecast shape {forecast.shape} recorded in {args.out}", flush=True)


if __name__ == "__main__":
    main()
