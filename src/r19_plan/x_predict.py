"""Fit matched FiLM X alternatives using only the predictor-visible cell and compound split."""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.goal_tracks.t2_upstream import SEED
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, LINE
from src.r12_multiaxis.data import Design, Fit
from src.r12_multiaxis.network import fit_network
from src.r15_lincs.design import FitSource
from src.r15_lincs.estimators import GAINS, loss
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.r18_access_plan import PLATE, keys
from src.r19_plan.r18_predict import PredictionBank, model_rows, vehicle_means
from src.r19_plan.sl_components import shuffle_rows
from src.r19_plan.x_features import CONTEXT_WIDTH, fit_basal_context, fit_cell_identity
from src.r19_plan.x_inputs import read_predict, reference, slice_axes

ARMS = ("X0", "X1", "X2", "X2_drug_off", "X2_context_off", "X2_drug_shuffle", "X2_context_shuffle")


class ContextBank(PredictionBank):
    def __init__(self, staged, key, cells):
        self.target, self.validation_cell, self.fold, self.fit, self.select, _ = slice_axes(key, cells)
        self.scenario = "A"
        self.landmarks, self.labels = staged.genes, staged.arrays["labels"]
        self.training, self.outer = (model_rows(staged.tables[name]) for name in ("labels", "query"))
        self.label_keys = keys(self.training)
        self.parents = pd.Index(sorted(set(self.training.parent) | set(self.outer.parent)))
        self.fingerprints = None
        self.role_control = {role: vehicle_means(staged.tables[name], staged.arrays[name], PLATE)
                             for name, role in (("fit_controls", self.fit), ("select_controls", self.select))}


def basal_asset(args, ref, genes):
    file = args.basal / "basal.npz"
    receipt = read_json(args.basal / "receipt.json")
    if receipt["reference_sha256"] != ref["basal_reference_sha256"] or sha256(file) != receipt["files"][file.name]:
        raise ValueError("The X2 basal asset differs from its registered provenance")
    with np.load(file, allow_pickle=False) as archive:
        cells, gene_axis, values = (archive[name] for name in ("cells", "genes", "values"))
    if list(cells) != ref["cells"] or not np.array_equal(gene_axis, genes):
        raise ValueError("X2 basal context or gene coordinates changed")
    return cells, gene_axis, checked_array(values, (len(cells), len(genes)), "X2 untreated basal RNA")


def make_design(fit, profiles, arm):
    train, query = fit.train.rows, fit.query.rows
    fitting, held = train[LINE], query[LINE]
    if set(fitting) & set(held) or set(train[GROUP]) & set(query[GROUP]):
        raise ValueError("X context features require held cell lines and canonical compounds")
    drugs = tuple(fit.drug_blocks)
    aux = {}
    if arm == "X0":
        d = fit.inputs("pca")
        # The same width is used for all arms, including a future rank-deficient physical DMSO design.
        context = tuple(np.pad(values[1], ((0, 0), (0, CONTEXT_WIDTH - values[1].shape[1])))
                        for values in (d.train, d.query))
    elif arm == "X1":
        model = fit_cell_identity(fitting)
        context = model.transform(fitting), model.transform(held)
    elif arm in ARMS[2:]:
        cells, genes, values = profiles
        basal = fit_basal_context(fitting, cells, genes, values)
        context_by_cell = pd.Index(cells)
        requested = [context_by_cell.get_indexer(rows) for rows in (fitting, held)]
        if any(np.any(positions < 0) for positions in requested):
            raise ValueError("X2 input lacks a registered held or fitting basal profile")
        if arm == "X2_context_shuffle":
            # Rotate the ordered training contexts by a reproducible nonzero shift. Held basal is unmodified.
            fit_cells = np.sort(fitting.unique())
            shift = np.random.default_rng(1501).integers(1, len(fit_cells))
            mapping = dict(zip(fit_cells, np.roll(fit_cells, shift), strict=True))
            requested[0] = context_by_cell.get_indexer(fitting.map(mapping))
        context = tuple(basal.transform(genes, values[pos]) for pos in requested)
        aux["basal_rank"] = int(basal.basis.n_components_)
        if arm == "X2_drug_off":
            drugs = tuple(np.zeros_like(x) for x in drugs)
        elif arm == "X2_drug_shuffle":
            drugs = shuffle_rows(drugs[0], train, [GROUP], 0), drugs[1]
        elif arm == "X2_context_off":
            context = tuple(np.zeros_like(x) for x in context)
    else:
        raise ValueError("Unknown X forecasting arm")
    return Design((drugs[0], context[0]), (drugs[1], context[1])), aux


def fit_slice(args, bank, profiles):
    inner, select, outer, held = bank.split()
    opts = config(args.config)
    genes = bank.landmarks.pr_gene_id.to_numpy()
    setups = ((inner, select), (outer, held))
    fits = [Fit(FitSource(bank, rows, opts, args.assets, bank.fit), rows, query, panel=genes)
            for rows, query in setups]
    if any(fit.query.treated is not None for fit in fits):
        raise ValueError("A treatment query unexpectedly entered X model training")
    validation = bank.response(select, bank.select)
    forecasts, selections = {}, {}
    for arm in ARMS:
        val, inner_meta = make_design(fits[0], profiles, arm)
        pred, outer_meta = make_design(fits[1], profiles, arm)
        checkpoints, runtimes = {}, {}
        for rate in R12_LRS:
            _, summary, checkpoints_at_rate, _ = fit_network(
                fits[0], val, "film", "L0", rate, SEED, "cpu", max(R12_EPOCHS), patience=False,
                inference_epochs=R12_EPOCHS,
            )
            checkpoints.update({(rate, row["epoch"]): row["prediction"] for row in checkpoints_at_rate
                                if "prediction" in row})
            runtimes[str(rate)] = summary
        scores = {(p, gain): loss(select, validation, gain * estimate)
                  for (p, estimate), gain in product(checkpoints.items(), GAINS)}
        if not np.isfinite(list(scores.values())).all():
            raise ValueError(f"X select loss is nonfinite for {arm}")
        params, gain = min(scores, key=lambda key: (round(scores[key], 9), key[1], str(key[0])))
        rate, epoch = params
        val, runtime, _, _ = fit_network(fits[1], pred, "film", "L0", rate, SEED, "cpu", epoch, patience=False)
        forecasts[arm] = checked_array(gain * val["contrast"][2], (len(held), len(genes)), arm)
        selections[arm] = dict(checkpoint=params, gain=gain, independent_loss=scores[params, gain],
                               inner_runtimes=runtimes, outer_runtime=runtime, inner=inner_meta, outer=outer_meta)
    return forecasts, selections


def code_fingerprints():
    module = Path(__file__)
    names = [module, module.with_name("x_features.py"), module.with_name("x_inputs.py"),
             module.parent.parent / "r12_multiaxis/network.py"]
    return {name.name: sha256(name) for name in names}


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    ref = reference(args.pin)
    staged = read_predict(args.bundle, args.pin, args.slice)
    bank = ContextBank(staged, args.slice, ref["cells"])
    profiles = basal_asset(args, ref, staged.genes.pr_gene_id.to_numpy())
    forecasts, selection = fit_slice(args, bank, profiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x-forecast-") as tmp:
        stage = Path(tmp)
        staged.tables["query"].to_csv(stage / "rows.tsv", sep="\t", index=False)
        bank.landmarks.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecasts.npz", **forecasts)
        write_json(stage / "selection.json", selection)
        write_json(stage / "receipt.json", dict(
            slice=args.slice, protocol_sha256=sha256(args.protocol), access_sha256=sha256(args.pin),
            code_sha256=code_fingerprints(), bundle_sha256=staged.receipt, config_sha256=sha256(args.config),
            basal_sha256=sha256(args.basal / "receipt.json"), files={file.name: sha256(file) for file in stage.iterdir()},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", required=True)
    for name in ("bundle", "pin", "basal", "protocol", "assets", "config", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
