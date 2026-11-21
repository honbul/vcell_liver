"""Compare basal and ID contexts on the same measured-donor residual task."""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.goal_tracks.t2_upstream import SEED
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r12_multiaxis.data import Fit
from src.r12_multiaxis.network import fit_network
from src.r15_lincs.design import FitSource
from src.r15_lincs.estimators import GAINS, loss
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_score import checked_rows
from src.r19_plan.sl_b import CONFIG_SHA256, DONORS
from src.r19_plan.x3_workflow import donor_inputs, tensorize
from src.r19_plan.x_inputs import matrix_axes, read_plan, read_predict, reference, slice_axes, validate_predict
from src.r19_plan.x_predict import ContextBank, basal_asset, code_fingerprints, make_design
from src.r19_plan.x_workflow import center_by_context, comparisons, interaction_axes, score_delta


MODELS = {"X1_B": "X1", "X2_B": "X2", "X2_B_context_off": "X2_context_off", "X2_B_context_shuffle": "X2_context_shuffle"}
ARMS = {"B_cell", *MODELS}
MAIN = ("X2_B", "X1_B")
PAIRS = (MAIN, ("X2_B", "B_cell"), ("X1_B", "B_cell"),
         ("X2_B", "X2_B_context_off"), ("X2_B", "X2_B_context_shuffle"))


def matched_donors(bank, source, target):
    """Average nonhepatic cells per parent, excluding each output cell itself."""
    parents = sorted(set(target.parent))
    chosen = source.loc[source.cell_id.isin(DONORS) & source.parent.isin(parents)]
    cells = sorted(chosen.cell_id.unique())
    if not cells or set(chosen.parent) != set(parents):
        raise ValueError("A residual fit or validation parent lacks a measured B donor")
    values = tensorize(chosen, bank.response(chosen, bank.fit), parents, cells)
    take = pd.Index(parents).get_indexer(target.parent)
    result = np.empty((len(target), values.shape[-1]), dtype=np.float32)
    for cell, idx in target.groupby("cell_id").indices.items():
        include = np.asarray(cells) != cell
        if not include.any():
            raise ValueError("The B donor residual would include its own treatment label")
        result[idx] = values[:, include].mean(axis=1)[take[idx]]
    return result


def load_held_donors(args, ref, bank, held):
    directory = args.donors
    receipt = read_json(directory/"receipt.json")
    path = directory/"donors.npz"
    cells = [cell for cell in ref["cells"] if cell in DONORS and cell != bank.target]
    if (receipt["slice"] != args.slice or receipt["access_sha256"] != sha256(args.pin)
            or receipt["data_sha256"] != sha256(path) or set(receipt["upstream"]) != set(cells)):
        raise ValueError("The matched nonhepatic query donor receipt changed")
    for cell, source in receipt["upstream"].items():
        key = f"{cell}-f{bank.fold}-{bank.fit}"
        if source != {"source": key, "stages": ref["slices"][key]}:
            raise ValueError("A held source cell has a different role, parent fold, or allowlist")
    with np.load(path, allow_pickle=False) as values:
        parents = pd.Index(values["parents"])
        if (list(values["cells"]) != cells or parents.to_list() != sorted(set(held.parent))
                or not np.array_equal(values["genes"], bank.landmarks.pr_gene_id)):
            raise ValueError("The B donor parent or physical gene axis changed")
        tensor = checked_array(values["values"], (len(parents), len(cells), len(bank.landmarks)), "B donor")
    return tensor.mean(axis=1)[parents.get_indexer(held.parent)], sha256(directory/"receipt.json"), len(cells)


def fit_arm(fits, arm, profiles, validate, truth, anchors):
    inner, outer = fits
    left, meta_left = make_design(inner, profiles, arm)
    right, meta_right = make_design(outer, profiles, arm)
    candidates = {}
    for lr in R12_LRS:
        _, _, checkpoints, _ = fit_network(inner, left, "film", "L0", lr, SEED, "cpu", max(R12_EPOCHS),
                                          patience=False, inference_epochs=R12_EPOCHS)
        candidates.update({(lr, cp["epoch"]): cp["prediction"] for cp in checkpoints if "prediction" in cp})
    objectives = {(key, gain): loss(validate, truth, anchors[0]+gain * prediction)
                  for (key, prediction), gain in product(candidates.items(), GAINS)}
    if not np.isfinite(list(objectives.values())).all() or len(candidates) != len(R12_EPOCHS)*len(R12_LRS):
        raise ValueError("B-context checkpoint or selection support differs from X")
    key, gain = min(objectives, key=lambda pair: (round(objectives[pair], 9), pair[1], str(pair[0])))
    lr, ep = key
    result, _, _, _ = fit_network(outer, right, "film", "L0", lr, SEED, "cpu", ep, patience=False)
    prediction = checked_array(anchors[1]+gain * result["contrast"][2], anchors[1].shape, arm)
    return prediction, dict(checkpoint=key, gain=gain, independent_loss=objectives[key,gain], inner=meta_left, outer=meta_right)


def lineage(args):
    files = ("x_b.py", "x3_workflow.py", "x_workflow.py", "x_transfer.py")
    here = Path(__file__).parent
    cfg = sha256(args.config)
    if cfg != CONFIG_SHA256:
        raise ValueError("X.B requires the matched X/R12 FiLM configuration")
    return dict(access_sha256=sha256(args.pin), protocol_sha256=sha256(args.protocol), config_sha256=cfg,
                basal_sha256=sha256(args.basal/"receipt.json"),
                code={**code_fingerprints(), **{name: sha256(here/name) for name in files}})


def predict(args):
    target = args.forecasts/args.slice
    if target.exists():
        raise FileExistsError(target)
    stamp = lineage(args)
    ref = reference(args.pin)
    staged = read_predict(args.inputs/args.slice, args.pin, args.slice)
    bank = ContextBank(staged, args.slice, ref["cells"])
    inner, validation, outer, held = bank.split()
    profiles = basal_asset(args, ref, bank.landmarks.pr_gene_id.to_numpy())
    held_donor, donor_hash, donor_cells = load_held_donors(args, ref, bank, held)
    available = outer.loc[outer.cell_id.isin(inner.cell_id.unique())]
    val_donor = matched_donors(bank, available, validation)
    truth = bank.response(validation, bank.select)
    scores = {value: loss(validation, truth, value * val_donor) for value in GAINS}
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("B source-only donor gain selection is nonfinite")
    donor_gain = min(scores, key=lambda g: (round(scores[g], 9), g))
    anchors = tuple(donor_gain * donor for donor in (val_donor, held_donor))
    options = config(args.config)
    fits = []
    for fit, query in ((inner, validation), (outer, held)):
        donor = matched_donors(bank, fit, fit)
        y = bank.response(fit, bank.fit) - donor_gain * donor
        source = FitSource(bank, fit, options, args.assets, bank.fit, train_delta=y)
        fits.append(Fit(source, fit, query, panel=bank.landmarks.pr_gene_id.to_numpy()))
    forecasts, choices = {"B_cell": anchors[1]}, {}
    for name, design in MODELS.items():
        forecasts[name], choices[name] = fit_arm(fits, design, profiles, validation, truth, anchors)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix=".xb-predict-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage/"forecasts.npz", **forecasts)
        write_json(stage/"selection.json", dict(donor_gain=donor_gain, donor_cells=donor_cells, models=choices))
        write_json(stage/"receipt.json", dict(stamp, slice=args.slice, donor_sha256=donor_hash,
            bundle_sha256=staged.receipt, query_sha256=ref["slices"][args.slice]["predict"]["query.tsv"],
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(target)


def donor_dir(args, target, key):
    return (args.hepg2_donors if target == "HEPG2" else args.new_donors)/key


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("Independent B scoring requires the frozen hash and fresh output")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    ref, stamp = reference(args.pin), lineage(args)
    cells = ref["cells"]
    genes, columns = matrix_axes(args.access, ref)
    queries, receipts, choices, predictions = {}, {}, [], {arm: [] for arm in sorted(ARMS)}
    for key in sorted(ref["slices"]):
        _, _, tables = read_plan(args.access, args.pin, key, "predict")
        validate_predict(tables, key, cells)
        query = queries[key] = tables["query"]
        folder = args.forecasts/key
        rec = read_json(folder/"receipt.json")
        held_cell = slice_axes(key, cells)[0]
        if (any(rec[k] != v for k, v in stamp.items()) or rec["slice"] != key
                or rec["query_sha256"] != ref["slices"][key]["predict"]["query.tsv"]
                or rec["bundle_sha256"] != sha256(args.inputs/key/"receipt.json")
                or rec["donor_sha256"] != sha256(donor_dir(args, held_cell, key)/"receipt.json")
                or set(rec["files"]) != {"forecasts.npz", "selection.json"}
                or any(sha256(folder/name) != value for name, value in rec["files"].items())):
            raise ValueError(f"The matched B forecast changed: {key}")
        with np.load(folder/"forecasts.npz", allow_pickle=False) as file:
            if set(file.files) != ARMS:
                raise ValueError("The B context forecast inventory changed")
            for name in predictions:
                values = checked_array(file[name], (len(query), len(genes)), name)
                if scoring:
                    predictions[name].append(values)
        receipts[key] = sha256(folder/"receipt.json")
        choice = read_json(folder/"selection.json")
        for name, settings in choice["models"].items():
            choices.append(dict(slice=key, arm=name, lr=settings["checkpoint"][0], epochs=settings["checkpoint"][1],
                                gain=settings["gain"], select_loss=settings["independent_loss"],
                                donor_gain=choice["donor_gain"], donor_cells=choice["donor_cells"]))
    rows = checked_rows(pd.concat([query.assign(fold=slice_axes(key, cells)[2], role=slice_axes(key, cells)[3])
                                   for key, query in queries.items()], ignore_index=True), None)
    if rows.parent.nunique() != 493 or rows.inst_id.nunique() != 14898 or len(receipts) != 45:
        raise ValueError("The matched B physical support differs from the frozen X population")
    axes, groups, counts, idx = interaction_axes(rows, cells)
    frozen = dict(stamp, slices=receipts)
    if not scoring:
        write_json(args.freeze, frozen)
        return
    if read_json(args.freeze) != frozen:
        raise ValueError("The B predictions or score code changed after freezing")
    truth = np.concatenate([score_delta(args, key, query, genes, columns) for key, query in queries.items()])
    values = {name: np.concatenate(parts) for name, parts in predictions.items()}
    del predictions
    paired, sensitivity = comparisons(rows, truth, values, cells, centered=False, pairs=PAIRS, primary_pair=MAIN)
    def centered(x):
        return center_by_context(x, groups, counts, idx, len(cells))
    interaction, _ = comparisons(axes, centered(truth), {arm: centered(v) for arm, v in values.items()},
                                 cells, centered=True, pairs=PAIRS, primary_pair=MAIN)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".xb-score-") as tmp:
        stage = Path(tmp)
        for name, data in (("paired", paired), ("interaction", interaction), ("sensitivity", sensitivity), ("selection", choices)):
            pd.DataFrame(data).to_csv(stage/(name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(frozen, freeze_sha256=args.expected_sha256,
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("donors", "predict", "freeze", "score"))
    for name in ("access", "pin", "inputs", "assets", "config", "basal", "protocol", "new-donors", "hepg2-donors",
                 "forecasts", "out", "freeze"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--slice")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.action == "donors":
        print(donor_inputs(args, all_registered_targets=True))
    elif args.action == "predict":
        args.donors = donor_dir(args, slice_axes(args.slice, reference(args.pin)["cells"])[0], args.slice)
        predict(args)
    else:
        evaluate(args)
