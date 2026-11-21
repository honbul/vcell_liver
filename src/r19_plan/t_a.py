"""Cross-fit shift calibrators for a held-context, held-compound FiLM predictor."""

import argparse
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t2_upstream import SEED
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r12_multiaxis.data import Fit
from src.r12_multiaxis.network import fit_network
from src.r15_lincs.design import FitSource
from src.r15_lincs.estimators import loss
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_score import checked_rows
from src.r19_plan.r18_access_plan import keys
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.sl_b import CONFIG_SHA256
from src.r19_plan.t123_workflow import SETTINGS, gains, responses, rule_factors
from src.r19_plan.t_a_scope import select_controls, source_folds
from src.r19_plan.x_inputs import matrix_axes, read_plan, read_predict, reference, slice_axes
from src.r19_plan.x_predict import ContextBank, code_fingerprints, make_design
from src.r19_plan.x_workflow import center_by_context, comparisons, interaction_axes, score_delta


LEARNING_RATE = .0003
EPOCHS = 10
ARMS = ("B0", "T0_A", "T1_A", "T2_A", "T3_A", "X0_published")
MAIN = ("T3_A", "T1_A")
PAIRS = (MAIN, ("T1_A", "T0_A"), ("T2_A", "T0_A"), ("T2_A", "T1_A"),
         ("T3_A", "T2_A"), ("T3_A", "X0_published"), ("T0_A", "X0_published"),
         *((arm, "B0") for arm in ARMS[1:]))


def provenance(args):
    if sha256(args.audit/"receipt.json") != sha256(args.audit_pin) or sha256(args.original) != sha256(args.original_pin):
        raise ValueError("The T.A registered metadata or original X forecasts changed")
    ref, audit, old = reference(args.pin), read_json(args.audit_pin), read_json(args.original)
    root = Path(__file__).parent
    if (sha256(args.config) != CONFIG_SHA256 or not audit["eligible"] or audit["expression_read"]
            or not audit["metadata_only"] or audit["access_sha256"] != sha256(args.pin)
            or audit["protocol_sha256"] != sha256(args.protocol) or old["access_sha256"] != sha256(args.pin)
            or audit["code_sha256"] != sha256(root/"t_a_scope.py")
            or sha256(args.audit/"support.tsv") != audit["files"]["support.tsv"]
            or set(old["slices"]) != set(ref["slices"]) or set(audit["slices"]) != set(ref["slices"])):
        raise ValueError("The source-only T.A registration is incomplete or infeasible")
    source = ("t_a.py", "t_a_scope.py", "t123_workflow.py", "shift_calibration.py", "x_workflow.py")
    stamp = dict(access_sha256=sha256(args.pin), protocol_sha256=sha256(args.protocol),
                 audit_sha256=sha256(args.audit_pin), config_sha256=sha256(args.config),
                 original_X_sha256=sha256(args.original), code={**code_fingerprints(),
                      **{name: sha256(root/name) for name in source}})
    return ref, audit, old, stamp


def original_forecast(args, old, query, shape):
    folder = args.original_forecasts/args.slice
    receipt = read_json(folder/"receipt.json")
    path = folder/"forecasts.npz"
    if (sha256(folder/"receipt.json") != old["slices"][args.slice] or sha256(path) != receipt["files"]["forecasts.npz"]
            or sha256(folder/"rows.tsv") != receipt["files"]["rows.tsv"]
            or (folder/"rows.tsv").read_bytes() != query.to_csv(sep="\t", index=False).encode()):
        raise ValueError("The published X0 prediction does not use the registered T.A query rows")
    with np.load(path, allow_pickle=False) as arrays:
        return checked_array(arrays["X0"], shape, "X0_published")


def fixed_prediction(bank, opts, assets, training, query):
    if (len(training) == 0 or len(query) == 0 or set(training.cell_id) & set(query.cell_id)
            or set(training.parent) & set(query.parent) or training.parent.nunique() < 20 or query.parent.nunique() < 20):
        raise ValueError("The fixed source model lacks independent domains and canonical parents")
    genes = bank.landmarks.pr_gene_id.to_numpy()
    model = Fit(FitSource(bank, training, opts, assets, bank.fit), training, query, panel=genes)
    if model.query.treated is not None:
        raise ValueError("A query label entered T.A base prediction")
    design, _ = make_design(model, None, "X0")
    forecasts, _, _, _ = fit_network(model, design, "film", "L0", LEARNING_RATE, SEED, "cpu", EPOCHS, patience=False)
    return checked_array(forecasts["contrast"][2], (len(query), len(genes)), "structure-only source prediction")


def effects(args, ref, audit, staged, bank):
    select, rotated = select_controls(args, ref, args.slice, staged.tables)
    expected = dict(predict=ref["slices"][args.slice]["predict"], rotated_key=rotated,
                    rotated_fit_sha256=ref["slices"][rotated]["predict"]["fit_controls.tsv"])
    if audit["slices"][args.slice] != expected or any(staged.tables["query"].cell_id.isin(select.cell_id)):
        raise ValueError("A selected control had the held target cell or was outside the frozen metadata support")
    frame, cols = matrix_axes(args.access, ref)
    if not frame.equals(staged.genes):
        raise ValueError("T.A source select controls differ from the original landmark axis")
    raw_select = extract_matrix(ref["matrix"], select, cols, frame.pr_gene_id.to_numpy())
    return responses(bank.training, bank.labels, select, raw_select)


def nested_calibration(args, ref, audit, staged, bank):
    training = bank.training
    source = tuple(sorted(training.cell_id.unique()))
    parent_groups = source_folds(training, bank.fold)
    select_delta = effects(args, ref, audit, staged, bank)
    fit_controls, fit_values = staged.tables["fit_controls"], staged.arrays["fit_controls"]
    genes = tuple(bank.landmarks.pr_gene_id)
    opts = config(args.config)

    @lru_cache(maxsize=None)
    def factors(cells, held):
        return rule_factors(fit_controls, fit_values, genes, cells, held)

    def prediction(fit, query):
        return fixed_prediction(bank, opts, args.assets, fit, query)

    def evaluate_source(fit_cells, fit_folds, query_cell, query_fold):
        fit = training.loc[training.cell_id.isin(fit_cells) & parent_groups.isin(fit_folds)]
        chosen = (training.cell_id.eq(query_cell) & parent_groups.eq(query_fold)).to_numpy()
        query = training.loc[chosen]
        if query_cell == bank.target or set(query.parent) & set(bank.outer.parent) or bank.target in fit_cells:
            raise ValueError("An outer-held treatment entered the nested calibrator")
        return query, prediction(fit, query), select_delta[chosen], factors(fit_cells, query_cell)

    inner_loss = {state: [] for state in SETTINGS}
    for hold in source:
        other = tuple(cell for cell in source if cell != hold)
        for validation in range(3):
            fit_fold, scale_fold = (validation+2) % 3, (validation+1) % 3
            anchors = [evaluate_source(tuple(c for c in other if c != pseudo), (fit_fold,), pseudo, scale_fold)
                       for pseudo in other]
            coefficients = gains(anchors, genes, SETTINGS)
            rows, p, y, scale = evaluate_source(other, (fit_fold,), hold, validation)
            for option in SETTINGS:
                inner_loss[option].append(loss(rows, y, scale[option] * coefficients[option] * p))
    objectives = {key: float(np.mean(items)) for key, items in inner_loss.items()}
    if not np.isfinite(list(objectives.values())).all() or any(len(v) != 12 for v in inner_loss.values()):
        raise ValueError("T.A did not score all nested calibration design cells")
    best = min(SETTINGS, key=lambda key: (round(objectives[key], 9), key))
    oof = [evaluate_source(tuple(c for c in source if c != hold), tuple(i for i in range(3) if i != f), hold, f)
           for hold in source for f in range(3)]
    whole = pd.concat([p[0] for p in oof], ignore_index=True)
    if len(whole) != len(training) or set(keys(whole)) != set(keys(training)) or whole.inst_id.duplicated().any():
        raise ValueError("The final scale fit does not use each independent source OOF well exactly once")
    coefficient = gains(oof, genes, [best])
    raw = prediction(training, bank.outer)
    adjusted = raw * factors(source, bank.target)[best]
    predictions = dict(B0=np.zeros_like(raw), T0_A=raw, T1_A=coefficient[None]*raw,
                       T2_A=adjusted, T3_A=coefficient[best]*adjusted)
    setting = dict(alpha=best[0], floor_multiplier=best[1], log_clip=best[2], selection_loss=objectives[best],
                   T1_gain=coefficient[None], T3_gain=coefficient[best], source_wells=len(whole),
                   source_parents=training.parent.nunique(), source_contexts=len(source),
                   target_fit_controls=int(fit_controls.cell_id.eq(bank.target).sum()))
    return predictions, setting


def predict(args):
    ref, audit, old, stamp = provenance(args)
    staged = read_predict(args.inputs/args.slice, args.pin, args.slice)
    bank = ContextBank(staged, args.slice, ref["cells"])
    arrays, settings = nested_calibration(args, ref, audit, staged, bank)
    shape = (len(bank.outer), len(bank.landmarks))
    arrays["X0_published"] = original_forecast(args, old, staged.tables["query"], shape)
    for arm in arrays:
        arrays[arm] = checked_array(arrays[arm], shape, arm).astype(np.float32)
    out = args.forecasts/args.slice
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".ta-predict-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage/"forecasts.npz", **arrays)
        write_json(stage/"selection.json", settings)
        write_json(stage/"receipt.json", dict(stamp, slice=args.slice, bundle_sha256=staged.receipt,
            query_sha256=ref["slices"][args.slice]["predict"]["query.tsv"],
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(out)


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("Independent scoring requires the exact frozen T.A digest")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    ref, _, _, stamp = provenance(args)
    cells = ref["cells"]
    genes, columns = matrix_axes(args.access, ref)
    queries, receipts, choices, forecasts = {}, {}, [], {arm: [] for arm in ARMS}
    for key in sorted(ref["slices"]):
        _, _, data = read_plan(args.access, args.pin, key, "predict")
        query = queries[key] = data["query"]
        folder = args.forecasts/key
        rec = read_json(folder/"receipt.json")
        if (any(rec[k] != value for k, value in stamp.items()) or rec["slice"] != key
                or rec["query_sha256"] != ref["slices"][key]["predict"]["query.tsv"]
                or rec["bundle_sha256"] != sha256(args.inputs/key/"receipt.json")
                or set(rec["files"]) != {"selection.json", "forecasts.npz"}
                or any(sha256(folder/name) != value for name, value in rec["files"].items())):
            raise ValueError("A T.A prediction changed from its committed population or input")
        with np.load(folder/"forecasts.npz", allow_pickle=False) as file:
            if set(file.files) != set(ARMS):
                raise ValueError("The T.A model inventory changed")
            for arm in ARMS:
                values = checked_array(file[arm], (len(query), len(genes)), arm)
                if scoring:
                    forecasts[arm].append(values)
        receipts[key] = sha256(folder/"receipt.json")
        choices.append(dict(slice=key, **read_json(folder/"selection.json")))
    rows = checked_rows(pd.concat([query.assign(fold=slice_axes(key, cells)[2], role=slice_axes(key, cells)[3])
                                   for key, query in queries.items()], ignore_index=True), None)
    if len(receipts) != 45 or rows.parent.nunique() != 493 or rows.inst_id.nunique() != 14898:
        raise ValueError("T.A did not cover the original cell and parent population")
    axes, groups, counts, index = interaction_axes(rows, cells)
    receipt = dict(stamp, slices=receipts)
    if not scoring:
        write_json(args.freeze, receipt)
        return
    if read_json(args.freeze) != receipt:
        raise ValueError("T.A input or forecast hashes changed after the independent freeze")
    truth = np.concatenate([score_delta(args, key, query, genes, columns) for key, query in queries.items()])
    forecasts = {arm: np.concatenate(values) for arm, values in forecasts.items()}
    paired, sensitivity = comparisons(rows, truth, forecasts, cells, centered=False, pairs=PAIRS, primary_pair=MAIN)

    def center(x):
        return center_by_context(x, groups, counts, index, len(cells))

    interaction, _ = comparisons(axes, center(truth), {arm: center(v) for arm, v in forecasts.items()}, cells,
                                 centered=True, pairs=PAIRS, primary_pair=MAIN)
    with TemporaryDirectory(dir=args.out.parent, prefix=".ta-score-") as tmp:
        stage = Path(tmp)
        outputs = dict(paired=paired, interaction=interaction, sensitivity=sensitivity, selection=choices)
        for name, data in outputs.items():
            pd.DataFrame(data).to_csv(stage/f"{name}.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(receipt, freeze_sha256=args.expected_sha256,
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "freeze", "score"))
    for name in ("access", "pin", "protocol", "audit", "audit-pin", "inputs", "config", "assets", "original", "original-pin",
                 "original-forecasts", "forecasts", "freeze", "out"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--slice")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    (predict if args.action == "predict" else evaluate)(args)
