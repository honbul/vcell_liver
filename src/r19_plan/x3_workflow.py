"""Run the separate B-information X3 comparison on independently scored HEPG2 wells."""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.estimators import GAINS, loss
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.r18_predict import vehicle_means
from src.r19_plan.x_features import fit_basal_context
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes, read_plan, read_predict, reference, slice_axes
from src.r19_plan.x_predict import ContextBank, basal_asset
from src.r19_plan.x_transfer import RIDGES, TEMPERATURES, donor_mean, fit_basal_response
from src.r19_plan.x_workflow import score_delta

TARGET = "HEPG2"
ARMS = ("B_cell", "X3_similarity", "X3_linear", "B_cell_parent_shuffle", "X3_similarity_parent_shuffle", "X3_linear_parent_shuffle")


def donor_inputs(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    ref, _, plan = read_plan(args.access, args.pin, args.slice, "predict")
    target, _, fold, fit_role, _, _ = slice_axes(args.slice, ref["cells"])
    expected = set(plan["query"].parent)
    if target != TARGET:
        raise ValueError("X3 is registered only for HEPG2 with non-hepatic donors")
    lineage, treatments, controls = {}, [], []
    for cell in (x for x in ref["cells"] if x != TARGET):
        source_key = f"{cell}-f{fold}-{fit_role}"
        _, _, source = read_plan(args.access, args.pin, source_key, "score")
        _, _, design = read_plan(args.access, args.pin, source_key, "predict")
        candidates = source["labels"]
        if (not candidates.cell_id.eq(cell).all() or set(candidates.parent) != expected
                or not candidates.drop(columns="level3_row_index").equals(design["query"])
                or not candidates.hours.astype(float).eq(6).all() or not candidates.dose.astype(float).eq(10).all()):
            raise ValueError("X3 B donors differ from registered matched nonhepatic query measurements")
        fit_controls = design["fit_controls"]
        dmsos = fit_controls.loc[fit_controls.rna_plate.isin(candidates.rna_plate)]
        control_axis(dmsos, candidates, fit_role)
        treatments.append(candidates)
        controls.append(dmsos)
        lineage[cell] = dict(source=source_key, stages=ref["slices"][source_key])
    rows, vehicles = pd.concat(treatments, ignore_index=True), pd.concat(controls, ignore_index=True)
    check_unique(pd.concat((rows, vehicles, plan["query"]), ignore_index=True))
    genes, cols = matrix_axes(args.access, ref)
    values = extract_matrix(ref["matrix"], pd.concat((rows, vehicles), ignore_index=True), cols, genes.pr_gene_id.to_numpy())
    mean = vehicle_means(vehicles, values[len(rows):], "rna_plate")
    delta = values[:len(rows)] - np.stack([mean[plate] for plate in rows.rna_plate])
    parents = sorted(expected)
    cells = [x for x in ref["cells"] if x != TARGET]
    aggregate = tensorize(rows, delta, parents, cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x3-donors-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage / "donors.npz", parents=parents, cells=cells, genes=genes.pr_gene_id.to_numpy(), values=aggregate)
        write_json(stage / "receipt.json", dict(slice=args.slice, access_sha256=sha256(args.pin), parent_count=len(parents),
            donor_wells=len(rows), donor_control_wells=len(vehicles), upstream=lineage,
            data_sha256=sha256(stage / "donors.npz")))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


def tensorize(rows, effects, parents, cells):
    p, c = pd.Index(parents).get_indexer(rows.parent), pd.Index(cells).get_indexer(rows.cell_id)
    if (p < 0).any() or (c < 0).any():
        raise ValueError("An X3 treatment is not on the admitted cell–parent coordinate")
    counts = np.zeros((len(parents), len(cells)), dtype=np.int32)
    np.add.at(counts, (p, c), 1)
    if np.any(counts == 0):
        raise ValueError("X3 requires the same actual donor parents in every included cell")
    result = np.zeros((len(parents), len(cells), effects.shape[1]), dtype=np.float32)
    np.add.at(result, (p, c), effects)
    return (result / counts[:, :, None]).astype(np.float32)


def inputs_for(bank, training, query, donor_tensor, query_cell, profiles):
    parent_train, parent_query = sorted(set(training.parent)), sorted(set(query.parent))
    cells = sorted(set(training.cell_id))
    if set(parent_train) & set(parent_query) or query_cell in cells:
        raise ValueError("A held compound or context entered X3 basal calibration")
    ids, genes, coordinates = profiles
    encoding = fit_basal_context(cells, ids, genes, coordinates)
    context_map = dict(zip(ids, encoding.transform(genes, coordinates), strict=True))
    effects = tensorize(training, bank.response(training, bank.fit), parent_train, cells)
    uniform = donor_mean(donor_tensor, cells, query_cell)
    expansion = pd.Index(parent_query).get_indexer(query.parent)
    results = {("B_cell", None): uniform}
    for t in TEMPERATURES:
        results["X3_similarity", t] = donor_mean(donor_tensor, cells, query_cell, context_map, t)
    source = np.stack([context_map[c] for c in cells])
    for ridge in RIDGES:
        fit = fit_basal_response(effects, parent_train, cells, source, ridge=ridge,
                                 held_parents=parent_query, held_context=query_cell)
        results["X3_linear", ridge] = fit.predict(uniform, query_cell, cells, context_map)
    return results, expansion


def predict(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    meta = reference(args.pin)
    staged = read_predict(args.bundle, args.pin, args.slice)
    bank = ContextBank(staged, args.slice, meta["cells"])
    inner, validation, outer, query = bank.split()
    profiles = basal_asset(args, meta, bank.landmarks.pr_gene_id.to_numpy())
    validation_parents = sorted(set(validation.parent))
    source = outer.loc[outer.parent.isin(validation_parents) & outer.cell_id.ne(bank.validation_cell)]
    val_donor = tensorize(source, bank.response(source, bank.fit), validation_parents, sorted(set(inner.cell_id)))
    checks = read_json(args.donors / "receipt.json")
    path = args.donors / "donors.npz"
    if checks["slice"] != args.slice or checks["access_sha256"] != sha256(args.pin) or sha256(path) != checks["data_sha256"]:
        raise ValueError("The X3 B-only donor receipt changed")
    with np.load(path, allow_pickle=False) as f:
        cells, parents, genes, source_values = (f[n] for n in ("cells", "parents", "genes", "values"))
    if (list(parents) != sorted(set(query.parent)) or not np.array_equal(genes, profiles[1])
            or set(cells) != set(outer.cell_id) or bank.target != TARGET):
        raise ValueError("X3 donors changed the complete nonhepatic query axes")
    order = pd.Index(cells).get_indexer(sorted(set(outer.cell_id)))
    source_values = source_values[:, order]
    validation_bank, val_index = inputs_for(bank, inner, validation, val_donor, bank.validation_cell, profiles)
    score_bank, score_index = inputs_for(bank, outer, query, source_values, TARGET, profiles)
    truth = bank.response(validation, bank.select)
    picks, arrays = {}, {}
    shuffled = np.random.default_rng(1501).permutation(len(parents))
    for arm in ARMS[:3]:
        candidates = [(name, value) for name, value in validation_bank if name == arm]
        objectives = {(value, gain): loss(validation, truth, gain * validation_bank[name, value][val_index])
                      for (name, value), gain in product(candidates, GAINS)}
        value, gain = min(objectives, key=lambda key: (round(objectives[key], 9), key[1], str(key[0])))
        matrix = checked_array(score_bank[arm, value], (len(parents), len(genes)), arm)
        arrays[arm] = (gain * matrix[score_index]).astype(np.float32)
        arrays[arm + "_parent_shuffle"] = (gain * matrix[shuffled][score_index]).astype(np.float32)
        picks[arm] = dict(parameter=value, gain=gain, independent_loss=objectives[value, gain])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x3-predict-") as tmp:
        stage = Path(tmp)
        staged.tables["query"].to_csv(stage / "rows.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecasts.npz", **arrays)
        write_json(stage / "selection.json", picks)
        write_json(stage / "receipt.json", dict(slice=args.slice, access_sha256=sha256(args.pin),
            protocol_sha256=sha256(args.protocol), donor_sha256=sha256(args.donors / "receipt.json"),
            genes_sha256=meta["files"]["landmarks.tsv"], code_sha256=sha256(__file__),
            files={f.name: sha256(f) for f in stage.iterdir()}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


def evaluate(args):
    if args.action == "score" and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("X3 scoring requires the independent freeze digest and a new output path")
    ref = reference(args.pin)
    genes, cols = matrix_axes(args.access, ref)
    keys = sorted(k for k in ref["slices"] if k.startswith(TARGET + "-"))
    receipts, queries, arrays = {}, {}, {arm: [] for arm in ARMS}
    for key in keys:
        _, _, source = read_plan(args.access, args.pin, key, "predict")
        queries[key] = source["query"]
        path = args.forecasts / key
        receipt = read_json(path / "receipt.json")
        if (receipt["slice"] != key or receipt["access_sha256"] != sha256(args.pin)
                or receipt["protocol_sha256"] != sha256(args.protocol) or receipt["code_sha256"] != sha256(__file__)
                or receipt["genes_sha256"] != ref["files"]["landmarks.tsv"]
                or set(receipt["files"]) != {"rows.tsv", "forecasts.npz", "selection.json"}
                or any(sha256(path/name) != h for name, h in receipt["files"].items())
                or (path/"rows.tsv").read_bytes() != source["query"].to_csv(sep="\t", index=False).encode()):
            raise ValueError("X3 predictions differ from the registered HEPG2 role, well or code")
        with np.load(path / "forecasts.npz", allow_pickle=False) as f:
            if set(f.files) != set(ARMS):
                raise ValueError("X3 prediction inventory changed")
            for arm in ARMS:
                values = checked_array(f[arm], (len(queries[key]), len(genes)), arm)
                if args.action == "score":
                    arrays[arm].append(values)
        receipts[key] = sha256(path / "receipt.json")
    stamp = dict(protocol_sha256=sha256(args.protocol), access_sha256=sha256(args.pin), slices=receipts,
                 code_sha256=sha256(__file__), transfer_sha256=sha256(Path(__file__).with_name("x_transfer.py")))
    if args.action == "freeze":
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if read_json(args.freeze) != stamp:
        raise ValueError("X3 predictions changed after freeze")
    rows = checked_rows(pd.concat([queries[k].assign(fold=slice_axes(k, ref["cells"])[2],
                        role=slice_axes(k, ref["cells"])[3]) for k in keys], ignore_index=True), None)
    truth = np.concatenate([score_delta(args, k, queries[k], genes, cols) for k in keys])
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, np.concatenate(arrays[arm])) for arm in ARMS}
    pairs = [(arm, "B_cell") for arm in ARMS[1:3]] + [(arm, arm+"_parent_shuffle") for arm in ARMS[:3]]
    results = []
    for scope in ("ALL", 0, 1, 2):
        mask = np.ones(len(rows), dtype=bool) if scope == "ALL" else rows.fold.eq(scope).to_numpy()
        for arm, baseline in pairs:
            values = parent_summaries(rows.loc[mask], len(genes), energy[mask], errors[arm][mask], errors[baseline][mask])
            points, ci, draws = paired_effects(*values)
            low, high, _ = relative_interval(values[0])
            for metric, (a, b), (lo, hi), n in zip(METRICS, points, ci, draws, strict=True):
                results.append(dict(scope=scope, arm=arm, reference=baseline, metric=metric, arm_value=a, reference_value=b,
                    difference=a-b, low99=lo, high99=hi, relative_gain=1-a/b if metric=="rmse" and b>0 else np.nan,
                    relative_low99=low if metric=="rmse" else np.nan, relative_high99=high if metric=="rmse" else np.nan,
                    parents=rows.loc[mask].parent.nunique(), wells=rows.loc[mask].inst_id.nunique(), valid_draws=int(n)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x3-score-") as tmp:
        stage = Path(tmp)
        pd.DataFrame(results).to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
                   files={"paired.tsv": sha256(stage / "paired.tsv")}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "predict", "freeze", "score"))
    for key in ("access", "pin", "out", "bundle", "basal", "donors", "protocol", "forecasts", "freeze"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--slice")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print({"stage": donor_inputs, "predict": predict, "freeze": evaluate, "score": evaluate}[args.action](args))
