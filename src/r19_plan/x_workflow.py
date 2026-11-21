"""Freeze the X predictions, then score held outcomes and cell–compound interactions."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.r18_predict import vehicle_means
from src.r19_plan.sl_workflow import parent_swap
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes, read_plan, reference, slice_axes, validate_predict
from src.r19_plan.x_predict import ARMS, code_fingerprints

COMPARISONS = (("X2", "X1"), ("X0", "X1"), ("X2", "X0"), *(("X2", name) for name in ARMS[3:]))
FILES = {"rows.tsv", "landmarks.tsv", "forecasts.npz", "selection.json"}


def verify_prediction(args, slice_id, query, genes):
    dir = args.forecasts / slice_id
    receipt = read_json(dir / "receipt.json")
    if (receipt["slice"] != slice_id or receipt["access_sha256"] != sha256(args.pin)
            or receipt["protocol_sha256"] != sha256(args.protocol) or receipt["code_sha256"] != code_fingerprints()
            or set(receipt["files"]) != FILES):
        raise ValueError(f"X forecast provenance changed in {slice_id}")
    for file, digest in receipt["files"].items():
        if sha256(dir / file) != digest:
            raise ValueError(f"An X forecast file changed in {slice_id}: {file}")
    if ((dir / "rows.tsv").read_bytes() != query.to_csv(sep="\t", index=False).encode()
            or (dir / "landmarks.tsv").read_bytes() != genes.to_csv(sep="\t", index=False).encode()):
        raise ValueError(f"X predictor changed the registered physical or gene axes in {slice_id}")
    with np.load(dir / "forecasts.npz", allow_pickle=False) as array:
        if set(array.files) != set(ARMS):
            raise ValueError("X forecasts do not contain the registered model and control arms")
        forecast = {arm: checked_array(array[arm], (len(query), len(genes)), arm) for arm in ARMS}
    return forecast, sha256(dir / "receipt.json")


def score_delta(args, key, query, genes, columns):
    ref, _, score = read_plan(args.access, args.pin, key, "score")
    _, _, _, _, _, score_role = slice_axes(key, ref["cells"])
    labels, controls = score["labels"], score["controls"]
    if not labels.drop(columns="level3_row_index").equals(query):
        raise ValueError("X held data differ from the frozen read-only query")
    check_unique(pd.concat([labels, controls], ignore_index=True))
    control_axis(controls, labels, score_role)
    merged = pd.concat((labels, controls), ignore_index=True)
    values = extract_matrix(ref["matrix"], merged, columns, genes.pr_gene_id.to_numpy())
    averages = vehicle_means(controls, values[len(labels):], "rna_plate")
    return values[:len(labels)] - np.stack([averages[plate] for plate in labels.rna_plate])


def interaction_axes(rows, cells):
    columns = ["parent", "role", "cell_id"]
    keys = pd.MultiIndex.from_frame(rows[columns])
    groups, distinct = pd.factorize(keys, sort=True)
    frame = distinct.to_frame(index=False)
    frame.columns = columns
    if not frame.groupby(["parent", "role"]).cell_id.nunique().eq(len(cells)).all() or set(frame.cell_id) != set(cells):
        raise ValueError("An X context interaction lacks a common parent in all five cells")
    counts = np.bincount(groups)
    index = frame.groupby(["parent", "role"], sort=False).ngroup().to_numpy()
    if not np.bincount(index).min() == np.bincount(index).max() == len(cells):
        raise ValueError("X interaction groups have unequal support")
    return frame, groups, counts, index


def center_by_context(data, groups, counts, index, n_cells):
    values = np.zeros((len(counts), data.shape[1]), dtype=np.float64)
    np.add.at(values, groups, data)
    values /= counts[:, None]
    center = np.zeros((index.max() + 1, data.shape[1]), dtype=np.float64)
    np.add.at(center, index, values)
    return values - (center / n_cells)[index]


def comparisons(rows, truth, forecasts, cells, *, centered):
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, estimates) for arm, estimates in forecasts.items()}
    results, strata = [], []
    for target in ("ALL", *cells):
        mask = np.ones(len(rows), dtype=bool) if target == "ALL" else rows.cell_id.eq(target).to_numpy()
        selected = rows.loc[mask]
        for arm, comparator in COMPARISONS:
            summary = parent_summaries(selected, truth.shape[1], energy[mask], errors[arm][mask], errors[comparator][mask])
            estimates, bounds, draws = paired_effects(*summary)
            practical = target == "ALL" and not centered and (arm, comparator) == ("X2", "X1")
            p, joint = parent_swap(*summary) if practical else ([np.nan, np.nan], np.nan)
            low_rel, high_rel, _ = relative_interval(summary[0])
            for metric, (a, b), (lo, hi), n, single in zip(METRICS, estimates, bounds, draws, p, strict=True):
                results.append(dict(target=target, arm=arm, reference=comparator, metric=metric, arm_value=a,
                    reference_value=b, difference=a-b, low99=lo, high99=hi, valid_draws=int(n), primary=practical,
                    one_sided_p=single, joint_p_unadjusted=joint, parents=selected.parent.nunique(), rows=len(selected),
                    relative_rmse_gain=1-a/b if metric=="rmse" and b>0 else np.nan,
                    relative_low99=low_rel if metric=="rmse" else np.nan,
                    relative_high99=high_rel if metric=="rmse" else np.nan))
            if practical:
                cuts = [("fold", str(k), rows.fold.eq(k)) for k in sorted(rows.fold.unique())]
                cuts.extend(("plate_removed", plate, rows.rna_plate.ne(plate)) for plate in sorted(rows.rna_plate.unique()))
                for scope, key, active in cuts:
                    rmse, sse, e = parent_summaries(rows.loc[active], truth.shape[1], energy[active],
                                                   errors[arm][active], errors[comparator][active])
                    strata.append(dict(scope=scope, key=key, arm=arm, reference=comparator,
                        delta_rmse=float((rmse[:, 0] - rmse[:, 1]).mean()),
                        delta_r2=float((sse[:, 1] - sse[:, 0]).sum()/e.sum()), parents=len(rmse)))
    return results, strata


def run(args):
    if args.action == "score" and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("X scoring requires the independent freeze digest and a fresh output location")
    ref = reference(args.pin)
    cells = ref["cells"]
    genes, columns = matrix_axes(args.access, ref)
    queries, receipts = {}, {}
    pieces = {arm: [] for arm in ARMS}
    for key in sorted(ref["slices"]):
        _, _, tables = read_plan(args.access, args.pin, key, "predict")
        validate_predict(tables, key, cells)
        queries[key] = tables["query"]
        forecasts, receipts[key] = verify_prediction(args, key, queries[key], genes)
        if args.action == "score":
            for arm, values in forecasts.items():
                pieces[arm].append(values)
    row_parts = [frame.assign(fold=slice_axes(key, cells)[2], role=slice_axes(key, cells)[3])
                 for key, frame in queries.items()]
    rows = checked_rows(pd.concat(row_parts, ignore_index=True), None)
    if rows.parent.nunique() != ref["parents"] or len(rows.drop_duplicates("inst_id")) != ref["treatments"]:
        raise ValueError("X did not cover its complete registered physical score population")
    frame, groups, counts, index = interaction_axes(rows, cells)
    stamp = dict(protocol_sha256=sha256(args.protocol), access_sha256=sha256(args.pin), slices=receipts,
                 predictor=code_fingerprints(), score_code_sha256=sha256(__file__))
    if args.action == "freeze":
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if read_json(args.freeze) != stamp:
        raise ValueError("An X predictor slice or scorer changed since freezing")
    observed = np.concatenate([score_delta(args, key, query, genes, columns) for key, query in queries.items()])
    forecast_full = {arm: np.concatenate(parts) for arm, parts in pieces.items()}
    del pieces
    absolute, sensitivity = comparisons(rows, observed, forecast_full, cells, centered=False)
    center = lambda vector: center_by_context(vector, groups, counts, index, len(cells))
    interaction, _ = comparisons(frame, center(observed), {arm: center(x) for arm, x in forecast_full.items()},
                                 cells, centered=True)
    outputs = {"paired": absolute, "interaction": interaction, "sensitivity": sensitivity}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x-score-") as tmp:
        stage = Path(tmp)
        for name, records in outputs.items():
            pd.DataFrame(records).to_csv(stage / f"{name}.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
                   files={p.name: sha256(p) for p in sorted(stage.iterdir())}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    for name in ("access", "pin", "protocol", "forecasts", "freeze"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.action == "score" and any(value is None for value in (args.out, args.expected_sha256)):
        parser.error("score requires --out and --expected-sha256")
    print(run(args))
