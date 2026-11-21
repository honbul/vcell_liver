"""Fit a metadata-only position rule and score the sealed R13 compound folds."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY, LINE, group_weights, load_data, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align
from src.r12_multiaxis.data import select_panel
from src.r12_multiaxis.split import eligible_pools
from src.r12_multiaxis.stage import load_staged
from src.r13_optimization.objectives import GainScores, center, row_products
from src.r13_optimization.score import controls, draw_contrast


WELL = KEY[0]
GAINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
ARMS = ("B_pos", "perm41", "perm42", "perm43")


def positions(path):
    book = table(path / "positions.tsv").sort_values(["plate", "barcode_position"])
    rec = read_json(path / "receipt.json")
    if sha256(path / "positions.tsv") != rec["positions_sha256"] or book["sample"].duplicated().any():
        raise ValueError("Position manifest changed")
    return book.set_index("sample")


def label_sets(book):
    labels = {"B_pos": book.barcode_position}
    for seed in (41, 42, 43):
        rng = np.random.default_rng(seed)
        labels[f"perm{seed}"] = book.groupby("plate", sort=True).barcode_position.transform(rng.permutation)
    return labels


def position_predict(train, values, query, labels):
    if set(train[GROUP]) & set(query[GROUP]) or set(train[WELL]) & set(query[WELL]):
        raise ValueError("A query compound group or physical well entered the position dictionary")
    bank = defaultdict(dict)
    for i, (plate, pos, line) in enumerate(zip(train.plate_id, labels.loc[train[WELL]], train[LINE])):
        if plate in bank[pos, line]:
            raise ValueError("Position and cell line identify multiple rows on one donor plate")
        bank[pos, line][plate] = i
    out = np.zeros((len(query), values.shape[1]), dtype=np.float32)
    support = np.zeros(len(query), dtype=int)
    for i, (plate, pos, line) in enumerate(zip(query.plate_id, labels.loc[query[WELL]], query[LINE])):
        donor = [j for donor_plate, j in bank[pos, line].items() if donor_plate != plate]
        if donor:
            out[i] = values[donor].mean(axis=0)
        support[i] = len(donor)
    return out, support


def checked_history(args, fold, role):
    path = args.history / "frozen" / f"task{2 * fold + int(role == 'B')}"
    rec = read_json(path / "receipt.json")
    prior = read_json(args.history / "freeze.json")
    if sha256(path / "receipt.json") != prior["task_receipts"][path.name]:
        raise ValueError("Historical prediction receipt changed")
    for name in ("rows.tsv", "panel.npy"):
        if sha256(path / name) != rec["files"][name]:
            raise ValueError("Historical rows or panel changed")
    return table(path / "rows.tsv"), np.load(path / "panel.npy", allow_pickle=False)


def fit(args, cfg, book):
    stage = args.assets / "staged"
    split = read_json(args.assets / "partition/split.json")
    stage_receipt = read_json(stage / "receipt.json")
    labels = label_sets(book)
    result = []
    args.out.mkdir(parents=True, exist_ok=False)
    for fold in range(3):
        train, counts = load_staged(stage / f"outer{fold}", "fit", stage_receipt["genes"])
        query, panel = checked_history(args, fold, "A")
        rows_b, panel_b = checked_history(args, fold, "B")
        if (not query[KEY + [GROUP]].equals(rows_b[KEY + [GROUP]]) or not np.array_equal(panel, panel_b)
                or query[CONTROL].eq(rows_b[CONTROL]).any()):
            raise ValueError("Historical R13 roles differ in assay rows, panel or DMSO assignment")
        if not np.array_equal(panel, select_panel(counts, cfg["panel_size"])):
            raise ValueError("Historical outer panel differs from fit-only reconstruction")
        if not set(train.dose_uM).union(query.dose_uM) == {0.5}:
            raise ValueError("Position fit must remain at 0.5 µM")
        all_rows = pd.concat((train, query), ignore_index=True)
        if not np.array_equal(book.loc[all_rows[WELL], "plate"], all_rows.plate_id):
            raise ValueError("Position metadata changed a release plate")

        membership = train[GROUP].map({int(k): v for k, v in split["inner"][str(fold)].items()})
        loss = {name: np.zeros(len(GAINS)) for name in ARMS}
        inner_weight = 0
        for k in (0, 1):
            aux = train.loc[membership.ne(k)].reset_index(drop=True)
            inner_query = eligible_pools(train.loc[membership.eq(k)], cfg["minimum_groups"])
            aux_values = counts[align(aux, train)]
            ip = select_panel(aux_values, cfg["panel_size"])
            observed = counts[align(inner_query, train)][:, ip].toarray()
            metric = GainScores(inner_query, observed, np.zeros_like(observed), cfg["minimum_groups"])
            donor_metric = Contrast(aux, cfg["minimum_groups"])
            reference = center(donor_metric, aux_values[:, ip].toarray())[donor_metric.active]
            donors = aux.iloc[donor_metric.active]
            weight = metric.metric.primary_weight.sum()
            inner_weight += weight
            for name in ARMS:
                prediction, _ = position_predict(donors, reference, inner_query, labels[name])
                loss[name] += weight * metric.points(prediction, GAINS)[0]

        donor_metric = Contrast(train, cfg["minimum_groups"])
        reference = center(donor_metric, counts[:, panel].toarray())[donor_metric.active]
        donors = train.iloc[donor_metric.active]
        for name in ARMS:
            gain = GAINS[np.argmin(loss[name])]
            prediction, support = position_predict(donors, reference, query, labels[name])
            np.save(args.out / f"{fold}-{name}.npy", prediction * gain, allow_pickle=False)
            result.append(dict(fold=fold, arm=name, gain=gain, inner_contrast=(loss[name] / inner_weight).tolist(),
                               query_rows=len(query), query_groups=int(query[GROUP].nunique()),
                               supported_rows=int((support > 0).sum()),
                               supported_groups=int(query.loc[support > 0, GROUP].nunique())))
        print(f"frozen prediction arrays for fold {fold}", flush=True)
    write_json(args.out / "freeze.json", dict(
        gains=GAINS.tolist(), fits=result, model_code_sha256=sha256(__file__), contract_sha256=sha256(args.protocol),
        config_sha256=sha256(args.config), history_freeze_sha256=sha256(args.history / "freeze.json"),
        positions_receipt_sha256=sha256(args.coordinates / "receipt.json"),
        staging_sha256=sha256(stage / "receipt.json"),
        arrays={p.name: sha256(p) for p in args.out.glob("*.npy")},
    ))


def cosine(pred, truth):
    denominator = np.sqrt(row_products(pred, pred) * row_products(truth, truth))
    return np.divide(row_products(pred, truth), denominator, out=np.zeros(len(pred)), where=denominator > 1e-14)


def replicate_probe(query, observed, donor, donor_values, book):
    references = defaultdict(list)
    for i, key in enumerate(zip(donor[GROUP], donor[LINE])):
        references[key].append(i)
    dp = book.loc[donor[WELL], "barcode_position"].to_numpy()
    plates = donor.plate_id.to_numpy()
    pair_wells, selected, predicted = set(), [], []
    for i, (group, line, plate, pos, well) in enumerate(zip(query[GROUP], query[LINE], query.plate_id,
                                                           book.loc[query[WELL], "barcode_position"], query[WELL])):
        take = [j for j in references[group, line] if plates[j] != plate and dp[j] != pos]
        if take:
            selected.append(i)
            predicted.append(np.mean([donor_values[[j for j in take if plates[j] == p]].mean(axis=0)
                                      for p in np.unique(plates[take])], axis=0))
            pair_wells.update(tuple(sorted((well, other))) for other in donor.iloc[take][WELL])
    if not selected:
        return dict(rows=0, groups=0, well_pairs=0, baseline=None, rmse=None, cosine=None)
    pred, target = np.stack(predicted), observed[selected]
    weights = group_weights(query.iloc[selected])
    def rmse(values):
        return float(np.average(np.sqrt(row_products(values, values)), weights=weights))

    return dict(rows=len(selected), groups=int(query.iloc[selected][GROUP].nunique()), well_pairs=len(pair_wells),
                baseline=rmse(target), rmse=rmse(target - pred),
                cosine=float(np.average(cosine(pred, target), weights=weights)))


def dmso_checks(r11, panel, book, plates):
    rows_a, values_a = load_data(r11 / "source", "A", panel)
    rows_b, values_b = load_data(r11 / "source", "B", panel)
    axes, groups = [], {}
    for plate, a in rows_a.loc[rows_a.plate_id.isin(plates)].groupby("plate_id"):
        b = rows_b[rows_b.plate_id.eq(plate)]
        lines = sorted(set(a[LINE]) & set(b[LINE]))
        ai, bi = a.reset_index().set_index(LINE).loc[lines], b.reset_index().set_index(LINE).loc[lines]
        contrast = values_a[ai["index"]].toarray() - values_b[bi["index"]].toarray()
        groups[plate] = dict(zip(lines, contrast))
        axes.append(dict(plate=plate, A=int(book.loc[ai[WELL].iloc[0], "barcode_position"]),
                         B=int(book.loc[bi[WELL].iloc[0], "barcode_position"])))
    pairs = []
    for i, (plate, a) in enumerate(groups.items()):
        for other, b in list(groups.items())[i + 1:]:
            lines = sorted(set(a) & set(b))
            similarity = cosine(np.stack([a[x] for x in lines]), np.stack([b[x] for x in lines]))
            pairs.append(dict(plate=plate, other=other, lines=len(lines), cosine=float(np.mean(similarity))))
    return axes, pairs


def score(args, cfg, book):
    stage = args.assets / "staged"
    frozen = read_json(args.out / "freeze.json")
    checks = {Path(__file__): frozen["model_code_sha256"], args.protocol: frozen["contract_sha256"],
              args.config: frozen["config_sha256"], args.history / "freeze.json": frozen["history_freeze_sha256"],
              args.coordinates / "receipt.json": frozen["positions_receipt_sha256"],
              stage / "receipt.json": frozen["staging_sha256"],
              **{args.out / name: value for name, value in frozen["arrays"].items()}}
    if any(sha256(path) != h for path, h in checks.items()):
        raise ValueError("Pre-scoring freeze mismatch")
    stage_rec = read_json(stage / "receipt.json")
    ids = sorted({g for fold in range(3) for g in checked_history(args, fold, "A")[0][GROUP]})
    draws = np.random.default_rng(130113).exponential(size=(10000, len(ids)))
    draws /= draws.mean(axis=1, keepdims=True)
    nums = {arm: np.zeros(len(draws)) for arm in ("B0", *ARMS)}
    points = defaultdict(float)
    denominator = np.zeros(len(draws))
    diagnostics, plate_pairs, axes, tasks = [], [], [], []
    panel_union = np.unique(np.concatenate([checked_history(args, fold, "A")[1] for fold in range(3)]))
    aux_rows, aux_counts = load_data(args.assets / "doses/source", "treated", panel_union,
                                     mask=lambda rows: rows[GROUP].isin(ids).to_numpy())
    usable = eligible_pools(aux_rows, cfg["minimum_groups"])
    aux_counts = aux_counts[align(usable, aux_rows)]
    r11 = Path(cfg["r11_assets"])
    for fold in range(3):
        true_rows, counts = load_staged(stage / f"outer{fold}", "score", stage_rec["genes"])
        obs_a = None
        for role in "AB":
            query, panel = checked_history(args, fold, role)
            actual = counts[align(query, true_rows)][:, panel].toarray()
            _, opposite = controls(r11, query, role, panel)
            observed = actual - opposite
            metric = Contrast(query, cfg["minimum_groups"])
            index = pd.Index(ids).get_indexer(query[GROUP])
            weights = metric.primary_weight
            denominator += draws @ np.bincount(index, weights=weights, minlength=len(ids))
            for arm in ("B0", *ARMS):
                prediction = np.zeros_like(actual) if arm == "B0" else np.load(args.out / f"{fold}-{arm}.npy")
                rmse = metric.point(prediction, observed)
                points[arm] += weights.sum() * rmse
                nums[arm] += draw_contrast(metric, prediction, observed, draws, index)
                if arm == "B_pos":
                    coverage, alignment = GainScores(query, observed, np.zeros_like(observed),
                                                     cfg["minimum_groups"]).diagnostics(prediction)
                    tasks.append(dict(fold=fold, role=role, rmse=rmse, alignment=alignment, coverage=coverage,
                                      weight=float(weights.sum())))
            if role == "A":
                obs_a = center(metric, observed)
            elif not np.allclose(obs_a, center(metric, observed), atol=1e-8, rtol=1e-7):
                raise ValueError("Matched DMSO did not cancel within the frozen contrast pools")
        diagnostic = replicate_probe(query, obs_a, query, obs_a, book)
        diagnostics.append(dict(fold=fold, donor="0.5uM", **diagnostic))
        aux = center(Contrast(usable, cfg["minimum_groups"]), aux_counts[:, np.searchsorted(panel_union, panel)].toarray())
        diagnostics.append(dict(fold=fold, donor="0.05_or_5uM", **replicate_probe(query, obs_a, usable, aux, book)))
        ax, pairs = dmso_checks(r11, panel, book, cfg["plates"])
        plate_pairs.extend(dict(fold=fold, **row) for row in pairs)
        axes = ax
        print(f"scored fold {fold}", flush=True)

    point = {key: float(value / (2 * len(ids))) for key, value in points.items()}
    if abs(point["B0"] - 0.0927918471289385) > 2e-8:
        raise ValueError(f"B0 does not match the published R13 point: {point['B0']}")
    effects = []
    for name in ARMS:
        alpha = 0.005 if name == "B_pos" else 0.025
        low, high = np.quantile((nums[name] - nums["B0"]) / denominator, [alpha, 1 - alpha])
        difference = point[name] - point["B0"]
        effects.append(dict(arm=name, rmse=point[name], difference=difference, lower=float(low), upper=float(high),
                            confidence=1-2*alpha, relative_percent=100*difference/point["B0"]))
    write_json(args.out / "results.json", dict(groups=len(ids), b0=point["B0"], effects=effects,
               confirmed=effects[0]["upper"] < 0, by_task=tasks, repeat_probe=diagnostics,
               dmso_positions=axes, dmso_pairs=plate_pairs, draws=len(draws), freeze_sha256=sha256(args.out / "freeze.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "score"))
    for name in ("assets", "history", "coordinates", "config", "out", "protocol"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    (fit if args.command == "fit" else score)(args, config(args.config), positions(args.coordinates))


if __name__ == "__main__":
    main()
