"""Measure response energy and different-position repeat prediction on frozen targets."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, POOL, group_weights, load_data, table
from src.r13_optimization.objectives import center, row_products

from .study import position_predict, positions
from .targets import metric_for


WELL = KEY[0]
DOSE_PAIRS = ((0.05, 0.05), (0.5, 0.5), (5.0, 5.0), (0.5, 0.05), (0.5, 5.0))
GAIN = 0.25
SEED = 130113
REPEATS = 10000
LIVER = "CVCL_0366"


def eligible(rows, minimum, mask):
    candidate = rows.loc[mask]
    keep = candidate.groupby(POOL, sort=False)[GROUP].transform("nunique").ge(minimum)
    return candidate.index[keep].to_numpy()


def esq(values, weights):
    return float(np.average(row_products(values, values), weights=weights))


def energy(rows, actual, nuisance, minimum, weighted):
    metric = metric_for(rows, minimum, weighted)
    true = center(metric, actual)
    remainder = center(metric, actual - nuisance)
    weights = metric.primary_weight
    kept = metric.active
    total, corrected = esq(true, weights), esq(remainder, weights)
    remaining = remainder.copy()
    for ix in rows.iloc[kept].groupby(WELL).groups.values():
        ids = ix.to_numpy()
        remaining[ids] -= np.average(remainder[ids], axis=0, weights=weights[ids])
    last = esq(remaining, weights)
    return dict(rows=len(kept), groups=int(rows.iloc[kept][GROUP].nunique()), total=total,
                position_removed=total - corrected, corrected_total=corrected,
                well_shared=corrected - last, remaining_including_sampling=last)


def energy_summary(components):
    measures = ("total", "position_removed", "corrected_total", "well_shared", "remaining_including_sampling")
    summary = {}
    for target in ("original", "combined"):
        folds = [x for x in components if x["target"] == target]
        weights = [x["groups"] for x in folds]
        estimates = {name: float(np.average([x[name] for x in folds], weights=weights)) for name in measures}
        summary[target] = dict(**estimates, groups=sum(weights), rows=sum(x["rows"] for x in folds))
    return summary


def read_halves(args):
    receipt = read_json(args.halves_receipt)
    for file in ("wells.tsv", "treated_halves.npy"):
        if sha256(args.halves / file) != receipt["files"][file]["sha256"]:
            raise ValueError(f"Legacy half-split receipt mismatch for {file}")
    wells = table(args.halves / "wells.tsv").rename(columns={"sample": WELL})
    cols = table(args.half_genes).sort_values("panel_index").token_id.to_numpy()
    return wells, cols, np.load(args.halves / "treated_halves.npy", mmap_mode="r")


def split_noise(rows, panel, legacy, minimum, weighted):
    wells, cols, values = legacy
    same_genes, _, legacy_cols = np.intersect1d(panel, cols, return_indices=True)
    linked = rows.merge(wells[[*KEY, "wid", "n_A", "n_B"]], on=KEY, validate="one_to_one", how="inner")
    available = linked.n_A.ge(10) & linked.n_B.ge(10)
    use = eligible(linked, minimum, available)
    if not len(use) or not len(same_genes):
        return dict(rows=0, genes=len(same_genes), noise=None, legacy_total=None, noise_fraction=None)
    linked = linked.loc[use].reset_index(drop=True)
    metric = metric_for(linked, minimum, weighted)
    ix = linked.wid.to_numpy(dtype=int)
    a, b = linked.n_A.to_numpy(), linked.n_B.to_numpy()
    scaling = np.sqrt(a * b) / (a + b)
    dif = (values[0][np.ix_(ix, legacy_cols)] - values[1][np.ix_(ix, legacy_cols)]) * scaling[:, None]
    noise = esq(center(metric, dif), metric.primary_weight)
    denom = esq(center(metric, values[2][np.ix_(ix, legacy_cols)]), metric.primary_weight)
    return dict(rows=len(linked), genes=len(same_genes), groups=int(linked[GROUP].nunique()), noise=noise,
                legacy_total=denom, noise_fraction=noise / denom if denom > 0 else None)


def macro_weights(rows):
    w = group_weights(rows) * rows.n_cells.to_numpy()
    totals = pd.Series(w).groupby(rows[GROUP].to_numpy()).transform("sum").to_numpy()
    return w / totals


def per_anchor(target, observed, donor, donor_values, labels):
    references = defaultdict(list)
    for j, key in enumerate(zip(donor[GROUP], donor[LINE])):
        references[key].append(j)
    donor_plates, donor_wells = donor.plate_id.to_numpy(), donor[WELL].to_numpy()
    donor_labels = labels.loc[donor[WELL]].to_numpy()
    selected, predictions, well_pairs = [], [], []
    for i, (group, line, plate, label, well) in enumerate(zip(target[GROUP], target[LINE], target.plate_id,
                                                              labels.loc[target[WELL]], target[WELL])):
        same_group = references[group, line]
        by_plate = defaultdict(list)
        for j in same_group:
            if donor_plates[j] != plate and donor_labels[j] != label:
                by_plate[donor_plates[j]].append(j)
        if by_plate:
            selected.append(i)
            predictions.append(np.mean([donor_values[ix].mean(axis=0) for ix in by_plate.values()], axis=0))
            well_pairs.append([tuple(sorted((well, donor_wells[j]))) for ix in by_plate.values() for j in ix])
    if not selected:
        return None
    obs = observed[selected]
    pred = np.stack(predictions)
    denom = np.sqrt(row_products(obs, obs) * row_products(pred, pred))
    result = target.iloc[selected][[*KEY, GROUP, ENTITY, "n_cells", "strength"]].copy()
    result["b0"] = np.sqrt(row_products(obs, obs))
    result["rmse025"] = np.sqrt(row_products(obs - GAIN * pred, obs - GAIN * pred))
    result["rmse1"] = np.sqrt(row_products(obs - pred, obs - pred))
    result["cosine"] = np.divide(row_products(obs, pred), denom, out=np.zeros(len(pred)), where=denom > 1e-14)
    result["pairs"] = well_pairs
    return result


def summarize_pairs(rows, target_dose, source_dose, stratum):
    result = dict(target_dose=target_dose, source_dose=source_dose, stratum=stratum, rows=len(rows),
                  groups=int(rows[GROUP].nunique()), well_pairs=len({pair for links in rows.pairs for pair in links}))
    if rows.empty:
        return result
    metrics = ["b0", "rmse025", "rmse1", "cosine"]
    weights = macro_weights(rows)
    grouped = rows[metrics].multiply(weights, axis=0).groupby(rows[GROUP]).sum()
    rng = np.random.default_rng(SEED)
    values = grouped.to_numpy()
    boot = values[rng.integers(len(grouped), size=(REPEATS, len(grouped)))].mean(axis=1)
    point = grouped.mean()
    for column in metrics:
        result[column] = float(point[column])
    for k, name in ((1, "gain025"), (2, "gain1")):
        result[f"{name}_improvement_percent"] = float(100 * (1 - point[metrics[k]] / point.b0)) if point.b0 > 0 else None
        result[f"{name}_improvement_ci95"] = (np.quantile(100 * (1 - boot[:, k] / boot[:, 0]), [0.025, 0.975]).tolist()
                                                if np.all(boot[:, 0] > 0) else None)
    result["cosine_ci95"] = np.quantile(boot[:, 3], [0.025, 0.975]).tolist()
    return result


def attach_strength(rows, lookup):
    key = [*KEY, "dose_uM"]
    linked = rows.merge(lookup[[*key, "strength"]], on=key, how="left", validate="one_to_one", sort=False)
    return linked.assign(strength=linked.strength.fillna("unlabeled"))


def corrected(rows, actual, nuisance, minimum):
    index = eligible(rows, minimum, rows.n_cells.ge(200))
    kept = rows.iloc[index].reset_index(drop=True)
    if not len(kept):
        return kept, np.empty((0, actual.shape[1]))
    return kept, center(metric_for(kept, minimum, True), actual[index] - nuisance[index])


def moa_auc(rows, values, annotations, cell_line=LIVER):
    use = rows[LINE].eq(cell_line) & rows[ENTITY].isin(annotations)
    rows = rows.loc[use].reset_index(drop=True)
    values = values[use.to_numpy()]
    norms = np.linalg.norm(values, axis=1)
    scores = (values @ values.T) / np.maximum(norms[:, None] * norms, 1e-20)
    groups, plates = rows[GROUP].to_numpy(), rows.plate_id.to_numpy()
    moa = rows[ENTITY].map(annotations).to_numpy()
    records = []
    for i, (group, plate, label) in enumerate(zip(groups, plates, moa)):
        candidates = (groups != group) & (plates != plate) & (norms > 1e-12)
        if norms[i] <= 1e-12:
            continue
        auroc = []
        for other in sorted(set(plates[candidates])):
            ids = candidates & (plates == other)
            by_group = pd.DataFrame({"score": scores[i, ids], "group": groups[ids], "matched": moa[ids] == label})
            if by_group.empty:
                continue
            labels_per_group = by_group.groupby("group").matched.nunique()
            stable = by_group.group.isin(labels_per_group[labels_per_group == 1].index)
            candidates_grouped = by_group[stable].groupby(["group", "matched"]).score.mean().reset_index()
            yes = candidates_grouped.loc[candidates_grouped.matched, "score"].to_numpy()
            no = candidates_grouped.loc[~candidates_grouped.matched, "score"].to_numpy()
            if len(yes) and len(no):
                delta = yes[:, None] - no
                auroc.append(float(np.mean((delta > 0) + (delta == 0) * 0.5)))
        if auroc:
            records.append((group, float(np.mean(auroc))))
    return records


def summarize_moa(records):
    groups = pd.DataFrame(records, columns=["group", "auc"]).groupby("group").auc.mean().to_numpy()
    if not len(groups):
        return dict(groups=0, anchors=0)
    boot = groups[np.random.default_rng(SEED).integers(len(groups), size=(REPEATS, len(groups)))].mean(axis=1)
    return dict(groups=int(len(groups)), anchors=len(records), auc=float(groups.mean()),
                auc_ci95=np.quantile(boot, [0.025, 0.975]).tolist())


def verified_inputs(args):
    receipt = read_json(args.common / "receipt.json")
    freeze = args.common / "position_freeze.json"
    frozen = read_json(freeze)
    if (receipt["position_freeze_sha256"] != sha256(freeze)
            or frozen["protocol_sha256"] != sha256(args.protocol)
            or frozen["coordinate_sha256"] != sha256(args.coordinates / "receipt.json")):
        raise ValueError("Common positions do not match the prescore freeze")
    needed = ["f4_labels.tsv"]
    for i in range(6):
        needed.extend(f"task{i}/{name}" for name in ("rows.tsv", "observed.npy", "observed_filtered.npy"))
    if any(sha256(args.common / name) != receipt["files"][name] for name in needed):
        raise ValueError("A common response or metadata file changed")
    if any(sha256(args.common / name) != digest for name, digest in frozen["files"].items()):
        raise ValueError("A prescore fold file changed")


def check_control_role(rows, other, a, b, minimum, weighted):
    if not rows[[*KEY, *POOL, GROUP]].equals(other[[*KEY, *POOL, GROUP]]):
        raise ValueError("Opposite DMSO roles have different response rows")
    metric = metric_for(rows, minimum, weighted)
    difference = 0.0
    for ix, weight in zip(metric.pools, metric.centers):
        offset = a[ix].astype(np.float64) - b[ix]
        difference = max(difference, float(np.max(np.abs(offset - np.average(offset, axis=0, weights=weight)))))
    if difference > 1e-6:
        raise ValueError(f"Opposite DMSO pools differ after centering: {difference}")
    return difference


def moa_annotations(path):
    moa = pd.read_parquet(path, columns=["drug", "moa-fine"]).dropna().drop_duplicates()
    moa = moa[~moa.drug.duplicated(keep=False)]
    return {k: v for k, v in zip(moa.drug, moa["moa-fine"].str.strip().str.casefold())
            if v not in {"", "unknown", "unclear", "unannotated", "other", "n/a", "na"}}


def run_moa(args):
    verified_inputs(args)
    minimum = config(args.config)["minimum_groups"]
    annotations = moa_annotations(args.moa)
    all_scores = defaultdict(list)
    for fold in range(3):
        task = args.common / f"task{2 * fold}"
        rows = table(task / "rows.tsv")
        actual = np.load(task / "observed_filtered.npy", mmap_mode="r", allow_pickle=False)
        nuisance = np.load(task / "nuisance_filtered.npy", mmap_mode="r", allow_pickle=False)
        rows, values = corrected(rows, actual, nuisance, minimum)
        for line in sorted(rows[LINE].unique()):
            all_scores[line].extend(moa_auc(rows, values, annotations, line))
        print(f"completed all-line MoA diagnostics for fold {fold}", flush=True)
    pooled = summarize_moa([item for records in all_scores.values() for item in records])
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "moa.json", dict(
        pooled=pooled, by_line={line: summarize_moa(records) for line, records in all_scores.items()},
        bootstrap_draws=REPEATS, bootstrap_seed=SEED, protocol_sha256=sha256(args.protocol),
        common_receipt_sha256=sha256(args.common / "receipt.json"), moa_sha256=sha256(args.moa),
        matching="same line and fold, different physical plate and compound group, same donor plate for the positive and negative",
        weighting="donor rows averaged per compound and donor plate; donor-plate AUC then averaged within anchor row and rows within anchor compound",
        uncertainty="conditional on corrected contrasts and donor candidates; resampling whole anchor groups with replacement",
    ))
    print(json.dumps(pooled), flush=True)


def run(args):
    verified_inputs(args)
    cfg = config(args.config)
    minimum = cfg["minimum_groups"]
    legacy = read_halves(args)
    book = positions(args.coordinates)
    labels = book.barcode_position
    f4 = table(args.strata)
    all_rows = [table(args.common / f"task{2 * fold}" / "rows.tsv") for fold in range(3)]
    panels = [np.load(args.common / f"task{2 * fold}" / "panel_filtered.npy", allow_pickle=False) for fold in range(3)]
    union = np.unique(np.concatenate(panels))
    groups = set().union(*(set(rows[GROUP]) for rows in all_rows))
    if sum(rows[GROUP].nunique() for rows in all_rows) != len(groups):
        raise ValueError("A scored compound belongs to more than one fold")
    aux_rows, aux_counts = load_data(args.assets / "doses/source", "treated", union,
                                     mask=lambda frame: frame[GROUP].isin(groups).to_numpy() & frame.n_cells.ge(200).to_numpy())
    complete = attach_strength(aux_rows, f4)
    if not complete[KEY].equals(aux_rows[KEY]):
        raise ValueError("Auxiliary metadata alignment changed")
    annotations = moa_annotations(args.moa)
    components, pair_rows, aucs, population, control_checks = [], defaultdict(list), [], [], []
    for fold, (rows, panel) in enumerate(zip(all_rows, panels)):
        task = args.common / f"task{2 * fold}"
        fit = args.common / f"fold{fold}"
        rec = read_json(fit / "receipt.json")
        scale_filtered = rec["scale_filtered"]
        orig, orig_pos = (np.load(task / name, allow_pickle=False) for name in ("observed.npy", "nuisance.npy"))
        all_panel = np.load(task / "panel.npy", allow_pickle=False)
        chosen = eligible(rows, minimum, rows.n_cells.ge(200))
        combined = rows.iloc[chosen].reset_index(drop=True)
        dense, dense_pos = (np.load(task / name, allow_pickle=False) for name in ("observed_filtered.npy", "nuisance_filtered.npy"))
        other = args.common / f"task{2 * fold + 1}"
        rows_b = table(other / "rows.tsv")
        orig_b = np.load(other / "observed.npy", mmap_mode="r", allow_pickle=False)
        dense_b = np.load(other / "observed_filtered.npy", mmap_mode="r", allow_pickle=False)
        control_checks.append(dict(fold=fold, original=check_control_role(rows, rows_b, orig, orig_b, minimum, False),
                                   combined=check_control_role(combined, rows_b.iloc[chosen].reset_index(drop=True),
                                                               dense[chosen], dense_b[chosen], minimum, True)))
        components.append(dict(fold=fold, target="original", **energy(rows, orig, orig_pos, minimum, False),
                               sampling_proxy=split_noise(rows, all_panel, legacy, minimum, False)))
        components.append(dict(fold=fold, target="combined", **energy(combined, dense[chosen], dense_pos[chosen], minimum, True),
                               sampling_proxy=split_noise(combined, panel, legacy, minimum, True)))
        pair_inputs = {0.5: corrected(rows, dense, dense_pos, minimum)}
        donors = table(fit / "donors.tsv")
        reference = np.load(fit / "donors_filtered.npy", allow_pickle=False)
        supported = {0.5: int(np.load(task / "supported_filtered.npy", allow_pickle=False)[chosen].sum())}
        for dose in (0.05, 5.0):
            mask = aux_rows[GROUP].isin(rows[GROUP]) & aux_rows.dose_uM.eq(dose)
            query = complete.loc[mask].reset_index(drop=True)
            actual = aux_counts[np.flatnonzero(mask), :][:, np.searchsorted(union, panel)].toarray()
            pred, support = position_predict(donors, reference, query, labels)
            pair_inputs[dose] = corrected(query, actual, scale_filtered * pred, minimum)
            supported[dose] = int(np.count_nonzero(support[eligible(query, minimum, query.n_cells.ge(200))]))
        population.extend(dict(fold=fold, dose=dose, rows=len(table), groups=int(table[GROUP].nunique()),
                               wells=int(table[WELL].nunique()), position_supported_rows=supported[dose])
                          for dose, (table, _) in pair_inputs.items())
        for target_dose, source_dose in DOSE_PAIRS:
            tgt, obs = pair_inputs[target_dose]
            src, ref = pair_inputs[source_dose]
            result = per_anchor(tgt, obs, src, ref, labels)
            if result is not None:
                pair_rows[target_dose, source_dose].append(result)
        aucs.extend(moa_auc(*pair_inputs[0.5], annotations))
        print(f"completed Tahoe diagnostics for fold {fold}", flush=True)
    empty = pd.DataFrame(columns=[GROUP, "pairs", "strength"])
    pairs = []
    for dose_pair in DOSE_PAIRS:
        frames = pair_rows[dose_pair]
        using = pd.concat(frames, ignore_index=True) if frames else empty
        for label in ("all", "weak", "detectable", "strong", "unlabeled"):
            at = using if label == "all" else using.loc[using.strength.eq(label)].reset_index(drop=True)
            pairs.append(summarize_pairs(at, *dose_pair, label))
    args.out.mkdir(parents=True, exist_ok=False)
    moa_result = summarize_moa(aucs)
    write_json(args.out / "ceilings.json", dict(components=components, component_summary=energy_summary(components),
               repeats=pairs, moa=moa_result, population=population, dmsorole_max_centered_abs_difference=control_checks,
               bootstrap_draws=REPEATS, bootstrap_seed=SEED, protocol_sha256=sha256(args.protocol),
               closure="total=position_removed+well_shared+remaining_including_sampling; combined evaluation starts at corrected_total",
               sampling="legacy cell-mean-log panel intersection; separate scale and never subtracted from exact closure",
               inputs=dict(common_receipt_sha256=sha256(args.common / "receipt.json"),
                           aux_receipt_sha256=sha256(args.assets / "doses/source/receipt.json"),
                           moa_sha256=sha256(args.moa)),
               sampling_inputs={str(args.halves_receipt): sha256(args.halves_receipt), str(args.half_genes): sha256(args.half_genes)},
               repeats_conditioning="whole group bootstrap with frozen parent-pool centers and matched different-plate different-position comparisons",
               moa_conditioning="conditional AUC with frozen target vectors and candidates on matching donor plate, clustered by anchor group"))
    print(json.dumps({"replicates": [x for x in pairs if x["stratum"] in {"all", "strong"}], "moa": moa_result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "common", "coordinates", "config", "out", "protocol", "halves", "halves_receipt", "half_genes", "strata", "moa"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--moa_only", action="store_true")
    args = parser.parse_args()
    (run_moa if args.moa_only else run)(args)


if __name__ == "__main__":
    main()
