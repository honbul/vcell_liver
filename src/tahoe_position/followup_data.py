"""Stage the prespecified Tahoe all-dose nuisance and matched response targets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.utils.extmath import randomized_svd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, POOL, load_data, table
from src.r13_optimization.objectives import center, row_products

from .ceilings import attach_strength, eligible, macro_weights
from .study import position_predict, positions
from .targets import GAINS, coding_tokens, donors_for, metric_for, panel_for


DOSES = (0.05, 0.5, 5.0)
PROTOCOL_SHA = "a8077df69d750f93db55a26bf3847df9dcd9dfecba2bef4d3d75b078b7b6eb00"
MIN_GROUPS = 3


def check_protocol(path):
    if sha256(path) != PROTOCOL_SHA:
        raise ValueError("Follow-up protocol differs from the committed version")


def pack_donors(rows, counts, panel):
    return donors_for(rows, counts[:, panel].toarray(), MIN_GROUPS)


def fit_gain(rows, counts, inner, labels):
    losses = np.zeros((2, len(DOSES), len(GAINS)))
    for inner_fold in range(2):
        fit = inner != inner_fold
        train = rows.loc[fit].reset_index(drop=True)
        panel = panel_for(counts[fit], 2000, None)
        donor, ref = pack_donors(train, counts[fit], panel)
        val = rows.loc[~fit].reset_index(drop=True)
        truth = counts[~fit][:, panel].toarray()
        prediction, _ = position_predict(donor, ref, val, labels)
        for d, dose in enumerate(DOSES):
            idx = eligible(val, MIN_GROUPS, val.dose_uM.eq(dose) & val.n_cells.ge(200))
            metric = metric_for(val.iloc[idx].reset_index(drop=True), MIN_GROUPS, True)
            y, p = center(metric, truth[idx]), center(metric, prediction[idx])
            yy, yp, pp = row_products(y, y), row_products(y, p), row_products(p, p)
            squared = yy - 2 * GAINS[:, None] * yp + GAINS[:, None] ** 2 * pp
            losses[inner_fold, d] = np.average(np.sqrt(squared.clip(0)), axis=1, weights=metric.primary_weight)
    mean = losses.mean(axis=(0, 1))
    return float(GAINS[mean.argmin()]), mean.tolist(), losses.tolist()


def corrected(rows, value, position):
    idx = eligible(rows, MIN_GROUPS, rows.n_cells.ge(200))
    active = rows.iloc[idx].reset_index(drop=True)
    return active, center(metric_for(active, MIN_GROUPS, True), value[idx] - position[idx])


def axes(rows, values, columns):
    dim = 16
    weight = np.sqrt(macro_weights(rows))[:, None]
    return randomized_svd(weight * values[:, columns], n_components=dim, n_iter=7,
                          random_state=130113)[2].T.astype(np.float32)


def file_receipt(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in folder.rglob("*") if p.is_file()}


def load_cache(root, protocol):
    check_protocol(protocol)
    info = read_json(root / "receipt.json")
    if info["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Cache differs from the committed protocol")
    for name, digest in info["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Cache file changed: {name}")
    return info


def array(folder, name):
    return np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def save(folder, **arrays):
    for name, value in arrays.items():
        np.save(folder / f"{name}.npy", value, allow_pickle=False)


def stage(args):
    check_protocol(args.protocol)
    prev_receipt = read_json(args.previous / "receipt.json")
    freeze = args.previous / "position_freeze.json"
    if sha256(freeze) != prev_receipt["position_freeze_sha256"]:
        raise ValueError("Prior position freeze changed")
    prev_fit = read_json(freeze)
    for name, digest in prev_fit["files"].items():
        if sha256(args.previous / name) != digest:
            raise ValueError(f"Prior fit artifact changed: {name}")
    split = read_json(args.assets / "partition/split.json")
    fold_of = {int(g): int(f) for g, f in split["outer"].items()}
    labels = positions(args.coordinates).barcode_position
    if sha256(args.previous / "f4_labels.tsv") != prev_receipt["files"]["f4_labels.tsv"]:
        raise ValueError("Prior F4 labels changed")
    strength = table(args.previous / "f4_labels.tsv")
    symbols = table(args.genes).set_index("token_id").gene_symbol
    coding = coding_tokens(args.gtf, args.genes)
    aux, aux_counts = load_data(
        args.assets / "doses/source", "treated", coding,
        mask=lambda frame: (frame[GROUP].isin(fold_of) & frame.dose_uM.isin((0.05, 5.0)) & frame.time_h.eq(24)).to_numpy(),
    )
    primary, primary_counts = load_data(
        args.primary, "treated", coding,
        mask=lambda frame: (frame[GROUP].isin(fold_of) & frame.dose_uM.eq(0.5) & frame.time_h.eq(24)).to_numpy(),
    )
    observed = pd.concat((primary, aux), ignore_index=True)
    counts = vstack((primary_counts, aux_counts), format="csr")
    del primary_counts, aux_counts
    observed[GROUP] = observed[GROUP].astype(int)
    observed["barcode_position"] = labels.loc[observed[KEY[0]]].to_numpy()
    observed = attach_strength(observed, strength)
    if not observed[GROUP].isin(fold_of).all():
        raise ValueError("Unexpected compound outside the historical partition")
    fold_id = observed[GROUP].map(fold_of).to_numpy()
    args.out.mkdir(parents=True, exist_ok=False)
    receipt_rows = []
    for fold in range(3):
        dest = args.out / f"fold{fold}"
        dest.mkdir()
        hist_fit = args.previous / f"fold{fold}"
        hist_task = args.previous / f"task{2 * fold}"
        for name in ("rows.tsv", "observed_filtered.npy"):
            if sha256(hist_task / name) != prev_receipt["files"][f"task{2 * fold}/{name}"]:
                raise ValueError(f"Prior score input changed: {name}")
        train_mask = fold_id != fold
        train = observed.loc[train_mask].reset_index(drop=True)
        train_count = counts[train_mask]
        inner = train[GROUP].map({int(g): int(f) for g, f in split["inner"][str(fold)].items()})
        if inner.isna().any():
            raise ValueError("Incomplete inner group assignment")
        gain, loss, loss_by_inner_dose = fit_gain(train, train_count, inner.to_numpy(), labels)
        panel = array(hist_fit, "panel_filtered")
        panel_columns = pd.Index(coding).get_indexer(panel)
        if np.min(panel_columns) < 0:
            raise ValueError("Historical filtered panel contains genes outside the coding set")
        donor, ref = pack_donors(train, train_count, panel_columns)
        hist_donor, hist_ref = table(hist_fit / "donors.tsv"), array(hist_fit, "donors_filtered")
        hist_gain = prev_fit["folds"][fold]["scale_filtered"]
        # The rank basis sees eligible train compounds only; auxiliary nuisance donors exclude this fold.
        donor_r, donor_c = corrected(train, train_count[:, panel_columns].toarray(),
                                     np.zeros((len(train), len(panel)), dtype=np.float32))
        train_position = np.zeros_like(donor_c, dtype=np.float32)
        inner_active = donor_r[GROUP].map({int(g): int(f) for g, f in split["inner"][str(fold)].items()})
        for k in range(2):
            use = inner.ne(k).to_numpy()
            fit_rows, fit_ref = pack_donors(train.loc[use].reset_index(drop=True), train_count[use], panel_columns)
            query_mask = inner_active.eq(k).to_numpy()
            train_position[query_mask] = gain * position_predict(
                fit_rows, fit_ref, donor_r.loc[query_mask], labels,
            )[0]
        _, train_corrected = corrected(donor_r, donor_c, train_position)
        overlap = np.flatnonzero(np.isin(panel, array(hist_fit, "panel")))
        all_columns = np.arange(len(panel))
        save(dest, panel=panel, overlap=overlap, basis=axes(donor_r, train_corrected, all_columns),
             basis_overlap=axes(donor_r, train_corrected, overlap))
        pd.DataFrame({"token": panel, "symbol": symbols.loc[panel].to_numpy()}).to_csv(
            dest / "panel.tsv", sep="\t", index=False,
        )
        score_mask = ~train_mask
        query = observed.loc[score_mask].reset_index(drop=True)
        q_values = counts[score_mask][:, panel_columns].toarray()
        hist_rows = table(hist_task / "rows.tsv")
        index = pd.MultiIndex.from_frame(query[KEY]).get_indexer(pd.MultiIndex.from_frame(hist_rows[KEY]))
        if np.any(index < 0) or np.any(query.dose_uM.to_numpy()[index] != 0.5):
            raise ValueError("Unable to recover the exact historical 0.5 µM cohort")
        # Other doses have their own assay pools; at 0.5 µM use the archived opposing-DMSO rows.
        keep = query.dose_uM.ne(0.5).to_numpy()
        keep[index] = True
        q_values[index] = array(hist_task, "observed_filtered")
        query, q_values = query.loc[keep].reset_index(drop=True), q_values[keep]
        p_all, support = position_predict(donor, ref, query, labels)
        p_old, support_old = position_predict(hist_donor, hist_ref, query, labels)
        active, target_all = corrected(query, q_values, gain * p_all)
        compare, target_old = corrected(query, q_values, hist_gain * p_old)
        if not active[KEY].equals(compare[KEY]):
            raise ValueError("Position regimes produced separate eligible cohorts")
        ix = pd.MultiIndex.from_frame(query[KEY]).get_indexer(pd.MultiIndex.from_frame(active[KEY]))
        active.to_csv(dest / "rows.tsv", sep="\t", index=False)
        save(dest, target_all=target_all.astype(np.float32), target_old=target_old.astype(np.float32))
        levels = []
        for dose in DOSES:
            ix_dose = np.flatnonzero(active.dose_uM.eq(dose))
            metric = metric_for(active.iloc[ix_dose].reset_index(drop=True), MIN_GROUPS, True)
            baseline = center(metric, q_values[ix[ix_dose]])
            scores = [np.average(np.sqrt(row_products(y, y)), weights=metric.primary_weight)
                      for y in (baseline, target_old[ix_dose], target_all[ix_dose])]
            levels.append(dict(dose=dose, groups=int(active.iloc[ix_dose][GROUP].nunique()), rows=len(ix_dose),
                               supported_all=int(np.count_nonzero(support[ix[ix_dose]])),
                               supported_old=int(np.count_nonzero(support_old[ix[ix_dose]])),
                               b0=float(scores[0]), old=float(scores[1]), all_dose=float(scores[2])))
        receipt_rows.append(dict(fold=fold, gain=gain, historical_gain=hist_gain, inner_loss=loss,
                                 inner_loss_by_dose=loss_by_inner_dose, levels=levels))
        print(f"staged fold {fold}, all-dose gain {gain}", flush=True)
    write_json(args.out / "receipt.json", dict(protocol_sha256=PROTOCOL_SHA, folds=receipt_rows,
               source_sha256=sha256(args.assets / "doses/source/receipt.json"), split_sha256=sha256(args.assets / "partition/split.json"),
               primary_sha256=sha256(args.primary / "receipt.json"),
               previous_sha256=sha256(args.previous / "receipt.json"), gtf_sha256=sha256(args.gtf), files=file_receipt(args.out)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "primary", "previous", "coordinates", "genes", "gtf", "protocol", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    stage(parser.parse_args())


if __name__ == "__main__":
    main()
