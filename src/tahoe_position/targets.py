"""Prepare fold-frozen position nuisance and the two agreed assay panels."""

import argparse
import gzip
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align
from src.r12_multiaxis.data import select_panel
from src.r12_multiaxis.split import eligible_pools
from src.r12_multiaxis.stage import load_staged
from src.r13_optimization.objectives import GainScores, center
from src.r13_optimization.score import controls

from .study import checked_history, position_predict, positions


GAINS = np.arange(9) / 8


def metric_for(rows, minimum, cell_weighted=False):
    metric = Contrast(rows, minimum)
    if cell_weighted:
        n = rows.n_cells.to_numpy(dtype=np.float64)
        metric.centers = [w * n[ix] for w, ix in zip(metric.centers, metric.pools)]
        for name in ("all_weight", "primary_weight"):
            w = getattr(metric, name) * n
            norm = np.bincount(metric.inverse, weights=w)[metric.inverse]
            setattr(metric, name, np.divide(w, norm, out=np.zeros_like(w), where=norm > 0))
    return metric


def coding_tokens(path, metadata):
    pattern = re.compile(r'gene_id "([^"]+)".*gene_biotype "protein_coding"')
    coding = set()
    with gzip.open(path, "rt") as stream:
        for line in stream:
            fields = line.split("\t", 8)
            if len(fields) == 9 and fields[2] == "gene" and fields[0] != "MT":
                match = pattern.search(fields[8])
                if match:
                    coding.add(match[1])
    genes = table(metadata)
    keep = genes.ensembl_id.isin(coding) & ~genes.gene_symbol.str.startswith("MT-")
    tokens = np.unique(np.r_[0:3, genes.loc[keep, "token_id"].to_numpy()]).astype(int)
    if len(tokens) < 10_000:
        raise ValueError("Incomplete Ensembl protein-coding vocabulary join")
    return tokens


def f4_labels(folder):
    inputs = {"snu423": "CVCL_0366", "hepg2c3a": "CVCL_1098"}
    cases = []
    for name, cell_line in inputs.items():
        frame = table(folder / f"{name}_effect_strength.tsv")
        frame = frame.loc[~frame.is_dmso].copy()
        frame["strength_code"] = np.select([frame.resp_strong, frame.resp_strict], [2, 1], default=0)
        condition = frame.groupby(["drug", "conc_uM"]).strength_code.transform("max")
        kinds = np.array(["weak", "detectable", "strong"])
        cases.append(pd.DataFrame({KEY[0]: frame["sample"], LINE: cell_line, "dose_uM": frame.conc_uM,
                                   "strength": kinds[condition], "strength_well": kinds[frame.strength_code]}))
    result = pd.concat(cases, ignore_index=True)
    if result.duplicated(KEY).any():
        raise ValueError("Duplicate F4 treatment labels")
    return result


def panel_for(values, size, allowed):
    return select_panel(values, size) if allowed is None else allowed[select_panel(values[:, allowed], size)]


def donors_for(rows, values, minimum):
    metric = Contrast(rows, minimum)
    return rows.iloc[metric.active], center(metric, values)[metric.active]


def fit_position(rows, counts, query, labels, inner, size, minimum, allowed):
    losses, weight_sum = np.zeros(len(GAINS)), 0.0
    for k in (0, 1):
        tr = rows.loc[inner.ne(k)].reset_index(drop=True)
        va = eligible_pools(rows.loc[inner.eq(k)], minimum)
        bound = counts[align(tr, rows)]
        panel = panel_for(bound, size, allowed)
        target = counts[align(va, rows)][:, panel].toarray()
        donor, ref = donors_for(tr, bound[:, panel].toarray(), minimum)
        pred, _ = position_predict(donor, ref, va, labels)
        scores = GainScores(va, target, np.zeros_like(target), minimum)
        weight = scores.metric.primary_weight.sum()
        losses += weight * scores.points(pred, GAINS)[0]
        weight_sum += weight
    scale = float(GAINS[np.argmin(losses)])
    panel = panel_for(counts, size, allowed)
    donor, ref = donors_for(rows, counts[:, panel].toarray(), minimum)
    pred, supported = position_predict(donor, ref, query, labels)
    return panel, scale * pred, supported > 0, donor, ref.astype(np.float32), scale, (losses / weight_sum).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("assets", "history", "coordinates", "config", "protocol", "gtf", "f4", "common"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    r11 = Path(cfg["r11_assets"])
    stage = args.assets / "staged"
    rec = read_json(stage / "receipt.json")
    historical_freeze = read_json(args.history / "freeze.json")
    if historical_freeze["stage_receipt_sha256"] != sha256(stage / "receipt.json"):
        raise ValueError("Historical assay stage changed")
    split = read_json(args.assets / "partition/split.json")
    book = positions(args.coordinates)
    allowed = coding_tokens(args.gtf, r11 / "metadata/genes.tsv")
    args.common.mkdir(parents=True, exist_ok=False)
    fit_files, fold_results = [], []
    for fold in range(3):
        dest = args.common / f"fold{fold}"
        dest.mkdir()
        query, original_panel = checked_history(args, fold, "A")
        rows, values = load_staged(stage / f"outer{fold}", "fit", rec["genes"])
        if not np.array_equal(book.loc[rows[KEY[0]], "plate"], rows.plate_id):
            raise ValueError("A fit row has a different position release plate")
        inner = rows[GROUP].map({int(k): v for k, v in split["inner"][str(fold)].items()})
        if inner.isna().any():
            raise ValueError("Incomplete historical inner membership")
        result = {"fold": fold, "fit_groups": int(rows[GROUP].nunique())}
        for suffix, candidates in (("", None), ("_filtered", allowed)):
            panel, pred, support, donor, centered, scale, losses = fit_position(
                rows, values, query, book.barcode_position, inner, cfg["panel_size"], cfg["minimum_groups"], candidates,
            )
            if suffix == "" and not np.array_equal(panel, original_panel):
                raise ValueError("Original panel changed")
            for name, array in (("panel", panel), ("nuisance", pred), ("supported", support)):
                np.save(dest / f"{name}{suffix}.npy", array, allow_pickle=False)
            if suffix:
                donor.to_csv(dest / "donors.tsv", sep="\t", index=False)
                np.save(dest / "donors_filtered.npy", centered, allow_pickle=False)
                result["intersection"] = len(np.intersect1d(panel, original_panel))
            result.update({f"scale{suffix}": scale, f"loss{suffix}": losses, f"supported_rows{suffix}": int(support.sum())})
        write_json(dest / "receipt.json", result)
        fold_results.append(result)
        fit_files.extend(dest.iterdir())
        print(f"frozen nuisance fold {fold}", flush=True)
    freeze_path = args.common / "position_freeze.json"
    write_json(freeze_path, dict(
        gains=GAINS.tolist(), folds=fold_results, protocol_sha256=sha256(args.protocol), gtf_sha256=sha256(args.gtf),
        historical_freeze_sha256=sha256(args.history / "freeze.json"), coordinate_sha256=sha256(args.coordinates / "receipt.json"),
        files={str(p.relative_to(args.common)): sha256(p) for p in fit_files},
    ))

    strata = f4_labels(args.f4)
    strata.to_csv(args.common / "f4_labels.tsv", sep="\t", index=False)
    outcomes = []
    for fold in range(3):
        src = args.common / f"fold{fold}"
        origin, counts = load_staged(stage / f"outer{fold}", "score", rec["genes"])
        query_a, _ = checked_history(args, fold, "A")
        for ri, role in enumerate("AB"):
            out = args.common / f"task{2 * fold + ri}"
            out.mkdir()
            rows, _ = checked_history(args, fold, role)
            if not query_a[KEY + [GROUP]].equals(rows[KEY + [GROUP]]):
                raise ValueError("The frozen physical query differs across roles")
            rows = rows.merge(strata, on=KEY + ["dose_uM"], how="left", validate="one_to_one", sort=False)
            rows[["strength", "strength_well"]] = rows[["strength", "strength_well"]].fillna("unlabeled")
            rows.to_csv(out / "rows.tsv", sep="\t", index=False)
            index = align(rows, origin)
            for suffix in ("", "_filtered"):
                for name in ("panel", "nuisance", "supported"):
                    (out / f"{name}{suffix}.npy").hardlink_to(src / f"{name}{suffix}.npy")
                panel = np.load(src / f"panel{suffix}.npy", allow_pickle=False)
                _, opposite = controls(r11, rows, role, panel)
                observed = counts[index][:, panel].toarray() - opposite
                np.save(out / f"observed{suffix}.npy", observed.astype(np.float32), allow_pickle=False)
            outcomes.extend(out.iterdir())
    write_json(args.common / "receipt.json", dict(
        position_freeze_sha256=sha256(freeze_path),
        f4_sources={p.name: sha256(p) for p in sorted(args.f4.glob("*_effect_strength.tsv"))},
        files={str(p.relative_to(args.common)): sha256(p) for p in outcomes + [args.common / "f4_labels.tsv"]},
    ))


if __name__ == "__main__":
    main()
