"""Frozen project-seen replay and limited physical-well-separated cell transfer."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, group_weights, load_data, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align

from .candidates import Candidate, Evaluator
from .data import Fit, Source
from .develop import CalibratorFits, settings
from .infer import FrozenPredictions
from .score import measure
from .split import common_rows
from .stage import load_staged


def stable_candidate(candidates):
    specs = [Candidate(**record) for record in candidates]

    def key(spec):
        return replace(spec, epoch=0, horizon=0)

    votes = Counter(map(key, specs))
    selected = min(votes, key=lambda x: (-votes[x], settings(x)))
    settings_median = int(np.median([x.epoch for x in specs if key(x) == selected]))
    return asdict(replace(selected, horizon=100, epoch=settings_median))


def modal(values):
    votes = Counter(values)
    return min(votes, key=lambda x: (-votes[x], str(x)))


def replay_choices(primary_path, role):
    choices = [read_json(primary_path / f"outer{k}" / role / "selection.json") for k in range(3)]
    picks = {arm: stable_candidate([c["winners"][arm]["candidate"] for c in choices])
             for arm in ("B_abs", "B_contrast", "Context", "Structure", "Loss")}
    pick_combo = stable_candidate([c["joint"]["raw"] for c in choices])
    shift = [c["shift"]["Shift"] for c in choices]
    shift_family = modal((s["family"], s["penalty"]) for s in shift)
    shifts = {
        "Shift": {"base": stable_candidate([s["base"] for s in shift]), "family": shift_family[0],
                  "penalty": shift_family[1]},
    }
    jtype = modal((c["joint"]["calibrated"]["family"], c["joint"]["calibrated"]["penalty"]) for c in choices)
    shifts["J2_calibrated"] = dict(base=pick_combo, family=jtype[0], penalty=jtype[1])
    picks["J2_raw"] = pick_combo
    return dict(candidates=picks, calibrated=shifts,
                integrated=modal(c["joint"]["selected"] for c in choices), source="outer inner selections only")


class FullSource(Source):
    def __init__(self, cfg, assets, role, transfer=False):
        super().__init__(cfg, assets, 0, role)
        receipt = read_json(assets / "staged/receipt.json")
        r, x = load_staged(assets / "staged/outer0", "score", receipt["genes"])
        self.fit_rows = pd.concat([self.fit_rows, r], ignore_index=True)
        self.fit_counts = sp.vstack([self.fit_counts, x], format="csr")
        self.transfer = transfer

    def check_fit_query(self, fit, query):
        if not self.transfer or set(fit.rows[GROUP]).isdisjoint(query.rows[GROUP]):
            return super().check_fit_query(fit, query)
        if (set(fit.rows[KEY[0]]) & set(query.rows[KEY[0]]) or
                set(fit.rows[LINE]) & set(query.rows[LINE]) or
                not set(query.rows[ENTITY]) <= set(fit.rows[ENTITY])):
            raise ValueError("Strict new-line transfer fit used a query well/line or unknown entity")


def screen_transfer(rows, line, minimum):
    per_line = rows.loc[rows[LINE].eq(line)].copy()
    others = rows.loc[rows[LINE].ne(line)]
    known = {drug: set(group[KEY[0]]) for drug, group in others.groupby(ENTITY)}
    per_line = per_line.loc[
        [bool(known.get(entity, set()) - {well}) for entity, well in zip(per_line[ENTITY], per_line[KEY[0]])]]
    for _, plate in per_line.groupby("plate_id"):
        groups = [tuple(grp.index) for _, grp in plate.groupby(GROUP)]
        if len(groups) < minimum:
            continue
        for ids in product(*groups):
            test = plate.loc[list(ids)]
            forbidden = set(test[KEY[0]])
            if all(known.get(entity, set()) - forbidden for entity in test[ENTITY]):
                train = others.loc[~others[KEY[0]].isin(forbidden)].reset_index(drop=True)
                return train, test.reset_index(drop=True)
    return None, None


def predict_secondary(cfg, assets, primary, out, device, mode):
    out.mkdir(parents=True, exist_ok=False)
    hashed = {}
    for role in ("A", "B"):
        source = FullSource(cfg, assets, role, transfer=mode == "transfer")
        choice = replay_choices(primary, role)
        tasks = {}
        if mode == "replay":
            query = common_rows(source.r11 / "final", source.r11 / "source", cfg["minimum_groups"])
            panel = table(source.r11 / "development/final_panel.tsv").token_id.to_numpy()
            tasks["replay"] = (source.fit_rows, query, panel)
        else:
            for cell in ("CVCL_1098", "CVCL_0366"):
                fit_rows, query = screen_transfer(source.fit_rows, cell, cfg["minimum_groups"])
                if fit_rows is not None:
                    tasks[cell] = (fit_rows, query, None)
        for label, (train, query, panel) in tasks.items():
            folder = out / label / role
            folder.mkdir(parents=True)
            (folder / "predictions").mkdir()
            (folder / "models").mkdir()
            fit = Fit(source, train, query, panel=panel)
            cross = CalibratorFits(source, fit, train, device, cfg["split_seed"] + 1900)
            results = FrozenPredictions(folder, fit, Evaluator(fit, device), cfg["training"], cross, choice)
            for arm, spec in choice["candidates"].items():
                results.train(arm, Candidate(**spec), confirm=True)
            for arm, calibration in choice["calibrated"].items():
                base = Candidate(**calibration["base"])
                if arm == "J2_calibrated":
                    reference = "J2_raw"
                else:
                    reference = "Shift_unadjusted"
                    results.train(reference, base, confirm=True)
                results.apply(arm, reference, calibration)
            results.alias("Integrated", choice["integrated"])
            np.save(folder / "panel.npy", fit.panel)
            fit.query.rows.to_csv(folder / "rows.tsv", sep="\t", index=False)
            write_json(folder / "selection.json", choice)
            write_json(folder / "prediction_index.json", results.entries)
            write_json(folder / "runtime.json", results.ledger)
            for file in folder.rglob("*"):
                if file.is_file():
                    hashed[str(file.relative_to(out))] = sha256(file)
    write_json(out / "freeze.json", dict(mode=mode, files=hashed, primary_path=str(primary.resolve()),
              provenance="Candidates fixed from primary inner selections independently by role."))


def score_secondary(cfg, assets, out, mode):
    frozen = read_json(out / "freeze.json")
    if frozen["mode"] != mode:
        raise ValueError("Secondary scoring mode differs from frozen run")
    if (out / "arm_metrics.tsv").exists():
        raise ValueError("Secondary predictions already scored")
    for file, expected in frozen["files"].items():
        if sha256(out / file) != expected:
            raise ValueError(f"Secondary frozen output changed: {file}")
    r11 = Path(cfg["r11_assets"])
    if mode == "replay":
        observed_rows, observed = load_data(r11 / "final", "treated")
    else:
        source = FullSource(cfg, assets, "A", transfer=True)
        observed_rows, observed = source.fit_rows, source.fit_counts
    reports = []
    for folder in sorted(out.glob("*/[AB]")):
        role = folder.name
        opposite = "B" if role == "A" else "A"
        rows = table(folder / "rows.tsv")
        panel = np.load(folder / "panel.npy")
        indexes = align(rows, observed_rows)
        y = observed[indexes].toarray() if mode == "replay" else observed[indexes][:, panel].toarray()
        ref, counts = load_data(r11 / "source", role, panel)
        other, views = load_data(r11 / "source", opposite, panel)
        ctrl_idx = align(rows, ref, ["plate_id", LINE])
        eval_idx = align(rows, other, ["plate_id", LINE])
        own = counts[ctrl_idx].toarray()
        score = y - views[eval_idx].toarray()
        expected = table(folder / "rows.tsv").control_well_id.to_numpy()
        if not np.array_equal(expected, ref.iloc[ctrl_idx][KEY[0]]) or np.any(
                expected == other.iloc[eval_idx][KEY[0]].to_numpy()):
            raise ValueError("Secondary frozen role join inconsistent")
        cc = Contrast(rows, cfg["minimum_groups"])
        w = group_weights(rows)
        entries = read_json(folder / "prediction_index.json")
        np.savez(folder / "score_truth.npz", absolute=y, measured_delta=score, own_control=own, panel=panel)
        for name, paths in entries.items():
            metrics = []
            for file in paths.values():
                values = measure(rows, cc, np.load(folder / file), score, own, y)
                metrics.append({k: float(np.average(v, weights=w)) for k, v in values.items()})
            reports.append(dict(mode=mode, scope=folder.parent.name, role=role, arm=name,
                                n_wells=rows[KEY[0]].nunique(), n_groups=rows[GROUP].nunique(), rows=len(rows),
                                seeds=len(paths), **{f: np.mean([x[f] for x in metrics]) for f in metrics[0]}))
    pd.DataFrame(reports).to_csv(out / "arm_metrics.tsv", sep="\t", index=False)
    write_json(out / "score_receipt.json", dict(freeze_sha256=sha256(out / "freeze.json"),
               truth_sha256={str(p.relative_to(out)): sha256(p) for p in out.glob("*/[AB]/score_truth.npz")},
               metric_sha256=sha256(out / "arm_metrics.tsv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--primary-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--mode", choices=["replay", "transfer"], required=True)
    parser.add_argument("command", choices=["predict", "score"])
    args = parser.parse_args()
    cfg = config(args.config)
    torch.set_num_threads(4)
    primary_manifest = read_json(args.primary_freeze)
    primary = Path(primary_manifest["predictions"])
    for name, expected in primary_manifest["predictions_receipts"].items():
        if sha256(primary / name) != expected:
            raise ValueError("Primary inference does not match its pre-scoring freeze")
    if args.command == "predict":
        predict_secondary(cfg, args.assets, primary, args.out, args.device, args.mode)
    else:
        score_secondary(cfg, args.assets, args.out, args.mode)


if __name__ == "__main__":
    main()
