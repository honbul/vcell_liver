"""Refit selected recipes on the complete outer training data and freeze predictions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP
from src.r12_multiaxis.data import Fit

from .tune import FAMILIES, checked_forecast, implementation, location


def competing_families(settings):
    return (*FAMILIES[1:], *settings.get("extra_families", []))


def choose(winners, baseline, settings):
    reference = baseline["contrast_by_fold"]
    promising = []
    for family in competing_families(settings):
        candidate = winners[family]
        if candidate["gain"] <= 0 or candidate["specification"].get("permute"):
            continue
        differences = [candidate["contrast_by_fold"][k] - ref for k, ref in reference.items()]
        if max(differences) < -settings["positive_tolerance"]:
            promising.append((family, candidate))
    return sorted(promising, key=lambda item: (item[1]["contrast"], item[1]["gain"], item[0]))


def receipt(directory, family, fold, role, protocol, config_path, partition):
    path = directory / "selection.json"
    report = read_json(path)
    if (report["family"], report["fold"], report["role"], report["protocol"]) != (family, fold, role, protocol):
        raise ValueError("Tuning receipt belongs to a different study slice")
    checks = {
        config_path: report["config_sha256"],
        directory / "grid.tsv": report["grid_sha256"],
        **{Path(p): h for p, h in report["code"].items()},
    }
    if any(sha256(path) != expected for path, expected in checks.items()):
        raise ValueError(f"Changed tuning code, manifest, or grid for {family}")
    if report["partition_sha256"] != partition:
        raise ValueError("Tuning used a different compound partition")
    return report


def record(directory, fit, winners, control, settings):
    ordered = choose(winners, control["best"], settings)
    selection = ordered[0][0] if ordered else "anchor"
    ensemble = [name for name, _ in ordered[:settings["ensemble_size"]]]
    index = defaultdict(list)
    files = {}

    def save(name, pred):
        if pred.shape != fit.offset.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid array for {name}")
        path = directory / f"{name}.{len(index[name])}.npy"
        np.save(path, np.asarray(pred, dtype=np.float32), allow_pickle=False)
        index[name].append(path.name)
        files[path.name] = sha256(path)

    save("B0", np.zeros_like(fit.offset))
    save("anchor", fit.offset)
    save("B_abs", implementation("control").predict(fit, control["best_absolute"]["specification"]))
    cached = {}
    for family, winner in winners.items():
        module = implementation(family)
        spec = winner["specification"]
        seeds = [42, 43, 44] if family == "neural" else [None]
        cached[family] = []
        for seed in seeds:
            params = dict(spec)
            if seed is not None:
                params["seed"] = seed
            pred = next(checked_forecast(module, fit, [params]))[1]
            prediction = fit.offset + winner["gain"] * (pred - fit.offset)
            cached[family].append(prediction)
            save(family, prediction)
    # Ensemble neural seeds within a family so it has the same weight as a deterministic family.
    for pred in cached.get(selection, [fit.offset]):
        save("selector", pred)
    average = np.mean([np.mean(cached[name], axis=0) for name in ensemble], axis=0) if ensemble else fit.offset
    save("ensemble", average)
    return dict(index), files, selection, ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", type=int, choices=range(6), required=True)
    parser.add_argument("--protocol", choices=["compound", "scaffold"], default="compound")
    args = parser.parse_args()
    fold, role = args.task // 2, "AB"[args.task % 2]
    cfg = config(args.config)
    study = config(args.study)
    src = location(cfg, args.assets, fold, role, args.protocol)
    family_names = competing_families(study["selection"])
    reports = {
        family: receipt(args.tuned / family / f"task{args.task}", family, fold, role,
                        args.protocol, args.config, src.split["manifest_sha256"])
        for family in ("control", *family_names)
    }
    winners = {family: reports[family]["best_nonzero"] for family in family_names}
    fit = Fit(src, src.fit_rows, src.query_rows, observe_query=False)
    if fit.query.treated is not None or set(fit.train.rows[GROUP]).intersection(fit.query.rows[GROUP]):
        raise ValueError("Unsafe outer fit/query construction")
    args.out.mkdir(parents=True, exist_ok=False)
    index, files, selected, ensemble = record(args.out, fit, winners, reports["control"], study["selection"])
    fit.query.rows.to_csv(args.out / "rows.tsv", sep="\t", index=False)
    np.save(args.out / "panel.npy", fit.panel, allow_pickle=False)
    files.update({name: sha256(args.out / name) for name in ("rows.tsv", "panel.npy")})
    active_sources = {*family_names, "control", "predict", "tune", "objectives"}
    implementations = {str(p): sha256(p) for p in sorted(Path("src/r13_optimization").glob("*.py"))
                       if p.stem in active_sources}
    manifest = dict(protocol=args.protocol, fold=fold, role=role,
                    partition_sha256=src.split["manifest_sha256"], study_sha256=sha256(args.study),
                    config_sha256=sha256(args.config), files=files, predictions=index,
                    chosen=selected, ensemble=ensemble, winners=winners, selected_pool=reports["control"]["best_absolute"],
                    tune_receipts={family: sha256(args.tuned / family / f"task{args.task}" / "selection.json")
                                   for family in ("control", *family_names)},
                    implementation=implementations)
    write_json(args.out / "receipt.json", manifest)
    print(json.dumps(dict(fold=fold, role=role, selected=selected, ensemble=ensemble)), flush=True)


if __name__ == "__main__":
    main()
