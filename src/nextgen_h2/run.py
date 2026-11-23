"""Run preregistered inner scaffold fits and emit outcome-blind outer forecasts."""
import argparse
import json
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vc_build.external import native_sensitivity
from vc_build.io import publish_json, sha256
from vc_build.v0plus_cv import molecular_inputs
from vc_build.v0plus_neural import load_auxiliary

from .assets import (
    AUXILIARY, COLD, HALLMARK, KEY, LIMORE, PRIOR, RECIPES, TARGET, canonical_registry, parquet_lock,
)
from .kernels import stress_mask
from .metrics import Evaluator, stack
from .model import Baseline, Fit, Fusion, Inputs, PENALTIES, Representation, WEIGHTS, molecular_features

TRAIN = COLD / "train.parquet"
META = ["row_id", "cell_id", "lineage_group", KEY, "source_id", "endpoint_id"]
R1_VIEWS = ("painting", "lincs")


def read_json(path):
    # Slurm dependencies can resolve before another host's latest NFS directory entry appears.
    for _ in range(20):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            time.sleep(2)
    return json.loads(path.read_text())


def base_key(base):
    return f"C1/{base.style}/{base.penalty:g}"


def load_context(assets, target_dir, device, *, external=False):
    receipt = read_json(assets / "data_receipt.json")
    targets_receipt = read_json(target_dir / "receipt.json")
    if sha256(target_dir / "targets.json") != targets_receipt["targets_sha256"]:
        raise ValueError("ChEMBL target mapping changed")
    targets = read_json(target_dir / "targets.json")
    registered = read_json(COLD / "receipt.json")
    if sha256(TRAIN) != registered["files"]["train.parquet"]:
        raise ValueError("Source response file changed")
    registry = parquet_lock(assets / "folds.parquet", receipt["files"]["folds.parquet"])
    coverage = parquet_lock(assets / "coverage.parquet", receipt["files"]["coverage.parquet"])
    pathway = parquet_lock(assets / "pathway_weights.parquet", receipt["files"]["pathway_weights.parquet"])
    profiles = {}
    for key in ["painting", *RECIPES, *("paths_" + recipe for recipe in RECIPES)]:
        prefix = "lincs_" if key in RECIPES else ""
        file = prefix + key + ".parquet"
        profiles[key] = parquet_lock(assets / file, receipt["files"][file])
    rna = parquet_lock(COLD / "rna.parquet", registered["files"]["rna.parquet"])
    edges, _, _ = molecular_inputs(PRIOR, HALLMARK, TARGET)
    aux, _ = load_auxiliary(AUXILIARY)
    all_registry = registry
    query = None
    if external:
        limore = read_json(LIMORE / "receipt.json")["artifacts"]
        external_limore = parquet_lock(LIMORE / "test_inputs/rows.parquet", limore["test_inputs/rows.parquet"]["sha256"])
        external_limore = external_limore.rename(columns={"cell_group": "lineage_group", "record_id": "row_id"})
        external_licob = parquet_lock(COLD / "query.parquet", registered["files"]["query.parquet"])
        external_gdsc = parquet_lock(assets / "gdsc_query.parquet", receipt["files"]["gdsc_query.parquet"])
        external_licob = external_licob.assign(source_id="LICOB", endpoint_id="AUC")
        external_limore = external_limore.assign(source_id="LIMORE")
        ext_rna = parquet_lock(LIMORE / "test_inputs/rna.parquet", limore["test_inputs/rna.parquet"]["sha256"])
        rna = pd.concat([rna, ext_rna.reindex(columns=rna.columns)], verify_integrity=True)
        query = pd.concat([external_limore, external_licob, external_gdsc], ignore_index=True)
        query = query.drop_duplicates("row_id").reset_index(drop=True)
        ext_registry = canonical_registry(query)
        all_registry = pd.concat([registry, ext_registry.loc[~ext_registry.index.isin(registry.index)]])
    data = Inputs(assets, molecular_features(all_registry), profiles, pathway, rna, edges, targets, aux, device)
    return data, registry, coverage, query


def subfold(registry, outer, inner=None):
    if outer not in range(3):
        raise ValueError("Invalid outer fold")
    held = registry.outer.eq(outer)
    if inner is not None:
        check = registry[f"inner{outer}"].eq(inner)
        if (held & check).any():
            raise AssertionError("Outer evaluation labels must remain sequestered")
        return registry.index[~held & ~check], registry.index[check]
    return registry.index[~held], registry.index[held]


def response(keys, labels):
    columns = None if labels else META
    return pd.read_parquet(TRAIN, columns=columns, filters=[(KEY, "in", list(keys))]).sort_values(
        "row_id", ignore_index=True)


def representations():
    for kernel in ("cos", "rbf"):
        yield Representation("painting", "cache", "raw", 0, kernel)
        for transform in ("pca", "white", "zca", "select"):
            for dimension in (32, 64, 128, 256):
                yield Representation("painting", "cache", transform, dimension, kernel)
        for recipe in RECIPES:
            for transform, dimension in (("raw", 0), ("pathway", 44), ("rank", 0), ("white", 64)):
                yield Representation("lincs", recipe, transform, dimension, kernel)


def coverage_for(coverage, config):
    if config.view == "painting":
        return coverage.painting
    return coverage["lincs_" + config.recipe]


def compare_set(coverage, view):
    if view == "painting":
        return coverage.painting
    return coverage[[f"lincs_{recipe}" for recipe in RECIPES]].all(axis=1)


def add_metrics(pairs, evaluator, model, name, prediction, modes):
    for support, keep in modes.items():
        mask = None if keep is None else evaluator.rows[KEY].map(keep).fillna(False).to_numpy()
        pairs.extend(evaluator.score(name, prediction, mask, support=support))


def config_dict(items):
    return [Representation(**item) for item in items]


def spec_dict(item):
    return Fusion(**item)


def baseline_options():
    return [Baseline(style, penalty) for style in ("molecular", "interaction") for penalty in PENALTIES]


def record_inner(stage, data, registry, coverage, outer, inner, root):
    training, validation = subfold(registry, outer, inner)
    fit = response(training, labels=True)
    hold = response(validation, labels=True)
    query = hold[META]
    engine = Fit(data, fit, query, registry=registry)
    evaluator = Evaluator(query.assign(truth=native_sensitivity(hold)), registry.scaffold_block, inner)
    pairs = []
    seed = 18407 + 201 * outer + inner

    def evaluate(model, name, modes):
        add_metrics(pairs, evaluator, model, name, engine.extract(model), modes)

    if stage == "r1base":
        for base in baseline_options():
            evaluate(engine.baseline(base), base_key(base), {"all": None})
    else:
        base = Baseline(**read_json(root / "base.json")[str(outer)])
        if stage == "r1":
            for view in R1_VIEWS:
                evaluate(engine.baseline(base, targets=False), "C0", {view + "_common": compare_set(coverage, view)})
                evaluate(engine.baseline(base), "C1", {view + "_common": compare_set(coverage, view)})
            mask_done = set()
            for config in representations():
                key = config.view + "/" + config.recipe
                common = {config.view + "_common": compare_set(coverage, config.view)}
                observed = key + "_observed"
                modes = common | {observed: coverage_for(coverage, config), "all": None}
                if key not in mask_done:
                    evaluate(engine.fusion(base, [config], Fusion("mask")), "C2/" + key, modes)
                    mask_done.add(key)
                evaluate(engine.fusion(base, [config], Fusion()), "C4/" + config.key, modes)
        elif stage == "r1controls":
            selected = read_json(root / "r1_select.json")[str(outer)]
            arms = {name: Representation(**item) for name, item in selected.items()}
            for view, config in arms.items():
                modes = {view + "_observed": coverage_for(coverage, config), view + "_all": None}
                for name, model in {
                    "C0": engine.baseline(base, targets=False),
                    "C1": engine.baseline(base),
                    "C2": engine.fusion(base, [config], Fusion("mask")),
                    "C3": engine.fusion(base, [config], Fusion(shuffle=True), seed=seed),
                    "C4": engine.fusion(base, [config], Fusion()),
                }.items():
                    evaluate(model, view + "/" + name, modes)
        elif stage == "r2":
            selected = read_json(root / "r1_gate.json")[str(outer)]
            passed = {view: Representation(**part["config"]) for view, part in selected.items() if part["pass"]}
            # If both assays miss the info gate, advance only the preregistered LINCS pathway diagnostic.
            if not passed:
                config = replace(Representation(**selected["lincs"]["config"]), transform="pathway", dimension=44)
                modes = {"lincs_path_observed": coverage_for(coverage, config), "lincs_path_all": None}
                for name, spec in (("C2", Fusion("mask")), ("C3", Fusion(shuffle=True)), ("C4", Fusion())):
                    evaluate(engine.fusion(base, [config], spec, seed=seed), "path_diag/" + name, modes)
                observed = coverage_for(coverage, config)
                _, mask, _ = engine.representations(config)
                for fraction in (0., .25, .50):
                    live = stress_mask(mask, engine.stop, fraction, seed)
                    kept = pd.Series(live[engine.stop:].cpu().numpy(), index=engine.query_drugs)
                    tag = f"path/{fraction:g}"
                    subsets = {tag + "/all": None, tag + "/originally_observed": observed,
                               tag + "/retained": kept, tag + "/masked": observed & ~kept.reindex(
                                   observed.index, fill_value=False)}
                    evaluate(engine.baseline(base), tag + "/C1", subsets)
                    evaluate(engine.fusion(base, [config], Fusion(stress=fraction), seed=seed),
                             tag + "/path_psd/0.25", subsets)
            else:
                fit_r2(engine, base, passed, coverage, seed, evaluate)
        elif stage == "r3":
            plan = read_json(root / "r2_select.json")[str(outer)]
            config = Representation(**plan["path_config"])
            trial = [config]
            chosen = plan["champion"]
            geometry = Fusion()
            if chosen and any(item["view"] == "lincs" for item in chosen["configs"]):
                trial = config_dict(chosen["configs"])
                [config] = [item for item in trial if item.view == "lincs"]
                geometry = spec_dict(chosen["spec"])
            supported = coverage_for(coverage, config)
            modes = {"match_observed": supported, "match_all": None}
            if chosen:
                incumbent = engine.fusion(base, config_dict(chosen["configs"]), spec_dict(chosen["spec"]), seed=seed)
                evaluate(incumbent, "match/incumbent", modes)
            for method, model in {
                "match/C2": engine.fusion(base, trial, Fusion("mask"), seed=seed),
                "match/C3": engine.fusion(base, trial, replace(geometry, shuffle=True), seed=seed),
                "match/geometry": engine.fusion(base, trial, geometry, seed=seed),
            }.items():
                evaluate(model, method, modes)
            for strength in (.05, .20):
                proposal = replace(geometry, match=strength)
                evaluate(engine.fusion(base, trial, proposal, seed=seed), f"match/direct/{strength:g}", modes)
                evaluate(engine.fusion(base, trial, replace(proposal, shuffle=True), seed=seed),
                         f"match/shuffled/{strength:g}", modes)
        else:
            raise ValueError(f"Unknown inner stage {stage}")
    target = root / stage / f"{outer}-{inner}"
    target.mkdir(parents=True, exist_ok=False)
    drug, cell = stack(pairs)
    drug.to_parquet(target / "drug.parquet", index=False)
    cell.to_parquet(target / "cell.parquet", index=False)
    publish_json(target / "receipt.json", {"stage": stage, "outer": outer, "inner": inner,
        "fit_rows": len(fit), "fit_drugs": len(training), "check_rows": len(hold), "check_drugs": len(validation),
        "outer_response_rows_loaded": 0, "solver": engine.notes, "sources": engine.calibration,
        "drug_sha256": sha256(target / "drug.parquet"), "cell_sha256": sha256(target / "cell.parquet")})


def fit_r2(engine, base, passed, coverage, seed, evaluate):
    for view, config in passed.items():
        observed = coverage_for(coverage, config)
        for fraction in (0., .25, .50):
            _, mask, _ = engine.representations(config)
            kept = stress_mask(mask, engine.stop, fraction, seed)
            active = pd.Series(kept[engine.stop:].cpu().numpy(), index=engine.query_drugs)
            unavailable = ~active
            bucket = f"{view}/{fraction:g}"
            modes = {bucket + "/all": None, bucket + "/originally_observed": observed,
                     bucket + "/retained": active, bucket + "/masked": observed & unavailable.reindex(
                         observed.index, fill_value=False)}
            evaluate(engine.baseline(base), bucket + "/C1", modes)
            for method in ("concat_zero", "concat_mean", "psd", "norm", "gated"):
                for weight in WEIGHTS:
                    spec = Fusion(method, weight, stress=fraction, protected=False)
                    evaluate(engine.fusion(base, [config], spec, seed=seed),
                             bucket + "/" + method + f"/{weight:g}", modes)
    if len(passed) == 2:
        joint = list(passed.values())
        both = coverage_for(coverage, joint[0]) & coverage_for(coverage, joint[1])
        evaluate(engine.fusion(base, joint, Fusion("norm")), "dual/norm/.25",
                 {"dual/all": None, "dual/observed": both})


def forecast_arms(engine, policy, seed):
    baseline = Baseline(**policy["base"])
    models = {"C0": engine.baseline(baseline, targets=False), "C1": engine.baseline(baseline)}
    flags = {}
    for view, item in policy["r1"].items():
        config = Representation(**item["config"])
        flags[view + "_observed"] = engine.representations(config)[1][engine.stop:]
        for name, spec in (("C2", Fusion("mask")), ("C3", Fusion(shuffle=True)), ("C4", Fusion())):
            models[view + "/" + name] = engine.fusion(baseline, [config], spec, seed=seed)
    for view, trial in policy["r2"]["views"].items():
        for method, entry in (("fusion_candidate", trial["challenger"]), ("naive", trial["naive"])):
            models[view + "/" + method] = engine.fusion(
                baseline, config_dict(entry["configs"]), spec_dict(entry["spec"]), seed=seed)
    if policy["champion"]:
        item = policy["champion"]
        configs = config_dict(item["configs"])
        spec = spec_dict(item["spec"])
        models["champion"] = engine.fusion(baseline, configs, spec, seed=seed)
        flags["champion_observed"] = models["champion"].present
        models["champion_mask"] = engine.fusion(baseline, configs, Fusion("mask"), seed=seed)
        models["champion_shuffle"] = engine.fusion(baseline, configs, replace(spec, shuffle=True), seed=seed)
    else:
        models["champion"] = models["C1"]
        flags["champion_observed"] = torch.zeros(len(engine.query_drugs), dtype=torch.bool, device=engine.device)
    # Explicit all-masked intervention through the same deployed selection policy.
    models["structure_only"] = models["C1"]
    if models["champion"].present is not None:
        masked = torch.zeros_like(models["champion"].present)
        models["structure_only"] = replace(models["champion"], present=masked)
    return models, flags


def offline_forecast(package, query, raw_rna):
    basis = package["basis"]
    cells = pd.Index(sorted(query.cell_id.unique()))
    rna = raw_rna.reindex(index=cells, columns=basis.genes)
    if rna.isna().all(axis=1).any():
        raise ValueError("Query cell requires measured basal RNA")
    rna = rna.fillna(package["medians"])
    raw = basis.cross(rna)["full"]
    context = None
    if package["pathway"] is not None:
        w, mean, scale = package["pathway"]
        p = rna.columns.get_indexer(w.index)
        ok = p >= 0
        proj = vc_rank(rna)[:, p[ok]] @ w.to_numpy(dtype=np.float32)[ok]
        context = torch.as_tensor(((proj - mean) / scale).clip(-4, 4))
    ix = cells.get_indexer(query.cell_id), package["drug_ids"].get_indexer(query[KEY])
    if (ix[1] < 0).any():
        raise ValueError("Drug is outside the sealed registered prediction panel")
    return {k: model.values(raw, package["center"], context).numpy()[ix] for k, model in package["models"].items()}


def vc_rank(rna):
    from vc_build.structured_kernel import ranked_expression
    return ranked_expression(rna)


def freeze_forecast(data, registry, coverage, external, outer, root):
    selection = root / "champion.json"
    policy = read_json(selection)[str(outer)]
    if outer == 3:
        fit = response(registry.index, labels=True)
        query = external[META].sort_values("row_id", ignore_index=True)
    else:
        fitting, held = subfold(registry, outer)
        fit = response(fitting, labels=True)
        query = response(held, labels=False)
        if set(fit[KEY]) & set(query[KEY]):
            raise AssertionError("Held exact parent entered source fitting")
    engine = Fit(data, fit, query, registry=registry)
    models, flags = forecast_arms(engine, policy, 64063 + outer)
    package = {"basis": engine.basis.cpu(), "center": engine.center.cpu(), "drug_ids": engine.query_drugs,
        "medians": engine.medians, "models": {k: m.cpu() for k, m in models.items()}, "pathway": None}
    if engine._context is not None:
        w = data.pathway
        indices = data.rna.columns.get_indexer(w.index)
        use = indices >= 0
        training = vc_rank(data.expression(engine.fit_cells))[:, indices[use]] @ w.to_numpy(dtype=np.float32)[use]
        package["pathway"] = (w, training.mean(axis=0), training.std(axis=0).clip(.05))
    freeze = root / "forecasts" / str(outer)
    freeze.mkdir(parents=True, exist_ok=False)
    # Production forecast uses the same offline CPU entry point as inference replay.
    before = offline_forecast(package, query, data.rna)
    path = freeze / "predictor.pkl"
    with path.open("xb") as handle:
        pickle.dump(package, handle, protocol=5)
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    after = offline_forecast(loaded, query, data.rna)
    delta = max(float(np.abs(after[name] - value).max()) for name, value in before.items())
    if delta >= 1e-6:
        raise AssertionError("Offline inference replay mismatch")
    frame = query.assign(**before, **{name: flag.cpu().numpy()[engine.selector[1]] for name, flag in flags.items()})
    missing = ~frame.champion_observed
    error = float(np.abs(frame.loc[missing, "champion"] - frame.loc[missing, "C1"]).max()) if missing.any() else 0.
    all_hidden = float(np.abs(frame.structure_only - frame.C1).max())
    if error != 0. or all_hidden != 0.:
        raise AssertionError("Protected missing-view prediction differs from C1")
    frame.to_parquet(freeze / "predictions.parquet", index=False)
    publish_json(freeze / "receipt.json", {"partition": outer, "champion_sha256": sha256(selection),
        "prediction_sha256": sha256(freeze / "predictions.parquet"), "predictor_sha256": sha256(path),
        "fit_rows": len(fit), "fit_parents": len(engine.fit_drugs), "query_rows": len(frame),
        "query_parents": len(engine.query_drugs), "query_outcomes_loaded": 0,
        "offline_max_abs_diff": delta, "natural_fallback_max_abs_diff": error,
        "structure_only_max_abs_diff": all_hidden, "natural_missing_rows": int(missing.sum()), "solver": engine.notes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("r1base", "r1", "r1controls", "r2", "r3", "freeze"))
    parser.add_argument("assets", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.index is None:
        import os
        args.index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not torch.cuda.is_available():
        raise RuntimeError("Study GPU fitting requires a Slurm CUDA allocation")
    gpu = torch.device("cuda")
    torch.set_num_threads(5)
    data, registry, coverage, external = load_context(args.assets, args.targets, gpu,
                                                     external=args.stage == "freeze" and args.index == 3)
    print(f"{args.stage} index={args.index} {torch.cuda.get_device_name(gpu)}", flush=True)
    start = time.monotonic()
    if args.stage == "freeze":
        freeze_forecast(data, registry, coverage, external, args.index, args.root)
    else:
        outer, inner = divmod(args.index, 3)
        record_inner(args.stage, data, registry, coverage, outer, inner, args.root)
    print(f"completed in {time.monotonic() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
