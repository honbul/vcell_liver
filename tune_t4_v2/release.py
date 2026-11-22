"""Produce and freeze role-specific T4 forecasts before any score response is read."""

import argparse
from pathlib import Path
from zlib import crc32

import joblib
import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.t4_context_transfer import original
from src.r19_plan.t4_scaffold_transfer import centered_fit, scaffold, summarize
from src.r19_plan.t4_state import state_forecast
from src.r19_plan.t4_workflow import axes, score_delta

from .data import SourceWells, all_pairs, contexts, parent_folds
from .linear import point_fits
from .model import TemporalHead, fit_adapter
from .pretrain import load_pan
from .neural import CONFIGS, cross_residual, fit as fit_neural
from .round2 import ARM_MODES
from .rounds34 import prior_predictions
from .transfer import BUDGETS

V1 = (False, "pcr", 1., 16, "")



def fit_release(cell, train, conf, seed, pan, genes, device):
    setting = tuple(conf["linear_config"])
    _, _, linear = next(point_fits([setting], train, train.x[:0], train.rows.iloc[:0], need_model=True))
    policy = conf["round4_champion"]
    local, pan_arm, pan_weight = policy, "", 0.
    if "+" in policy:
        local, remainder = policy.split("+")
        pan_arm, weight = remainder.split("/")
        pan_weight = float(weight)
    elif policy.startswith("pan_"):
        local, pan_arm, pan_weight = "", policy, 1.
    nn_model, nn_weight, adapter = None, 0., None
    if local == "linear_neural":
        name, weight = conf["round2"].split("/")
        target = cross_residual(train, setting)
        nn_weight = float(weight)
        nn_model = fit_neural(train, target, ARM_MODES[name], CONFIGS[name], seed, device)
    if pan_arm.startswith("pan_all_rank"):
        rank = int(pan_arm.rsplit("rank", 1)[1])
        adapter = fit_adapter(pan, train, rank, device=device)
    return TemporalHead(policy, linear, nn_model, nn_weight, train.x.mean(axis=0), adapter,
                        pan_weight, set(train.rows.scaffold), genes, seed)


def old_group(folder, pin, group):
    receipts = read_json(pin)
    source = folder / "forecasts" / group
    if sha256(source / "receipt.json") != receipts["groups"][group]:
        raise ValueError("Historical T4 comparator group diverged from its registered freeze")
    inventory = read_json(source / "receipt.json")
    return source, inventory


def old_array(source, inventory, key, suffix):
    path = source / f"{key}.{suffix}"
    if sha256(path) != inventory["slices"][key]:
        raise ValueError("Historical T4 comparator array diverged from its registered freeze")
    return np.load(path, allow_pickle=False)


def predict(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ctx = contexts(args.reports, args.hepg2, args.phh)[args.cell]
    config = read_json(args.settings)
    if config["cell"] != args.cell or config["group"] != args.group:
        raise ValueError("Settings and outer slice mismatch")
    prior_path = args.pretrain / "pan_model.joblib"
    if sha256(prior_path) != config["pretrain_sha256"]:
        raise ValueError("Pan model mismatched the frozen inner choices")
    pan = load_pan(prior_path)
    holdout = parent_folds(ctx)[int(args.group[1])]
    source = SourceWells(ctx, args.group, {scaffold(p) for p in holdout})
    repeat_refs = old_group(args.repeat / args.cell,
                            args.reports / "data/t4_repeat" / args.cell / "freeze.json", args.group)
    if args.cell == "HEPG2":
        orig = old_group(args.hep_mean_shift, args.reports / "data/t4_mean_shift/freeze.json", args.group)
        state_refs = old_group(args.hep_state, args.reports / "data/t4_state/freeze.json", args.group)
    else:
        from src.goal_tracks.t2_state import load_network

        state_network = load_network(args.state_upstream)
    variants = {}
    for conf in config["slices"]:
        key = conf["slice"]
        query = source.plans[key]["query"].copy()
        query["dose_uM"] = query.actual_dose_uM.astype(float)
        existing = original(ctx, key, query)
        x = existing["identity6"]
        early, late = source.data(key)
        train = all_pairs(early, late)
        seed = crc32((key+"|fit").encode())
        trained = fit_release(args.cell, train, conf, seed, pan, ctx.genes, args.device)
        p = trained.predict(query, x, pan, device=args.device)
        release_file = args.out / f"{key}.joblib"
        joblib.dump(trained, release_file, compress=2)
        samples = type("Held", (), {"rows": query, "x": x})
        transferable = prior_predictions(pan, train, samples, cell=args.cell, seed=seed, device=args.device)
        arms = {"v2": p, "v2_linear": trained.linear.predict(x, query),
                "pan_zero_centered": transferable["pan_zero_centered"], "direct24": existing["direct24"]}
        with old_array(*repeat_refs, key, "npz") as  reference:
            arms["direct24_calibrated"] = reference["direct24_scalar"]
        if args.cell == "HEPG2":
            with old_array(*orig, key, "npz") as reference:
                arms.update({"v1_rank16": reference["purged_operator"], "mean_scalar": reference["purged_mean_scalar"],
                             "diagonal": reference["purged_diagonal"]})
            arms["state"] = old_array(*state_refs, key, "npy")
        else:
            arms["v1_rank16_archived"] = existing["local_operator"]
            arms["v1_scalar_archived"] = existing["local_scalar"]
            _, arms["v1_rank16"], _ = next(point_fits([V1], train, x, query))
            arms["mean_scalar"], arms["diagonal"], *_ = centered_fit((train.rows, train.x, train.y), x)
            arms["state"] = state_forecast(state_network, (train.rows, train.x, train.y), x)
            for k in BUDGETS:
                arm = conf["fewshot_pan"].get(str(k), f"pan_{k}_rank0")
                arms[f"pan_fewshot_{k}"] = transferable[arm]
                arms[f"scalar_fewshot_{k}"] = transferable[f"scalar_{k}"]
                arms[f"linear_fewshot_{k}"] = transferable[f"linear_{k}"]
        payload = args.out / f"{key}.npz"
        expected = (len(query), len(ctx.genes))
        if any(a.shape != expected or not np.isfinite(a).all() for a in arms.values()):
            raise ValueError(f"Malformed temporal estimate for {key}")
        np.savez_compressed(payload, **{name: value.astype(np.float32) for name, value in arms.items()})
        variants[key] = {"forecast_sha256": sha256(payload), "model_sha256": sha256(release_file),
                         "fitted_parent_count": int(train.rows.parent.nunique()),
                         "holdout_scaffold_overlap": len(trained.train_scaffolds & set(query.parent.map(scaffold))),
                         "head": trained.policy, "wells": len(query),
                         "rows_sha256": ctx.ref["slices"][key]["predict"]["query.tsv"]}
        print(key, len(query), trained.policy, flush=True)
    write_json(args.out / "receipt.json", {"cell": args.cell, "group": args.group, "slices": variants,
                                          "settings_sha256": sha256(args.settings), "pan_sha256": sha256(prior_path),
                                          "score_labels_accessed": False})


def freeze(args):
    record = {}
    for cell in ("HEPG2", "PHH"):
        receipts = {}
        for fold in range(3):
            for role in "ABC":
                group = f"f{fold}-{role}"
                folder = args.forecasts / cell / group
                stamp = read_json(folder / "receipt.json")
                if stamp["cell"] != cell or stamp["group"] != group or stamp["score_labels_accessed"]:
                    raise ValueError("Unexpected forecast role")
                for key, part in stamp["slices"].items():
                    if (part["holdout_scaffold_overlap"] or sha256(folder / f"{key}.npz") != part["forecast_sha256"]
                            or sha256(folder / f"{key}.joblib") != part["model_sha256"]):
                        raise ValueError(f"T4 forecast or fitted scope mismatch: {key}")
                receipts[group] = sha256(folder / "receipt.json")
        record[cell] = receipts
    if args.lock.exists():
        raise FileExistsError(args.lock)
    write_json(args.lock, {"cell_groups": record, "no_score_input": True, "code_sha256": sha256(__file__)})


def score(args):
    if args.out.exists() or sha256(args.lock) != args.digest:
        raise ValueError("Score requires the original single-final lock")
    refs = contexts(args.reports, args.hepg2, args.phh)
    freeze_data = read_json(args.lock)
    args.out.mkdir(parents=True, exist_ok=False)
    if freeze_data["code_sha256"] != sha256(__file__):
        raise ValueError("T4 scoring implementation changed after lock")
    for cell, groups in freeze_data["cell_groups"].items():
        ctx = refs[cell]
        frames, actual, predictions = [], [], {}
        seen = set()
        for group, checksum in sorted(groups.items()):
            folder = args.forecasts / cell / group
            if sha256(folder / "receipt.json") != checksum:
                raise ValueError("Forecast receipt changed after lock")
            records = read_json(folder / "receipt.json")
            from src.r19_plan.t4_context_transfer import group_plans

            for key, tables in group_plans(ctx, group).items():
                q = tables["query"]
                part = records["slices"][key]
                path = folder / f"{key}.npz"
                if sha256(path) != part["forecast_sha256"] or key in seen:
                    raise ValueError("Duplicate or non-immutable forecast payload")
                seen.add(key)
                y = score_delta(ctx.scope, ctx.ref, key, q, ctx.cols, ctx.genes)
                frames.append(q.assign(fold=axes(key)[0], role=axes(key)[1]))
                actual.append(y)
                with np.load(path, allow_pickle=False) as diagnostics:
                    for arm in diagnostics.files:
                        predictions.setdefault(arm, []).append(diagnostics[arm])
        rows = pd.concat(frames, ignore_index=True)
        y = np.concatenate(actual)
        if (len(seen) != len(ctx.ref["slices"]) or rows.parent.nunique() != ctx.ref["parents"]
                or rows.inst_id.nunique() != (268 if cell=="HEPG2" else 105)):
            raise ValueError(f"The full registered {cell} scope was not scored")
        mapping = {p: scaffold(p) for p in rows.parent.unique()}
        arms = {tag: np.concatenate(blocks) for tag, blocks in predictions.items()}
        comparisons = [("v2", name) for name in arms if name != "v2"]
        if cell == "PHH":
            comparators = ("v1_rank16_archived", "v1_scalar_archived", "mean_scalar")
            comparisons.extend(("pan_zero_centered", control) for control in comparators)
            for k in BUDGETS:
                comparisons.extend((f"pan_fewshot_{k}", arm) for arm in (
                    f"linear_fewshot_{k}", f"scalar_fewshot_{k}", *comparators))
        summarize(rows, y, arms, mapping, comparisons=comparisons).to_csv(args.out / f"{cell}.tsv",
                                                                          sep="\t", index=False, na_rep="NA")
    write_json(args.out / "receipt.json", {"freeze_sha256": args.digest, "scores": {
        p.name: sha256(p) for p in args.out.glob("*.tsv")}, "score_process_completed": True})


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "freeze", "score"))
    for key in ("reports", "hepg2", "phh", "hep-mean-shift", "hep-state", "repeat", "state-upstream", "settings",
                "pretrain", "forecasts", "lock", "out"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--cell", choices=("HEPG2", "PHH"))
    parser.add_argument("--group")
    parser.add_argument("--digest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    {"predict": predict, "freeze": freeze, "score": score}[args.action](args)


if __name__ == "__main__":
    cli()
