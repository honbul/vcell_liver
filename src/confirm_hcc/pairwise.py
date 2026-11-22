"""Refit two fixed V0+ v2 recipes on the corrected source and save OOF and full fits."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vc_build.external import KEY, SOURCE, native_sensitivity
from vc_build.io import freeze_table, publish_json, sha256
from vc_build.v0plus_cv import panel_targets, validation_roster
from v0plus_v2 import lean, source, stack, trees
from v0plus_v2.release_types import SourceMember

from .contract import CMAP, FM, OLD_SETTINGS, TARGETS, TRAIN_SHA, corrected, fold_indices, read_json, v2_settings

RECIPES = ("pairwise_rank", "pairwise_top5")


def neural_banks(limore=False):
    banks, hashes = lean.fm_paths(FM, limore=limore)
    banks["cmap"], hashes["cmap"] = lean.fm_read(CMAP)
    return banks, hashes


def selection(fold):
    cfg = v2_settings(fold)
    adaptive = {}
    for row in cfg["round1-02"]["selection"]:
        adaptive.setdefault(row["compound_id"], {})[row["model"]] = tuple(row["cfg"])
    tree = cfg["round2-02"]["selected_trees"]
    amplitudes = {}
    for row in cfg["round4-01"]["inner_top_by_recipe"]:
        name, value = row["candidate"].split("/")
        if name in RECIPES:
            amplitudes[name] = float(value)
    if set(amplitudes) != set(RECIPES):
        raise ValueError("Both historic Pairwise amplitudes are required")
    required = set().union(*(stack.RECIPES[name][0] for name in RECIPES))
    adaptive = {drug: {k: v for k, v in options.items() if k in required}
                for drug, options in adaptive.items()}
    return adaptive, tree, amplitudes, required


def neural_models(fit, expression, banks, names, device, seed):
    return {
        name: lean.train(fit, expression, banks if name == "two_tower_cmap" else
                         {k: v for k, v in banks.items() if k != "cmap"}, cfg, device=device, seed=seed)
        for name, cfg in lean.CONFIGS.items() if name in names
    }


def component(args):
    train, expression, _ = corrected()
    outer, held = list(fold_indices(train))[args.fold]
    linear, tree, _, names = selection(args.fold)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.part == "linear":
        targets = read_json(TARGETS)
        banks, hashes = {}, {}
    else:
        banks, hashes = neural_banks()

    stages = [(f"inner{i}", fit, validation_roster(test), (781 + args.fold) * 37 + i)
              for i, (fit, test) in enumerate(source.grouped(outer, 2))]
    stages += [("outer", outer, validation_roster(held), 32912 + args.fold),
               ("full", train, None, 32912 + args.fold)]
    for stage, fit, query, seed in stages:
        fit_rna = (expression.loc[fit.cell_id.unique()] if args.part == "neural"
                   else expression.loc[expression.index.isin(fit.cell_id)])
        if args.part == "linear":
            first = source.fit_models(fit, fit_rna, targets, linear)
            boosted = trees.fit_models(fit, fit_rna, targets, tree)
            first.heads = {k: v for k, v in first.heads.items() if k in names}
            fitted = (first, boosted)
            predict = lambda q: first.predict(q, expression.loc[q.cell_id.unique()]) | boosted.predict(
                q, expression.loc[q.cell_id.unique()])
        else:
            fitted = neural_models(fit, fit_rna, banks, names, args.device, seed)
            predict = lambda q: {k: lean.forecast(v, q, expression, banks, device=args.device)
                                 for k, v in fitted.items()}
        if query is not None:
            frame = query[KEY].assign(**predict(query))
            frame.to_parquet(args.out / f"{stage}.parquet", index=False)
        if stage in ("full", "outer"):
            joblib.dump(fitted, args.out / f"{stage}.joblib", compress=2)
        print(f"fold={args.fold} part={args.part} stage={stage} n_fit={len(fit)}", flush=True)
    files = {p.name: sha256(p) for p in args.out.iterdir()}
    publish_json(args.out / "receipt.json", {"training_sha256": TRAIN_SHA, "fold": args.fold,
                 "settings_sha256": {k: v[args.fold] for k, v in OLD_SETTINGS.items()},
                 "embedding_sha256": hashes, "part": args.part, "artifacts": files})


def measurements(outer):
    parts, native = [], []
    for i, (train, held) in enumerate(source.grouped(outer, 2)):
        _, anchor, aligned, transforms = panel_targets(train)
        means = aligned.groupby([*SOURCE, "compound_id"]).source_mean.first().rename("source_mean")
        checked = held.merge(pd.DataFrame(transforms), on=SOURCE, validate="many_to_one")
        measured = native_sensitivity(checked) * checked.slope + checked.intercept
        checked = checked.assign(aligned=measured).join(means, on=[*SOURCE, "compound_id"], validate="many_to_one")
        checked["residual"] = checked.aligned - checked.source_mean
        checked["anchor"] = checked.compound_id.map(anchor)
        parts.append(checked.groupby(["cell_id", "compound_id", "lineage_group"], as_index=False)[
            ["residual", "anchor"]].mean().assign(inner=i))
        native.append(held.assign(inner=i, observed=native_sensitivity(held))[
            ["inner", *SOURCE, "cell_id", "compound_id", "observed"]])
    return pd.concat(parts, ignore_index=True), pd.concat(native, ignore_index=True)


def member(args):
    train, _, _ = corrected()
    outer, held = list(fold_indices(train))[args.fold]
    _, _, amplitudes, _ = selection(args.fold)
    receipts = {}
    for part in ("linear", "neural"):
        directory = args.root / part / f"fold{args.fold}"
        receipt = read_json(directory / "receipt.json")
        if (receipt["training_sha256"], receipt["fold"], receipt["part"]) != (TRAIN_SHA, args.fold, part):
            raise ValueError("Unexpected component freeze")
        if any(sha256(directory / f) != h for f, h in receipt["artifacts"].items()):
            raise ValueError("Pairwise fitted artifact changed")
        receipts[part] = sha256(directory / "receipt.json")

    def artifact(part, name):
        return args.root / part / f"fold{args.fold}" / name

    def forecast(stage):
        a, b = [pd.read_parquet(artifact(part, stage + ".parquet")) for part in ("linear", "neural")]
        return a.merge(b, on=KEY, validate="one_to_one")

    truth, native = measurements(outer)
    keys = ["inner", "cell_id", "compound_id"]
    oof = pd.concat([forecast(f"inner{i}").assign(inner=i) for i in (0, 1)])
    arms = sorted(set(oof.columns) - set([*KEY, "inner"]))
    aligned = truth.merge(oof[[*keys, *arms]], on=keys, validate="one_to_one")
    if len(aligned) != len(truth):
        raise ValueError("Incomplete inner OOF coverage")
    estimators = {}
    for recipe in RECIPES:
        model = stack.build(aligned[arms], aligned, recipe)
        model.amplitude = amplitudes[recipe]
        scores = model.predict(aligned[arms], aligned.anchor.to_numpy(), aligned.compound_id)
        ranking = native.merge(aligned[keys].assign(predicted_value=scores), on=keys, validate="many_to_one")
        focus = recipe == "pairwise_top5"
        model.offsets = stack.fit_offsets(ranking, sorted(truth.compound_id.unique()), float(focus),
                                          .55 if focus else .85)
        estimators[recipe] = model

    args.out.mkdir(exist_ok=False, parents=True)
    columns = set().union(*(head.columns for head in estimators.values()))
    adaptive, boosted = joblib.load(artifact("linear", "full.joblib"))
    neural = joblib.load(artifact("neural", "full.joblib"))
    fitted = SourceMember(adaptive, boosted, {k: v for k, v in neural.items() if k in columns}, estimators)
    joblib.dump(fitted, args.out / "member.joblib", compress=2)

    query = forecast("outer")
    _, anchor, _, _ = panel_targets(outer)
    bases = query.compound_id.map(anchor).to_numpy()
    values = [
        query[KEY].assign(method=name, predicted_value=head.predict(query[arms], bases, query.compound_id),
                          fold=args.fold, track="source_oof")
        for name, head in estimators.items()
    ]
    freeze_table(pd.concat(values, ignore_index=True), args.out / "oof.parquet", provenance={
        "train_data_sha256": TRAIN_SHA, "fit_rows": len(outer), "held_rows": len(held),
        "n_held_lineages": held.lineage_group.nunique(), "components": receipts,
        "fixed_amplitudes": amplitudes, "member_sha256": sha256(args.out / "member.joblib"),
        "refit_rows": len(train),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("component", "member"))
    parser.add_argument("--part", choices=("linear", "neural"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--fold", type=int, choices=range(3), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    {"component": component, "member": member}[args.stage](args)


if __name__ == "__main__":
    main()
