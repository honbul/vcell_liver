"""Fit a same-measurement singleton State comparator on the locked scaffold T4 scope."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.t4_context_transfer import group_plans, original
from src.r19_plan.t4_scaffold_transfer import cluster_effects, cut, fit_effects, inputs, query_plan, registry
from src.r19_plan.t4_workflow import average_conditions, axes, pairs, score_delta


OLD = ("purged_operator", "purged_scalar", "identity6", "direct24")
STATE = "State_6h"
COMPARISONS = (("purged_operator", STATE), *((STATE, name) for name in OLD[1:]))


def lineage(args):
    source = SimpleNamespace(previous=args.previous, reports=args.reports, protocol=args.reports / "T4_SCAFFOLD_PROTOCOL.md")
    ctx = inputs(source)
    pin = args.reports / "data/t4_scaffold/freeze.json"
    prior = read_json(pin)
    if sha256(args.scaffold / "freeze.json") != sha256(pin) or any(prior[k] != v for k, v in ctx.stamp.items()):
        raise ValueError("The published scaffold-filtered comparator is not the pinned source")
    stamp = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__), access_sha256=ctx.stamp["access_sha256"],
                 original_sha256=ctx.stamp["previous_freeze_sha256"], scaffold_freeze_sha256=sha256(pin))
    return ctx, prior, stamp


def prior_case(args, ctx, old, group, key, query):
    folder = args.scaffold / "forecasts" / group
    receipt = read_json(folder / "receipt.json")
    payload = folder / f"{key}.npz"
    if sha256(folder / "receipt.json") != old["groups"][group] or sha256(payload) != receipt["slices"][key]:
        raise ValueError("Original scaffold forecast changed")
    originals = original(ctx, key, query)
    result = {name: originals[name] for name in ("identity6", "direct24")}
    with np.load(payload, allow_pickle=False) as f:
        result.update({name: checked_array(f[name], result["identity6"].shape, name) for name in OLD[:2]})
    fit, = [x for x in receipt["settings"] if x["slice"] == key and x["mode"] == "purged"]
    return result, fit


def state_forecast(network, triplet, query):
    import torch
    from src.goal_tracks.t2_state import build_network, forward

    rows, x, y = triplet
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = build_network(network, y.shape[1], 1, cell_set_len=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)

    def tensor(values):
        return torch.as_tensor(values, device=device, dtype=torch.float32)

    base = tensor(x)[:, None, :]
    target = tensor(y)
    code = torch.ones((len(x), 1), device=device)
    weight = 1. / rows.groupby("parent").parent.transform("size").to_numpy()
    weight = tensor(weight / weight.mean())
    for _ in range(40):
        model.train()
        order = torch.randperm(len(rows), device=device)
        for idx in order.split(128):
            predicted = forward(model, base[idx], code[idx])
            loss = (weight[idx] * (predicted-target[idx]).square().mean(dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        model.eval()
        data = tensor(query)[:, None, :]
        return forward(model, data, torch.ones((len(query), 1), device=device)).cpu().numpy()


def predict(args, ctx, old, stamp):
    import torch
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    held_folds, cores = registry(ctx)
    data = group_plans(ctx, args.group)
    fold, role, *_ = axes(next(iter(data)))
    index, values = fit_effects(ctx, data, role)
    dest = args.run / "forecasts" / args.group
    if dest.exists():
        raise FileExistsError(dest)
    network = load_network(args.state)
    upstream_model = args.state / "src/state/tx/models/state_transition.py"
    args.run.joinpath("forecasts").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".t4-state-") as tmp:
        stage = Path(tmp)
        assets, support = {}, []
        for key, entry in data.items():
            late, early, query = [entry[name] for name in ("source24", "source6", "query")]
            available, _, _ = cut(late.parent, held_folds[fold], cores, fold, axes(key)[4])
            use = available["purged"]
            x, y = (table.loc[table.parent.isin(use)] for table in (early, late))
            triplet = pairs(average_conditions(x, values[index.get_indexer(x.inst_id)]), y, values[index.get_indexer(y.inst_id)])
            comparators, reference = prior_case(args, ctx, old, args.group, key, query)
            parents, conditions = triplet[0].parent.nunique(), len(triplet[0])
            if parents != reference["fit_parent"] or conditions != reference["fit_conditions"] or set(y.parent) & held_folds[fold]:
                raise ValueError("State sees a different set of fitted scaffold-filtered conditions")
            prediction = checked_array(state_forecast(network, triplet, comparators["identity6"]),
                                       comparators["identity6"].shape, STATE)
            file = stage / f"{key}.npy"
            np.save(file, prediction, allow_pickle=False)
            assets[key] = sha256(file)
            support.append(dict(slice=key, source_parents=parents, source_conditions=conditions, query_wells=len(query)))
            print(f"{key}: {parents} fit parents, {conditions} source time pairs", flush=True)
        write_json(stage / "receipt.json", dict(stamp, group=args.group, slices=assets, support=support,
               model_upstream=UPSTREAM, model_sha256=sha256(upstream_model), torch_version=torch.__version__,
               device=torch.cuda.get_device_name(), epochs=40, seed=42, cell_tokens=1))
        stage.rename(dest)


def compare(rows, truth, predicted, mapping):
    energy = squared_rows(truth)
    errors = {name: squared_rows(truth, array) for name, array in predicted.items()}
    parents = sorted(set(rows.parent))
    result = []
    for arm, baseline in COMPARISONS:
        moments = parent_summaries(rows, truth.shape[1], energy, errors[arm], errors[baseline])
        points, ci, draws = paired_effects(*moments)
        bounds = relative_interval(moments[0])[:2]
        cl_ci, cl_bounds = cluster_effects(parents, mapping, *moments)
        for unit, paired, bounds in (("parent", ci, bounds), ("scaffold", cl_ci, cl_bounds)):
            for metric, (a, b), (lo, hi), n in zip(METRICS, points, paired, draws, strict=True):
                rmse = metric == "rmse"
                result.append(dict(unit=unit, arm=arm, reference=baseline, metric=metric, arm_value=a, reference_value=b,
                    difference=a-b, low99=lo, high99=hi, relative_gain=1-a/b if rmse else np.nan,
                    relative_low99=bounds[0] if rmse else np.nan, relative_high99=bounds[1] if rmse else np.nan, draws=n))
    return pd.DataFrame(result)


def finish(args, ctx, prior, stamp):
    scoring = args.phase == "score"
    freeze = args.run / "freeze.json"
    if scoring and (args.run.joinpath("score").exists() or sha256(freeze) != args.digest):
        raise ValueError("Separate State scoring requires the independent forecast freeze digest")
    if not scoring and freeze.exists():
        raise FileExistsError(freeze)
    queries, digests, fitting = {}, {}, []
    arrays = {name: [] for name in (*OLD, STATE)}
    for group in sorted(prior["groups"]):
        folder = args.run / "forecasts" / group
        receipt = read_json(folder / "receipt.json")
        keys = {k for k in ctx.ref["slices"] if k.startswith(group + "-")}
        if (any(receipt[k] != value for k, value in stamp.items()) or receipt["group"] != group
                or set(receipt["slices"]) != keys):
            raise ValueError("The State forecast differs from the registered split or provenance")
        digests[group] = sha256(folder / "receipt.json")
        fitting.extend(receipt["support"])
        for key, digest in sorted(receipt["slices"].items()):
            query = queries[key] = query_plan(ctx, key)
            known, _ = prior_case(args, ctx, prior, group, key, query)
            file = folder / f"{key}.npy"
            if sha256(file) != digest:
                raise ValueError("A State prediction differs from the forecast receipt")
            value = checked_array(np.load(file, allow_pickle=False), known["identity6"].shape, STATE)
            if scoring:
                for name, array in {**known, STATE: value}.items():
                    arrays[name].append(array)
    rows = checked_rows(pd.concat([q.assign(fold=axes(k)[0], role=axes(k)[1]) for k, q in queries.items()], ignore_index=True), None)
    if rows.parent.nunique() != 132 or rows.inst_id.nunique() != 268 or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("The complete paired 132-parent, 268-well support is required")
    manifest = dict(stamp, groups=digests)
    if not scoring:
        write_json(freeze, manifest)
        return
    if read_json(freeze) != manifest:
        raise ValueError("Candidate arrays and reference lineage changed after the freeze")
    truth = np.concatenate([score_delta(ctx.scope, ctx.ref, k, q, ctx.cols, ctx.genes) for k, q in queries.items()])
    _, mapping = registry(ctx)
    table = compare(rows, truth, {name: np.concatenate(values) for name, values in arrays.items()}, mapping)
    with TemporaryDirectory(dir=args.run, prefix=".t4-state-score-") as tmp:
        stage = Path(tmp)
        table.to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(fitting).to_csv(stage / "fit_support.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(manifest, freeze_sha256=args.digest, parents=rows.parent.nunique(),
            wells=rows.inst_id.nunique(), scaffold_components=len(set(mapping.values())),
            files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(args.run / "score")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("predict", "freeze", "score"))
    for field in ("reports", "previous", "scaffold", "protocol", "run", "state"):
        cli.add_argument("--" + field, type=Path)
    cli.add_argument("--group")
    cli.add_argument("--digest")
    args = cli.parse_args()
    lineage_values = lineage(args)
    (predict if args.phase == "predict" else finish)(args, *lineage_values)
