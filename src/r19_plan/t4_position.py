"""Audit time-transfer predictions on independently positioned RNA inputs and fixed drug pools."""

import argparse
from hashlib import sha256 as hash_text
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import checked_array, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_access import CONDITION
from src.r19_plan.t4_context_transfer import group_plans
from src.r19_plan.t4_scaffold_transfer import cut, fit_effects, query_plan, registry
from src.r19_plan.t4_state import lineage, prior_case
from src.r19_plan.t4_workflow import average_conditions, axes, deltas, pairs, scalar, score_delta
from src.r19_plan.time_operator import fit_time_operator
from src.r19_plan.x_inputs import check_unique, control_axis


OP, SC, ST = "purged_operator", "purged_scalar", "State_6h"
BASE = (OP, SC, ST)
NEW = ("op_different", "scalar_different", "op_pair_different", "op_pair_same", "scalar_pair_different", "scalar_pair_same")
POOL = ["fold", "role", "rna_plate", "actual_dose_uM"]


def selections(query, donors):
    groups = donors.groupby(CONDITION, sort=False).indices
    result = []
    for parent, dose, location, target in query[[*CONDITION, "rna_well", "inst_id"]].itertuples(index=False, name=None):
        candidate = groups[parent, dose]
        if not donors.iloc[candidate].rna_well.str.fullmatch(r"[A-Z]+\d{2}").all():
            raise ValueError("Unknown donor RNA position")
        same = donors.iloc[candidate].rna_well.eq(location).to_numpy()
        early_same = candidate[same]
        early_different = candidate[~same]
        n = min(len(early_same), len(early_different))

        def ordered(indices):
            return sorted(indices, key=lambda i: hash_text(f"T4-RNA-position|{target}|{donors.inst_id.iat[i]}".encode()).digest())[:n]

        result.append((early_different, ordered(early_different), ordered(early_same)))
    return result


def setup(args):
    registered = SimpleNamespace(previous=args.previous, reports=args.reports,
        protocol=args.reports / "T4_STATE_PROTOCOL.md", scaffold=args.scaffold)
    ctx, scaffold, state_stamp = lineage(registered)
    pin = args.reports / "data/t4_state/freeze.json"
    state = read_json(pin)
    if sha256(args.state / "freeze.json") != sha256(pin) or any(state[k] != v for k, v in state_stamp.items()):
        raise ValueError("State input differs from the pinned source")
    stamp = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
        access_sha256=state_stamp["access_sha256"], scaffold_freeze_sha256=state_stamp["scaffold_freeze_sha256"],
        state_freeze_sha256=sha256(pin), t4_freeze_sha256=state_stamp["original_sha256"])
    return ctx, scaffold, state, stamp


def predict(args, ctx, scaffold, state, stamp):
    held, cores = registry(ctx)
    data = group_plans(ctx, args.group)
    fold, role, *_ = axes(next(iter(data)))
    index, values = fit_effects(ctx, data, role)
    out = args.run / "forecasts" / args.group
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    state_folder = args.state / "forecasts" / args.group
    state_receipt = read_json(state_folder / "receipt.json")
    if sha256(state_folder / "receipt.json") != state["groups"][args.group]:
        raise ValueError("Changed State forecast receipt")
    with TemporaryDirectory(dir=out.parent, prefix=".t4-position-") as tmp:
        stage = Path(tmp)
        digests, selection_hashes = {}, {}
        for key, block in data.items():
            late, early, query, donor = [block[n] for n in ("source24", "source6", "query", "donor6")]
            controls = block["fit_controls"].loc[lambda rows: rows.rna_plate.isin(donor.rna_plate)]
            admitted = pd.concat((donor, controls), ignore_index=True)
            check_unique(admitted)
            control_axis(controls, donor, role)
            expression = extract_matrix(ctx.ref["matrix"], admitted, ctx.cols, ctx.genes)
            donor_values = deltas(donor, expression[:len(donor)], controls, expression[len(donor):])
            partitions, _, _ = cut(late.parent, held[fold], cores, fold, axes(key)[4])
            early_fit, late_fit = [table.loc[table.parent.isin(partitions["purged"])] for table in (early, late)]
            triplet = pairs(average_conditions(early_fit, values[index.get_indexer(early_fit.inst_id)]),
                            late_fit, values[index.get_indexer(late_fit.inst_id)])
            model = fit_time_operator(*triplet, ctx.genes, rank=16, ridge=1, held_parents=held[fold])
            gain = scalar(*triplet)
            old, fit = prior_case(args, ctx, scaffold, args.group, key, query)
            full = old["identity6"]
            if (fit["fit_parent"] != model.fit_parent_count or fit["fit_conditions"] != len(triplet[0])
                    or not np.array_equal(old[OP], model.predict(full, ctx.genes).astype("float32"))
                    or not np.array_equal(old[SC], (gain * full).astype("float32"))):
                raise ValueError("Position comparison does not reproduce the frozen references")
            state_file = state_folder / f"{key}.npy"
            if sha256(state_file) != state_receipt["slices"][key]:
                raise ValueError("Changed State prediction")
            products = {OP: old[OP], SC: old[SC], ST: checked_array(np.load(state_file, allow_pickle=False), full.shape, ST)}
            chosen = selections(query, donor)
            selected_ids = [(target, *[tuple(donor.inst_id.iloc[ids]) for ids in parts])
                            for target, parts in zip(query.inst_id, chosen, strict=True)]
            selection_hashes[key] = hash_text(repr(selected_ids).encode()).hexdigest()
            masks = []
            for j in range(3):
                subset = [indices[j] for indices in chosen]
                has = np.asarray([len(ids) > 0 for ids in subset])
                masks.append(has)
                subset_data = np.stack([donor_values[ids].mean(axis=0) if len(ids) else np.zeros(full.shape[1])
                                        for ids in subset]).astype(full.dtype)
                products[NEW[2*j]] = model.predict(subset_data, ctx.genes)
                products[NEW[2*j+1]] = gain * subset_data
            if not np.array_equal(masks[1], masks[2]):
                raise ValueError("Matched-position masks differ")
            file = stage / f"{key}.npz"
            np.savez_compressed(file, **products, different=masks[0], pair=masks[1])
            digests[key] = sha256(file)
        write_json(stage / "receipt.json", dict(stamp, group=args.group, slices=digests, selection_sha256=selection_hashes))
        stage.rename(out)


def compare(rows, truth, arrays, comparisons, label):
    result = []
    energy = squared_rows(truth)
    for arm, reference in comparisons:
        moments = parent_summaries(rows, truth.shape[1], energy, squared_rows(truth, arrays[arm]), squared_rows(truth, arrays[reference]))
        points, ci, counts = paired_effects(*moments)
        lower, upper, _ = relative_interval(moments[0])
        for metric, (a, b), (lo, hi), draws in zip(("rmse", "r2_pooled"), points, ci, counts, strict=True):
            rmse = metric == "rmse"
            result.append(dict(scope=label, parents=rows.parent.nunique(), wells=rows.inst_id.nunique(),
                arm=arm, reference=reference, metric=metric, arm_value=a, reference_value=b, difference=a-b, low99=lo,
                high99=hi, relative_gain=1-a/b if rmse else np.nan, relative_low99=lower if rmse else np.nan,
                relative_high99=upper if rmse else np.nan, draws=draws))
    return result


def contrast(rows, truth, arrays):
    predictions = {name: arrays[name].astype("float64") for name in BASE}
    target = truth.astype("float64")
    details = []
    for positions in rows.groupby(POOL, sort=False).indices.values():
        labels, group = np.unique(rows.parent.iloc[positions], return_inverse=True)
        if len(labels) < 3:
            raise ValueError("A contrast pool has fewer than three query parents")
        centered = {}
        for name, value in dict(predictions, target=target).items():
            means = np.stack([value[positions[group == i]].mean(axis=0) for i in range(len(labels))])
            center = means.mean(axis=0)
            value[positions] -= center
            means -= center
            centered[name] = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
        for arm in BASE:
            cosine = centered[arm] @ centered["target"].T
            correct = cosine.diagonal()[:, None]
            ties = np.isclose(cosine, correct, atol=1e-10, rtol=1e-8)
            better = ((cosine > correct) & ~ties).sum(axis=1)
            for k in (1, 5):
                hit = np.clip((k - better) / ties.sum(axis=1), 0, 1)
                for i, parent in enumerate(labels):
                    details.append(dict(parent=parent, arm=arm, k=k, top=hit[i], chance=min(k/len(labels), 1), pool_size=len(labels)))
    return target, predictions, pd.DataFrame(details)


def retrieval(results):
    output = []
    for label, frame in (("min3", results), ("min6", results.loc[results.pool_size.ge(6)])):
        for k, block in frame.groupby("k"):
            points = block.groupby(["parent", "arm"])[["top", "chance"]].mean()
            top = points.top.unstack()
            rng = np.random.default_rng(38105)
            draw = rng.integers(len(top), size=(10_000, len(top)))
            for reference in (ST, SC, "chance"):
                baseline = top[reference] if reference != "chance" else points.xs(OP, level="arm").chance
                diff = (top[OP] - baseline).to_numpy()
                lo, hi = np.quantile(diff[draw].mean(axis=1), [.005, .995])
                output.append(dict(scope=label, parents=len(top), k=k, arm=OP, reference=reference,
                    arm_value=top[OP].mean(), reference_value=baseline.mean(), difference=diff.mean(), low99=lo, high99=hi))
    return pd.DataFrame(output)


def finish(args, ctx, scaffold, state, stamp):
    scores = args.phase == "score"
    freeze_path = args.run / "freeze.json"
    if scores and (sha256(freeze_path) != args.digest or (args.run / "score").exists()):
        raise ValueError("Score requires a new directory and valid independent freeze digest")
    if not scores and freeze_path.exists():
        raise FileExistsError(freeze_path)
    queries, pieces, groups, selections_by_roles = {}, {}, {}, {}
    for group in sorted(scaffold["groups"]):
        folder = args.run / "forecasts" / group
        receipt_path = folder / "receipt.json"
        receipt = read_json(receipt_path)
        keys = {k for k in ctx.ref["slices"] if k.startswith(group + "-")}
        if (receipt["group"] != group or set(receipt["slices"]) != keys or set(receipt["selection_sha256"]) != keys
                or any(receipt[k] != v for k, v in stamp.items())):
            raise ValueError("Position forecast does not match the registered slices")
        groups[group] = sha256(receipt_path)
        for key, digest in sorted(receipt["slices"].items()):
            fold, _, *_, plate = axes(key)
            selection_hash = receipt["selection_sha256"][key]
            if selections_by_roles.setdefault((fold, plate), selection_hash) != selection_hash:
                raise ValueError("Vehicle rotation changed the selected donor wells")
            query = query_plan(ctx, key)
            path = folder / f"{key}.npz"
            if sha256(path) != digest:
                raise ValueError("Position forecast content changed")
            with np.load(path, allow_pickle=False) as stored:
                for arm in (*BASE, *NEW):
                    checked_array(stored[arm], (len(query), len(ctx.genes)), arm)
                different, pair = stored["different"], stored["pair"]
                if any(a.dtype.kind != "b" or a.shape != (len(query),) for a in (different, pair)) or (pair & ~different).any():
                    raise ValueError("Invalid location support")
                queries[key] = query.assign(fold=axes(key)[0], role=axes(key)[1], different=different, pair=pair)
                if scores:
                    pieces[key] = {arm: stored[arm] for arm in (*BASE, *NEW)}

    rows = checked_rows(pd.concat(queries.values(), ignore_index=True), None)
    if not rows.groupby("inst_id")[["pair", "different"]].nunique().eq(1).all().all():
        raise ValueError("Position support changed across vehicle roles")
    rows["pool"] = rows.groupby(POOL).parent.transform("nunique").ge(3)
    support = []
    for scope, expected in (("different", (59, 82)), ("pair", (35, 46)), ("pool", (69, 118))):
        selected = rows.loc[rows[scope]]
        if (selected.parent.nunique(), selected.inst_id.nunique()) != expected:
            raise ValueError(f"Registered position or pool support changed for {scope}")
        for fold, block in selected.groupby("fold"):
            support.append(dict(scope=scope, fold=int(fold), parents=block.parent.nunique(),
                wells=block.inst_id.nunique(), plates=block.rna_plate.nunique(), pools=block[POOL].drop_duplicates().shape[0]//3))
    locked = dict(stamp, groups=groups, support=support, folds=3, roles=3)
    if not scores:
        write_json(freeze_path, locked)
        return
    if read_json(freeze_path) != locked:
        raise ValueError("Position and retrieval support or predictions changed after freezing")

    truth = np.concatenate([score_delta(ctx.scope, ctx.ref, k, q.drop(columns=["fold", "role", "different", "pair"]),
                                      ctx.cols, ctx.genes) for k, q in queries.items()])
    arrays = {arm: np.concatenate([p[arm] for p in pieces.values()]) for arm in (*BASE, *NEW)}
    comparisons = {
        "different": (("op_different", "scalar_different"), (OP, SC), (OP, ST), ("op_different", OP)),
        "pair": (("op_pair_different", "op_pair_same"), ("scalar_pair_different", "scalar_pair_same"),
                 ("op_pair_different", "scalar_pair_different"), ("op_pair_same", "scalar_pair_same")),
        "pool": ((OP, ST), (OP, SC)),
    }
    results = []
    for scope, pairs_to_score in comparisons.items():
        mask = rows[scope].to_numpy()
        sample, y = rows.loc[mask].reset_index(drop=True), truth[mask]
        forecasts = {name: value[mask] for name, value in arrays.items()}
        if scope == "pool":
            y, forecasts, hits = contrast(sample, y, forecasts)
        results.extend(compare(sample, y, forecasts, pairs_to_score, scope))
    with TemporaryDirectory(dir=args.run, prefix=".t4-position-score-") as tmp:
        stage = Path(tmp)
        pd.DataFrame(results).to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        retrieval(hits).to_csv(stage / "retrieval.tsv", sep="\t", index=False)
        pd.DataFrame(support).to_csv(stage / "support.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(locked, freeze_sha256=args.digest,
            files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(args.run / "score")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("predict", "freeze", "score"))
    for field in ("reports", "previous", "scaffold", "state", "protocol", "run"):
        cli.add_argument("--"+field, type=Path, required=True)
    cli.add_argument("--group")
    cli.add_argument("--digest")
    args = cli.parse_args()
    (predict if args.phase == "predict" else finish)(args, *setup(args))
