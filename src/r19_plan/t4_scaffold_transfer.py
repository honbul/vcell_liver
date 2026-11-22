"""Refit T4 after excluding the evaluated scaffolds from all source times."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zlib import crc32

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.design import fold_id
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import BOOTSTRAPS, METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.i_stage import parse_tsv, pinned_bytes
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_access import CELL, PHH_RANK, PHH_RIDGE
from src.r19_plan.t4_context_transfer import group_plans, original
from src.r19_plan.t4_workflow import average_conditions, axes, deltas, pairs, reference, scalar, score_delta
from src.r19_plan.time_operator import fit_time_operator
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes


MODES = ("full", "purged", "count")
ARMS = ("identity6", "local_operator", "direct24", *(f"{m}_{method}" for m in MODES for method in ("operator", "scalar")))
COMPARISONS = (
    *(("purged_operator", baseline) for baseline in ("purged_scalar", "identity6", "full_operator", "count_operator", "direct24")),
    ("full_operator", "full_scalar"), ("full_operator", "local_operator"), ("count_operator", "count_scalar"),
)
MEAN_SHIFT_ARMS = ("purged_mean_scalar", "purged_diagonal")


def centered_fit(fit, query):
    """Fit the centered scalar and a diagonal ridge selected within source parents."""
    rows, earlier, later = fit
    group, parents = pd.factorize(rows.parent, sort=True)
    counts = np.bincount(group)
    weights = 1.0 / counts[group]
    x_mean = np.average(earlier, axis=0, weights=weights)
    y_mean = np.average(later, axis=0, weights=weights)
    x = earlier - x_mean
    y = later - y_mean
    energy = (weights[:, None] * x * x).sum(axis=0)
    cross = (weights[:, None] * x * y).sum(axis=0)
    gain = cross.sum() / energy.sum()

    # Exact leave-one-parent-out moments with a newly fitted intercept for each holdout.
    parent_stats = []
    for values in (x, y, x * x, x * y):
        totals = np.zeros((len(parents), x.shape[1]))
        np.add.at(totals, group, values * weights[:, None])
        parent_stats.append(totals)
    sum_x, sum_y, sum_xx, sum_xy = parent_stats
    mass = len(parents) - 1
    x_loo = -sum_x / mass
    y_loo = -sum_y / mass
    xx_loo = np.maximum(energy - sum_xx - mass * x_loo**2, 0)
    xy_loo = cross - sum_xy - mass * x_loo * y_loo
    losses = []
    for ridge in (.01, .1, 1., 10., 100.):
        slopes = (xy_loo / (xx_loo + ridge))[group]
        residual = y - y_loo[group] - (x - x_loo[group]) * slopes
        losses.append(((weights[:, None] * residual**2).sum(), ridge))
    _, penalty = min(losses)
    centered = query - x_mean
    return y_mean + gain * centered, y_mean + centered * cross / (energy + penalty), gain, penalty


def scaffold(parent):
    molecule = Chem.MolFromSmiles(parent)
    if molecule is None:
        raise ValueError("The pinned T4 canonical parent is not a structure")
    core = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    return core or "acyclic:" + parent


def inputs(args):
    public = args.reports / "data/t4"
    scope = SimpleNamespace(access=args.previous / "access", pin=public / "access.json",
                            protocol=args.reports / "T4_PROTOCOL.md")
    ref = reference(scope)
    pin = public / "freeze.json"
    frozen = read_json(pin)
    if (sha256(args.previous / "freeze.json") != sha256(pin)
            or frozen["access_sha256"] != sha256(scope.pin) or frozen["protocol_sha256"] != sha256(scope.protocol)
            or set(frozen["slices"]) != set(ref["slices"])):
        raise ValueError("The actual T4 source artifacts differ from the published freeze")
    genes, cols = matrix_axes(scope.access, ref)
    stamp = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                 access_sha256=sha256(scope.pin), previous_freeze_sha256=sha256(pin))
    if args.mean_shift:
        operator = Path(__file__).with_name("time_operator.py")
        stamp.update(time_operator_sha256=sha256(operator))
    return SimpleNamespace(scope=scope, ref=ref, frozen=frozen, forecasts=args.previous / "forecasts",
                           genes=genes.pr_gene_id.to_numpy(), cols=cols, stamp=stamp)


def query_plan(ctx, key):
    name = "query.tsv"
    path = ctx.scope.access / "predict-plan" / key / name
    return parse_tsv(pinned_bytes(path, ctx.ref["slices"][key]["predict"][name]))


def registry(ctx):
    parent_sets = {i: set() for i in range(3)}
    for key in sorted(ctx.ref["slices"]):
        fold, role, *_ = axes(key)
        if role == "A":
            rows = query_plan(ctx, key)
            if not rows.parent.map(lambda p: fold_id(p, "R15-outer", 3)).eq(fold).all():
                raise ValueError("A scaffold registry parent has a different evaluation fold")
            parent_sets[fold].update(rows.parent)
    parents = set().union(*parent_sets.values())
    if len(parents) != ctx.ref["parents"] or sum(map(len, parent_sets.values())) != len(parents):
        raise ValueError("The official 132 T4 parents or their folds changed")
    mapping = {p: scaffold(p) for p in sorted(parents)}
    return parent_sets, mapping


def cut(available, held, cores, fold, plate):
    parents = sorted(set(available))
    held_cores = {cores[p] for p in held}
    if set(parents) & held:
        raise ValueError("A held parent entered the temporal fit")
    removed = {p for p in parents if cores[p] in held_cores}
    seed = crc32(f"T4scaffold|{fold}|{plate}".encode())
    count = set(np.random.default_rng(seed).choice(np.asarray(parents, dtype=str), size=len(removed), replace=False))
    purged = set(parents) - removed
    if len(purged) <= PHH_RANK or {cores[p] for p in purged} & held_cores:
        raise ValueError("Scaffold purge lacks independent rank-16 source support")
    return dict(full=set(parents), purged=purged, count=set(parents)-count), removed, count


def fit_effects(ctx, data, role):
    earlier, later, controls = (
        pd.concat([tables[name] for tables in data.values()], ignore_index=True).drop_duplicates()
        for name in ("source6", "source24", "fit_controls")
    )
    treated = pd.concat((earlier, later), ignore_index=True)
    controls = controls.loc[controls.rna_plate.isin(treated.rna_plate)]
    admitted = pd.concat((treated, controls), ignore_index=True)
    check_unique(admitted)
    control_axis(controls, treated, role)
    raw = extract_matrix(ctx.ref["matrix"], admitted, ctx.cols, ctx.genes)
    values = deltas(treated, raw[:len(treated)], controls, raw[len(treated):])
    return pd.Index(treated.inst_id), values


def predict(args, ctx):
    parent_sets, mapping = registry(ctx)
    data = group_plans(ctx, args.group)
    fold, role, *_ = axes(next(iter(data)))
    destination = args.forecasts / args.group
    if destination.exists():
        raise FileExistsError(destination)
    args.forecasts.mkdir(parents=True, exist_ok=True)
    index, values = fit_effects(ctx, data, role)
    held = parent_sets[fold]
    with TemporaryDirectory(dir=args.forecasts, prefix=".t4-scaffold-") as tmp:
        stage = Path(tmp)
        hashes, settings = {}, []
        for key, tables in data.items():
            late, early, query = [tables[n] for n in ("source24", "source6", "query")]
            plate = axes(key)[4]
            partitions, removed, count = cut(late.parent, held, mapping, fold, plate)
            early_values, late_values = [values[index.get_indexer(t.inst_id)] for t in (early, late)]
            old = original(ctx, key, query)
            query6 = old["identity6"]
            fits = {}
            for mode, members in partitions.items():
                masks = [t.parent.isin(members).to_numpy() for t in (early, late)]
                early_pairs = average_conditions(early.loc[masks[0]], early_values[masks[0]])
                fit = pairs(early_pairs, late.loc[masks[1]], late_values[masks[1]])
                operator = fit_time_operator(*fit, ctx.genes, rank=PHH_RANK, ridge=PHH_RIDGE, held_parents=held)
                gain = scalar(*fit)
                fits[f"{mode}_operator"] = operator.predict(query6, ctx.genes).astype("float32")
                fits[f"{mode}_scalar"] = (gain * query6).astype("float32")
                settings.append(dict(slice=key, fold=fold, role=role, plate=plate, mode=mode, fit_parent=operator.fit_parent_count,
                                     fit_conditions=len(fit[0]), removed=len(removed) if mode != "full" else 0,
                                     held_scaffold_in_fit=len({mapping[p] for p in members} & {mapping[p] for p in held}),
                                     scalar_gain=gain, removal_count_control=len(count)))
                if args.mean_shift and mode == "purged":
                    centered, diagonal, centered_gain, penalty = centered_fit(fit, query6)
                    fits["purged_mean_scalar"] = centered.astype("float32")
                    fits["purged_diagonal"] = diagonal.astype("float32")
                    settings[-1].update(centered_scalar_gain=centered_gain, diagonal_ridge=penalty)
            path = stage / f"{key}.npz"
            np.savez_compressed(path, **fits)
            hashes[key] = sha256(path)
        write_json(stage / "receipt.json", dict(ctx.stamp, group=args.group, slices=hashes, settings=settings))
        stage.rename(destination)


def cluster_effects(parent, mapping, rmse, sse, energy):
    groups, labels = pd.factorize([mapping[p] for p in parent], sort=True)
    if len(labels) < 2:
        raise ValueError("Scaffold uncertainty needs multiple components")
    rng = np.random.default_rng(191705)
    mass = np.bincount(groups)
    gain = np.bincount(groups, weights=rmse[:, 1]-rmse[:, 0])
    ref = np.bincount(groups, weights=rmse[:, 1])
    energy_sum = np.bincount(groups, weights=energy)
    sse_gain = np.bincount(groups, weights=sse[:, 1]-sse[:, 0])
    draw = rng.integers(len(labels), size=(BOOTSTRAPS, len(labels)))
    effect = np.stack((-gain[draw].sum(axis=1)/mass[draw].sum(axis=1),
                       sse_gain[draw].sum(axis=1)/energy_sum[draw].sum(axis=1)))
    relative = gain[draw].sum(axis=1)/ref[draw].sum(axis=1)
    return np.quantile(effect, [.005, .995], axis=1).T, np.quantile(relative, [.005, .995])


def summarize(rows, truth, arrays, mapping, comparisons=COMPARISONS):
    results = []
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, data) for arm, data in arrays.items()}
    parent = sorted(set(rows.parent))
    for arm, baseline in comparisons:
        moments = parent_summaries(rows, truth.shape[1], energy, errors[arm], errors[baseline])
        points, intervals, draws = paired_effects(*moments)
        low, high, _ = relative_interval(moments[0])
        cl_intervals, cl_relative = cluster_effects(parent, mapping, *moments)
        for mode, ci, rel in (("parent", intervals, (low, high)), ("scaffold", cl_intervals, cl_relative)):
            for metric, (a, b), (lo, hi), n in zip(METRICS, points, ci, draws, strict=True):
                results.append(dict(scope=mode, arm=arm, reference=baseline, metric=metric, arm_value=a, reference_value=b,
                    difference=a-b, low99=lo, high99=hi, bootstrap_draws=n, relative_gain=1-a/b if metric=="rmse" else np.nan,
                    relative_low99=rel[0] if metric=="rmse" else np.nan, relative_high99=rel[1] if metric=="rmse" else np.nan))
    return pd.DataFrame(results)


def collect(args, ctx):
    groups = sorted({key[:4] for key in ctx.ref["slices"]})
    hashes, queries, settings = {}, {}, []
    scoring = args.action == "score"
    arrays = {name: [] for name in ARMS + (MEAN_SHIFT_ARMS if args.mean_shift else ())}
    for group in groups:
        directory = args.forecasts / group
        receipt = read_json(directory / "receipt.json")
        keys = {key for key in ctx.ref["slices"] if key.startswith(group + "-")}
        if (any(receipt[k] != value for k, value in ctx.stamp.items()) or receipt["group"] != group
                or set(receipt["slices"]) != keys):
            raise ValueError("The scope of the scaffold forecast differs from its registered group")
        hashes[group] = sha256(directory / "receipt.json")
        settings.extend(receipt["settings"])
        for key, digest in sorted(receipt["slices"].items()):
            query = queries[key] = query_plan(ctx, key)
            old = original(ctx, key, query)
            path = directory / f"{key}.npz"
            if sha256(path) != digest:
                raise ValueError("The scaffold prediction array changed")
            with np.load(path, allow_pickle=False) as packed:
                if set(packed.files) != set(arrays) - set(old):
                    raise ValueError("Unexpected model inventory in the scaffold prediction")
                new = {name: checked_array(packed[name], old["identity6"].shape, name) for name in packed.files}
            if scoring:
                for name, parts in arrays.items():
                    parts.append((old | new)[name])
    return dict(ctx.stamp, groups=hashes), queries, settings, arrays


def evaluate(args, ctx):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.digest):
        raise ValueError("Separate scaffold scoring requires the freeze digest")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    stamp, queries, settings, arrays = collect(args, ctx)
    rows = checked_rows(pd.concat([q.assign(fold=axes(k)[0], role=axes(k)[1]) for k, q in queries.items()],
                                  ignore_index=True), None)
    if rows.parent.nunique() != ctx.ref["parents"] or rows.inst_id.nunique() != 268 or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("A registered T4 parent or well or vehicle rotation was lost")
    if not scoring:
        write_json(args.freeze, stamp)
        return
    if read_json(args.freeze) != stamp:
        raise ValueError("The candidates differ from the prospective freeze")
    truth = np.concatenate([score_delta(ctx.scope, ctx.ref, k, q, ctx.cols, ctx.genes) for k, q in queries.items()])
    _, mapping = registry(ctx)
    comparisons = COMPARISONS
    if args.mean_shift:
        comparisons += tuple(("purged_operator", baseline) for baseline in MEAN_SHIFT_ARMS)
    table = summarize(rows, truth, {arm: np.concatenate(parts) for arm, parts in arrays.items()}, mapping, comparisons)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-scaffold-score-") as tmp:
        stage = Path(tmp)
        table.to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(settings).to_csv(stage / "fit_support.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.digest, parents=len(mapping),
            scaffold_components=len(set(mapping.values())), wells=rows.inst_id.nunique(), plates=rows.rna_plate.nunique(),
            files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("predict", "freeze", "score"))
    for field in ("reports", "previous", "protocol", "forecasts", "freeze", "out"):
        cli.add_argument("--" + field, type=Path)
    cli.add_argument("--group")
    cli.add_argument("--digest")
    cli.add_argument("--mean-shift", action="store_true")
    args = cli.parse_args()
    ctx = inputs(args)
    (predict if args.action == "predict" else evaluate)(args, ctx)
