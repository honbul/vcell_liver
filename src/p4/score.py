"""Freeze aligned forecasts before scoring Tahoe line-specific responses."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import DOSES, MIN_CELLS, MIN_GROUPS, WELL, eligibility, slice_name
from src.goal_tracks.t1_io import score_truth
from src.goal_tracks.t1_score import STRATA, STRATUM_LINES, registry
from src.p123.p1_fair import CPU_MODELS
from src.p123.p1_score import ENDPOINTS, StudyBook
from src.p123.p123_io import load_prediction
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.objectives import center, row_products

from .common import History, NEW, NULL, WellProjection, common_args, finish_args


REFITS = {f"P4-refit-{name}" for name in CPU_MODELS}
BEST = "best-VC"
REPEATS = ("plate6-line-x0.25", "plate6-line-x1")
CI = (.005, .995)


def binding(args, history):
    directory = Path(__file__).parent
    return {"protocol_sha256": sha256(args.protocol), "stage_sha256": sha256(args.stage / "receipt.json"),
            "dose_uM": args.dose, "history": history.references,
            "code": {file: sha256(directory / file) for file in ("common.py", "predict.py", "score.py")}}


def load_slice(args, history, minimum, fold, role):
    part = slice_name(args.dose, minimum, fold, role)
    rows, panel, forecasts, evidence = history.forecasts(minimum, fold, role)
    _, source_hashes = history.source_selection(minimum, fold, role)
    settings = {}
    for lane, expected in (("chemical", set(NEW)), ("vc", REFITS)):
        path = args.out / "predictions" / lane / part
        stamp, compared, genes, arrays = load_prediction(
            path, args.stage, args.protocol, args.dose, minimum, fold, role)
        selection = stamp["selection"]
        if (expected != set(arrays) or not rows[KEY].equals(compared[KEY]) or not np.array_equal(panel, genes)
                or selection["historical"] != history.references or selection["source_receipts"] != source_hashes
                or selection["module_sha256"] != sha256(Path(__file__).with_name("predict.py"))):
            raise ValueError(f"P4 {lane} predictions differ from the fixed fit or score axes")
        forecasts.update(arrays)
        settings.update(selection["gains"])
        evidence[lane] = sha256(path / "receipt.json")
    return rows, panel, forecasts, settings, evidence


def freeze(args, history, output):
    tasks = {}
    universe = None
    for minimum in MIN_CELLS:
        for fold in range(3):
            for role in "AB":
                _, _, arms, _, evidence = load_slice(args, history, minimum, fold, role)
                if universe is not None and sorted(arms) != universe:
                    raise ValueError("The forecast universe changes across axes")
                universe = sorted(arms)
                tasks[slice_name(args.dose, minimum, fold, role)] = evidence
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {**binding(args, history), "tasks": tasks, "arms": universe})
    print(f"Frozen {len(tasks)} slices with {len(universe)} predictions each", flush=True)


def reference_names(arms, history):
    return {arm for arm in arms if arm in history.known or arm in REFITS or arm.startswith("fair-")}


def comparisons(points, boot, best):
    results = []
    for scope, rows in points.groupby("stratum"):
        values = rows.set_index("arm")
        available = sorted(best & set(values.index))
        for metric in ENDPOINTS:
            rmse = metric == "contrast_rmse"
            optimal = min if rmse else max
            reduce = np.minimum.reduce if rmse else np.maximum.reduce
            choices = {"B0": ("B0", boot[scope, "B0", metric])}
            if available:
                winner = optimal(available, key=lambda k: values.at[k, metric])
                choices[BEST] = (winner, reduce([boot[scope, arm, metric] for arm in available]))
            for arm, value in values.iterrows():
                for label, (winner, ref_draw) in choices.items():
                    if arm == "B0" or (label == BEST and arm in best):
                        continue
                    lo, hi = np.quantile(boot[scope, arm, metric] - ref_draw, CI)
                    results.append(dict(dose_uM=value.dose_uM, minimum_cells=value.minimum_cells,
                                        stratum=scope, arm=arm, reference=label, best_at_point=winner, metric=metric,
                                        difference=value[metric] - values.at[winner, metric], ci99_low=lo, ci99_high=hi,
                                        passes=bool(hi < 0 if rmse else lo > 0),
                                        primary=bool(arm in NEW and value.dose_uM == 5 and value.minimum_cells == 20
                                                     and scope in {"all", "liver"})))
    return pd.DataFrame(results)


def projected_repeat(rows, observed, forecasts):
    keys = [GROUP, LINE, "barcode_position"]
    index = rows.reset_index(names="row_position")
    a = index.loc[index.plate_id.eq("plate14")].set_index(keys, verify_integrity=True)
    b = index.loc[index.plate_id.eq("plate6")].set_index(keys, verify_integrity=True)
    paired = a.join(b[["row_position"]], how="inner", rsuffix="_plate6").reset_index()
    if paired.empty:
        return None
    target = eligibility(rows.iloc[paired.row_position], 0)
    if target.empty:
        return None
    pair = target.merge(paired[[*keys, "row_position", "row_position_plate6"]], on=keys, validate="one_to_one")
    target_i, donor_i = pair.row_position.to_numpy(), pair.row_position_plate6.to_numpy()
    donor = rows.iloc[donor_i].reset_index(drop=True)
    repeated = center(Contrast(donor, MIN_GROUPS), observed[donor_i])
    predictions = {k: v[target_i] for k, v in forecasts.items()}
    predictions.update({label: gain*repeated for label, gain in zip(REPEATS, (.25, 1.))})
    return target, observed[target_i], predictions


def repeat_reach(points, boot):
    result = []
    for scope, rows in points.groupby("stratum"):
        point = rows.set_index("arm")
        for metric in ENDPOINTS:
            sign = -1 if metric == "contrast_rmse" else 1
            base = point.at["B0", metric]
            base_draw = boot[scope, "B0", metric]
            for ref in REPEATS:
                denom = sign * (point.at[ref, metric] - base)
                draws = sign * (boot[scope, ref, metric] - base_draw)
                low, high = np.quantile(draws, CI)
                for arm in (*NEW, "proposed-v0", "proposed-v1"):
                    gain = sign * (point.at[arm, metric] - base)
                    left, right = (np.nan, np.nan)
                    if low > 0:
                        numerator = sign * (boot[scope, arm, metric] - base_draw)
                        ratio = np.divide(100*numerator, draws, out=np.full_like(draws, np.nan), where=draws != 0)
                        left, right = np.nanquantile(ratio, CI)
                    result.append(dict(dose_uM=5, minimum_cells=point.at[arm, "minimum_cells"], stratum=scope,
                                       parent_groups=point.at[arm, "parent_groups"], arm=arm, reference=ref, metric=metric,
                                       model_gain=gain, repeat_gain=denom, repeat_gain_ci99_low=low,
                                       repeat_gain_ci99_high=high, reach_pct=100*gain/denom if denom > 0 else np.nan,
                                       reach_ci99_low=left, reach_ci99_high=right, stable=bool(low > 0)))
    return pd.DataFrame(result)


def line_sets(rows):
    for name in STRATA:
        include = np.ones(len(rows), dtype=bool) if name == "all" else rows[LINE].isin(STRATUM_LINES[name]).to_numpy()
        yield name, include


def summary_rows(projection, observed, original, minimum, fold, role):
    rows = projection.rows
    metric = Contrast(rows, MIN_GROUPS)
    raw_contrast = center(metric, original[projection.index])
    line_contrast = center(metric, observed)
    raw = row_products(raw_contrast, raw_contrast)
    specific = row_products(line_contrast, line_contrast)
    for stratum, mask in line_sets(rows):
        support = projection.support[mask]
        yield dict(minimum_cells=minimum, fold=fold, role=role, stratum=stratum,
                   original_rows=int(mask.sum()), parent_groups=rows.loc[mask, GROUP].nunique(),
                   wells=rows.loc[mask, WELL].nunique(), all_lines=projection.distinct_lines(),
                   median_mean_lines=float(np.median(support)) if len(support) else np.nan,
                   min_mean_lines=int(support.min()) if len(support) else np.nan,
                   original_energy=raw[mask].sum()*observed.shape[1], projected_energy=specific[mask].sum()*observed.shape[1])


def contributions(cases, dose, minimum):
    energies = []
    for rows, truth, preds in cases:
        metric = Contrast(rows, MIN_GROUPS)
        observed = center(metric, truth)
        sst = row_products(observed, observed)*truth.shape[1]
        labels = rows[[GROUP, ENTITY, LINE]].copy().assign(sst=sst)
        for name in NEW:
            error = center(metric, preds[name])-observed
            labels[f"sse_{name}"] = row_products(error, error)*truth.shape[1]
        energies.append(labels)
    ledger = pd.concat(energies, ignore_index=True)
    groups, exclusions = [], []
    for scope, mask in line_sets(ledger):
        for arm in NEW:
            selected = ledger.loc[mask].assign(sse=ledger.loc[mask, f"sse_{arm}"])
            table = selected.groupby(GROUP, as_index=False).agg(
                entities=(ENTITY, lambda s: "|".join(sorted(set(s)))), rows_roles=(LINE, "size"),
                sst=("sst", "sum"), sse=("sse", "sum"))
            table["savings"] = table.sst-table.sse
            table = table.sort_values(["savings", GROUP], ascending=[False, True])
            meta = dict(dose_uM=dose, minimum_cells=minimum, stratum=scope, arm=arm)
            groups.append(table.assign(**meta))
            positive = table.loc[table.savings.gt(0)]
            net, total = table.savings.sum(), positive.savings.sum()
            for k in (0, 5, 10, 20):
                excluded = positive.head(k)
                values = removed_scores(cases, scope, arm, set(excluded[GROUP]))
                removed = excluded.savings.sum()
                exclusions.append(dict(**meta, **values, top_k=k, removed_parents=len(excluded),
                                       removed_ids="|".join(excluded[GROUP].astype(str)),
                                       positive_share_pct=100*removed/total if total > 0 else np.nan,
                                       net_share_pct=100*removed/net if net != 0 else np.nan))
    return pd.concat(groups, ignore_index=True), pd.DataFrame(exclusions)


def removed_scores(cases, scope, arm, excluded):
    mass, sst, sse, count = 0., 0., 0., 0
    rmse = np.zeros(2)
    parents = set()
    for rows, truth, pred in cases:
        keep = ~rows[GROUP].isin(excluded)
        if scope != "all":
            keep &= rows[LINE].isin(STRATUM_LINES[scope])
        selected = eligibility(rows.loc[keep].assign(_row=np.flatnonzero(keep)), 0)
        if selected.empty:
            continue
        idx = selected.pop("_row").to_numpy(dtype=int)
        metric = Contrast(selected, MIN_GROUPS)
        observed = center(metric, truth[idx])
        error = center(metric, pred[arm][idx])-observed
        t, r = row_products(observed, observed), row_products(error, error)
        rmse += np.sqrt((t, r)) @ metric.primary_weight
        sst, sse = sst+t.sum(), sse+r.sum()
        mass += metric.primary_weight.sum()
        parents.update(selected[GROUP])
        count += len(selected)
    return dict(parent_groups=len(parents), rows_roles=count, rmse_b0=rmse[0]/mass if mass else np.nan,
                rmse_model=rmse[1]/mass if mass else np.nan, r2_pooled=1-sse/sst if sst else np.nan)


def score(args, history, manifest):
    frozen = read_json(manifest)
    if any(frozen[key] != value for key, value in binding(args, history).items()):
        raise ValueError("P4 scoring provenance differs from the prediction freeze")
    groups = registry(args.stage)
    names = ("points", "paired", "support", "well_only", "selection", "group_contribution",
             "top_group_exclusion", "repeat_points", "repeat_reach")
    tables = {key: [] for key in names}
    for minimum in MIN_CELLS:
        cases = []
        book = StudyBook(groups, args.dose, minimum, "development")
        repeated = StudyBook(groups, args.dose, minimum, "plate14-repeat-supported") if args.dose == 5 else None
        for fold in range(3):
            for role in "AB":
                part = slice_name(args.dose, minimum, fold, role)
                rows, panel, forecasts, settings, evidence = load_slice(args, history, minimum, fold, role)
                if evidence != frozen["tasks"][part] or sorted(forecasts) != frozen["arms"]:
                    raise ValueError("Prediction changed after P4 freeze")
                original = score_truth(args.stage, args.dose, minimum, fold, role, rows, panel)
                projection = WellProjection(rows)
                observed = projection.centered(original)
                scored = {k: projection.centered(v) for k, v in forecasts.items()}
                if not np.allclose(scored["B0"], 0., atol=1e-12, rtol=0):
                    raise ValueError("B0 differs from a zero P4 response")
                scored[NULL] = scored["B0"]
                book.add(projection.rows, observed, scored)
                cases.append((projection.rows, observed, {k: scored[k] for k in NEW}))
                if repeated is not None:
                    arms = {k: scored[k] for k in ("B0", "B_pos-all", "proposed-v0", "proposed-v1", *NEW)}
                    paired = projected_repeat(projection.rows, observed, arms)
                    if paired is not None:
                        repeated.add(*paired)
                key = dict(dose_uM=args.dose, minimum_cells=minimum, fold=fold, role=role)
                support = summary_rows(projection, observed, original, minimum, fold, role)
                tables["support"].append(pd.DataFrame(support).assign(
                    dose_uM=args.dose, registered_wells=rows[WELL].nunique(), registered_rows=len(rows)))
                tables["well_only"].append(pd.DataFrame(
                    dict(**key, arm=k, projected_max_abs=float(np.max(np.abs(v))),
                         zero_equivalent=bool(np.max(np.abs(v)) <= 1e-12)) for k, v in scored.items()))
                tables["selection"].append(pd.DataFrame(dict(**key, arm=k, gain=v["gain"],
                                                            receipt_sha256=evidence["chemical" if k in NEW else "vc"])
                                                        for k, v in settings.items()))
                print(f"Scored {part} with {len(projection.rows)} projected rows", flush=True)
        points, boot = book.results()
        new = points.arm.isin({*NEW, *REFITS})
        points["evidence"] = np.where(new, "P4_refit", np.where(points.arm.isin({NULL, "B0"}), "zero_reference",
                                                                "posthoc_rescore"))
        tables["points"].append(points)
        tables["paired"].append(comparisons(points, boot, reference_names(frozen["arms"], history)))
        energy, removal = contributions(cases, args.dose, minimum)
        tables["group_contribution"].append(energy)
        tables["top_group_exclusion"].append(removal)
        if repeated is not None:
            rp, rb = repeated.results()
            tables["repeat_points"].append(rp)
            tables["repeat_reach"].append(repeat_reach(rp, rb))
    output = manifest.parent / "scored"
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix=".p4-score-") as temporary:
        folder = Path(temporary)
        for label, entries in tables.items():
            if entries:
                pd.concat(entries, ignore_index=True).to_csv(folder / f"{label}.tsv", sep="\t", index=False)
        primary = pd.concat(tables["paired"], ignore_index=True)
        primary = primary.loc[primary.primary].copy()
        key = ["minimum_cells", "stratum", "arm", "reference"]
        primary["both_pass"] = primary.groupby(key).passes.transform("all")
        primary.to_csv(folder / "decision.tsv", sep="\t", index=False)
        write_json(folder / "receipt.json", {"freeze_sha256": sha256(manifest),
                                              "files": {p.name: sha256(p) for p in sorted(folder.iterdir())}})
        folder.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("action", choices=("freeze", "score"))
    parser.add_argument("--dose", type=float, choices=DOSES, required=True)
    args = finish_args(parser.parse_args())
    history = History(args, args.dose)
    output = args.out / f"d{args.dose:g}" / "freeze.json"
    (freeze if args.action == "freeze" else score)(args, history, output)


if __name__ == "__main__":
    main()
