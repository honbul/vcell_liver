"""Freeze and independently score P1/P2 Tahoe arms and historical comparators."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import DOSES, MIN_CELLS, MIN_GROUPS, eligible_pools, slice_name
from src.goal_tracks.t1_io import load_task, score_truth
from src.goal_tracks.t1_score import DRAW_COUNT, STRATA, STRATUM_LINES, axis_draw, registry
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.objectives import center, row_products
from src.r6_followup_v1.metrics import grams

from .p123_io import load_prediction
from .p1_predict import REQUIRED


REFITS = {"ChemBERTa_R11", "FiLM_R12", "Concat_R12", "scGPT_C7", "UCE_C8"}
P2_ARMS = {"P2-Tanimoto", "P2-LINCS-cosine", "P2-Tanimoto-LINCS", "P2-LINCS-direct", "P2-LINCS-imputed"}
BASES = ("B0", "B_pos-all", "proposed-v0", "p1-trim", "proposed-v1")
ENDPOINTS = ("contrast_rmse", "r2_pooled")


def historical(args):
    root = args.prior.resolve()
    t1 = root / f"t1/first/d{args.dose:g}"
    t2 = root / f"t2/d{args.dose:g}"
    t1stamp, t2stamp = read_json(t1 / "freeze.json"), read_json(t2 / "freeze.json")
    actual = sha256(args.stage / "receipt.json")
    if (t1stamp["stage_sha256"] != actual or t2stamp["stage_sha256"] != actual
            or t2stamp["t1_freeze_sha256"] != sha256(t1 / "freeze.json")):
        raise ValueError("T1/T2 freeze does not agree with staged source or its linked baseline")
    roots = {k: Path(v) for k, v in t2stamp["eligible_known"].items()}
    if len(roots) != 10 or not REFITS.issubset(roots):
        raise ValueError("T2 comparative universe changed")
    return t1, t2, t1stamp, t2stamp, roots


def bind(args, prev):
    prior1, prior2, *_ = prev
    if args.track == "P2" and (len(args.extra_root) != 1 or args.coverage is None):
        raise ValueError("P2 needs one prediction root and its frozen LINCS coverage")
    if args.track == "P1-fair" and len(args.extra_root) != 10:
        raise ValueError("Fairness needs roots for all 10 T2 comparators")
    if args.track == "P1" and (args.extra_root or args.coverage is not None):
        raise ValueError("Native P1 uses only the registered prediction root")
    if args.track != "P2" and args.coverage is not None:
        raise ValueError("Only P2 adds LINCS coverage")
    return dict(
        track=args.track, dose_uM=args.dose, protocol_sha256=sha256(args.protocol),
        stage_manifest_sha256=sha256(args.stage / "receipt.json"),
        t1_source_freeze_sha256=sha256(prior1 / "freeze.json"),
        t2_source_freeze_sha256=sha256(prior2 / "freeze.json"),
        p1_root=str(args.p1_root.resolve()), extra_root=[str(p.resolve()) for p in args.extra_root],
        coverage_sha256=sha256(args.coverage) if args.coverage else None,
        scorer_code_sha256=sha256(Path(__file__)),
        reader_code_sha256=sha256(Path(__file__).with_name("p123_io.py")),
    )


def check_arms(track, previous_names, new, extra):
    if new != REQUIRED:
        raise ValueError(f"P1 raw arms differ from the fixed set: {sorted(new)}")
    expected = set()
    if track == "P1-fair":
        expected = {f"fair-{name}" for name in previous_names} | {f"fair-refit-{name}" for name in REFITS}
    if track == "P2":
        expected = P2_ARMS
    if extra != expected:
        raise ValueError(f"Additional prediction arms differ from the fixed set: {sorted(extra ^ expected)}")


def load_case(args, prev, n, fold, role, sealed=None):
    t1, _, a, b, roots = prev
    part = slice_name(args.dose, n, fold, role)
    case = t1 / "predictions" / part
    contract = args.prior / "t1-code/reports/goal_tracks/PROTOCOL.md"
    record = {"T1": sha256(case / "receipt.json")}
    if record["T1"] != a["tasks"][part] or record["T1"] != b["tasks"][part]["T1"]:
        raise ValueError("The two earlier studies did not bind the same T1 forecasts")
    _, rows, panel, arms = load_task(case, args.stage, contract, args.dose, n, fold, role)

    def merge(new_rows, new_panel, predictions):
        if not rows[KEY].equals(new_rows[KEY]) or not np.array_equal(panel, new_panel):
            raise ValueError("Arm does not use a common held data axis")
        if set(arms) & set(predictions):
            raise ValueError("Duplicate prediction name")
        arms.update(predictions)

    for name, root in roots.items():
        path = root / part
        record[name] = sha256(path / "receipt.json")
        if record[name] != b["tasks"][part][name]:
            raise ValueError(f"Earlier T2 forecast changed: {name}")
        _, new_rows, new_panel, pred = load_task(path, args.stage, contract, args.dose, n, fold, role)
        if set(pred) != {name}:
            raise ValueError("T2 task name differs from previous freeze")
        merge(new_rows, new_panel, pred)
    path = args.p1_root / part
    record["P1"] = sha256(path / "receipt.json")
    stamp, new_rows, new_panel, new = load_prediction(path, args.stage, args.protocol, args.dose, n, fold, role)
    if stamp["selection"]["t1_freeze_sha256"] != sha256(t1 / "freeze.json"):
        raise ValueError("P1 used another historical anchor")
    merge(new_rows, new_panel, new)
    extra = set()
    for k, root in enumerate(args.extra_root):
        path = root / part
        record[f"extra-{k}"] = sha256(path / "receipt.json")
        _, new_rows, new_panel, pred = load_prediction(path, args.stage, args.protocol, args.dose, n, fold, role)
        extra.update(pred)
        merge(new_rows, new_panel, pred)
    check_arms(args.track, set(roots), set(new), extra)
    if sealed is not None and record != sealed["tasks"][part]:
        raise ValueError("A prediction receipt changed after freeze")
    return part, record, stamp, rows, panel, arms


def freeze(args):
    prev = historical(args)
    stamp = bind(args, prev)
    records = {}
    universe = None
    for n in MIN_CELLS:
        for fold in range(3):
            for role in "AB":
                part, record, _, _, _, arms = load_case(args, prev, n, fold, role)
                current = sorted(arms)
                if universe is not None and universe != current:
                    raise ValueError("Arm universe differs across slabs")
                universe = current
                records[part] = record
    args.freeze.parent.mkdir(parents=True, exist_ok=True)
    if args.freeze.exists():
        raise FileExistsError(args.freeze)
    write_json(args.freeze, dict(stamp, tasks=records, arms=universe, existing_vc=sorted(prev[-1])))
    print(f"Froze {len(records)} slabs, {len(universe)} arms.", flush=True)


def pool_weights(rows, metric):
    w = np.zeros((len(STRATA), len(rows)), dtype=np.float64)
    w[0] = metric.primary_weight
    for k, stratum in enumerate(STRATA[1:], 1):
        include = rows[LINE].isin(STRATUM_LINES[stratum]).to_numpy()
        if include.any():
            w[k, include] = group_weights(rows.loc[include])
    return w


def bootstrap_error(rows, metric, g, draw, entity_index, weights):
    rmse = np.zeros((len(STRATA), len(draw)))
    sse = np.zeros_like(rmse)
    lines = rows[LINE].to_numpy()
    for kernel, ix, center_weight in zip(g, metric.pools, metric.centers):
        positive = draw[:, entity_index[ix]]
        normalized = positive * center_weight
        normalized /= normalized.sum(axis=1, keepdims=True)
        products = normalized @ kernel
        variance = np.maximum(np.diag(kernel)-2*products+np.sum(products*normalized, axis=1, keepdims=True), 0)
        root = np.sqrt(variance) * positive
        energy = np.einsum("ij,ij->i", positive, variance)
        rmse[0] += root @ weights[0, ix]
        sse[0] += energy
        for local, liver in enumerate(STRATUM_LINES["liver"]):
            if lines[ix[0]] == liver:
                rmse[1] += root @ weights[1, ix]
                sse[1] += energy
                rmse[local+2] += root @ weights[local+2, ix]
                sse[local+2] += energy
    return rmse, sse


class StudyBook:
    """Compute macro RMSE and row-energy R² using paired parent re-centering."""

    def __init__(self, groups, dose, n, cohort):
        self.index, self.draw = axis_draw(groups)
        self.dose, self.n, self.cohort = dose, n, cohort
        self.point = defaultdict(lambda: np.zeros(len(STRATA)))
        self.point_sse = defaultdict(lambda: np.zeros(len(STRATA)))
        self.boot_rmse = defaultdict(lambda: np.zeros((len(STRATA), DRAW_COUNT)))
        self.boot_sse = defaultdict(lambda: np.zeros((len(STRATA), DRAW_COUNT)))
        self.mass = np.zeros(len(STRATA))
        self.mass_boot = np.zeros((len(STRATA), DRAW_COUNT))
        self.counts = np.zeros(len(STRATA), dtype=int)
        self.compounds = [set() for _ in STRATA]

    def add(self, rows, obs, arms):
        rows = rows.reset_index(drop=True)
        metric = Contrast(rows, MIN_GROUPS)
        if len(metric.active) != len(rows):
            raise ValueError("A pool lacks minimum parent support")
        index = self.index.get_indexer(rows[GROUP])
        if np.any(index < 0):
            raise ValueError("Unregistered parent in evaluated rows")
        weights = pool_weights(rows, metric)
        keep = weights > 0
        self.mass += weights.sum(axis=1)
        self.counts += keep.sum(axis=1)
        for k in range(len(STRATA)):
            self.compounds[k].update(rows.loc[keep[k], GROUP])
            self.mass_boot[k] += self.draw @ np.bincount(index, weights=weights[k], minlength=len(self.index))
        for arm, pred in arms.items():
            err = np.asarray(pred, dtype=np.float64) - obs
            centered = center(metric, err)
            squared = row_products(centered, centered)
            self.point[arm] += weights @ np.sqrt(squared)
            self.point_sse[arm] += keep @ squared
            rmse, energy = bootstrap_error(rows, metric, grams(err, metric.pools), self.draw, index, weights)
            self.boot_rmse[arm] += rmse
            self.boot_sse[arm] += energy

    def results(self):
        values, samples = [], {}
        sst = self.point_sse["B0"]
        sst_boot = self.boot_sse["B0"]
        for arm in self.point:
            for k, stratum in enumerate(STRATA):
                if self.mass[k] <= 0 or sst[k] <= 0:
                    continue
                raw = self.point[arm][k] / self.mass[k]
                r2 = 1 - self.point_sse[arm][k] / sst[k]
                a = self.boot_rmse[arm][k] / self.mass_boot[k]
                b = 1 - self.boot_sse[arm][k] / sst_boot[k]
                samples[stratum, arm, "contrast_rmse"] = a
                samples[stratum, arm, "r2_pooled"] = b
                row = dict(dose_uM=self.dose, minimum_cells=self.n, cohort=self.cohort, stratum=stratum,
                           arm=arm, parent_groups=len(self.compounds[k]), rows_roles=int(self.counts[k]),
                           contrast_rmse=raw, r2_pooled=r2, sse=self.point_sse[arm][k]*2000,
                           sst=sst[k]*2000)
                for label, draws in zip(ENDPOINTS, (a, b)):
                    low, high = np.quantile(draws, (0.005, 0.995))
                    row.update({f"{label}_ci99_low": low, f"{label}_ci99_high": high})
                values.append(row)
        return pd.DataFrame(values), samples


def formal(track, row, tested, reference):
    if row.dose_uM != 5 or row.minimum_cells != 20:
        return False
    if track == "P1" and tested == "proposed-v1":
        return row.stratum in {"all", "liver"} and reference in {"p1-trim", "B0", "best-T2"}
    if track == "P1-fair" and tested == "proposed-v1":
        return row.stratum in {"all", "liver"} and reference in {"best-fair", "best-fair-refit", "best-fair-all"}
    if track == "P2":
        return (tested == "P2-Tanimoto-LINCS" and reference == "P2-Tanimoto"
                and row.stratum == "liver" and row.cohort in {"full", "lincs_observed"})
    return False


def pair_rows(points, boot, track, known):
    family = {"best-T2": known}
    if track == "P1-fair":
        family["best-fair"] = {f"fair-{x}" for x in known}
        family["best-fair-refit"] = {f"fair-refit-{x}" for x in REFITS}
        family["best-fair-all"] = family["best-fair"] | family["best-fair-refit"]
    candidates = set(REQUIRED)
    if track == "P1-fair":
        candidates |= family["best-fair-all"]
    if track == "P2":
        candidates |= P2_ARMS
    references = set(BASES) | set(family)
    if track == "P2":
        references |= {"P2-Tanimoto"}
    result = []
    for stratum, group in points.groupby("stratum", sort=False):
        info = group.set_index("arm")
        labels = set(info.index)
        for metric in ENDPOINTS:
            use_min = metric == "contrast_rmse"
            select = min if use_min else max
            fold_op = np.minimum if use_min else np.maximum
            record = {}
            for ref in sorted(references):
                r = family.get(ref, {ref})
                if not r <= labels:
                    continue
                picked = select(sorted(r), key=lambda x: info.at[x, metric])
                curve = fold_op.reduce([boot[stratum, x, metric] for x in sorted(r)])
                record[ref] = (picked, info.at[picked, metric], curve)
            for arm in sorted(candidates & labels):
                row = next(group.loc[group.arm.eq(arm)].itertuples(index=False))
                for ref, (picked, point, curve) in record.items():
                    if arm == ref or arm in family.get(ref, ()):
                        continue
                    diffs = boot[stratum, arm, metric] - curve
                    lo, hi = np.quantile(diffs, (.005, .995))
                    improvement = (hi < 0 if use_min else lo > 0)
                    is_formal = formal(track, row, arm, ref)
                    result.append(dict(
                        dose_uM=row.dose_uM, minimum_cells=row.minimum_cells, cohort=row.cohort,
                        stratum=stratum, arm=arm, reference=ref, best_reference_at_point=picked,
                        metric=metric, difference=float(info.at[arm, metric] - point),
                        ci99_low=lo, ci99_high=hi, improvement=bool(improvement),
                        separates_zero=bool(lo > 0 or hi < 0), formal=is_formal,
                    ))
    return pd.DataFrame(result)


def coverage(path, groups):
    raw = table(path)
    key = GROUP if GROUP in raw else "group"
    if key not in raw or "has_pan_anydose" not in raw:
        raise ValueError("LINCS coverage needs the canonical group key and has_pan_anydose")
    case = raw.set_index(key, verify_integrity=True).reindex(groups)["has_pan_anydose"]
    value = case.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if value.isna().any():
        raise ValueError("Incomplete canonical parent coverage manifest")
    return value


def cohort_indices(rows, status):
    if status is None:
        return {"full": np.arange(len(rows))}
    known = rows[GROUP].map(status)
    if known.isna().any():
        raise ValueError("Outer parent missing from coverage registry")
    return {"full": np.arange(len(rows)),
            "lincs_observed": np.flatnonzero(known),
            "lincs_unobserved": np.flatnonzero(~known)}


def cohort_eligible(rows, indices):
    use = rows.iloc[indices].copy().assign(_p123_index=indices)
    if len(use):
        use = eligible_pools(use, MIN_GROUPS)
    return use.drop(columns="_p123_index"), use["_p123_index"].to_numpy(dtype=int)


def per_group_energy(cases):
    result = []
    for rows, observed, preds in cases:
        m = Contrast(rows, MIN_GROUPS)
        outcome = center(m, observed)
        trimmed = center(m, preds["p1-trim"])
        combined = center(m, preds["proposed-v1"])
        loss0, loss1 = trimmed-outcome, combined-outcome
        e = rows[[GROUP, ENTITY, LINE]].copy()
        e["sse_trim"] = row_products(loss0, loss0)*observed.shape[1]
        e["sse_v1"] = row_products(loss1, loss1)*observed.shape[1]
        e["sst"] = row_products(outcome, outcome)*observed.shape[1]
        result.append(e)
    raw = pd.concat(result, ignore_index=True)
    raw["savings"] = raw.sse_trim-raw.sse_v1
    for stratum in STRATA:
        subset = raw if stratum == "all" else raw[raw[LINE].isin(STRATUM_LINES[stratum])]
        grouped = subset.groupby(GROUP, as_index=False).agg(
            entities=(ENTITY, lambda x: "|".join(sorted(set(x)))),
            rows_roles=(ENTITY, "size"), sse_trim=("sse_trim", "sum"), sse_v1=("sse_v1", "sum"),
            sst=("sst", "sum"), savings=("savings", "sum"),
        ).sort_values(["savings", GROUP], ascending=[False, True])
        yield stratum, grouped


def exclusion_points(cases, excluded, stratum):
    results = defaultdict(lambda: np.zeros(2))
    total_mass = 0.0
    sst = 0.0
    count = 0
    groups = set()
    for rows, obs, preds in cases:
        mask = ~rows[GROUP].isin(excluded)
        if stratum != "all":
            mask &= rows[LINE].isin(STRATUM_LINES[stratum])
        case, indices = cohort_eligible(rows, np.flatnonzero(mask))
        if not len(case):
            continue
        metric = Contrast(case, MIN_GROUPS)
        truth = center(metric, obs[indices])
        sst += float(np.square(truth).sum())
        total_mass += metric.primary_weight.sum()
        count += len(case)
        groups.update(case[GROUP])
        for name in BASES:
            err = center(metric, preds[name][indices]) - truth
            row_mse = row_products(err, err)
            results[name] += (np.dot(np.sqrt(row_mse), metric.primary_weight),
                              row_mse.sum()*truth.shape[1])
    for arm, (rmse, sse) in results.items():
        yield dict(arm=arm, contrast_rmse=rmse/total_mass, r2_pooled=1-sse/sst,
                   parent_groups=len(groups), rows_roles=count)


def contributions(cases, dose, n):
    energy, exclusions = [], []
    for scope, grouped in per_group_energy(cases):
        full = grouped.savings.sum()
        positive = grouped[grouped.savings.gt(0)]
        positive_sum = positive.savings.sum()
        energy.append(grouped.assign(dose_uM=dose, minimum_cells=n, stratum=scope,
                                     signed_share=grouped.savings/full if full != 0 else np.nan))
        for count in (0, 5, 10, 20):
            removed = positive.iloc[:count]
            keys = set(removed[GROUP])
            mass = removed.savings.sum()
            for row in exclusion_points(cases, keys, scope):
                exclusions.append(dict(row, dose_uM=dose, minimum_cells=n, stratum=scope, top_k=count,
                    removed_positive_parents=len(keys), removed_ids=",".join(map(str, sorted(keys))),
                    savings_removed_positive_pct=100*mass/positive_sum if positive_sum > 0 else np.nan,
                    savings_removed_net_pct=100*mass/full if full != 0 else np.nan))
    return pd.concat(energy, ignore_index=True), pd.DataFrame(exclusions)


def score(args):
    import time

    prev = historical(args)
    frozen = read_json(args.freeze)
    stamp = bind(args, prev)
    if any(frozen[key] != expect for key, expect in stamp.items()):
        raise ValueError("Frozen assumptions changed")
    groups = registry(args.stage)
    status = coverage(args.coverage, groups) if args.coverage else None
    args.out.mkdir(parents=True, exist_ok=False)
    points, intervals, energy, excluded, selection = [], [], [], [], []
    for n in MIN_CELLS:
        books = {}
        bound = set()
        cases = []
        for fold in range(3):
            for role in "AB":
                part, _, chosen, rows, panel, arms = load_case(args, prev, n, fold, role, frozen)
                if sorted(arms) != frozen["arms"]:
                    raise ValueError("Arm order changed after freeze")
                obs = score_truth(args.stage, args.dose, n, fold, role, rows, panel,
                                  verify=(fold, role) not in bound)
                bound.add((fold, role))
                if args.track == "P1":
                    cases.append((rows, obs, {k: arms[k] for k in BASES}))
                    selection.append(dict(dose_uM=args.dose, minimum_cells=n, fold=fold, role=role,
                                          winner=chosen["selection"]["winner"],
                                          choices=str(chosen["selection"]["chosen"])))
                for cohort, ix in cohort_indices(rows, status).items():
                    part_rows, ids = cohort_eligible(rows, ix)
                    if not len(ids):
                        continue
                    if cohort not in books:
                        books[cohort] = StudyBook(groups, args.dose, n, cohort)
                    observed = obs if len(ids) == len(obs) else obs[ids]
                    chosen_arms = arms
                    if args.track != "P1":
                        use = set(BASES) | set(prev[-1]) | (P2_ARMS if args.track == "P2"
                                                          else {a for a in arms if a.startswith("fair-")})
                        chosen_arms = {k: v for k, v in arms.items() if k in use}
                    subset = chosen_arms if len(ids) == len(obs) else {k: v[ids] for k, v in chosen_arms.items()}
                    books[cohort].add(part_rows, observed, subset)
                print(f"Scored {part}, t={time.monotonic():.1f}", flush=True)
        for book in books.values():
            table_points, boot = book.results()
            table_points["evidence"] = np.where(table_points.arm.isin(prev[-1]) |
                                                table_points.arm.isin({"B0", "B_abs", "B_pos", "B_pos-all", "proposed-v0"}) |
                                                ~table_points.arm.str.startswith(("p1-", "proposed-v1", "P2-", "fair-")),
                                                "posthoc_T1_T2", "P123")
            points.append(table_points)
            intervals.append(pair_rows(table_points, boot, args.track, set(prev[-1])))
        if cases:
            a, b = contributions(cases, args.dose, n)
            energy.append(a)
            excluded.append(b)
    main = pd.concat(points, ignore_index=True)
    pairs = pd.concat(intervals, ignore_index=True)
    main.to_csv(args.out / "points.tsv", sep="\t", index=False)
    pairs.to_csv(args.out / "paired.tsv", sep="\t", index=False)
    gate = pairs.loc[pairs.formal].copy()
    if not gate.empty:
        key = [c for c in ("dose_uM", "minimum_cells", "cohort", "stratum", "arm", "reference")]
        ok = gate.groupby(key)["improvement"].all().rename("both_metrics")
        gate = gate.merge(ok, left_on=key, right_index=True, validate="many_to_one")
    gate.to_csv(args.out / "decision.tsv", sep="\t", index=False)
    if energy:
        pd.concat(energy, ignore_index=True).to_csv(args.out / "group_contribution.tsv", sep="\t", index=False)
        pd.concat(excluded, ignore_index=True).to_csv(args.out / "top_group_exclusion.tsv", sep="\t", index=False)
        pd.DataFrame(selection).to_csv(args.out / "selection.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(freeze_sha256=sha256(args.freeze),
               files={p.name: sha256(p) for p in sorted(args.out.iterdir())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    parser.add_argument("--track", choices=("P1", "P1-fair", "P2"), required=True)
    parser.add_argument("--dose", choices=DOSES, type=float, required=True)
    for key in ("protocol", "prior", "stage", "p1-root", "freeze"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "score" and args.out is None:
        parser.error("score requires --out")
    (freeze if args.action == "freeze" else score)(args)


if __name__ == "__main__":
    main()
