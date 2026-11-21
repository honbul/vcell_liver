"""Measure different-position reliability on already scored Tahoe development wells."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import MIN_GROUPS, WELL, eligibility, slice_name, verify_one
from src.goal_tracks.t1_io import score_truth
from src.goal_tracks.t1_score import STRATUM_LINES, registry
from src.p4.common import History, WellProjection, common_args, finish_args
from src.p4.score import binding, load_slice
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE, table
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.objectives import center
from src.r19_plan.hcc_scaffold import CASES
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, paired_effects, parent_summaries, squared_rows


DOSES = (.5, 5.)
MINIMUMS = (20, 200)
LINES = {name: STRATUM_LINES[name][0] for name in ("SNU-423", "HepG2-C3A")}
POSITION = "barcode_position"
PLATE = "plate_id"
CHEMISTRY = [GROUP, ENTITY, "dose_uM", "time_h", LINE]
MODELS = ("P4-sibling", "fair-State-Morgan-well-adaptation")
REPEATS = ("repeat025", "repeat1")


def contracts(args, dose):
    args.dose = dose
    root = args.archive/"p4/data"/f"d{dose:g}"
    source = args.out/f"d{dose:g}"/"freeze.json"
    digest = sha256(root/"freeze.json")
    scored = read_json(root/"scored/receipt.json")
    frozen = read_json(source)
    stage = read_json(args.stage/"receipt.json")
    if sha256(source) != digest or scored["freeze_sha256"] != digest or sha256(args.stage/"receipt.json") != frozen["stage_sha256"]:
        raise ValueError("A P4 freeze, stage, or original score receipt changed")
    return frozen, stage, dict(protocol_sha256=sha256(args.design), code_sha256=sha256(__file__),
                                original_freeze_sha256=digest, stage_sha256=frozen["stage_sha256"], dose_uM=dose)


def source_rows(args, stage, fold, dose, minimum):
    path = args.stage/f"f{fold}"/"score_rows.tsv"
    verify_one(args.stage, stage, path)
    rows = eligibility(table(path), minimum, dose)
    if rows[PLATE].eq("plate13").any() or not rows[GROUP].isin(registry(args.stage)).all():
        raise ValueError("A well from outside the exposed P4 cohort was encountered")
    return WellProjection(rows).rows


def pairs(rows):
    result = []
    contexts = dict(tuple(rows.groupby([LINE, PLATE], sort=True)))
    for _, conditions in rows.groupby(CHEMISTRY, sort=True):
        for _, target in conditions.iterrows():
            alternatives = conditions.loc[conditions[PLATE].ne(target[PLATE])]
            for _, donor in alternatives.iterrows():
                same = donor[POSITION] == target[POSITION]
                plate = donor[PLATE]
                allowed = contexts[target[LINE], plate]
                decoys = allowed.loc[allowed[GROUP].ne(target[GROUP])]
                if not same:
                    decoys = decoys.loc[decoys[POSITION].ne(target[POSITION])]
                sham = min(decoys[WELL], default="", key=lambda w: (crc32(f"HCC-repeat|{target[WELL]}|{plate}|{w}".encode()), w))
                result.append(dict(group=target[GROUP], entity=target[ENTITY], line_id=target[LINE],
                    relation="same" if same else "different", target_well=target[WELL], target_plate=target[PLATE],
                    target_position=target[POSITION], donor_well=donor[WELL], donor_plate=plate, donor_position=donor[POSITION],
                    decoy_well=sham))
    return result


def scope_support(data):
    records = []
    for (minimum, cell), rows in data.groupby(["minimum_cells", "line_id"], sort=True):
        key = ["fold", "target_well"]
        grouping = rows.groupby(key).relation.nunique()
        common = set(grouping.loc[grouping.eq(2)].index)
        both = rows.loc[pd.MultiIndex.from_frame(rows[key]).isin(common)]
        for name, group in [*rows.groupby("relation", sort=True), ("same_target_both", both)]:
            unique = group.groupby(key).decoy_well.apply(lambda w: w.ne("").all())
            records.append(dict(minimum_cells=minimum, line_id=cell, relation=name, parents=group.group.nunique(),
                entity_count=group.entity.nunique(), target_wells=group.target_well.nunique(), donor_wells=group.donor_well.nunique(),
                directed_pairs=len(group), donor_plates=group.donor_plate.nunique(), decoy_supported_wells=int(unique.sum()),
                above_ten_parent_floor=group.group.nunique() >= 10))
    return pd.DataFrame(records)


def audit(args):
    for dose in DOSES:
        _, stage, proof = contracts(args, dose)
        rows = []
        for minimum in MINIMUMS:
            for fold in range(3):
                subset = source_rows(args, stage, fold, dose, minimum)
                scope = subset.loc[subset[LINE].isin(LINES.values())]
                rows.extend(dict(record, fold=fold, minimum_cells=minimum) for record in pairs(scope))
        data = pd.DataFrame(rows)
        product = args.study/f"d{dose:g}"/"audit"
        product.mkdir(parents=True, exist_ok=False)
        data.to_csv(product/"pairs.tsv", sep="\t", index=False)
        scope_support(data).to_csv(product/"support.tsv", sep="\t", index=False)
        write_json(product/"receipt.json", dict(proof, metadata_only=True, expression_read=False,
                   files={p.name: sha256(p) for p in product.iterdir()}))


def vectors(link, axes, truth, predictors):
    rows, outcomes, arrays = [], [], {name: [] for name in (*REPEATS, "decoy025", *predictors)}
    support = []
    index = axes.set_index([LINE, WELL], verify_integrity=True).assign(idx=np.arange(len(axes))).idx
    for (line, target), target_pairs in link.groupby(["line_id", "target_well"], sort=True):
        pos = index.at[line, target]
        donor_arrays, decoy_arrays = [], []
        for _, links in target_pairs.groupby("donor_plate", sort=True):
            donor_arrays.append(truth[[index.at[line, well] for well in links.donor_well]].mean(axis=0))
            candidates = links.decoy_well.unique()
            if len(candidates) != 1:
                raise ValueError("A matched decoy is ambiguous within donor plate")
            if candidates[0]:
                decoy_arrays.append(truth[index.at[line, candidates[0]]])
        repeated = np.mean(donor_arrays, axis=0)
        valid = len(decoy_arrays) == len(donor_arrays)
        arrays["decoy025"].append(.25*np.mean(decoy_arrays, axis=0) if valid else np.zeros_like(repeated))
        support.append(valid)
        arrays["repeat025"].append(.25*repeated)
        arrays["repeat1"].append(repeated)
        for name, values in predictors.items():
            arrays[name].append(values[pos])
        outcomes.append(truth[pos])
        rows.append(axes.iloc[pos])
    return pd.DataFrame(rows), np.stack(outcomes), {k: np.stack(v) for k, v in arrays.items()}, np.array(support)


def cosine_stats(rows, truth, pred):
    energy = np.linalg.norm(truth, axis=1)*np.linalg.norm(pred, axis=1)
    dot = np.einsum("ij,ij->i", truth, pred)
    value = np.divide(dot, energy, out=np.full(len(rows), np.nan), where=energy>1e-12)
    by_parent = pd.Series(value).groupby(rows.parent.to_numpy()).mean().dropna().to_numpy()
    if not len(by_parent):
        return np.nan, np.nan, np.nan, 0
    draws = np.random.default_rng(191423).exponential(size=(10000, len(by_parent)))
    lo, hi = np.quantile((draws @ by_parent)/draws.sum(axis=1), [.005, .995])
    return float(by_parent.mean()), lo, hi, len(by_parent)


def aggregate(cases, minimum, coordinate, relation, line):
    tables, observations, predictions, masks, common = zip(*cases)
    rows = pd.concat(tables, ignore_index=True).rename(columns={GROUP: "parent", WELL: "inst_id"})
    truth = np.concatenate(observations)
    values = {name: np.concatenate([p[name] for p in predictions]) for name in predictions[0]}
    energy = squared_rows(truth)
    errors = {name: squared_rows(truth, value) for name, value in values.items()}
    results, direction = [], []
    for name, covers in (("all", np.ones(len(rows), bool)), ("decoy_matched", np.concatenate(masks)),
                         ("same_target_both", np.concatenate(common))):
        if not covers.any():
            continue
        selected = rows.loc[covers]
        meta = dict(minimum_cells=minimum, coordinate=coordinate, line_id=line, relation=relation, scope=name,
                    parents=selected.parent.nunique(), wells=selected.inst_id.nunique())
        for arm, error in errors.items():
            if (arm == "decoy025" and name != "decoy_matched") or arm=="B0":
                continue
            for ref in (("B0", "decoy025") if name == "decoy_matched" and arm == "repeat025" else ("B0",)):
                moments = parent_summaries(selected, truth.shape[1], energy[covers], error[covers], errors[ref][covers])
                points, ci, _ = paired_effects(*moments)
                lo_rel, hi_rel, _ = relative_interval(moments[0])
                for metric, (a, b), (lo, hi) in zip(METRICS, points, ci, strict=True):
                    results.append(dict(meta, arm=arm, reference=ref, metric=metric, arm_value=a, reference_value=b,
                        difference=a-b, low99=lo, high99=hi, relative_gain=1-a/b if metric=="rmse" and b>0 else np.nan,
                        relative_low99=lo_rel if metric=="rmse" else np.nan, relative_high99=hi_rel if metric=="rmse" else np.nan))
            mean, lo, hi, n = cosine_stats(selected, truth[covers], values[arm][covers])
            direction.append(dict(meta, arm=arm, cosine=mean, low99=lo, high99=hi, finite_parents=n))
    return results, direction


def common_difference(assembled):
    keys = ["fold", "role", LINE, WELL]
    results = []
    for minimum, coordinate, relation, line in sorted(assembled):
        equal = (minimum, coordinate, "same", line)
        distinct = (minimum, coordinate, "different", line)
        if relation != "same" or distinct not in assembled:
            continue
        vectors_ = []
        for key in (equal, distinct):
            cases = [item for item in assembled[key] if item[-1].any()]
            if not cases:
                break
            rows = pd.concat([x[0].loc[x[-1]] for x in cases], ignore_index=True)
            order = rows.sort_values(keys).index.to_numpy()
            truth = np.concatenate([x[1][x[-1]] for x in cases])[order]
            scores = {arm: np.concatenate([x[2][arm][x[-1]] for x in cases])[order] for arm in REPEATS}
            vectors_.append((rows.iloc[order].reset_index(drop=True), truth, scores))
        if len(vectors_) != 2:
            continue
        a, y, same = vectors_[0]
        b, z, different = vectors_[1]
        if not a[keys].equals(b[keys]) or not np.array_equal(y, z):
            raise ValueError("Same and different positions did not retain an identical target and outcome")
        rows = a.rename(columns={GROUP: "parent", WELL: "inst_id"})
        for arm in REPEATS:
            moments = parent_summaries(rows, y.shape[1], squared_rows(y), squared_rows(y, different[arm]), squared_rows(y, same[arm]))
            points, ci, _ = paired_effects(*moments)
            for metric, (p, q), (lo, hi) in zip(METRICS, points, ci, strict=True):
                results.append(dict(minimum_cells=minimum, coordinate=coordinate, line_id=line,
                    parents=rows.parent.nunique(), wells=rows.inst_id.nunique(), arm=arm, metric=metric,
                    different=p, same=q, difference=p-q, low99=lo, high99=hi))
    return pd.DataFrame(results)


def evaluate(args):
    product = args.study/f"d{args.dose:g}"
    frozen, _, proof = contracts(args, args.dose)
    original = History(args, args.dose)
    if any(frozen[k] != value for k, value in binding(args, original).items()):
        raise ValueError("The P4 code and registered benchmark no longer match")
    audited = read_json(product/"audit/receipt.json")
    if (not audited["metadata_only"] or audited["expression_read"] or any(audited[k] != v for k,v in proof.items())
            or any(sha256(product/"audit"/n) != h for n,h in audited["files"].items())):
        raise ValueError("The repeat audit differs from the registered structural Well data")
    link = table(product/"audit/pairs.tsv")
    pending = []
    evidence = {}
    for minimum in MINIMUMS:
        for fold, role in CASES:
            key = slice_name(args.dose, minimum, fold, role)
            rows, panel, preds, _, stamp = load_slice(args, original, minimum, fold, role)
            if stamp != frozen["tasks"][key] or sorted(preds) != frozen["arms"]:
                raise ValueError("An original P4 forecast changed")
            evidence[key] = stamp
            pending.append((minimum, fold, role, rows, panel, {k: preds[k] for k in ("B_pos-all", *MODELS)}))
    required = dict(proof, audit_sha256=sha256(product/"audit/receipt.json"), original_slices=evidence)
    if args.action == "lock":
        write_json(product/"freeze.json", required)
        return
    if sha256(product/"freeze.json") != args.expected_sha256 or read_json(product/"freeze.json") != required:
        raise ValueError("A score would use repeats before the independent prediction lock")
    assembled = {}
    for minimum, fold, role, rows, panel, preds in pending:
        partial = link.loc[link.minimum_cells.eq(minimum) & link.fold.eq(fold)]
        if partial.empty:
            continue
        full = score_truth(args.stage, args.dose, minimum, fold, role, rows, panel)
        projection = WellProjection(rows)
        target = projection.rows
        contrast = Contrast(target, MIN_GROUPS)
        shared = {name: center(contrast, projection.centered(preds[name])) for name in MODELS}
        native = center(contrast, full[projection.index] - preds["B_pos-all"][projection.index])
        projected = center(contrast, projection.centered(full))
        grouping = partial.groupby(["line_id", "target_well"]).relation.nunique()
        common = set(grouping.loc[grouping.eq(2)].index)
        for (line, relation), pairs_ in partial.groupby(["line_id", "relation"]):
            for name, truth, forecast in (("P4_line", projected, shared), ("native_less_Bpos", native, {})):
                forecasts = dict(forecast, B0=np.zeros_like(truth))
                out, measure, expected, supported = vectors(pairs_, target, truth, forecasts)
                matched = pd.MultiIndex.from_frame(out[[LINE, WELL]]).isin(common)
                out = out.assign(fold=fold, role=role)
                assembled.setdefault((minimum, name, relation, line), []).append((out, measure, expected, supported, matched))
    scores, directions = [], []
    for key, cases in assembled.items():
        score, cosine = aggregate(cases, *key)
        scores.extend(score)
        directions.extend(cosine)
    output = product/"score"
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=product, prefix=".hcc-repeat-score-") as temp:
        path = Path(temp)
        pd.DataFrame(scores).to_csv(path/"paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(directions).to_csv(path/"direction.tsv", sep="\t", index=False, na_rep="NA")
        common_difference(assembled).to_csv(path/"common_difference.tsv", sep="\t", index=False, na_rep="NA")
        write_json(path/"receipt.json", dict(required, freeze_sha256=args.expected_sha256,
                   files={p.name: sha256(p) for p in path.iterdir()}))
        path.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("action", choices=("audit", "lock", "score"))
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--dose", type=float, choices=DOSES)
    parser.add_argument("--expected-sha256")
    args = finish_args(parser.parse_args())
    (audit if args.action=="audit" else evaluate)(args)
