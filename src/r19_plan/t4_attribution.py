"""Evaluate fit-mean and exactly dose-matched wrong-parent controls for T4."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_access import CONDITION, PREDICT
from src.r19_plan.t4_workflow import (
    average_conditions, axes, deltas, lookup, pairs, plans, reference, score_delta, validate_predict,
)
from src.r19_plan.time_operator import fit_time_operator
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes

ORIGINAL_SHA256 = "a6a931d6b0c4b88b6adae8244ad41c9e5e5e464b50286bed9728da0b5255febe"
ARMS = ("operator6", "mean24", "same_dose_swap6")


def shuffle_axis(conditions):
    """Build one derangement per dose, with singleton doses explicitly unsupported."""
    mapping = {}
    for dose, rows in conditions.groupby("actual_dose_uM", sort=True):
        parents = sorted(set(rows.parent), key=lambda p: (crc32(f"T4-null|{p}|{dose}".encode()), p))
        if len(parents) > 1:
            mapping.update({(p, dose): alt for p, alt in zip(parents, parents[1:] + parents[:1])})
    return mapping


def lineage(args):
    if sha256(args.original) != ORIGINAL_SHA256:
        raise ValueError("T4 attribution requires the original tested prediction freeze")
    old = read_json(args.original)
    if old["access_sha256"] != sha256(args.pin):
        raise ValueError("T4 attribution uses a different access axis from the original")
    here = Path(__file__)
    hashes = {name: sha256(here.with_name(name)) for name in (here.name, "t4_workflow.py", "time_operator.py")}
    return dict(access_sha256=sha256(args.pin), original_sha256=ORIGINAL_SHA256,
                protocol_sha256=sha256(args.protocol), code_sha256=hashes), old


def donor_profiles(ref, data, columns, genes, role):
    donors = pd.concat([plan["donor6"] for plan in data.values()], ignore_index=True).drop_duplicates()
    vehicles = pd.concat([plan["fit_controls"] for plan in data.values()], ignore_index=True).drop_duplicates()
    vehicles = vehicles.loc[vehicles.rna_plate.isin(donors.rna_plate)]
    check_unique(pd.concat((donors, vehicles), ignore_index=True))
    control_axis(vehicles, donors, role)
    raw = extract_matrix(ref["matrix"], pd.concat((donors, vehicles), ignore_index=True), columns, genes)
    return average_conditions(donors, deltas(donors, raw[:len(donors)], vehicles, raw[len(donors):]))


def recover_original(args, key, old, plan, ref, columns, genes, donors):
    folder = args.original_forecasts/key
    receipt = read_json(folder/"receipt.json")
    if (sha256(folder/"receipt.json") != old["slices"][key]
            or any(sha256(folder/file) != digest for file, digest in receipt["files"].items())):
        raise ValueError("A T4 source forecast changed since scoring")
    setting = read_json(folder/"selection.json")
    later, earlier = plan["source24"], plan["source6"]
    source = pd.concat((earlier, later), ignore_index=True)
    controls = plan["fit_controls"].loc[plan["fit_controls"].rna_plate.isin(source.rna_plate)]
    raw = extract_matrix(ref["matrix"], pd.concat((source, controls), ignore_index=True), columns, genes)
    deltas_source = deltas(source, raw[:len(source)], controls, raw[len(source):])
    source6 = average_conditions(earlier, deltas_source[:len(earlier)])
    fit = pairs(source6, later, deltas_source[len(earlier):])
    query = plan["query"]
    model = fit_time_operator(*fit, genes, rank=setting["rank"], ridge=setting["ridge"], held_parents=set(query.parent))
    matched = model.predict(lookup(donors, query), genes).astype("float32")
    with np.load(folder/"forecasts.npz", allow_pickle=False) as values:
        if not np.array_equal(matched, values["operator6"]):
            raise ValueError("A reconstructed operator failed the original frozen forecast equivalence check")
    return model, matched


def predict(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    ref = reference(args)
    upstream, old = lineage(args)
    use = sorted(k for k in ref["slices"] if k.startswith(args.slice+"-"))
    if not use:
        raise ValueError("Unknown T4 attribution fold and rotation")
    role = axes(use[0])[1]
    tables = {k: plans(args, ref, k, "predict", PREDICT) for k in use}
    for k in use:
        validate_predict(tables[k], k)
    gene_table, columns = matrix_axes(args.access, ref)
    genes = gene_table.pr_gene_id.to_numpy()
    bank = donor_profiles(ref, tables, columns, genes, role)
    conditions = pd.concat([plan["query"][CONDITION] for plan in tables.values()], ignore_index=True)
    shuffle = shuffle_axis(conditions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-attr-") as temp:
        stage = Path(temp)
        slices = {}
        for key, plan in tables.items():
            query = plan["query"]
            model, matched = recover_original(args, key, old, plan, ref, columns, genes, bank)
            mean = np.broadcast_to(model.output_mean, matched.shape).astype("float32")
            alternatives = query.assign(parent=[shuffle.get(pair, "") for pair in
                                        query[CONDITION].itertuples(index=False, name=None)])
            supported = alternatives.parent.ne("").to_numpy()
            swapped = mean.copy()
            if supported.any():
                swapped[supported] = model.predict(lookup(bank, alternatives.loc[supported]), genes)
            path = stage/(key+".npz")
            np.savez_compressed(path, operator6=matched, mean24=mean, same_dose_swap6=swapped, supported=supported)
            slices[key] = sha256(path)
        write_json(stage/"receipt.json", dict(upstream, group=args.slice, slices=slices))
        stage.rename(args.out)
    return sha256(args.out/"receipt.json")


def comparisons(rows, truth, predictions, support):
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, values) for arm, values in predictions.items()}
    result = []
    subsets = (("ALL", np.ones(len(rows), bool), ("mean24",)), ("same_dose", support, ARMS[1:]),
               ("10_uM", support & rows.actual_dose_uM.eq("10").to_numpy(), ARMS[1:]),
               ("2.5_uM", support & rows.actual_dose_uM.eq("2.5").to_numpy(), ARMS[1:]))
    for name, mask, arms in subsets:
        for arm in arms:
            subset = rows.loc[mask]
            stats = parent_summaries(subset, truth.shape[1], energy[mask], errors["operator6"][mask], errors[arm][mask])
            values, intervals, draws = paired_effects(*stats)
            gain_lo, gain_hi, _ = relative_interval(stats[0])
            for metric, (a, b), (lo, hi), n in zip(METRICS, values, intervals, draws, strict=True):
                result.append(dict(scope=name, arm="operator6", reference=arm, metric=metric, arm_value=a,
                    reference_value=b, difference=a-b, low99=lo, high99=hi, parents=subset.parent.nunique(),
                    wells=subset.inst_id.nunique(), doses=subset.actual_dose_uM.nunique(), valid_bootstraps=n,
                    relative_gain=1-a/b if metric == "rmse" and b > 0 else np.nan,
                    relative_low99=gain_lo if metric=="rmse" else np.nan, relative_high99=gain_hi if metric=="rmse" else np.nan))
    return pd.DataFrame(result)


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("New T4 attribution score requires its separate freeze digest")
    ref = reference(args)
    upstream, old = lineage(args)
    genes_frame, columns = matrix_axes(args.access, ref)
    genes = genes_frame.pr_gene_id.to_numpy()
    digests, seen, queries, support, forecasts = {}, set(), {}, [], {arm: [] for arm in ARMS}
    for group in sorted({"-".join(key.split("-", 2)[:2]) for key in ref["slices"]}):
        path = args.forecasts/group
        receipt = read_json(path/"receipt.json")
        expect = {key for key in ref["slices"] if key.startswith(group+"-")}
        if (any(receipt[k] != v for k, v in upstream.items()) or receipt["group"] != group
                or set(receipt["slices"]) != expect or seen & expect):
            raise ValueError("Unexpected scope in T4 attribution forecast")
        for key, digest in sorted(receipt["slices"].items()):
            file = path/(key+".npz")
            if sha256(file) != digest:
                raise ValueError("A T4 attribution forecast file changed")
            query = queries[key] = plans(args, ref, key, "predict", PREDICT)["query"]
            with np.load(file, allow_pickle=False) as response:
                if set(response.files) != set(ARMS) | {"supported"}:
                    raise ValueError("T4 attribution arm inventory changed")
                available = response["supported"]
                if available.dtype != bool or available.shape != (len(query),):
                    raise ValueError("T4 attribution support mask has the wrong row axis")
                support.append(available)
                for arm in ARMS:
                    array = checked_array(response[arm], (len(query), len(genes)), arm)
                    if scoring:
                        forecasts[arm].append(array)
        seen |= expect
        digests[group] = sha256(path/"receipt.json")
    if seen != set(old["slices"]):
        raise ValueError("Some original T4 score slices are missing")
    rows = checked_rows(pd.concat([q.assign(fold=axes(k)[0], role=axes(k)[1])
                        for k, q in queries.items()], ignore_index=True), None)
    coverage = np.concatenate(support)
    if (rows.parent.nunique() != ref["parents"] or not rows.groupby("inst_id").role.nunique().eq(3).all()
            or rows.assign(covered=coverage).groupby("inst_id").covered.nunique().ne(1).any()):
        raise ValueError("T4 attribution lost a parent, role, or exposure-matched support")
    stamp = dict(upstream, groups=digests)
    if not scoring:
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if read_json(args.freeze) != stamp:
        raise ValueError("T4 attribution arrays changed after freeze")
    truth = np.concatenate([score_delta(args, ref, k, q, columns, genes) for k, q in queries.items()])
    result = comparisons(rows, truth, {k: np.concatenate(v) for k, v in forecasts.items()}, coverage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-attr-score-") as temp:
        stage = Path(temp)
        result.to_csv(stage/"paired.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
            files={"paired.tsv": sha256(stage/"paired.tsv")}))
        stage.rename(args.out)
    return sha256(args.out/"receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "freeze", "score"))
    for name in ("access", "pin", "protocol", "original", "original-forecasts", "forecasts", "freeze", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--slice")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print((predict if args.action == "predict" else evaluate)(args))
