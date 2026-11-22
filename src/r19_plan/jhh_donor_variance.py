"""Disentangle gene-specific donor variance from attenuation and response magnitude."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_donor_panel_size import choices
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.hcc_off_position_pilot_freeze import old_panels
from src.r19_plan.i_score import METRICS, checked_array, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.jhh5_donor_attribution import shrink_consensus
from src.r19_plan.jhh5_off_position import CALIBRATED, COUNT, SCALAR, SHRINK
from src.r19_plan.jhh5_off_position_native import references as jhh5
from src.r19_plan.jhh7_donor_transfer import read_plan
from src.r19_plan.jhh7_off_position_native import references as jhh7
from src.r19_plan.jhh_low_dose_transfer import load_plan
from src.r19_plan.jhh_off_position_robustness import SCOPE
from src.r19_plan.t4_workflow import axes, scalar


SHUFFLE, MAGNITUDE, TYPICAL = "norm_matched_shuffle", "magnitude_only", "fit_gene_variance"
ARMS = (CALIBRATED, SCALAR, SHUFFLE, MAGNITUDE, TYPICAL)
N_SEEDS, BINS = 16, 16


def variance(panels, mean):
    return np.square(panels - mean[:, None]).sum(axis=1)/(COUNT*(COUNT-1))


def shrink(mean, var):
    signal = np.square(mean)
    total = signal + var
    return mean*np.divide(signal, total, out=np.zeros_like(total), where=total > 0)


def shuffle_direction(parent, mean, var):
    rng = np.random.default_rng(crc32(f"191203|{parent}".encode()))
    shuffled = np.broadcast_to(var, (N_SEEDS, len(var))).copy()
    for group in np.array_split(np.argsort(np.abs(mean), kind="stable"), BINS):
        for sample in shuffled:
            sample[group] = var[rng.permutation(group)]
    return shrink(mean[None, :], shuffled)


def match_norm(reference, shuffled):
    a = np.linalg.norm(reference.astype("float64"), axis=-1)
    b = np.linalg.norm(shuffled.astype("float64"), axis=-1)
    if np.any((a[:, None] > 0) & (b == 0)):
        raise ValueError("A nonzero fitted forecast has an empty shuffled direction")
    multipliers = np.divide(a[:, None], b, out=np.zeros_like(b), where=b > 0)
    output = (shuffled*multipliers[..., None]).astype("float32")
    np.testing.assert_allclose(np.linalg.norm(output.astype("float64"), axis=-1),
        np.broadcast_to(a[:, None], b.shape), atol=1e-10, rtol=3e-7)
    return output


def references(args):
    prior_args = argparse.Namespace(**(vars(args) | {"protocol": args.native_protocol}))
    is_jhh5 = args.cell == "JHH5"
    old, ref, genes, cols, freeze, _ = (jhh5 if is_jhh5 else jhh7)(prior_args)
    stamp = dict(cell=args.cell, previous_freeze_sha256=sha256(args.previous_pin),
                 protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__))
    return old, ref, genes, cols, freeze, stamp


def keys(args, ref, role):
    return ([key for key in sorted(ref["slices"]) if axes(key)[1] == role]
            if args.cell == "JHH5" else [role])


def tables(args, old, ref, phase, key):
    if args.cell == "JHH7":
        return read_plan(old, ref, phase, key)
    return {name.removesuffix(".tsv"): table for name, table in
            load_plan(old, phase, key, ref["slices"][key][phase]).items()}


def donor_profiles(args, old, ref, genes, cols, frozen, role):
    if args.cell == "JHH5":
        return old_panels(args, ref, genes, frozen, role)
    plan = tables(args, old, ref, "predict", role)
    source = plan["source"]
    if not source.view.eq("nonhepatic").all() or not source.actual_dose_uM.eq("10").all():
        raise ValueError("The frozen JHH-7 donor plan changed")
    values = effect(source, plan["source_controls"], ref, genes, cols, role)
    plate, vectors = group_means(values, source, ["parent", "cell_id", "rna_plate"])
    cell, vectors = group_means(vectors, plate, ["parent", "cell_id"])
    names, panels = [], []
    for parent, idx in cell.groupby("parent", sort=True).indices.items():
        listed = cell.cell_id.iloc[idx].tolist()
        if len(listed) != COUNT:
            raise ValueError("Each parent needs the same eight nonhepatic donor lines")
        names.append(parent)
        panels.append(vectors[idx[choices(parent, listed, 0)]])
    return pd.Index(names), np.stack(panels)


def historical(args, frozen, key, shape):
    five = args.cell == "JHH5"
    path = args.previous / "forecasts" / (key[:4] if five else key)
    receipt = path / "receipt.json"
    digest = frozen["groups"][key[:4]] if five else frozen["roles"][key]
    if sha256(receipt) != digest:
        raise ValueError("Original prediction receipt changed")
    record = read_json(receipt)
    file = path / (key + ".npz" if five else "predictions.npz")
    if sha256(file) != (record["files"][key] if five else record["array_sha256"]):
        raise ValueError("The original donor-average or shrinkage prediction changed")
    with np.load(file, allow_pickle=False) as data:
        return {arm: checked_array(data[arm], shape, arm) for arm in (CALIBRATED, SCALAR, SHRINK)}


def predict(args, old, ref, genes, cols, frozen, stamp):
    role = args.role
    names, panels = donor_profiles(args, old, ref, genes, cols, frozen, role)
    mean = panels.mean(axis=1)
    spread = variance(panels, mean)
    shuffled = np.stack([shuffle_direction(p, m, v) for p, m, v in zip(names, mean, spread, strict=True)])
    reduced = shrink_consensus(panels, [COUNT]*len(panels), mean)
    amplitude = shrink(mean, np.median(spread, axis=1, keepdims=True))
    dest = args.run / "predictions" / role
    dest.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".donor-variance-") as temp:
        stage = Path(temp)
        outputs, fits = {}, []
        for key in keys(args, ref, role):
            plan = tables(args, old, ref, "predict", key)
            fit, query = plan["fit"], plan["query"]
            if (set(query.parent) & set(fit.parent) or set(query.scaffold) & set(fit.scaffold)
                    or set(query.rna_plate) & set(fit.rna_plate) or "level3_row_index" in query
                    or not query.cell_id.eq(args.cell).all()):
                raise ValueError("Predictor input crosses the held scaffold, parent, plate or expression boundary")
            labels, y = group_means(effect(fit, plan["fit_controls"], ref, genes, cols, role), fit, ["parent"])
            i, j = (names.get_indexer(column) for column in (labels.parent, query.parent))
            if (i < 0).any() or (j < 0).any():
                raise ValueError("A registered parent lacks donor measurements")
            original = historical(args, frozen, key, (len(query), len(genes)))
            np.testing.assert_array_equal(reduced[j].astype("float32"), original.pop(SHRINK))
            for arm, value in ((CALIBRATED, reduced), (SCALAR, mean)):
                np.testing.assert_array_equal((scalar(labels, value[i], y)*value[j]).astype("float32"), original[arm])
            typical = shrink(mean[np.r_[i, j]], np.median(spread[i], axis=0))
            ma, tv = scalar(labels, amplitude[i], y), scalar(labels, typical[:len(i)], y)
            dest_file = stage / (key+".npz")
            np.savez_compressed(dest_file, **original,
                **{SHUFFLE: match_norm(original[CALIBRATED], shuffled[j]),
                   MAGNITUDE: (ma*amplitude[j]).astype("float32"),
                   TYPICAL: (tv*typical[len(i):]).astype("float32")})
            outputs[key] = sha256(dest_file)
            fits.append(dict(slice=key, fit_parents=len(i), query_wells=len(j), magnitude_gain=ma, gene_variance_gain=tv))
        write_json(stage/"receipt.json", dict(stamp, role=role, target_expression_read=False, files=outputs, fits=fits))
        stage.rename(dest)


def verified_forecasts(args, old, ref, genes, stamp):
    queries, receipts, estimates, fits = {}, {}, {}, []
    for role in "ABC":
        directory = args.run / "predictions" / role
        file = directory / "receipt.json"
        record = read_json(file)
        expected = keys(args, ref, role)
        if (any(record[k] != v for k, v in stamp.items()) or record["role"] != role
                or record["target_expression_read"] or set(expected) != set(record["files"])):
            raise ValueError("A forecast differs from the registered unscored prediction plan")
        receipts[role] = sha256(file)
        fits.extend(record["fits"])
        for key in expected:
            query = tables(args, old, ref, "predict", key)["query"]
            array = directory / (key+".npz")
            if sha256(array) != record["files"][key]:
                raise ValueError("A forecast payload changed")
            lead = (len(query),)
            shape = {arm: lead+((N_SEEDS,) if arm == SHUFFLE else ())+(len(genes),) for arm in ARMS}
            with np.load(array, allow_pickle=False) as values:
                estimates[key] = {a: checked_array(values[a], shape[a], a) for a in ARMS}
            queries[key] = (role, query)
    rows = checked_rows(pd.concat([q.assign(role=role) for role, q in queries.values()], ignore_index=True), None)
    n, well, plate = rows.parent.nunique(), rows.inst_id.nunique(), rows.rna_plate.nunique()
    if (n, well, plate) != SCOPE[args.cell] or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("Frozen forecasts lost the original physical support")
    return dict(stamp, roles=receipts, support=[n, well, plate]), rows, queries, estimates, fits


def score_rows(rows, y, arrays, scope):
    energy = squared_rows(y)
    summary = {}
    for arm, a in arrays.items():
        e = np.stack([squared_rows(y, a[:, k]) for k in range(N_SEEDS)]) if arm == SHUFFLE else squared_rows(y, a)[None]
        summary[arm] = (np.sqrt(e/y.shape[1]).mean(axis=0), e.mean(axis=0))
    a, ae = summary[CALIBRATED]
    results = []
    for arm in ARMS[1:]:
        b, be = summary[arm]
        _, sse, exposure = parent_summaries(rows, y.shape[1], energy, ae, be)
        rmse = pd.DataFrame(dict(parent=rows.parent.to_numpy(), a=a, b=b)).groupby("parent").mean().to_numpy()
        points, intervals, n = paired_effects(rmse, sse, exposure)
        low, high, _ = relative_interval(rmse)
        for m, (x, ref), (lo, hi), draws in zip(METRICS, points, intervals, n, strict=True):
            props = dict(relative_gain=1-x/ref, relative_low99=low, relative_high99=high) if m == "rmse" else {}
            results.append(dict(scope=scope, parents=rows.parent.nunique(), wells=rows.inst_id.nunique(),
                 arm=CALIBRATED, reference=arm, metric=m, arm_value=x, reference_value=ref,
                 difference=x-ref, low99=lo, high99=hi, draws=draws, **props))
    return results


def finish(args, old, ref, genes, cols, frozen, stamp):
    path = args.run / "freeze.json"
    if args.phase == "freeze" and path.exists():
        raise FileExistsError(path)
    if args.phase == "score" and (sha256(path) != args.digest or (args.run/"score").exists()):
        raise ValueError("Independent scoring requires the immutable forecast digest")
    manifest, rows, queries, estimates, fits = verified_forecasts(args, old, ref, genes, stamp)
    if args.phase == "freeze":
        write_json(path, manifest)
        return
    if read_json(path) != manifest:
        raise ValueError("Forecast inputs changed after the freeze")
    truth = []
    for key, (role, query) in queries.items():
        plan = tables(args, old, ref, "score", key)
        target = plan["query"]
        if not target.drop(columns="level3_row_index").equals(query):
            raise ValueError("The new score and old forecast target coordinates disagree")
        ctl = plan["score_controls"] if args.cell == "JHH5" else plan["controls"]
        truth.append(effect(target, ctl, ref, genes, cols, "ABC"[("ABC".index(role)+2)%3]))
    y = np.concatenate(truth)
    arrays = {arm: np.concatenate([e[arm] for e in estimates.values()]) for arm in ARMS}
    all_rows = score_rows(rows, y, arrays, "all")
    scored, old_table = pd.DataFrame(all_rows), args.previous/"score/paired.tsv"
    old_score = read_json(args.previous_score_pin)
    if (sha256(args.previous/"score/receipt.json") != sha256(args.previous_score_pin)
            or sha256(old_table) != old_score["files"]["paired.tsv"]):
        raise ValueError("Original score receipt or table changed")
    expected = pd.read_csv(old_table, sep="\t").query("scope == 'all' and arm == @CALIBRATED and reference == @SCALAR")
    fields = ["arm_value", "reference_value", "difference", "low99", "high99"]
    np.testing.assert_allclose(scored.loc[scored.reference.eq(SCALAR), fields], expected[fields], atol=1e-12, rtol=0)
    folds = []
    for n in range(3):
        keep = rows.fold.eq(n)
        folds.extend(score_rows(rows.loc[keep], y[keep], {arm: a[keep] for arm, a in arrays.items()}, n))
    with TemporaryDirectory(dir=args.run, prefix=".donor-variance-score-") as temp:
        stage = Path(temp)
        for name, values in (("paired", all_rows), ("folds", folds), ("fits", fits)):
            pd.DataFrame(values).to_csv(stage/(name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(manifest, freeze_sha256=args.digest, original_scores_replayed=True,
            files={f.name: sha256(f) for f in stage.iterdir()}))
        stage.rename(args.run/"score")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("predict", "freeze", "score"))
    cli.add_argument("--cell", choices=tuple(SCOPE), required=True)
    cli.add_argument("--role", choices=tuple("ABC"))
    cli.add_argument("--digest")
    for name in ("run", "previous", "previous-pin", "previous-score-pin", "access-pin", "previous-protocol", "native-protocol", "protocol"):
        cli.add_argument("--"+name, type=Path, required=True)
    args = cli.parse_args()
    (predict if args.phase == "predict" else finish)(args, *references(args))
