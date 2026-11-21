"""Compare frozen JHH-5 donor baselines and fit-free donor consensus on identical support."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_transfer import context, donors, effect, group_means
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array, checked_rows, parent_summaries, squared_rows
from src.r19_plan.t4_position import OP, SC, ST, compare, contrast, retrieval
from src.r19_plan.t4_scaffold_transfer import cluster_effects
from src.r19_plan.t4_workflow import axes


MEAN = "donor_mean"
FIX = "donor_cell_disagreement_shrinkage"
SHUFFLE = "other_parent_same_pool"
FIT_MEAN = "fit_parent_donor_mean"
STATE = "State_24h_donor"
OLD = {"B0": "B0", MEAN: MEAN, "scalar": "scalar", STATE: ST}
NEW = (FIX, SHUFFLE, FIT_MEAN)
ARMS = (*OLD, *NEW)
ATTRIBUTION = tuple((MEAN, arm) for arm in ("B0", STATE, SHUFFLE, FIT_MEAN))
CANDIDATE = ((FIX, MEAN), (FIX, "scalar"))
PAIRS = (*ATTRIBUTION, (MEAN, "scalar"), *CANDIDATE)


def setup(args):
    previous = SimpleNamespace(access=args.previous / "access", access_pin=args.access_pin,
        protocol=args.previous_protocol, run=args.previous)
    ref, genes, cols, prior_stamp = context(previous)
    actual = args.previous / "freeze.json"
    frozen = read_json(args.previous_pin)
    if sha256(actual) != sha256(args.previous_pin) or any(frozen[k] != v for k, v in prior_stamp.items()):
        raise ValueError("The original JHH-5 prediction freeze is different")
    stamp = dict(access_sha256=prior_stamp["access_sha256"], previous_freeze_sha256=sha256(actual),
        protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__))
    return SimpleNamespace(previous=previous, ref=ref, genes=genes, cols=cols,
        prior_stamp=prior_stamp, frozen=frozen, stamp=stamp)


def shrink_consensus(cell_profiles, group_sizes, means):
    sumsq = np.stack([((values - mean)**2).sum(axis=0) for values, mean in zip(cell_profiles, means, strict=True)])
    n = np.asarray(group_sizes)[:, None]
    if (n < 2).any():
        raise ValueError("Cell disagreement needs at least two donor contexts")
    variance_of_mean = sumsq / (n*(n-1))
    signal = means**2
    total = signal + variance_of_mean
    weight = np.divide(signal, total, out=np.zeros_like(signal), where=total>0)
    return means * weight


def wrong_parent(query, fold, plate):
    pool = np.unique(query.parent)
    if len(pool) < 3:
        raise ValueError("The fixed wrong-parent pool has too few candidates")
    order = np.random.default_rng(crc32(f"68241|{fold}|{plate}".encode())).permutation(pool)
    swapped = dict(zip(order, np.roll(order, 1), strict=True))
    return query.parent.map(swapped)


def generate(args, source):
    ref, role = source.ref, args.role
    parents, published = donors(source.previous, role, ref, source.genes, source.prior_stamp)
    tables = {name: read_pinned(source.previous.access / "donor-plan" / role, name, h)
        for name, h in ref["donors"][role].items()}
    raw, control = tables["source.tsv"], tables["controls.tsv"]
    if len(raw) != ref["donor_wells"] or not raw.view.eq("nonhepatic").all() or not raw.actual_dose_uM.eq("10").all():
        raise ValueError("The frozen matched dose and context source inventory differs")
    values = effect(raw, control, ref, source.genes, source.cols, role)
    plate_rows, plate_vectors = group_means(values, raw, ["parent", "cell_id", "rna_plate"])
    cell_rows, cell_vectors = group_means(plate_vectors, plate_rows, ["parent", "cell_id"])
    grouped, mean = group_means(cell_vectors, cell_rows, ["parent"])
    if not parents.equals(pd.Index(grouped.parent)) or not np.array_equal(mean, published):
        raise ValueError("Source contexts do not reproduce the original donor vector exactly")
    profile = [cell_vectors[ix] for ix in cell_rows.groupby("parent", sort=True).indices.values()]
    counts = np.array([len(a) for a in profile], dtype=np.int64)
    shrunk = shrink_consensus(profile, counts, mean)

    out = args.run / "forecasts" / role
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent, prefix=".jhh5-donor-check-") as tmp:
        stage = Path(tmp)
        assets, pairing = {}, {}
        for key, split in sorted(ref["slices"].items()):
            fold, selected, _, _, plate = axes(key)
            if selected != role:
                continue
            files = source.previous.access / "predict-plan" / key
            query, fit = [read_pinned(files, name, split["predict"][name]) for name in ("query.tsv", "fit.tsv")]
            group = key[:4]
            old = args.previous / "forecasts" / group
            receipt = old / "receipt.json"
            p = old / (key+".npz")
            record = read_json(receipt)
            if sha256(receipt) != source.frozen["groups"][group] or sha256(p) != record["slices"][key]:
                raise ValueError("JHH-5 original forecast changed")
            with np.load(p, allow_pickle=False) as packed:
                expected = (len(query), len(source.genes))
                fixed = {name: checked_array(packed[arm], expected, name) for name, arm in OLD.items()}
            swapped = wrong_parent(query, fold, plate)
            index, replace, fitting = (parents.get_indexer(names) for names in (query.parent, swapped, sorted(fit.parent.unique())))
            if any((ix < 0).any() for ix in (index, replace, fitting)) or not np.array_equal(
                    mean[index].astype("float32"), fixed[MEAN]) or index.size != replace.size or (index==replace).any():
                raise ValueError("The donor arm or matched incorrect-parent comparator is not aligned")
            pooled = np.broadcast_to(mean[fitting].mean(axis=0), expected)
            new = {FIX: shrunk[index], SHUFFLE: mean[replace], FIT_MEAN: pooled}
            dest = stage / (key+".npz")
            np.savez_compressed(dest, **fixed, **{name: arr.astype("float32") for name, arr in new.items()}, count=counts[index])
            assets[key] = sha256(dest)
            pairing[key] = crc32("\n".join(swapped).encode())
        write_json(stage / "receipt.json", dict(source.stamp, role=role, slices=assets, wrong_parent_registry=pairing))
        stage.rename(out)


def finish(args, source):
    path = args.run / "freeze.json"
    score = args.phase == "score"
    if score and (sha256(path) != args.digest or (args.run/"score").exists()):
        raise ValueError("Separate scoring requires the independent JHH-5 donor freeze")
    if not score and path.exists():
        raise FileExistsError(path)
    receipts, queries, arrays, pairing = {}, {}, {}, {}
    for role in "ABC":
        root = args.run / "forecasts" / role
        receipt = read_json(root/"receipt.json")
        keys = {k for k in source.ref["slices"] if axes(k)[1] == role}
        if (receipt["role"] != role or set(receipt["slices"]) != keys or set(receipt["wrong_parent_registry"]) != keys
                or any(receipt[k] != v for k, v in source.stamp.items())):
            raise ValueError("JHH-5 donor forecasts have a changed role or lineage")
        receipts[role] = sha256(root/"receipt.json")
        for key, h in sorted(receipt["slices"].items()):
            fold, _, _, _, plate = axes(key)
            files = source.previous.access/"predict-plan"/key
            query = read_pinned(files, "query.tsv", source.ref["slices"][key]["predict"]["query.tsv"])
            matrix = root/(key+".npz")
            if sha256(matrix) != h:
                raise ValueError("A JHH-5 donor prediction changed after forecasting")
            match = receipt["wrong_parent_registry"][key]
            if pairing.setdefault((fold, plate), match) != match:
                raise ValueError("The wrong-parent pairing differs across roles")
            with np.load(matrix, allow_pickle=False) as archive:
                found = {name: checked_array(archive[name], (len(query), len(source.genes)), name) for name in ARMS}
                n = archive["count"]
            if n.shape != (len(query),) or n.dtype.kind not in "iu" or n.min() < 2:
                raise ValueError("Invalid availability on the frozen donor axis")
            queries[key] = query.assign(role=role, n_cells=n)
            if score:
                arrays[key] = found
    rows = checked_rows(pd.concat(queries.values(), ignore_index=True), None)
    if (rows.parent.nunique() != 609 or rows.inst_id.nunique() != 615 or rows.rna_plate.nunique() != 6
            or not rows.groupby("inst_id").role.nunique().eq(3).all() or not rows.groupby("parent").n_cells.nunique().eq(1).all()):
        raise ValueError("JHH-5 donor scoring axis lost its original paired support")
    manifest = dict(source.stamp, roles=receipts, parents=609, wells=615, plates=6)
    if not score:
        write_json(path, manifest)
        return
    if read_json(path) != manifest:
        raise ValueError("The forecast or source support differs from the independent freeze")

    outcomes = []
    for key, query in queries.items():
        root = source.previous.access/"score-plan"/key
        p = source.ref["slices"][key]["score"]
        rows_y, controls = [read_pinned(root, name, p[name]) for name in ("query.tsv", "score_controls.tsv")]
        if not rows_y.drop(columns="level3_row_index").equals(query.drop(columns=["role", "n_cells"])):
            raise ValueError("JHH-5 score role does not map to the locked prediction rows")
        outcomes.append(effect(rows_y, controls, source.ref, source.genes, source.cols, axes(key)[3]))
    truth = np.concatenate(outcomes)
    predictions = {name: np.concatenate([a[name] for a in arrays.values()]) for name in ARMS}
    report = score_tables(rows, truth, predictions)
    with TemporaryDirectory(dir=args.run, prefix=".jhh5-donor-score-") as tmp:
        stage = Path(tmp)
        for name, frame in report.items():
            frame.to_csv(stage / (name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(manifest, freeze_sha256=args.digest,
            files={f.name: sha256(f) for f in stage.iterdir()}))
        stage.rename(args.run/"score")


def score_tables(rows, y, predictions):
    y_centered, raw_centered, hits = contrast(rows, y, {OP: predictions[MEAN], SC: predictions[FIX], ST: predictions[STATE]})
    centered = {MEAN: raw_centered[OP], FIX: raw_centered[SC], STATE: raw_centered[ST]}
    energy = squared_rows(y)
    mapping = rows[["parent","scaffold"]].drop_duplicates().set_index("parent", verify_integrity=True).scaffold.to_dict()
    clusters, sensitivities, strata = [], [], []
    for arm, reference in PAIRS:
        moments = parent_summaries(rows, y.shape[1], energy, squared_rows(y, predictions[arm]), squared_rows(y, predictions[reference]))
        intervals, relative = cluster_effects(sorted(mapping), mapping, *moments)
        for metric, bounds in zip(("rmse", "r2_pooled"), intervals, strict=True):
            clusters.append(dict(arm=arm, reference=reference, metric=metric, low99=bounds[0], high99=bounds[1],
                gain_low99=relative[0] if metric=="rmse" else np.nan, gain_high99=relative[1] if metric=="rmse" else np.nan))
    for plate in sorted(rows.rna_plate.unique()):
        keep = rows.rna_plate.ne(plate)
        sensitivities.extend(compare(rows.loc[keep].reset_index(drop=True), y[keep], {a:v[keep] for a,v in predictions.items()},
            PAIRS, plate))
    for group, keep in (("under_16", rows.n_cells.lt(16)), ("at_least_16", rows.n_cells.ge(16))):
        if keep.any():
            strata.extend(compare(rows.loc[keep].reset_index(drop=True), y[keep], {a:v[keep] for a,v in predictions.items()},
                                CANDIDATE, group))
    return dict(paired=pd.DataFrame(compare(rows, y, predictions, PAIRS, "absolute")),
        within_plate=pd.DataFrame(compare(rows, y_centered, centered, ((MEAN,STATE),(FIX,MEAN)), "centered")),
        retrieval=retrieval(hits).replace({OP:MEAN,SC:FIX,ST:STATE}), scaffold=pd.DataFrame(clusters),
        leave_plate=pd.DataFrame(sensitivities), donor_count=pd.DataFrame(strata))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("predict", "freeze", "score"))
    for field in ("previous", "previous-pin", "previous-protocol", "protocol", "access-pin", "run"):
        cli.add_argument("--"+field, type=Path, required=True)
    cli.add_argument("--role", choices=list("ABC"))
    cli.add_argument("--digest")
    args = cli.parse_args()
    (generate if args.phase == "predict" else finish)(args, setup(args))
