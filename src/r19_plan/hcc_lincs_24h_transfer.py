"""Compare equal-information 24-hour transfer models on held adult HCC plates."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_access import CONTEXT
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array, checked_rows, parent_summaries, squared_rows
from src.r19_plan.i_stage import parse_landmarks, pinned_bytes
from src.r19_plan.r18_input_bundle import extract_matrix, indices
from src.r19_plan.t4_position import OP, SC, ST, compare, contrast, retrieval
from src.r19_plan.t4_scaffold_transfer import cluster_effects
from src.r19_plan.t4_state import state_forecast
from src.r19_plan.t4_workflow import axes, scalar
from src.r19_plan.time_operator import fit_time_operator
from src.r19_plan.x_inputs import check_unique, control_axis


CANDIDATE, ANCHOR, SHUFFLE = "anchored_rank16", "affine", "anchored_parent_shuffle"
ARMS = ("B0", "target_mean", "donor_mean", "scalar", ANCHOR, "rank16", CANDIDATE, SHUFFLE, ST)
COMPARISONS = tuple((CANDIDATE, name) for name in ARMS if name != CANDIDATE)


def context(args):
    file = args.access / "receipt.json"
    ref = read_json(file)
    if (sha256(file) != sha256(args.access_pin) or not ref["metadata_only"] or ref["expression_read"]
            or ref["cell"] != "JHH5" or ref["protocol_sha256"] != sha256(args.protocol)):
        raise ValueError("CMap expression steps require the pinned, eligible JHH-5 access plan")
    genes = parse_landmarks(pinned_bytes(args.access / "landmarks.tsv", ref["files"]["landmarks.tsv"])).pr_gene_id.to_numpy()
    cols = {}
    for key in ("treated", "controls"):
        name = key + "_genes.tsv"
        axis = read_pinned(args.access, name, ref["files"][name])
        if not np.array_equal(indices(axis.gene_id), genes):
            raise ValueError("Aligned CMap axes have a different landmark order")
        cols[key] = indices(axis.level3_gene_index)
    stamp = dict(access_sha256=sha256(file), protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__))
    return ref, genes, cols, stamp


def effect(rows, controls, ref, genes, cols, role):
    check_unique(pd.concat((rows, controls), ignore_index=True))
    control_axis(controls, rows, role)
    treated = extract_matrix(ref["matrices"]["treated"], rows, cols["treated"], genes)
    vehicles = extract_matrix(ref["matrices"]["controls"], controls, cols["controls"], genes)
    means = {key: vehicles[ix].mean(axis=0, dtype="float64") for key, ix in controls.groupby(CONTEXT).indices.items()}
    baseline = np.stack([means[key] for key in rows[CONTEXT].itertuples(index=False, name=None)])
    return treated - baseline


def group_means(values, rows, fields):
    groups = rows.groupby(fields, sort=True).indices
    labels = pd.DataFrame(list(groups), columns=fields) if len(fields) > 1 else pd.DataFrame({fields[0]: list(groups)})
    return labels, np.stack([values[ix].mean(axis=0) for ix in groups.values()])


def make_donor(args, ref, genes, cols, stamp):
    role = args.group
    tables = {name: read_pinned(args.access / "donor-plan" / role, name, h) for name, h in ref["donors"][role].items()}
    rows = tables["source.tsv"]
    if (not rows.view.eq("nonhepatic").all() or not rows.actual_dose_uM.eq("10").all()
            or rows.cell_id.eq("JHH5").any() or rows.inst_id.nunique() != ref["donor_wells"]):
        raise ValueError("CMap source support differs from the frozen donor contract")
    values = effect(rows, tables["controls.tsv"], ref, genes, cols, role)
    plate, averaged = group_means(values, rows, ["parent", "cell_id", "rna_plate"])
    cells, averaged = group_means(averaged, plate, ["parent", "cell_id"])
    parents, averaged = group_means(averaged, cells, ["parent"])
    if len(parents) != ref["target_parents"]:
        raise ValueError("A held compound has no equal-information donor input")
    out = args.run / "donors" / role
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent, prefix=".donor-") as tmp:
        stage = Path(tmp)
        data = stage / "vectors.npz"
        np.savez_compressed(data, parents=parents.parent.to_numpy(dtype=str), values=averaged)
        write_json(stage / "receipt.json", dict(stamp, role=role, parents=len(parents), vector_sha256=sha256(data)))
        stage.rename(out)


def donors(args, role, ref, genes, stamp):
    out = args.run / "donors" / role
    receipt = read_json(out / "receipt.json")
    data = out / "vectors.npz"
    if (any(receipt[k] != v for k, v in stamp.items()) or receipt["role"] != role
            or receipt["parents"] != ref["target_parents"] or sha256(data) != receipt["vector_sha256"]):
        raise ValueError("Donor array differs from the locked role and cohort")
    with np.load(data, allow_pickle=False) as src:
        parents = pd.Index(src["parents"])
        values = checked_array(src["values"], (len(parents), len(genes)), "donors")
    if not parents.is_unique or len(parents) != receipt["parents"]:
        raise ValueError("Donor parent axis changed")
    return parents, values


def model_predictions(network, fit, x, y, query, wanted, genes, fold, plate):
    train = fit.assign(study="CMap2020", cell_id="JHH5", dose_uM=10.)
    held = set(query.parent)
    xmean, ymean = x.mean(axis=0), y.mean(axis=0)
    scale = scalar(train, x-xmean, y-ymean)
    added_mean = ymean - scale*xmean
    anchor = added_mean + scale*wanted
    residual = y - (added_mean + scale*x)
    order = np.random.default_rng(crc32(f"38105|{fold}|{plate}".encode())).permutation(len(x))
    model = fit_time_operator(train, x, y, genes, rank=16, ridge=1, held_parents=held)
    anchored = fit_time_operator(train, x, residual, genes, rank=16, ridge=1, held_parents=held)
    shuffled = fit_time_operator(train, x[order], residual, genes, rank=16, ridge=1, held_parents=held)
    predictions = {
        "B0": np.zeros_like(anchor),
        "target_mean": np.broadcast_to(ymean, anchor.shape),
        "donor_mean": wanted,
        "scalar": scalar(train, x, y)*wanted,
        ANCHOR: anchor,
        "rank16": model.predict(wanted, genes),
        CANDIDATE: anchor + anchored.predict(wanted, genes),
        SHUFFLE: anchor + shuffled.predict(wanted, genes),
        ST: state_forecast(network, (train, x, y), wanted),
    }
    return predictions, scale


def predict(args, ref, genes, cols, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    network = load_network(args.state)
    _, role = args.group.split("-")
    parent_ids, vectors = donors(args, role, ref, genes, stamp)
    donor_receipt = args.run / "donors" / role / "receipt.json"
    output = args.run / "forecasts" / args.group
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix=".forecast-") as tmp:
        stage = Path(tmp)
        assets, settings = {}, []
        for key, entry in sorted(ref["slices"].items()):
            if not key.startswith(args.group + "-"):
                continue
            fold, _, _, _, plate = axes(key)
            values = {name: read_pinned(args.access/"predict-plan"/key, name, h) for name, h in entry["predict"].items()}
            fit, controls, query = (values[name] for name in ("fit.tsv", "fit_controls.tsv", "query.tsv"))
            if (set(fit.parent) & set(query.parent) or set(fit.scaffold) & set(query.scaffold)
                    or fit.fold.eq(str(fold)).any() or fit.rna_plate.eq(plate).any() or not query.fold.eq(str(fold)).all()
                    or not query.rna_plate.eq(plate).all() or "level3_row_index" in query):
                raise ValueError("The JHH-5 query enters its prediction training support")
            p, target = group_means(effect(fit, controls, ref, genes, cols, role), fit, ["parent"])
            pidx, qidx = parent_ids.get_indexer(p.parent), parent_ids.get_indexer(query.parent)
            if (pidx < 0).any() or (qidx < 0).any():
                raise ValueError("Fit or query has no matching CMap donor")
            forecast, scale = model_predictions(network, p, vectors[pidx], target, query, vectors[qidx], genes, fold, plate)
            file = stage / (key+".npz")
            np.savez_compressed(file, **{name: checked_array(value, (len(query), len(genes)), name).astype("float32")
                for name, value in forecast.items()})
            assets[key] = sha256(file)
            settings.append(dict(slice=key, parents=len(p), n_query=len(query), fit_wells=len(fit), affine_gain=scale))
        if not assets:
            raise ValueError("Unknown prediction group")
        upstream = args.state / "src/state/tx/models/state_transition.py"
        write_json(stage / "receipt.json", dict(stamp, group=args.group, slices=assets, settings=settings,
            donor_sha256=sha256(donor_receipt), state_upstream=UPSTREAM, state_model_sha256=sha256(upstream)))
        stage.rename(output)


def finish(args, ref, genes, cols, stamp):
    scoring = args.phase == "score"
    frozen = args.run / "freeze.json"
    if scoring and (sha256(frozen) != args.digest or (args.run/"score").exists()):
        raise ValueError("JHH-5 scoring requires the verified prediction freeze")
    if not scoring and frozen.exists():
        raise FileExistsError(frozen)
    groups, queries, arrays, settings = {}, {}, {}, []
    for group in sorted({key[:4] for key in ref["slices"]}):
        folder = args.run / "forecasts" / group
        receipt = read_json(folder/"receipt.json")
        keys = {k for k in ref["slices"] if k.startswith(group+"-")}
        if (receipt["group"] != group or set(receipt["slices"]) != keys or any(receipt[k] != v for k,v in stamp.items())
                or receipt["donor_sha256"] != sha256(args.run/"donors"/group[-1]/"receipt.json")):
            raise ValueError("JHH-5 forecast roles or source hash changed")
        groups[group] = sha256(folder/"receipt.json")
        settings.extend(receipt["settings"])
        for key, digest in sorted(receipt["slices"].items()):
            query = read_pinned(args.access/"predict-plan"/key, "query.tsv", ref["slices"][key]["predict"]["query.tsv"])
            queries[key] = query
            path = folder/(key+".npz")
            if sha256(path) != digest:
                raise ValueError("Forecast file changed")
            with np.load(path, allow_pickle=False) as archive:
                components = {arm: checked_array(archive[arm], (len(query), len(genes)), arm) for arm in ARMS}
            if scoring:
                arrays[key] = components
    rows = checked_rows(pd.concat([query.assign(role=axes(key)[1]) for key,query in queries.items()], ignore_index=True), None)
    parents, wells = rows.parent.nunique(), rows.inst_id.nunique()
    if (parents != ref["target_parents"] or wells != ref["target_wells"] or not rows.groupby("inst_id").role.nunique().eq(3).all()):
        raise ValueError("JHH-5 paired support changed")
    manifest = dict(stamp, groups=groups, parents=parents, wells=wells, plates=rows.rna_plate.nunique(), roles=3)
    if not scoring:
        write_json(frozen, manifest)
        return
    if read_json(frozen) != manifest:
        raise ValueError("JHH-5 prediction support changed after freezing")

    truth = []
    for key, registered in queries.items():
        plan = {name: read_pinned(args.access/"score-plan"/key, name, h) for name,h in ref["slices"][key]["score"].items()}
        target, controls = plan["query.tsv"], plan["score_controls.tsv"]
        if not target.drop(columns="level3_row_index").equals(registered):
            raise ValueError("JHH-5 score well order differs from the prediction registry")
        truth.append(effect(target, controls, ref, genes, cols, axes(key)[3]))
    y = np.concatenate(truth)
    predicted = {arm: np.concatenate([values[arm] for values in arrays.values()]) for arm in ARMS}
    scores = compare(rows, y, predicted, COMPARISONS, "absolute")
    energy = squared_rows(y)
    scaffold = rows[["parent", "scaffold"]].drop_duplicates().set_index("parent", verify_integrity=True).scaffold.to_dict()
    clusters = []
    for arm, reference in COMPARISONS:
        moments = parent_summaries(rows, y.shape[1], energy, squared_rows(y, predicted[arm]), squared_rows(y, predicted[reference]))
        ci, rel = cluster_effects(sorted(scaffold), scaffold, *moments)
        for metric, bound in zip(("rmse", "r2_pooled"), ci, strict=True):
            clusters.append(dict(arm=arm, reference=reference, metric=metric, low99=bound[0], high99=bound[1],
                relative_low99=rel[0] if metric=="rmse" else np.nan, relative_high99=rel[1] if metric=="rmse" else np.nan))
    leave = []
    main_pairs = tuple((CANDIDATE, a) for a in (ANCHOR, ST, SHUFFLE))
    for plate in sorted(rows.rna_plate.unique()):
        keep = rows.rna_plate.ne(plate)
        leave.extend(compare(rows.loc[keep].reset_index(drop=True), y[keep],
            {arm: d[keep] for arm, d in predicted.items()}, main_pairs, plate))
    relevant = rows.groupby(["fold", "role", "rna_plate", "actual_dose_uM"]).parent.transform("nunique").ge(3)
    centered_y, centered_p, hits = contrast(rows.loc[relevant].reset_index(drop=True), y[relevant],
        {OP: predicted[CANDIDATE][relevant], SC: predicted[ANCHOR][relevant], ST: predicted[ST][relevant]})
    centered_scores = compare(rows.loc[relevant].reset_index(drop=True), centered_y, centered_p,
        ((OP, SC), (OP, ST)), "within_plate")
    with TemporaryDirectory(dir=args.run, prefix=".score-") as tmp:
        stage = Path(tmp)
        for name, table in (("paired", pd.DataFrame(scores)), ("scaffold", pd.DataFrame(clusters)),
                            ("leave_plate", pd.DataFrame(leave)), ("within_plate", pd.DataFrame(centered_scores)),
                            ("retrieval", retrieval(hits)), ("fit_support", pd.DataFrame(settings))):
            if name in ("within_plate", "retrieval"):
                table = table.replace({OP:CANDIDATE, SC:ANCHOR})
            table.to_csv(stage/(name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(manifest, freeze_sha256=args.digest,
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.run/"score")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("donor", "predict", "freeze", "score"))
    for field in ("access", "access-pin", "protocol", "state", "run"):
        cli.add_argument("--"+field, type=Path, required=True)
    cli.add_argument("--group")
    cli.add_argument("--digest")
    args = cli.parse_args()
    {"donor":make_donor, "predict":predict, "freeze":finish, "score":finish}[args.phase](args, *context(args))
