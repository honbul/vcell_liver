"""Evaluate the existing eight-context comparator contract on pinned JHH-5 low-dose wells."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_donor8_state import MEAN, SCALAR, SHRINK, STATE, paired, state_panel_forecast
from src.r19_plan.hcc_donor_panel_size import SEEDS, choices
from src.r19_plan.hcc_lincs_24h_access import CELL, CONTEXT, VIEW, register
from src.r19_plan.hcc_lincs_24h_transfer import context, effect, group_means
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array, checked_rows
from src.r19_plan.jhh5_donor_attribution import shrink_consensus
from src.r19_plan.t4_workflow import axes, scalar
from src.r19_plan.x_inputs import check_unique


DOSE = "3.33333"
PANEL = 8
ARMS = (MEAN, SHRINK, STATE, SCALAR)


def access(args):
    if sha256(args.audit / "receipt.json") != sha256(args.audit_pin):
        raise ValueError("The low-dose support audit differs from the pinned metadata")
    stamp = read_json(args.audit_pin)
    if not stamp["metadata_only"] or stamp["expression_read"] or not stamp["registered_control_block"]:
        raise ValueError("Low-dose access requires a metadata-only audit with the R18 control block")
    scope = {name: read_pinned(args.audit, name, digest) for name, digest in stamp["files"].items()}
    support, = scope["support.tsv"].loc[lambda rows: rows.cell.eq(CELL) & rows.dose.eq(DOSE)
        & rows.minimum_donor_cells_and_plates.eq(str(PANEL))].to_dict("records")
    if support["sufficient"] != "True":
        raise ValueError("The registered low-dose support gate did not pass")
    target = scope["targets.tsv"].loc[lambda rows: rows.cell_id.eq(CELL) & rows.actual_dose_uM.eq(DOSE)]
    source = scope["donors.tsv"].loc[lambda rows: rows.view.eq(VIEW) & rows.actual_dose_uM.eq(DOSE)
                                    & rows.parent.isin(target.parent)]
    counts = source.groupby("parent").agg(cells=("cell_id", "nunique"), plates=("rna_plate", "nunique"))
    allowed = counts.index[counts.cells.ge(PANEL) & counts.plates.ge(PANEL)]
    target = target.loc[target.parent.isin(allowed)]
    source = source.loc[source.parent.isin(allowed)]
    contexts = pd.concat([target[CONTEXT], source[CONTEXT]]).drop_duplicates()
    controls = scope["controls.tsv"].merge(contexts, on=CONTEXT, validate="many_to_one")
    if (target.parent.nunique() != int(support["parents"]) or len(target) != int(support["wells"])
            or set(target.rna_plate) & set(source.rna_plate)):
        raise ValueError("The treatment axis has changed since the low-dose support audit")
    check_unique(pd.concat((target, source, controls), ignore_index=True))
    splits = scope["splits.tsv"].loc[lambda rows: rows.dose.eq(DOSE)]
    register(args, target, source, controls, splits)


def load_plan(args, stage, key, recipe):
    return {name: read_pinned(args.access / (stage + "-plan") / key, name, digest) for name, digest in recipe.items()}


def donor_panels(args, ref, genes, cols, stamp):
    role = args.role
    tables = {name: read_pinned(args.access / "donor-plan" / role, name, digest)
              for name, digest in ref["donors"][role].items()}
    rows = tables["source.tsv"]
    if not rows.actual_dose_uM.eq(DOSE).all() or not rows.view.eq(VIEW).all() or len(rows) != ref["donor_wells"]:
        raise ValueError("Unexpected donor measurements in the low-dose access plan")
    values = effect(rows, tables["controls.tsv"], ref, genes, cols, role)
    plates, vectors = group_means(values, rows, ["parent", "cell_id", "rna_plate"])
    cells, vectors = group_means(vectors, plates, ["parent", "cell_id"])
    parents, means, shrunk, selected = [], [], [], []
    for parent, index in cells.groupby("parent", sort=True).indices.items():
        contexts = cells.cell_id.iloc[index].tolist()
        if len(index) < PANEL:
            raise ValueError("A parent lost its declared donor coverage")
        panels = [choices(parent, contexts, seed)[:PANEL] for seed in SEEDS]
        blocks = [vectors[index[members]] for members in panels]
        averages = np.stack([block.mean(axis=0) for block in blocks])
        parents.append(parent)
        means.append(averages)
        shrunk.append(shrink_consensus(blocks, [PANEL]*len(SEEDS), averages))
        selected.extend((parent, seed, contexts[i]) for seed, members in zip(SEEDS, panels, strict=True) for i in members)
    if len(parents) != ref["target_parents"]:
        raise ValueError("The donor panel registry has lost target parents")
    target = args.run / "donors" / role
    target.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix=".panels-") as temporary:
        staged = Path(temporary)
        data = staged / "panels.npz"
        np.savez_compressed(data, parents=np.asarray(parents), mean=np.asarray(means, dtype="float32"),
                            shrink=np.asarray(shrunk, dtype="float32"))
        panel_roster = staged / "panel_members.tsv"
        pd.DataFrame(selected, columns=["parent", "seed", "cell_id"]).to_csv(panel_roster, sep="\t", index=False)
        write_json(staged / "receipt.json", dict(stamp, role=role, array_sha256=sha256(data),
                                                  panel_sha256=sha256(panel_roster)))
        staged.rename(target)


def read_panels(args, role, stamp, ref, genes):
    donor = args.run / "donors" / role
    receipt = read_json(donor / "receipt.json")
    vectors = donor / "panels.npz"
    if (receipt["role"] != role or any(receipt[k] != v for k, v in stamp.items())
            or sha256(vectors) != receipt["array_sha256"] or sha256(donor / "panel_members.tsv") != receipt["panel_sha256"]):
        raise ValueError("The donor panels changed before prediction")
    size = (ref["target_parents"], len(SEEDS), len(genes))
    with np.load(vectors, allow_pickle=False) as packed:
        names = pd.Index(packed["parents"])
        arrays = [checked_array(packed[key], size, key) for key in ("mean", "shrink")]
    if not names.is_unique or len(names) != size[0]:
        raise ValueError("Donor panel parent axis differs from the access plan")
    return names, *arrays


def predict(args, ref, genes, cols, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    role = args.group[-1]
    names, mean, shrunk = read_panels(args, role, stamp, ref, genes)
    network = load_network(args.state)
    dest = args.run / "forecasts" / args.group
    dest.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".predict-") as temporary:
        staged = Path(temporary)
        files, fit_stats = {}, []
        for key, entry in sorted(ref["slices"].items()):
            if not key.startswith(args.group + "-"):
                continue
            fold, _, _, _, plate = axes(key)
            tables = load_plan(args, "predict", key, entry["predict"])
            fit, controls, query = (tables[name] for name in ("fit.tsv", "fit_controls.tsv", "query.tsv"))
            if (set(fit.parent) & set(query.parent) or set(fit.scaffold) & set(query.scaffold)
                    or fit.fold.eq(str(fold)).any() or fit.rna_plate.eq(plate).any() or not query.fold.eq(str(fold)).all()
                    or not query.rna_plate.eq(plate).all() or "level3_row_index" in query):
                raise ValueError("Low-dose fit intersects the query fold, scaffolds or plate")
            labels, y = group_means(effect(fit, controls, ref, genes, cols, role), fit, ["parent"])
            i, j = names.get_indexer(labels.parent), names.get_indexer(query.parent)
            if (i < 0).any() or (j < 0).any():
                raise ValueError("A fit or score parent lacks fixed donor panels")
            fitting, wanted = mean[i], mean[j]
            replicated = labels.iloc[np.repeat(np.arange(len(labels)), len(SEEDS))].reset_index(drop=True)
            gain = scalar(replicated, fitting.reshape(-1, len(genes)), np.repeat(y, len(SEEDS), axis=0))
            results = {MEAN: wanted, SHRINK: shrunk[j], SCALAR: (gain*wanted).astype("float32"),
                       STATE: state_panel_forecast(network, labels, fitting, y, wanted)}
            data = staged / (key + ".npz")
            np.savez_compressed(data, **{arm: checked_array(values, wanted.shape, arm) for arm, values in results.items()})
            files[key] = sha256(data)
            fit_stats.append(dict(slice=key, fit_parents=len(labels), query_wells=len(query), gain=gain))
        if not files:
            raise ValueError("The forecast group has no pinned slices")
        write_json(staged / "receipt.json", dict(stamp, group=args.group, files=files, fits=fit_stats,
            donor_sha256=sha256(args.run / "donors" / role / "receipt.json"), state_upstream=UPSTREAM,
            state_model_sha256=sha256(args.state / "src/state/tx/models/state_transition.py")))
        staged.rename(dest)


def finish(args, ref, genes, cols, stamp):
    scoring = args.phase == "score"
    path = args.run / "freeze.json"
    if scoring and (sha256(path) != args.digest or (args.run / "score").exists()):
        raise ValueError("Low-dose scoring requires the independent forecast freeze")
    if not scoring and path.exists():
        raise FileExistsError(path)
    hashes, queries, predictions, fits = {}, {}, {}, []
    for group in sorted({key[:4] for key in ref["slices"]}):
        directory = args.run / "forecasts" / group
        receipt = read_json(directory / "receipt.json")
        keys = {key for key in ref["slices"] if key.startswith(group + "-")}
        if (receipt["group"] != group or any(receipt[k] != v for k, v in stamp.items())
                or set(receipt["files"]) != keys
                or sha256(args.run / "donors" / group[-1] / "receipt.json") != receipt["donor_sha256"]):
            raise ValueError("Forecast receipt or donor lineage changed")
        hashes[group] = sha256(directory / "receipt.json")
        fits.extend(receipt["fits"])
        for key, digest in sorted(receipt["files"].items()):
            pred = ref["slices"][key]["predict"]
            query = read_pinned(args.access / "predict-plan" / key, "query.tsv", pred["query.tsv"])
            queries[key] = query
            data = directory / (key + ".npz")
            if sha256(data) != digest:
                raise ValueError("A low-dose predicted array changed")
            with np.load(data, allow_pickle=False) as archive:
                arrays = {arm: checked_array(archive[arm], (len(query), len(SEEDS), len(genes)), arm) for arm in ARMS}
            if scoring:
                predictions[key] = arrays
    rows = checked_rows(pd.concat([q.assign(role=axes(key)[1]) for key, q in queries.items()], ignore_index=True), None)
    support = dict(parents=rows.parent.nunique(), wells=rows.inst_id.nunique(), plates=rows.rna_plate.nunique())
    if (any(n != ref["target_" + key] for key, n in support.items())
            or not rows.groupby("inst_id").role.nunique().eq(3).all()):
        raise ValueError("Prediction and audited parent, well and role axes differ")
    frozen = dict(stamp, support=support, groups=hashes)
    if not scoring:
        write_json(path, frozen)
        return
    if read_json(path) != frozen:
        raise ValueError("Low-dose scoring inputs differ from the independent freeze")

    targets = []
    for key, query in queries.items():
        tables = load_plan(args, "score", key, ref["slices"][key]["score"])
        target = tables["query.tsv"]
        if not target.drop(columns="level3_row_index").equals(query):
            raise ValueError("Score wells do not have the registered query order")
        targets.append(effect(target, tables["score_controls.tsv"], ref, genes, cols, axes(key)[3]))
    y = np.concatenate(targets)
    outcomes = {arm: np.concatenate([values[arm] for values in predictions.values()]) for arm in ARMS}
    outcomes["B0"] = np.zeros_like(y)
    folds = [(f"fold_{n}", rows.fold.eq(n)) for n in range(3)]
    plates = [(plate, rows.rna_plate.ne(plate)) for plate in sorted(rows.rna_plate.unique())]
    tables = {"paired": paired(rows, y, outcomes, CELL, "all"), "fits": fits}
    for key, conditions in (("folds", folds), ("leave_plate", plates)):
        tables[key] = [record for label, keep in conditions for record in paired(rows.loc[keep].reset_index(drop=True),
                y[keep], {arm: values[keep] for arm, values in outcomes.items()}, CELL, label)]
    with TemporaryDirectory(dir=args.run, prefix=".score-") as temporary:
        staged = Path(temporary)
        for name, table in tables.items():
            pd.DataFrame(table).to_csv(staged / (name + ".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(staged / "receipt.json", dict(frozen, freeze_sha256=args.digest,
                                                  files={file.name: sha256(file) for file in staged.iterdir()}))
        staged.rename(args.run / "score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("access", "donor", "predict", "freeze", "score"))
    for key in ("access", "access-pin", "protocol", "run", "audit", "audit-pin", "cmap-pin", "landmarks", "out", "state"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--role", choices=tuple("ABC"))
    parser.add_argument("--group")
    parser.add_argument("--digest")
    args = parser.parse_args()
    if args.phase == "access":
        access(args)
    else:
        ref, genes, cols, stamp = context(args)
        stamp["transfer_code_sha256"] = sha256(__file__)
        {"donor": donor_panels, "predict": predict, "freeze": finish, "score": finish}[args.phase](args, ref, genes, cols, stamp)
