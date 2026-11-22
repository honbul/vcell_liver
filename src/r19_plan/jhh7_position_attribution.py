"""Pair same-compound off-position donors with wrong-compound same-position donors."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_transfer import effect
from src.r19_plan.i_score import checked_array, checked_rows, parent_summaries, squared_rows
from src.r19_plan.jhh5_donor_attribution import SHUFFLE, STATE
from src.r19_plan.jhh7_donor_transfer import SCALED, read_plan
from src.r19_plan.jhh7_low_dose import DOSES, provenance
from src.r19_plan.t4_position import OP, SC, ST, compare, contrast, retrieval
from src.r19_plan.t4_scaffold_transfer import cluster_effects


OFF, SAME = "same_parent_different_position", "other_parent_same_position"
U2OS_PAIRS = ((OFF, SAME), (OFF, "B0"), (OFF, SCALED), (SAME, "B0"), (SCALED, "B0"))
OLD_PAIRS = tuple((SCALED, reference) for reference in (SHUFFLE, STATE, "B0"))


def base_reference(args):
    previous = SimpleNamespace(run=args.previous, access_pin=args.access_pin, protocol=args.previous_protocol)
    ref, genes, cols, stamp = provenance(previous)
    path = args.previous / "freeze.json"
    frozen = read_json(path)
    if sha256(path) != sha256(args.previous_pin) or any(frozen[k] != v for k, v in stamp.items()):
        raise ValueError("The prior observational JHH-7 cohort or its freeze changed")
    return previous, ref, genes, cols, frozen, dict(previous_freeze_sha256=sha256(path),
        protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__), access_sha256=stamp["access_sha256"])


def original(previous, frozen, role, query, gene_count):
    directory = previous.run / "forecasts" / role
    receipt = read_json(directory / "receipt.json")
    path = directory / "predictions.npz"
    if sha256(directory / "receipt.json") != frozen["roles"][role] or sha256(path) != receipt["file_sha256"]:
        raise ValueError("A previous role prediction or fitted gain changed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {arm: checked_array(archive[arm], (len(query), gene_count), arm) for arm in (SCALED, SHUFFLE, STATE)}
    return float(receipt["scalar_gain"]), arrays


def paired_locations(query, source):
    by_parent = source.groupby(["actual_dose_uM", "parent"], sort=False).indices
    by_position = source.set_index(["actual_dose_uM", "rna_plate", "rna_well"], verify_integrity=True)
    links = []
    for target in query.itertuples(index=False):
        rows = source.iloc[by_parent.get((target.actual_dose_uM, target.parent), [])]
        for hit in rows.itertuples(index=False):
            key = (target.actual_dose_uM, hit.rna_plate, target.rna_well)
            if hit.rna_well == target.rna_well or key not in by_position.index:
                continue
            wrong = by_position.loc[key]
            if wrong.parent != target.parent:
                links.append(dict(target_id=target.inst_id, source_plate=hit.rna_plate, off_id=hit.inst_id,
                                  same_id=wrong.inst_id, target_well=target.rna_well, off_well=hit.rna_well,
                                  dose=target.actual_dose_uM))
    return pd.DataFrame(links)


def predict(args, previous, ref, genes, cols, frozen, stamp):
    role = args.role
    plan = read_plan(previous, ref, "predict", role)
    query = plan["query"]
    source = plan["source"].loc[lambda rows: rows.cell_id.eq("U2OS") & rows.actual_dose_uM.isin(DOSES)]
    links = paired_locations(query, source)
    if links.empty:
        raise ValueError("No plate-matched position controls were found")
    eligible = query.loc[query.inst_id.isin(links.target_id)]
    primary = eligible.loc[eligible.actual_dose_uM.eq("1")]
    if primary.parent.nunique() < 50 or primary.groupby("fold").parent.nunique().reindex(list("012"), fill_value=0).min() < 10:
        raise ValueError("The prespecified position attribution support gate did not pass")
    used = source.loc[source.inst_id.isin(set(links.off_id) | set(links.same_id))].reset_index(drop=True)
    controls = plan["source_controls"].loc[lambda rows: rows.cell_id.eq("U2OS") & rows.rna_plate.isin(used.rna_plate)]
    values = effect(used, controls, ref, genes, cols, role)
    lookup = pd.Index(used.inst_id)
    gain, _ = original(previous, frozen, role, query, len(genes))
    collected = {name: [] for name in (OFF, SAME)}
    for q in eligible.inst_id:
        blocks = links.loc[links.target_id.eq(q)].groupby("source_plate", sort=True)
        for arm, key in ((OFF, "off_id"), (SAME, "same_id")):
            averaged = [values[lookup.get_indexer(plate[key])].mean(axis=0) for _, plate in blocks]
            collected[arm].append((gain*np.mean(averaged, axis=0)).astype("float32"))
    output = args.run / "forecasts" / role
    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".position-") as temp:
        stage = Path(temp)
        links.to_csv(stage / "links.tsv", sep="\t", index=False)
        support = links.groupby("target_id").source_plate.nunique().reindex(eligible.inst_id)
        np.savez_compressed(stage / "arrays.npz", ids=eligible.inst_id.to_numpy(dtype=str), pairs=support.to_numpy(),
                            **{name: np.stack(values) for name, values in collected.items()})
        write_json(stage / "receipt.json", dict(stamp, role=role,
            files={name: sha256(stage / name) for name in ("links.tsv", "arrays.npz")}))
        stage.rename(output)


def diagnostics(rows, y, arrays, dose):
    reconstructed = compare(rows, y, arrays, OLD_PAIRS, dose)
    centered, models, matches = contrast(rows, y, {OP: arrays[SCALED], SC: arrays[SHUFFLE], ST: arrays[STATE]})
    mapped = {SCALED: models[OP], SHUFFLE: models[SC], STATE: models[ST], "B0": np.zeros_like(centered)}
    ranking = retrieval(matches).replace({OP: SCALED, SC: SHUFFLE, ST: STATE}).assign(dose=dose)
    scaffold = rows[["parent", "scaffold"]].drop_duplicates().set_index("parent", verify_integrity=True).scaffold.to_dict()
    influence, resampled = [], []
    for arm, reference in OLD_PAIRS:
        parent_order = sorted(scaffold)
        metrics = parent_summaries(rows, y.shape[1], squared_rows(y), squared_rows(y, arrays[arm]), squared_rows(y, arrays[reference]))
        bounds, relative = cluster_effects(parent_order, scaffold, *metrics)
        resampled.append(dict(dose=dose, reference=reference, scaffolds=len(set(scaffold.values())), relative_gain_low99=relative[0],
                              relative_gain_high99=relative[1], r2_low99=bounds[1, 0], r2_high99=bounds[1, 1]))
        rmse, sse, energy = metrics
        gains = sse[:, 1] - sse[:, 0]
        ranked = np.argsort(-gains, kind="stable")
        positive = ranked[gains[ranked] > 0]
        for remove in (0, 5, 10, 20):
            remaining = np.ones(len(gains), dtype=bool)
            remaining[positive[:remove]] = False
            influence.append(dict(dose=dose, reference=reference, removed=int((~remaining).sum()), parents=int(remaining.sum()),
                rmse_positive=int((rmse[remaining, 0] < rmse[remaining, 1]).sum()),
                sse_positive=int((gains[remaining] > 0).sum()),
                relative_gain=1-rmse[remaining, 0].mean()/rmse[remaining, 1].mean(),
                r2_delta=gains[remaining].sum()/energy[remaining].sum()))
    return dict(original=reconstructed, centered=compare(rows, centered, mapped, OLD_PAIRS, dose),
                retrieval=ranking.to_dict("records"), influence=influence, scaffold=resampled)


def finish(args, previous, ref, genes, cols, frozen, stamp):
    scoring = args.phase == "score"
    freeze = args.run / "freeze.json"
    if scoring and (sha256(freeze) != args.digest or (args.run / "score").exists()):
        raise ValueError("Matched-position scoring requires the new freeze")
    if not scoring and freeze.exists():
        raise FileExistsError(freeze)
    roles, pieces, shared = {}, {}, None
    for role in "ABC":
        root = args.run / "forecasts" / role
        path = root / "receipt.json"
        receipt = read_json(path)
        digest = receipt["files"]["links.tsv"]
        if shared is None:
            shared = digest
        if (receipt["role"] != role or any(receipt[k] != v for k,v in stamp.items())
                or shared != digest or any(sha256(root / name) != h for name,h in receipt["files"].items())):
            raise ValueError("The source pairs or their lineage differ across roles")
        roles[role] = sha256(path)
        q = read_plan(previous, ref, "predict", role)["query"]
        with np.load(root / "arrays.npz", allow_pickle=False) as archive:
            index = pd.Index(q.inst_id).get_indexer(archive["ids"])
            if (index < 0).any() or len(set(index)) != len(index) or (archive["pairs"] < 1).any():
                raise ValueError("An eligible target or its matched-pair count changed")
            arrays = {arm: checked_array(archive[arm], (len(index), len(genes)), arm) for arm in (OFF, SAME)}
            pieces[role] = (q, index, archive["pairs"], arrays)
    manifest = dict(stamp, roles=roles, pair_roster_sha256=shared)
    if not scoring:
        write_json(freeze, manifest)
        return
    if read_json(freeze) != manifest:
        raise ValueError("The position-attribution freeze differs from the scorer input")
    population, match_q, truths, matches, old_arrays, paired_arrays = [], [], [], [], {}, {}
    for i, (role, (q, index, counts, arrays)) in enumerate(pieces.items()):
        score = read_plan(previous, ref, "score", role)
        if not score["query"].drop(columns="level3_row_index").equals(q):
            raise ValueError("Physical target wells differ from their original registration")
        y = effect(score["query"], score["controls"], ref, genes, cols, "ABC"[(i+2)%3])
        _, old = original(previous, frozen, role, q, len(genes))
        population.append(q.assign(role=role))
        match_q.append(q.iloc[index].assign(role=role, source_plate_pairs=counts))
        truths.append(y)
        matches.append(y[index])
        old_arrays[role] = {**old, "B0": np.zeros_like(y)}
        paired_arrays[role] = {**arrays, SCALED: old[SCALED][index], "B0": np.zeros_like(y[index])}
    tables = {name: [] for name in ("original", "centered", "retrieval", "scaffold", "influence", "matched")}
    all_rows = checked_rows(pd.concat(population, ignore_index=True), None)
    pair_rows = checked_rows(pd.concat(match_q, ignore_index=True), None)
    for dose in DOSES:
        full = all_rows.actual_dose_uM.eq(dose)
        measured = np.concatenate(truths)[full]
        previous_arms = {k: np.concatenate([v[k] for v in old_arrays.values()])[full] for k in old_arrays["A"]}
        for name, records in diagnostics(all_rows.loc[full].reset_index(drop=True), measured, previous_arms, dose).items():
            tables[name].extend(records)
        for n in (1, 3):
            use = pair_rows.actual_dose_uM.eq(dose) & pair_rows.source_plate_pairs.ge(n)
            values = {k: np.concatenate([v[k] for v in paired_arrays.values()])[use] for k in paired_arrays["A"]}
            digest = compare(pair_rows.loc[use].reset_index(drop=True), np.concatenate(matches)[use], values, U2OS_PAIRS, dose)
            tables["matched"].extend(dict(r, min_donor_plates=n) for r in digest)
    with TemporaryDirectory(dir=args.run, prefix=".position-score-") as temp:
        stage = Path(temp)
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(stage / (name + ".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(manifest, freeze_sha256=args.digest,
            files={path.name: sha256(path) for path in stage.iterdir()}))
        stage.rename(args.run / "score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "freeze", "score"))
    for key in ("run", "previous", "access-pin", "previous-pin", "previous-protocol", "protocol"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--role", choices=tuple("ABC"))
    parser.add_argument("--digest")
    args = parser.parse_args()
    (predict if args.phase == "predict" else finish)(args, *base_reference(args))
