"""Replay fixed contrasts and quantify parent and physical-plate sensitivity."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.i_stage import parse_tsv, pinned_bytes
from src.r19_plan.t4_workflow import axes, reference as time_reference, score_delta as time_score
from src.r19_plan.x_inputs import matrix_axes, read_plan, reference as matched_reference, slice_axes
from src.r19_plan.x_workflow import score_delta as matched_score

TRACKS = {
    "T4": (("operator6", "scalar6"), ("operator6", "identity6"), ("operator6", "direct24")),
    "SL_B": (("B_R17", "B_blend_anchor"),),
}
COUNTS = {"T4": (132, 268, 12), "SL_B": (493, 2324, 41)}
DRAW_NAMES = ("relative_rmse_gain", "delta_r2")
DRAW_COUNT = 10_000


def frozen(args):
    run = args.run
    receipt = read_json(run/"score/receipt.json")
    stamp = read_json(run/"freeze.json")
    if (sha256(run/"freeze.json") != receipt["freeze_sha256"] or stamp["access_sha256"] != sha256(args.pin)
            or stamp["protocol_sha256"] != sha256(args.original_protocol)
            or any(receipt[k] != value for k, value in stamp.items())
            or sha256(run/"score/paired.tsv") != receipt["files"]["paired.tsv"]):
        raise ValueError("The original score and frozen forecast lineages disagree")
    use_t4 = args.track == "T4"
    ref = (time_reference if use_t4 else matched_reference)(args if use_t4 else args.pin)
    slice_names = set(ref["slices"]) if use_t4 else {s for s in ref["slices"] if s.startswith("HEPG2-")}
    if set(stamp["slices"]) != slice_names:
        raise ValueError("The forecast freeze omits or adds an evaluation slice")
    genes, indices = matrix_axes(args.access, ref)
    required = {arm for pair in TRACKS[args.track] for arm in pair}
    filename = "forecasts.npz" if use_t4 else "full.npz"
    coordinates, parts = {}, {arm: [] for arm in required}
    for key, digest in sorted(stamp["slices"].items()):
        if use_t4:
            fold, role, _, _, _ = axes(key)
            source = ref["slices"][key]["predict"]["query.tsv"]
            query = parse_tsv(pinned_bytes(args.access/"predict-plan"/key/"query.tsv", source))
        else:
            _, _, fold, role, _, _ = slice_axes(key, ref["cells"])
            _, _, design = read_plan(args.access, args.pin, key, "predict")
            query = design["query"]
        path = run/"forecasts"/key
        prediction = read_json(path/"receipt.json")
        if (sha256(path/"receipt.json") != digest
                or prediction["query_sha256"] != ref["slices"][key]["predict"]["query.tsv"]
                or sha256(path/filename) != prediction["files"][filename]):
            raise ValueError("An original forecast or row register changed")
        with np.load(path/filename, allow_pickle=False) as values:
            for arm in required:
                parts[arm].append(checked_array(values[arm], (len(query), len(genes)), arm))
        coordinates[key] = (query, query.assign(fold=fold, role=role))
    rows = checked_rows(pd.concat([table[1] for table in coordinates.values()], ignore_index=True), None)
    shape = rows.parent.nunique(), rows.inst_id.nunique(), rows.rna_plate.nunique()
    if shape != COUNTS[args.track] or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("The original physical comparison support changed")
    # Read Score-only expression after every required forecast has been checked against its freeze.
    if use_t4:
        deltas = [time_score(args, ref, key, pair[0], indices, genes.pr_gene_id.to_numpy())
                  for key, pair in coordinates.items()]
    else:
        deltas = [matched_score(args, key, pair[0], genes, indices) for key, pair in coordinates.items()]
    truth = np.concatenate(deltas)
    energy = squared_rows(truth)
    errors = {name: squared_rows(truth, np.concatenate(seq)) for name, seq in parts.items()}
    return rows, len(genes), energy, errors, receipt


def verify_replay(original, track, arm, base, points, bounds):
    if track == "T4":
        selected = original.loc[original.scope.eq("ALL") & original.arm.eq(arm) & original.reference.eq(base)]
    else:
        selected = original.loc[original.track.eq("full") & original.arm.eq(base) & original.reference.eq(arm)]
    selected = selected.set_index("metric", verify_integrity=True).loc[list(METRICS)]
    expected_points = selected[["arm_value", "reference_value"]].to_numpy()
    expected_bounds = selected[["low99", "high99"]].to_numpy()
    if track == "SL_B":
        expected_points = expected_points[:, ::-1]
        expected_bounds = -expected_bounds[:, ::-1]
    if (not np.allclose(expected_points, points, rtol=0, atol=1e-12)
            or not np.allclose(expected_bounds, bounds, rtol=0, atol=1e-12)):
        raise ValueError("Frozen score replay does not reproduce the original paired comparison")


def points(rmse, sse, energy):
    prediction, base = rmse.mean(axis=0)
    return dict(rmse=prediction, baseline_rmse=base, relative_rmse_gain=1-prediction/base,
                delta_r2=(sse[:, 1]-sse[:, 0]).sum()/energy.sum())


def crossed_weights(rows, genes, energy, candidate, base, *, draws=DRAW_COUNT):
    pcodes, p = pd.factorize(rows.parent, sort=True)
    qcodes, q = pd.factorize(rows.rna_plate, sort=True)
    shape = len(p), len(q)
    flattened = pcodes*len(q)+qcodes

    def reduce(values):
        return np.bincount(flattened, weights=values, minlength=len(p)*len(q)).reshape(shape)

    n = reduce(None)
    a, b = (reduce(np.sqrt(value/genes)) for value in (candidate, base))
    e, reduction = reduce(energy), reduce(base-candidate)
    rng = np.random.default_rng(150104)
    p_weights = rng.exponential(size=(draws, len(p)))
    q_weights = rng.exponential(size=(draws, len(q)))
    samples = np.empty((draws, len(DRAW_NAMES)))
    for start in range(0, draws, 128):
        stop = min(start+128, draws)
        parents, plates = p_weights[start:stop], q_weights[start:stop]
        denominators = plates @ n.T
        scaled_arms = [(parents*(plates @ matrix.T)/denominators).sum(axis=1) for matrix in (a, b)]
        samples[start:stop, 0] = 1-scaled_arms[0]/scaled_arms[1]
        numerator = (parents*(plates @ reduction.T)).sum(axis=1)
        samples[start:stop, 1] = numerator/(parents*(plates @ e.T)).sum(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError("The crossed plate and parent sensitivity draws include invalid support")
    return np.quantile(samples, [.005, .995], axis=0).T


def sensitivity(rows, genes, energy, candidate, base):
    def part(keep):
        return parent_summaries(rows.loc[keep], genes, energy[keep], candidate[keep], base[keep])

    rmse, sse, exposure = part(np.ones(len(rows), dtype=bool))
    parents = sorted(rows.parent.unique())
    per_parent = pd.DataFrame(dict(parent=parents, rmse_decrease=rmse[:,1]-rmse[:,0],
        sse_decrease=sse[:,1]-sse[:,0], target_energy=exposure))
    removals, plates = [], []
    for label in ("rmse_decrease", "sse_decrease"):
        positives = per_parent.loc[per_parent[label].gt(0)].sort_values([label, "parent"], ascending=[False, True])
        per_parent[label+"_rank"] = per_parent.parent.map(dict(zip(positives.parent, range(1,len(positives)+1))))
        denom = positives[label].sum()
        for n in (5,10,20):
            selected = positives.head(n)
            keep = ~rows.parent.isin(selected.parent)
            removals.append(dict(metric_rank=label, omit_limit=n, omitted=len(selected), remaining=rows.loc[keep].parent.nunique(),
                share_positive=selected[label].sum()/denom if denom else np.nan, **points(*part(keep))))
    for plate in sorted(rows.rna_plate.unique()):
        keep = rows.rna_plate.ne(plate)
        plates.append(dict(plate=plate, remaining=rows.loc[keep].parent.nunique(),
                           remaining_wells=rows.loc[keep].inst_id.nunique(), **points(*part(keep))))
    model_points, bounds, _ = paired_effects(rmse, sse, exposure)
    influence = crossed_weights(rows, genes, energy, candidate, base)
    joint = [dict(metric=name, point=points(rmse, sse, exposure)[name], low99=lo, high99=hi, draws=DRAW_COUNT)
             for name, (lo, hi) in zip(DRAW_NAMES, influence, strict=True)]
    return per_parent, removals, plates, joint, model_points, bounds


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    rows, genes, energy, errors, score_receipt = frozen(args)
    original = pd.read_csv(args.run/"score/paired.tsv", sep="\t")
    merged = {table: [] for table in ("parents", "top_removed", "plate_removed", "crossed")}
    for arm, base in TRACKS[args.track]:
        result = sensitivity(rows, genes, energy, errors[arm], errors[base])
        verify_replay(original, args.track, arm, base, *result[4:])
        for dataset, frame in zip(merged, result[:4], strict=True):
            merged[dataset].append(pd.DataFrame(frame).assign(arm=arm, reference=base))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".robustness-") as tmp:
        stage = Path(tmp)
        for name, pieces in merged.items():
            pd.concat(pieces, ignore_index=True).to_csv(stage/(name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(track=args.track, original_freeze_sha256=score_receipt["freeze_sha256"],
            original_score_sha256=sha256(args.run/"score/receipt.json"), original_replayed=True,
            protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
            parents=rows.parent.nunique(), physical_wells=rows.inst_id.nunique(), plates=rows.rna_plate.nunique(),
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=tuple(TRACKS), required=True)
    for name in ("access","pin","run","protocol","original-protocol","out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
