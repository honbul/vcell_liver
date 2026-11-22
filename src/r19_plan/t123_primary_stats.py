"""Replay the registered T3–T1 comparison for the DEV joint directional family."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.sl_workflow import parent_swap
from src.r19_plan.x_inputs import matrix_axes, read_plan, reference, slice_axes
from src.r19_plan.x_workflow import score_delta


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    freeze = read_json(args.freeze)
    score = read_json(args.score / "receipt.json")
    if (score["freeze_sha256"] != sha256(args.freeze) or freeze["access_sha256"] != sha256(args.pin)
            or any(score[k] != v for k, v in freeze.items())
            or sha256(args.score/"paired.tsv") != score["files"]["paired.tsv"]):
        raise ValueError("T1 and T3 official forecast and score lineage changed")
    ref = reference(args.pin)
    genes, columns = matrix_axes(args.access, ref)
    pieces, known, targets, forecasts = {}, [], [], {"T1": [], "T3": []}
    if set(freeze["slices"]) != {key for key in ref["slices"] if key.startswith("HEPG2-")}:
        raise ValueError("T1–T3 freeze lacks the complete HEPG2 comparison")
    for key, digest in sorted(freeze["slices"].items()):
        path = args.forecasts/key
        manifest = read_json(path/"receipt.json")
        if (sha256(path/"receipt.json") != digest or set(manifest["files"]) != {"selection.json", "forecasts.npz"}
                or any(sha256(path/name) != h for name, h in manifest["files"].items())):
            raise ValueError("A frozen T1–T3 forecast changed")
        _, _, plan = read_plan(args.access, args.pin, key, "predict")
        query = pieces[key] = plan["query"]
        _, _, fold, role, _, _ = slice_axes(key, ref["cells"])
        known.append(query.assign(fold=fold, role=role))
        with np.load(path/"forecasts.npz", allow_pickle=False) as array:
            for arm in forecasts:
                forecasts[arm].append(checked_array(array[arm], (len(query), len(genes)), arm))
    rows = checked_rows(pd.concat(known, ignore_index=True), None)
    for key, query in pieces.items():
        targets.append(score_delta(args, key, query, genes, columns))
    truth = np.concatenate(targets)
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, np.concatenate(seq)) for arm, seq in forecasts.items()}

    def by_parent(mask):
        return parent_summaries(rows.loc[mask], truth.shape[1], energy[mask], errors["T3"][mask], errors["T1"][mask])

    stats = by_parent(np.ones(len(rows), dtype=bool))
    observed, bounds, _ = paired_effects(*stats)
    table = pd.read_csv(args.score/"paired.tsv", sep="\t")
    selected = table.loc[table.scope.eq("ALL") & table.arm.eq("T3") & table.reference.eq("T1")].set_index("metric")
    order = selected.loc[list(METRICS)]
    if (not np.allclose(observed, order[["arm_value", "reference_value"]], rtol=0, atol=1e-12)
            or not np.allclose(bounds, order[["low99","high99"]], rtol=0, atol=1e-12)):
        raise ValueError("T1–T3 score replay failed the official point and interval check")
    p, joint = parent_swap(*stats)
    effects = []
    for name, values in (("fold", rows.fold), ("plate_removed", rows.rna_plate)):
        for key in sorted(values.unique()):
            mask = values.eq(key) if name == "fold" else values.ne(key)
            rmse, sse, total = by_parent(mask)
            effects.append(dict(scope=name, level=key, parents=len(rmse), delta_rmse=(rmse[:,0]-rmse[:,1]).mean(),
                                delta_r2=(sse[:,1]-sse[:,0]).sum()/total.sum()))
    args.out.mkdir(parents=True)
    pd.DataFrame({"metric": METRICS, "one_sided_p": p, "joint_p_unadjusted": joint}).to_csv(
        args.out/"primary_p.tsv", sep="\t", index=False)
    pd.DataFrame(effects).to_csv(args.out/"sensitivity.tsv", sep="\t", index=False)
    write_json(args.out/"receipt.json", dict(scored_replay=True, fitted=False, release="GSE92742",
        freeze_sha256=sha256(args.freeze), score_sha256=sha256(args.score/"receipt.json"),
        primary="T3-T1", contract="B", parents=rows.parent.nunique(), wells=rows.inst_id.nunique(),
        code_sha256=sha256(__file__), files={f.name: sha256(f) for f in args.out.iterdir()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("access", "pin", "freeze", "forecasts", "score", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())
