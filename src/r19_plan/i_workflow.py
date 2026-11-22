"""Freeze all I-track predictions or score them in the independent score process."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r19_plan.i_evaluate import evaluate_i_run
from src.r19_plan.i_freeze import FOLD_ROLES, freeze_run
from src.r19_plan.i_stage import load_score_delta, parse_landmarks, read_query_metadata


def registered_axes(plan, reference, cache):
    registered = read_json(reference)
    if sha256(plan / "receipt.json") != sha256(reference) or not registered["metadata_only"]:
        raise ValueError("The I-track split differs from its metadata lock")
    landmark_path = cache / "landmarks.tsv"
    if sha256(landmark_path) != registered["landmarks_sha256"]:
        raise ValueError("The I-track gene axis differs from the registered landmark order")
    genes = parse_landmarks(landmark_path.read_bytes())
    raw, queries, fit_parents = {}, {}, {}
    for fold, role in sorted(FOLD_ROLES):
        slice_id = f"f{fold}-{role}"
        raw[slice_id] = read_query_metadata(plan, reference, slice_id)
        queries[fold, role] = raw[slice_id].assign(fold=fold, role=role)
        if role == "A":
            filename = "inner_and_outer_labels.tsv"
            label_path = plan / "predict-plan" / slice_id / filename
            if sha256(label_path) != registered["files"][slice_id]["predict"][filename]:
                raise ValueError("The I-track fit-parent inventory changed")
            fit_parents[fold] = set(table(label_path).parent)
    return raw, queries, genes, fit_parents


def run(args):
    raw, queries, genes, fit = registered_axes(args.plan, args.split_reference, args.cache)
    if args.action == "freeze":
        return freeze_run(args.freeze, args.forecasts, queries, genes, args.protocol)
    if args.out.exists():
        raise FileExistsError(args.out)

    def read_delta(slice_id, _query):
        return load_score_delta(args.cache, args.vehicles, args.plan, args.split_reference,
                                slice_id, raw[slice_id], genes)

    tables = evaluate_i_run(args.freeze, args.expected_sha256, args.forecasts, queries, genes, args.protocol,
                            read_score_delta=read_delta, fit_parents_by_fold=fit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".i-score-") as tmp:
        stage = Path(tmp)
        for name, rows in tables.items():
            rows.to_csv(stage / f"{name}.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(
            freeze_sha256=args.expected_sha256, protocol_sha256=sha256(args.protocol),
            split_sha256=sha256(args.split_reference), evaluator_sha256=sha256(__file__),
            files={path.name: sha256(path) for path in sorted(stage.iterdir())},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    for name in ("plan", "split-reference", "cache", "forecasts", "protocol", "freeze"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("vehicles", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.action == "score" and any(value is None for value in (args.vehicles, args.out, args.expected_sha256)):
        parser.error("score requires --vehicles, --out, and --expected-sha256")
    print(run(args))
