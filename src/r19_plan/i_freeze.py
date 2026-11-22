"""Freeze I-track predictions and availability together before any outer score."""

from pathlib import Path
from tempfile import TemporaryDirectory
import re

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_predict import ALPHAS, GAINS, WELL_FIELDS

FILES = {"rows.tsv", "landmarks.tsv", "forecasts.npz", "kernel_only.npz", "coverage.npz", "selection.tsv"}
ROW_FIELDS = [*WELL_FIELDS, "parent", "fold", "role"]
HASH = re.compile(r"[a-f0-9]{64}")
FOLD_ROLES = {(fold, role) for fold in range(3) for role in "ABC"}


def check_prediction_shapes(rows, genes, forecasts, reference):
    if (rows.empty or rows[ROW_FIELDS].isna().any().any()
            or rows.duplicated(["study", "rna_plate", "rna_well", "role"]).any()):
        raise ValueError("An I-track forecast's registered physical-well and role axis is invalid")
    if len(rows.fold.unique()) != 1 or rows.fold.iloc[0] not in (0, 1, 2):
        raise ValueError("An I-track forecast must have one registered outer fold")
    if len(rows.role.unique()) != 1 or rows.role.iloc[0] not in ("A", "B", "C"):
        raise ValueError("An I-track forecast must have one registered fit role")
    if len(genes) == 0 or genes.pr_gene_id.duplicated().any() or genes.pr_gene_id.isna().any():
        raise ValueError("The I-track landmark order must be nonempty and unique")
    shape = (len(rows), len(genes))
    checked_array(reference, shape, "B_R17 locked reference")
    if "B_R17" in forecasts or "I0" not in forecasts:
        raise ValueError("The I-track reference and structure fallback must be separate")
    for name, prediction in forecasts.items():
        checked_array(prediction.full, shape, name)
        checked_array(prediction.kernel_only, shape, f"{name} kernel diagnostic")
        count = prediction.view_count
        if (count.shape != (len(rows),) or count.dtype.kind not in "iu"
                or (count < 0).any() or (count > len(prediction.views)).any()):
            raise ValueError(f"Invalid availability axis: {name}")
        if not prediction.views or name.endswith(("_mask_only", "_morphology_only")):
            continue
        for key in ("full", "kernel_only"):
            if not np.array_equal(getattr(prediction, key)[count == 0], getattr(forecasts["I0"], key)[count == 0]):
                raise ValueError(f"The no-view forecast must use the identical I0 value: {name}")


def freeze_fold(path, rows, genes, forecasts, reference, choices, locked_inputs, protocol, reference_selection=None):
    """Write the forecast and its row, gene, support, and input hashes atomically."""
    if path.exists():
        raise FileExistsError(path)
    if not locked_inputs or any(HASH.fullmatch(digest) is None for digest in locked_inputs.values()):
        raise ValueError("The I-track forecast requires pinned input receipts")
    check_prediction_shapes(rows, genes, forecasts, reference)
    if set(choices) != set(forecasts) or any(
        choice.alpha not in ALPHAS or choice.gain not in GAINS or not np.isfinite(choice.loss)
        for choice in choices.values()
    ):
        raise ValueError("The I-track forecast differs from the registered selection")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".i-predict-") as temp:
        stage = Path(temp)
        rows.to_csv(stage / "rows.tsv", sep="\t", index=False)
        genes.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecasts.npz", B_R17=reference,
                            **{name: forecast.full for name, forecast in forecasts.items()})
        np.savez_compressed(stage / "kernel_only.npz", **{name: f.kernel_only for name, f in forecasts.items()})
        np.savez_compressed(stage / "coverage.npz", **{name: f.view_count for name, f in forecasts.items()})
        pd.DataFrame([dict(arm=name, alpha=choice.alpha, gain=choice.gain, selection_loss=choice.loss)
                      for name, choice in choices.items()]).to_csv(stage / "selection.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(
            fold=int(rows.fold.iloc[0]), fit_role=str(rows.role.iloc[0]), protocol_sha256=sha256(protocol),
            inputs=locked_inputs, views={name: list(forecast.views) for name, forecast in forecasts.items()},
            reference_selection=reference_selection,
            files={file.name: sha256(file) for file in sorted(stage.iterdir())},
        ))
        stage.rename(path)
    return sha256(path / "receipt.json")


def verify_fold(path, expected_sha256, expected_rows, expected_genes, protocol):
    """Check the separately handed-off digest and the registered row and landmark axes."""
    file = path / "receipt.json"
    if sha256(file) != expected_sha256:
        raise ValueError("The I-track fold receipt has changed since freezing")
    receipt = read_json(file)
    if receipt["protocol_sha256"] != sha256(protocol) or set(receipt["files"]) != FILES:
        raise ValueError("The I-track freeze used a different contract")
    for name, digest in receipt["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"The frozen I-track artifact changed: {name}")
    if (path / "rows.tsv").read_bytes() != expected_rows.to_csv(sep="\t", index=False).encode():
        raise ValueError("I-track predictions differ from the independently registered score row order")
    if (path / "landmarks.tsv").read_bytes() != expected_genes.to_csv(sep="\t", index=False).encode():
        raise ValueError("I-track predictions differ from the independently registered score gene order")
    return receipt


def slice_receipts(forecasts, queries, genes, protocol):
    if set(queries) != FOLD_ROLES:
        raise ValueError("The I-track freeze requires all nine registered fold and role slices")
    seen, seen_wells = set(), set()
    receipts = {}
    for fold in range(3):
        source = queries[fold, "A"].drop(columns="role")
        parents = set(source.parent)
        wells = set(source[["study", "rna_plate", "rna_well"]].itertuples(index=False, name=None))
        if seen & parents or seen_wells & wells:
            raise ValueError("An I-track parent or physical well belongs to multiple outer score folds")
        seen |= parents
        seen_wells |= wells
        for role in "ABC":
            rows = queries[fold, role]
            if (not rows.fold.eq(fold).all() or not rows.role.eq(role).all()
                    or not rows.drop(columns="role").equals(source)):
                raise ValueError("I-track role rotations must share their physical query axis")
            key = f"f{fold}-{role}"
            path = forecasts / key
            digest = sha256(path / "receipt.json")
            stamp = verify_fold(path, digest, rows, genes, protocol)
            if (stamp["fold"], stamp["fit_role"]) != (fold, role):
                raise ValueError("The I-track fold receipt reports a different role")
            receipts[key] = digest
    return receipts


def run_stamp(forecasts, queries, genes, protocol):
    module = Path(__file__)
    return dict(protocol_sha256=sha256(protocol), code_sha256=sha256(module),
                score_code_sha256={name: sha256(module.with_name(name))
                                  for name in ("i_evaluate.py", "i_score.py", "i_workflow.py")},
                folds=slice_receipts(forecasts, queries, genes, protocol))


def freeze_run(path, forecasts, queries, genes, protocol):
    """Register all nine receipts before the score workflow may load held RNA."""
    stamp = run_stamp(forecasts, queries, genes, protocol)
    write_json(path, stamp)
    return sha256(path)


def verify_run(path, expected_sha256, forecasts, queries, genes, protocol):
    if sha256(path) != expected_sha256:
        raise ValueError("The I-track run freeze digest has changed")
    stamp = read_json(path)
    current = run_stamp(forecasts, queries, genes, protocol)
    if stamp != current:
        raise ValueError("An I-track fold changed after the run freeze")
    return stamp
