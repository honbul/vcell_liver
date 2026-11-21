"""Snapshot and verify Tahoe forecasts before opening reserved treatment outcomes."""

from pathlib import Path

import numpy as np

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY, LINE, table
from src.r11_chemical_probe.report import align, align_pool

from .t1_data import PROTOCOL_SHA256, WELL, array, eligibility, slice_name, validate_protocol, verify_one


def write_task(path, source, fit, predictions, selection=None):
    path = Path(path)
    if fit.query.treated is not None or not np.array_equal(fit.panel, source.panel):
        raise ValueError("Outer scorer input is not the registered forecast panel")
    if not fit.query.rows[KEY].equals(source.query_rows[KEY]):
        raise ValueError("Outer scorer input is not the registered forecast row universe")
    for name, values in predictions.items():
        if values.shape != fit.query.control.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid forecast array: {name}")
        if not name.replace("-", "_").replace("_", "").isalnum():
            raise ValueError(f"Invalid forecast name: {name}")
    path.mkdir(parents=True, exist_ok=False)
    fit.query.rows.to_csv(path / "rows.tsv", sep="\t", index=False)
    np.save(path / "panel.npy", fit.panel, allow_pickle=False)
    for name, values in predictions.items():
        np.save(path / f"{name}.npy", np.asarray(values, dtype=np.float32), allow_pickle=False)
    write_json(path / "receipt.json", dict(
        protocol_sha256=PROTOCOL_SHA256, stage_sha256=sha256(source.staged / "receipt.json"),
        dose=source.dose, minimum_cells=source.minimum_cells, fold=source.outer, role=source.role,
        models=sorted(predictions), selection=selection or {}, rows=len(fit.query.rows),
        held_parents=int(fit.query.rows[GROUP].nunique()),
        files={p.name: sha256(p) for p in sorted(path.iterdir())},
    ))


def load_task(path, stage, protocol, dose, n, fold, role):
    path = Path(path)
    validate_protocol(protocol)
    receipt = read_json(path / "receipt.json")
    if ((receipt["dose"], receipt["minimum_cells"], receipt["fold"], receipt["role"]) != (dose, n, fold, role)
            or receipt["stage_sha256"] != sha256(stage / "receipt.json")
            or receipt["protocol_sha256"] != PROTOCOL_SHA256):
        raise ValueError("Forecast slice, cache or contract differs from its receipt")
    for name, expected in receipt["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Changed prediction artifact: {path / name}")
    folder = stage / f"f{fold}"
    staged_receipt = read_json(stage / "receipt.json")
    for name in ("score_rows.tsv", "panels.json"):
        verify_one(stage, staged_receipt, folder / name)
    rows = table(path / "rows.tsv")
    registered = eligibility(table(folder / "score_rows.tsv"), n, dose)
    expected_cols = KEY + [GROUP, "barcode_position", "plate_id", "dose_uM", "time_h", "n_cells"]
    if not (rows[expected_cols].equals(registered[expected_cols])
            and rows[CONTROL].equals(registered[f"control_{role}"])):
        raise ValueError("Forecast rows or role disagree with the registered evaluation")
    panel = array(path / "panel.npy")
    expected = read_json(folder / "panels.json")[slice_name(dose, n, fold, "panel") + "_-1"]
    if not np.array_equal(panel, expected):
        raise ValueError("Forecast panel disagrees with the fit-only registered panel")
    arrays = {name: array(path / f"{name}.npy") for name in receipt["models"]}
    return receipt, rows, panel, arrays


def score_truth(stage, dose, n, fold, role, rows, panel, verify=True):
    folder = stage / f"f{fold}"
    other = "B" if role == "A" else "A"
    expected = read_json(stage / "receipt.json")
    if verify:
        for name in ("tokens.npy", "score_rows.tsv", "score.npy", f"{other}_rows.tsv", f"{other}.npy"):
            verify_one(stage, expected, folder / name)
    tokens = array(folder / "tokens.npy")
    columns = np.searchsorted(tokens, panel)
    if np.any(columns >= len(tokens)) or not np.array_equal(tokens[columns], panel):
        raise ValueError("Scored panel is not in staged token union")
    staged = table(folder / "score_rows.tsv")
    selected = align(rows, staged)
    assert staged.iloc[selected].dose_uM.eq(dose).all()
    assert staged.iloc[selected].n_cells.ge(n).all()
    controls = table(folder / f"{other}_rows.tsv")
    cc = align_pool(rows, controls, ["plate_id", LINE])
    if np.array_equal(rows[CONTROL], controls.iloc[cc][WELL]) or (rows[CONTROL].to_numpy() == controls.iloc[cc][WELL].to_numpy()).any():
        raise ValueError("Predictor and scorer have the same physical control")
    return (array(folder / "score.npy")[np.ix_(selected, columns)]
            - array(folder / f"{other}.npy")[np.ix_(cc, columns)])
