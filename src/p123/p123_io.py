"""Write and load P1–P3 forecasts on the released, verified T1 task axis."""

from pathlib import Path
import shutil
import tempfile

import numpy as np

from src.goal_tracks.t1_data import array, eligibility, slice_name, verify_one
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY, table


def write_prediction(path, source, fit, predictions, selection, protocol):
    path = Path(path)
    if fit.query.treated is not None or not np.array_equal(fit.panel, source.panel):
        raise ValueError("Forecast has outer treated data or a different gene panel")
    if not fit.query.rows[KEY].equals(source.query_rows[KEY]):
        raise ValueError("Forecast rows differ from the registered task")
    if not predictions:
        raise ValueError("No prediction arms")
    for name, values in predictions.items():
        if values.shape != fit.query.control.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid forecast values for {name}")
        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid forecast name {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        fit.query.rows.to_csv(temporary / "rows.tsv", sep="\t", index=False)
        np.save(temporary / "panel.npy", fit.panel, allow_pickle=False)
        for name, value in predictions.items():
            np.save(temporary / f"{name}.npy", np.asarray(value, dtype=np.float32), allow_pickle=False)
        write_json(temporary / "receipt.json", dict(
            protocol_sha256=sha256(protocol), stage_sha256=sha256(source.staged / "receipt.json"),
            dose=source.dose, minimum_cells=source.minimum_cells, fold=source.outer, role=source.role,
            models=sorted(predictions), selection=selection,
            rows=len(fit.query.rows), held_parents=int(fit.query.rows[GROUP].nunique()),
            files={p.name: sha256(p) for p in sorted(temporary.iterdir())},
        ))
        if path.exists():
            raise FileExistsError(path)
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_prediction(path, stage, protocol, dose, n, fold, role):
    path = Path(path)
    stamp = read_json(path / "receipt.json")
    if (stamp["protocol_sha256"] != sha256(protocol)
            or stamp["stage_sha256"] != sha256(stage / "receipt.json")
            or (stamp["dose"], stamp["minimum_cells"], stamp["fold"], stamp["role"]) != (dose, n, fold, role)):
        raise ValueError(f"Prediction receipt differs from fixed study context: {path}")
    if set(stamp["files"]) != {"rows.tsv", "panel.npy", *(f"{name}.npy" for name in stamp["models"])}:
        raise ValueError("Prediction is incomplete")
    for name, expected in stamp["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Prediction artifact changed: {path / name}")
    base = stage / f"f{fold}"
    receipt = read_json(stage / "receipt.json")
    for name in ("score_rows.tsv", "panels.json"):
        verify_one(stage, receipt, base / name)
    rows = table(path / "rows.tsv")
    baseline = eligibility(table(base / "score_rows.tsv"), n, dose)
    fields = KEY + [GROUP, "barcode_position", "plate_id", "dose_uM", "time_h", "n_cells"]
    if not (rows[fields].equals(baseline[fields]) and rows[CONTROL].equals(baseline[f"control_{role}"])):
        raise ValueError("Prediction does not use the registered rows and role")
    panel = array(path / "panel.npy")
    panel_label = slice_name(dose, n, fold, "panel") + "_-1"
    if not np.array_equal(panel, read_json(base / "panels.json")[panel_label]):
        raise ValueError("Prediction does not use the registered fit-only panel")
    predictions = {name: array(path / f"{name}.npy") for name in stamp["models"]}
    if any(matrix.shape != (len(rows), len(panel)) for matrix in predictions.values()):
        raise ValueError("Prediction shape does not match axes")
    return stamp, rows, panel, predictions
