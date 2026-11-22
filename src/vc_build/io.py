"""Read study contracts and seal immutable forecast artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORECAST_KEYS = ("row_id", "track", "fold", "method")
OUTCOME_COLUMNS = {
    "actual", "actual_value", "endpoint_value", "observed", "observed_value",
    "target", "truth", "y", "y_true",
}


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_contract(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != 1 or document.get("study") != "vc_build_v1":
        raise ValueError("Unsupported study contract")
    return document


def read_table(path: str | Path):
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".tsv":
        key_types = {key: str for key in (*FORECAST_KEYS, "cell_id", "lineage_group", "compound_id")}
        return pd.read_csv(path, sep="\t", dtype=key_types, keep_default_na=False, na_values=[""])
    raise ValueError("Expected a .tsv or .parquet table")


def _publish(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish_json(path: str | Path, data: dict[str, Any]) -> None:
    def write(path: Path) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    _publish(Path(path), write)


def receipt_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".freeze.json")


def _check_forecasts(frame) -> None:
    keys = list(FORECAST_KEYS)
    missing = set(keys + ["predicted_value"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast table lacks columns {sorted(missing)}")
    if set(map(str.lower, frame.columns)) & OUTCOME_COLUMNS:
        raise ValueError("Forecast artifacts cannot contain observed outcomes")
    if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Expected nonempty, unique forecast identities")


def freeze_table(frame, path: str | Path, *, provenance: dict[str, Any]) -> dict[str, Any]:
    """Create one forecast artifact and hash receipt. A partial seal fails closed."""
    path = Path(path)
    seal = receipt_path(path)
    if path.exists() or seal.exists():
        raise FileExistsError(path)
    if path.suffix not in {".tsv", ".parquet"}:
        raise ValueError("Expected a .tsv or .parquet forecast")
    _check_forecasts(frame)
    json.dumps(provenance, allow_nan=False)

    def write(path_out: Path) -> None:
        if path.suffix == ".parquet":
            frame.to_parquet(path_out, index=False)
        else:
            frame.to_csv(path_out, sep="\t", index=False, lineterminator="\n")

    _publish(path, write)
    receipt = {
        "schema_version": 1,
        "artifact": path.name,
        "sha256": sha256(path),
        "rows": len(frame),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
    }
    publish_json(seal, receipt)
    return receipt


def verify_forecast(path: str | Path):
    path = Path(path)
    receipt = json.loads(receipt_path(path).read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact") != path.name
        or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("sha256", ""))
        or receipt["sha256"] != sha256(path)
    ):
        raise ValueError("Forecast does not match its freeze receipt")
    frame = read_table(path)
    _check_forecasts(frame)
    if receipt.get("rows") != len(frame):
        raise ValueError("Forecast row count differs from freeze receipt")
    return frame, receipt
