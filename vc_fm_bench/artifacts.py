"""Content-address the data used at the feature, prediction and scoring boundaries."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import tempfile


def digest(path: Path) -> str:
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


@contextmanager
def new_directory(path: Path):
    """Publish a directory only after the complete artifact has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as tmp:
        staging = Path(tmp)
        yield staging
        # A nonempty completed destination cannot be replaced by rename.
        staging.rename(path)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_record(path: Path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def verify_record(path: Path, record):
    if file_record(path) != record:
        raise ValueError(f"Artifact differs from its receipt: {path}")
