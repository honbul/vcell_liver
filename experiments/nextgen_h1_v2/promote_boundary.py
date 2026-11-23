"""Publish one checked final boundary receipt and preserve its predecessor."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def publish(old, new, archive):
    prev = old / "result.json"
    revision = json.loads((new / "result.json").read_text())
    if revision["resume_sha256"] != sha256(prev.read_bytes()).hexdigest():
        raise ValueError("The resumed evidence does not match its predecessor")
    if revision["outer_labels_opened"] or any(
            v["remaining_numerical_boundary_trials"] != 0 for v in revision["modes"].values()):
        raise ValueError("The selected comparison still has an unchecked numerical boundary")
    if archive.exists() or old.is_symlink():
        raise FileExistsError("Boundary receipt was already promoted")
    archive.parent.mkdir(parents=True, exist_ok=True)
    old.rename(archive)
    try:
        old.symlink_to(new.resolve(), target_is_directory=True)
    except BaseException:
        archive.rename(old)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "new", "archive"):
        parser.add_argument(name, type=Path)
    publish(**vars(parser.parse_args()))
