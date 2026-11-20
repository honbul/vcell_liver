"""Use the R11 many-to-one pool join with unchanged frozen R12 predictions."""

import argparse
from pathlib import Path
import sys

from src.r11_chemical_probe.common import sha256, write_json
from src.r11_chemical_probe.dataset import KEY, LINE
from src.r11_chemical_probe.report import align, align_pool

from . import score, secondary


def match_rows(left, right, right_key=KEY):
    if list(right_key) == ["plate_id", LINE]:
        return align_pool(left, right, list(right_key))
    return align(left, right, right_key)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--target", choices=["score", "secondary"], required=True)
    args, remaining = parser.parse_known_args()
    module = score if args.target == "score" else secondary
    module.align = match_rows
    sys.argv = [sys.argv[0], *remaining]
    module.main()
    output = Path(remaining[remaining.index("--out") + 1])
    write_json(output / "join_revision.json", dict(
        source_sha256=sha256(__file__),
        target=args.target,
        frozen_scorer_sha256=sha256(module.__file__),
        change="Restore the R11 checked many-to-one plate and cell line DMSO lookup. "
               "The group manifest, score-control roles, prediction files, estimator and contrasts are identical.",
    ))


if __name__ == "__main__":
    main()
