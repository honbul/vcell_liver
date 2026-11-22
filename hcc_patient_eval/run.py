"""Freeze and audit a read-only HCC functional model on public TCGA-LIHC tumors."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
RUN = ROOT / "hcc_patient_eval_v1"
FUNCTION = ROOT / "vc_build_v1/function"
RELEASE = FUNCTION / "function-registered.ekrmqfpZ"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", help="Alternative release adapter as module:factory")
    parser.add_argument("--stage", choices=("all", "freeze", "score"), default="all")
    parser.add_argument("--digest", help="Required freeze SHA-256 for independent scoring")
    parser.add_argument("--score-out", type=Path, help="New score directory for the verified existing freeze")
    defaults = dict(upstream=RELEASE / "repo", baseline=None, cohort=RUN / "gdc-20220418-v1",
                    raw=RUN / "raw", annotations=RUN / "annotations-v1", limore=FUNCTION / "limore-v1",
                    limore_maf=Path("/mnt/S1/sdata/pubdata/LIMORE/Cancer_functional_genes_of_81_cell_lines.txt"),
                    cross=FUNCTION / "cross-v1/train", depmap=Path("/mnt/S1/sdata/pubdata/DepMap/25Q3/Model.csv"),
                    out=RUN / "audit")
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    args = parser.parse_args()
    if args.model.is_file():
        args.model = args.model.parent
    if args.baseline is not None and args.baseline.is_file():
        args.baseline = args.baseline.parent
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    if args.stage in ("freeze", "all"):
        from .forecast import freeze

        freeze(args)
    if args.stage in ("score", "all"):
        from .cohort import digest
        from .score import score

        frozen = args.out / "freeze.json"
        requested = args.digest if args.stage == "score" else digest(frozen)
        if not requested:
            parser.error("--stage score requires --digest")
        score(args, requested)


if __name__ == "__main__":
    main()
