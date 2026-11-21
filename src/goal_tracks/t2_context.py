"""Build frozen scGPT and UCE representations of the additional Tahoe DMSO wells."""

import argparse
from pathlib import Path

from src.r11_chemical_probe.common import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample", "scgpt", "uce"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--r12-assets", type=Path, required=True)
    parser.add_argument("--scgpt-code", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.stage == "sample":
        from src.r12_multiaxis.cells import collect

        from src.r11_chemical_probe.dataset import table

        metadata = args.r12_assets / "doses"
        samples = table(metadata / "metadata/samples.tsv")
        primary = set(cfg["plates"])
        cfg["plates"] = sorted(set(samples.plate_id) - primary)
        cfg["r11_assets"] = str(metadata)
        collect(cfg, args.root / "cells-v2")
    elif args.stage == "scgpt":
        from src.r12_multiaxis.scgpt import embed

        embed(cfg, args.root, args.root / "scgpt", args.scgpt_code)
    else:
        from src.r12_multiaxis.uce import embed

        embed(cfg, args.root, args.r12_assets / "uce-checkpoint",
              args.r12_assets / "uce-code", args.root / "uce", batch_size=16)


if __name__ == "__main__":
    main()
