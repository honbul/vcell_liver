"""Prepare read-only public inputs for the practical virtual cell."""

import argparse
from pathlib import Path
import os

import yaml

from .function_data import write_limore


DEFAULT_CONFIG = Path("configs/vc_build_v1/contract.yaml")


def load_settings(config, function_root=None):
    config_path = Path(config)
    top = yaml.safe_load(config_path.read_text())
    function_sources = Path(top.get("function_source_config", config_path))
    settings = yaml.safe_load(function_sources.read_text())
    function_root = Path(function_root or os.environ.get("VC_BUILD_FUNCTION_ROOT") or settings["function_root"])

    def resolve(item):
        if isinstance(item, dict):
            return {key: resolve(value) for key, value in item.items()}
        if isinstance(item, list):
            return [resolve(value) for value in item]
        if isinstance(item, str):
            return item.replace("${function_root}", str(function_root))
        return item

    settings = resolve(settings)
    settings["function_root"] = str(function_root)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--function-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", choices=("limore", "cross"), default="limore")
    parser.add_argument("--limore", type=Path)
    args = parser.parse_args()
    settings = load_settings(args.config, args.function_root)
    root = Path(settings["function_root"])
    if args.bundle == "cross":
        from .cross_source_data import prep_cross

        output = args.output or root / "cross-v1"
        receipt = prep_cross(settings, args.limore or root / "limore-v1", output)
        print(f"{output} | {receipt['train_rows']} fit rows and {receipt['query_rows']} distinct query rows")
    else:
        output = args.output or root / "limore-v1"
        receipt = write_limore(settings, output)
        print(f"{output} | {receipt['n_cells']} cells, {receipt['n_compounds']} compounds, {receipt['n_rows']} rows")


if __name__ == "__main__":
    main()
