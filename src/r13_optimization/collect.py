"""Collect small optimization evidence and retain digest links to server artifacts."""

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json


def collect(run, out):
    out.mkdir(parents=True, exist_ok=False)
    results, receipts, choices = [], [], []
    for protocol in ("compound", "scaffold"):
        source = run / protocol
        conf = read_json(source / "freeze.json")
        score = read_json(source / "scored/receipt.json")
        if score["freeze_sha256"] != sha256(source / "freeze.json"):
            raise ValueError(f"{protocol} score does not match freeze")
        for name, check in score["files"].items():
            p = source / "scored" / name
            if sha256(p) != check:
                raise ValueError(f"Changed score result {p}")
            if p.suffix == ".tsv":
                shutil.copyfile(p, out / f"{protocol}_{name}")
        shutil.copyfile(source / "freeze.json", out / f"{protocol}_freeze.json")
        for i in range(6):
            task = f"task{i}"
            receipt = source / "frozen" / task / "receipt.json"
            if conf["task_receipts"][task] != sha256(receipt):
                raise ValueError(f"Changed predictor receipt {receipt}")
            rec = read_json(receipt)
            baseline = read_json(source / "control" / task / "selection.json")["best"]["contrast"]
            choices.append(dict(protocol=protocol, task=task, fold=rec["fold"], role=rec["role"],
                                chosen=rec["chosen"], ensemble=",".join(rec["ensemble"]) or "-"))
            for family, candidate in rec["winners"].items():
                tuned = source / family / task / "selection.json"
                t = read_json(tuned)
                if rec["tune_receipts"][family] != sha256(tuned):
                    raise ValueError(f"Changed tuning receipt {tuned}")
                if t["grid_sha256"] != sha256(tuned.parent / "grid.tsv"):
                    raise ValueError(f"Changed tuning grid {tuned}")
                results.append(dict(protocol=protocol, family=family, task=task, fold=rec["fold"],
                                    role=rec["role"], names=t["n_specs"], scored_settings=t["n_inner_metrics"],
                                    selected=candidate["name"], gain=candidate["gain"],
                                    inner_contrast=candidate["contrast"], compared_to_b0=candidate["contrast"] - baseline,
                                    inner_absolute=candidate["absolute"], seconds=t["wall_seconds"]))
                receipts.append(dict(path=str(tuned.parent / "grid.tsv"), sha256=t["grid_sha256"]))
            for name, h in rec["files"].items():
                receipts.append(dict(path=str(receipt.parent / name), sha256=h))
        receipts.append(dict(path=str(source / "scored/bootstrap.npz"), sha256=score["files"]["bootstrap.npz"]))
    pd.DataFrame(results).to_csv(out / "inner_selection.tsv", sep="\t", index=False)
    pd.DataFrame(choices).to_csv(out / "outer_choices.tsv", sep="\t", index=False)
    pd.DataFrame(receipts).to_csv(out / "server_artifacts.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", {p.name: sha256(p) for p in out.iterdir()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args.run.resolve(strict=True), args.out)


if __name__ == "__main__":
    main()
