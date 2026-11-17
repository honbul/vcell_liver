"""Pilot helper: give long method ids (ddpm:<run id>, cells:<path>) short names in a copy of an
evaluation table, so that training seeds of one model are pooled by the unchanged aggregate.py.

Scores are not recomputed or altered; only the `method` column is renamed (prefix before '@' kept
apart from an optional mean-kind suffix). The mapping is a JSON object {old method: new name}.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import data as D
from .setting import run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--map", required=True, help="JSON file {old: new}")
    a = ap.parse_args()
    out = run_dir("eval", a.setting)
    dst = out / f"{a.out_tag}.tsv.gz"
    if dst.exists():
        raise SystemExit(f"{dst} exists; evaluation tags are immutable")
    mp = json.load(open(a.map))
    R = pd.read_csv(out / f"{a.tag}.tsv.gz", sep="\t")
    missing = sorted(set(R.method) - set(mp))
    if missing:
        raise SystemExit(f"methods without a name: {missing}")
    R["method"] = R.method.map(mp)
    R.to_csv(dst, sep="\t", index=False, float_format="%.6g")
    D.dump_json(dict(tag=a.out_tag, source_tag=a.tag, map=mp, sha256=D.sha256_file(dst)), out / f"{a.out_tag}.json")
    print(R.groupby(["method", "train_seed", "sample_seed"]).size().to_string())


if __name__ == "__main__":
    main()
