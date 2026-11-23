"""Offline single-sample prediction from a sealed H1-v2 package."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h1_anchor.cli import read_anchors
from nextgen_h1_anchor.models import rna_forecast

from .boost import region
from .data import BUDGETS, rna_projection
from .freeze import ANCHOR, DUAL, read_package


def bounds(model, k):
    a = model.order[:k]
    if model.ctx.levels:
        return [[-.25, 1.25] for _ in a]
    t = model.ctx.train[:, a]
    buffer = 3 * model.ctx.sd[a]
    return np.column_stack((np.nanmin(t, axis=0) - buffer, np.nanmax(t, axis=0) + buffer)).tolist()


def single(package, name, k, values, rna=None):
    model = package["models"][name]
    ctx = model.ctx
    if model.dual:
        if rna is None or rna.ndim != 1 or len(rna) != package["train_rna"].shape[1]:
            raise ValueError("Dual inference requires one registered ranked RNA vector")
        if not np.isfinite(rna).all() or np.abs(rna).max() > .501:
            raise ValueError("RNA must be ranked on the registered gene axis with range [-0.5, 0.5]")
        x = package["train_rna"].astype(float)
        rna = rna.astype(float)[None]
        p, cross, _ = rna_forecast(ctx.train, x, rna, ctx.chem)
        _, pc = rna_projection(x, rna)
    else:
        p, pc, cross = np.array([ctx.mean]), np.zeros((1, 16)), np.zeros((1, len(ctx.train)))
    one = replace(ctx, valid=None, bvalid=p, pc_valid=pc, rna_valid_kernel=cross,
                  valid_groups=np.array(["offline"]), tissue=np.array(["unspecified"]))
    adapted = replace(model, ctx=one)
    q = region(model.order, k, len(ctx.mean))
    result = adapted.predict(values, k)[0, q]
    return q, result


def infer(frozen, anchors, k, name, rna):
    _, package = read_package(frozen)
    model = package["models"][name]
    values = read_anchors(anchors, list(package["panel"][model.order[:k]]), bounds(model, k))
    if model.dual and rna is None:
        raise ValueError("Dual inference requires --rna")
    if model.dual:
        rna = np.load(rna, allow_pickle=False)
    q, p = single(package, name, k, values, rna)
    frame = pd.DataFrame({"compound_id": package["panel"][q], "score": p})
    return frame.sort_values(["score", "compound_id"], ascending=(False, True)).assign(rank=np.arange(1, len(q) + 1))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, type=Path)
    cli.add_argument("--anchors", required=True, type=Path)
    cli.add_argument("--rna", type=Path)
    cli.add_argument("--k", type=int, choices=BUDGETS, default=5)
    cli.add_argument("--arm", choices=(ANCHOR, DUAL), default=ANCHOR)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    try:
        table = infer(args.model, args.anchors, args.k, args.arm, args.rna)
    except (ValueError, KeyError, FileNotFoundError) as error:
        cli.exit(2, f"{error}\n")
    table.to_csv(args.out, index=False, sep="\t", float_format="%.17g")


if __name__ == "__main__":
    main()
