"""Run the full E5 kernel grid on source-only inner roles, with shared ridge solves."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import pandas as pd

from followup_h2.common import sha256
from followup_h2_e4.data import cpu_threads, registry, utc_now
from followup_h2_e4.model import GRID_PARTS, PENALTIES, WEIGHTS, Candidate, settings
from followup_h2_e4.selection import InnerMetrics, role_data
from nextgen_h2.run import META, read_json
from vc_build.external import native_sensitivity
from vc_build.io import publish_json

from .kernel import BASELINE, BALANCED, DEBIASED, TASK, TAUS, ScreenKernel, System, solve_shifts
from .registration import read_split, source_only


def all_roles(split):
    names = {r for value in split["selections"].values() for r in value}
    # Start large grids first to reduce the tail of the capped array.
    return sorted(names, key=lambda r: (-split["roles"][r]["fit"]["rows"], r))


def add_scores(rows, evaluator, role, family, tau, cfg, results):
    for p, (pred, iterations, error, method) in results.items():
        theta = Candidate(cfg.base, cfg.shape, cfg.weight, p)
        for screen, value, n in evaluator.rank(pred):
            rows.append(asdict(theta) | {"candidate": theta.key, "role": role, "screen": screen,
                        "family": family, "tau": tau, "value": value, "n": n, "iterations": iterations,
                        "relative_residual": error, "solver": method})


def single_screen_task(engine, params, mode):
    """Reuse one weighted solve for every effective (lambda, tau) on a one-screen fit."""
    gram, avail = engine.geometry(params)
    cross = gram[engine.stop:, :engine.stop]
    mass, value, _, _ = engine.targets_for(TASK)
    assert len(mass) == 1
    system = System(engine.cell, gram[:engine.stop,:engine.stop].contiguous(), mass[0], value[0])
    shifts = {(tau,p):p/(1.+tau) for tau in TAUS for p in PENALTIES}
    global_solutions = solve_shifts(system, sorted(set(shifts.values())))
    for tau in TAUS:
        answers = {}
        for p in PENALTIES:
            coef, n, err, method = global_solutions[shifts[tau,p]]
            pred = engine.project(coef[None]/(1.+tau), cross, avail, tau, TASK, mode, 1.)
            answers[p] = pred, n, err, method
        yield tau, answers


def shard(preflight, out, index):
    source_only()
    split = read_split(preflight)
    roles = all_roles(split)
    i, quadrant = divmod(index, len(GRID_PARTS))
    role = roles[i]
    evidence = split["roles"][role]
    reg = registry()
    data, fit, query = role_data(evidence, reg, inner=True)
    model = ScreenKernel(data, fit, query[META], reg, role)
    mask = model.support("lincs", "landmark_mean")
    x = query[META].assign(truth=native_sensitivity(query).to_numpy())
    ev = InnerMetrics(x, reg.scaffold_block, mask)
    mode = "shared" if evidence["kind"] == "loso" else "in_screen"
    extra_baseline = role.startswith("T2/") and evidence["kind"] == "pooled"
    frames, proofs = [], {}
    base, shape = GRID_PARTS[quadrant]
    directory = out / f"{index:03d}"
    directory.mkdir(exist_ok=True, parents=True)
    locked = sha256(preflight / "split.json")
    if (directory/"receipt.json").exists():
        receipt = read_json(directory/"receipt.json")
        assert receipt["registration_sha256"] == locked and receipt["scores_sha256"] == sha256(directory/"scores.tsv")
        print("Already sealed", index, flush=True)
        return
    for w in WEIGHTS:
        partial = directory / f"w-{w:g}.tsv"
        check = partial.with_suffix(".json")
        if check.exists():
            sealed = read_json(check)
            assert sealed["role"] == role and sealed["registration_sha256"] == locked
            assert sealed["partial_sha256"] == sha256(partial)
            frames.append(pd.read_csv(partial, sep="\t"))
            proofs.update(sealed["fit_transform"])
            continue
        records = []
        cfg = Candidate(base, shape, w, PENALTIES[0])
        params = settings(role, cfg)
        for family in (DEBIASED, BALANCED):
            result, _, _, proof = model.solutions(params, family, 0., PENALTIES)
            add_scores(records, ev, role, family, 0., cfg, result)
            proofs[family] = proof
            print(utc_now(), role, base, shape, "w", w, family, flush=True)
            if family == BALANCED and extra_baseline and fit.source_id.nunique() == 1:
                add_scores(records, ev, role, BASELINE, 0., cfg, result)
        if extra_baseline and fit.source_id.nunique() > 1:
            result, _, _, _ = model.solutions(params, BASELINE, 0., PENALTIES)
            add_scores(records, ev, role, BASELINE, 0., cfg, result)
        if fit.source_id.nunique() == 1:
            for tau, result in single_screen_task(model, params, mode):
                if tau:
                    add_scores(records, ev, role, TASK, tau, cfg, result)
        else:
            for tau in TAUS[1:]:
                result, _, _, proof = model.solutions(params, TASK, tau, PENALTIES, mode=mode)
                add_scores(records, ev, role, TASK, tau, cfg, result)
                proofs[TASK] = proof
                print(utc_now(), role, base, shape, "w", w, "tau", tau, flush=True)
        piece = pd.DataFrame(records)
        piece.to_csv(partial, index=False, sep="\t")
        publish_json(check, {"role":role,"registration_sha256":locked,
                            "partial_sha256":sha256(partial),"fit_transform":proofs})
        frames.append(piece)
        print(utc_now(), role, base, shape, "w", w, "finished", flush=True)
    table = pd.concat(frames, ignore_index=True)
    file = directory / "scores.tsv"
    table.to_csv(file, index=False, sep="\t")
    publish_json(directory / "receipt.json", {
        "registration_sha256": locked, "role": role, "index": index,
        "grid_part": [base, shape], "scores_sha256": sha256(file),
        "row_count": len(table), "fit_rows": len(fit), "inner_rows": len(query),
        "scope": mode, "external_cohort_access": 0, "fit_transform": proofs,
        "shards": table.groupby("family").candidate.nunique().to_dict(),
        "written_at_utc": utc_now(),
    })


if __name__ == "__main__":
    cpu_threads()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preflight", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    shard(args.preflight, args.out, int(os.environ["SLURM_ARRAY_TASK_ID"]))
