"""Run source-only kernel searches and publish both stages of selection as sealed evidence."""
import argparse
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from followup_h2.common import panel_hash, parquet_lock, sha256
from nextgen_h2.assets import KEY
from nextgen_h2.metrics import ScoreBlock, correlation
from nextgen_h2.run import META, read_json
from vc_build.external import native_sensitivity
from vc_build.io import publish_json

from .data import PLAN, cpu_threads, internal_only, registry, source_context, source_rows, train_hash, utc_now
from .model import (
    ANCHORS, BASES, BLENDS, GRID, GRID_PARTS, PENALTIES, SHRINKS, Candidate, Kernel, anchor_key, settings,
)


def split_contract(preflight):
    path = preflight / "e4_split_manifest.json"
    note = read_json(path)
    if (note["plan_sha256"] != sha256(PLAN) or note["source_file_sha256"] != train_hash()
            or note["response_columns_read"] or note["external_cohort_access"] or note["below_minimum_lincs"]
            or note["support_sha256"] != sha256(preflight / "e4_step0_support.tsv")):
        raise ValueError("Unsealed or infeasible Step 0 registration")
    return note


def ordered_roles(manifest):
    return sorted({r for folds in manifest["selections"].values() for r in folds})


def role_data(role, reg, *, inner):
    fit = source_rows(role["fit_parents"], role["fit_screens"], labels=True)
    query = source_rows(role["query_parents"], role["query_screens"], labels=inner)
    if panel_hash(fit) != role["fit_roster_sha256"] or panel_hash(query) != role["query_roster_sha256"]:
        raise ValueError("Role input differs from outcome-blind registration")
    ids = pd.Index(sorted(set(fit[KEY]) | set(query[KEY])))
    cells = pd.Index(sorted(set(fit.cell_id) | set(query.cell_id)))
    return source_context(reg, cells, ids), fit, query


class InnerMetrics:
    def __init__(self, query, groups, mask=None):
        self.data = query.reset_index(drop=True)
        self.blocks = []
        for screen, block in self.data.groupby("source_id"):
            idx = block.index.to_numpy()
            if mask is not None:
                idx = idx[mask[idx]]
            if len(idx):
                layout = ScoreBlock(self.data, idx, "inner", 0, screen, groups, False)
                self.blocks.append((screen, idx, layout))

    def rank(self, linear):
        for screen, idx, block in self.blocks:
            eligible = block.drug_eligible
            if not eligible.any():
                raise ValueError(f"No rank-eligible LINCS parent in {screen}")
            pred = block.wide(linear[idx])
            values = correlation(block.truth_rank, rankdata(pred, axis=0, nan_policy="omit"))
            yield screen, float(values[eligible].mean()), int(eligible.sum())

    def ndcg(self, scores):
        for screen, idx, block in self.blocks:
            if not block.cell_eligible.any():
                raise ValueError(f"No Top-5-eligible cell in {screen}")
            pred = block.wide(scores[idx])
            top = np.argsort(-np.where(np.isfinite(pred), pred, -np.inf), axis=1, kind="stable")[:, :5]
            dcg = np.take_along_axis(block.relevance, top, axis=1) @ block.discount[:top.shape[1]]
            valid = block.cell_eligible
            yield screen, float((dcg[valid] / block.ideal[valid]).mean()), int(valid.sum())


def stage1(preflight, out, index):
    split = split_contract(preflight)
    roles = ordered_roles(split)
    position, part = divmod(index, len(GRID_PARTS))
    name = roles[position]
    reg = registry()
    data, fit, query = role_data(split["roles"][name], reg, inner=True)
    engine = Kernel(data, fit, query[META], reg, name)
    held = query[META].assign(truth=native_sensitivity(query).to_numpy())
    group = GRID_PARTS[part]
    records = []
    for point in GRID:
        if (point.base, point.shape) != group:
            continue
        personal, _, available, steps, err = engine.solve_personal(settings(name, point))
        if not records:
            evaluator = InnerMetrics(held, reg.scaffold_block, available[engine.selector[1]])
        for screen, value, n in evaluator.rank(personal[engine.selector]):
            records.append(asdict(point) | {"role": name, "candidate": point.key,
                                           "screen": screen, "value": value, "n": n,
                                           "steps": steps, "solver_residual": err})
    dest = out / f"{index:03d}"
    dest.mkdir(exist_ok=False, parents=True)
    pd.DataFrame(records).to_csv(dest / "scores.tsv", sep="\t", index=False)
    publish_json(dest / "receipt.json", {
        "index": index, "role": name, "grid_part": list(group), "kernel_candidates": 40,
        "fit_rows": len(fit), "inner_rows": len(query), "fit_calibration": engine.calibration,
        "table_sha256": sha256(dest / "scores.tsv"),
        "split_sha256": sha256(preflight / "e4_split_manifest.json"),
        "external_cohort_access": 0, "role_outside_outer_fit_labels_read": 0,
    })
    print(name, group, len(records), flush=True)


def read_stage1(preflight, stage1_dir, split):
    out = []
    for i in range(len(ordered_roles(split)) * len(GRID_PARTS)):
        folder = stage1_dir / f"{i:03d}"
        proof = read_json(folder / "receipt.json")
        filename = folder / "scores.tsv"
        if (proof["index"] != i or proof["table_sha256"] != sha256(filename)
                or proof["split_sha256"] != sha256(preflight / "e4_split_manifest.json")
                or proof["external_cohort_access"] or proof["role_outside_outer_fit_labels_read"]):
            raise ValueError("Invalid stage 1 shard")
        out.append(pd.read_csv(filename, sep="\t"))
    result = pd.concat(out, ignore_index=True)
    if not result.groupby("role").candidate.nunique().eq(len(GRID)).all():
        raise ValueError("Incomplete Stage 1 exploration")
    return result


def macros(table, candidate="candidate"):
    return table.groupby([candidate, "screen"], sort=False).value.mean().groupby(level=0, sort=False).mean()


def stable_pick(points, scores, slack):
    cutoff = scores.max() - slack
    eligible = [p for p in points if scores[p.key] >= cutoff]
    return max(eligible, key=lambda p: (p.penalty, scores[p.key]))


def choose_kernels(preflight, stage1_dir, output):
    split = split_contract(preflight)
    scores = read_stage1(preflight, stage1_dir, split)
    selected, all_rows = {}, []
    for name, names in split["selections"].items():
        subset = scores.loc[scores.role.isin(names)].copy()
        means = macros(subset)
        raw = max(GRID, key=lambda p: means[p.key])
        stability = stable_pick(GRID, means, .001)
        values = subset[subset.candidate.eq(raw.key)].groupby("screen").value.mean()
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        one_se = stable_pick(GRID, means, se)
        selected[name] = {
            "selected": asdict(stability), "selected_value": float(means[stability.key]),
            "raw_max": asdict(raw), "raw_max_value": float(means[raw.key]), "one_se": asdict(one_se),
            "one_se_value": float(means[one_se.key]), "raw_max_screen_se": se,
        }
        all_rows.append(subset.assign(selection=name, stage=1, tier="screen",
                                      chosen=subset.candidate.eq(stability.key)))
        all_rows.append(pd.DataFrame([
            asdict(p) | {"selection": name, "stage": 1, "tier": "macro", "candidate": p.key,
                         "value": means[p.key], "chosen": p == stability} for p in GRID]))
    output.mkdir(exist_ok=False, parents=True)
    pd.concat(all_rows, ignore_index=True).to_csv(output / "grid.tsv", sep="\t", index=False)
    publish_json(output / "kernels.json", {
        "groups": selected, "external_cohort_access": 0, "created_at_utc": utc_now(),
        "split_sha256": sha256(preflight / "e4_split_manifest.json"),
        "grid_sha256": sha256(output / "grid.tsv"),
    })


def stage2(preflight, kernels, out, index):
    split = split_contract(preflight)
    name = ordered_roles(split)[index]
    group = name.rsplit("/", 1)[0]
    locked = read_json(kernels / "kernels.json")
    assert locked["split_sha256"] == sha256(preflight / "e4_split_manifest.json")
    choice = Candidate(**locked["groups"][group]["selected"])
    reg = registry()
    data, fit, query = role_data(split["roles"][name], reg, inner=True)
    engine = Kernel(data, fit, query[META], reg, name)
    personal, gram, supported, _, _ = engine.solve_personal(settings(name, choice))
    anchors = engine.evaluate_anchors(gram, supported)
    for key, offset in anchors.items():
        query[key] = offset[engine.selector[1]]
    query["residual"] = personal[engine.selector]
    query["warm"] = engine.warm[engine.selector[1]]
    query["truth"] = native_sensitivity(query).to_numpy()
    dest = out / f"{index:02d}"
    dest.mkdir(exist_ok=False, parents=True)
    file = dest / "inner.parquet"
    query.drop(columns=["endpoint_value", "endpoint_direction"]).to_parquet(file, index=False)
    publish_json(dest / "receipt.json", {
        "name": name, "grand_mean": float(engine.target_mean.mean()), "rows": len(query),
        "selection_role": group, "kernel": asdict(choice), "panel_sha256": sha256(file),
        "kernel_selection_sha256": sha256(kernels / "kernels.json"), "external_cohort_access": 0,
    })
    print(name, len(query), flush=True)


def anchor_scores(evaluators, name, calculate):
    out = []
    for role, scorer, grand in evaluators:
        values = calculate(scorer.data, grand)
        for screen, value, n in scorer.ndcg(values):
            out.append({"role": role, "screen": screen, "value": value, "n": n, "candidate": name})
    return out


def choose_anchors(preflight, kernels, stage2_dir, out):
    split, reg = split_contract(preflight), registry()
    file = kernels / "kernels.json"
    locked = read_json(file)
    assert locked["grid_sha256"] == sha256(kernels / "grid.tsv")
    assert locked["external_cohort_access"] == 0
    stage1_grid = pd.read_csv(kernels / "grid.tsv", sep="\t")
    roles = ordered_roles(split)
    grid, choices, evidence = [], {}, {}
    for group, names in split["selections"].items():
        evaluators = []
        for name in names:
            folder = stage2_dir / f"{roles.index(name):02d}"
            proof = read_json(folder / "receipt.json")
            assert proof["name"] == name and proof["selection_role"] == group
            assert proof["kernel_selection_sha256"] == sha256(file) and not proof["external_cohort_access"]
            data = parquet_lock(folder / "inner.parquet", proof["panel_sha256"])
            assert panel_hash(data) == split["roles"][name]["query_roster_sha256"]
            evidence[name] = sha256(folder / "receipt.json")
            evaluators.append((name, InnerMetrics(data, reg.scaffold_block), proof["grand_mean"]))

        rows = []
        simple = [anchor_key(*spec) for spec in ANCHORS]
        configs = {key: {"family": f, "parameter": p} for key, (f, p) in zip(simple, ANCHORS)}
        for key in simple:
            rows.extend(anchor_scores(evaluators, key, lambda d, _: d.residual + d[key]))
        basic_means = macros(pd.DataFrame(rows))
        chem = max([k for k in simple if k.startswith("chem")], key=basic_means.get)
        lincs = max([k for k in simple if k.startswith("lincs")], key=basic_means.get)

        def blend(alpha, gamma, d, grand):
            raw = (1. - alpha) * d[chem] + alpha * d[lincs]
            return d.residual + np.where(d.warm, d["chem_top5/5"], grand + gamma * (raw - grand))

        def mix_config(family, alpha, gamma):
            return {"family": family, "alpha": alpha, "gamma": gamma,
                    "chem": list(ANCHORS[simple.index(chem)]), "lincs": list(ANCHORS[simple.index(lincs)])}

        blend_keys = []
        for a in BLENDS:
            key = f"convex_blend/{a:g}"
            configs[key] = mix_config("convex_blend", a, 1.)
            blend_keys.append(key)
            rows.extend(anchor_scores(evaluators, key, lambda d, g: blend(a, 1., d, g)))

        subgrid = []
        for g in SHRINKS:
            key = f"shrinkage/gamma={g:g}"
            configs[key] = mix_config("shrinkage", .5, g)
            subgrid.extend(anchor_scores(evaluators, key, lambda d, mean: blend(.5, g, d, mean)))
        gamma_means = macros(pd.DataFrame(subgrid))
        shrinkage = max(configs.keys() & set(gamma_means.index), key=lambda k: (gamma_means[k], float(k.split("=")[1])))
        configs["shrinkage"] = configs[shrinkage]
        rows.extend([r | {"candidate": "shrinkage"} for r in subgrid if r["candidate"] == shrinkage])

        table = pd.DataFrame(rows)
        mean = macros(table)
        candidates = simple + blend_keys + ["shrinkage"]
        assert len(candidates) == 16 and set(candidates) == set(mean.index)
        winner = max(candidates, key=mean.get)
        choices[group] = locked["groups"][group] | {
            "intercept": configs[winner], "intercept_candidate": winner, "intercept_ndcg_at_5": float(mean[winner]),
            "best_chemistry": chem, "best_lincs": lincs, "shrinkage_gamma_scores": gamma_means.to_dict(),
            "expanded_kernel": asdict(settings(group, Candidate(**locked["groups"][group]["selected"]))),
        }
        grid.append(table.assign(selection=group, stage=2, tier="screen", chosen=table.candidate.eq(winner)))
        grid.append(pd.DataFrame([{"selection": group, "stage": 2, "tier": "macro",
                                   "candidate": k, "value": mean[k], "chosen": k == winner} for k in candidates]))
        grid.append(pd.DataFrame(subgrid).assign(selection=group, stage=2, tier="shrinkage_subgrid", chosen=False))
        grid.append(pd.DataFrame(anchor_scores(evaluators, "zero_intercept", lambda d, _: d.residual))
                    .assign(selection=group, stage=2, tier="diagnostic", chosen=False))
    output_grid = pd.concat([stage1_grid, *grid], ignore_index=True)
    out.mkdir(exist_ok=False, parents=True)
    output_grid.to_csv(out / "E4_INNER_GRID.tsv", sep="\t", index=False)
    publish_json(out / "e4_selection_receipt.json", {
        "locked_at_utc": utc_now(), "external_cohort_access": 0, "external_score_access": 0,
        "plan_sha256": sha256(PLAN), "split_sha256": sha256(preflight / "e4_split_manifest.json"),
        "grid_sha256": sha256(out / "E4_INNER_GRID.tsv"), "stage1_selection_sha256": sha256(file),
        "stage2_role_receipts": evidence, "kernel_candidates_per_method": 160,
        "anchor_strategies_per_method": 16, "policy": choices,
    })
    print("sealed policies", len(choices), "at", utc_now(), flush=True)


if __name__ == "__main__":
    cpu_threads()
    internal_only()
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("stage1", "kernels", "stage2", "seal"))
    cli.add_argument("preflight", type=Path)
    cli.add_argument("paths", type=Path, nargs="+")
    args = cli.parse_args()
    if args.phase == "stage1":
        stage1(args.preflight, *args.paths, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.phase == "stage2":
        stage2(args.preflight, *args.paths, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.phase == "kernels":
        choose_kernels(args.preflight, *args.paths)
    else:
        choose_anchors(args.preflight, *args.paths)
