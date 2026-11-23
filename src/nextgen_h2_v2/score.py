"""Read each held native response exactly once after all prediction and replay locks."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.metrics import CELL, DRUG, Evaluator, contrast, stack
from nextgen_h2.run import META, read_json, response, subfold
from nextgen_h2.score import adult_licob, external_labels, summary
from vc_build.external import native_sensitivity
from vc_build.io import publish_json, sha256

from .assets import V1_ASSETS, V1_RUN, registries
from .freeze import ARMS, V1_SHA
from .residual_run import STAGE

ASSAY = ARMS[:8]
STRUCTURE = ARMS[8:]
LADDER = [("H2-v1", "C1"), ("H2-R1", "H2-v1"), ("H2-R2a", "H2-R1"),
          ("H2-R2b", "H2-R2a"), ("H2-R3-GBDT", "H2-R2b"), ("H2-R3-Neural", "H2-R2b"),
          ("H2-v2", "H2-v1"), ("H2-v2", "C1")]
STATS = {"drug": DRUG, "cell": CELL}


def validate(work, decision):
    selected = sha256(decision / "champion.json")
    selection = read_json(decision / "receipt.json")
    if (selection["selection_sha256"] != selected or selection["query_outcomes_opened"]
            or selection["strict_stage"] != STAGE):
        raise ValueError("H2-v2 selection mismatch")
    cli_path = work.parent / "portable-cli" / "receipt.json"
    cli = read_json(cli_path)
    if (cli["fit_parent_member"] or not cli["structural_max_abs_diff"] < 1e-6
            or not cli["assisted_max_abs_diff"] < 1e-6):
        raise ValueError("Verified inference on a novel exact parent is required before scoring")
    registry, _ = registries()
    data, proof = [], []
    for part in range(4):
        base = work / "forecasts" / str(part)
        note = read_json(base / "receipt.json")
        replay = read_json(base / "replay.json")
        f = parquet_lock(base / "forecast.parquet", note["files"]["forecast.parquet"])
        old = parquet_lock(V1_RUN / "forecasts" / str(part) / "predictions.parquet", V1_SHA[part])
        if (note["part"] != part or note["query_outcomes_loaded"] != 0
                or note["selection_sha256"] != selected or note["v1_sha256"] != V1_SHA[part]
                or not f[META].equals(old[META])
                or not np.array_equal(f.C1, old.C1) or not np.array_equal(f["H2-v1"], old.champion)
                or not np.isfinite(f[list(ARMS)]).all().all()):
            raise ValueError(f"Forecast registration or C1/H2-v1 replay changed in partition {part}")
        if not np.array_equal(f["Structure-C1"], f.C1):
            raise ValueError("Structure-only baseline changed")
        if (replay["part"] != part or replay["forecast_sha256"] != note["files"]["forecast.parquet"]
                or replay["network_allowed"] or not replay["max_abs_diff"] < 1e-6
                or replay["RNA_fallback_max_diff"] != 0. or replay["structure_fallback_max_diff"] != 0.):
            raise ValueError("Offline replay must pass before outcomes are opened")
        if sha256(base / "bundle.pkl") != note["files"]["bundle.pkl"]:
            raise ValueError("Checked prediction package differs")
        if part != 3 and not f[META].equals(response(subfold(registry, part)[1], False)):
            raise ValueError("Frozen outer scaffold query differs")
        if part == 3 and cli["package_sha256"] != note["files"]["portable.pkl"]:
            raise ValueError("Verified novel SMILES CLI package differs")
        proof.append({"forecast_receipt_sha256": sha256(base / "receipt.json"),
                      "offline_receipt": replay, "v1_sha256": V1_SHA[part],
                      "frozen_forecast_sha256": note["files"]["forecast.parquet"]})
        data.append(f)
    if sum(len(x) for x in data[:3]) != 670830:
        raise ValueError("Outer response denominator differs from v1")
    return data, proof, registry, selected, sha256(cli_path)


def scopes_by_track(scope):
    return (("assisted/lincs_observed", ASSAY), ("assisted/mixed_deployment", ASSAY),
            ("assisted/lincs_unobserved", ASSAY), ("structure/all_masked", STRUCTURE),
            ("structure/462_support" if scope == "DRP" else "structure/lincs_unobserved", STRUCTURE))


def scopes(frame, groups, fold, name, interaction=True, *, source_panel=False):
    frame = frame.reset_index(drop=True)
    ev = Evaluator(frame, groups, fold, scope=name, interaction=interaction)
    measured = frame.lincs_observed.astype(bool)
    selections = (measured, None, ~measured, None, ~measured)
    results = []
    for (mode, arms), use in zip(scopes_by_track("DRP" if source_panel else name), selections):
        for arm in arms:
            results.extend(ev.score(arm, frame[arm].to_numpy(), use, support=mode))
    return results


def contrasts(d, c):
    rows = []
    for typ, frame in (("drug", d), ("cell", c)):
        for (scope, support), block in frame.groupby(["scope", "support"], sort=True):
            axes = set(block.arm.unique())
            trials = LADDER if support.startswith("assisted/") else [("H2-StructOnly", "Structure-C1")]
            for candidate, reference in trials:
                if not {candidate, reference} <= axes:
                    continue
                for metric in STATS[typ]:
                    if metric not in block:
                        continue
                    delta, lo, hi, clusters = contrast(block, candidate, reference, metric)
                    rows.append({"scope": scope, "support": support, "axis": typ, "candidate": candidate,
                                 "reference": reference, "metric": metric, "delta": delta,
                                 "lo95": lo, "hi95": hi, "clusters": clusters})
    return pd.DataFrame(rows)


def gate(effects):
    def value(scope, support, candidate, reference, metric):
        match = effects[(effects.scope == scope) & (effects.support == support) &
                        (effects.candidate == candidate) & (effects.reference == reference) &
                        (effects.metric == metric)]
        if len(match) != 1:
            raise ValueError(f"Missing required decision contrast for {scope}/{support}/{metric}")
        return match.iloc[0]

    criteria = [
        ("assisted/lincs_observed", "spearman", .01),
        ("assisted/lincs_observed", "interaction_spearman", .02),
        ("assisted/mixed_deployment", "ndcg_at_5", .01),
    ]
    results = [value("DRP", support, "H2-v2", "H2-v1", name) for support, name, minimum in criteria]
    success = [p.delta >= level and p.lo95 > 0 for p, (_, _, level) in zip(results, criteria)]
    gd = [value("GDSC1", "assisted/lincs_observed", "H2-v2", "H2-v1", "spearman"),
          value("GDSC1", "assisted/mixed_deployment", "H2-v2", "H2-v1", "spearman")]
    virtual = value("DRP", "structure/462_support", "H2-StructOnly", "Structure-C1", "spearman")
    return {"upgrade_pass": bool(any(success) and all(x.lo95 >= -.005 for x in [*results, *gd])),
            "gds_noninferiority": all(x.lo95 >= -.005 for x in gd),
            "criterion_passes": dict(zip((m for _, m, _ in criteria), [bool(x) for x in success])),
            "structure_462_pass": bool(virtual.delta >= .01 and virtual.lo95 > 0)}


def tissue_summary(tissue_drug, tissue_cell, tissue_support):
    tissue = pd.concat([summary(tissue_drug, DRUG).assign(axis="drug"), summary(tissue_cell, CELL).assign(axis="cell")])
    reported = set(tissue.loc[tissue.screen.eq("macro"), ["scope", "support", "arm", "axis"]].itertuples(
        index=False, name=None))
    empty = []
    for label in tissue_support:
        for support, arms in scopes_by_track("DRP"):
            for arm in arms:
                for axis in STATS:
                    if (label, support, arm, axis) not in reported:
                        empty.append({"scope": label, "support": support, "arm": arm, "axis": axis,
                                      "screen": "macro", "fold": "all", "units": 0})
    return pd.concat([tissue, pd.DataFrame(empty)], ignore_index=True)


def run(work, chosen):
    parts, sealed, registry, digest, cli_sha = validate(work, chosen)
    asset = read_json(V1_ASSETS / "data_receipt.json")
    cells = parquet_lock(V1_ASSETS / "cells.parquet", asset["files"]["cells.parquet"])
    source = pd.concat([x[[KEY, "cell_id", "lincs_observed"]] for x in parts[:3]], ignore_index=True)
    source["tissue"] = source.cell_id.map(cells.tissue)
    if source.tissue.isna().any() or source.tissue.nunique() != 28:
        raise ValueError("The 28 registered tissue lineages must be present before scoring")
    tissue_support = {label: support_counts(block) for label, block in source.groupby("tissue", sort=True)}
    current = work / "score"
    current.mkdir(exist_ok=False)
    publish_json(current / "outcome_access.json", {"score_passes": 1, "proofs": sealed,
                                                    "portable_cli_receipt_sha256": cli_sha})
    groups = registry.scaffold_block
    assessments, tissues = [], []
    for idx, predictions in enumerate(parts[:3]):
        targets = response(subfold(registry, idx)[1], True)
        if not targets.row_id.equals(predictions.row_id):
            raise ValueError("Native outcome keys differ")
        frame = predictions.assign(truth=native_sensitivity(targets), tissue=predictions.cell_id.map(cells.tissue))
        assessments.extend(scopes(frame, groups, idx, "DRP"))
        for label, block in frame.groupby("tissue", sort=True):
            tissues.extend(scopes(block, groups, idx, label, interaction=False, source_panel=True))
        print(f"Scored DRP outer {idx}", flush=True)
    y, licob, ext_chem = external_labels(V1_ASSETS, asset)
    ext = parts[3].merge(y, on="row_id", validate="one_to_one", how="left")
    if set(y.row_id) != set(ext.row_id) or not np.isfinite(ext.truth).all():
        raise ValueError("External outcome support differs from the v1 sealed registration")
    external_support = {}
    for scope, block in ext.groupby("source_id", sort=True):
        if scope == "LICOB":
            block = adult_licob(block, licob)
            novel = block[~block.in_source & block.remaining_unspecified_stereo.eq(0)]
            assert novel[KEY].nunique() == 18 and block[KEY].nunique() == 76 and block.cell_id.nunique() == 38
            assessments.extend(scopes(novel, ext_chem, 0, "LICOB18"))
            external_support["LICOB18"] = support_counts(novel)
        assessments.extend(scopes(block, ext_chem, 0, scope))
        external_support[scope] = support_counts(block)
        print(f"Scored {scope}", flush=True)
    d, c = stack(assessments)
    td, tc = stack(tissues)
    for name, frame in (("drug", d), ("cell", c), ("tissue_drug", td), ("tissue_cell", tc)):
        frame.to_parquet(current / f"unit_{name}.parquet", index=False)
    scores = pd.concat([summary(d, DRUG).assign(axis="drug"), summary(c, CELL).assign(axis="cell")])
    tissue = tissue_summary(td, tc, tissue_support)
    paired = contrasts(d, c)
    ladder = paired.merge(scores[scores.screen.eq("macro")].drop(columns=["screen", "fold"]),
                          left_on=["scope", "support", "axis", "candidate"],
                          right_on=["scope", "support", "axis", "arm"], validate="many_to_one")
    for name, frame in (("H2V2_METRICS", scores), ("H2V2_PAIRED_95CI", paired),
                        ("H2V2_ABLATION_LADDER", ladder), ("H2V2_PANCANCER_BY_TISSUE", tissue)):
        frame.to_csv(current / (name + ".tsv"), sep="\t", index=False, na_rep="NA")
    publish_json(current / "score_receipt.json", {"score_passes": 1, "selection_sha256": digest,
        "source_rows": sum(len(x) for x in parts[:3]), "source_parents": len(registry),
        "source_support": support_counts(source), "tissue_support": tissue_support,
        "tissues": list(tissue_support), "external_rows": ext.groupby("source_id").size().to_dict(),
        "external_support": external_support, "portable_cli_receipt_sha256": cli_sha,
        "result": gate(paired), "forecasts": sealed,
        "files": {f.name: sha256(f) for f in sorted(current.iterdir())}})


def support_counts(rows):
    return {"rows": len(rows), "parents": rows[KEY].nunique(), "cells": rows.cell_id.nunique(),
            "LINCS_parents": rows.loc[rows.lincs_observed, KEY].nunique()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("decision", type=Path)
    a = parser.parse_args()
    run(a.work, a.decision)
