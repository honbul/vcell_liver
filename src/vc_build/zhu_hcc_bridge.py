"""Adapt a 2D molecular predictor on already evaluated public Zhu organoids."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .culture_consistency import kernel_between
from .evaluate import join_observations
from .expanded_panel_cv import PENALTY, expanded_targets
from .external import KEY, SOURCE, _validate
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_icc_bridge import affine_map, centered_chemistry, corrections
from .licob_node_recovery import joined_outcomes
from .limore_corrected import OBSERVATIONS, shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import EXTERNAL_TRACKS, external_inputs, external_reference
from .response_components import centered_kernels, orthogonal_parts
from .structured_kernel import drug_kernels, fit_cell_kernels, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs, panel_targets
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash
from .zhu_hcc_confirmation import ASSETS, EXPANDED, SCOPE, ZHU, observed, read_rosters
from .zhu_dose_transfer import FORECAST as ZHU_PREDICTIONS

DEV_SCORE = "d4201de70dca3988bb4ebcf92457c23bfe75f1b530d104bf30733b461c657b11"
BASE, OFFSET, BOTH, AUC, DOSE, NULL = (
    "matched_2d", "zhu_auc_offset", "zhu_dual_residual", "zhu_auc_residual", "zhu_10uM_residual", "zhu_donor_shuffle",
)
ARMS = (BASE, OFFSET, BOTH, AUC, DOSE, NULL)
PEERS = ("drug_mean", "drug_ridge", "structured", "cellhit", "codeae_inductive",
         "velodrome_inductive", "v0plus_inductive")
SHIFTED = tuple("same_offset_" + m for m in PEERS)
WEIGHT = .5


def prepare(args):
    require_hash(args.zhu_score, DEV_SCORE)
    past = json.loads(args.zhu_score.read_text())
    require_hash(args.zhu_labels, ASSETS["zhu-s4.xlsx"])
    if past["forecast_sha256"] != ZHU_PREDICTIONS or past["label_sha256"] != ASSETS["zhu-s4.xlsx"]:
        raise ValueError("The Zhu development role needs the completed earlier score")
    rosters = read_rosters(args.zhu, expression=True)
    with closing(load_workbook(args.zhu_labels, read_only=True, data_only=True)) as wb:
        panels = {key: observed(wb, key, frame) for key, frame in rosters.items()}
    with new_output(args.out) as path:
        hashes = {}
        for label, panel in panels.items():
            if len(panel) != len(rosters[label]) or panel.duplicated(["cell_id", "compound_id"]).any():
                raise ValueError("The development panel has missing or duplicate labels")
            out = path / (label + ".parquet")
            panel[[*KEY, "endpoint_value"]].to_parquet(out, index=False)
            hashes[label] = sha256(out)
        publish_json(path / "receipt.json", {
            "cohort": ZHU, "earlier_score": DEV_SCORE, "labels": ASSETS["zhu-s4.xlsx"], "files": hashes,
            "endpoint_rows": {label: len(panel) for label, panel in panels.items()},
            "overlapping_donors": len(set(panels["auc"].lineage_group) & set(panels["single_dose"].lineage_group)),
        })


def development(args):
    receipt = args.dev / "receipt.json"
    stamp = json.loads(receipt.read_text())
    if (stamp["cohort"], stamp["earlier_score"], stamp["labels"]) != (ZHU, DEV_SCORE, ASSETS["zhu-s4.xlsx"]):
        raise ValueError("Unexpected HCC organoid development role")
    rosters = read_rosters(args.zhu, expression=True)
    records = {}
    for label in SCOPE:
        path = args.dev / (label + ".parquet")
        require_hash(path, stamp["files"][label])
        rows = pd.read_parquet(path)
        if not rows[KEY].equals(rosters[label][KEY]) or not np.isfinite(rows.endpoint_value).all():
            raise ValueError("The fit rows disagree with the fixed public endpoint")
        if label == "single_dose":
            # Zhu reports inhibition on a percent scale. The affine bounds assume fractional responses.
            rows = rows.assign(endpoint_value=rows.endpoint_value / 100.)
        records[label] = rows.pivot(index="cell_id", columns="compound_id", values="endpoint_value")
    a, b = records["auc"], records["single_dose"]
    if not set(a.index) <= set(b.index) or not set(a.columns) <= set(b.columns):
        raise ValueError("The endpoint-overlap contract changed")
    rna = pd.read_parquet(args.zhu / "rna.parquet")
    return rna.loc[records["single_dose"].index], records, sha256(receipt)


def extend(fit, args, observed):
    receipt = args.expanded / "receipt.json"
    require_hash(receipt, EXPANDED)
    curve = args.expanded / "train.parquet"
    require_hash(curve, json.loads(receipt.read_text())["files"]["train.parquet"])
    additional = set(observed["single_dose"].columns) - set(fit.compound_id)
    extra = pd.read_parquet(curve, filters=[("compound_id", "in", sorted(additional))])
    if additional != set(extra.compound_id) or not set(extra.cell_id) <= set(fit.cell_id):
        raise ValueError("Expanded exact parents must use the same registered cell pool")
    rows = pd.concat([fit, extra], ignore_index=True)
    if rows.duplicated(["cell_id", "compound_id", *SOURCE]).any():
        raise ValueError("Duplicate source response identities")
    return rows, extra


def rotated(shared, others):
    rng = np.random.default_rng(34861)
    result = {}
    for members in (shared, others):
        ordered = rng.permutation(members)
        result.update(zip(ordered, np.roll(ordered, 1)))
    return result


def fit_bridge(fit, query, rna, zhu_rna, assays, edges, targets, auxiliary):
    _validate(fit, query)
    if set(zhu_rna.index) & (set(fit.cell_id) | set(query.cell_id)):
        raise ValueError("Zhu and scoring compartments must use separate registered cultures")
    device = device_cuda()
    genes = rna.columns.intersection(zhu_rna.columns).sort_values()
    source = rna.loc[sorted(fit.cell_id.unique()), genes]
    held = rna.loc[sorted(query.cell_id.unique()), genes]
    zhu = zhu_rna[genes]
    basis, gram = fit_cell_kernels(source, edges, auxiliary, device=device)
    _, mean, aligned, _ = panel_targets(fit)
    compounds, kernels = drug_kernels(aligned, targets, device=device)
    chem = kernels["structured"]
    w, y = training_grid(aligned, source.index, compounds, device=device)
    coefficient, iterations, solver_error = solve_residuals(gram["full"], chem, w, y, PENALTY)
    coef = coefficient @ chem
    intercept = mean.loc[compounds].to_numpy()
    base = (basis.cross(held)["full"] @ coef).cpu().numpy() + intercept
    pdo = pd.DataFrame((basis.cross(zhu)["full"] @ coef).cpu().numpy() + intercept,
                       index=zhu.index, columns=compounds)
    diag = {"core_solver_steps": iterations, "core_solver_error": solver_error, "genes": len(genes)}
    shuffle = rotated(assays["auc"].index.to_numpy(), zhu.index.difference(assays["auc"].index).to_numpy())
    vectors = basis.vectors(held)
    changes, controls, offsets = {}, {}, {}
    for label, observed_matrix in assays.items():
        row = observed_matrix.index
        names = observed_matrix.columns
        estimate = pdo.loc[row, names].to_numpy()
        actual, audit = affine_map(observed_matrix.to_numpy(), estimate)
        _, shift, residual = orthogonal_parts(actual - estimate, np.ones_like(estimate))
        reference = basis.vectors(zhu.loc[row])
        train, cross = centered_kernels(kernel_between(basis, reference, reference),
                                        kernel_between(basis, vectors, reference))
        j = compounds.get_indexer(names)
        drug = centered_chemistry(chem, j)
        fit_delta, details = corrections(train, cross, drug, residual)
        rotation = row.get_indexer([shuffle[name] for name in row])
        if (rotation < 0).any() or (rotation == np.arange(len(row))).any():
            raise ValueError("The negative control must preserve shared assay donor identity and derange all donors")
        null, null_audit = corrections(train, cross, drug, residual[rotation])
        changes[label] = pd.DataFrame(fit_delta, index=held.index, columns=names)
        controls[label] = pd.DataFrame(null, index=held.index, columns=names)
        offsets[label] = pd.Series(shift, index=names)
        diag[label] = {"donors": len(row), "compounds": len(names), "affine": audit,
                       "residual_fit": details, "shuffle_fit": null_audit,
                       "residual_train_rms": float(np.sqrt(np.square(residual).mean()))}

    offset = WEIGHT * offsets["auc"].reindex(compounds, fill_value=0)

    def dual(parts):
        single = parts["single_dose"]
        count = 1 + single.columns.isin(parts["auc"].columns).astype(int)
        return single.add(parts["auc"].reindex(columns=single.columns, fill_value=0)).div(count, axis=1)

    locus = held.index.get_indexer(query.cell_id), compounds.get_indexer(query.compound_id)
    if min(locus[0].min(), locus[1].min()) < 0:
        raise ValueError("Unmatched recipient response identities")
    delta = {BOTH: dual(changes), AUC: changes["auc"], DOSE: changes["single_dose"], NULL: dual(controls)}
    anchored = base + offset.to_numpy()
    values = {BASE: base, OFFSET: anchored}
    for arm, change in delta.items():
        values[arm] = anchored + WEIGHT * change.reindex(columns=compounds, fill_value=0).to_numpy()
    output = pd.concat([query[KEY].assign(method=name, predicted_value=value[locus])
                        for name, value in values.items()], ignore_index=True)
    diag["joint_delta_rms"] = float(np.sqrt(np.square((values[BOTH]-anchored)[locus]).mean()))
    return output, offset, diag


def predict(args):
    zhu, profiles, dev_sha = development(args)
    blocks, queries, priors, hashes = [], {}, {}, {}
    for mode in EXTERNAL_TRACKS:
        core, query, rna, receipt = external_inputs(args, mode)
        prior, ref = external_reference(args, mode, query)
        queries[mode], priors[mode], hashes[mode] = query, prior, {"cohort": receipt, "reference": ref}
        blocks.append(rna.loc[sorted(query.cell_id.unique())])
    fit, extra = extend(core, args, profiles)
    genes = sorted(set.intersection(*(set(x.columns) for x in [*blocks, zhu, rna])))
    train = rna.loc[sorted(fit.cell_id.unique()), genes]
    expression = pd.concat([train, *[block[genes] for block in blocks]], verify_integrity=True)
    query = pd.concat([q[KEY] for q in queries.values()], ignore_index=True)
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    targets, _ = expanded_targets(extra, targets, expression.columns)
    auxiliary, _ = load_auxiliary(args.auxiliary)
    rows, offset, diagnostics = fit_bridge(fit, query, expression, zhu, profiles, edges, targets, auxiliary)
    parts = []
    for mode in EXTERNAL_TRACKS:
        q, previous = queries[mode], priors[mode]
        shifted = previous.assign(method="same_offset_" + previous.method,
                                  predicted_value=previous.predicted_value + previous.compound_id.map(offset))
        part = pd.concat([rows[rows.row_id.isin(q.row_id)], previous, shifted], ignore_index=True)
        shared_support(part, q, (*ARMS, *PEERS, *SHIFTED))
        parts.append(part.assign(track=mode, fold=0))
    freeze_table(pd.concat(parts, ignore_index=True), args.out, provenance={
        "zhu_dev_sha256": dev_sha, "zhu_earlier_score": DEV_SCORE, "recipients": hashes,
        "fit_rows": len(fit), "fit_drugs": fit.compound_id.nunique(), "diagnostics": diagnostics,
        "direct_compounds": {name: list(labels.columns) for name, labels in profiles.items()},
        "auc_offsets": offset.to_dict(),
    })


def assess(rows, query):
    controls = (BASE, OFFSET, AUC, DOSE, NULL, *SHIFTED)
    pairs = [(BOTH, b) for b in controls] + [(OFFSET, BASE)]
    report, drug, _ = axis_metrics(rows, pairs)
    bounds = {b: crossed_patient_drug_ci(rows, BOTH, b, repeats=2000) for b in controls}
    rank = {(p["baseline"], p["metric"]): p["ci95"][0] for p in report["paired_contrasts"] if p["candidate"] == BOTH}
    gates = {b: all(bounds[b][k][0] > 0 for k in ("spearman", "pearson"))
             and all(rank[b, k] > -.005 for k in ("spearman", "ndcg_at_5", "top5_overlap")) for b in controls}
    strata = {}
    for name, identifiers in compound_strata(query).items():
        partial = drug[drug.compound_id.isin(identifiers)]
        if not partial.empty:
            stats = partial.groupby("method")[["spearman", "pearson"]].mean()
            strata[name] = {"drugs": partial.compound_id.nunique(),
                            "values": stats.astype(object).where(stats.notna(), None).to_dict("index")}
    return {"metrics": report, "ci95": bounds, "gates": gates, "strata": strata,
            "donors": rows.lineage_group.nunique(), "drugs": rows.compound_id.nunique(), "rows": rows.row_id.nunique()}


def score(args):
    output, seal = verify_forecast(args.forecast)
    metadata = seal["provenance"]
    if metadata["zhu_earlier_score"] != DEV_SCORE or sha256(args.dev / "receipt.json") != metadata["zhu_dev_sha256"]:
        raise ValueError("Expected the complete Zhu bridge freeze before label access")
    direct = set(metadata["direct_compounds"]["single_dose"])
    results = {}
    for mode in EXTERNAL_TRACKS:
        _, query, _, receipt = external_inputs(args, mode)
        if receipt != metadata["recipients"][mode]["cohort"]:
            raise ValueError("Recipient manifest changed")
        rows = output[output.track.eq(mode)]
        shared_support(rows, query, (*ARMS, *PEERS, *SHIFTED))
        if mode == "LIMORE":
            require_hash(args.limore_labels, OBSERVATIONS)
            joined = join_observations(rows, pd.read_parquet(args.limore_labels))
        else:
            require_hash(args.licob_labels, ASSET_HASHES["supplement"])
            joined = adult_donor_outcomes(joined_outcomes(query, rows, args.licob_labels),
                                         constant_methods=("drug_mean", "same_offset_drug_mean"))
        results[mode] = {"full": assess(joined, query),
                         "direct": assess(joined[joined.compound_id.isin(direct)], query)}
    publish_json(args.out, {"forecast_sha256": seal["sha256"], "cohorts": results,
                            "labels": [OBSERVATIONS, ASSET_HASHES["supplement"]],
                            "joint_pass": all(all(panel["gates"].values()) for r in results.values() for panel in r.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    common = ("dev", "core", "original", "licob")
    for stage, fields in {
        "prepare": ("zhu", "zhu-labels", "zhu-score"),
        "predict": (*common, "zhu", "expanded", "archive", "priors", "hallmark", "targets", "auxiliary"),
        "score": (*common, "forecast", "limore-labels", "licob-labels"),
    }.items():
        p = sub.add_parser(stage)
        for field in ("out", *fields):
            p.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
