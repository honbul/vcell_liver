"""Compare source-only HCC response models with matched, measured DNA inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import rankdata

from . import measured_drivers
from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS
from .external import KEY, _validate
from .hcc_3d_study_transfer import COHORT_HASHES, DONORS_HASH
from .io import freeze_table, publish_json, sha256, verify_forecast
from .limore_corrected import OBSERVATIONS, shared_support
from .v0plus_cv import candidate_forecasts, molecular_inputs, panel_targets, training_inputs
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes, checked_registration, cohort_metadata
from .v0plus_release import load_query
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_sota import verify_upstream
from .yang_hcc_external import SETTINGS, require_hash

LIMORE_ARCHIVE = "fe7f9e656801d63fd1263ec3e94093cad86b049130aa038d4f2e8fdd059b02bb"
STUDIES = {"LIMORE": (50, 68), "LICOB": (34, 53)}
REFERENCES = ("drug_mean", "drug_ridge", "cellhit", "velodrome_inductive", "codeae_inductive", "v0plus_inductive")
SHARDS = 3
PENALTY = 4.0
MEASURED_WEIGHT = 0.10
TARGET_WEIGHT = 0.10
PERMUTATION_SEED = 37819


def data(args, study):
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    train, fit_rna, _ = training_inputs(args.core)
    if study == "LIMORE":
        require_hash(args.limore / "receipt.json", RECEIPTS["original"])
        query, held, _ = load_query(args.limore, train)
    else:
        require_hash(args.licob / "receipt.json", COHORT_HASHES["LICOB"])
        checked_registration(args.licob)
        query = pd.read_parquet(args.licob / "query.parquet")
        held = pd.read_parquet(args.licob / "rna.parquet")
    _validate(train, query)
    models, drugs = STUDIES[study]
    if len(query) != models * drugs or query.cell_id.nunique() != models or query.compound_id.nunique() != drugs:
        raise ValueError(f"Unexpected {study} registered support")
    genes = fit_rna.columns.intersection(held.columns)
    fit_rna = fit_rna.loc[sorted(train.cell_id.unique()), genes]
    held = held.loc[sorted(query.cell_id.unique()), genes]
    return train, query, fit_rna, held


def prepare(args):
    train, limore, _, _ = data(args, "LIMORE")
    _, licob, _, _ = data(args, "LICOB")
    measured_drivers.prepare(args, train, limore, licob)


def normalized_block(fit, held):
    positive = fit.sum()
    support = fit.count()
    keep = positive.ge(5) & (support - positive).ge(5) & support.ge(40)
    if not keep.any():
        raise ValueError("No measured DNA feature has both fit classes")
    columns = sorted(keep[keep].index)
    center = fit[columns].mean()
    spread = fit[columns].std(ddof=0).clip(lower=0.20)
    transformed = [((values[columns] - center) / spread).fillna(0).to_numpy(dtype=np.float32)
                   for values in (fit, held)]
    return (*transformed, columns)


def target_feature(fit, held, drugs, targets, device):
    import torch

    fit_rank, held_rank = ((rankdata(frame.to_numpy(), axis=1) - 0.5) / frame.shape[1]
                           for frame in (fit, held))
    blocks = [np.zeros((len(frame), len(drugs)), dtype=np.float32) for frame in (fit, held)]
    covered = 0
    for j, name in enumerate(drugs):
        genes = fit.columns.intersection(targets.get(name, ()))
        if not len(genes):
            continue
        covered += 1
        indices = fit.columns.get_indexer(genes)
        fit_mean = fit_rank[:, indices].mean(axis=1)
        query_mean = held_rank[:, indices].mean(axis=1)
        scale = max(float(fit_mean.std()), 0.05)
        for block, value in zip(blocks, (fit_mean, query_mean)):
            block[:, j] = (value - fit_mean.mean()) / scale * np.sqrt(TARGET_WEIGHT)
    return *(torch.as_tensor(block, device=device) for block in blocks), covered


def permutation(query, indices):
    ordered = query[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id").loc[indices]
    group = [np.sort(indices.get_indexer(part.index)) for _, part in ordered.groupby("lineage_group")]
    rng = np.random.default_rng(PERMUTATION_SEED)
    shuffled = np.arange(len(indices))
    for size in sorted({len(x) for x in group}):
        peers = [x for x in group if len(x) == size]
        for recipient, reference_index in zip(peers, rng.permutation(len(peers))):
            shuffled[recipient] = peers[reference_index]
    return shuffled


def structured_models(train, query, fit, held, edges, targets, auxiliary, blocks, groups):
    import torch

    from .structured_kernel import KernelGrid, dot_kernel, solve_residuals
    if not torch.cuda.is_available():
        raise RuntimeError("Measured-driver kernels require a G1 CUDA device")
    device = torch.device("cuda")
    _, anchors, rows, _ = panel_targets(train)
    grid = KernelGrid.prepare(rows, fit, held, edges, targets, auxiliary, device=device)
    cell, cross = grid.cell["full"]
    drug = grid.drug["structured"]
    target_fit, target_held, target_covered = target_feature(fit, held, grid.drugs, targets, device)
    measured, diagnostic = {}, {}
    for name, (fit_block, test_block) in blocks.items():
        train_data = fit_block.reindex(fit.index)
        test_data = test_block.reindex(held.index)
        if not test_data.notna().all(axis=None):
            raise ValueError("Registered evaluation must have the measured evidence used by the model")
        a, b, columns = normalized_block(train_data, test_data)
        a, b = (torch.as_tensor(values, device=device) for values in (a, b))
        measured[name] = (MEASURED_WEIGHT * dot_kernel(a, a), MEASURED_WEIGHT * dot_kernel(b, a))
        diagnostic[name] = {"columns": columns, "fit_models": int(train_data.notna().any(axis=1).sum()),
                            "held_models": len(test_data)}
    arms = {"structured": (), "target_expression": (), "measured_coding": ("coding",)}
    if "copy" in blocks:
        arms |= {"measured_copy": ("copy",), "measured_genome": ("coding", "copy")}
    primary = "measured_genome" if "copy" in blocks else "measured_coding"
    shuffled = permutation(groups, grid.cells)
    key = (grid.cells.get_indexer(query.cell_id), grid.drugs.get_indexer(query.compound_id))
    anchor = query.compound_id.map(anchors).to_numpy()
    values, diagnostics = [], {}
    for arm, inputs in arms.items():
        fit_cell = cell + sum(measured[x][0] for x in inputs)
        held_cell = cross + sum(measured[x][1] for x in inputs)
        use_target = arm != "structured"
        coefficients, steps, error = solve_residuals(fit_cell, drug, grid.weights, grid.response, PENALTY,
                                                      per_drug_prior=target_fit if use_target else None)
        diagnostics[arm] = {"iterations": steps, "residual": error}
        extra = target_held * (target_fit * coefficients).sum(dim=0) if use_target else 0.0
        residual = held_cell @ coefficients @ drug + extra
        values.append(query[KEY].assign(method=arm, predicted_value=anchor + residual.cpu().numpy()[key]))
        if arm == primary:
            reindexed = cross + sum(measured[x][1][shuffled] for x in inputs)
            residual = reindexed @ coefficients @ drug + extra
            values.append(query[KEY].assign(method="genotype_permutation",
                                            predicted_value=anchor + residual.cpu().numpy()[key]))
    return pd.concat(values, ignore_index=True), {"target_mapped_training_drugs": target_covered, "DNA": diagnostic,
        "permutation_moved_models": int(np.count_nonzero(shuffled != np.arange(len(shuffled)))), "solver": diagnostics}


def structured(args, train, query, fit, held, edges, targets):
    from .v0plus_neural import load_auxiliary

    measurements, receipt = measured_drivers.read(args.genomics)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    suffix = args.study.lower()
    views = ("coding", "copy") if args.study == "LIMORE" else ("coding",)
    blocks = {name: (measurements[f"source_{name}"], measurements[f"{suffix}_{name}"]) for name in views}
    groups = query
    if args.study == "LICOB":
        require_hash(args.patient_audit, DONORS_HASH)
        patient_map = json.loads(args.patient_audit.read_text())["models"]
        donors = pd.DataFrame(patient_map).set_index("cell_id").donor_id
        groups = query.assign(lineage_group=query.cell_id.map(donors))
    prediction, report = structured_models(train, query, fit, held, edges, targets, auxiliary, blocks, groups)
    return prediction, report | {"genomics_sha256": receipt, "auxiliary": aux_receipt}


def expected_components(study):
    return ("structured",) if study == "LIMORE" else (
        "structured", "linear", "velodrome_inductive", "codeae_inductive", "v0plus_inductive",
        *(f"cellhit-{i}" for i in range(SHARDS)),
    )


def predict(args):
    require_hash(args.settings, SETTINGS)
    train, query, fit, held = data(args, args.study)
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    extra = {}
    if args.component == "structured":
        predictions, extra = structured(args, train, query, fit, held, edges, targets)
    else:
        predictions, query = fit_reference(args, train, query, fit, held, edges, targets)
    shared_support(predictions, query, predictions.method.unique())
    freeze_table(predictions.assign(track=args.study, fold=0), args.out, provenance={
        "component": args.component, "study": args.study, "core_sha256": RECEIPTS["core"],
        "settings_sha256": SETTINGS, **extra,
    })


def fit_reference(args, train, query, fit, held, edges, targets):
    component = args.component
    rna = pd.concat([fit, held], verify_integrity=True)
    auxiliary = None
    if component.startswith("cellhit-"):
        shard = int(component.split("-")[1])
        drugs = np.array_split(sorted(query.compound_id.unique()), SHARDS)[shard]
        query = query[query.compound_id.isin(drugs)]
        component = "cellhit"
    if component in {"cellhit", "velodrome_inductive", "codeae_inductive"}:
        verify_upstream(args.upstream)
    if component == "v0plus_inductive":
        from .v0plus_neural import load_auxiliary

        auxiliary, _ = load_auxiliary(args.auxiliary)
    config = json.loads(args.settings.read_text())["settings"][component]
    locked = [tuple(item) for item in config if component != "linear" or item[0] in REFERENCES]
    predictions, _ = candidate_forecasts(component, train, query, rna, edges, targets,
                                          selected=locked, gpu=args.gpu, upstream=args.upstream,
                                          auxiliary=auxiliary)
    predictions.method = predictions.method.str.split("/").str[0]
    return predictions, query


def merge(args):
    _, query, _, _ = data(args, args.study)
    columns = [*KEY, "track", "fold", "method", "predicted_value"]
    parts, hashes = [], {}
    for component in expected_components(args.study):
        piece, seal = verify_forecast(args.parts / f"{component}.parquet")
        info = seal["provenance"]
        if (info["component"], info["study"], info["core_sha256"], info["settings_sha256"]) != (
                component, args.study, RECEIPTS["core"], SETTINGS):
            raise ValueError("Unexpected measured context prediction")
        hashes[component] = seal["sha256"]
        parts.append(piece[columns])
    if args.study == "LIMORE":
        archived, archive = verify_forecast(args.archive)
        if archive["sha256"] != LIMORE_ARCHIVE:
            raise ValueError("Unexpected corrected reference archive")
        preserved = archived[archived.method.isin({f"corrected/{name}" for name in REFERENCES})].copy()
        preserved.method = preserved.method.str.removeprefix("corrected/")
        parts.append(preserved[columns].assign(track=args.study))
        hashes["corrected_archive"] = LIMORE_ARCHIVE
    joint = pd.concat(parts, ignore_index=True)
    primary = primary_arm(args.study)
    pairs = joint[joint.method.isin(["drug_ridge", primary])]
    blend = pairs.groupby([*KEY, "track", "fold"], as_index=False).predicted_value.mean()
    joint = pd.concat([joint, blend.assign(method="ridge_measured_average")])
    expected = {*REFERENCES, "structured", "target_expression", "measured_coding", "genotype_permutation",
                "ridge_measured_average"}
    if args.study == "LIMORE":
        expected |= {"measured_copy", "measured_genome"}
    shared_support(joint, query, expected)
    labels = OBSERVATIONS if args.study == "LIMORE" else ASSET_HASHES["supplement"]
    freeze_table(joint, args.out, provenance={"study": args.study, "settings_sha256": SETTINGS,
        "core_sha256": RECEIPTS["core"], "component_hashes": hashes, "label_sha256": labels})


def primary_arm(study):
    return "measured_genome" if study == "LIMORE" else "measured_coding"


def comparisons(study):
    primary = primary_arm(study)
    references = ("structured", "target_expression", "genotype_permutation", "drug_mean", "drug_ridge", "cellhit")
    result = [(primary, name) for name in references]
    result += [("target_expression", "structured"), ("ridge_measured_average", "drug_ridge"),
               ("ridge_measured_average", "cellhit"), ("ridge_measured_average", "drug_mean")]
    if study == "LIMORE":
        result += [("measured_copy", "target_expression"), ("measured_coding", "target_expression")]
    return result


def paired_gates(study, across_cell, across_drug):
    primary = primary_arm(study)
    within_drug = {row["baseline"]: row["ci95"] for row in across_cell if row["candidate"] == primary}
    within_cell = {(row["baseline"], row["metric"]): row["ci95"]
                   for row in across_drug if row["candidate"] == primary}
    measured_gain = {name: all(within_drug[name][metric][0] > 0 for metric in ("spearman", "pearson"))
                     for name in ("target_expression", "genotype_permutation")}
    joint_gain = {name: all(within_drug[name][metric][0] > 0 for metric in ("spearman", "pearson"))
                  and all(within_cell[name, metric][0] > 0 for metric in ("spearman", "ndcg_at_5"))
                  for name in ("drug_mean", "drug_ridge", "cellhit")}
    return {"measured_gain": all(measured_gain.values()), "measured_gain_by_control": measured_gain,
            "joint_gain": all(joint_gain.values()), "joint_gain_by_reference": joint_gain}


def evaluate(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    _, query, _, _ = data(args, args.study)
    forecast, seal = verify_forecast(args.forecast)
    provenance = seal["provenance"]
    if (provenance["study"], provenance["settings_sha256"], provenance["core_sha256"]) != (
            args.study, SETTINGS, RECEIPTS["core"]):
        raise ValueError("Unexpected joint forecast lineage")
    require_hash(args.labels, provenance["label_sha256"])
    if args.study == "LIMORE":
        joined = join_observations(forecast, pd.read_parquet(args.labels))
    else:
        book = load_workbook(args.labels, read_only=True, data_only=True)
        sheet = list(book["S6. AUC"].values)
        outcome = []
        for row in query.itertuples(index=False):
            values = sheet[row.response_row - 1]
            if values[0] != row.cell_id or not np.isfinite(values[row.response_column]):
                raise ValueError("The LICOB label differs from its registered patient and drug")
            outcome.append(-float(values[row.response_column]))
        book.close()
        require_hash(args.patient_audit, DONORS_HASH)
        patient_info = json.loads(args.patient_audit.read_text())
        if patient_info != cohort_metadata(args.licob, args.labels):
            raise ValueError("Patient grouping changed")
        joined = join_observations(forecast, query.assign(endpoint_value=outcome)).merge(
            pd.DataFrame(patient_info["models"]), on="cell_id", validate="many_to_one")
        joined = adult_donor_outcomes(joined)
    pairs = comparisons(args.study)
    overall, drugs, _ = axis_metrics(joined, pairs)
    intervals = [{"candidate": a, "baseline": b, "ci95": crossed_patient_drug_ci(joined, a, b)}
                 for a, b in pairs]
    strata = {}
    for name, parents in compound_strata(query).items():
        if parents:
            section = joined[joined.compound_id.isin(parents)]
            values = axis_metrics(section, ())[0]
            if len(parents) < 6:
                for block in values["scores"]:
                    block.pop("macro_ndcg_at_5", None)
                    block.pop("macro_top5_overlap", None)
            strata[name] = {"compounds": len(parents), "metrics": values}
    publish_json(args.out, {"study": args.study, "scope": "previously evaluated, source-only technical comparison",
        "forecast_sha256": seal["sha256"], "label_sha256": provenance["label_sha256"],
        "models": query.cell_id.nunique(), "lineages": joined.lineage_group.nunique(),
        "rows": joined.row_id.nunique(), "primary": primary_arm(args.study), "overall": overall,
        "crossed_ci95": intervals, "strata": strata,
        "gates": paired_gates(args.study, intervals, overall["paired_contrasts"]),
        "compounds": query.drop_duplicates("compound_id").set_index("compound_id").source_compound_label.to_dict(),
        "per_drug": drugs.astype(object).where(drugs.notna(), None).to_dict("records"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="action", required=True)
    fields = {
        "prepare": (*measured_drivers.INPUT_HASHES,),
        "predict": ("genomics", "priors", "hallmark", "targets", "auxiliary", "upstream", "settings", "patient_audit"),
        "merge": ("parts", "archive"),
        "score": ("forecast", "labels", "patient_audit"),
    }
    for action, names in fields.items():
        stage = stages.add_parser(action)
        for name in (*names, "core", "limore", "licob", "out"):
            stage.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
        if action != "prepare":
            stage.add_argument("--study", choices=STUDIES, required=True)
        if action == "predict":
            stage.add_argument("--component", choices=expected_components("LICOB"), required=True)
            stage.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "predict": predict, "merge": merge, "score": evaluate}[args.action](args)


if __name__ == "__main__":
    main()
