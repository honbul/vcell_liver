"""Fit a drug-ID-free molecular model and score it against frozen scaffold baselines."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import join_observations
from .expanded_panel_cv import expanded_targets
from .external import KEY
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import ARMS as COLD_REFS, PRIMARY as STRUCTURED, checked, double_cold
from .licob_dose_sources import COLD
from .licob_node_recovery import joined_outcomes
from .limore_corrected import OBSERVATIONS, registered, shared_support
from .structured_cv import attach, paired_intervals, with_assay
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import fit_molecular, load_auxiliary
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

COLD_DIGESTS = (
    "111859666c6ba43d059b5d01cffa8c7d01fc5c00421fecc55689b2e22064eb8a",
    "8abaf10f6f18d789a4f5ef2f84741d1d62a5084fde489b42d0d16d17566378b2",
    "421bafdd56943e0150d4b525894bb75f9b39957d5c2e84754298e9a27cefedad",
    "4cc7e3afa71fee91c0ca9a12ce06002d7a1016f9d4951b1833237830e3d7a2a1",
)
LIMORE_DIGEST = "fe7f9e656801d63fd1263ec3e94093cad86b049130aa038d4f2e8fdd059b02bb"
LIMORE_REFS = tuple("corrected/" + name for name in (
    "drug_mean", "drug_ridge", "drug_elasticnet", "lightgbm", "catboost",
    "cellhit", "codeae_inductive", "velodrome_inductive", "structured", "v0plus_inductive",
))
MOLECULAR = "id_free_molecular"
ABLATION = "id_free_rna_ecfp"
PRIMARY = "structured_molecular_blend"
ABLATION_BLEND = "structured_rna_ecfp_blend"
NEW = (MOLECULAR, ABLATION, PRIMARY, ABLATION_BLEND)
WIDTH, EPOCHS, BATCH = 192, 45, 4096
TRACKS = {"licob": "LICOB_COLD", "limore": "LIMORE50"}


def registered_panel(args, names):
    digest = checked(args.inputs, names)
    if digest != COLD:
        raise ValueError("Expected the signed full 76-drug input")
    return digest


def reference(path, digest, methods):
    table, seal = verify_forecast(path)
    if seal["sha256"] != digest:
        raise ValueError("Unexpected archived comparator")
    rows = table[table.method.isin(methods)]
    if set(rows.method) != set(methods):
        raise ValueError("Some archived comparator forecasts are absent")
    return rows


def source_query(fit, labels, refs, fold):
    query = validation_roster(labels)
    expected = with_assay(query, labels, fold)
    key = [*KEY, "track", "fold"]
    if set(fit.lineage_group) & set(query.lineage_group):
        raise ValueError("Fit and scaffold-test lineage groups overlap")
    if not expected[key].sort_values(key).reset_index(drop=True).equals(
        refs[refs.method.eq("chemical_mean")][key].sort_values(key).reset_index(drop=True)
    ):
        raise ValueError("The restored scaffold split differs from the signed comparator")
    return query


def predict_nn(fit, queries, rna, edges, targets, auxiliary):
    fit_rna = rna.loc[sorted(fit.cell_id.unique())]
    _, anchor, observations, scale = panel_targets(fit)
    registry = pd.concat(queries, ignore_index=True)
    registry = registry[["compound_id", "isomeric_smiles"]].drop_duplicates()
    fit_grid = pd.MultiIndex.from_product(
        [fit_rna.index, sorted(registry.compound_id.unique())],
        names=["cell_id", "compound_id"],
    ).to_frame(index=False)
    targets, _ = expanded_targets(
        fit[~fit.compound_id.isin(targets)], targets, rna.columns,
    )
    result = {}
    for arm, options in (
        (MOLECULAR, {"drug_identity": False}),
        (ABLATION, {"drug_identity": False, "paths": False, "targets": False,
                    "chronos": False, "drivers": False}),
    ):
        model = fit_molecular(
            "v0plus_inductive", WIDTH, EPOCHS, observations, anchor, fit_rna, None,
            edges, targets, auxiliary, gpu=True, interaction_target=True,
            registry=registry, options=options, batch_size=BATCH,
        )
        _, fit_values = model.predict(fit_grid, fit_rna)
        mean = fit_grid.assign(value=fit_values).groupby("compound_id").value.mean()
        result[arm] = {}
        for name, query in queries.items():
            _, values = model.predict(query, rna.loc[sorted(query.cell_id.unique())])
            values -= query.compound_id.map(mean).to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("The molecular model did not cover the held catalogue")
            result[arm][name] = values
        print(f"Fitted {arm} on {len(fit)} rows", flush=True)
        del model
    return result, scale


def combined(query, refs, values, anchor_name, structure_name):
    expected = query[KEY].set_index("row_id")
    arms = {}
    for method in (anchor_name, structure_name):
        table = refs[refs.method.eq(method)][[*KEY, "predicted_value"]].drop_duplicates()
        keyed = table.set_index("row_id").reindex(expected.index)
        if not keyed[expected.columns].equals(expected):
            raise ValueError("Frozen reference identities differ from the query")
        arms[method] = keyed.predicted_value.to_numpy()
    mean, structured = arms[anchor_name], arms[structure_name]
    individual = {name: mean + vals for name, vals in values.items()}
    forecast = individual | {
        PRIMARY: .5 * (individual[MOLECULAR] + structured),
        ABLATION_BLEND: .5 * (individual[ABLATION] + structured),
    }
    return pd.concat(
        [query[KEY].assign(method=name, predicted_value=values) for name, values in forecast.items()],
        ignore_index=True,
    )


def predict(args):
    digest = registered_panel(args, ("train", "rna", "query"))
    train = pd.read_parquet(args.inputs / "train.parquet")
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    cold = reference(args.cold_reference / f"fold-{args.fold}.parquet", COLD_DIGESTS[args.fold], COLD_REFS)
    queries, references = {}, {}
    if args.fold < 3:
        train, labels = double_cold(train, args.fold)
        queries["source"] = source_query(train, labels, cold, args.fold)
        references["source"] = (cold, "chemical_mean", STRUCTURED)
    else:
        queries["licob"] = pd.read_parquet(args.inputs / "query.parquet")
        _, queries["limore"], all_2d_rna = registered(args, "corrected")
        genes = rna.columns
        limore_rna = all_2d_rna.loc[sorted(queries["limore"].cell_id.unique()), genes]
        rna = pd.concat([rna, limore_rna], verify_integrity=True)
        limore = reference(args.limore_reference, LIMORE_DIGEST, LIMORE_REFS)
        references = {
            "licob": (cold, "chemical_mean", STRUCTURED),
            "limore": (limore, "corrected/drug_mean", "corrected/structured"),
        }
    if any(set(train.lineage_group) & set(query.lineage_group) for query in queries.values()):
        raise ValueError("External training and query lineages overlap")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    # Query catalogues can supply static target annotations; RNA feature and target vocabulary
    # fitting is limited to targets of actual fit compounds inside MolecularData.prepare.
    if args.fold == 3:
        targets, _ = expanded_targets(queries["licob"][~queries["licob"].compound_id.isin(targets)],
                                      targets, rna.columns)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    outputs, calibration = predict_nn(train, queries, rna, edges, targets, auxiliary)
    for mode, query in queries.items():
        refs, mean, structure = references[mode]
        values = {arm: outputs[arm][mode] for arm in (MOLECULAR, ABLATION)}
        forecast = combined(query, refs, values, mean, structure)
        if mode == "source":
            forecast = with_assay(forecast, labels, args.fold)
            name = f"fold-{args.fold}"
        else:
            shared_support(forecast, query, NEW)
            forecast = forecast.assign(track=TRACKS[mode], fold=0)
            name = mode
        freeze_table(pd.concat([forecast, refs], ignore_index=True), args.out / f"{name}.parquet", provenance={
            "stage": mode, "fold": args.fold, "input_sha256": digest,
            "comparator_sha256": LIMORE_DIGEST if mode == "limore" else COLD_DIGESTS[args.fold],
            "fit_rows": len(train), "fit_drugs": int(train.compound_id.nunique()),
            "fit_cells": int(train.cell_id.nunique()), "genes": rna.shape[1],
            "query_seen_drugs": int(query.loc[query.compound_id.isin(train.compound_id), "compound_id"].nunique()),
            "calibration": calibration, "priors": priors, "auxiliary": aux,
        })


def summarize(rows, query, comparators, *, source=False):
    pairs = [(PRIMARY, comparator) for comparator in comparators]
    metrics, drugs, _ = axis_metrics(rows, pairs)
    boots = (lambda baseline: paired_intervals(rows, PRIMARY, baseline, repeats=800)) if source else (
        lambda baseline: crossed_patient_drug_ci(rows, PRIMARY, baseline, repeats=1600)
    )
    intervals = {comparator: boots(comparator) for comparator in comparators}
    ranking = {(value["baseline"], value["metric"]): value["ci95"][0] for value in metrics["paired_contrasts"]}
    gates = {name: all(intervals[name].get(metric, [-1])[0] > 0 for metric in ("spearman", "pearson"))
             and all(ranking.get((name, metric), -1) > -.005 for metric in ("spearman", "ndcg_at_5", "top5_overlap"))
             for name in comparators}
    groups = {}
    if query is not None:
        for name, identities in compound_strata(query).items():
            matched = drugs[drugs.compound_id.isin(identities)].dropna(subset=["spearman", "pearson"])
            if not matched.empty:
                groups[name] = {"drugs": int(matched.compound_id.nunique()),
                                "mean": matched.groupby("method")[["spearman", "pearson"]].mean().to_dict("index")}
    return {"metrics": metrics, "intervals": intervals, "gates": gates, "strata": groups,
            "drug_count": int(rows.compound_id.nunique()), "lineage_count": int(rows.lineage_group.nunique())}


def sealed(directory, name, mode, fold, digest):
    rows, proof = verify_forecast(directory / f"{name}.parquet")
    meta = proof["provenance"]
    expected = LIMORE_DIGEST if mode == "limore" else COLD_DIGESTS[fold]
    if (meta["stage"], meta["fold"], meta["input_sha256"], meta["comparator_sha256"]) != (
        mode, fold, digest, expected
    ):
        raise ValueError("Unexpected molecular forecast provenance")
    return rows, proof["sha256"]


def score(args):
    mode = args.mode
    if mode == "source":
        digest = registered_panel(args, ("train",))
        parts = [sealed(args.parts, f"fold-{fold}", mode, fold, digest) for fold in range(3)]
        train = pd.read_parquet(args.inputs / "train.parquet")
        joined = pd.concat([attach(rows, double_cold(train, fold)[1]) for fold, (rows, _) in enumerate(parts)])
        contrasts = ("chemical_mean", "neighbor_transfer", STRUCTURED, ABLATION_BLEND)
        results = {assay: summarize(rows, None, contrasts, source=True)
                   for assay, rows in joined.groupby("track")}
        proof = {"hashes": [digest for _, digest in parts],
                 "gate": sum(all(r["gates"].values()) for r in results.values()) >= 2}
    else:
        digest = registered_panel(args, ("query",))
        rows, incoming = sealed(args.parts, mode, mode, 3, digest)
        if mode == "licob":
            query = pd.read_parquet(args.inputs / "query.parquet")
            shared_support(rows, query, (*COLD_REFS, *NEW))
            require_hash(args.supplement, ASSET_HASHES["supplement"])
            scored = adult_donor_outcomes(joined_outcomes(query, rows, args.supplement),
                                          constant_methods=("grand_mean", "chemical_mean"))
            contrasts = ("chemical_mean", "neighbor_transfer", STRUCTURED, ABLATION_BLEND)
            support = query[query.in_source]
            cold = query[~query.in_source]
            fully_specified = cold[cold.remaining_unspecified_stereo.eq(0)]
            scopes = {"full": query, "warm": support, "cold": cold, "cold_fully_specified": fully_specified}
            labels = ASSET_HASHES["supplement"]
        else:
            _, query, _ = registered(args, "corrected")
            shared_support(rows, query, (*LIMORE_REFS, *NEW))
            require_hash(args.labels, OBSERVATIONS)
            scored = join_observations(rows, pd.read_parquet(args.labels))
            contrasts = (*LIMORE_REFS, ABLATION_BLEND)
            scopes = {"full": query}
            labels = OBSERVATIONS
        results = {name: summarize(scored[scored.compound_id.isin(block.compound_id)], block, contrasts)
                   for name, block in scopes.items()}
        required = ["full", "cold_fully_specified"] if mode == "licob" else ["full"]
        proof = {"hashes": [incoming], "gate": all(all(results[name]["gates"].values()) for name in required),
                 "label_sha256": labels}
    publish_json(args.out, {"stage": mode, "inputs_sha256": digest, "results": results} | proof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("cold_reference", "limore_reference", "priors", "hallmark", "targets", "auxiliary"),
        "score": ("parts", "supplement", "labels"),
    }
    for stage, paths in fields.items():
        cli = sub.add_parser(stage)
        for field in ("inputs", "core", "original", "out", *paths):
            cli.add_argument("--" + field.replace("_", "-"), type=Path, required=True)
        if stage == "predict":
            cli.add_argument("--fold", type=int, choices=range(4), required=True)
        else:
            cli.add_argument("--mode", choices=("source", "licob", "limore"), required=True)
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
