"""Register three prior-exposed PDO cohorts and separate calibration from scoring labels."""

import argparse
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vc_build.external import KEY
from vc_build.hcc_3d_study_transfer import DRUGS, recipient_outcomes, target_inputs
from vc_build.io import freeze_table, publish_json, sha256, verify_forecast
from vc_build.yang_hcc_adaptation import BENCHMARK_INPUTS
from vc_build.yang_hcc_external import EXPOSED_DONORS, outcomes, read_inputs

from .base_models import as_forecast, predict_part
from .contract import BUILD, RUNS, TRAIN_SHA, corrected, verified
from .ledger import checkpoint, frozen_pairwise, load_base, members
from .pairwise import neural_banks

STUDIES = ("LICOB", "Wong", "Yang")
BUDGETS = (2, 4, 8)
REPEATS = 100
MIN_RNA = .85
LICOB = RUNS / "licob-author-2f1328b"
WONG = RUNS / "wong-84d021d/cohort"
YANG = RUNS / "yang-external-54fcad5/inputs"
LICOB_BOOK = BUILD / "licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
WONG_BOOK = BUILD / "licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx"
YANG_BOOK = BUILD / "v0plus-public-assets/yang-plc-rv2w3dv9rs-v2/mmc8.xlsx"
EXPECTED_PATIENTS = {"LICOB": 29, "Wong": 10, "Yang": 74}


def inputs(study):
    if study == "Yang":
        read_inputs(YANG, BENCHMARK_INPUTS)
        query = pd.read_parquet(YANG / "query.parquet")
        query = query.loc[~query.lineage_group.isin(EXPOSED_DONORS)]
        rna = pd.read_parquet(YANG / "rna.parquet")
        rna = rna.loc[sorted(query.cell_id.unique())]
    else:
        directory = LICOB / "cohort" if study == "LICOB" else WONG
        parents = pd.read_parquet(directory / "query.parquet", columns=["compound_id"]).compound_id.unique()
        query, rna = target_inputs(study, directory, LICOB / "patient-audit.json", compounds=parents)
    if query.lineage_group.nunique() != EXPECTED_PATIENTS[study] or not set(DRUGS) <= set(query.compound_id):
        raise ValueError("Adult HCC PDO donor and exact TKI panel changed")
    if "endpoint_value" in query or query.duplicated(["cell_id", "compound_id"]).any():
        raise ValueError("Unexpected labels or duplicated PDO input")
    return query, rna


def align(rna, fit):
    matching = rna.reindex(columns=fit.columns)
    support = matching.notna().mean(axis=1)
    if (support < MIN_RNA).any() or np.isinf(matching.to_numpy()).any():
        raise ValueError("Public PDO RNA support below transfer threshold")
    # An absent gene is imputed with the frozen source median. This leaves all fit transforms fixed.
    return matching.fillna(fit.median()), support


def source(args):
    train, fit, _ = corrected()
    known = set(train.compound_id)
    paths = list(members(args.task1))
    linear = load_base(checkpoint(args.task1, "linear"))
    structured = joblib.load(checkpoint(args.task1, "structured"))
    banks, hashes = neural_banks()
    banks = {k: v for k, v in banks.items() if k not in ("scgpt", "geneformer")}
    predictions, support = [], {}
    for study in STUDIES:
        q, raw_rna = inputs(study)
        eligible = q[q.compound_id.isin(known)].copy()
        rna, coverage = align(raw_rna, fit)
        if set(q.cell_id) & set(train.cell_id) or set(q.lineage_group) & set(train.lineage_group):
            raise ValueError("PDO crossed into corrected 2D training identities")
        measures = {"drug_mean": eligible.compound_id.map(linear.models["drug_mean"][0].anchor).to_numpy()}
        measures.update(predict_part("structured", structured, eligible, rna))
        measures["pairwise_rank"] = frozen_pairwise(paths, eligible, rna, banks)["pairwise_rank"]
        predictions.append(as_forecast(eligible, measures, fold=0, track=study))
        registry = q[["compound_id", "response_label"]].drop_duplicates() if study != "Wong" else (
            q[["compound_id", "source_compound_label"]].drop_duplicates().rename(
                columns={"source_compound_label": "response_label"}))
        support[study] = {"patients": int(q.lineage_group.nunique()), "pdos": int(q.cell_id.nunique()),
                          "pdo_drug_rows": len(q), "drugs": int(q.compound_id.nunique()),
                          "registered_68_exact_drugs": int(eligible.compound_id.nunique()),
                          "rna_genes_min_fraction": float(coverage.min()),
                          "outside_68": registry.loc[~registry.compound_id.isin(known)].to_dict("records")}
    freeze_table(pd.concat(predictions, ignore_index=True), args.out, provenance={
        "train_data_sha256": TRAIN_SHA, "support": support, "fixed_drug_embeddings_sha256": {
            k: v for k, v in hashes.items() if k in banks}, "pdo_fm_policy": "offline missing cell-FM flags",
        "task2_decision_sha256": sha256(args.task2 / "reference_decision.json"),
        "contract_sha256": sha256(args.contract),
    })


def treatment_labels(study, query):
    if study == "Yang":
        frames = outcomes(YANG_BOOK, query)
    else:
        frames = recipient_outcomes(study, query, LICOB_BOOK if study == "LICOB" else WONG_BOOK)
    if len(frames) != len(query) or not np.isfinite(frames.endpoint_value).all():
        raise ValueError(f"Expected wholly observed registered PDO roster for {study}")
    return frames


def balance(patients, k, seed):
    """Equalize the per-patient exposure count and spread uneven pair/episode reuse."""
    rng = np.random.default_rng(seed)
    n = len(patients)
    if k >= n - 1:
        raise ValueError("At least two independent test patients are required")
    uses = np.zeros(n, dtype=int)
    pairs, sets = {}, {}
    for _ in range(REPEATS):
        # A patient whose count is below the kth count must be selected.
        kth = np.partition(uses, k - 1)[k - 1]
        fixed = np.flatnonzero(uses < kth)
        pool = np.flatnonzero(uses == kth)
        candidates = [tuple(sorted([*fixed, *rng.choice(pool, size=k - len(fixed), replace=False)]))
                      for _ in range(72)]
        chosen = min(candidates, key=lambda v: (sets.get(v, 0), sum(pairs.get(p, 0) for p in combinations(v, 2))))
        sets[chosen] = sets.get(chosen, 0) + 1
        for pair in combinations(chosen, 2):
            pairs[pair] = pairs.get(pair, 0) + 1
        uses[list(chosen)] += 1
        yield tuple(patients[i] for i in chosen)
    if uses.max() - uses.min() > 1:
        raise ValueError("Calibration exposure is unbalanced")


def patient_table(frame, columns):
    groups = frame.groupby(["lineage_group", "compound_id"], sort=True)
    result = groups[columns].mean()
    # Preserve exact constants across unequal numbers of regional samples.
    for key in columns:
        unique = groups[key].nunique()
        result[key] = result[key].where(unique.ne(1), groups[key].first())
    return result.reset_index()


def register(args):
    _, prior = verify_forecast(args.prior)
    if prior["provenance"]["contract_sha256"] != sha256(args.contract):
        raise ValueError("Received 3D signal from an incompatible contract")
    args.out.mkdir(parents=True, exist_ok=False)
    coverage, artifacts = {}, {}
    for j, study in enumerate(STUDIES):
        q, _ = inputs(study)
        actual = treatment_labels(study, q)
        frame = patient_table(actual, ["endpoint_value"])
        patients = sorted(frame.lineage_group.unique())
        if len(frame) != len(patients) * frame.compound_id.nunique():
            raise ValueError("Patient×compound rectangle unexpectedly incomplete")
        output = args.out / study
        output.mkdir()
        truth = output / "test-outcomes.parquet"
        frame.to_parquet(truth, index=False)
        artifacts[str(truth.relative_to(args.out))] = sha256(truth)
        coverage[study] = {}
        for k in BUDGETS:
            layouts, labels, calibration_count = [], [], {p: 0 for p in patients}
            for repeat, fitted in enumerate(balance(patients, k, 20261123 + 100 * j + k)):
                held = sorted(set(patients) - set(fitted))
                if len(fitted) != k or set(fitted) & set(held):
                    raise ValueError("Patient overlaps between fit and evaluation roles")
                for patient in fitted:
                    calibration_count[patient] += 1
                layouts.append({"repeat": repeat, "fit": list(fitted), "held": held})
                labels.append(frame[frame.lineage_group.isin(fitted)].assign(repeat=repeat))
            p = output / f"k{k}-calibration-only.parquet"
            pd.concat(labels, ignore_index=True).to_parquet(p, index=False)
            q = output / f"k{k}-layout.json"
            publish_json(q, {"layouts": layouts, "k": k})
            for f in (p, q):
                artifacts[str(f.relative_to(args.out))] = sha256(f)
            coverage[study][str(k)] = {"test_patients_per_repeat": len(patients) - k,
                "calibration_min_per_patient": min(calibration_count.values()),
                "calibration_max_per_patient": max(calibration_count.values()),
                "unique_calibration_sets": len({tuple(t["fit"]) for t in layouts})}
    publish_json(args.out / "registration.json", {"forecast_sha256": prior["sha256"], "artifacts": artifacts,
                 "contract_sha256": sha256(args.contract), "calibration_balance": coverage,
                 "source_workbook_sha256": {k: sha256(v) for k, v in zip(STUDIES, (LICOB_BOOK, WONG_BOOK, YANG_BOOK))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "register"))
    for name in ("task1", "task2", "out", "prior", "contract"):
        parser.add_argument("--" + name, type=Path, required=name in ("out", "contract"))
    args = parser.parse_args()
    {"source": source, "register": register}[args.stage](args)


if __name__ == "__main__":
    main()
