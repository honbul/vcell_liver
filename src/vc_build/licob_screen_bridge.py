"""Transfer exact-parent single-dose PRISM and NCI GI50 to LICOB donor ranks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_source_data import alias_token, unambiguous_parent
from .external import KEY
from .function_data import new_output
from .io import freeze_table, publish_json, verify_forecast
from .licob_cold import checked
from .licob_node_recovery import joined_outcomes
from .licob_prism_restored import OLD_INPUT, RESTORED
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda, domain_kernels
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

TRACK = "LICOB_SCREEN_BRIDGE"
PRIOR = "e7286a8eb88b003c8ccc365469ebcbb926e4a9ce187f68634b2fa22d1605103f"
INPUTS = {
    "broad_samples.tsv": "f0bff0ce9a049e08b6728adb2beb65583f9bf225b4fc49960dc3193196ba5ebb",
    "prism_compounds.csv": "7e78f5901c1a97d2baab0789ab89832e716388da4eacaa9f094e7d2f2f5a3463",
    "prism_treatments.csv": "6be6422ba804ad0775e78b457677bdf088707b9354746a03e110ae63f5eb2061",
    "prism_matrix.csv": "3b6554cfc6c765af53088a676edc7bce00ee7d84fe808b93bbfa892de607bc3d",
    "nci_smiles.csv": "1e5cb58713d0a22b8cbb8865687d357bce3bda2caa2b1b0a2e7b20c58ddd75d3",
    "nci_gi50.zip": "65212008012e0fc4648709f5f081c18e577986a40a8ba45ea271f8c3bb81f728",
}
MODELS_SHA256 = "9dbb9de8805696c1345816ab07edd23fb4fd95e117739f3c5c3b1cf062c1233b"
PRISM_IDS = ("BRD-K83030136-001-02-9", "BRD-K00003509-001-01-9", "BRD-K00003460-001-01-9",
             "BRD-K42728290-001-09-1", "BRD-K22149900-001-03-9")
NCI_IDS = (654509, 764487, 764757, 776422, 777193, 780041, 789702, 800072, 803600, 803789)
ARMS = ("drug_mean", "historical", "prism_transfer", "nci_transfer", "equal_transfer")
PRIMARY = ARMS[-1]
RANK = ("spearman", "ndcg_at_5", "top5_overlap")


def exact_catalogue(rows, identity, smiles, registered):
    catalogue = {}
    for key, group in rows.groupby(identity):
        parents = [unambiguous_parent(value) for value in group[smiles].drop_duplicates()]
        if parents and all(parent is not None for parent in parents):
            ids = {parent[1] for parent in parents}
            if len(ids) == 1 and ids <= registered:
                catalogue[key] = ids.pop()
    return catalogue


def prism_rows(raw, supported, registered):
    samples = pd.read_csv(raw / "broad_samples.tsv", sep="\t", comment="!").dropna(subset=["smiles"])
    lots = samples[samples.broad_id.isin(PRISM_IDS)]
    parents = exact_catalogue(lots, "broad_id", "smiles", registered)
    if len(parents) != 5:
        raise ValueError("PRISM physical lot mapping differs from preflight")
    ids = ["BRD:" + native for native in parents]
    metadata = pd.read_csv(raw / "prism_compounds.csv")
    metadata = metadata[metadata.IDs.isin(ids)].set_index("IDs", verify_integrity=True).loc[ids]
    design = pd.read_csv(raw / "prism_treatments.csv")
    if not metadata.dose.eq(2.5).all():
        raise ValueError("Unexpected treatment concentrations")
    cohorts = {}
    for native, screen in zip(parents, metadata.screen):
        if screen == "REP.1M":
            selected = design[design.broad_id.eq(native)]
            if (len(selected) != 6 or not selected.screen.eq("REP1M").all()
                    or not selected.dose.eq(2.5).all() or set(selected.culture) != {"PR500A", "PR500B"}):
                raise ValueError(f"Unexpected REP1M design for {native}")
        elif screen != "REP.PRIMARY":
            raise ValueError(f"Unknown measurement source {screen}")
        cohorts[native] = screen
    matrix = pd.read_csv(raw / "prism_matrix.csv", index_col=0).loc[ids]
    if not matrix.index.is_unique:
        raise ValueError("Registered PRISM lots must have one response row each")
    matrix = matrix.where(np.isfinite(matrix))
    table = -matrix.loc[:, matrix.columns.isin(supported)].T
    table.columns = table.columns.str.removeprefix("BRD:")
    return table, parents, {"release_rows": matrix.notna().sum(axis=1).to_dict(), "screens": cohorts}


def nci_rows(raw, models, supported, registered):
    samples = pd.read_csv(raw / "nci_smiles.csv")
    candidates = samples[samples.NSC.isin(NCI_IDS)].dropna(subset=["SMILES"])
    parents = exact_catalogue(candidates, "NSC", "SMILES", registered)
    if len(parents) != 9 or 654509 in parents:
        raise ValueError("NCI lot mapping differs from preflight")
    aliases = pd.concat([models.set_index("ModelID")[column].dropna()
                         for column in ("CellLineName", "StrippedCellLineName", "CCLEName")])
    alias_table = aliases.rename("alias").reset_index().assign(alias=lambda data: data.alias.map(alias_token))
    alias_table = alias_table.drop_duplicates()
    unique = alias_table[~alias_table.alias.duplicated(keep=False)].set_index("alias").ModelID
    cols = ["NSC", "EXPID", "CONCENTRATION_UNIT", "CELL_NAME", "PANEL_NAME", "AVERAGE"]
    with pd.read_csv(raw / "nci_gi50.zip", usecols=cols, chunksize=200_000) as chunks:
        measurements = pd.concat([chunk[chunk.NSC.isin(NCI_IDS)] for chunk in chunks], ignore_index=True)
    selected = measurements[measurements.NSC.isin(parents) & measurements.CONCENTRATION_UNIT.eq("M")
                            & np.isfinite(measurements.AVERAGE)].copy()
    selected["model"] = selected.CELL_NAME.map(alias_token).map(unique)
    # Count raw support before alias filtering and aggregation.
    support = selected.groupby("NSC").agg(rows=("NSC", "size"), cells=("CELL_NAME", "nunique"),
                                          experiments=("EXPID", "nunique")).to_dict("index")
    selected = selected[selected.model.isin(supported)]
    table = -selected.groupby(["model", "NSC"]).AVERAGE.median().unstack("NSC")
    details = {"release_counts": support, "panels": sorted(measurements.PANEL_NAME.unique()),
               "rejected_lots": sorted(set(NCI_IDS) - parents.keys())}
    return table, parents, details


def ranked(source):
    result = source.rank(pct=True, axis=0)
    return (result - result.mean()) / result.std(ddof=0)


def fit_sources(rna, query_cells, edges, targets, name, table, parents, patient, device):
    import torch

    supported = {}
    values = ranked(table)
    # Patients carrying more than one source cell model receive a total weight of one per lot.
    fit_cells = table.index
    main, cross = domain_kernels(rna, fit_cells, query_cells, edges, targets, device)
    for native, parent in parents.items():
        x = values[native].dropna()
        n_groups = x.index.map(patient).nunique()
        if n_groups < 20 or x.nunique() < 5:
            continue
        idx = torch.as_tensor(fit_cells.get_indexer(x.index), device=device)
        weights = x.index.map(patient).value_counts()
        penalties = x.index.map(patient).map(weights).to_numpy()
        system = main[idx[:, None], idx] + 4 * torch.diag(torch.as_tensor(penalties, device=device,
                                                                          dtype=main.dtype))
        response = torch.as_tensor(x.to_numpy(), dtype=main.dtype, device=device)
        prediction = (cross[:, idx] @ torch.linalg.solve(system, response)).cpu().numpy()
        spread = prediction.std()
        if not np.isfinite(spread) or spread <= 1e-8:
            raise ValueError(f"Degenerate {name} prediction for {native}")
        supported[native] = (parent, (prediction - prediction.mean()) / spread, n_groups, len(x))
    vectors = {}
    for parent in sorted({entry[0] for entry in supported.values()}):
        vectors[parent] = np.mean([entry[1] for entry in supported.values() if entry[0] == parent], axis=0)
    support = {str(native): {"parent": parent, "groups": groups, "cells": cells}
               for native, (parent, _, groups, cells) in supported.items()}
    return vectors, support


def replace_order(historical, vectors):
    updated = historical.copy()
    for parent, signal in vectors.items():
        # Match the previously frozen per-drug score distribution. No query labels are used.
        positions = pd.Series(signal, index=historical.index).rank(method="average").to_numpy() - 1
        prior = np.sort(historical[parent].to_numpy())
        updated[parent] = np.interp(positions, np.arange(len(prior)), prior)
    return updated


def predict(args):
    area = args.historical / "inputs"
    if checked(area, ("train", "query", "rna")) != OLD_INPUT:
        raise ValueError("LICOB inputs have changed")
    old, freeze = verify_forecast(args.prior)
    if freeze["sha256"] != PRIOR:
        raise ValueError("Restored historical forecast has changed")
    for file, digest in INPUTS.items():
        require_hash(args.raw / file, digest)
    require_hash(args.models, MODELS_SHA256)
    train = pd.read_parquet(area / "train.parquet", columns=["cell_id", "compound_id"])
    query = pd.read_parquet(area / "query.parquet")
    rna = pd.read_parquet(area / "rna.parquet")
    query_cells = pd.Index(sorted(query.cell_id.unique()))
    candidates = query[query.remaining_unspecified_stereo.eq(0)]
    registered = set(candidates.compound_id) - set(train.compound_id) - set(RESTORED)
    if len(registered) != 11 or set(train.cell_id) & set(query_cells):
        raise ValueError("Unmatched parent or cell roster changed")
    models = pd.read_csv(args.models, usecols=["ModelID", "PatientID", "CellLineName",
                                              "StrippedCellLineName", "CCLEName", "Age", "OncotreeCode"])
    patient = models.set_index("ModelID").PatientID
    patient = patient.fillna(patient.index.to_series())
    edges, targets, features = molecular_inputs(args.priors, args.hallmark, args.targets)
    pars = (set(train.cell_id), registered)
    screens = {"prism": prism_rows(args.raw, *pars), "nci": nci_rows(args.raw, models, *pars)}
    vectors, support = {}, {}
    device = device_cuda()
    for name, (table, parents, audit) in screens.items():
        vectors[name], fitted = fit_sources(rna, query_cells, edges, targets, name, table, parents, patient, device)
        audit["fitted"] = fitted
        audit["adult_hcc_by_lot"] = {
            str(lot): int(table[lot].dropna().index.isin(
                models.loc[models.OncotreeCode.eq("HCC") & models.Age.ge(18), "ModelID"]).sum())
            for lot in parents
        }
        support[name] = audit
    combined = {key: np.mean([item[key] for item in vectors.values() if key in item], axis=0)
                for key in sorted(set(vectors["prism"]) | set(vectors["nci"]))}
    if len(combined) != 6:
        raise ValueError("Expected six independently measured exact parents")
    history = old[old.method.eq("restored_residual")].pivot(index="cell_id", columns="compound_id",
                                                           values="predicted_value").loc[query_cells]
    means = old[old.method.eq("old_mean")].set_index("row_id").predicted_value
    outputs = [query[KEY].assign(method="drug_mean", predicted_value=query.row_id.map(means))]
    methods = {"historical": history, "prism_transfer": replace_order(history, vectors["prism"]),
               "nci_transfer": replace_order(history, vectors["nci"]), PRIMARY: replace_order(history, combined)}
    indices = history.index.get_indexer(query.cell_id), history.columns.get_indexer(query.compound_id)
    for name, table in methods.items():
        outputs.append(query[KEY].assign(method=name, predicted_value=table.to_numpy()[indices]))
    output = pd.concat(outputs, ignore_index=True)
    shared_support(output, query, ARMS)
    with new_output(args.run) as run:
        freeze_table(output.assign(track=TRACK, fold=0), run / "forecast.parquet", provenance={
            "sources": INPUTS, "models_sha256": MODELS_SHA256, "rna_features": features,
            "previous_sha256": PRIOR, "query_input_sha256": OLD_INPUT, "support": support,
            "with_transfer": {**{name: sorted(value) for name, value in vectors.items()}, PRIMARY: sorted(combined)},
        })


def score(args):
    area = args.historical / "inputs"
    if checked(area, ("query",)) != OLD_INPUT:
        raise ValueError("LICOB query has changed")
    query = pd.read_parquet(area / "query.parquet")
    rows, seal = verify_forecast(args.run / "forecast.parquet")
    shared_support(rows, query, ARMS)
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    values = adult_donor_outcomes(joined_outcomes(query, rows, args.supplement))
    selected = seal["provenance"]["with_transfer"]
    comparisons = [(name, reference) for name in ARMS[2:] for reference in ARMS[:2]]
    result = {}
    for scope, drugs in {**selected, "full": sorted(query.compound_id.unique())}.items():
        subset = values[values.compound_id.isin(drugs)]
        report, per_drug, _ = axis_metrics(subset, comparisons)
        result[scope] = {"report": report, "donors": subset.lineage_group.nunique(), "drugs": len(drugs),
                         "per_drug": per_drug.astype(object).where(per_drug.notna(), None).to_dict("records"),
                         "ci95": {ref: crossed_patient_drug_ci(subset, PRIMARY, ref) for ref in ARMS[:2]}}
    gates = {}
    for reference in ARMS[:2]:
        same = result[PRIMARY]["ci95"][reference]
        full = {row["metric"]: row["ci95"][0] for row in result["full"]["report"]["paired_contrasts"]
                if row["candidate"] == PRIMARY and row["baseline"] == reference}
        gates[reference] = (set(same) == {"pearson", "spearman"} and all(ci[0] > 0 for ci in same.values())
                            and all(full.get(metric, -np.inf) > -.005 for metric in RANK))
    publish_json(args.run / "score.json", {
        "freeze_sha256": seal["sha256"], "outcome_sha256": ASSET_HASHES["supplement"],
        "results": result, "gates": gates, "pass": all(gates.values()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, task, paths in (
        ("predict", predict, ("raw", "prior", "models", "priors", "hallmark", "targets")),
        ("score", score, ("supplement",)),
    ):
        sub = stages.add_parser(name)
        for path in ("historical", "run", *paths):
            sub.add_argument("--" + path, type=Path, required=True)
        sub.set_defaults(task=task)
    args = parser.parse_args()
    args.task(args)


if __name__ == "__main__":
    main()
