"""Freeze and separately score registered LICOB dose endpoints."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .evaluate import join_observations
from .expanded_panel_cv import expanded_targets
from .external import KEY, SOURCE
from .function_data import alias_token
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import PENALTY, checked as checked_cold
from .licob_dose_sources import COLD, SCREENS, registration
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import centered_kernels, orthogonal_parts
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

PRIOR = "4cc7e3afa71fee91c0ca9a12ce06002d7a1016f9d4951b1833237830e3d7a2a1"
SHEETS = {"auc": "S6. AUC", "ic50_bounded": "S7. IC50", "emax": "S8. Emax"}
METHODS = ("drug_mean", "matched_auc", "matched_endpoint")
PRIOR_METHODS = ("chemical_mean", "interaction_transfer")
MATCHED = ("emax", "ic50_bounded")


def read_prior(path, query):
    forecasts, seal = verify_forecast(path)
    if seal["sha256"] != PRIOR or seal["provenance"]["input_sha256"] != COLD:
        raise ValueError("Unexpected 76-drug frozen forecast")
    selected = forecasts[forecasts.method.isin(PRIOR_METHODS)]
    shared_support(selected, query, PRIOR_METHODS)
    return selected.assign(track="prior")


def source_curves(run):
    digest = registration(run / "inputs")
    blocks, receipts = [], {}
    for study in SCREENS:
        directory = run / study
        path = directory / "receipt.json"
        receipt = json.loads(path.read_text())
        curve = directory / "curves.parquet"
        if (receipt["input_sha256"], receipt["source"]) != (digest, study):
            raise ValueError("Source extract does not match the registered dose design")
        require_hash(curve, receipt["curve_sha256"])
        frame = pd.read_parquet(curve)
        if len(frame) != receipt["matched_rows"]:
            raise ValueError("Source receipt and curve rows disagree")
        if not frame.empty:
            blocks.append(frame)
        receipts[study] = receipt | {"receipt_sha256": sha256(path)}
    return pd.concat(blocks, ignore_index=True) if blocks else None, digest, receipts


def drug_gram(training, targets, device):
    import torch
    from .structured_kernel import drug_kernels

    names, terms = drug_kernels(training.drop_duplicates("compound_id"), targets, device=device)
    identity = torch.eye(len(names), device=device)
    tanimoto = ((terms["rna_chem_only"] - .75 * identity - .1) / .15).clamp(0, 1)
    target = ((terms["structured"] - .65 * identity - .15 * tanimoto - .1) / .1).clamp(0, 1)
    gram = .35 * identity + .35 * tanimoto + .2 * target + .1
    return names, centered_kernels(gram, gram)[0]


def fit_matched(fit, query, rna, edges, targets, aux, endpoint):
    import torch
    from .structured_kernel import cell_kernels, solve_residuals, training_grid

    device = device_cuda()
    cells = pd.Index(sorted(fit.cell_id.unique()))
    held = pd.Index(sorted(query.cell_id.unique()))
    if len(cells.intersection(held)):
        raise ValueError("External RNA overlaps a fit cell")
    drugs, gram = drug_gram(fit, targets, device)
    terms, features = cell_kernels(rna.loc[cells], rna.loc[held], edges, aux, device=device)
    train_cell, query_cell = centered_kernels(*terms["full"])
    ix = held.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if min(ix[0].min(), ix[1].min()) < 0:
        raise ValueError("Missing model or compound in matched forecast")
    preds, solves = {}, {}
    for outcome, method in ((endpoint, "matched_endpoint"), ("source_auc", "matched_auc")):
        data = fit.assign(y=-fit[outcome])
        means = data.groupby([*SOURCE, "compound_id"]).y
        data["residual"] = data.y - means.transform("mean")
        anchor = means.mean().groupby("compound_id").mean().reindex(drugs).to_numpy()
        weights, y = training_grid(data, cells, drugs, device=device)
        _, _, residual = orthogonal_parts(y.cpu().numpy(), weights.cpu().numpy())
        coef, steps, norm = solve_residuals(
            train_cell, gram, weights, torch.as_tensor(residual, dtype=weights.dtype, device=device), PENALTY,
        )
        preds[method] = (query_cell @ coef @ gram.T).cpu().numpy()[ix] + anchor[ix[1]]
        solves[method] = {"steps": steps, "residual_norm": norm}
        if outcome == endpoint:
            preds["drug_mean"] = anchor[ix[1]]
    return pd.concat([query[KEY].assign(method=m, predicted_value=preds[m]) for m in METHODS],
                     ignore_index=True).assign(track=endpoint, fold=0), features, solves


def freeze(args):
    from .v0plus_neural import load_auxiliary

    if checked_cold(args.cold, ("query", "rna")) != COLD:
        raise ValueError("Unexpected LICOB registration")
    query = pd.read_parquet(args.cold / "query.parquet")
    prior = read_prior(args.prior, query)
    curves, digest, receipts = source_curves(args.run)
    panels, parts = {}, [prior]
    for endpoint in MATCHED:
        observed = curves.dropna(subset=[endpoint]) if curves is not None else pd.DataFrame(
            columns=["cell_id", "compound_id", "source_id"])
        counts = observed.groupby("compound_id").cell_id.nunique()
        selected = counts[counts.ge(30)].index
        fit = observed[observed.compound_id.isin(selected)]
        target = query[query.compound_id.isin(selected)]
        panels[endpoint] = {
            "drugs": len(selected), "fit_rows": len(fit),
            "all_source_curves": len(observed), "all_source_drugs": len(counts),
            "sources": fit.groupby("source_id").size().to_dict(),
            "support": {key: {"cells": int(block.cell_id.nunique()),
                              "rows_by_source": block.source_id.value_counts().to_dict()}
                        for key, block in fit.groupby("compound_id")},
        }
        if len(selected) < 5:
            continue
        if len(parts) == 1:
            rna = pd.read_parquet(args.cold / "rna.parquet")
            edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
            aux, aux_receipt = load_auxiliary(args.auxiliary)
        annotations = pd.concat([fit, target], ignore_index=True)
        supported, _ = expanded_targets(annotations[~annotations.compound_id.isin(targets)], targets, rna.columns)
        part, used_features, solves = fit_matched(fit, target, rna, edges, supported, aux, endpoint)
        shared_support(part, target, METHODS)
        panels[endpoint].update(features=used_features, solves=solves, priors=priors, auxiliary=aux_receipt)
        parts.append(part)
    forecasts = pd.concat(parts, ignore_index=True)
    if not np.isfinite(forecasts.predicted_value).all():
        raise ValueError("Nonfinite value in registered forecasts")
    freeze_table(forecasts, args.run / "forecast.parquet",
                 provenance={"registration_sha256": digest, "cold_sha256": COLD, "prior_sha256": PRIOR,
                             "extracts": receipts, "panels": panels})


def read_outcomes(workbook, query, dose, selected=tuple(SHEETS)):
    doses = dose.set_index("compound_id")
    results = {}
    with closing(load_workbook(workbook, read_only=True, data_only=True)) as book:
        anchor = None
        for endpoint, sheet in SHEETS.items():
            rows = book[sheet].values
            next(rows)
            header = list(map(alias_token, next(rows)))
            if anchor is None:
                anchor = header
            if endpoint not in selected:
                continue
            records = [row for row in rows if row[0] is not None]
            by_model = {str(row[0]).strip(): row for row in records}
            if len(by_model) != len(records):
                raise ValueError(f"Duplicate response row in {sheet}")
            labels = []
            for key in query.itertuples(index=False):
                name = anchor[key.response_column]
                if header.count(name) != 1 or key.cell_id not in by_model:
                    raise ValueError(f"Unregistered response identity in {sheet}")
                raw = by_model[key.cell_id][header.index(name)]
                if raw is None or str(raw).strip().lower() in ("", "na", "nan", "n/a"):
                    labels.append((np.nan, "missing"))
                    continue
                number = float(raw)
                side = "within"
                if endpoint == "ic50_bounded":
                    if np.isfinite(number) and number < 0:
                        raise ValueError("Negative finite IC50 concentration in S7")
                    limits = doses.loc[key.compound_id]
                    if number < limits.minimum:
                        side = "below"
                    if number > limits.maximum:
                        side = "above"
                    number = np.log10(np.clip(number, limits.minimum, limits.maximum))
                if not np.isfinite(number):
                    raise ValueError(f"Nonfinite labeled observation in {sheet}")
                labels.append((-number, side))
            results[endpoint] = query[KEY].assign(
                endpoint_value=[x[0] for x in labels], bound_status=[x[1] for x in labels],
                donor_id=query.donor_id, adult_age_confirmed=query.adult_age_confirmed,
            )
    return results


def paired_donors(forecast, labels):
    valid = labels.dropna(subset=["endpoint_value"])
    joined = join_observations(forecast[forecast.row_id.isin(valid.row_id)], valid)
    donors = adult_donor_outcomes(joined.merge(
        valid[["row_id", "adult_age_confirmed", "donor_id"]], on="row_id", validate="many_to_one",
    ))
    # Preserve the exact invariant chemical means after averaging different numbers of organoids.
    for method in ("drug_mean", "chemical_mean"):
        anchor = forecast[forecast.method.eq(method)].groupby("compound_id").predicted_value
        if not anchor.nunique().eq(1).all():
            raise ValueError("Drug mean must be invariant across models")
        select = donors.method.eq(method)
        donors.loc[select, "predicted_value"] = donors.loc[select, "compound_id"].map(anchor.first())
    return donors


def summarize(donors, comparisons, strata=None):
    summary, per_drug, _ = axis_metrics(donors, comparisons)
    ci = {f"{a}/{b}": crossed_patient_drug_ci(donors, a, b) for a, b in comparisons}
    rows = donors[["row_id", "compound_id", "lineage_group"]].drop_duplicates()
    result = {"drugs": rows.compound_id.nunique(), "donors": rows.lineage_group.nunique(), "pairs": len(rows),
              "metrics": summary, "within_drug_ci95": ci}
    if strata is not None:
        result["strata"] = {name: per_drug[per_drug.compound_id.isin(ids)]
                            .groupby("method")[["spearman", "pearson"]].mean().dropna().to_dict("index")
                            for name, ids in strata.items() if ids}
    return result


def gates(result):
    output = {}
    patient = {(r["baseline"], r["metric"]): r["ci95"][0] for r in result["metrics"]["paired_contrasts"]}
    eligible = min(r["n_drugs"] for r in result["metrics"]["across_cell_scores"])
    for base in ("drug_mean", "matched_auc"):
        if eligible < 10:
            output[base] = "insufficient_support"
        else:
            intervals = result["within_drug_ci95"][f"matched_endpoint/{base}"]
            passing = (all(intervals.get(k, [-1])[0] > 0 for k in ("spearman", "pearson"))
                       and all(patient.get((base, k), -1) > -.005 for k in ("spearman", "ndcg_at_5")))
            output[base] = "pass" if passing else "fail"
    return output


def score(args):
    if checked_cold(args.cold, ("query",)) != COLD:
        raise ValueError("Unexpected LICOB registration")
    query = pd.read_parquet(args.cold / "query.parquet")
    prior = read_prior(args.prior, query)
    forecasts, seal = verify_forecast(args.run / "forecast.parquet")
    proof = seal["provenance"]
    if (proof["registration_sha256"], proof["cold_sha256"], proof["prior_sha256"]) != (
            registration(args.run / "inputs"), COLD, PRIOR):
        raise ValueError("Forecast and registered design disagree")
    if not forecasts[forecasts.track.eq("prior")].reset_index(drop=True).equals(prior.reset_index(drop=True)):
        raise ValueError("Archived AUC reference changed at freeze")
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    dose = pd.read_parquet(args.run / "inputs/dose.parquet")
    outcomes = read_outcomes(args.supplement, query, dose)
    scopes, pairs, counts = {}, {}, {}
    strata = compound_strata(query)
    historic = (("interaction_transfer", "chemical_mean"),)
    matched = (("matched_endpoint", "drug_mean"), ("matched_endpoint", "matched_auc"))
    for endpoint, labels in outcomes.items():
        counts[endpoint] = labels.loc[labels.adult_age_confirmed, "bound_status"].value_counts().to_dict()
        prior_donors = paired_donors(prior, labels)
        scopes[f"prior/{endpoint}"] = summarize(prior_donors, historic, strata)
        if endpoint in MATCHED:
            concordance = outcomes["auc"].dropna(subset=["endpoint_value"])[["row_id", "endpoint_value"]]
            concordance = query[KEY].merge(concordance, on="row_id").assign(
                predicted_value=lambda rows: rows.endpoint_value, method="observed_auc", track=endpoint, fold=0,
            ).drop(columns="endpoint_value")
            pairs[endpoint] = summarize(paired_donors(concordance, labels), ())
            fitted = forecasts[forecasts.track.eq(endpoint)]
            if not fitted.empty:
                panel = query[query.compound_id.isin(proof["panels"][endpoint]["support"])]
                shared_support(fitted, panel, METHODS)
                result = summarize(paired_donors(fitted, labels), matched, compound_strata(panel))
                result["gate"] = gates(result)
                scopes[f"matched/{endpoint}"] = result
            else:
                scopes[f"matched/{endpoint}"] = {"gate": "insufficient_support"}
    by_drug = query[["compound_id", "source_compound_label"]].drop_duplicates().merge(dose, on="compound_id")
    publish_json(args.run / "score.json", {"forecast_sha256": seal["sha256"], "prior_sha256": PRIOR,
                 "labels_sha256": ASSET_HASHES["supplement"], "source_only_receipts": proof["extracts"],
                 "fit_panels": proof["panels"], "adult_organoid_response_status": counts,
                 "dose_design": by_drug.to_dict("records"), "outcome_concordance": pairs, "scopes": scopes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    for name, fields in {"freeze": ("priors", "targets", "auxiliary", "hallmark"), "score": ("supplement",)}.items():
        cli = subs.add_parser(name)
        for field in ("run", "prior", "cold", *fields):
            cli.add_argument("--" + field, required=True, type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
