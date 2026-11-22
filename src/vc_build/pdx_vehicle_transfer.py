"""Test untreated-adjusted pharmacological transfer on tissue-held Gao PDXs."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from openpyxl import load_workbook

from .external import KEY, native_sensitivity
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import checked
from .licob_prism_restored import OLD_INPUT
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda, domain_kernels
from .v0plus_cv import molecular_inputs
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

WORKBOOK = "c4b9a6903a4d1f76e3ddca4199039776d56bb99970aa5b7abe4f3abd732a0c6d"
CHEMISTRY = "e68701450b5e11e3b5635d7483fd57516e15de35c116585ecbcbd04acd463a"
TISSUES = ("BRCA", "CRC", "NSCLC", "PDAC", "CM")
DAY = 14
PENALTY = 4.0
MIN_FIT = 20
MIN_HELD = 5
PRIMARY = "blend_adjusted"
REFERENCES = ("drug_mean", "source_2d", "blend_unadjusted")
METHODS = (*REFERENCES, "pdx_unadjusted", "pdx_adjusted", PRIMARY)
FIT_COLUMNS = [*KEY, "tissue", "native", "adjusted"]


def read_table(sheet, columns):
    rows = sheet.values
    headings = next(rows)
    idx = [headings.index(name) for name in columns]
    return pd.DataFrame.from_records((tuple(row[i] for i in idx) for row in rows), columns=columns)


def growth_curve(rows):
    """Log relative volume at day 14, using only curves with baseline and bracketing observations."""
    if (not np.isfinite(rows.day).all() or (rows.day < 0).any() or rows.day.duplicated().any()
            or 0 not in rows.day.values or rows.day.max() < DAY):
        return np.nan
    if not np.isfinite(rows.volume).all() or (rows.volume <= 0).any():
        return np.nan
    ordered = rows.sort_values("day")
    log_volumes = np.log(ordered.volume.to_numpy())
    return float(np.interp(DAY, ordered.day, log_volumes) - log_volumes[0])


def response_rows(book, names):
    cols = ["Model", "Tumor Type", "Treatment", "Volume (mm3)", "Days Post T0"]
    raw = read_table(book["PCT raw data"], cols)
    raw.columns = ["model", "tissue", "arm", "volume", "day"]
    raw.volume = pd.to_numeric(raw.volume, errors="coerce")
    raw.day = pd.to_numeric(raw.day, errors="coerce")
    raw = raw[raw.model.isin(names) & raw.tissue.isin(TISSUES)]
    tissue = raw[["model", "tissue"]].drop_duplicates()
    if not tissue.model.is_unique:
        raise ValueError("PDX ID maps to multiple primary cancer types")

    metric = read_table(book["PCT curve metrics"], ["Model", "Treatment", "Treatment type"])
    metric.columns = ["model", "arm", "kind"]
    metric = metric[metric.kind.eq("single")].drop(columns="kind").drop_duplicates()
    records = [{"model": model, "arm": arm, "growth": growth_curve(rows)}
               for (model, arm), rows in raw.groupby(["model", "arm"], sort=True)]
    growing = pd.DataFrame(records).dropna(subset=["growth"])
    control = growing[growing.arm.eq("untreated")].set_index("model").growth
    eligible = growing[~growing.arm.eq("untreated")].merge(metric, on=["model", "arm"], validate="one_to_one")
    eligible["control"] = eligible.model.map(control)
    eligible = eligible.dropna(subset=["control"]).merge(tissue, on="model", validate="many_to_one")
    return eligible, {"rna_available_models": len(names), "raw_models_by_tissue": tissue.tissue.value_counts().to_dict(),
                      "unambiguous_day14_vehicle_models": len(control)}


def source_links(registry, source):
    """Use exact parent if measured and otherwise three nearby source compounds."""
    compounds = source.groupby("compound_id").agg(n=("lineage_group", "nunique"),
                                                    smiles=("isomeric_smiles", "first"))
    compounds = compounds[compounds.n.ge(100)].sort_index()
    ids = list(compounds.index)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    vectors = [generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in compounds.smiles]
    links = {}
    for item in registry.values():
        parent = item["parent"]
        if parent in compounds.index:
            links[parent] = {parent: 1.}
            continue
        probe = generator.GetFingerprint(Chem.MolFromSmiles(item["smiles"]))
        sim = DataStructs.BulkTanimotoSimilarity(probe, vectors)
        nearest = sorted(zip(ids, sim), key=lambda x: (-x[1], x[0]))[:3]
        links[parent] = {key: float(score) for key, score in nearest if score >= .25}
    return links


def register(args):
    require_hash(args.workbook, WORKBOOK)
    require_hash(args.chemistry, CHEMISTRY)
    if checked(args.source, ("train", "rna")) != OLD_INPUT:
        raise ValueError("Unexpected 2D viability reference source")
    registry = json.loads(args.chemistry.read_text())
    chemistry = pd.DataFrame.from_dict(registry, orient="index")[["parent"]].rename_axis("arm").reset_index()
    reference = pd.read_parquet(args.source / "train.parquet", columns=[
        *KEY, "source_id", "endpoint_id", "endpoint_value", "endpoint_direction", "isomeric_smiles",
    ])
    links = source_links(registry, reference)
    first_rna = pd.read_parquet(args.source / "rna.parquet")
    # The frozen RNA bundle also contains LICOB profiles. Use only source RNA rows to fit.
    first_rna = first_rna.loc[sorted(reference.cell_id.unique())]
    with closing(load_workbook(args.workbook, read_only=True, data_only=True)) as book:
        rna = book["RNAseq_fpkm"].values
        header = next(rna)
        if header[0] != "Sample" or len(header) != 400 or len(set(header)) != 400:
            raise ValueError("Unexpected PDX basal FPKM roster")
        observations, info = response_rows(book, set(header[1:]))
        donors = observations.merge(chemistry, on="arm", validate="many_to_one")
        obs = donors.assign(compound_id=donors.parent, native=-donors.growth,
                            adjusted=donors.control - donors.growth, cell_id="PDXE:" + donors.model)
        obs = obs.assign(lineage_group=obs.cell_id, row_id=obs.cell_id + "|" + obs.compound_id)
        if not obs.row_id.is_unique:
            raise ValueError("More than one registered regimen for the same PDX and exact parent")
        columns = [(pos, name) for pos, name in enumerate(header) if name in set(donors.model)]
        features = {name: tuple(row[pos] for pos, _ in columns)
                    for row in rna if (name := row[0]) in first_rna.columns}
    feature = pd.DataFrame(features, index=["PDXE:" + name for _, name in columns], dtype=np.float32)
    genes = first_rna.columns.intersection(feature.columns)
    if len(genes) < 12_000 or len(obs) < 500 or not np.isfinite(feature[genes].to_numpy()).all():
        raise ValueError("Insufficient shared clinical source support")
    both = pd.concat([first_rna[genes], feature[genes]], verify_integrity=True)
    resolved = {str(k): links[k] for k in sorted(obs.compound_id.unique())}
    source_drugs = set().union(*(set(donors) for donors in resolved.values()))
    ref = reference[reference.compound_id.isin(source_drugs)].copy()
    ref["native"] = native_sensitivity(ref)
    support = (obs.groupby(["tissue", "arm"]).size().unstack(fill_value=0).astype(int).to_dict("index"))
    with new_output(args.out) as output:
        roster = obs[[*KEY, "tissue"]]
        roster.to_parquet(output / "roster.parquet")
        both.to_parquet(output / "rna.parquet")
        ref[[*KEY, "source_id", "native"]].to_parquet(output / "reference.parquet")
        for name in TISSUES:
            obs.loc[obs.tissue.eq(name), FIT_COLUMNS].to_parquet(output / f"{name}.parquet")
        publish_json(output / "receipt.json", {
            "workbook_sha256": WORKBOOK, "chemistry_sha256": CHEMISTRY, "source_sha256": OLD_INPUT,
            "day": DAY, "pdx_models": obs.cell_id.nunique(), "pdx_observations": len(obs),
            "pdx_compounds": obs.compound_id.nunique(), "shared_rna_genes": len(genes),
            "pdx_raw_support": info, "matched_tissue_by_regimen": support,
            "registered_but_no_match": sorted(set(registry) - set(obs.arm)),
            "source_models": ref.cell_id.nunique(), "source_observations": len(ref), "source_links": resolved,
            "files": {p.name: sha256(p) for p in sorted(output.glob("*.parquet"))},
        })


def load_registered(root, names):
    receipt = json.loads((root / "receipt.json").read_text())
    if (receipt["workbook_sha256"], receipt["chemistry_sha256"], receipt["source_sha256"], receipt["day"]) != (
            WORKBOOK, CHEMISTRY, OLD_INPUT, DAY):
        raise ValueError("Unexpected PDX outcome version")
    for name in names:
        require_hash(root / name, receipt["files"][name])
    return [pd.read_parquet(root / name) for name in names], sha256(root / "receipt.json"), receipt


def standardized_rank(frame, value, groups):
    rank = frame.groupby(groups)[value].rank(pct=True, method="average")
    strata = [frame[g] for g in groups]
    center = rank.groupby(strata).transform("mean")
    spread = rank.groupby(strata).transform("std")
    return (rank - center) / spread.replace(0, np.nan)


def forecast_residuals(train, response, context, rna, query, edges, targets, device):
    cells = pd.Index(sorted(train.cell_id.unique()))
    held = pd.Index(sorted(query.cell_id.unique()))
    cell, cross = domain_kernels(rna, cells, held, edges, targets, device)
    table = train.assign(ranks=standardized_rank(train, response, [context, "compound_id"]))
    summary = table.groupby(["compound_id", "cell_id"]).ranks.mean()
    parts = []
    for compound, actual in query.groupby("compound_id", sort=True):
        vals = summary.loc[compound].dropna()
        if len(vals) < MIN_FIT:
            raise ValueError("Response ranks lost the registered fit support")
        selected = torch.as_tensor(cells.get_indexer(vals.index), device=device)
        kernel = cell[selected[:, None], selected]
        y = torch.as_tensor(vals.to_numpy(), device=device, dtype=kernel.dtype)
        beta = torch.linalg.solve(kernel + PENALTY * torch.eye(len(kernel), device=device), y)
        prediction = (cross[:, selected] @ beta).cpu().numpy()
        parts.append(pd.Series(prediction[held.get_indexer(actual.cell_id)], index=actual.index))
    return pd.concat(parts).loc[query.index]


def source_forecast(source, query, links, rna, edges, targets, device):
    results = pd.Series(0., index=query.index)
    matches = [{key: (weight / sum(links[drug].values())) for key, weight in links[drug].items()}
               for drug in query.compound_id]
    queries = []
    for i, row in query.iterrows():
        queries.extend(dict(cell_id=row.cell_id, compound_id=source_id, weight=weight, query_index=i)
                       for source_id, weight in matches[i].items())
    if queries:
        rows = pd.DataFrame(queries)
        source = source[source.compound_id.isin(rows.compound_id)]
        predictions = forecast_residuals(source, "native", "source_id", rna, rows, edges, targets, device)
        weighted = (rows.weight * predictions).groupby(rows.query_index).sum()
        results.loc[weighted.index] = weighted
    return results, source.cell_id.nunique()


def predict(args):
    hold = args.tissue
    labels = tuple(f"{t}.parquet" for t in TISSUES if t != hold)
    (rna, roster, source, *parts), registered, receipt = load_registered(
        args.registered, ("rna.parquet", "roster.parquet", "reference.parquet", *labels),
    )
    fit = pd.concat(parts, ignore_index=True)
    query = roster[roster.tissue.eq(hold)]
    if set(query.lineage_group) & set(fit.lineage_group):
        raise ValueError("Tissue holdout has overlapping PDXs")
    pop = fit.groupby("compound_id").lineage_group.nunique()
    fit_groups = fit.groupby(["compound_id", "tissue"]).size()
    supported = fit_groups[fit_groups.ge(MIN_HELD)].groupby(level=0).size()
    proposed = query.groupby("compound_id").size()
    common = set(pop.index[pop.ge(MIN_FIT)]) & set(proposed.index[proposed.ge(MIN_HELD)]) & set(supported.index)
    query = query[query.compound_id.isin(common)].reset_index(drop=True)
    fit = fit[fit.compound_id.isin(common)]
    if len(common) < 4:
        raise ValueError(f"Insufficient supported drugs for tissue {hold}: {sorted(common)}")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    device = device_cuda()
    anchor = query.compound_id.map(fit.groupby("compound_id").adjusted.mean())
    scale = query.compound_id.map(fit.groupby("compound_id").adjusted.std())
    source_residual, source_models = source_forecast(source, query, receipt["source_links"], rna, edges, targets,
                                                     device)
    residual = {name: forecast_residuals(fit, response, "tissue", rna, query, edges, targets, device)
                for name, response in (("pdx_unadjusted", "native"), ("pdx_adjusted", "adjusted"))}
    residual["source_2d"] = source_residual
    residual["blend_unadjusted"] = .5 * (residual["source_2d"] + residual["pdx_unadjusted"])
    residual[PRIMARY] = .5 * (residual["source_2d"] + residual["pdx_adjusted"])
    methods = {"drug_mean": anchor} | {name: anchor + scale * estimates for name, estimates in residual.items()}
    rows = pd.concat([query[KEY].assign(method=name, predicted_value=values) for name, values in methods.items()])
    shared_support(rows, query, METHODS)
    freeze_table(rows.assign(track=hold, fold=0), args.out, provenance={
        "scope": hold, "registration_sha256": registered, "features": priors,
        "model_support": query.cell_id.nunique(), "compound_support": sorted(common),
        "source_2d_fit_models": source_models, "other_pdx_fit_models": fit.cell_id.nunique(),
        "source_2d_exact_parents": sorted(key for key in common if key in receipt["source_links"][key]),
    })


def score(args):
    result, hashes = {}, {}
    for name in TISSUES:
        (observed,), receipt, _ = load_registered(args.registered, (f"{name}.parquet",))
        rows, seal = verify_forecast(args.forecasts / f"{name}.parquet")
        meta = seal["provenance"]
        if (meta["registration_sha256"], meta["scope"]) != (receipt, name):
            raise ValueError("Unexpected PDX tissue forecast")
        observed = observed[observed.compound_id.isin(meta["compound_support"])]
        shared_support(rows, observed, METHODS)
        joined = rows.merge(observed[[*KEY, "adjusted", "native"]], on=KEY, validate="many_to_one")
        comparisons = [(PRIMARY, ref) for ref in REFERENCES]
        def metrics(f):
            report, per_drug, _ = axis_metrics(f, comparisons)
            return report | {"observations": f.row_id.nunique(),
                             "per_drug": per_drug.astype(object).where(per_drug.notna(), None).to_dict("records")}
        main = joined.assign(endpoint_value=joined.adjusted)
        overall = metrics(main)
        panel = main[main.row_id.isin(main[["row_id", "cell_id"]].drop_duplicates().groupby("cell_id").row_id
                                     .filter(lambda ids: len(ids) >= 6))]
        panel_score = metrics(panel) if len(panel) else {"observations": 0, "paired_contrasts": []}
        intervals = {ref: crossed_patient_drug_ci(main, PRIMARY, ref, repeats=800) for ref in REFERENCES}
        full = {(r["baseline"], r["metric"]): r["ci95"][0] for r in panel_score["paired_contrasts"]}
        gates = {ref: len(ci) == 2 and all(bounds[0] > 0 for bounds in ci.values())
                 and panel.cell_id.nunique() >= 10
                 and all(full.get((ref, m), -1.) > -.005 for m in ("spearman", "ndcg_at_5", "top5_overlap"))
                 for ref, ci in intervals.items()}
        corr = {m: part.adjusted.corr(part.native, method="spearman")
                for m, part in observed.groupby("compound_id")}
        exact = main[main.compound_id.isin(meta["source_2d_exact_parents"])]
        result[name] = {"adjusted_outcome": overall, "exact_source_support": metrics(exact) if len(exact) else None, "native_outcome": metrics(joined.assign(endpoint_value=joined.native)),
                        "true_panel": panel_score, "gates": gates, "paired_ci95": intervals,
                        "pdxs": observed.cell_id.nunique(), "drugs": len(meta["compound_support"]),
                        "eligible_ranking_pdxs": panel.cell_id.nunique(), "observed_native_adjusted_spearman": corr}
        hashes[name] = seal["sha256"]
    n = sum(all(value["gates"].values()) for value in result.values())
    publish_json(args.out, {"pass": n >= 3, "tissues_passed": n, "forecast_sha256": hashes, "results": result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    stages = {
        "register": ("workbook", "chemistry", "source"),
        "predict": ("registered", "priors", "hallmark", "targets"),
        "score": ("registered", "forecasts"),
    }
    for action, fields in stages.items():
        sub = subs.add_parser(action)
        for field in (*fields, "out"):
            sub.add_argument("--" + field, type=Path, required=True)
        if action == "predict":
            sub.add_argument("--tissue", choices=TISSUES, required=True)
    args = parser.parse_args()
    {"register": register, "predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
