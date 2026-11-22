"""Test transfer from growth-unadjusted 2D screens to published HCC GR50."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_source_data import MODEL_COLS, parse_cellosaurus_identity, read_raw, related_cellosaurus
from .expanded_panel_cv import RECEIPTS
from .external import KEY, SOURCE, _validate
from .function_data import alias_token, new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .limore_corrected import shared_support
from .measured_driver_transfer import fit_reference
from .pdo_transfer_cv import device_cuda
from .v0plus_cv import molecular_inputs, panel_targets, training_inputs
from .yang_hcc_external import PENALTY, SETTINGS, require_hash

COHORT = "SAGMEISTER_ADULT_HCC"
ARTICLE = "4aa52d413a0a7732caccfcce6b837d167c6b83f5427c3562c16d6d5ca297842d"
GROWTH = "cdbc9f3c6c7b2388ade1780bc93feb6579f6868b73ba3dbc80dfb57d60c0a4dc"
MODELS = ("Hep3B", "HepG2", "HLE", "HLF", "HuH1", "HuH7", "PLC-PRF5", "Snu398", "Snu475")
PANEL = {
    "Sorafenib": ("MLDQJTXFUGDVEO-UHFFFAOYSA-N", .31, 5.),
    "Lenvatinib": ("WOSKHXYHFSIKNG-UHFFFAOYSA-N", .25, 20.),
    "Regorafenib": ("FNHKPVJBJVTLMP-UHFFFAOYSA-N", .31, 5.),
    "Cabozantinib": ("ONIQOQHATWINJY-UHFFFAOYSA-N", 1.25, 20.),
}
PROFILE = ("structured", "growth_adjusted", "growth_shuffled")
COMPONENTS = {
    "structured": PROFILE,
    "linear": ("drug_mean", "drug_ridge"),
    **{name: (name,) for name in ("cellhit", "velodrome_inductive", "codeae_inductive", "v0plus_inductive")},
}


def training(core, held):
    require_hash(core / "receipt.json", RECEIPTS["core"])
    rows, rna, _ = training_inputs(core)
    fit = rows[~rows.lineage_group.isin(held)]
    return fit, rna.loc[sorted(fit.cell_id.unique())], rows


def protected_lineages(registry, models, cells, rows):
    full = registry[registry.family_id.isin(cells.family_id)]
    seed = set(full.cellosaurus_id) | set(full.root_ancestor)
    accessions, aliases, _ = related_cellosaurus(seed - {""}, *models)
    matched = (registry[registry.cellosaurus_id.isin(accessions)].depmap_model_id.tolist()
               + list(aliases))
    seed_rows = rows[rows.cell_id.isin(set(matched) | set(full.depmap_model_id))]
    return set(seed_rows.lineage_group) | set(full.depmap_patient_id) - {""}


def growth_data(path, model_file, fit):
    require_hash(path, GROWTH)
    g = pd.read_csv(path, dtype={"model_id": str})
    valid = g[g.replicates.ge(3) & g.day4_day1_ratio.gt(1) & np.isfinite(g.day4_day1_ratio)]
    values = np.log2(valid.set_index("model_id").day4_day1_ratio).groupby(level=0).median()
    depmap = read_raw(model_file, MODEL_COLS)
    known = depmap[depmap.ModelID.isin(fit.cell_id)].set_index("ModelID", verify_integrity=True)
    return known.SangerModelID.map(values).dropna().rename("log2_ratio")


def register(args):
    require_hash(args.original / "receipt.json", RECEIPTS["original"])
    evidence = json.loads((args.original / "receipt.json").read_text())["inputs"]
    for name in ("depmap_models", "cellosaurus"):
        require_hash(Path(evidence[name]["path"]), evidence[name]["sha256"])
    for name in ("cell_registry.tsv", "rna.parquet"):
        require_hash(args.limore / name, evidence["limore_bundle"]["artifacts"][name]["sha256"])
    registry = pd.read_csv(args.limore / "cell_registry.tsv", sep="\t", dtype=str).fillna("")
    model_names = {alias_token(name): name for name in MODELS}
    roster = registry[registry.source_name.map(alias_token).isin(model_names)].copy()
    if len(roster) != len(MODELS):
        raise ValueError("Unexpected source model roster")
    models = parse_cellosaurus_identity(evidence["cellosaurus"]["path"])
    _, _, all_rows = training(args.core, ())
    held = protected_lineages(registry, models, roster, all_rows)
    train, rna, _ = training(args.core, held)
    query = roster[roster.included.eq("True")].rename(columns={"model_id": "cell_id", "family_id": "lineage_group"})
    query = query[["cell_id", "source_name", "lineage_group", "depmap_model_id"]].assign(
        published_cell=query.source_name.map(alias_token).map(model_names))
    if len(query) != 7 or query.lineage_group.nunique() != 6:
        raise ValueError("Unexpected adult HCC lineage cohort")
    assay_rna = pd.read_parquet(args.limore / "rna.parquet").loc[sorted(query.cell_id)]
    genes = sorted(set(assay_rna.columns) & set(rna.columns))
    panel = pd.DataFrame([(drug, *identity) for drug, identity in PANEL.items()],
                         columns=["source_compound_label", "compound_id", "low_uM", "high_uM"])
    chemical = train[train.compound_id.isin(panel.compound_id)][["compound_id", "isomeric_smiles"]].drop_duplicates()
    if not chemical.compound_id.is_unique or len(chemical) != len(PANEL) or len(genes) < 16000:
        raise ValueError("Missing registered exact drug structure or expression support")
    query = query.merge(panel.merge(chemical, on="compound_id", validate="one_to_one"), how="cross")
    query = query.assign(row_id=COHORT + "|" + query.cell_id + "|" + query.compound_id)
    _validate(train, query)
    if set(roster.depmap_model_id) & set(train.cell_id):
        raise ValueError("The same source cell is in fit and holdout")
    growth = growth_data(args.growth, evidence["depmap_models"]["path"], train)
    if len(growth) < 200:
        raise ValueError("Insufficient independently measured fit-only proliferation")
    with new_output(args.out) as work:
        frames = {"query": query, "rna": assay_rna[genes], "growth": growth.to_frame()}
        for name, frame in frames.items():
            frame.to_parquet(work / f"{name}.parquet", index=name != "query")
        publish_json(work / "receipt.json", {
            "cohort": COHORT, "core_sha256": RECEIPTS["core"], "growth_sha256": GROWTH, "pdf_sha256": ARTICLE,
            "limore_registry_sha256": evidence["limore_bundle"]["artifacts"]["cell_registry.tsv"]["sha256"],
            "limore_rna_sha256": evidence["limore_bundle"]["artifacts"]["rna.parquet"]["sha256"],
            "limore_response_sha256": evidence["limore_bundle"]["artifacts"]["rows.parquet"]["sha256"],
            "protected_lineages": sorted(held), "fit_cells": int(train.cell_id.nunique()), "fit_rows": len(train),
            "excluded_fit_rows": len(all_rows) - len(train), "fit_growth_cells": len(growth),
            "fit_growth_adult_hcc_cells": len(set(growth.index) & set(train.loc[train.lineage_role.eq("adult_hcc"), "cell_id"])),
            "genes": len(genes), "adult_models": 7, "adult_lineages": 6, "drugs": len(PANEL),
            "files": {name: sha256(work / f"{name}.parquet") for name in frames},
        })


def registration(path):
    paper = path / "receipt.json"
    doc = json.loads(paper.read_text())
    if (doc["cohort"], doc["core_sha256"], doc["growth_sha256"], doc["pdf_sha256"]) != (
            COHORT, RECEIPTS["core"], GROWTH, ARTICLE):
        raise ValueError("Unexpected independent GR50 cohort")
    for name, digest in doc["files"].items():
        require_hash(path / f"{name}.parquet", digest)
    return doc, sha256(paper)


def subtract_growth(rows, growth):
    fixed = rows.residual.copy()
    known = rows.cell_id.map(growth)
    counts = []
    for _, group in rows[known.notna()].groupby([*SOURCE, "compound_id"]):
        if group.cell_id.nunique() < 30:
            continue
        x = known.loc[group.index].to_numpy()
        x = x - x.mean()
        slope = float(x @ group.residual / (x @ x + 50))
        fixed.loc[group.index] -= x * slope
        counts.append(slope)
    return rows.assign(residual=fixed), {
        "adjusted_assay_drug_groups": len(counts),
        "absolute_slope_quantiles": np.quantile(np.abs(counts), [.25, .5, .75]).tolist() if counts else [],
    }


def structure(args, train, query, fit, held, edges, targets, growth):
    from .structured_kernel import KernelGrid, solve_residuals, training_grid
    from .v0plus_neural import load_auxiliary

    aux, receipt = load_auxiliary(args.auxiliary)
    device = device_cuda()
    _, anchor, rows, calibration = panel_targets(train)
    # Use unique indexes for all per-assay operations.
    rows = rows.reset_index(drop=True)
    grid = KernelGrid.prepare(rows, fit, held, edges, targets, aux, device=device)
    usage = train[["cell_id", "tissue_origin"]].drop_duplicates().set_index("cell_id").tissue_origin
    growth = (growth - growth.mean()) / growth.std(ddof=0)
    rng = np.random.default_rng(43675)
    shuffled = growth.groupby(usage).transform(lambda part: rng.permutation(part.to_numpy()))
    known, first = subtract_growth(rows, growth)
    null, second = subtract_growth(rows, shuffled)
    cell, cross = grid.cell["full"]
    gram = grid.drug["structured"]
    where = held.index.get_indexer(query.cell_id), grid.drugs.get_indexer(query.compound_id)
    base = query.compound_id.map(anchor).to_numpy()
    records, diagnostics = [], {}
    for method, data in zip(PROFILE, (rows, known, null)):
        weight, response = training_grid(data, fit.index, grid.drugs, device=device)
        coef, steps, norm = solve_residuals(cell, gram, weight, response, PENALTY)
        value = base + (cross @ coef @ gram).cpu().numpy()[where]
        records.append(query[KEY].assign(method=method, predicted_value=value))
        diagnostics[method] = {"iterations": steps, "residual_norm": norm}
    return pd.concat(records, ignore_index=True), {"solves": diagnostics, "growth": first, "shuffled": second,
        "growth_fraction_changed_by_permutation": float(shuffled.ne(growth).mean()),
        "calibration": calibration, "auxiliary": receipt}


def predict(args):
    info, digest = registration(args.registration)
    require_hash(args.settings, SETTINGS)
    train, fit, _ = training(args.core, info["protected_lineages"])
    query = pd.read_parquet(args.registration / "query.parquet")
    held = pd.read_parquet(args.registration / "rna.parquet")
    fit = fit[held.columns]
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    _validate(train, query)
    if args.component == "structured":
        growth = pd.read_parquet(args.registration / "growth.parquet").log2_ratio
        if not set(growth.index) <= set(fit.index):
            raise ValueError("Held-cell growth values entered fit-only inputs")
        result, notes = structure(args, train, query, fit, held, edges, targets, growth)
    else:
        result, _ = fit_reference(args, train, query, fit, held, edges, targets)
        notes = {}
    shared_support(result, query, COMPONENTS[args.component])
    freeze_table(result.assign(track=COHORT, fold=0), args.out,
                 provenance={"registration_sha256": digest, "settings_sha256": SETTINGS,
                             "component": args.component, "priors": priors, "details": notes})


def combine(args):
    _, digest = registration(args.registration)
    query = pd.read_parquet(args.registration / "query.parquet")
    parts, hashes = [], {}
    for component, methods in COMPONENTS.items():
        rows, stamp = verify_forecast(args.parts / f"{component}.parquet")
        record = stamp["provenance"]
        if (record["registration_sha256"], record["settings_sha256"], record["component"]) != (
                digest, SETTINGS, component):
            raise ValueError("Unexpected GR50 forecast registration")
        shared_support(rows, query, methods)
        parts.append(rows)
        hashes[component] = stamp["sha256"]
    freeze_table(pd.concat(parts, ignore_index=True), args.out, provenance={
        "registration_sha256": digest, "article_sha256": ARTICLE, "component_sha256": hashes,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "register": ("original", "core", "limore", "growth"),
        "predict": ("registration", "core", "settings", "priors", "targets", "hallmark", "auxiliary", "upstream"),
        "freeze": ("registration", "parts"),
        "score": ("registration", "forecast", "article", "values", "limore"),
    }
    for stage, params in fields.items():
        sub = stages.add_parser(stage)
        for name in (*params, "out"):
            sub.add_argument("--" + name, type=Path, required=True)
        if stage == "predict":
            sub.add_argument("--component", choices=COMPONENTS, required=True)
            sub.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.stage == "score":
        from .tki_gr50_score import score

        score(args)
    else:
        {"register": register, "predict": predict, "freeze": combine}[args.stage](args)


if __name__ == "__main__":
    main()
