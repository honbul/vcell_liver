"""Register public pancreatic PDO pharmacology and test joint 2D/3D HCC transfer."""

import argparse
from collections import Counter
from functools import partial
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem
import torch

from .expanded_panel_cv import PENALTY, RECEIPTS
from .external import KEY, SOURCE, _validate
from .function_data import canonical_chemistry, new_output
from .io import freeze_table, publish_json, sha256
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference,
)
from .rna_state_projection import score_external
from .structured_kernel import dot_kernel, drug_kernels, fit_cell_kernels, radial, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs, panel_targets
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

DATA = {
    "data1.xlsx": "a89e694d1271454272d237e60d087cd3954a6e03cc27f188a8ad8920bb97f621",
    "data7.xlsx": "8741a4c5be80e6229f58c6c4450fd3ccb3464cfa9d7273615b424d48156c50b8",
    "data8.xlsx": "68843d04c8871619ab3c1666dea5d8e5d90bdffc64b0ab41e1f27d21ff0e3503",
    "rna.txt.gz": "479517fbed1d8f5bb9d141b6b932b6ab56ba2ba72d4394d13aea2a3da0a50f27",
    "series.soft.gz": "81a6c210d510a3a3d94525ff84888e367497bc2fd477df2dcc2b2877cd0739e1",
    "catalog.xlsx": "95092a4577933781ca3e96b8015eb8133f07bd2a1fbac2e15696b59b801d6c57",
}
COMMON = "pdo_common_axis_2d"
PRIMARY = "pdo_joint_domain"
POOLED = "pdo_joint_pooled"
SHUFFLED = "pdo_joint_donor_shuffle"
EXACT = "pdo_joint_exact_parent"
ARMS = (COMMON, PRIMARY, POOLED, SHUFFLED, EXACT)
PDO_MASS = 100.
PDO_DOMAIN_VARIANCE = 1.


def formula_counts(formula):
    counts = Counter()
    for section in re.sub(r"\s", "", formula).split("."):
        segment = re.fullmatch(r"(\d*)(.*)", section)
        factor, formula = segment.groups()
        terms = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        if not terms or "".join(a+b for a,b in terms) != formula:
            raise ValueError(f"Invalid published formula: {section}")
        for element, number in terms:
            counts[element] += int(number or 1) * int(factor or 1)
    return counts


def unspecified(smiles):
    return any(site.specified == Chem.StereoSpecified.Unspecified
               for site in Chem.FindPotentialStereo(Chem.MolFromSmiles(smiles)))


def chemistry(data, assets):
    original = pd.read_excel(data / "data7.xlsx", sheet_name="Secondary screening", header=1)
    vendor = pd.read_excel(data / "catalog.xlsx", sheet_name=1,
                           usecols=["Cat", "SMILES", "Formula", "CAS Number", "URL"])
    roster = original.merge(vendor, left_on="Catalog Number", right_on="Cat", how="left",
                            suffixes=("_screen", "_vendor"), validate="one_to_one")
    if len(roster) != 64:
        raise ValueError("Published secondary screen drug count changed")
    rows = []
    for _, row in roster.iterrows():
        code, cas = row["Catalog Number"], row["CAS Number"]
        item = {"catalogue": code, "label": row["Product Name"].strip(),
                "highest_reported": row["Maximum concentration"], "cas": None if pd.isna(cas) else str(cas)}
        if pd.isna(row.SMILES):
            rows.append(item | {"status": "absent_from_catalog"})
            continue
        if formula_counts(row.Formula_screen) != formula_counts(row.Formula_vendor):
            rows.append(item | {"status": "formula_discrepancy"})
            continue
        smiles, key, connectivity, _ = canonical_chemistry(row.SMILES)
        evidence = row.URL
        if unspecified(smiles):
            path = data / "pug" / f"{cas}.json"
            require_hash(path, assets["pubchem_sha256"][path.name])
            props = json.loads(path.read_text())["PropertyTable"]["Properties"]
            choices = {canonical_chemistry(p["SMILES"]) for p in props}
            choices = {v for v in choices if v[2] == connectivity and not unspecified(v[0])}
            if len(choices) != 1:
                rows.append(item | {"status": "unresolved_stereochemistry"})
                continue
            smiles, key, _, _ = choices.pop()
            evidence = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/SMILES,InChIKey/JSON"
        rows.append(item | {"status": "eligible", "compound_id": key, "isomeric_smiles": smiles, "evidence": evidence})
    table = pd.DataFrame(rows).sort_values("catalogue", ignore_index=True)
    duplicate = table.status.eq("eligible") & table.compound_id.duplicated()
    return table.assign(status=table.status.mask(duplicate, "duplicate_parent"))


def register(args):
    assets = json.loads(args.assets.read_text())
    for name, digest in DATA.items():
        require_hash(args.data / name, digest)
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    catalog = chemistry(args.data, assets)
    eligible = catalog[catalog.status.eq("eligible")].set_index("catalogue")
    meta = pd.read_excel(args.data / "data1.xlsx", header=1)
    auc = pd.read_excel(args.data / "data8.xlsx", header=1).set_index("Drug list", verify_integrity=True)
    expr = pd.read_csv(args.data / "rna.txt.gz", sep="\t").set_index("GeneSymbol", verify_integrity=True).T
    with gzip.open(args.data / "series.soft.gz", "rt") as handle:
        titles = {line.split(" = ", 1)[1].strip() for line in handle if line.startswith("!Sample_title = ")}
    if set(auc.index) != set(catalog.catalogue) or not set(auc.columns) <= set(expr.index) & titles:
        raise ValueError("GEO and supplemental study identities do not match")
    participants = meta[meta["Sample ID"].isin(auc.columns) & meta["Tumorr Type Detailed"].eq("Pancreatic Adenocarcinoma")]
    if len(participants) != 38 or not participants["Number of Samples Per Patient"].eq(1).all():
        raise ValueError("Expected 38 patient-distinct PDAC models")
    if not participants["Diagnosis Age (y)"].ge(18).all() or not participants["Organoid transcriptomic sequence"].eq("Yes").all():
        raise ValueError("Incomplete registration of adult baseline RNA")
    cells = sorted(participants["Sample ID"])
    response = auc.loc[eligible.index, cells].set_axis(eligible.compound_id).T
    rna = expr.loc[cells].astype(np.float32)
    if not np.isfinite(rna.to_numpy()).all() or not np.isfinite(response.to_numpy()).all() or response.lt(0).any().any():
        raise ValueError("Unusable pre-treatment expression or normalized AUC")
    ids = [f"GSE194249:{cell}" for cell in cells]
    with new_output(args.out) as out:
        rna.set_axis(ids).to_parquet(out / "rna.parquet")
        response.set_axis(ids).to_parquet(out / "auc.parquet")
        eligible.reset_index().to_parquet(out / "chemistry.parquet")
        catalog.to_csv(out / "catalogue_audit.tsv", sep="\t", index=False)
        publish_json(out / "receipt.json", {
            "input_files": DATA, "asset_manifest_sha256": sha256(args.assets),
            "core_sha256": RECEIPTS["core"], "geo_rna_samples": len(expr),
            "rna_genes": rna.shape[1], "auc_models": len(auc.columns),
            "adult_pdac_models": len(cells), "chemical_status": catalog.status.value_counts().to_dict(),
            "auc_drugs": response.shape[1], "training_observations": int(response.size),
            "files": {p.name: sha256(p) for p in sorted(out.iterdir())},
        })


def load_pdo(path):
    receipt = json.loads((path / "receipt.json").read_text())
    if receipt["input_files"] != DATA or receipt["adult_pdac_models"] != 38:
        raise ValueError("Unexpected public PDO cohort")
    frames = []
    for name in ("rna", "auc", "chemistry"):
        source = path / f"{name}.parquet"
        require_hash(source, receipt["files"][source.name])
        frames.append(pd.read_parquet(source))
    return (*frames, sha256(path / "receipt.json"))


def extend_cell_kernel(basis, other):
    """Apply the fit-only 2D RNA transform to auxiliary PDO cells and queries."""
    rna, pathways, auxiliary = (torch.cat((x, y), dim=0) for x, y in zip(basis.reference, basis.vectors(other)))
    linear = dot_kernel(rna, rna) / basis.reference_scale
    energy = linear.diagonal()
    return (.35 * (linear + radial(linear, energy, energy, 2.))
            + .20 * dot_kernel(pathways, pathways) + .10 * dot_kernel(auxiliary, auxiliary))


def pdo_training_grid(auc, chem, rows, training_cells, compounds, device):
    """Keep PDO mean shifts out of the label. Set a fixed per-drug 3D observation weight."""
    measured = -auc.to_numpy()
    average_scale = rows.groupby([*SOURCE, "compound_id"]).residual.std().groupby("compound_id").mean()
    scale = chem.compound_id.map(average_scale).fillna(average_scale.median()).clip(lower=.02).to_numpy()
    centered = measured - measured.mean(axis=0)
    standardized = centered / np.maximum(measured.std(axis=0), .05) * scale
    positions = compounds.get_indexer(chem.compound_id)
    if not auc.columns.equals(pd.Index(chem.compound_id)) or (positions < 0).any():
        raise ValueError("PDO training AUC and chemical registry are misaligned")
    y = torch.zeros((len(auc), len(compounds)), device=device)
    w = torch.zeros_like(y)
    y[:, positions] = torch.as_tensor(standardized, dtype=y.dtype, device=device)
    w[:, positions] = PDO_MASS / len(auc)
    old_w, old_y = training_grid(rows, training_cells, compounds, device=device)
    return torch.cat((old_w, w)), torch.cat((old_y, y))


def forecast(mode, fit, query, rna, pdo_rna, auc, chem, edges, targets, auxiliary):
    _validate(fit, query)
    device = device_cuda()
    genes = rna.columns.intersection(pdo_rna.columns).sort_values()
    native = rna.loc[sorted(fit.cell_id.unique()), genes]
    held = rna.loc[sorted(query.cell_id.unique()), genes]
    pdo = pdo_rna.loc[auc.index, genes]
    all_groups = pd.concat([native, pdo, held], verify_integrity=True).index
    if not all_groups.is_unique or len(genes) < 10000:
        raise ValueError("Unexpected PDO RNA support")
    basis, grams = fit_cell_kernels(native, edges, auxiliary, device=device)
    block = extend_cell_kernel(basis, pd.concat([pdo, held]))
    fit_end = len(native) + len(pdo)
    if (block[:len(native), :len(native)] - grams["full"]).abs().max().item() > 2e-5:
        raise ValueError("The extended basis changed the native subkernel")
    combined = block[:fit_end, :fit_end]
    cross = block[fit_end:, :fit_end]
    domain = torch.cat((torch.zeros(len(native), device=device), torch.ones(len(pdo), device=device)))
    domain_train = combined * (1 + PDO_DOMAIN_VARIANCE * domain[:, None] * domain)
    domain_cross = cross * (1 + PDO_DOMAIN_VARIANCE * domain * (mode == "LICOB"))
    _, anchor, rows, _ = panel_targets(fit)
    core_drug = pd.Index(sorted(anchor.index))
    chemical = pd.concat([rows[["compound_id", "isomeric_smiles"]],
                          chem[~chem.compound_id.isin(core_drug)][["compound_id", "isomeric_smiles"]]])
    compounds, kernels = drug_kernels(chemical, targets, device=device)
    drug = kernels["structured"]
    weights, responses = pdo_training_grid(auc, chem, rows, native.index, compounds, device)
    is_exact = torch.as_tensor(compounds.isin(core_drug), device=device)
    exact_weights = weights.clone()
    exact_weights[len(native):, ~is_exact] = 0.
    rng = np.random.default_rng(81141)
    order = rng.permutation(len(pdo))
    permuted = np.empty_like(order)
    permuted[order] = np.roll(order, 1)
    shuffled = torch.cat((responses[:len(native)], responses[len(native):][permuted]))
    key = held.index.get_indexer(query.cell_id), compounds.get_indexer(query.compound_id)
    anchor_values = query.compound_id.map(anchor).to_numpy()
    plain_shape = len(native)
    models = {
        COMMON: (combined[:plain_shape, :plain_shape], cross[:, :plain_shape], weights[:plain_shape], responses[:plain_shape]),
        POOLED: (combined, cross, weights, responses),
        PRIMARY: (domain_train, domain_cross, weights, responses),
        SHUFFLED: (domain_train, domain_cross, weights, shuffled),
        EXACT: (domain_train, domain_cross, exact_weights, responses),
    }
    output, trace = [], {}
    for name, (kernel, link, weight, response) in models.items():
        coef, steps, residual = solve_residuals(kernel, drug, weight, response, PENALTY)
        value = anchor_values + (link @ coef @ drug).cpu().numpy()[key]
        output.append(query[KEY].assign(method=name, predicted_value=value))
        trace[name] = {"iterations": steps, "solver_residual": residual}
    return pd.concat(output, ignore_index=True), {"genes": len(genes), "fit_pdac_cells": len(pdo),
        "pdo_drugs": auc.shape[1], "core_drugs_with_exact_pdo": len(core_drug.intersection(auc.columns)),
        "exact_query_drugs": len(pd.Index(query.compound_id.unique()).intersection(auc.columns)),
        "solver": trace}


def predict(args):
    pdo_rna, auc, chem, receipt = load_pdo(args.pdo)
    fit, query, rna, registered = external_inputs(args, args.mode)
    refs, reference = external_reference(args, args.mode, query)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    new, detail = forecast(args.mode, fit, query, rna, pdo_rna, auc, chem, edges, targets, auxiliary)
    all_methods = pd.concat([refs, new.assign(track=args.mode, fold=0)], ignore_index=True)
    shared_support(all_methods, query, (*BASELINES, *PUBLIC, *ARMS))
    freeze_table(all_methods[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": 0, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "pdo_receipt_sha256": receipt, "priors": priors, "auxiliary": aux,
        "diagnostics": detail,
    })


def assess(frame, *, external, exact):
    into = (*BASELINES, *PUBLIC, COMMON, POOLED, SHUFFLED, EXACT)
    overall, _, _ = axis_metrics(frame, [(PRIMARY, c) for c in into])
    across = {c: crossed_patient_drug_ci(frame, PRIMARY, c, repeats=1000) for c in into}
    within = {(x["baseline"], x["metric"]): x["ci95"] for x in overall["paired_contrasts"]}
    gates = {c: all(across[c][m][0] > 0 for m in ("spearman", "pearson"))
             and all(within[c, m][0] > -.005 for m in ("spearman", "ndcg_at_5", "top5_overlap"))
             for c in into}
    matched = frame[frame.compound_id.isin(exact)]
    support = {"drugs": matched.compound_id.nunique(), "observations": matched.row_id.nunique()}
    if support["drugs"] >= 6:
        support["metrics"] = axis_metrics(matched, ())[0]
    return {"metrics": overall, "across_cell_ci95": across, "gates": gates,
            "observations": frame.row_id.nunique(), "groups": frame.lineage_group.nunique(), "exact_pdo": support}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for action, fields in {
        "register": ("data", "assets", "core"),
        "predict": ("pdo", "licob", "archive", "priors", "hallmark", "targets", "auxiliary", "original", "core"),
        "score": ("pdo", "licob", "forecasts", "limore-labels", "licob-labels", "original", "core"),
    }.items():
        sub = stages.add_parser(action)
        for field in (*fields, "out"):
            sub.add_argument("--" + field, type=Path, required=True)
        if action == "predict":
            sub.add_argument("--mode", choices=EXTERNAL_TRACKS, required=True)
    args = parser.parse_args()
    if args.stage == "register":
        register(args)
    elif args.stage == "predict":
        predict(args)
    else:
        _, auc, _, _ = load_pdo(args.pdo)
        score_external(args, assessor=partial(assess, exact=set(auc.columns)), arms=ARMS)


if __name__ == "__main__":
    main()
