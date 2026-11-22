"""Register exact LICOB structures and evaluate prediction on unseen drug scaffolds."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from rdkit import Chem
from sklearn.model_selection import GroupKFold

from .expanded_panel_cv import expanded_targets, registered_inputs
from .external import KEY, SOURCE
from .function_data import canonical_chemistry, new_output
from .hcc_panel_support import GDSC_RECEIPT
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_full_axis import validated as validated_wide
from .licob_node_recovery import joined_outcomes
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import centered_kernels, component_scores, orthogonal_parts
from .structured_cv import attach, paired_intervals, with_assay
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_licob import ASSET_HASHES, CAS_EVIDENCE, adult_donor_outcomes, checked_registration
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

WIDE = "1f37cdff695ad7c9cbac06430a86b77d5ef5dfca5f9fa03d4557954b464102a0"
TRACK = "LICOB_COLD"
PRIMARY = "interaction_transfer"
CONTROL = ("chemical_mean", "neighbor_transfer")
ARMS = ("grand_mean", *CONTROL, "chemical_kernel", "molecular_kernel", PRIMARY)
STEREO_CAS = {
    "1037184-44-3": "d74201b41b09c2569a8bf48f60517b4b0e2e895129c6cc71d88f00d9783038b0",
    "1071992-99-8": "a69180dc6b6c52b7a9a5f5f10b79053ddc3cfb63ab24bf14d9a833342de65788",
    "122111-03-9": "4044727b894f1737afa24ed700877094c29bb5b25e7a62b17a489e50e9176cd4",
    "1223405-08-0": "fff15ae354c1675a3b37d10134dcf8193f7ec0394efea8c01b9b5cd41c0b3daf",
    "1346574-57-9": "b344833a40532a2abf7b629bbfe224516e297a6baffa13bae8f95237cfdaa85f",
    "1352226-88-0": "b247102dfbace9a7d716f1c3a0e948e58676db862f110b4455f23c3c55689934",
    "143-67-9": "4ccfc1cc5905affe6961985655d407e26058d2d172e6c07f6da333661abbcd00",
    "1644545-52-7": "8e079eae3b0a211dabf00238fc23bc57df964a08afc699c14ec62e4b310ac792",
    "1949837-12-0": "2609db2b728939ddde3e8b86f08710cf6390bf6d82b72ca0f8b07c779841af91",
    "1950634-92-0": "0d8dcf1aecb8403154612b0ce14878b2488c24bf51301922b038879a4b773a14",
    "2064292-12-0": "70a5e81b2cb2b48d156ab22495067e4910e97b2c7adfa56615a5739f11168142",
    "2688842-08-0": "4dc3369d25d295f9daba0d576de123906ac74bd8c37ec4ccd8fd8830ac437fec",
    "336113-53-2": "c9699c0ffe321fe279d8763a9d3990a3c99e64b41c81a2125b532ae19339cefa",
    "571203-78-6": "ba2c113e02f8e1064446134be41862dea52e2089cbd91529633d7fcf1a2221fb",
    "72496-41-4": "0b0faf411aa8d731493b037255b4fa89fc59fc3786c99b8f9bc4bc65f6e6942a",
    "867160-71-2": "609b962f2b177e8caad56231203b55e5a244e53b0af7efb6379356e117fdc60f",
}
EXPECTED_QUERY = (43, 38, 76)
PENALTY = 4.0


def chemistry(args):
    checked_registration(args.author)
    require_hash(args.catalogue, ASSET_HASHES["catalog"])
    original = pd.read_parquet(args.author / "chemistry.parquet").rename(columns={"compound_id": "original_parent"})
    vendor = pd.read_excel(args.catalogue, sheet_name=1, usecols=["Cat", "CAS Number", "SMILES"])
    updated = []
    for row in original.itertuples():
        code = row.catalogue
        known = CAS_EVIDENCE.get(code)
        supplier = vendor[vendor.Cat.eq(code)]
        cas = known[0] if known else str(supplier["CAS Number"].iloc[0])
        if cas in STEREO_CAS:
            path = args.stereo / f"{cas}.json"
            require_hash(path, STEREO_CAS[cas])
        elif known:
            path = args.cas / f"{cas}.json"
            require_hash(path, known[1])
        else:
            path = None
        if path is not None:
            props = json.loads(path.read_text())["PropertyTable"]["Properties"]
            possibilities = {canonical_chemistry(item["SMILES"]) for item in props}
            evidence = f"CAS {cas} / PubChem " + ",".join(str(item["CID"]) for item in props)
        else:
            possibilities = {canonical_chemistry(value) for value in supplier.SMILES}
            evidence = f"Selleck L1700 {code}"
        if len(possibilities) != 1:
            raise ValueError(f"Conflicting structure evidence for {code}")
        smiles, key, family, _ = possibilities.pop()
        if family != row.original_parent.split("-")[0]:
            raise ValueError(f"Resolved stereochemistry changed chemical connectivity for {code}")
        gaps = sum(site.specified == Chem.StereoSpecified.Unspecified
                   for site in Chem.FindPotentialStereo(Chem.MolFromSmiles(smiles)))
        updated.append({"catalogue": code, "compound_id": key, "isomeric_smiles": smiles, "connectivity_group": family,
                        "cas": cas, "remaining_unspecified_stereo": gaps, "structure_evidence": evidence})
    clean = original.drop(columns="isomeric_smiles").merge(pd.DataFrame(updated), on="catalogue", validate="one_to_one")
    if not clean.compound_id.is_unique:
        raise ValueError("No two LICOB catalog entries may identify the same exact parent")
    return clean


def register(args):
    if validated_wide(args.wide) != WIDE:
        raise ValueError("Unexpected LICOB RNA roster")
    _, core, rna, extra = registered_inputs(args, include_extension=True)
    require_hash(args.gdsc / "receipt.json", GDSC_RECEIPT)
    gdsc_file = args.gdsc / "additional_rows.parquet"
    require_hash(gdsc_file, json.loads((args.gdsc / "receipt.json").read_text())["artifacts"][gdsc_file.name])
    gdsc = pd.read_parquet(gdsc_file)
    cols = ["cell_id", "lineage_group", "compound_id", "isomeric_smiles", "source_compound_label",
            "drug_target_source_labels", "row_id", "endpoint_direction", "endpoint_value", *SOURCE]
    train = pd.concat([x[cols].assign(core_rows=(i == 0)) for i, x in enumerate((core, extra, gdsc))],
                      ignore_index=True)
    if train.duplicated([*SOURCE, "cell_id", "compound_id"]).any() or train.row_id.duplicated().any():
        raise ValueError("Conflicting frozen source experiment identities")
    catalog = chemistry(args)
    learned = set(train.compound_id)
    catalog["in_source"] = catalog.compound_id.isin(learned)
    unsupported = set(catalog.loc[~catalog.in_source, "connectivity_group"])
    excluded = train[train.compound_id.str.split("-").str[0].isin(unsupported)]
    train = train.drop(index=excluded.index)
    count = train.groupby("compound_id").size()
    catalog["training_rows"] = catalog.compound_id.map(count).fillna(0).astype(int)
    catalog["in_source"] = catalog.training_rows.gt(0)
    profile = pd.read_parquet(args.wide / "query.parquet", columns=[
        "cell_id", "donor_id", "adult_age_confirmed", "new_donor", "response_row"
    ]).drop_duplicates()
    query = profile.merge(catalog, how="cross").assign(lineage_group=lambda rows: rows.donor_id)
    query = query.assign(row_id=TRACK + "|" + query.cell_id + "|" + query.compound_id)
    counts = (query.cell_id.nunique(), query.loc[query.adult_age_confirmed, "donor_id"].nunique(),
              query.compound_id.nunique())
    if counts != EXPECTED_QUERY or set(query.cell_id) & set(train.cell_id) or query.row_id.duplicated().any():
        raise ValueError(f"Unexpected RNA/support/overlap for {counts}")
    held = pd.read_parquet(args.wide / "rna.parquet")
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    with closing(load_workbook(args.supplement, read_only=True, data_only=True)) as book:
        if len(next(book["S6. AUC"].iter_rows(min_row=2, max_row=2, values_only=True))) != 77:
            raise ValueError("Unexpected response header width")
    with new_output(args.out) as area:
        joint = pd.concat([rna.loc[sorted(train.cell_id.unique()), held.columns], held], verify_integrity=True)
        for name, frame in {"train": train, "query": query, "rna": joint}.items():
            frame.to_parquet(area / f"{name}.parquet", index=name == "rna")
        publish_json(area / "receipt.json", {"stage": TRACK, "wide_sha256": WIDE,
            "workbook_sha256": ASSET_HASHES["supplement"], "resolved_cas_sha256": STEREO_CAS,
            "models": counts[0], "adult_donors": counts[1], "genes": len(held.columns),
            "fit_compounds": int(train.compound_id.nunique()), "fit_rows": len(train),
            "sources": train.source_id.value_counts().to_dict(),
            "excluded_unresolved_source": excluded.groupby("compound_id").size().to_dict(),
            "support": catalog.drop(columns="isomeric_smiles").to_dict("records"),
            "files": {f"{name}.parquet": sha256(area / f"{name}.parquet") for name in ("train", "query", "rna")}})


def checked(area, names):
    receipt = area / "receipt.json"
    doc = json.loads(receipt.read_text())
    if (doc["stage"], doc["wide_sha256"], doc["resolved_cas_sha256"]) != (TRACK, WIDE, STEREO_CAS):
        raise ValueError("Wrong full LICOB registration")
    for key in names:
        require_hash(area / f"{key}.parquet", doc["files"][f"{key}.parquet"])
    return sha256(receipt)


def double_cold(train, fold):
    chemistry = train[["compound_id", "isomeric_smiles"]].drop_duplicates()
    if not chemistry.compound_id.is_unique:
        # Equal full-parent keys can have several interchangeable representations.
        chemistry = chemistry.drop_duplicates("compound_id")
    chemistry["scaffold"] = chemistry.isomeric_smiles.map(lambda x: canonical_chemistry(x)[3])
    chemistry = chemistry.set_index("compound_id").sort_index()
    core = chemistry.loc[sorted(train.loc[train.core_rows, "compound_id"].unique())]
    _, held_index = list(GroupKFold(n_splits=3).split(core, groups=core.scaffold))[fold]
    held = core.iloc[held_index]
    scaffolds = set(held.scaffold)
    connection = {drug.split("-")[0] for drug in held.index}
    blocked = set(chemistry.index[chemistry.scaffold.isin(scaffolds)]) | {
        key for key in chemistry.index if key.split("-")[0] in connection}
    ids = train[["cell_id", "lineage_group"]].drop_duplicates().sort_values("cell_id")
    _, held_cells = list(GroupKFold(n_splits=3).split(ids, groups=ids.lineage_group))[fold]
    lineages = set(ids.iloc[held_cells].lineage_group)
    fit = train[~train.lineage_group.isin(lineages) & ~train.compound_id.isin(blocked)]
    labels = train[train.core_rows & train.lineage_group.isin(lineages) & train.compound_id.isin(held.index)]
    if len(labels) < 1000 or set(labels.lineage_group) & set(fit.lineage_group) or set(labels.compound_id) & set(fit.compound_id):
        raise ValueError("Invalid scaffold × lineage split")
    return fit, labels


def fit_panel(fit, registry, rna, edges, targets, auxiliary, device):
    import torch
    from .structured_kernel import (
        KernelExpressionSchema, PanelPredictor, drug_kernels, fit_cell_kernels, solve_residuals, training_grid,
    )

    _, anchor, data, calibration = panel_targets(fit)
    structural_rows = pd.concat([fit[["compound_id", "isomeric_smiles"]],
        registry[["compound_id", "isomeric_smiles"]]], ignore_index=True).drop_duplicates("compound_id")
    ids, blocks = drug_kernels(structural_rows, targets, device=device)
    identity = torch.eye(len(ids), device=device)
    tanimoto = ((blocks["rna_chem_only"] - .75 * identity - .1) / .15).clamp(0, 1)
    shared_target = ((blocks["structured"] - .65 * identity - .15 * tanimoto - .1) / .1).clamp(0, 1)
    fitted, held = pd.Index(sorted(fit.compound_id.unique())), pd.Index(sorted(registry.compound_id.unique()))
    fit_index = torch.as_tensor(ids.get_indexer(fitted), device=device)
    held_index = torch.as_tensor(ids.get_indexer(held), device=device)

    def blocks_for(matrix):
        return matrix[fit_index[:, None], fit_index], matrix[held_index[:, None], fit_index]

    fit_cells = pd.Index(sorted(fit.cell_id.unique()))
    fit_rna = rna.loc[fit_cells]
    basis, cell_terms = fit_cell_kernels(fit_rna, edges, auxiliary, device=device)
    cell = cell_terms["full"]
    schema = KernelExpressionSchema(basis.genes, basis.indices, np.median(fit_rna.to_numpy(), axis=0))
    weights, response = training_grid(data, fit_cells, fitted, device=device)
    _, sim = blocks_for(tanimoto)
    selected = sim.argsort(dim=1, descending=True, stable=True)[:, :5]
    score = sim.gather(1, selected)
    neighbor = torch.zeros_like(sim)
    neighbor.scatter_(1, selected, score.pow(4))
    neighbor /= neighbor.sum(dim=1, keepdims=True).clamp_min(1e-12)
    _, lookup = blocks_for(identity)
    warm = lookup.sum(dim=1) > 0
    taper = torch.where(warm[:, None], lookup, neighbor)
    base_train = torch.as_tensor(anchor.reindex(fitted).to_numpy(dtype=np.float32), device=device)
    overall = float(base_train.mean().item())
    base = torch.where(taper.sum(dim=1) > 0, taper @ base_train, overall)
    intercept = pd.Series(base.cpu().numpy(), index=held)
    predictors = {"grand_mean": PanelPredictor(schema, basis, held, pd.Series(overall, index=held), None),
                  "chemical_mean": PanelPredictor(schema, basis, held, intercept, None)}
    _, _, residue = orthogonal_parts(response.cpu().numpy(), weights.cpu().numpy())
    c_centered, _ = centered_kernels(cell, cell)
    center = cell.mean(dim=0, keepdim=True)
    residues = torch.as_tensor(residue, dtype=weights.dtype, device=device)
    notes = {}
    for name, gram in (("neighbor_transfer", identity), ("chemical_kernel", .35 * identity + .55 * tanimoto + .1),
                       ("molecular_kernel", .35 * identity + .35 * tanimoto + .2 * shared_target + .1),
                       (PRIMARY, .35 * identity + .35 * tanimoto + .2 * shared_target + .1)):
        drug_fit, drug_query = blocks_for(gram)
        lhs, y = cell, response
        if name == PRIMARY:
            drug_fit, drug_query = centered_kernels(drug_fit, drug_query)
            lhs, y = c_centered, residues
        coef, steps, norm = solve_residuals(lhs, drug_fit, weights, y, PENALTY)
        if name == "neighbor_transfer":
            drug_query = taper
        predictors[name] = PanelPredictor(schema, basis, held, intercept, coef @ drug_query.T,
                                          center if name == PRIMARY else None)
        notes[name] = {"iterations": steps, "residual": norm}
    sims = dict(zip(held, score[:, 0].cpu().tolist()))
    return predictors, {"solves": notes, "fit_rows": len(fit), "fit_drugs": len(fitted), "calibration": calibration,
                        "features": int(basis.reference[-1].shape[1]), "nearest_tanimoto": sims,
                        "target_annotated_query": sum(bool(targets.get(d)) for d in held)}


def models(fit, query, rna, edges, targets, auxiliary, device):
    registry = query[["compound_id", "isomeric_smiles"]].drop_duplicates()
    predictors, notes = fit_panel(fit, registry, rna, edges, targets, auxiliary, device)
    held_rna = rna.loc[sorted(query.cell_id.unique())]
    cross = predictors[PRIMARY].basis.cross(held_rna)["full"]
    forecasts = []
    for name, model in predictors.items():
        anchor, residual = model.predict(query, held_rna, cross=cross)
        forecasts.append(query[KEY].assign(method=name, predicted_value=anchor + residual))
    return pd.concat(forecasts, ignore_index=True), notes


def predict(args):
    from .v0plus_neural import load_auxiliary

    digest = checked(args.inputs, ("train", "query", "rna"))
    fit = pd.read_parquet(args.inputs / "train.parquet")
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    if args.fold == 3:
        query = pd.read_parquet(args.inputs / "query.parquet")
        annotations = pd.concat([fit, query], ignore_index=True)
    else:
        fit, labels = double_cold(fit, args.fold)
        query = validation_roster(labels)
        annotations = pd.concat([fit, labels[["compound_id", "drug_target_source_labels"]]], ignore_index=True)
    targets, _ = expanded_targets(annotations[~annotations.compound_id.isin(targets)], targets, rna.columns)
    outputs, notes = models(fit, query, rna, edges, targets, auxiliary, device_cuda())
    if args.fold == 3:
        shared_support(outputs, query, ARMS)
        outputs = outputs.assign(track=TRACK, fold=0)
    else:
        outputs = with_assay(outputs, labels, args.fold)
    freeze_table(outputs, args.out, provenance={"input_sha256": digest, "fold": args.fold, "model": notes,
        "priors": priors, "auxiliary": aux})


def contrast_gates(summary, cis):
    rank = {(x["baseline"], x["metric"]): x["ci95"][0] for x in summary["paired_contrasts"]
            if x["candidate"] == PRIMARY}
    return {base: all(cis[base].get(k, [-1])[0] > 0 for k in ("pearson", "spearman"))
            and all(rank.get((base, k), -1) > -.005 for k in ("spearman", "ndcg_at_5"))
            for base in CONTROL}


def summarize(data, scope, *, external):
    comparisons = tuple((PRIMARY, base) for base in (*CONTROL, "molecular_kernel", "chemical_kernel"))
    summary, drugs, _ = axis_metrics(data, comparisons)
    ci_fn = crossed_patient_drug_ci if external else paired_intervals
    cis = {base: ci_fn(data, PRIMARY, base) for _, base in comparisons}
    if data.compound_id.nunique() <= 5:
        for row in summary["scores"]:
            row.pop("macro_ndcg_at_5", None)
            row.pop("macro_top5_overlap", None)
        summary["paired_contrasts"] = [x for x in summary["paired_contrasts"] if x["metric"] == "spearman"]
    output = {"summary": summary, "within_drug_ci95": cis, "gate": contrast_gates(summary, cis),
              "rows": int(data.row_id.nunique()), "drugs": int(data.compound_id.nunique())}
    if external:
        output["components"] = component_scores(data, cell_axis="lineage_group")
        output["per_drug"] = drugs.astype(object).where(pd.notna(drugs), None).to_dict("records")
        strata = {}
        for name, ids in compound_strata(scope).items():
            measures = drugs[drugs.compound_id.isin(ids)].groupby("method")[["spearman", "pearson"]].mean().dropna()
            if not measures.empty:
                strata[name] = {"drugs": len(ids), "scores": measures.to_dict("index")}
        output["strata"] = strata
    return output


def verified(args, fold, digest):
    rows, seal = verify_forecast(args.parts / f"fold-{fold}.parquet")
    if (seal["provenance"]["fold"], seal["provenance"]["input_sha256"]) != (fold, digest):
        raise ValueError("Mismatched frozen fold")
    return rows, seal


def source_score(args):
    digest = checked(args.inputs, ("train",))
    train = pd.read_parquet(args.inputs / "train.parquet")
    sections, seals = [], {}
    for fold in range(3):
        rows, seal = verified(args, fold, digest)
        _, labels = double_cold(train, fold)
        sections.append(attach(rows, labels))
        seals[str(fold)] = seal["sha256"]
    results = {name: summarize(section, None, external=False)
               for name, section in pd.concat(sections, ignore_index=True).groupby("track")}
    publish_json(args.out, {"input_sha256": digest, "forecast_sha256": seals, "results": results,
        "gate": sum(all(value["gate"].values()) for value in results.values()) >= 2})


def external_score(args):
    digest = checked(args.inputs, ("query",))
    query = pd.read_parquet(args.inputs / "query.parquet")
    rows, seal = verified(args, 3, digest)
    shared_support(rows, query, ARMS)
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    donors = adult_donor_outcomes(joined_outcomes(query, rows, args.supplement))
    for name in ("grand_mean", "chemical_mean"):
        constants = rows[rows.method.eq(name)].groupby("compound_id").predicted_value
        if not constants.nunique().eq(1).all():
            raise ValueError("Expected a sample-invariant baseline")
        selected = donors.method.eq(name)
        donors.loc[selected, "predicted_value"] = donors.loc[selected, "compound_id"].map(constants.first())
    by_support = query.drop_duplicates("compound_id")
    cold = by_support[~by_support.in_source]
    unequivocal = cold[cold.remaining_unspecified_stereo.eq(0)]
    cases = {"all_76": by_support, "source_support": by_support[by_support.in_source], "cold": cold,
        "cold_fully_specified": unequivocal,
        "restored_source": by_support[by_support.in_source & by_support.compound_id.ne(by_support.original_parent)]}
    results = {}
    for key, sub in cases.items():
        results[key] = summarize(donors[donors.compound_id.isin(sub.compound_id)], sub, external=True)
    publish_json(args.out, {"input_sha256": digest, "forecast_sha256": seal["sha256"],
        "labels_sha256": ASSET_HASHES["supplement"], "results": results,
        "cold_gate": all(results["cold_fully_specified"]["gate"].values()),
        "nearest_tanimoto": seal["provenance"]["model"]["nearest_tanimoto"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    q = parser.add_subparsers(dest="stage", required=True)
    for stage, paths in {
        "register": ("original", "core", "extension", "gdsc", "author", "wide", "catalogue", "cas", "stereo", "supplement"),
        "predict": ("inputs", "priors", "hallmark", "targets", "auxiliary"),
        "source": ("inputs", "parts"), "external": ("inputs", "parts", "supplement"),
    }.items():
        sub = q.add_parser(stage)
        for key in (*paths, "out"):
            sub.add_argument("--" + key, type=Path, required=True)
        if stage == "predict":
            sub.add_argument("--fold", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    {"register": register, "predict": predict, "source": source_score, "external": external_score}[args.stage](args)


if __name__ == "__main__":
    main()
