"""Register the author-distributed LICOB RNA and score a source-only forecast freeze."""

import argparse
import json
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .evaluate import join_observations
from .external import KEY, _validate
from .function_data import alias_token, canonical_chemistry
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_broutier import writer
from .v0plus_cv import candidate_forecasts, molecular_inputs, training_inputs
from .v0plus_licob_audit import LICOB_RNA
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_sota import verify_upstream
from .v0plus_wong import SETTINGS_HASH

COHORT = "LICOB_IHCC"
AUTHOR_COMMIT = "c65cb2f9305defbf5b87b56a7216d673d6b94a39"
ASSET_HASHES = {
    "rna": "8d2025fe76cb9306928ec12ca9095c45c07986c2a91b2efce58443d78ac35c11",
    "author": "5fb1460932286215d8368e1d0f00a481345ae1ac88fafb21e51068c72b43b135",
    "supplement": "0e158e1129349e3e32dc6840104f92959b95788705edb9e191ffe47ac21fa906",
    "catalog": "95092a4577933781ca3e96b8015eb8133f07bd2a1fbac2e15696b59b801d6c57",
    "node_archive": "c40ef61dcb7453415fcf1fdbeb557016e1935c9097715dfb22d75225b6fbf69e",
}
CAS_EVIDENCE = {
    "S1001": ("923564-51-6", "c6ea44893baadf1acd2ded2454a36801a0aa88d990edd2ca8f02009ea78fb1fc"),
    "S1011": ("850140-72-6", "a31ed15c97bbda7a3795552372493173c5acd181a70c191562f42dcee57a46ca"),
    "S1039": ("53123-88-9", "83d9aa79e733490114574f2070a652fc18e3bd887790ea417aef05ab9dce5b2d"),
    "S1044": ("162635-04-3", "867901ccd3d4d172b95afd5df90674eacbc52e3c2bec705005819e8e9c9de6eb"),
    "S1068": ("877399-52-5", "820b975c04b60dcfac103d9b3082cf00ad95b71163b6cebf3f4ef47e94b57e31"),
    "S1109": ("755038-02-9", "ff18907fdfa350f3fe930775ac98660e12fc3dec2cf9248f3cb212547ae66659"),
    "S1141": ("75747-14-7", "9c484a10816a01cef2281b2aba16aff595d92394dfc0889f6f6fab09edf4951e"),
    "S1148": ("114977-28-5", "8be7632b2bf787bd7b5eb466180286be38de72c1bc025cee74b1292809d6fe43"),
    "S1150": ("33069-62-4", "de8ab3d208537290adf9c8663883f896543ff8eb9a5fb21c8c13b368531d3a2b"),
    "S1210": ("59-05-2", "35d934382862c7245bbc5857510ce9e324e9adfd66010a64e19dccfa6fc0771c"),
    "S1225": ("33419-42-0", "6a8837b76c84150f4f7462404ef9a6dbbc9ab7656b13f766b8d8fce38d3e636b"),
    "S1231": ("119413-54-6", "ff1798b8b600ba22d09ca3ebbbb72c0b8d1c6bbe7da72a1df7e3d50d46f2ce03"),
    "S1453": ("192185-72-1", "7b83456bc67991af345349a0c9cb8e01d874ac44f533c688a7a1755ca2370839"),
    "S1532": ("860352-01-8", "cfabdb40b55e9097eeaacc13a7d5b8f9ab9c93c661b7bf3c42e176ab0ed30a71"),
    "S1792": ("79902-63-9", "400951a997845e96030b86713ff802f7b91392fda4513c87e9d695750b4a1b59"),
    "S2753": ("905854-02-6", "c8026b38995b204aeee55472a82c2068d6a45a3feee3bcb255df4e3026800cb2"),
    "S2780": ("1300031-49-5", "b91d5289db2924c4c6f3e80fa1f503e058974a0ca8b55a2dca41aa2f6ae80b71"),
    "S4505": ("143-67-9", "050142067fedc3c6c4017a335098f021371c50956bf92cf5207fc8b214ac7444"),
    "S7048": ("1207456-01-6", "b39c440f6579907c699c864fe98bb6d13d9a75e476f4fdff47e34bce9b754eb8"),
    "S7110": ("1268524-70-4", "952ccf87edf470169ae3e872b0874726ba607e675c478c866fd1b9a6f7aac5b1"),
    "S8041": ("934660-93-2", "3588e4589eb01be2c3d662531f89b063021f252fc6c56345d64c9aa458583c53"),
    "S1208": ("25316-40-9", "f49f55f68294da573b3f1ccdba89f65655854950a672877292a99393a8168d01"),
    "S1223": ("56390-09-1", "19e99a1aed1a93bed93fccc811f6360ccd724d76747f555b30a87af96bcc7f0b"),
    "S2245": ("1080622-86-1", "72036a8783d611651ee4ea5345d66058ee73bf077d71076d35a773f638832c95"),
    "S7120": ("120964-45-6", "301153e0dda76806a33f2584a0390a1b08290db5c62268f9edc603117a59d7cc"),
    "S7242": ("571203-78-6", "a5b9dc5cfbc1b0b683fd243c9be5fca76a1d280ca88d288e3dca42c2d7056b9a"),
    "S1009": ("915019-65-7", "00ca74f0a354f6e12d9e963e5b616ae276f5451a3bcfccfe3b23c9ea7c923ed6"),
}
ARMS = {
    "linear": ("drug_mean", "additive", "drug_ridge", "drug_elasticnet"),
    "lightgbm": ("lightgbm",), "catboost": ("catboost",), "cellhit": ("cellhit",),
    "velodrome_inductive": ("velodrome_inductive",), "codeae_inductive": ("codeae_inductive",),
    "v0plus_inductive": ("v0plus_inductive",),
}
PUBLIC_ARMS = {"cellhit", "velodrome_inductive", "codeae_inductive"}
BLEND = "ridge_molecular_average"


def check_hash(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"Registered input hash changed for {path.name}")


def paired_axis(supplement):
    sheet = supplement["S6. AUC"]
    header = next(sheet.iter_rows(min_row=2, max_row=2, values_only=True))
    keys = [alias_token(name) for name in header[1:]]
    rows = [(row[0], position) for position, row in enumerate(
        sheet.iter_rows(min_row=3, max_col=1, values_only=True), start=3,
    ) if re.fullmatch(r"(?:HCCO|ICCO|HBO|HCCICO|CHCO)\d+", str(row[0]))]
    if len(keys) != 76 or len(set(keys)) != 76 or len(dict(rows)) != len(rows):
        raise ValueError("Unexpected LICOB response header")
    return keys, dict(rows)


def chemical_registry(supplement, catalog_path, cas_dir, response_drugs, train):
    records = load_workbook(catalog_path, read_only=True, data_only=True).worksheets[1].values
    catalog = pd.DataFrame(records, columns=next(records)).dropna(subset=["Cat"]).set_index("Cat")
    rows = supplement["S5. Drug information"].iter_rows(min_row=3, values_only=True)
    registered = []
    for name, _, targets, category, pathway, _, _, _, _, vendor, code in rows:
        code = str(code).strip()
        if vendor != "Selleck" or not re.fullmatch(r"S\d{4}", code):
            raise ValueError(f"Unexpected LICOB catalogue identifier for {name}")
        if code in CAS_EVIDENCE:
            cas, digest = CAS_EVIDENCE[code]
            path = cas_dir / f"{cas}.json"
            check_hash(path, digest)
            products = json.loads(path.read_text())["PropertyTable"]["Properties"]
            smiles = [product["SMILES"] for product in products]
            evidence = f"Selleck CAS {cas} / PubChem CID " + ",".join(str(p["CID"]) for p in products)
        else:
            smiles = catalog.loc[[code], "SMILES"].tolist()
            evidence = f"Selleck L1700 {code}"
        structures = {canonical_chemistry(value)[1] for value in smiles}
        if len(structures) != 1:
            raise ValueError(f"Ambiguous exact parent identity for {name}")
        registered.append({
            "response_label": str(name).strip(), "source_compound_label": str(name).strip(),
            "drug_target_source_labels": " ".join(str(x) for x in (targets, category, pathway) if x),
            "catalogue": code, "compound_id": next(iter(structures)), "evidence": evidence,
            "response_column": response_drugs.index(alias_token(name)) + 1,
        })
    drugs = pd.DataFrame(registered)
    if (len(drugs) != 76 or not drugs.response_column.is_unique or not drugs.compound_id.is_unique):
        raise ValueError("LICOB requires a one-to-one S5/S6/chemical parent mapping")
    structures = train[["compound_id", "isomeric_smiles"]].drop_duplicates().set_index(
        "compound_id", verify_integrity=True,
    )
    return drugs.join(structures, on="compound_id", validate="1:1")


def prefix_concordance(path, author):
    """Compare a readable prefix for identity only. All modeling RNA is from the author bundle."""
    with ZipFile(path) as archive:
        strings = ["".join(item.itertext()) for item in ElementTree.fromstring(
            archive.read("xl/sharedStrings.xml"),
        )]
        with archive.open(LICOB_RNA) as stream:
            rows = re.findall(rb"<row\s.*?</row>", stream.read(4 * 1024 * 1024))
    cells = [re.findall(rb"<v>(.*?)</v>", row) for row in rows]
    columns = [strings[int(value)] for value in cells[1][1:]]
    prefix = pd.DataFrame(
        [[float(value) for value in row[1:]] for row in cells[2:]], columns=columns,
        index=[strings[int(row[0])] for row in cells[2:]],
    )
    genes = prefix.index.intersection(author.index)
    samples = prefix.columns.intersection(author.columns)
    correlations = prefix.loc[genes, samples].T.corrwith(author.loc[genes, samples].T).dropna()
    return {
        "reconciled_genes_in_prefix": len(correlations), "shared_samples": len(samples),
        "median_gene_pearson": float(correlations.median()),
        "p05_gene_pearson": float(correlations.quantile(.05)),
    }


def register(args):
    for field, digest in ASSET_HASHES.items():
        check_hash(getattr(args, field), digest)
    supplement = load_workbook(args.supplement, read_only=True, data_only=True)
    train, source_rna, _ = training_inputs(args.bundle)
    drugs, response_rows = paired_axis(supplement)
    chemical = chemical_registry(supplement, args.catalog, args.cas, drugs, train)
    rna = pd.read_csv(args.rna, sep="\t", index_col=0)
    if rna.shape != (26078, 50) or not rna.index.is_unique or not rna.columns.is_unique:
        raise ValueError("Unexpected author RNA sample or gene axis")
    if not np.isfinite(rna.to_numpy()).all():
        raise ValueError("Nonfinite author RNA")
    concordance = prefix_concordance(args.node_archive, rna)
    if concordance["shared_samples"] != len(rna.columns) or concordance["median_gene_pearson"] < .90:
        raise ValueError("Author RNA failed cross-check against the NODE prefix")

    diagnoses = {row[0]: row[1] for row in supplement["S1. Clinical information"].iter_rows(
        min_row=4, max_col=6, values_only=True,
    ) if row[5] == "HCC"}
    profile_ids = sorted(set(diagnoses) & set(rna.columns) & set(response_rows))
    adult_ids = [name for name in profile_ids if isinstance(diagnoses[name], (int, float)) and diagnoses[name] >= 18]
    if len(adult_ids) < 5 or len(chemical.dropna(subset=["isomeric_smiles"])) < 5:
        raise ValueError("Insufficient registered adult organoid or compound support")
    if any(not re.fullmatch(r"HCCO\d+", name) for name in profile_ids):
        raise ValueError("Unexpected LICOB HCC model identifiers")
    genes = sorted(set(rna.index) & set(source_rna.columns))
    expression = rna.loc[genes, profile_ids].T.rename_axis("cell_id")
    models = pd.DataFrame({"cell_id": profile_ids, "adult_age_confirmed": [s in adult_ids for s in profile_ids]})
    query = models.merge(chemical.dropna(subset=["isomeric_smiles"]), how="cross")
    query["lineage_group"] = query.cell_id
    query["row_id"] = COHORT + "|" + query.cell_id + "|" + query.compound_id
    query["response_row"] = query.cell_id.map(response_rows)
    _validate(train, query)
    files = writer(args.out, {"rna.parquet": expression, "query.parquet": query, "chemistry.parquet": chemical})
    publish_json(args.out / "receipt.json", {
        "cohort": COHORT, "author_commit": AUTHOR_COMMIT, "asset_sha256": ASSET_HASHES,
        "bundle_sha256": sha256(args.bundle / "receipt.json"), "files": files,
        "cas_sha256": {cas: digest for cas, digest in CAS_EVIDENCE.values()},
        "node_prefix_rna_identity": concordance, "author_profiles": len(rna.columns),
        "source_rna_common_genes": len(genes), "all_biobank_hcc_models": len(diagnoses),
        "hcc_with_s6": len(set(diagnoses) & set(response_rows)),
        "paired_adult_hcc": len(adult_ids), "paired_all_hcc": len(profile_ids),
        "screened_compounds": len(chemical), "source_supported_compounds": int(query.compound_id.nunique()),
        "registered_forecast_rows_per_method": len(query),
        "rna_missing_hcc_with_s6": sorted(set(diagnoses) & set(response_rows) - set(profile_ids)),
        "outside_training_compounds": chemical.loc[chemical.isomeric_smiles.isna(), "response_label"].tolist(),
    })


def checked_registration(path):
    receipt = json.loads((path / "receipt.json").read_text())
    if receipt["cohort"] != COHORT or receipt["asset_sha256"] != ASSET_HASHES:
        raise ValueError("Unexpected LICOB registration")
    for name, digest in receipt["files"].items():
        check_hash(path / name, digest)
    return receipt


def predict(args):
    registration = checked_registration(args.cohort)
    check_hash(args.settings, SETTINGS_HASH)
    check_hash(args.bundle / "receipt.json", registration["bundle_sha256"])
    train, source_rna, _ = training_inputs(args.bundle)
    query = pd.read_parquet(args.cohort / "query.parquet")
    rna = pd.read_parquet(args.cohort / "rna.parquet")
    _validate(train, query)
    selected = [tuple(item) for item in json.loads(args.settings.read_text())["settings"][args.arm]]
    if args.arm in PUBLIC_ARMS:
        verify_upstream(args.upstream)
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary = None
    if args.arm == "v0plus_inductive":
        from .v0plus_neural import load_auxiliary

        auxiliary, _ = load_auxiliary(args.auxiliary)
    forecast, _ = candidate_forecasts(
        args.arm, train, query, pd.concat([source_rna[rna.columns], rna]), edges, targets,
        gpu=args.gpu, selected=selected, upstream=args.upstream, auxiliary=auxiliary,
    )
    forecast.method = forecast.method.str.split("/").str[0]
    freeze_table(forecast.assign(track=COHORT, fold=0), args.out, provenance={
        "arm": args.arm, "registration_sha256": sha256(args.cohort / "receipt.json"), "settings_sha256": SETTINGS_HASH,
    })


def merge(args):
    checked_registration(args.cohort)
    registration_hash = sha256(args.cohort / "receipt.json")
    roster = pd.read_parquet(args.cohort / "query.parquet", columns=KEY).set_index("row_id").sort_index()
    parts, hashes = [], {}
    for arm, methods in ARMS.items():
        prediction, receipt = verify_forecast(args.parts / f"{arm}.parquet")
        expected = {"arm": arm, "registration_sha256": registration_hash, "settings_sha256": SETTINGS_HASH}
        if receipt["provenance"] != expected or set(prediction.method) != set(methods):
            raise ValueError(f"Unauthorized LICOB prediction component {arm}")
        for _, group in prediction.groupby("method"):
            if not group[KEY].set_index("row_id").sort_index().equals(roster):
                raise ValueError(f"Prediction identities mismatch in {arm}")
        parts.append(prediction)
        hashes[arm] = receipt["sha256"]
    forecast = pd.concat(parts, ignore_index=True)
    blends = (forecast[forecast.method.isin({"v0plus_inductive", "drug_ridge"})]
              .groupby([*KEY, "track", "fold"], as_index=False).predicted_value.mean().assign(method=BLEND))
    forecast = pd.concat([forecast, blends], ignore_index=True)
    if not np.isfinite(forecast.predicted_value.to_numpy()).all():
        raise ValueError("Nonfinite LICOB forecast")
    freeze_table(forecast, args.out, provenance={
        "registration_sha256": registration_hash, "settings_sha256": SETTINGS_HASH, "components_sha256": hashes,
    })


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    checked_registration(args.cohort)
    prediction, receipt = verify_forecast(args.forecast)
    provenance = receipt["provenance"]
    if (provenance["registration_sha256"] != sha256(args.cohort / "receipt.json")
            or provenance["settings_sha256"] != SETTINGS_HASH or set(provenance["components_sha256"]) != set(ARMS)):
        raise ValueError("LICOB score requires the joint registered freeze")
    check_hash(args.supplement, ASSET_HASHES["supplement"])
    query = pd.read_parquet(args.cohort / "query.parquet")
    sheet = load_workbook(args.supplement, read_only=True, data_only=True)["S6. AUC"]
    rows = list(sheet.values)
    measured = []
    for row in query.itertuples(index=False):
        record = rows[row.response_row - 1]
        value = record[row.response_column]
        if record[0] != row.cell_id or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("An audited LICOB response is missing or has the wrong identity")
        measured.append(-float(value))
    joined = join_observations(prediction, query.assign(endpoint_value=measured)).merge(
        query[["row_id", "adult_age_confirmed"]], on="row_id", validate="many_to_one",
    )
    methods = ("v0plus_inductive", BLEND)
    comparisons = [(candidate, baseline) for candidate in methods
                   for baseline in ("drug_mean", "drug_ridge", "cellhit")]
    outcomes = {}
    for name, block in (("adult", joined[joined.adult_age_confirmed]), ("all_hcc", joined)):
        table, drugs, _ = axis_metrics(block, comparisons if name == "adult" else ())
        if name == "adult":
            table["crossed_patient_drug_ci95"] = [
                {"candidate": a, "baseline": b, "intervals": crossed_patient_drug_ci(block, a, b)}
                for a, b in comparisons
            ]
        outcomes[name] = {
            "lineages": int(block.lineage_group.nunique()), "drugs": int(block.compound_id.nunique()),
            "observations": int(block.row_id.nunique()), "metrics": table,
            "per_drug": drugs.astype(object).where(pd.notna(drugs), None).to_dict("records"),
        }
    strata = {}
    adult = joined[joined.adult_age_confirmed]
    for name, parents in compound_strata(query).items():
        subset = adult[adult.compound_id.isin(parents)]
        if not subset.empty:
            table, _, _ = axis_metrics(subset, ())
            strata[name] = {"compounds": len(parents), "across_cell": table["across_cell_scores"]}
    primary = outcomes["adult"]["metrics"]
    gates = {}
    for baseline in ("drug_mean", "drug_ridge", "cellhit"):
        across = next(item["intervals"] for item in primary["crossed_patient_drug_ci95"]
                      if item["candidate"] == "v0plus_inductive" and item["baseline"] == baseline)
        within = [item["ci95"] for item in primary["paired_contrasts"]
                  if item["candidate"] == "v0plus_inductive" and item["baseline"] == baseline
                  and item["metric"] in {"spearman", "ndcg_at_5"}]
        gates[baseline] = len(across) == len(within) == 2 and all(
            interval[0] > 0 for interval in [*across.values(), *within]
        )
    publish_json(args.out, {
        "cohort": COHORT, "forecast_sha256": receipt["sha256"], "endpoint": "Negative published S6 AUC",
        "strata": strata, "outcomes": outcomes,
        "gates": {"by_baseline": gates, "comparative_superiority": gates["drug_ridge"] and gates["cellhit"]},
        "compound_labels": query.drop_duplicates("compound_id").set_index("compound_id").response_label.to_dict(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    modes = {
        "register": (*ASSET_HASHES, "cas", "bundle", "out"),
        "predict": ("cohort", "bundle", "settings", "priors", "hallmark", "targets", "upstream", "auxiliary", "out"),
        "merge": ("cohort", "parts", "out"),
        "score": ("cohort", "forecast", "supplement", "out"),
    }
    for mode, arguments in modes.items():
        command = commands.add_parser(mode)
        for name in arguments:
            command.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
        if mode == "predict":
            command.add_argument("--arm", choices=ARMS, required=True)
            command.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"register": register, "predict": predict, "merge": merge, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
