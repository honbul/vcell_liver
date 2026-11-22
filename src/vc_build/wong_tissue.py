"""Compare matched original-tumor and organoid RNA for source-only response prediction."""

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import join_observations
from .external import KEY
from .function_data import new_output
from .io import publish_json, sha256, verify_forecast
from .limore_corrected import shared_support
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .v0plus_wong import WORKBOOK_HASH, workbook
from .wong_expanded import METHODS, PRIMARY, checked, observations, summarize


def paired_tpm(sheet, fit_genes, pdo_ids):
    records = iter(sheet.values)
    header = next(records)
    if header[0] != "Gene symbol" or len(set(pdo_ids)) != 10:
        raise ValueError("Unexpected Wong paired-sample or gene header")
    tissue_ids = [cell.replace("PDTO_", "PDT_", 1) + "_ParentalTissue" for cell in pdo_ids]
    if set(tissue_ids) != {name for name in header if str(name).startswith("PDT_")} or any(
        header.count(cell) != 1 for cell in tissue_ids
    ):
        raise ValueError("Each registered organoid requires exactly one matching parental-tumor profile")
    columns = [header.index(cell) for cell in tissue_ids]
    records = [(row[0], *(row[i] for i in columns)) for row in records if row[0] in fit_genes]
    values = pd.DataFrame(records, columns=["gene", *pdo_ids])
    values = values[~values.gene.duplicated(keep=False)].set_index("gene").reindex(fit_genes)
    tpm = values.apply(pd.to_numeric, errors="raise").T
    if (tpm < 0).any(axis=None) or not np.isfinite(tpm.to_numpy()).all():
        raise ValueError("Tissue expression needs the full registered genes with nonnegative TPM")
    return np.log2(tpm + 1).rename_axis("cell_id")


def register(args):
    digest = checked(args.pdo, ("train.parquet", "rna.parquet", "query.parquet"))
    ordinary = json.loads((args.pdo / "receipt.json").read_text())
    expression = pd.read_parquet(args.pdo / "rna.parquet")
    pdo_ids = sorted(pd.read_parquet(args.pdo / "query.parquet", columns=["cell_id"]).cell_id.unique())
    with closing(workbook(args.workbook)) as handle:
        tissue = paired_tpm(handle["Supplementary Table 6"], expression.columns, pdo_ids)
    identical = all((tissue.loc[cell].to_numpy() == expression.loc[cell].to_numpy()).all() for cell in pdo_ids)
    if identical or tissue.shape != (10, len(expression.columns)):
        raise ValueError("Expected ten distinct tissue expression profiles")
    joint = pd.concat([expression.drop(index=pdo_ids), tissue], verify_integrity=True)
    with new_output(args.out) as out:
        for filename in ("train.parquet", "query.parquet"):
            (out / filename).symlink_to((args.pdo / filename).resolve())
        joint.to_parquet(out / "rna.parquet")
        publish_json(out / "receipt.json", ordinary | {
            "files": ordinary["files"] | {"rna.parquet": sha256(out / "rna.parquet")},
            "expression_input": "matched parental tumour TPM",
            "paired_pdo_registration_sha256": digest, "paired_profiles": len(tissue),
        })


def read_forecasts(args):
    digests = {}
    tables = {}
    query_stamps = set()
    for view, directory in (("organoid", args.pdo), ("tissue", args.tissue)):
        inputs = directory / "inputs"
        digest = checked(inputs, ("query.parquet",))
        receipt = json.loads((inputs / "receipt.json").read_text())
        rows, seal = verify_forecast(directory / "final.parquet")
        provenance = seal["provenance"]
        if (provenance["registration_sha256"], provenance["workbook_sha256"]) != (digest, WORKBOOK_HASH):
            raise ValueError("Frozen prediction and paired registration do not match")
        if view == "tissue" and receipt.get("paired_pdo_registration_sha256") != digests["organoid"]["registration"]:
            raise ValueError("Organ and tissue predictions do not use the same matched inputs")
        query_stamps.add(receipt["files"]["query.parquet"])
        tables[view] = rows
        digests[view] = {"registration": digest, "forecast": seal["sha256"]}
    if len(query_stamps) != 1:
        raise ValueError("Tissue and organoid predictions require the same roster")
    return tables, digests, pd.read_parquet(args.pdo / "inputs/query.parquet")


def matched_views(joined):
    candidate, baseline = "tissue", "organoid"
    contrasts = []
    for method in METHODS:
        pair = pd.concat([frame[frame.method.eq(method)].assign(method=view) for view, frame in joined.items()])
        metrics, _, _ = axis_metrics(pair, ((candidate, baseline),))
        drug = crossed_patient_drug_ci(pair, candidate, baseline)
        rank = {item["metric"]: item["ci95"] for item in metrics["paired_contrasts"]}
        entry = {"model": method, "drug_ci95": drug, "rank_ci95": rank}
        contrasts.append(entry)
    main = next(item for item in contrasts if item["model"] == PRIMARY)
    gate = all(main["drug_ci95"][name][0] > -.020 for name in ("spearman", "pearson")) and all(
        main["rank_ci95"][name][0] > -.020 for name in ("spearman", "ndcg_at_5"))
    return {"contrasts": contrasts, "primary_noninferior": gate}


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    tables, stamps, query = read_forecasts(args)
    for frame in tables.values():
        shared_support(frame, query, METHODS)
    actual = observations(query, args.workbook)
    scopes = {"new_drugs": query[~query.previously_scored], "all_matched": query,
              "previous_drugs": query[query.previously_scored]}
    results = {}
    for name, scope in scopes.items():
        truth = actual[actual.row_id.isin(scope.row_id)]
        if truth.lineage_group.nunique() != 10 or truth.groupby("compound_id").size().ge(5).sum() < 20:
            raise ValueError("Measured support below the preregistered comparison")
        joined = {view: join_observations(frame[frame.row_id.isin(truth.row_id)], truth)
                  for view, frame in tables.items()}
        results[name] = {view: summarize(frame, scope, main=name == "new_drugs") for view, frame in joined.items()}
        results[name]["paired"] = matched_views(joined)
    novel = results["new_drugs"]
    publish_json(args.out, {
        "primary": PRIMARY, "workbook_sha256": WORKBOOK_HASH, "sha256": stamps, "results": results,
        "organoid_joint_gate": all(novel["organoid"]["gate"].values()),
        "tissue_joint_gate": all(novel["tissue"]["gate"].values()) and novel["paired"]["primary_noninferior"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="action", required=True)
    for action, paths in {"register": ("pdo", "workbook"), "score": ("pdo", "tissue", "workbook")}.items():
        command = stages.add_parser(action)
        for name in (*paths, "out"):
            command.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    {"register": register, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
