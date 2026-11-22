"""Read-only audits of frozen 3D sensitivity and TCGA transcriptional concordance."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from vc_build.external import KEY, _rna
from vc_build.io import freeze_table, publish_json, sha256, verify_forecast
from vc_build.v0plus_cli import predict_offline
from vc_build.v0plus_score import axis_metrics, join_observations

from .release import HISTORICAL, historical, load_release, predict_table

PDX = "GSE90653_16_PDX"
TCGA = "TCGA_LIHC_primary_adult"
MARKERS = ("absolute_purity", "immune_score", "stromal_score")


def side_cohort(args, stem, track, location, release):
    snapshot = args.v1_runs / stem
    cohort = snapshot / ("bundle" if track.startswith("Broutier") else "cohort")
    stamp = json.loads((cohort / "receipt.json").read_text())
    for role in ("query.parquet", "rna.parquet"):
        if sha256(cohort / role) != stamp["files"][role]:
            raise ValueError(f"Changed archived {track} RNA or query registry")
    query = pd.read_parquet(cohort / "query.parquet")
    expression = _rna(cohort / "rna.parquet")
    new, cov = predict_table(query, expression, release, {}, track, min_coverage=.65)
    comparison, old_hash = historical(snapshot / "forecast.parquet", HISTORICAL)
    extras = ({"labels": stamp["files"]["labels.parquet"]} if track.startswith("Broutier") else
              {"figure": stamp["figure_pdf_sha256"]})
    freeze_table(pd.concat([new, comparison.assign(track=track, fold=0)], ignore_index=True), location,
                 provenance={"release_sha256": sha256(args.package/"release.json"),
                             "receipt_sha256": sha256(cohort/"receipt.json"), "archived_forecast_sha256": old_hash,
                             "min_rna_measured_fraction": float(cov.min().min()), **extras})


def tcga_expression(directory):
    receipt = json.loads((directory / "receipt.json").read_text())
    expected = receipt["outputs"]
    names = ("participants.tsv", "genes.tsv", "tpm_unstranded.npy")
    for key in names:
        value = expected[key]
        if isinstance(value, dict):
            value = value["sha256"]
        if sha256(directory/key) != value:
            raise ValueError("TCGA TPM input no longer matches the locked cohort")
    participants = pd.read_csv(directory / names[0], sep="\t", dtype={"participant": "str"})
    genes = pd.read_csv(directory / names[1], sep="\t")
    rna = np.load(directory / names[2], allow_pickle=False)
    use = genes.gene_type.eq("protein_coding") & genes.gene_name.notna() & ~genes.gene_name.duplicated(keep=False)
    if len(participants) != 368 or rna.shape != (len(participants), len(genes)):
        raise ValueError("TCGA adult primary roster changed")
    return pd.DataFrame(np.log2(1.+rna[:, use]), index=participants.participant,
                        columns=genes.loc[use, "gene_name"])


def patient_forecast(args, release):
    source = args.patient / "gdc-20220418-v1"
    expression = tcga_expression(source)
    roster = pd.DataFrame({"cell_id": expression.index, "lineage_group": expression.index})
    query = roster.merge(release.compounds, how="cross")
    query["row_id"] = "TCGA|" + query.cell_id + "|" + query.compound_id
    new, cov = predict_table(query, expression, release, {}, TCGA)
    store = args.out / "tcga_log2tpm.tsv"
    expression.rename_axis("sample_id").to_csv(store, sep="\t")
    earlier = []
    for method in ("v0plus_inductive", "drug_ridge"):
        next_path = args.out / f"tcga_{method}.tsv"
        old_settings = SimpleNamespace(package=args.v1_runs/"final-8e5d3f4/V0plus",
                                       model=method, compounds=None, rna=store, scale="bulk_log2_tpm_plus_1",
                                       gene_map=None, out=next_path)
        predict_offline(old_settings)
        prepared = pd.read_csv(next_path, sep="\t").rename(columns={"sample_id": "cell_id",
                                                                    "sensitivity_score": "predicted_value"})
        old = query[KEY].merge(prepared[["cell_id", "compound_id", "predicted_value"]],
                               on=["cell_id", "compound_id"], validate="one_to_one")
        if not np.isfinite(old.predicted_value).all():
            raise ValueError("Legacy V0+ did not produce matched TCGA predictions")
        earlier.append(old.assign(method=method, track=TCGA, fold=0))
    freeze_table(pd.concat([new, *earlier], ignore_index=True), args.out / "tcga.parquet",
                 provenance={"cohort_sha256": sha256(source / "receipt.json"),
                             "release_sha256": sha256(args.package/"release.json"),
                             "rna_min_coverage": float(cov.min().min()), "patient_outcome_access": False})


def forecast(args):
    args.out.mkdir(exist_ok=False, parents=True)
    release, _ = load_release(args.package)
    side_cohort(args, "broutier-e67f953", "Broutier_HCC_organoids", args.out / "broutier.parquet", release)
    side_cohort(args, "pdx-45ff510", PDX, args.out / "pdx.parquet", release)
    patient_forecast(args, release)


def correlated(truth, forecasts):
    observed, predicted = rankdata(truth, axis=-1), rankdata(forecasts, axis=-1)
    observed -= observed.mean(axis=-1, keepdims=True)
    predicted -= predicted.mean(axis=-1, keepdims=True)
    scale = np.sqrt(np.square(observed).sum(axis=-1)*np.square(predicted).sum(axis=-1))
    return np.divide((observed*predicted).sum(axis=-1), scale, out=np.zeros_like(scale), where=scale>0)


def pdx_score(runs, frozen):
    table, seal = verify_forecast(frozen)
    source = runs / "pdx-45ff510/outcomes"
    if sha256(source/"hu.pdf") != seal["provenance"]["figure"]:
        raise ValueError("PDX response figure changed since the source registration")
    order = (source/"figure4c_order.txt").read_text().split()
    measured = table.pivot(index="cell_id", columns="method", values="predicted_value")
    if (len(order) != 16 or set(order) != set(measured.index) or not np.isfinite(measured).all().all()):
        raise ValueError("Unexpected GSE90653 forecast or outcome roster")
    y = np.arange(1,17)
    p = measured.loc[order].to_numpy().T
    indices = np.random.default_rng(90653).integers(16, size=(10000,16))
    base = correlated(y,p)
    boot = correlated(y[indices],p[:,indices])
    names = measured.columns.tolist()
    old = names.index("v0plus_inductive")
    return {"rows": len(order), "bootstrap_n": len(indices), "receipts": {
        "freeze_sha256": seal["sha256"], "order_sha256": sha256(source/"figure4c_order.txt")},
        "methods": [{"method": name, "spearman": float(base[j]),
                     "ci95": np.quantile(boot[j],[.025,.975]).tolist(),
                     "delta_v1": float(base[j]-base[old]),
                     "paired_delta_ci95": np.quantile(boot[j]-boot[old],[.025,.975]).tolist()}
                    for j,name in enumerate(names)]}


def patient_score(args):
    from hcc_patient_eval.score import estimate_annotations, tests

    p, manifest = verify_forecast(args.out/"tcga.parquet")
    annot = args.patient/"annotations-v1"
    meta = json.loads((annot/"receipt.json").read_text())
    fn = annot/"annotations.parquet"
    check = meta["outputs"]["annotations.parquet"]
    if isinstance(check, dict):
        check = check["sha256"]
    if sha256(fn) != check:
        raise ValueError("Frozen TCGA annotation checksum differs")
    covariates = pd.read_parquet(fn)
    if "participant" in covariates:
        covariates = covariates.set_index("participant")
    covariates = covariates.join(estimate_annotations(annot), how="inner")
    source, _ = load_release(args.package)
    reg = source.compounds.set_index("compound_id")
    data = (p.rename(columns={"cell_id": "participant", "method": "arm", "predicted_value": "value"})
            .assign(preprocessing="fixed_inductive_log2tpm"))
    panels, contrasts, strata, purity, _ = tests(data, covariates, reg)
    for name, frame in (("panels",panels),("driver_subtype",contrasts),("strata",strata),("purity",purity)):
        frame.to_csv(args.out/f"tcga_{name}.tsv", sep="\t", index=False, na_rep="NA")
    comparisons = []
    fixed = data.pivot(index=["participant","compound_id"], columns="arm", values="value")
    for option in p.method.unique():
        if not option.startswith("v2_"):
            continue
        for tested in ("v0plus_inductive", "drug_ridge"):
            per_case = [spearmanr(m[option],m[tested]).statistic for _,m in fixed.groupby(level="participant")]
            comparisons.append({"v2": option, "comparator": tested, "patient_drug_rank_mean": float(np.mean(per_case))})
    result = {"patients": covariates.shape[0], "drugs": len(reg), "outcomes_available": False,
        "annotation_sha256": sha256(fn), "forecast_sha256": manifest["sha256"],
        "drug_rank_agreement": comparisons,
        "significant_fdr_05": [{"arm": arm, "contrast": c, "positive": int(n)} for (arm,c),n in
                              contrasts.assign(significant=contrasts.fdr.lt(.05)).groupby(
                                  ["arm","contrast"]).significant.sum().items()],
        "median_abs_partial_rho": [{"arm": arm, "marker": marker, "rho": float(value)}
                                   for (arm,marker), value in purity.assign(
                                       abs_rho=purity.partial_rank_rho.abs()).groupby(["arm","marker"]).abs_rho.median().items()]}
    return result


def score(args):
    score_name = args.out/"audit_score.json"
    if score_name.exists():
        raise FileExistsError(score_name)
    br, br_replay = verify_forecast(args.out/"broutier.parquet")
    labels = args.v1_runs/"broutier-e67f953/bundle/labels.parquet"
    if sha256(labels) != br_replay["provenance"]["labels"]:
        raise ValueError("Broutier observations changed after freeze")
    matched = join_observations(br, pd.read_parquet(labels))
    overview, _, details = axis_metrics(matched, ())
    detail = details.astype(object).where(details.notna(), None).to_dict("records")
    record = {"Broutier": {"rows": 14, "independent_organoids": 2, "full": overview, "details": detail},
              "PDX": pdx_score(args.v1_runs, args.out/"pdx.parquet"), "TCGA": patient_score(args)}
    publish_json(score_name, record)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("forecast","score"))
    for key in ("v1-runs","package","patient","out"):
        cli.add_argument("--"+key, type=Path, required=True)
    args = cli.parse_args()
    {"forecast": forecast, "score": score}[args.action](args)
