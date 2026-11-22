"""Measure whether public CRC organoid response adds to a frozen 2D HCC model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.integrate import trapezoid

from .external import KEY, SOURCE, native_sensitivity
from .function_data import canonical_chemistry
from .io import freeze_table, publish_json, verify_forecast
from .licob_cold import checked as checked_licob
from .licob_curves import DOSE_REGISTRATION
from .licob_dose_sources import COLD, registration
from .licob_endpoints import PRIOR, paired_donors, read_outcomes, source_curves, summarize
from .limore_corrected import shared_support
from .oslo_library_transfer import forecasts
from .oslo_public_curves import CURVE, OSLO_REGISTRATION
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES
from .v0plus_score import axis_metrics, compound_strata
from .yang_hcc_external import require_hash

CHEM = "edebd1872873c379b0f208175c9ca86aa97ca23f298784189465a818d566e535"
ENDPOINT = "5cc6c93c5ccc9adf4b6e9f6ec271e3031835bda8928bf444af5b098b0493e15a"
BASELINES = ("chemical_mean", "neighbor_transfer", "interaction_transfer")
EMAX_BASELINES = ("drug_mean", "matched_endpoint")
ADDED = ("oslo_ridge", "oslo_rna", "oslo_pathway", "combined")
PRIMARY = "combined"


def register_chemistry(args):
    receipt = args.oslo / "receipt.json"
    require_hash(receipt, OSLO_REGISTRATION)
    meta = json.loads(receipt.read_text())
    for name in ("counts.parquet", "dose_points.parquet", "name_overlap.tsv"):
        require_hash(args.oslo / name, meta["outputs"][name])
    require_hash(args.chemistry, CHEM)
    evidence = pd.read_csv(args.chemistry, sep="\t")
    candidates = pd.read_csv(args.oslo / "name_overlap.tsv", sep="\t").merge(
        evidence, left_on="compound_name", right_on="name", validate="many_to_one")
    accepted, excluded = [], []
    for name in evidence.itertuples(index=False):
        path = args.pubchem / f"{name.cas}.json"
        require_hash(path, name.sha256)
        properties = json.loads(path.read_text())["PropertyTable"]["Properties"]
        if len(properties) != 1 or properties[0]["CID"] != name.cid:
            raise ValueError("CAS lookup failed to identify the registered PubChem record")
        structure, parent, *_ = canonical_chemistry(properties[0]["SMILES"])
        if set(candidates[candidates.name.eq(name.name)].compound_id) != {parent}:
            raise ValueError("The normalized PubChem parent and LICOB parent differ")
        sites = Chem.FindPotentialStereo(Chem.MolFromSmiles(structure))
        if any(site.specified == Chem.StereoSpecified.Unspecified for site in sites):
            excluded.append(name.name)
        else:
            accepted.append(name.name)
    if len(accepted) != 13 or excluded != ["Erastin"]:
        raise ValueError("Unexpected isomer-resolved chemical overlap")
    return candidates[candidates.name.isin(accepted)], excluded


def oslo_endpoints(args, mapping, dose, endpoint):
    raw = pd.read_parquet(args.oslo / "dose_points.parquet",
                         filters=[("compound_fimm", "in", mapping.compound_fimm.tolist())])
    raw = raw.merge(mapping[["compound_fimm", "compound_id"]], on="compound_fimm", validate="many_to_one")
    maxima = dose.set_index("compound_id").maximum
    rows = []
    for _, part in raw.groupby(CURVE, sort=False):
        part = part.sort_values("dose_uM")
        tag = part.iloc[0]
        doses = np.log10(part.dose_uM.to_numpy())
        curve = -np.clip(part.viability.to_numpy(), 0, 1)
        if endpoint == "auc":
            value = trapezoid(curve, x=doses) / np.ptp(doses)
        else:
            highest = np.log10(maxima[tag.compound_id])
            if highest < doses[0] or highest > doses[-1]:
                continue
            value = np.interp(highest, doses, curve)
        rows.append((tag.sample_id, tag.lineage_group, tag.compound_id, tag.library_id, tag.compound_fimm, value))
    key = ["cell_id", "lineage_group", "compound_id", "library_id", "compound_fimm"]
    table = pd.DataFrame(rows, columns=[*key, "response"]).groupby(key, as_index=False).response.median()
    levels = ["library_id", "compound_fimm", "compound_id"]
    patients = table.groupby([*levels, "lineage_group"], as_index=False).response.mean()
    std = patients.groupby(levels).response.agg(["mean", "std"])
    table = table.join(std, on=levels)
    table["standardized"] = (table.response - table["mean"]) / table["std"].fillna(0.).clip(lower=.05)
    identities = ["cell_id", "lineage_group", "compound_id"]
    return table.groupby(identities, as_index=False).standardized.median()


def source_scales(args, old, codes, endpoint):
    if endpoint == "auc":
        source = pd.read_parquet(args.cold / "train.parquet", filters=[("compound_id", "in", sorted(codes))])
        slopes = pd.DataFrame(old["provenance"]["model"]["calibration"])[[*SOURCE, "slope"]]
        source = source.merge(slopes, on=SOURCE, validate="many_to_one")
        source["value"] = native_sensitivity(source) * source.slope
    else:
        source, _, _ = source_curves(args.dose)
        source = source[source.compound_id.isin(codes)].assign(value=lambda r: -r.emax)
    sds = source.groupby([*SOURCE, "compound_id"]).value.std().groupby("compound_id").mean()
    if set(sds.index) != set(codes) or not np.isfinite(sds).all() or not sds.gt(0).all():
        raise ValueError("An eligible parent lacks a valid dispersion in 2D sources")
    return sds


def augment(table, roster, annotated, source, x, edges, scales, endpoint):
    if endpoint == "auc":
        controls = BASELINES
        prior = "interaction_transfer"
    else:
        controls = EMAX_BASELINES
        prior = "matched_endpoint"
    existing = table[table.method.isin(controls) & table.row_id.isin(roster.row_id)]
    shared_support(existing, roster, controls)
    estimates, details = forecasts(source, annotated, x, edges, "standardized")
    values = estimates.pivot(index="row_id", columns="method", values="predicted_value")
    values = values.reindex(roster.row_id)
    conversion = roster.compound_id.map(scales).fillna(0.).to_numpy()
    offsets = values[["drug_ridge", "rna_kernel", "pathway_kernel"]].sub(values.drug_mean, axis=0).fillna(0.)
    offsets = offsets.to_numpy() * conversion[:, None]
    comparator = existing.pivot(index="row_id", columns="method", values="predicted_value").reindex(roster.row_id)
    anchor = comparator[controls[0]].to_numpy()
    full = {name: anchor + offsets[:, j] for j, name in enumerate(ADDED[:-1])}
    full[PRIMARY] = comparator[prior].to_numpy() + .5 * offsets[:, 1]
    extra = pd.concat([roster[KEY].assign(method=name, predicted_value=y) for name, y in full.items()], ignore_index=True)
    predictions = pd.concat([existing, extra], ignore_index=True).assign(track=endpoint, fold=0)
    shared_support(predictions, roster, (*controls, *ADDED))
    return predictions, details


def fit(args):
    if checked_licob(args.cold, ("query", "rna", "train")) != COLD or registration(args.dose / "inputs") != DOSE_REGISTRATION:
        raise ValueError("Unexpected HCC target registration")
    mapping, removed = register_chemistry(args)
    query = pd.read_parquet(args.cold / "query.parquet")
    dose = pd.read_parquet(args.dose / "inputs/dose.parquet")
    old, freeze = verify_forecast(args.baseline)
    old_dose, dose_freeze = verify_forecast(args.dose / "forecast.parquet")
    if freeze["sha256"] != PRIOR or dose_freeze["sha256"] != ENDPOINT:
        raise ValueError("Unexpected frozen 2D references")
    rna = pd.read_parquet(args.oslo / "counts.parquet")
    held = pd.read_parquet(args.cold / "rna.parquet").loc[sorted(query.cell_id.unique())]
    axis = held.columns.intersection(rna.columns)
    if len(axis) != 14421:
        raise ValueError("Unexpected common Oslo–LICOB RNA panel")
    x = pd.concat([rna[axis], held[axis]], verify_integrity=True)
    edges, _, signatures = molecular_inputs(args.priors, args.hallmark, args.targets)
    packs, info = [], {}
    for endpoint, prior in (("auc", old), ("emax", old_dose[old_dose.track.eq("emax")])):
        train = oslo_endpoints(args, mapping, dose, endpoint)
        count = train.groupby("compound_id").lineage_group.nunique()
        chosen = set(count[count.ge(20)].index)
        if len(chosen) != (13 if endpoint == "auc" else 3):
            raise ValueError(f"Unexpected number of Oslo-supported drugs for {endpoint}")
        train = train[train.compound_id.isin(chosen)]
        annotated = query[query.compound_id.isin(chosen)]
        scale = source_scales(args, freeze, chosen, endpoint)
        part, diagnostics = augment(prior, query if endpoint == "auc" else annotated, annotated,
                                    train, x, edges, scale, endpoint)
        packs.append(part)
        names = query.drop_duplicates("compound_id").set_index("compound_id").source_compound_label
        info[endpoint] = {"fit_rows": len(train), "compounds": names.reindex(sorted(chosen)).to_dict(),
            "oslo_donors_per_compound": count.reindex(sorted(chosen)).to_dict(),
            "aligned_2d_spread": scale.to_dict(), "adaptation": diagnostics}
    freeze_table(pd.concat(packs, ignore_index=True), args.out / "forecast.parquet", provenance={
        "oslo_sha256": OSLO_REGISTRATION, "cold_sha256": COLD, "dose_sha256": DOSE_REGISTRATION, "chemistry_sha256": CHEM,
        "prior_sha256": {"auc": PRIOR, "emax": ENDPOINT}, "source_signatures": signatures, "excluded": removed, "panels": info})


def gate(result):
    eligible = min(x["n_drugs"] for x in result["metrics"]["across_cell_scores"])
    if eligible < 10 or result["donors"] < 20:
        return {name: "insufficient_support" for name in BASELINES}
    rank = {(x["baseline"], x["metric"]): x["ci95"][0] for x in result["metrics"]["paired_contrasts"]}
    return {name: all(result["within_drug_ci95"][f"{PRIMARY}/{name}"][metric][0] > 0
                      for metric in ("spearman", "pearson"))
            and all(rank[name, metric] > -.005 for metric in ("spearman", "ndcg_at_5", "top5_overlap"))
            for name in BASELINES}


def score(args):
    rows, freeze = verify_forecast(args.out / "forecast.parquet")
    proof = freeze["provenance"]
    if (proof["oslo_sha256"], proof["cold_sha256"], proof["dose_sha256"], proof["chemistry_sha256"]) != (
        OSLO_REGISTRATION, checked_licob(args.cold, ("query",)), registration(args.dose / "inputs"), CHEM
    ) or (proof["cold_sha256"], proof["dose_sha256"]) != (COLD, DOSE_REGISTRATION):
        raise ValueError("Unexpected prediction/input registration")
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    query = pd.read_parquet(args.cold / "query.parquet")
    dose = pd.read_parquet(args.dose / "inputs/dose.parquet")
    labels = read_outcomes(args.supplement, query, dose, selected=("auc", "emax"))
    cases = {}
    for endpoint in ("auc", "emax"):
        panel = proof["panels"][endpoint]
        held = query[query.compound_id.isin(panel["compounds"])]
        part = rows[rows.track.eq(endpoint)]
        shared_support(part, query if endpoint == "auc" else held,
                       (*(BASELINES if endpoint == "auc" else EMAX_BASELINES), *ADDED))
        donors = paired_donors(part, labels[endpoint])
        comparisons = [(PRIMARY, name) for name in (BASELINES if endpoint == "auc" else EMAX_BASELINES)]
        if endpoint == "auc":
            case = summarize(donors[donors.compound_id.isin(panel["compounds"])], comparisons, compound_strata(held))
            case["gate"] = gate(case)
            cases["auc_common"] = case
            cases["auc_full"] = summarize(donors, comparisons)
        else:
            summary, per_drug, _ = axis_metrics(donors, ())
            cases["emax_support"] = {"within_drug": summary["across_cell_scores"], "per_drug": per_drug.to_dict("records")}
    publish_json(args.out / "score.json", {"freeze_sha256": freeze["sha256"],
                 "label_sha256": ASSET_HASHES["supplement"], "results": cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, function, more in (
        ("fit", fit, ("oslo", "chemistry", "pubchem", "baseline", "priors", "hallmark", "targets")),
        ("score", score, ("supplement",)),
    ):
        sub = stages.add_parser(name)
        for field in ("out", "cold", "dose", *more):
            sub.add_argument(f"--{field}", type=Path, required=True)
        sub.set_defaults(execute=function)
    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()
