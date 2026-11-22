"""Fit vinorelbine sensitivity across public assays and score held HCC PDX models."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch

from .cross_source_data import gdsc_rows, read_raw, response_rows, roster_table
from .expanded_panel_cv import registered_inputs
from .io import freeze_table, publish_json, sha256, verify_forecast
from .shuen_pdx import agreements, donor_axis, response_sources
from .structured_kernel import Standardization, dot_kernel, fit_cell_kernels, ranked_expression
from .v0plus_cli import read_expression
from .v0plus_cv import molecular_inputs
from .v0plus_features import ExpressionFeatures
from .v0plus_neural import load_auxiliary


DRUG = "GBABOYUKABKIAF-IELIFDKJSA-N"
GENES = ("TUBB", "TUBB3", "ABCB1", "ABCC10", "DLGAP5", "FBXW7", "RB1", "NF2")
ASSAYS = {"GDSC1": "140", "GDSC2": "2048", "PRISM20Q2": "BRD-K69280563-001-01-8"}
EXTENSION_SHA = "832b92789f6e5c0f1c7e39607dee9a0c78f71b4c7725948af51e06e09b1ceecf"
PDX_SHA = "73795c04ee31bfccafa928451ded763283beb5b805541d72810e18ff11301957"
BASES = ("mean", "ridge", "full")
PRIMARY = "mechanism"
GRIDS = {"ridge": (1., 16., 256., 4096.)} | {
    name: (.05, .5, 4., 16.) for name in ("rna", "full", PRIMARY)
}


def observations(args, core, extra):
    inputs = json.loads((args.original / "receipt.json").read_text())["inputs"]
    roles = ("gdsc1", "gdsc2", "gdsc_labels", "depmap_models")
    for name in roles:
        item = inputs[name]
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Registered source changed: {name}")
    official = read_raw(inputs["gdsc_labels"]["path"], ["DRUG_ID", "DRUG_NAME", "SYNONYMS"])
    depmap = read_raw(inputs["depmap_models"]["path"], ["ModelID", "SangerModelID", "COSMICID"])
    # DepMap serializes integer COSMIC accessions as decimal strings in this release.
    depmap["COSMICID"] = depmap.COSMICID.str.replace(r"\.0+$", "", regex=True)
    depmap = depmap[depmap.ModelID.isin(core.cell_id)
                    & depmap.SangerModelID.ne("") & depmap.COSMICID.ne("")]
    if depmap.duplicated(["SangerModelID", "COSMICID"]).any():
        raise ValueError("Ambiguous DepMap Sanger–COSMIC join")
    all_rows = []
    for assay in ("GDSC1", "GDSC2"):
        native = ASSAYS[assay]
        info = official.loc[official.DRUG_ID.eq(native)]
        if (len(info) != 1 or not info.DRUG_NAME.eq("Vinorelbine").all()
                or not info.SYNONYMS.str.contains("vinorelbine tartrate", case=False).all()):
            raise ValueError(f"Unexpected native identity: {assay}")
        path = inputs[assay.lower()]["path"]
        header = read_raw(path, ["SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME"])
        header = header[header.DRUG_ID.eq(native)]
        if not header.DRUG_NAME.eq("Vinorelbine").all():
            raise ValueError(f"Unexpected fitted name: {assay}")
        roster = roster_table(header, assay, "CELL_LINE_NAME", "SANGER_MODEL_ID", "COSMIC_ID")
        roster = roster.merge(depmap, left_on=["source_key", "source_key_2"],
                              right_on=["SangerModelID", "COSMICID"], validate="many_to_one")
        lookup = pd.DataFrame({"native_compound_id": [native], "compound_id": [DRUG]})
        raw = gdsc_rows(path, roster.rename(columns={"ModelID": "cell_id"}), lookup)
        raw["source_run"] = raw.source_run.str.replace("GDSC2_", assay + "_", regex=False)
        rows, _ = response_rows(raw, assay, assay, 72, "native fitted AUC", ["dose_min_uM", "dose_max_uM"])
        all_rows.append(rows)
    prism = extra[extra.source_id.eq("PRISM20Q2") & extra.compound_id.eq(DRUG)]
    if len(prism) != 462 or not prism.native_compound_id.eq(ASSAYS["PRISM20Q2"]).all():
        raise ValueError("Unexpected PRISM vinorelbine support")
    all_rows.append(prism)
    fields = ["source_id", "source_run", "cell_id", "endpoint_value"]
    roster = core[["cell_id", "lineage_group", "lineage_role"]].drop_duplicates().set_index("cell_id")
    data = pd.concat([part[fields] for part in all_rows], ignore_index=True)
    data = data.join(roster, on="cell_id", validate="many_to_one")
    if (set(data.source_id) != set(ASSAYS) or data.duplicated(["cell_id", "source_id"]).any()
            or data.lineage_group.isna().any()
            or not np.isfinite(data.endpoint_value).all() or data.groupby("source_id").size().min() < 100):
        raise ValueError("Inadequate and inconsistent Vinorelbine source support")
    return data.assign(sensitivity=-data.endpoint_value), {key: inputs[key]["sha256"] for key in roles}


def basis(rna, query, edges, auxiliary, device):
    fitted, train = fit_cell_kernels(rna, edges, auxiliary, device=device)
    held = fitted.cross(query)
    if not set(GENES) <= set(rna.columns):
        raise ValueError("A registered vinorelbine gene is missing")
    lookup = rna.columns.get_indexer(GENES)
    rx, rq = ranked_expression(rna)[:, lookup], ranked_expression(query)[:, lookup]
    scaling = Standardization.fit(rx, .05)
    rx, rq = scaling.apply(rx, device), scaling.apply(rq, device)
    for dest, left in ((train, rx), (held, rq)):
        dest[PRIMARY] = .75 * dest["full"] + .25 * dot_kernel(left, rx)
    features = ExpressionFeatures.fit(rna, edges, {DRUG: GENES}, n_variable=768, rank=True)
    x = torch.as_tensor(features.transform(rna).to_numpy(), device=device)
    z = torch.as_tensor(features.transform(query).to_numpy(), device=device)
    train["ridge"], held["ridge"] = x @ x.T, z @ x.T
    return train, held


def fit_grid(rows, ids, gram, cross, penalties):
    outcomes = rows.groupby("source_id").sensitivity
    size = outcomes.transform("size")
    spread = outcomes.transform(lambda v: v.std(ddof=0))
    if (spread < 1e-6).any():
        raise ValueError("A fitting assay has a constant endpoint")
    scaled = (rows.sensitivity - outcomes.transform("mean")) / spread
    weight = len(rows) / (size * rows.source_id.nunique())
    joined = rows.assign(total=weight, weighted=scaled * weight).groupby("cell_id").agg(
        total=("total", "sum"), weighted=("weighted", "sum"),
    )
    positions = ids.get_indexer(joined.index)
    if (positions < 0).any():
        raise ValueError("Assay cells are missing from the fit Gram matrix")
    idx = torch.as_tensor(positions, device=gram.device)
    root = torch.as_tensor(joined.total.to_numpy(), device=gram.device, dtype=gram.dtype).sqrt()
    y = torch.as_tensor(joined.weighted.to_numpy(), device=gram.device, dtype=gram.dtype) / root
    k = gram[idx[:, None], idx] * (root[:, None] * root)
    query = cross[:, idx] * root
    identity = torch.eye(len(idx), device=k.device, dtype=k.dtype)
    return {penalty: (query @ torch.linalg.solve(k + penalty * identity, y)).cpu().numpy()
            for penalty in penalties}


def dev_select(data, rna, edges, auxiliary, device):
    roster = data[["cell_id", "lineage_group"]].drop_duplicates().sort_values("cell_id")
    evaluated = defaultdict(list)
    for train_idx, test_idx in GroupKFold(3).split(roster, groups=roster.lineage_group):
        train_ids, test_ids = (pd.Index(roster.iloc[i].cell_id) for i in (train_idx, test_idx))
        for assay in ASSAYS:
            known = data[data.cell_id.isin(train_ids) & data.source_id.ne(assay)]
            hold = data[data.cell_id.isin(test_ids) & data.source_id.eq(assay)]
            fit_ids = pd.Index(sorted(known.cell_id.unique()))
            hold_ids = pd.Index(hold.cell_id)
            fit, cross = basis(rna.loc[fit_ids], rna.loc[hold_ids], edges, auxiliary, device)
            for name, penalties in GRIDS.items():
                for penalty, output in fit_grid(known, fit_ids, fit[name], cross[name], penalties).items():
                    evaluated[name, penalty, assay].append(
                        hold[["sensitivity", "lineage_role"]].assign(prediction=output),
                    )
    records = []
    for (name, penalty, assay), values in evaluated.items():
        block = pd.concat(values, ignore_index=True)
        records.append({"method": name, "penalty": penalty, "assay": assay,
                        **correlations(block), "adult_hcc": correlations(block[block.lineage_role.eq("adult_hcc")])})
    table = pd.DataFrame(records)
    overall = table.groupby(["method", "penalty"], as_index=False).spearman.mean()
    best = overall.sort_values(["spearman", "penalty"], ascending=[False, False]).drop_duplicates("method")
    return best.set_index("method").penalty.to_dict(), records


def correlations(block):
    values = {"n": len(block)}
    for kind in ("spearman", "pearson"):
        measured = block[["sensitivity", "prediction"]].corr(method=kind).iloc[0, 1]
        values[kind] = float(measured) if np.isfinite(measured) else None
    return values


def observed_agreement(data):
    results = []
    for left, right in combinations(ASSAYS, 2):
        wide = data[data.source_id.isin([left, right])].pivot(
            index=["cell_id", "lineage_role"], columns="source_id", values="sensitivity",
        ).dropna(subset=[left, right]).reset_index().rename(columns={left: "sensitivity", right: "prediction"})
        results.append({"left": left, "right": right, **correlations(wide),
                        "adult_hcc": correlations(wide[wide.lineage_role.eq("adult_hcc")])})
    return results


def prepare_query(pdx, fit_rna):
    _, seal = verify_forecast(pdx / "forecast.parquet")
    if seal["sha256"] != PDX_SHA:
        raise ValueError("Unexpected preexisting PDX RNA contract")
    counts, lengths = (pdx / f"{name}.tsv" for name in ("counts", "lengths"))
    for key, path in (("counts_sha256", counts), ("length_sha256", lengths)):
        if seal["provenance"][key] != sha256(path):
            raise ValueError("Previously verified PDX normalization has changed")
    forms = {}
    for name, length in (("cpm", None), ("length_tpm", lengths)):
        form = read_expression(counts, "bulk_counts", None, length).reindex(columns=fit_rna.columns)
        if len(form) != 17 or form.notna().mean(axis=1).min() < .85:
            raise ValueError("Invalid PDX RNA coverage")
        forms[name] = form
    return forms


def predict(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Vinorelbine source fits need the assigned G1 GPU")
    device = torch.device("cuda")
    extension_receipt = args.extension / "receipt.json"
    if sha256(extension_receipt) != EXTENSION_SHA:
        raise ValueError("Unexpected PRISM corrected extension")
    provenance = json.loads(extension_receipt.read_text())
    _, core, rna, _ = registered_inputs(SimpleNamespace(original=args.original, core=args.core))
    table = args.extension / "additional_rows.parquet"
    if (provenance["original_bundle_sha256"] != sha256(args.original / "receipt.json")
            or provenance["artifacts"][table.name] != sha256(table)):
        raise ValueError("Corrected PRISM rows differ from their registration")
    extension = pd.read_parquet(table, filters=[("compound_id", "==", DRUG)])
    data, source_hashes = observations(args, core, extension)
    edges, _, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    ids = pd.Index(sorted(data.cell_id.unique()))
    rna = rna.loc[ids]
    selected, tuning = dev_select(data, rna, edges, auxiliary, device)
    forms = prepare_query(args.pdx, rna)
    query = pd.concat(forms, names=["track", "row_id"])
    fit, cross = basis(rna, query.fillna(rna.median()), edges, auxiliary, device)
    predictions = {"mean": np.zeros(sum(map(len, forms.values())))}
    for method, penalty in selected.items():
        predictions[method] = fit_grid(data, ids, fit[method], cross[method], [penalty])[penalty]
    for assay in ("GDSC2", "PRISM20Q2"):
        penalty = selected["full"]
        known = data[data.source_id.eq(assay)]
        assay_ids = pd.Index(sorted(known.cell_id.unique()))
        assay_rna = rna.loc[assay_ids]
        train, held = basis(assay_rna, query.fillna(assay_rna.median()), edges, auxiliary, device)
        predictions[assay] = fit_grid(known, assay_ids, train["full"], held["full"], [penalty])[penalty]
    query_index = query.index.to_frame(index=False)
    frames = [query_index.assign(method=method, fold=0, predicted_value=value) for method, value in predictions.items()]
    args.run.mkdir(exist_ok=True)
    freeze_table(pd.concat(frames, ignore_index=True), args.run / "pdx.parquet", provenance={
        "source_hashes": source_hashes, "extension_sha256": EXTENSION_SHA, "pdx_rna_from": PDX_SHA,
        "source_support": {name: {"cells": len(group), "hcc": int(group.lineage_role.eq("adult_hcc").sum()),
                                  "protocol": sorted(group.source_run.unique())}
                           for name, group in data.groupby("source_id")},
        "priors": priors, "auxiliary": aux_receipt["endpoints"], "selected": selected, "source_cv": tuning,
        "source_observed_agreement": observed_agreement(data),
    })


def score(args):
    rows, seal = verify_forecast(args.run / "pdx.parquet")
    source = response_sources(args.sources)
    labels = {year: source[year].query("drug == 'vinorelbine'").assign(response=lambda d: -d.tc)
              for year in ("2023", "2024")}
    methods = sorted(rows.method.unique())
    results, checks = {}, {}
    for normalization, part in rows.groupby("track"):
        wide = part.pivot(index="row_id", columns="method", values="predicted_value")
        matched = {year: table.merge(wide, left_on="model_id", right_index=True, validate="one_to_one")
                   for year, table in labels.items()}
        shared_models = set.intersection(*(set(frame.model_id) for frame in matched.values()))
        for year, table in matched.items():
            shared = table[table.model_id.isin(shared_models)]
            for scope, frame in (("all", table), ("common", shared)):
                data = donor_axis(frame[["model_id", "response", *methods]])
                if len(data) < 6:
                    raise ValueError("Insufficient support for paired PDX scoring")
                name = f"{normalization}/{year}/{scope}"
                changes = {method: agreements(data, PRIMARY, "response", method) for method in BASES}
                results[name] = {"n": len(data), "models": len(frame), "scores": {
                    method: agreements(data, method, "response") for method in methods}, "contrasts": changes}
                if normalization == "cpm" and scope == "common":
                    checks[year] = all(stat["ci95"][0] > 0 for comp in changes.values() for stat in comp.values())
    publish_json(args.run / "score.json", {"freeze_sha256": seal["sha256"], "comparisons": results,
                                            "primary_checks": checks, "pass": all(checks.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, handler, paths in (
        ("predict", predict, ("original", "core", "extension", "pdx", "priors", "hallmark", "targets", "auxiliary")),
        ("score", score, ("sources",)),
    ):
        stage = stages.add_parser(name)
        for arg in ("run", *paths):
            stage.add_argument("--" + arg, type=Path, required=True)
        stage.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
