"""Score paired source-library and donor holdouts on common measured doses."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix, diags
from sklearn.linear_model import Ridge

from .evaluate import join_observations
from .external import KEY
from .io import freeze_table, publish_json, verify_forecast
from .licob_endpoints import summarize
from .limore_corrected import shared_support
from .oslo_public_curves import CURVE, OSLO_REGISTRATION
from .pdo_transfer_cv import device_cuda
from .v0plus_cv import molecular_inputs
from .yang_hcc_external import require_hash

BRIDGE = "935cd73545d42871fb8d757d28f6bb63ad0da88c0bc2a5ad378e3e4b76b48551"
LIBRARIES = ("lib1", "lib2")
ENDPOINTS = ("auc", "emax")
CONTROLS = ("drug_mean", "drug_ridge", "rna_kernel")
METHODS = (*CONTROLS, "pathway_kernel")
IDENTITY = ["sample_id", "compound_fimm", "lineage_group", "library_id"]


def positions(args):
    require_hash(args.inputs / "receipt.json", OSLO_REGISTRATION)
    require_hash(args.bridge, BRIDGE)
    meta = json.loads((args.inputs / "receipt.json").read_text())
    for name in ("counts.parquet", "dose_points.parquet"):
        require_hash(args.inputs / name, meta["outputs"][name])
    public = json.loads(args.bridge.read_text())
    codes = public["exact_products_with_ten_donors_each_library"]
    design = pd.read_parquet(args.inputs / "dose_points.parquet", columns=[*CURVE, "lineage_group", "dose_uM"],
                             filters=[("compound_fimm", "in", codes)])
    roles = design.groupby("lineage_group").library_id.nunique()
    if roles.max() != 1 or set(design.library_id) != set(LIBRARIES):
        raise ValueError("Both registered libraries must have disjoint donors")
    ends = design.groupby(CURVE).dose_uM.agg(["min", "max"]).reset_index()
    shared = ends.groupby("compound_fimm").agg(lo=("min", "max"), hi=("max", "min"))
    shared = shared[shared.hi.div(shared.lo).ge(10)]
    query = design[design.compound_fimm.isin(shared.index)][IDENTITY].drop_duplicates()
    if len(shared) < 10 or min(query.groupby("library_id").lineage_group.nunique()) < 20:
        raise ValueError("Insufficient donor/library support after physical dose matching")
    return identities(query), shared


def identities(rows):
    roster = rows.rename(columns={"sample_id": "cell_id", "compound_fimm": "compound_id"}).copy()
    roster["row_id"] = roster.cell_id + "|" + roster.compound_id
    return roster


def outcomes(args, library, shared):
    points = pd.read_parquet(args.inputs / "dose_points.parquet",
        filters=[("compound_fimm", "in", shared.index.tolist()), ("library_id", "==", library)])
    results = []
    for _, raw in points.groupby(CURVE, sort=False):
        probe = raw.iloc[0]
        window = shared.loc[probe.compound_fimm]
        raw = raw.sort_values("dose_uM")
        log_dose = np.log10(raw.dose_uM)
        if log_dose.iloc[0] > np.log10(window.lo) or log_dose.iloc[-1] < np.log10(window.hi):
            raise ValueError("A raw curve fails to cover the registered common dose window")
        network = np.linspace(np.log10(window.lo), np.log10(window.hi), 33)
        observed = -np.interp(network, log_dose, np.clip(raw.viability, 0, 1))
        results.append([*(probe[key] for key in IDENTITY),
                        trapezoid(observed, x=network) / np.ptp(network), observed[-1]])
    measured = pd.DataFrame(results, columns=[*IDENTITY, *ENDPOINTS])
    return identities(measured.groupby(IDENTITY, as_index=False)[list(ENDPOINTS)].median())


def cell_grams(rna, held, edges, device):
    from .structured_kernel import Standardization, dot_kernel, radial, ranked_expression

    donors, queries = (ranked_expression(frame) for frame in (rna, held))
    scale = Standardization.fit(donors, .05)
    fit, query = (scale.apply(frame, device) for frame in (donors, queries))
    energy = fit.square().mean(dim=1)
    normalizer = energy.mean()
    raw = dot_kernel(fit, fit) / normalizer
    cross = dot_kernel(query, fit) / normalizer
    norm = energy / normalizer
    baseline = (.5 * (raw + radial(raw, norm, norm, 2)),
                .5 * (cross + radial(cross, query.square().mean(dim=1) / normalizer, norm, 2)))
    edges = edges[edges.gene.isin(rna.columns)]
    counts = edges.groupby("signature").gene.nunique()
    edges = edges[edges.signature.isin(counts[counts.ge(10)].index)]
    names = pd.Index(sorted(edges.signature.unique()))
    weights = csr_matrix((edges.weight.to_numpy(dtype=np.float32),
                         (rna.columns.get_indexer(edges.gene), names.get_indexer(edges.signature))),
                         shape=(len(rna.columns), len(names)))
    weights = weights @ diags(1 / np.sqrt(np.asarray(weights.power(2).sum(axis=0)).ravel()))
    signature = Standardization.fit(donors @ weights, .05)
    p, q = (signature.apply(x @ weights, device) for x in (donors, queries))
    enriched = (.7 * baseline[0] + .3 * dot_kernel(p, p), .7 * baseline[1] + .3 * dot_kernel(q, p))
    index = np.argsort(-donors.std(axis=0), kind="stable")[:512]
    return baseline, enriched, fit[:, index].cpu().numpy(), query[:, index].cpu().numpy(), len(names)


def forecasts(train, roster, expression, edges, endpoint):
    import torch
    from .structured_kernel import solve_residuals

    device = device_cuda()
    cells, held = (pd.Index(sorted(x.cell_id.unique())) for x in (train, roster))
    matrices = cell_grams(expression.loc[cells], expression.loc[held], edges, device)
    simple, enriched, fit_rna, test_rna, paths = matrices
    drugs = pd.Index(sorted(train.compound_id.unique()))
    donors = train.groupby(["lineage_group", "compound_id"])[endpoint].transform("size")
    means = train.groupby(["lineage_group", "compound_id"])[endpoint].mean().groupby("compound_id").mean()
    row, col = cells.get_indexer(train.cell_id), drugs.get_indexer(train.compound_id)
    ii, jj = held.get_indexer(roster.cell_id), drugs.get_indexer(roster.compound_id)
    if (jj < 0).any() or set(train.lineage_group) & set(roster.lineage_group):
        raise ValueError("Unsupported compound or overlapping patient")
    delta = train[endpoint].to_numpy() - train.compound_id.map(means).to_numpy()
    weight = torch.zeros((len(cells), len(drugs)), device=device)
    response = torch.zeros_like(weight)
    weight[row, col] = torch.as_tensor(1 / donors.to_numpy(), dtype=weight.dtype, device=device)
    response[row, col] = torch.as_tensor(delta, dtype=weight.dtype, device=device)
    drug = .8 * torch.eye(len(drugs), device=device) + .2
    anchor = roster.compound_id.map(means).to_numpy()
    ridge = np.zeros(len(roster))
    for j in range(len(drugs)):
        source, target = col == j, jj == j
        model = Ridge(alpha=1000).fit(fit_rna[row[source]], delta[source], sample_weight=1 / donors.to_numpy()[source])
        ridge[target] = model.predict(test_rna[ii[target]])
    result = {"drug_mean": anchor, "drug_ridge": anchor + ridge}
    diagnostics = {"signatures": paths}
    for name, (fit_gram, cross) in (("rna_kernel", simple), ("pathway_kernel", enriched)):
        coefficient, steps, norm = solve_residuals(fit_gram, drug, weight, response, 4.)
        result[name] = anchor + (cross @ coefficient @ drug).cpu().numpy()[ii, jj]
        diagnostics[name] = {"iterations": steps, "residual_norm": norm}
    return pd.concat([roster[KEY].assign(method=name, predicted_value=values) for name, values in result.items()]), diagnostics


def fit(args):
    roster, shared = positions(args)
    expression = pd.read_parquet(args.inputs / "counts.parquet")
    edges, _, provenance = molecular_inputs(args.priors, args.hallmark, args.targets)
    parts, stats = [], {}
    for library in LIBRARIES:
        source = outcomes(args, library, shared)
        held = roster[roster.library_id.ne(library)]
        for endpoint in ENDPOINTS:
            frame, evidence = forecasts(source, held, expression, edges, endpoint)
            shared_support(frame, held, METHODS)
            track = f"{library}_{endpoint}"
            parts.append(frame.assign(track=track, fold=0))
            stats[track] = evidence
    freeze_table(pd.concat(parts, ignore_index=True), args.run / "forecast.parquet", provenance={
        "inputs_sha256": OSLO_REGISTRATION, "bridge_sha256": BRIDGE, "dose_windows": shared.to_dict("index"),
        "prior_sources": provenance, "fit": stats,
    })


def evaluate_gate(result):
    rank = {(row["baseline"], row["metric"]): row["ci95"][0] for row in result["metrics"]["paired_contrasts"]}
    return {method: all(result["within_drug_ci95"][f"pathway_kernel/{method}"][metric][0] > 0
                        for metric in ("spearman", "pearson"))
            and all(rank[method, metric] > -.005 for metric in ("spearman", "ndcg_at_5", "top5_overlap"))
            for method in CONTROLS}


def score(args):
    roster, shared = positions(args)
    rows, seal = verify_forecast(args.run / "forecast.parquet")
    proof = seal["provenance"]
    if (proof["inputs_sha256"], proof["bridge_sha256"], proof["dose_windows"]) != (
        OSLO_REGISTRATION, BRIDGE, shared.to_dict("index")
    ):
        raise ValueError("Forecast does not use the fixed common dose windows")
    results = {}
    for source, held in (LIBRARIES, LIBRARIES[::-1]):
        test = outcomes(args, held, shared)
        for endpoint in ENDPOINTS:
            track = f"{source}_{endpoint}"
            fitted = rows[rows.track.eq(track)]
            shared_support(fitted, roster[roster.library_id.eq(held)], METHODS)
            joined = join_observations(fitted, test[KEY].assign(endpoint_value=test[endpoint]))
            donors = joined.groupby(["track", "fold", "method", "lineage_group", "compound_id"], as_index=False)[
                ["predicted_value", "endpoint_value"]].mean()
            reference = fitted[fitted.method.eq("drug_mean")].groupby("compound_id").predicted_value.first()
            anchor = donors.method.eq("drug_mean")
            donors.loc[anchor, "predicted_value"] = donors.loc[anchor, "compound_id"].map(reference)
            donors = donors.assign(cell_id=donors.lineage_group, row_id=donors.lineage_group + "|" + donors.compound_id)
            result = summarize(donors, [("pathway_kernel", name) for name in CONTROLS])
            if endpoint == "auc":
                result["gate"] = evaluate_gate(result)
            results[track] = result
    publish_json(args.run / "score.json", {"forecast_sha256": seal["sha256"], "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, function, fields in (("fit", fit, ("priors", "hallmark", "targets")), ("score", score, ())):
        stage = stages.add_parser(name)
        for field in ("run", "inputs", "bridge", *fields):
            stage.add_argument(f"--{field}", type=Path, required=True)
        stage.set_defaults(execute=function)
    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()
