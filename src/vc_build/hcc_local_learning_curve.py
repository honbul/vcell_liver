"""Separate target-assay sensitivity learning from cross-assay transfer."""

import argparse
from hashlib import sha256 as hash_text
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .culture_consistency import kernel_between
from .evaluate import join_observations
from .external import KEY, _validate
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import checked
from .licob_dose_sources import COLD
from .licob_icc_bridge import affine_map, corrections
from .licob_node_recovery import joined_outcomes
from .limore_corrected import OBSERVATIONS, registered, shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import centered_kernels, orthogonal_parts
from .structured_cv import paired_intervals
from .structured_kernel import drug_kernels, fit_cell_kernels
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES, adult_donor_outcomes
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics, compound_strata
from .yang_hcc_external import require_hash

REFS = {
    "LICOB": ("4cc7e3afa71fee91c0ca9a12ce06002d7a1016f9d4951b1833237830e3d7a2a1",
              {"interaction_transfer": "original_2d", "chemical_mean": "original_2d_mean",
               "neighbor_transfer": "original_neighbor"}),
    "LIMORE": ("fe7f9e656801d63fd1263ec3e94093cad86b049130aa038d4f2e8fdd059b02bb",
               {"corrected/structured": "original_2d", "corrected/drug_mean": "original_2d_mean",
                "corrected/drug_ridge": "original_ridge", "corrected/cellhit": "original_cellhit",
                "corrected/codeae_inductive": "original_codeae",
                "corrected/velodrome_inductive": "original_velodrome"}),
}
LABELS = {"LICOB": ASSET_HASHES["supplement"], "LIMORE": OBSERVATIONS}
PRIMARY, ANCHOR, MEAN, SHIFT, LOCAL, NULL = (
    "calibrated_2d_plus_local", "local_shrunk_mean", "local_mean", "local_calibrated_2d",
    "local_rna_only", "local_shuffled_response",
)
ARMS = (PRIMARY, ANCHOR, MEAN, SHIFT, LOCAL, NULL)
BUDGETS = ("8", "full")


def inputs(args):
    train, limore, rna = registered(args, "corrected")
    if args.mode == "LIMORE":
        query = limore
    else:
        if checked(args.cold, ("query", "rna")) != COLD:
            raise ValueError("Expected the registered 76-drug LICOB panel")
        roster = pd.read_parquet(args.cold / "query.parquet")
        query = roster[roster.adult_age_confirmed].copy()
        query["lineage_group"] = query.donor_id
        source = rna.loc[sorted(train.cell_id.unique())]
        held = pd.read_parquet(args.cold / "rna.parquet").loc[sorted(query.cell_id.unique())]
        genes = source.columns.intersection(held.columns).sort_values()
        rna = pd.concat([source[genes], held[genes]], verify_integrity=True)
    digest, names = REFS[args.mode]
    old, seal = verify_forecast(args.reference)
    if seal["sha256"] != digest:
        raise ValueError("Unexpected fixed source predictor")
    old = old[old.row_id.isin(query.row_id) & old.method.isin(names)].copy()
    if args.mode == "LICOB":
        old["lineage_group"] = old.cell_id.map(query.drop_duplicates("cell_id").set_index("cell_id").donor_id)
    old.method = old.method.map(names)
    shared_support(old, query, names.values())
    return train, query, rna, old


def partitions(query, mode):
    groups = sorted(query.lineage_group.unique(), key=lambda v: hash_text(f"491881:{mode}:{v}".encode()).hexdigest())
    for fold in range(3):
        test = groups[fold::3]
        train = [g for g in groups if g not in test]
        yield fold, train, test


def observations(query, path, mode):
    require_hash(path, LABELS[mode])
    base = query[KEY].assign(method="identity", predicted_value=0., track=mode, fold=0)
    if mode == "LICOB":
        raw = joined_outcomes(query, base, path)
    else:
        raw = join_observations(base, pd.read_parquet(path))
    return raw[[*KEY, "endpoint_value"]]


def prepare(args):
    _, query, _, _ = inputs(args)
    truth = observations(query, args.labels, args.mode)
    with new_output(args.out) as path:
        hashes, counts = {}, {}
        for fold, train, test in partitions(query, args.mode):
            frame = truth[truth.lineage_group.isin(train)].copy()
            if set(frame.lineage_group) != set(train) or set(frame.lineage_group) & set(test):
                raise ValueError("The fold contains observations outside the development donor block")
            artifact = path / f"fold-{fold}.parquet"
            frame.to_parquet(artifact, index=False)
            hashes[str(fold)] = sha256(artifact)
            counts[str(fold)] = {"training_donors": len(train), "held_donors": len(test), "training_rows": len(frame)}
        publish_json(path / "receipt.json", {"mode":args.mode, "source_reference":REFS[args.mode][0],
                                             "label_sha256":LABELS[args.mode], "files":hashes, "folds":counts})


def pair_parts(fit, basis, fitted_vectors, held_vectors):
    model_panel = fit[KEY].drop_duplicates("cell_id").set_index("cell_id").sort_index()
    donors = pd.Index(sorted(model_panel.lineage_group.unique()))
    weights = (model_panel.lineage_group.to_numpy()[None, :] == donors.to_numpy()[:, None]).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    agg = torch.as_tensor(weights, device=fitted_vectors[0].device)
    gram = agg @ kernel_between(basis, fitted_vectors, fitted_vectors) @ agg.T
    cross = kernel_between(basis, held_vectors, fitted_vectors) @ agg.T
    return donors, centered_kernels(gram, cross)


def local(fit, query, prior, rna, basis, chem):
    _validate(fit, query)
    train = sorted(fit.cell_id.unique())
    held = pd.Index(sorted(query.cell_id.unique()))
    donors, (cell, cross) = pair_parts(fit, basis, basis.vectors(rna.loc[train]), basis.vectors(rna.loc[held]))
    data = fit.merge(prior[[*KEY, "original_2d"]], on=KEY, validate="one_to_one")
    if len(data) != len(fit):
        raise ValueError("The archived forecast must cover every fit observation")
    drugs, kernel = chem
    pooled = data.groupby(["lineage_group", "compound_id"])[["endpoint_value", "original_2d"]].mean().unstack()
    real, base = (pooled[name].loc[donors, drugs].to_numpy() for name in ("endpoint_value", "original_2d"))
    if not np.isfinite(real).all() or not np.isfinite(base).all():
        raise ValueError("The donor-by-drug fit panel must be complete")
    raw_anchor = real.mean(axis=0)
    baseline_anchor = base.mean(axis=0)
    # Convert the source compound intercept into the target response scale.
    source_mean, _ = affine_map(base, real)
    anchor = (len(donors) * raw_anchor + 4 * source_mean.mean(axis=0)) / (len(donors) + 4)
    lower, upper = np.quantile(real, [.25, .75], axis=0)
    base_lower, base_upper = np.quantile(base, [.25, .75], axis=0)
    spread = upper - lower
    median_spread = float(np.median(spread))
    slope = float(np.clip(median_spread / max(np.median(base_upper - base_lower), 1e-8), .25, 4.))
    spread = np.maximum(spread, .25 * max(median_spread, 1e-6))
    errors = (real - raw_anchor) - slope * (base - baseline_anchor)
    # Keep the two-way interaction as the personalized learning target.
    residual = orthogonal_parts(errors / spread, np.ones_like(errors))[2]
    specific = orthogonal_parts((real - raw_anchor) / spread, np.ones_like(real))[2]
    rng = np.random.default_rng(43386)
    order = rng.permutation(len(donors))
    shuffled = np.empty_like(order)
    shuffled[order] = np.roll(order, 1)
    declared = kernel["structured"]
    changes = {}
    for name, target in ((PRIMARY, residual), (LOCAL, specific), (NULL, residual[shuffled])):
        adjusted, _ = corrections(cell, cross, declared, target)
        changes[name] = adjusted * spread
    index = held.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    source = query.merge(
        prior[[*KEY, "original_2d"]], on=KEY, how="left", validate="one_to_one",
    ).original_2d.to_numpy()
    if not np.isfinite(source).all():
        raise ValueError("The archived forecast must cover every held observation")
    m = anchor[index[1]]
    delta = slope*(source-baseline_anchor[index[1]])
    values = {MEAN:raw_anchor[index[1]],ANCHOR:m,SHIFT:m+delta,LOCAL:m+changes[LOCAL][index],
              PRIMARY:m+delta+.5*changes[PRIMARY][index],NULL:m+delta+.5*changes[NULL][index]}
    return pd.concat([query[KEY].assign(method=name,predicted_value=p) for name,p in values.items()],
                     ignore_index=True), {"train_donors":len(donors), "slope":slope,
                                          "personalization_rms":float(np.sqrt(np.mean(np.square(.5*changes[PRIMARY][index]))))}


def predict(args):
    train, roster, rna, previous = inputs(args)
    proof_path = args.dev / "receipt.json"
    proof = json.loads(proof_path.read_text())
    if (proof["mode"], proof["source_reference"], proof["label_sha256"]) != (
        args.mode, REFS[args.mode][0], LABELS[args.mode]
    ):
        raise ValueError("Expected the fixed donor-fold development labels")
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, _ = load_auxiliary(args.auxiliary)
    device = device_cuda()
    basis, _ = fit_cell_kernels(rna.loc[sorted(train.cell_id.unique())], edges, auxiliary, device=device)
    chem = drug_kernels(roster, targets, device=device)
    prior = previous[previous.method.eq("original_2d")].rename(columns={"predicted_value":"original_2d"})
    tables, audit = [], {}
    for fold, donors, test in partitions(roster,args.mode):
        if fold != args.fold:
            continue
        path = args.dev / f"fold-{fold}.parquet"
        require_hash(path,proof["files"][str(fold)])
        current = pd.read_parquet(path)
        expected = roster[roster.lineage_group.isin(donors)]
        if not current[KEY].set_index("row_id").sort_index().equals(expected[KEY].set_index("row_id").sort_index()):
            raise ValueError("Only the declared fit donors may be loaded by a forecast process")
        query = roster[roster.lineage_group.isin(test)]
        for budget in BUDGETS:
            sample = donors if budget == "full" else donors[:8]
            fold_fit = current[current.lineage_group.isin(sample)]
            part, audit[budget] = local(fold_fit,query[KEY],prior,rna,basis,chem)
            joined = pd.concat([part,previous[previous.row_id.isin(query.row_id)]],ignore_index=True)
            shared_support(joined,query,(*ARMS,*REFS[args.mode][1].values()))
            tables.append(joined.assign(track=args.mode+"/"+budget,fold=fold))
    freeze_table(pd.concat(tables,ignore_index=True),args.out,provenance={
        "mode":args.mode,"fold":args.fold,"source_reference":REFS[args.mode][0],"dev_sha256":sha256(proof_path),
        "endpoint":LABELS[args.mode],"fit":audit,
    })


def assessment(rows, roster, method, comparators):
    comparisons = [(method,name) for name in comparators]
    metrics, drug, _ = axis_metrics(rows,comparisons)
    ranking = {(p["baseline"],p["metric"]):p["ci95"][0] for p in metrics["paired_contrasts"]}
    bounds = {name:paired_intervals(rows,method,name,repeats=1000) for name in comparators}
    gates = {b:all(bounds[b][k][0]>0 for k in ("spearman","pearson"))
             and all(ranking[b,k]>-.005 for k in ("spearman","ndcg_at_5","top5_overlap")) for b in comparators}
    strata = {}
    for name, parents in compound_strata(roster).items():
        sub = drug[drug.compound_id.isin(parents)]
        if not sub.empty:
            values = sub.groupby("method")[["spearman","pearson"]].mean()
            strata[name] = {"drugs":sub.compound_id.nunique(),
                            "scores":values.astype(object).where(values.notna(),None).to_dict("index")}
    return {"metrics":metrics,"intervals":bounds,"gates":gates,"strata":strata}


def score(args):
    _, roster, _, _ = inputs(args)
    verified, hashes = [], {}
    dev_hash = sha256(args.dev / "receipt.json")
    for fold, _, test in partitions(roster, args.mode):
        part, seal = verify_forecast(args.predictions / f"fold-{fold}.parquet")
        p = seal["provenance"]
        if (p["fold"], p["mode"], p["source_reference"], p["endpoint"], p["dev_sha256"]) != (
            fold, args.mode, REFS[args.mode][0], LABELS[args.mode], dev_hash
        ):
            raise ValueError("Unexpected donor-fold forecast")
        subset = roster[roster.lineage_group.isin(test)]
        for budget in BUDGETS:
            block = part[part.track.eq(f"{args.mode}/{budget}")]
            shared_support(block, subset, (*ARMS, *REFS[args.mode][1].values()))
            verified.append((block, subset))
        hashes[str(fold)] = seal["sha256"]
    truth = observations(roster, args.labels, args.mode)
    pieces = []
    for block, subset in verified:
        joined = join_observations(block, truth)
        if args.mode == "LICOB":
            patients = subset[["row_id", "donor_id", "adult_age_confirmed"]]
            joined = adult_donor_outcomes(
                joined.merge(patients, on="row_id", validate="many_to_one"),
                constant_methods=(MEAN, ANCHOR, "original_2d_mean"),
            )
        pieces.append(joined)
    scored = pd.concat(pieces, ignore_index=True)
    decisions = {}
    comparisons = (MEAN, ANCHOR, SHIFT, LOCAL, NULL, *REFS[args.mode][1].values())
    for budget in BUDGETS:
        joined = scored[scored.track.eq(f"{args.mode}/{budget}")]
        decisions[budget] = assessment(joined, roster, PRIMARY, comparisons)
    shift = scored[scored.method.eq(PRIMARY)].assign(
        method=lambda x: x.track.str.split("/").str[1], track=args.mode,
    )
    decisions["gain"] = assessment(shift, roster, "full", ("8",))
    publish_json(args.out,{"mode":args.mode,"labels":LABELS[args.mode],"hashes":hashes,
                           "donors":roster.lineage_group.nunique(),"drugs":roster.compound_id.nunique(),"results":decisions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage",required=True)
    common = ("core","original","cold","reference")
    for stage, fields in {
        "prepare":(*common,"labels"),
        "predict":(*common,"dev","priors","hallmark","targets","auxiliary"),
        "score":(*common,"dev","predictions","labels"),
    }.items():
        job = sub.add_parser(stage)
        for field in ("out",*fields):
            job.add_argument("--"+field,type=Path,required=True)
        job.add_argument("--mode",choices=REFS,required=True)
        if stage=="predict":
            job.add_argument("--fold",type=int,choices=range(3),required=True)
    args = parser.parse_args()
    {"prepare":prepare,"predict":predict,"score":score}[args.stage](args)


if __name__=="__main__":
    main()
