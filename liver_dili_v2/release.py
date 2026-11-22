"""Seal all DILI projections before a single paired FDA score job."""

import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from liver_dili.splits import digest

from .calibration import calibrate, diagnostics, risk_coverage
from .data import KINDS, RegisteredData, check_disjoint, inner_groups
from .features import Features, subset
from .rankers import train
from .release_types import ReleaseFit
from .round1 import ranking
from .round2 import CONFIGS, safe_auxiliary, workflow
from .round3 import REWEIGHT, add_mechanisms, error_features, hard_weights, nested_error_weights
from .stack import GRID, construct, tune


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def feature_loader(args):
    return Features(args.fm, args.chemberta)


def mechanism_model(oof, model):
    if "signed_mechanism" not in model.transform.features:
        return None, []
    columns = [c for c in oof if "__" in c]
    predictor = make_pipeline(StandardScaler(), LogisticRegression(C=.018, max_iter=500))
    return predictor.fit(oof[columns], oof.label), columns


def fold_fit(args, data, selected):
    scope = f"{args.kind}-{args.fold}"
    paths = {
        "r1_cpu": args.runs/"round1-03-cpu"/scope, "r1_cat": args.runs/"round1-03-cat"/scope,
        "r2": args.runs/"round2-01"/scope, "r3": args.runs/"round3-02"/scope, "r4": args.runs/"round4-01"/scope,
    }
    if digest(paths["r4"]/"receipt.json") != selected["fold_sha256"][scope]:
        raise ValueError("A source ensemble changed after policy lock")
    model, calibrated = joblib.load(paths["r4"]/"meta.joblib")[selected["selected"]]
    fits = joblib.load(paths["r1_cpu"]/"models.joblib") | joblib.load(paths["r1_cat"]/"models.joblib")
    fits |= joblib.load(args.runs/"round1-04-kernels"/scope/"models.joblib")
    fits |= joblib.load(paths["r3"]/"reweighted.joblib")
    graph = joblib.load(paths["r2"]/"models.joblib")
    raw = pd.read_parquet(paths["r3"]/"inner.parquet")
    projection, columns = mechanism_model(raw, model)
    train_table = data.fit_table(args.kind, args.fold)
    ref = feature_loader(args).arrays(train_table)["base"]
    result = ReleaseFit(fits, graph, model, calibrated, ref, projection, columns)
    args.out.mkdir(exist_ok=False, parents=True)
    path = args.out/"checkpoint.joblib"
    joblib.dump(result, path, compress=2)
    write(args.out/"release.json", {"checkpoint_sha256": digest(path), "policy_sha256": digest(args.policy),
                                   "increment_labels_seen": False, "fit_rows": len(train_table), "scope": scope,
                                   "embedding_sha256": feature_loader(args).hashes})


def majority_setting(args):
    population = {name: Counter() for name in ("v1_lgbm", "lgbm", "catboost", "xgboost", "extra",
                                               "linear", "tsvm", "countsvm", "tlogit", "minmaxsvm", "rbfsvm")}
    configuration = {name: Counter() for name in CONFIGS}
    for kind in KINDS:
        for i in range(5):
            scope = f"{kind}-{i}"
            for snapshot in ("round1-03-cpu", "round1-03-cat", "round1-04-kernels"):
                collected = read(args.runs/snapshot/scope/"receipt.json")["selected"]
                for arm, settings in collected.items():
                    population[arm].update([(settings["representation"], json.dumps(settings["config"]))])
            graph = read(args.runs/"round2-01"/scope/"receipt.json")["inner_results"]
            for arm in CONFIGS:
                winner = max((c for c in graph if c["method"]==arm), key=lambda c: c["utility"])
                configuration[arm].update([tuple(winner["setting"])])
    choices = {
        arm: [(r, json.loads(s)) for (r,s),_ in configurations.most_common(2)]
        for arm, configurations in population.items()
    }
    configs = {arm: tuple(k for k,_ in options.most_common(2)) for arm,options in configuration.items()}
    return choices, configs


def history(args, data, selected):
    rows = data.fit_table("release", 0)
    bank = feature_loader(args)
    x = bank.arrays(rows)
    y = rows.label.to_numpy()
    aux = safe_auxiliary(args.aux, args.fda_structures)
    tree_settings, graph_settings = majority_setting(args)
    graph_x, _, graph_models, graph_guidance = workflow(rows, rows.iloc[:1], aux, device=args.device,
                                                       configurations=graph_settings)
    oof = pd.DataFrame(graph_x, index=rows.index).assign(label=y, group=rows.group)
    models, chosen = {}, {}
    splits = list(inner_groups(rows))
    for name, configs in tree_settings.items():
        candidates = []
        for representation, cfg in configs:
            raw = np.empty(len(y))
            utility = []
            for t, v in splits:
                local = train(name, representation, cfg, subset(x,t), y[t], args.device=="cuda")
                raw[v] = local.raw(subset(x,v))
                metrics = ranking(y[v], raw[v])
                utility.append(.6*metrics["auroc"]+.4*metrics["auprc"])
            candidates.append((float(np.mean(utility)), representation, cfg, raw))
        _, representation, cfg, raw = max(candidates, key=lambda r:r[0])
        oof[name] = raw
        models[name] = train(name, representation, cfg, x, y, args.device=="cuda")
        chosen[name] = {"representation": representation, "config": cfg}
    masks = error_features(x)
    weight = hard_weights(masks,y,oof.v1_lgbm.to_numpy())
    local_weights = nested_error_weights(rows, x, y, masks, chosen["v1_lgbm"])
    for name in REWEIGHT:
        representation, cfg = (chosen[name][key] for key in ("representation","config"))
        raw = np.empty(len(y))
        for tr, va, local in local_weights:
            fit = train(name, representation, cfg, subset(x,tr), y[tr], sample_weight=local)
            raw[va] = fit.raw(subset(x,va))
        oof[name+"_reweighted"] = raw
        models[name+"_reweighted"] = train(name, representation, cfg, x, y, sample_weight=weight)
    add_mechanisms(oof, oof.copy())
    _, _, _, search = tune(oof, y, rows.group)
    recipe = selected["selected"]
    penalty = next(i["penalty"] for i in search if i["recipe"]==recipe)
    columns = list(GRID.get(recipe, GRID["heterogeneous"] if recipe=="weighted_heterogeneous" else GRID["kernel_tree"]))
    if recipe == "weighted_heterogeneous":
        columns += [c for c in oof if c.endswith("_reweighted")]
    if recipe == "mechanism_oof":
        columns += [c for c in oof if "__" in c or c=="signed_mechanism"]
    columns = [c for c in columns if c in oof]
    raw = np.empty(len(oof))
    for t,v in splits:
        learner = construct(oof.iloc[t], columns, y[t], recipe, penalty)
        raw[v] = learner.raw(oof.iloc[v])
    meta = construct(oof, columns, y, recipe, penalty)
    cal, calibration, _ = calibrate(raw, y, rows.group.to_numpy())
    signed, column_names = mechanism_model(oof, meta)
    package = ReleaseFit(models, graph_models, meta, cal, x["base"], signed, column_names)
    args.out.mkdir(exist_ok=False, parents=True)
    file = args.out/"checkpoint.joblib"
    joblib.dump(package, file, compress=2)
    write(args.out/"release.json", {
        "scope": "release-0", "fit_rows": len(rows), "parent_count": int(rows.connectivity.nunique()),
        "trained_on_prior_release": True, "increment_labels_seen": False, "policy_sha256": digest(args.policy),
        "checkpoint_sha256": digest(file), "embedding_sha256": bank.hashes,
        "recipe": recipe, "configurations": chosen, "graph_tuning": graph_guidance, "selected_calibration": cal.method,
        "calibration_grid": calibration, "ood_cutoff": .195,
    })


def release_model(root):
    card = read(root/"release.json")
    checkpoint = root/"checkpoint.joblib"
    if digest(checkpoint) != card["checkpoint_sha256"] or card["increment_labels_seen"]:
        raise ValueError("The DILI release did not match its freeze")
    return joblib.load(checkpoint), card


def forecast(args, dataset):
    bundle, card = release_model(args.checkpoint)
    bank = feature_loader(args)
    if card["embedding_sha256"] != bank.hashes or card["policy_sha256"] != digest(args.policy):
        raise ValueError("Forecast representation or final policy changed")
    query = dataset.outer_query(args.kind, args.fold, development=False)
    train = dataset.fit_table(args.kind, args.fold)
    check_disjoint(train, query)
    output = bundle.forecast(query, bank, device=args.device, cohort=True)
    args.out.mkdir(exist_ok=False, parents=True)
    path = args.out/"forecast.parquet"
    output.to_parquet(path)
    write(args.out/"receipt.json", {"predictions_sha256": digest(path), "scope": card["scope"],
                                   "checkpoint_sha256": card["checkpoint_sha256"], "policy_sha256": digest(args.policy),
                                   "rows": len(output), "outcome_columns": []})


def tickets():
    return [(kind, fold) for kind in KINDS for fold in range(5)] + [("release",0)]


def lock(args):
    manifest = {}
    for kind, fold in tickets():
        location = args.runs / f"{kind}-{fold}"
        stamp = read(location/"receipt.json")
        if (stamp["policy_sha256"] != digest(args.policy) or stamp["scope"] != f"{kind}-{fold}"
                or stamp["outcome_columns"] or stamp["predictions_sha256"] != digest(location/"forecast.parquet")):
            raise ValueError("Label-free DILI forecast differed from the preregistered prediction")
        manifest[stamp["scope"]] = digest(location/"receipt.json")
    write(args.out, {"predictions": manifest, "selected": read(args.policy)["selected"],
                     "policy_sha256": digest(args.policy), "code_sha256": digest(Path(__file__))})


def group_lookup(dataset, kind):
    groups = {}
    for i in range(5):
        fit = dataset.read(f"{kind}-{i}/train.parquet", columns=["group"])
        for row, value in fit.group.items():
            if row in groups and groups[row] != value:
                raise ValueError("Historical scaffold group is inconsistent")
            groups[row] = value
    return groups


def bootstrap(frame, *, seed, n=10000):
    groups = [g for _,g in frame.groupby("cluster", sort=True).indices.items()]
    data = frame[["label", "v2", "v1", "v2_decision", "v1_decision"]].to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    result = []
    for _ in range(n):
        ids = np.concatenate([groups[ix] for ix in generator.integers(0, len(groups), len(groups))])
        y,a,b,da,db = data[ids].T
        if len(np.unique(y)) < 2:
            continue
        first, second = diagnostics(y,a,da), diagnostics(y,b,db)
        result.append([first[key]-second[key] for key in first])
    metrics = list(diagnostics(data[:,0], data[:,1], data[:,3]))
    draws = np.asarray(result)
    return {name: np.quantile(draws[:,i],[.005,.995]).tolist() for i,name in enumerate(metrics)}, len(draws)


def score(args, data):
    if args.out.exists() or digest(args.policy) != read(args.freeze)["policy_sha256"]:
        raise ValueError("DILI score requires the final freeze and a fresh score destination")
    sealed = read(args.freeze)
    if sealed["code_sha256"] != digest(Path(__file__)):
        raise ValueError("DILI scoring implementation changed after freeze")
    results = {}
    cluster = {kind: group_lookup(data,kind) for kind in KINDS}
    for kind, fold in tickets():
        slot = f"{kind}-{fold}"
        path = args.runs/slot
        stamp = read(path/"receipt.json")
        if digest(path/"receipt.json") != sealed["predictions"][slot] or digest(path/"forecast.parquet") != stamp["predictions_sha256"]:
            raise ValueError("A DILI forecast changed after freeze")
        new = pd.read_parquet(path/"forecast.parquet")
        archive = args.baselines/slot
        old_path = archive/"forecast.parquet"
        original = read(archive/"freeze.json")
        if digest(old_path) != original["forecast_sha256"]:
            raise ValueError("DILI v1 paired forecast provenance changed")
        predictions = pd.read_parquet(old_path)
        prev = predictions.loc[predictions["mode"].eq("structure") & predictions.method.eq("lightgbm")]
        targets = data.outcomes(kind,fold,development=False)
        if not (set(new.index) == set(prev.index) == set(targets.index)):
            raise ValueError("DILI v1 and v2 refer to different FDA rows")
        frame = new.rename(columns={"probability":"v2","decision":"v2_decision"}).join(
            prev[["probability","decision"]].rename(columns={"probability":"v1","decision":"v1_decision"})).join(targets[["label"]])
        frame["cluster"] = (frame.index.map(cluster[kind]) if kind in cluster else
                            data.outer_query(kind,fold,development=False).loc[frame.index,"connectivity"])
        if frame.cluster.isna().any():
            raise ValueError("DILI bootstrap cluster index incomplete")
        results.setdefault(kind,[]).append(frame)
    report = {}
    for kind, parts in results.items():
        a = pd.concat(parts)
        expected = 155 if kind=="release" else 887
        if len(a)!=expected or not a.index.is_unique:
            raise ValueError("DILI final evaluation partition size changed")
        y = a.label.to_numpy()
        ci, draws = bootstrap(a, seed=20261122)
        report[kind] = {
            "rows": len(a), "positives": int(y.sum()), "v1": diagnostics(y,a.v1,a.v1_decision),
            "v2": diagnostics(y,a.v2,a.v2_decision), "paired_99_ci_v2_minus_v1": ci, "valid_draws": draws,
            "risk_coverage_v2": risk_coverage(y,a.v2.to_numpy(),a.v2_decision.to_numpy(),a.nearest_tanimoto.to_numpy()),
        }
    write(args.out, {"freeze_sha256": digest(args.freeze), "selected_recipe": sealed["selected"],
                     "external_score_passes": 1, "splits": report})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit_fold", "fit_history", "predict", "freeze", "score"))
    for key in ("data","policy","runs","out","fm","chemberta","baselines","aux","fda-structures","checkpoint","freeze"):
        parser.add_argument("--"+key, type=Path)
    parser.add_argument("--kind", choices=(*KINDS,"release"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    args = parser.parse_args()
    if args.action == "freeze":
        lock(args)
    else:
        dataset = RegisteredData(args.data)
        if args.action == "score":
            score(args,dataset)
        elif args.action == "predict":
            forecast(args,dataset)
        else:
            selected = read(args.policy)
            if selected["selection_uses_increment"]:
                raise ValueError("Invalid DILI final choice provenance")
            (history if args.action=="fit_history" else fold_fit)(args,dataset,selected)
