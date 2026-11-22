"""Nested, group-held FDA forecasting with exact assay joins and delayed scoring."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .chemistry import features, identity
from .splits import digest, receipt


GRIDS = {"logistic": (.02, .2, 2.), "lightgbm": ((7, 80), (15, 120), (31, 160)),
         "multitask": ((64, .15), (128, .25))}
MODES = ("ecfp_properties", "structure", "predicted_pathways", "pathway_shuffle",
         "cell_painting", "painting_mask", "painting_shuffle", "measured_rna")
PATH_PREFIX = ("wikipathways__", "hallmark__", "progeny__")
SEED = 20261122


@lru_cache(maxsize=10000)
def chemistry(smiles):
    return features(smiles)


def matrix(rows, embedding=None):
    x = np.stack([chemistry(s)[1] for s in rows.smiles])
    if embedding is not None:
        z = embedding.reindex(rows.inchikey).to_numpy(dtype=np.float32)
        if not np.isfinite(z).all():
            raise ValueError("Missing frozen embedding for evaluated structure")
        x = np.c_[x, z]
    return x


def panel(path):
    frames = []
    for name in ("GSE284321", "GSE296254", "TGGATEs_PHH"):
        assay = pd.read_parquet(path / f"{name}.parquet")
        names = sorted(col for col in assay if col.startswith(PATH_PREFIX))
        rows = assay.loc[assay.dose_uM.between(1, 100)].copy()
        rows = rows.groupby(["inchikey", "smiles", "connectivity", "dose_uM"])[names].median().reset_index()
        maximum = rows.groupby("inchikey").dose_uM.transform("max")
        measured = rows.loc[rows.dose_uM.eq(maximum)].groupby(
            ["inchikey", "smiles", "connectivity"])[names].median().reset_index()
        measured["scaffold"] = measured.smiles.map(lambda x: identity(x)[3])
        measured["platform"] = "PHH" if name == "TGGATEs_PHH" else "HepaRG"
        frames.append(measured)
    # Both HepaRG studies count once per exact structure in the pathway-regression target.
    return pd.concat(frames).groupby(
        ["inchikey", "smiles", "connectivity", "scaffold", "platform"], as_index=False).median(numeric_only=True)


def permitted(assays, excluded):
    legal = (~assays.connectivity.isin(excluded.connectivity) & ~assays.scaffold.isin(excluded.scaffold))
    return assays.loc[legal]


class PathHead:
    """One train-only structure-to-pathway regressor per independent assay platform."""

    def __init__(self, assays, embedding):
        self.models = {}
        for platform, rows in assays.groupby("platform", sort=True):
            target = rows.filter(regex="^(wikipathways|hallmark|progeny)__")
            target = target.loc[:, target.notna().all()]
            if len(rows) < 30 or not len(target.columns):
                raise ValueError(f"Insufficient registered {platform} pathway training support")
            x = matrix(rows, embedding)
            scale = StandardScaler().fit(x)
            weights = Ridge(alpha=100., solver="lsqr").fit(scale.transform(x), target)
            self.models[platform] = (list(target.columns), scale, weights)

    def predict(self, x, index):
        arrays = []
        for platform, (names, scale, estimator) in self.models.items():
            result = estimator.predict(scale.transform(x))
            arrays.append(pd.DataFrame(result, index=index, columns=[f"{platform}:{col}" for col in names]))
        return pd.concat(arrays, axis=1)


def shuffled(values, seed):
    order = np.random.default_rng(seed).permutation(len(values))
    return np.asarray(values)[order]


class Dataset:
    def __init__(self, embeddings, assays):
        self.embedding = pd.read_parquet(embeddings).drop(columns="chemberta_truncated") if embeddings else None
        self.assays = panel(assays) if assays else None
        self.measured = self.assays
        self.paint = pd.read_parquet(assays / "painting.parquet") if assays else None
        self.path_cache = {}

    def paths(self, train, query, x_train, x_query):
        key = (tuple(train.index), tuple(query.index))
        if key in self.path_cache:
            return self.path_cache[key]
        available = permitted(self.assays, query)
        result = []
        for _, held in GroupKFold(4).split(train, groups=train.group):
            path = PathHead(permitted(available, train.iloc[held]), self.embedding)
            result.append(path.predict(x_train[held], train.index[held]))
        fitted = PathHead(available, self.embedding)
        target = fitted.predict(x_query, query.index)
        values = (pd.concat(result).reindex(index=train.index, columns=target.columns), target, fitted)
        self.path_cache[key] = values
        return values

    def representation(self, train, query, mode):
        embedding = None if mode == "ecfp_properties" else self.embedding
        xf, xq = matrix(train, embedding), matrix(query, embedding)
        fit_mask, query_mask = np.ones(len(train), bool), np.ones(len(query), bool)
        context = {}
        if mode in ("predicted_pathways", "pathway_shuffle"):
            pf, pq, head = self.paths(train, query, xf, xq)
            if mode == "pathway_shuffle":
                pf, pq = shuffled(pf, SEED), shuffled(pq, SEED + 1)
            xf, xq = np.c_[xf, pf], np.c_[xq, pq]
            context["path_head"] = head
        elif mode.startswith("painting") or mode == "cell_painting":
            tf, tq = [self.paint.reindex(rows.inchikey) for rows in (train, query)]
            fit_mask, query_mask = tf.notna().any(axis=1).to_numpy(), tq.notna().any(axis=1).to_numpy()
            if fit_mask.sum() < 12:
                return xf, xq, fit_mask & False, query_mask & False, context
            if mode == "painting_mask":
                xf, xq = np.c_[xf, fit_mask], np.c_[xq, query_mask]
                return xf, xq, np.ones(len(train), bool), query_mask, context
            keep = tf.iloc[fit_mask].notna().mean().ge(.8)
            center = tf.loc[:, keep].median()
            tr, te = (f.loc[:, keep].fillna(center).to_numpy() for f in (tf, tq))
            scaler = StandardScaler().fit(tr[fit_mask])
            tr, te = scaler.transform(tr), scaler.transform(te)
            transform = PCA(n_components=min(32, fit_mask.sum() - 1, tr.shape[1]),
                            svd_solver="randomized", random_state=SEED).fit(tr[fit_mask])
            tr, te = transform.transform(tr), transform.transform(te)
            if mode == "painting_shuffle":
                tr[fit_mask], te[query_mask] = shuffled(tr[fit_mask], SEED), shuffled(te[query_mask], SEED + 1)
            tr[~fit_mask], te[~query_mask] = 0, 0
            xf, xq = np.c_[xf, tr, fit_mask], np.c_[xq, te, query_mask]
            fit_mask = np.ones(len(train), bool)
        elif mode == "measured_rna":
            measured = self.measured.drop(columns=["smiles", "connectivity", "scaffold"]).pivot(
                index="inchikey", columns="platform")
            measured.columns = [f"{platform}:{path}" for path, platform in measured.columns]
            tf, tq = [measured.reindex(rows.inchikey) for rows in (train, query)]
            fit_mask, query_mask = tf.notna().any(axis=1).to_numpy(), tq.notna().any(axis=1).to_numpy()
            cols = tf.notna().sum().ge(10)
            fit_median = tf.loc[:, cols].median()
            xf = np.c_[xf, tf.loc[:, cols].fillna(fit_median), tf.loc[:, cols].notna()]
            xq = np.c_[xq, tq.loc[:, cols].fillna(fit_median), tq.loc[:, cols].notna()]
        return xf, xq, fit_mask, query_mask, context


class MultiTask:
    """An embedding-based MLP with shared DILI and masked pathway supervision."""

    def fit(self, x, y, aux_x, aux_y, setting):
        import torch
        from torch import nn

        torch.manual_seed(SEED)
        width, dropout = setting
        self.scaler = StandardScaler().fit(np.vstack([x, aux_x]))
        tx = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32)
        ty = torch.as_tensor(y, dtype=torch.float32)
        ax = torch.as_tensor(self.scaler.transform(aux_x), dtype=torch.float32)
        aux_values = aux_y.to_numpy()
        aux_mask = np.isfinite(aux_values)
        aux_mean = np.nanmean(aux_values, axis=0)
        aux_scale = np.maximum(np.nanstd(aux_values, axis=0), 1e-5)
        ay = torch.as_tensor(np.nan_to_num((aux_values - aux_mean) / aux_scale), dtype=torch.float32)
        mask = torch.as_tensor(aux_mask)
        shared = nn.Sequential(nn.Linear(x.shape[1], width), nn.LayerNorm(width), nn.GELU(),
                               nn.Dropout(dropout), nn.Linear(width, width // 2), nn.GELU())
        classifier = nn.Linear(width // 2, 1)
        pathway = nn.Linear(width // 2, ay.shape[1])
        optimizer = torch.optim.AdamW(
            [*shared.parameters(), *classifier.parameters(), *pathway.parameters()], lr=.0015, weight_decay=.035)
        for _ in range(28):
            batch = torch.randperm(len(tx)).split(96)
            aux_order = torch.randperm(len(ax))
            for step, train_ids in enumerate(batch):
                aux_ids = aux_order[(torch.arange(96) + 96 * step) % len(aux_order)]
                classification = nn.functional.binary_cross_entropy_with_logits(
                    classifier(shared(tx[train_ids])).flatten(), ty[train_ids])
                error = pathway(shared(ax[aux_ids])) - ay[aux_ids]
                active = mask[aux_ids]
                loss = classification + .25 * error[active].square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.encoder = nn.Sequential(shared, classifier).eval()
        return self

    def raw(self, x):
        import torch

        with torch.inference_mode():
            return self.encoder(torch.as_tensor(self.scaler.transform(x), dtype=torch.float32)).flatten().numpy()


def trained_score(method, setting, xf, yf, xq, aux=None):
    if method == "multitask":
        aux_x, aux_y = aux
        fit = MultiTask().fit(xf, yf, aux_x, aux_y, setting)
        return fit.raw(xq), fit
    if method == "logistic":
        scale = StandardScaler().fit(xf)
        fit = LogisticRegression(C=setting, solver="liblinear", intercept_scaling=10.,
                                 max_iter=1000, random_state=SEED).fit(scale.transform(xf), yf)
        return fit.decision_function(scale.transform(xq)), (scale, fit)
    leaves, iterations = setting
    fit = LGBMClassifier(num_leaves=leaves, n_estimators=iterations, learning_rate=.04,
                         min_child_samples=20, reg_lambda=2., colsample_bytree=.7,
                         verbose=-1, n_jobs=2, random_state=SEED).fit(xf, yf)
    return np.asarray(fit.booster_.predict(xq, raw_score=True)), fit


def calibrate(oof, labels, groups):
    def platt():
        return LogisticRegression(C=10., solver="liblinear", intercept_scaling=10., max_iter=500, random_state=SEED)

    fit = platt().fit(oof[:, None], labels)
    probabilities = fit.predict_proba(oof[:, None])[:, 1]
    thresholds = np.unique(np.r_[.5, np.quantile(probabilities, np.linspace(.04, .96, 65))])
    threshold = max(thresholds, key=lambda t: (balanced_accuracy_score(labels, probabilities >= t), -abs(t-.5)))
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(SEED)
    parameters = []
    for draw in rng.integers(len(unique), size=(200, len(unique))):
        mass = np.bincount(draw, minlength=len(unique))[inverse]
        if len(np.unique(labels[mass > 0])) < 2:
            continue
        boot = platt().fit(oof[:, None], labels, sample_weight=mass)
        parameters.append([float(boot.coef_[0, 0]), float(boot.intercept_[0])])
    return fit, float(threshold), np.asarray(parameters)


def aux_arrays(dataset, held):
    aux = permitted(dataset.assays, held)
    x = matrix(aux, dataset.embedding)
    paths = aux.drop(columns=["smiles", "connectivity", "scaffold"]).pivot(index="inchikey", columns="platform")
    paths = paths.reindex(aux.inchikey).dropna(axis=1, how="all")
    return x, paths


def inner_views(dataset, train, mode, method, folds):
    parts = []
    for fit, held in folds:
        tr, va = train.iloc[fit], train.iloc[held]
        xf, xq, ft, qt, _ = dataset.representation(tr, va, mode)
        aux = aux_arrays(dataset, va) if method == "multitask" else None
        parts.append((fit, held, xf, xq, ft, qt, aux))
    return parts


def choose(dataset, train, query, mode, method, folds, views=None):
    parts = views if views is not None else inner_views(dataset, train, mode, method, folds)
    y = train.label.to_numpy()
    scores = []
    for setting in GRIDS[method]:
        oof = np.full(len(train), np.nan)
        for fit, held, xf, xq, ft, qt, aux in parts:
            yf = y[fit][ft]
            if len(np.unique(yf)) < 2 or not qt.any():
                continue
            values, _ = trained_score(method, setting, xf[ft], yf, xq[qt], aux)
            oof[held[qt]] = values
        good = np.isfinite(oof)
        if good.sum() >= 20 and len(np.unique(y[good])) == 2:
            scores.append((average_precision_score(y[good], oof[good]), setting, oof, good))
    if not scores:
        return None
    pr, setting, oof, good = max(scores, key=lambda entry: entry[0])
    calibration, threshold, coefficients = calibrate(oof[good], y[good], train.group.to_numpy()[good])
    xf, xq, fit_mask, query_mask, context = dataset.representation(train, query, mode)
    aux = aux_arrays(dataset, query) if method == "multitask" else None
    if not query_mask.any() or len(np.unique(y[fit_mask])) < 2:
        return None
    raw, model = trained_score(method, setting, xf[fit_mask], y[fit_mask], xq[query_mask], aux)
    prob = calibration.predict_proba(raw[:, None])[:, 1]
    lower, upper = np.quantile(expit(raw[:, None] * coefficients[:, 0] + coefficients[:, 1]), [.05, .95], axis=1)
    q = pd.DataFrame(dict(raw=raw, probability=prob, decision=prob >= threshold,
                          probability_lo90=lower, probability_hi90=upper), index=query.index[query_mask])
    return q, dict(method=method, setting=setting, calibration=calibration, threshold=threshold,
                   coefficients=coefficients, model=model, inner_ap=pr, fit_rows=int(fit_mask.sum()), **context)


def predict(args):
    plan = args.splits / args.fold
    files = json.loads((args.splits / "receipt.json").read_text())["files"]
    for name in ("train.parquet", "query.parquet", "nearest.tsv"):
        if digest(plan / name) != files[f"{args.fold}/{name}"]:
            raise ValueError(f"Split input changed: {name}")
    train, query = (pd.read_parquet(plan / f"{name}.parquet") for name in ("train", "query"))
    if set(train.connectivity) & set(query.connectivity):
        raise ValueError("Outer split connectivity overlap")
    ds = Dataset(args.embeddings, args.assays)
    if ds.assays is not None:
        ds.assays = permitted(ds.assays, query)
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(train, train.label, train.group))
    names = ["ecfp_properties"]
    if ds.embedding is not None:
        names.append("structure")
    if ds.assays is not None:
        names.extend(MODES[2:])
    frames, fit_metadata = {}, []
    for mode in names:
        standard = inner_views(ds, train, mode, "logistic", folds)
        methods = ["logistic", "lightgbm"]
        if mode == "structure" and ds.assays is not None:
            methods.append("multitask")
        for method in methods:
            chosen = choose(ds, train, query, mode, method, folds, standard if method != "multitask" else None)
            if chosen is None:
                continue
            table, fit = chosen
            table["measured_available"] = table.index.isin(query.index)
            if mode in ("cell_painting", "painting_mask", "painting_shuffle"):
                anchor = frames[("structure" if ds.embedding is not None else "ecfp_properties", method)]
                fallback = anchor.loc[~anchor.index.isin(table.index)].copy()
                fallback["measured_available"] = False
                table = pd.concat([table, fallback])
            frames[mode, method] = table
            fit_metadata.append(dict(mode=mode, method=method, setting=fit["setting"],
                                     fit_rows=fit["fit_rows"], threshold=fit["threshold"], inner_ap=fit["inner_ap"]))
    args.out.mkdir(parents=True, exist_ok=False)
    output = pd.concat([frame.assign(mode=mode, method=method, fold=args.fold)
                        for (mode, method), frame in frames.items()])
    output.to_parquet(args.out / "forecast.parquet")
    receipt(args.out / "freeze.json", dict(split=args.fold, split_receipt=digest(args.splits / "receipt.json"),
        embedding_sha256=digest(args.embeddings) if args.embeddings else None,
        assay_receipt=digest(args.assays / "receipt.json") if args.assays else None,
        fit_metadata=fit_metadata, forecast_sha256=digest(args.out / "forecast.parquet"),
        code_sha256=digest(Path(__file__))))


def ece(y, p):
    groups = np.minimum(np.floor(p * 10).astype(int), 9)
    return float(sum((groups == i).mean() * abs(y[groups == i].mean() - p[groups == i].mean())
                     for i in np.unique(groups)))


def metrics(data):
    y, p = data.label.to_numpy(), data.probability.to_numpy()
    if len(np.unique(y)) < 2:
        return dict(n=len(y), positive=int(y.sum()), auroc=None, auprc=None,
                    balanced_accuracy=None, brier=None, ece=None)
    return dict(n=len(y), positive=int(y.sum()), auroc=roc_auc_score(y, p),
                auprc=average_precision_score(y, p), balanced_accuracy=balanced_accuracy_score(y, data.decision),
                brier=brier_score_loss(y, p), ece=ece(y, p))


def score(args):
    files = json.loads((args.splits / "receipt.json").read_text())["files"]
    frames, freezes = [], {}
    for folder in sorted(args.forecasts.iterdir()):
        freeze_path = folder / "freeze.json"
        frozen = json.loads(freeze_path.read_text())
        if frozen["split_receipt"] != digest(args.splits / "receipt.json"):
            raise ValueError("Forecast was registered against a different split")
        file = folder / "forecast.parquet"
        if frozen["forecast_sha256"] != digest(file):
            raise ValueError("Frozen model prediction changed")
        fold = frozen["split"]
        labels = args.splits / fold / "score_labels.parquet"
        if digest(labels) != files[f"{fold}/{labels.name}"]:
            raise ValueError("Scoring label registry changed")
        predictions = pd.read_parquet(file)
        frames.append(predictions.join(pd.read_parquet(labels), how="left", validate="many_to_one")
                      .assign(split=fold.rsplit("-", 1)[0]))
        freezes[fold] = digest(freeze_path)
    if set(freezes) != {Path(key).parent.name for key in files if key.endswith("/query.parquet")}:
        raise ValueError("Entire split inventory must finish forecasting before scoring")
    results = pd.concat(frames)
    if results.label.isna().any() or results.concern.eq("Ambiguous").any():
        raise ValueError("Unknown or ambiguous outcome entered scoring")
    tables = []
    for (split, mode, method), group in results.groupby(["split", "mode", "method"]):
        tables.append(dict(split=split, mode=mode, method=method, support="all", **metrics(group)))
        if mode in ("cell_painting", "painting_mask", "painting_shuffle", "measured_rna"):
            covered = group.loc[group.measured_available]
            base_mode = "structure" if results["mode"].eq("structure").any() else "ecfp_properties"
            baseline = results.loc[results.split.eq(split) & results["mode"].eq(base_mode) & results.method.eq(method)]
            baseline = baseline.loc[baseline.index.isin(covered.index)]
            tables.extend([dict(split=split, mode=mode, method=method, support="matched", **metrics(covered)),
                           dict(split=split, mode=base_mode, method=method, support=f"matched_{mode}", **metrics(baseline))])
    args.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(tables).to_csv(args.out / "metrics.tsv", sep="\t", index=False, na_rep="NA")
    receipt(args.out / "receipt.json", dict(forecast_freezes=freezes,
        split_receipt=digest(args.splits / "receipt.json"), output_sha256=digest(args.out / "metrics.tsv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    forecast = actions.add_parser("predict")
    forecast.add_argument("--fold", required=True)
    forecast.add_argument("--embeddings", type=Path)
    forecast.add_argument("--assays", type=Path)
    scored = actions.add_parser("score")
    scored.add_argument("--forecasts", type=Path, required=True)
    for action in (forecast, scored):
        action.add_argument("--splits", type=Path, required=True)
        action.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (predict if args.action == "predict" else score)(args)


if __name__ == "__main__":
    main()
