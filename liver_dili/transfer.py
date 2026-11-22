"""Scaffold-blocked, bidirectional HepaRG and PHH pathway transfer."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from .assays import pathway_matrix, register_sets, valid
from .chemistry import identity
from .splits import data_groups, digest, receipt


def read_assay(root, name, compounds=None):
    table = pd.read_parquet(root / f"{name}.parquet")
    genes = pd.read_parquet(root / f"{name}-genes.parquet")
    keys = ["inchikey", "connectivity", "smiles", "dose_uM"]
    valid = table.dose_uM.between(1, 1000)
    if compounds is not None:
        valid &= table.inchikey.isin(compounds)
    data = pd.concat([table.loc[valid, keys], genes.loc[valid]], axis=1)
    return data.groupby(keys, sort=True).median().reset_index()


def match(early, late):
    matches = early[["inchikey", "dose_uM"]].reset_index().merge(
        late[["inchikey", "dose_uM"]].reset_index(), on="inchikey", suffixes=("_a", "_b"), validate="many_to_many")
    matches["distance"] = np.abs(np.log2(matches.dose_uM_a / matches.dose_uM_b))
    matches["min_dose"] = np.minimum(matches.dose_uM_a, matches.dose_uM_b)
    matches = matches.loc[matches.distance.le(np.log2(3))].sort_values(
        ["distance", "min_dose", "index_a", "index_b"], ascending=[True, False, True, True])
    return matches.drop_duplicates("inchikey").sort_values("inchikey").reset_index(drop=True)


def heparg_plates(source):
    rows = pd.read_csv(source, sep="\t", low_memory=False, usecols=["smiles", "dose_uM", "time_h", "plate_id"])
    rows = rows.loc[rows.time_h.eq(24) & rows.dose_uM.between(1, 1000)].copy()
    rows["identity"] = rows.smiles.map(valid)
    rows = rows.dropna(subset="identity")
    rows[["inchikey", "connectivity", "scaffold"]] = pd.DataFrame(
        [(x[1], x[2], x[3]) for x in rows.identity], index=rows.index)
    return (rows.groupby(["inchikey", "connectivity", "scaffold", "dose_uM"])
            .plate_id.agg(lambda ids: set(ids.dropna())).reset_index())


def normalized(x):
    magnitude = np.linalg.norm(x, axis=1)
    return np.divide(x, magnitude[:, None], out=np.zeros_like(x), where=magnitude[:, None] > 0)


def direct_gene(x):
    ranks = rankdata(x, axis=1)
    return normalized(ranks - np.mean(ranks, axis=1, keepdims=True))


def reduced_modules(x, y):
    donor = np.concatenate([x, y], axis=0).T
    donor -= donor.mean(axis=1, keepdims=True)
    fit = KMeans(n_clusters=24, n_init=5, random_state=20261122).fit(normalized(donor))
    return np.eye(24, dtype=float)[fit.labels_]


def ridge(train_x, train_y, query, alpha):
    """Exact centered multioutput ridge through the small compound Gram matrix."""
    x_center = train_x.mean(axis=0)
    y_center = train_y.mean(axis=0)
    centered = train_x - x_center
    kernel = centered @ centered.T + alpha * np.eye(len(train_x))
    coefficients = np.linalg.solve(kernel, train_y - y_center)
    return y_center + ((query - x_center) @ centered.T) @ coefficients


def fit_transfer(x, y, tr, te, groups):
    choices = []
    for alpha in (1., 10., 100.):
        errors = []
        for fit, val in GroupKFold(3).split(tr, groups=groups[tr]):
            error = ridge(x[tr[fit]], y[tr[fit]], x[tr[val]], alpha) - y[tr[val]]
            errors.append(np.mean(error**2))
        choices.append((np.mean(errors), alpha))
    _, alpha = min(choices)
    return ridge(x[tr], y[tr], x[te], alpha), alpha


def prepare(args):
    raw = args.raw
    map_file = pd.read_csv(args.hgnc, sep="\t", usecols=["entrez_id", "symbol"], dtype=str).dropna()
    entrez = map_file.drop_duplicates("entrez_id", keep=False).set_index("entrez_id").symbol
    sets = register_sets(raw, entrez)
    sets = {group: {name: genes for name, genes in sets.items() if name.startswith(group + "__")}
            for group in ("wikipathways", "hallmark", "progeny")}
    phh = read_assay(args.assays, "TGGATEs_PHH")
    plate_sources = {"GSE284321": args.heparg_284321_rows, "GSE296254": args.heparg_296254_rows}
    if any(plate_sources.values()) and not all(plate_sources.values()):
        raise ValueError("Physical plate control requires both HepaRG sample registries")
    all_predictions, fits, support = [], [], []
    for heparg in ("GSE284321", "GSE296254"):
        hep = read_assay(args.assays, heparg, set(phh.inchikey))
        matches = match(hep, phh)
        paired = hep.iloc[matches.index_a].reset_index(drop=True)
        paired["scaffold"] = paired.smiles.map(lambda s: identity(s)[3])
        plates = None
        if plate_sources[heparg] is not None:
            names = ["inchikey", "dose_uM"]
            plate_source = heparg_plates(plate_sources[heparg]).set_index(names, verify_integrity=True)
            plates = plate_source.loc[pd.MultiIndex.from_frame(paired[names]), "plate_id"].reset_index(drop=True)
            if any(not len(row) for row in plates):
                raise ValueError("A paired treatment is missing its physical plate ID")
        groups = data_groups(paired, None, "murcko").to_numpy()
        support.append(dict(hep=heparg, pairs=len(paired), scaffolds=len(set(groups)),
                            exact_dose=int(matches.distance.lt(1e-8).sum()),
                            median_fold_dose_ratio=float(np.median(2**matches.distance)) if len(paired) else None))
        if len(set(groups)) < 12:
            continue
        x = paired.filter(regex="^(?!inchikey$|connectivity$|smiles$|dose_uM$|scaffold$)")
        y = phh.iloc[matches.index_b].reset_index(drop=True).loc[:, x.columns]
        gene_x, gene_y = direct_gene(x.to_numpy()), direct_gene(y.to_numpy())
        for a, b, direction in ((x, y, "HepaRG_to_PHH"), (y, x, "PHH_to_HepaRG")):
            outcomes = {"genes": (gene_x, gene_y) if direction == "HepaRG_to_PHH" else (gene_y, gene_x)}
            outcomes.update({name: (pathway_matrix(a, definition)[0].to_numpy(),
                                    pathway_matrix(b, definition)[0].to_numpy())
                             for name, definition in sets.items()})
            for fold, (train, test) in enumerate(GroupKFold(4).split(paired, groups=groups)):
                if set(groups[train]) & set(groups[test]):
                    raise ValueError("A held scaffold entered the cross-platform fit")
                if plates is not None:
                    held_plates = set().union(*plates.iloc[test])
                    train = train[plates.iloc[train].map(lambda p: not bool(p & held_plates)).to_numpy()]
                if len(set(groups[train])) < 3:
                    raise ValueError("Insufficient independent source groups for nested transfer selection")
                fit_x, fit_y = outcomes["genes"]
                module_weights = reduced_modules(fit_x[train], fit_y[train])
                norm = np.maximum(module_weights.sum(axis=0, keepdims=True), 1)
                by_space = outcomes | {"fit_only_modules": (fit_x @ module_weights / norm,
                                                           fit_y @ module_weights / norm)}
                for name, (input_space, target) in by_space.items():
                    model, alpha = fit_transfer(input_space, target, train, test, groups)
                    permutation = np.random.default_rng(20261122 + fold).permutation(train)
                    shuffled = ridge(input_space[train], target[permutation], input_space[test], alpha)
                    fits.append(dict(hep=heparg, direction=direction, space=name, fold=fold,
                                     ridge=alpha, train=len(train), held=len(test), features=target.shape[1]))
                    # Delayed scoring binds observations via rows; predictions contain no truth vectors.
                    base = np.repeat(target[train].mean(axis=0, keepdims=True), len(test), axis=0)
                    for arm, forecast in (("direct_source", input_space[test]), ("ridge", model),
                                          ("train_mean", base), ("wrong_source_pairing", shuffled)):
                        for i, vec in zip(test, forecast, strict=True):
                            all_predictions.append(dict(hep=heparg, direction=direction, space=name,
                                fold=fold, arm=arm, inchikey=paired.inchikey[i], scaffold=groups[i],
                                forecast=vec.astype(np.float32).tolist()))
                    # Test-space module membership varies over folds and is retained for reconstruction.
                    if name == "fit_only_modules":
                        for feature, module in zip(x.columns, module_weights.argmax(axis=1), strict=True):
                            fits.append(dict(hep=heparg, direction=direction, space=name, fold=fold,
                                             module=int(module), symbol=feature))
    if not all_predictions:
        raise ValueError("No predeclared cross-platform pairs support held-scaffold prediction")
    args.out.mkdir(parents=True, exist_ok=False)
    forecasts = pd.DataFrame(all_predictions)
    forecasts.to_parquet(args.out / "forecast.parquet", index=False)
    p = pd.DataFrame(fits)
    p.to_parquet(args.out / "fits.parquet", index=False)
    receipt(args.out / "freeze.json", dict(assay_receipt=digest(args.assays / "receipt.json"),
        code_sha=digest(Path(__file__)), pathway_receipt=digest(args.raw / "pathways.sha256"),
        plate_exclusion=all(plate_sources.values()),
        physical_plate_metadata={name: digest(path) for name, path in plate_sources.items() if path},
        support=support, outputs={p.name: digest(p) for p in sorted(args.out.iterdir())}))


def score(args):
    freeze = args.forecast / "freeze.json"
    frozen = json.loads(freeze.read_text())
    if digest(freeze) != args.digest or frozen["assay_receipt"] != digest(args.assays / "receipt.json"):
        raise ValueError("Wrong cross-platform forecast freeze")
    for name, sha in frozen["outputs"].items():
        if digest(args.forecast / name) != sha:
            raise ValueError(f"Changed cross-platform forecast data {name}")
    table = pd.read_parquet(args.forecast / "forecast.parquet")
    fits = pd.read_parquet(args.forecast / "fits.parquet")
    registry = pd.read_csv(args.hgnc, sep="\t", usecols=["entrez_id", "symbol"], dtype=str).dropna()
    entrez = registry.drop_duplicates("entrez_id", keep=False).set_index("entrez_id").symbol
    definitions = register_sets(args.raw, entrez)
    conditions = {}
    phh = read_assay(args.assays, "TGGATEs_PHH", set(table.inchikey))
    metadata_fields = ["inchikey", "connectivity", "smiles", "dose_uM"]
    for name in sorted(table.hep.unique()):
        keys = set(table.loc[table.hep.eq(name), "inchikey"])
        hep = read_assay(args.assays, name, keys)
        pairs = match(hep, phh)
        for direction, target, axis in (("HepaRG_to_PHH", phh, "index_b"), ("PHH_to_HepaRG", hep, "index_a")):
            values = target.iloc[pairs[axis]].set_index("inchikey").drop(columns=metadata_fields[1:])
            conditions[name, direction, "genes"] = pd.DataFrame(direct_gene(values.to_numpy()),
                index=values.index, columns=values.columns)
            for group in ("wikipathways", "hallmark", "progeny"):
                group_sets = {key: value for key, value in definitions.items() if key.startswith(group + "__")}
                conditions[name, direction, group] = pathway_matrix(values, group_sets)[0]
    results = []
    for (hep, direction, space, fold, arm), pred in table.groupby(["hep", "direction", "space", "fold", "arm"]):
        if space == "fit_only_modules":
            source = conditions[hep, direction, "genes"].loc[pred.inchikey]
            selection = fits.loc[fits.hep.eq(hep) & fits.direction.eq(direction) & fits.space.eq(space)
                                 & fits.fold.eq(fold) & fits.symbol.notna()].set_index("symbol").module
            weights = np.eye(24)[selection.loc[source.columns].astype(int)]
            y = source.to_numpy() @ weights / np.maximum(1, weights.sum(axis=0))
        else:
            y = conditions[hep, direction, space].loc[pred.inchikey].to_numpy()
        forecast = np.stack(pred.forecast)
        cos = (normalized(y) * normalized(forecast)).sum(axis=1)
        sq = ((y - forecast)**2).mean(axis=1)
        sign = (np.sign(y) == np.sign(forecast)).mean(axis=1)
        results.append(pred.drop(columns="forecast").assign(cosine=cos, sq_error=sq, sign_agreement=sign))
    outcome = pd.concat(results, ignore_index=True)
    aggregate = outcome.groupby(["hep", "direction", "space", "arm"], as_index=False).agg(
        n=("inchikey", "size"), cosine=("cosine", "mean"), mse=("sq_error", "mean"),
        sign_agreement=("sign_agreement", "mean"))
    aggregate["rmse"] = np.sqrt(aggregate.pop("mse"))
    paired = []
    for (hep, direction, space), direction_rows in outcome.groupby(["hep", "direction", "space"]):
        base = direction_rows.loc[direction_rows.arm.eq("ridge")].set_index("inchikey")
        rng = np.random.default_rng(20261122)
        draw = rng.integers(len(base), size=(10000, len(base)))
        for control in ("direct_source", "train_mean", "wrong_source_pairing"):
            comparator = direction_rows.loc[direction_rows.arm.eq(control)].set_index("inchikey").loc[base.index]
            for metric in ("cosine", "sq_error", "sign_agreement"):
                delta = base[metric].to_numpy() - comparator[metric].to_numpy()
                low, high = np.quantile(delta[draw].mean(axis=1), [.005, .995])
                paired.append(dict(hep=hep, direction=direction, space=space, reference=control, metric=metric,
                                   n=len(delta), change=delta.mean(), low99=low, high99=high))
    args.out.mkdir(parents=True, exist_ok=False)
    aggregate.to_csv(args.out / "cross_platform.tsv", sep="\t", index=False)
    pd.DataFrame(paired).to_csv(args.out / "paired.tsv", sep="\t", index=False)
    receipt(args.out / "receipt.json", dict(forecast_sha=digest(freeze), support=frozen["support"],
        code_sha=digest(Path(__file__)), outputs={p.name: digest(p) for p in sorted(args.out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    for name in ("assays", "raw", "hgnc", "out", "forecast"):
        parser.add_argument(f"--{name}", type=Path, required=name != "forecast")
    parser.add_argument("--digest")
    parser.add_argument("--heparg-284321-rows", type=Path)
    parser.add_argument("--heparg-296254-rows", type=Path)
    args = parser.parse_args()
    (prepare if args.action == "freeze" else score)(args)
