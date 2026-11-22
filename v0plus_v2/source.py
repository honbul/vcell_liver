"""Grouped source-only development for the sensitivity head."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

from vc_build.external import KEY, SOURCE, align_endpoints, native_sensitivity
from vc_build.io import publish_json
from vc_build.v0plus_cv import attach_native, panel_targets, summarize, training_inputs, validation_roster

from .adaptive import FAMILY, NORMS, Projection, fit_candidates

MIN_TRAIN = 20


def grouped(fit, n):
    for left, right in GroupKFold(n).split(fit, groups=fit.lineage_group):
        develop, held = fit.iloc[left], fit.iloc[right]
        if set(develop.lineage_group) & set(held.lineage_group):
            raise ValueError("Source cell/patient lineage cross-over")
        yield develop, held


def load(path):
    rows, rna, meta = training_inputs(path)
    rna = rna.loc[:, sorted(rna.columns)]
    if len(rows) != 65_685 or rows.lineage_group.nunique() != 991:
        raise ValueError("V0+ source development roster changed")
    return rows, rna, meta


def correlation(x, y):
    a = x - np.mean(x, axis=0)
    b = y - y.mean()
    denominator = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b))
    return np.divide(b @ a, denominator, out=np.zeros(x.shape[1]), where=denominator > 0)


def selection_value(frame, keys, panel):
    """Keep assay-specific rankings and equal source weight in every drug's selection."""
    accum = []
    for _, source in frame.groupby(SOURCE):
        actual = source.endpoint_value.to_numpy()
        loc = keys.get_indexer(source.cell_id)
        if np.any(loc < 0):
            raise ValueError("Unexpected inner validation identity")
        if len(actual) >= 7 and np.std(actual) > 1e-7:
            predictions = panel[loc]
            pearson = correlation(predictions, actual)
            spearman = correlation(rankdata(predictions, axis=0), rankdata(actual))
            accum.append(.55 * spearman + .45 * pearson)
    if not accum:
        return np.zeros(panel.shape[1])
    return np.mean(accum, axis=0)


def setup(fit, rna):
    unique = rna.loc[rna.index.intersection(fit.cell_id)]
    projections = {norm: Projection.fit(unique, norm) for norm in NORMS}
    views = {name: obj.transform(rna) for name, obj in projections.items()}
    return projections, views


@dataclass
class DrugModels:
    projections: dict
    heads: dict
    anchor: pd.Series

    def predict(self, query, rna):
        transformed = {norm: view.transform(rna) for norm, view in self.projections.items()}
        base = query.compound_id.map(self.anchor).to_numpy()
        result = {method: base.copy() for method in self.heads}
        for method, drugs in self.heads.items():
            for drug, positions in query.groupby("compound_id").indices.items():
                if drug in drugs:
                    cell = rna.index.get_indexer(query.iloc[positions].cell_id)
                    if np.any(cell < 0):
                        raise ValueError("Missing expression for registered query cell")
                    head = drugs[drug]
                    result[method][positions] += head.apply(transformed[head.norm][cell])
        return result


def load_drug_models(path):
    """Read the originally submitted module-mode artifact under its trained class."""
    import __main__

    already = getattr(__main__, "DrugModels", None)
    __main__.DrugModels = DrugModels
    try:
        return joblib.load(path)
    finally:
        if already is None:
            del __main__.DrugModels
        else:
            __main__.DrugModels = already


def known_genes(rna, targets, drug):
    a = rna.columns.get_indexer(targets.get(drug, []))
    return a[a >= 0]


def tune(fit, rna, targets):
    evaluations = defaultdict(lambda: defaultdict(list))
    inner_values = []
    for i, (train, val) in enumerate(grouped(fit, 2)):
        projections, views = setup(train, rna)
        panel, anchor, _, _ = panel_targets(train)
        observed = val.assign(endpoint_value=native_sensitivity(val))
        for drug, tr in panel.groupby("compound_id", sort=True):
            if len(tr) < MIN_TRAIN:
                continue
            check = observed.loc[observed.compound_id.eq(drug)]
            if check.empty:
                continue
            cells = pd.Index(sorted(check.cell_id.unique()))
            before, after = rna.index.get_indexer(tr.cell_id), rna.index.get_indexer(cells)
            results = list(fit_candidates(views, projections, before, after,
                                          tr.residual.to_numpy(), known_genes(rna, targets, drug)))
            if not results:
                continue
            cfg, _, forecasts = zip(*results)
            values = selection_value(check, cells, np.stack(forecasts, axis=1))
            for setting, value in zip(cfg, values):
                evaluations[drug][setting].append(value)
            pred_matrix = np.stack(forecasts, axis=1)
            for col, setting in enumerate(cfg):
                inner_values.append(pd.DataFrame({
                    "inner": i, "cell_id": cells, "compound_id": drug, "model": cfg_id(setting),
                    "forecast": anchor[drug] + pred_matrix[:, col],
                }))
    selected, stats = {}, []
    for drug, opts in evaluations.items():
        best = {}
        for family in FAMILY:
            options = {k: np.mean(v) for k, v in opts.items() if k[3] == family}
            if options:
                best[family] = max(options, key=options.get)
        if best:
            best["adaptive"] = max(best.values(), key=lambda c: np.mean(opts[c]))
        selected[drug] = best
        for name, setting in best.items():
            stats.append({"compound_id": drug, "model": name, "cfg": list(setting),
                          "objective": float(np.mean(opts[setting]))})
    return selected, stats, pd.concat(inner_values, ignore_index=True)


def cfg_id(setting):
    return "/".join(map(str, setting))


def fit_models(fit, rna, targets, selected):
    panel, anchor, _, _ = panel_targets(fit)
    projections, views = setup(fit, rna)
    heads = {name: {} for name in (*FAMILY, "adaptive")}
    for drug, rows in panel.groupby("compound_id"):
        options = selected.get(drug, {})
        if not options:
            continue
        ids = rna.index.get_indexer(rows.cell_id)
        cfgs = list(dict.fromkeys(options.values()))
        found = {setting: model for setting, model, _ in fit_candidates(
            views, projections, ids, np.array([], dtype=int),
            rows.residual.to_numpy(), known_genes(rna, targets, drug), configs=cfgs)}
        for name, setting in options.items():
            heads[name][drug] = found[setting]
    return DrugModels(projections, heads, anchor)


def inner_chosen(raw, selected):
    rows = []
    for (drug, model), block in raw.groupby(["compound_id", "model"], sort=False):
        for name, cfg in selected.get(drug, {}).items():
            if model == cfg_id(cfg):
                rows.append(block.drop(columns="model").assign(method=name))
    return pd.concat(rows, ignore_index=True)


def report_failures(validation, predicted, protocols):
    actual = validation.merge(predicted[KEY + ["predicted_value"]], on=["cell_id", "lineage_group", "compound_id"],
                              validate="many_to_one").assign(endpoint_value=native_sensitivity(validation).to_numpy())
    # Use the aligned join's native labels so the diagnosis is independent of input row ordering.
    actual["endpoint_value"] = native_sensitivity(actual)
    diagnostics = []
    for (source, compound), group in actual.groupby(["source_id", "compound_id"]):
        value = group.endpoint_value.to_numpy()
        p = group.predicted_value.to_numpy()
        diagnostics.append({"source_id": source, "compound_id": compound, "n": len(group),
                            "response_sd": float(np.std(value)),
                            "signed_error_mean": float(np.mean(p-value)),
                            "drug_spearman": float(correlation(rankdata(p)[:, None], rankdata(value))[0]),
                            "drug_pearson": float(correlation(p[:, None], value)[0])})
    out = pd.DataFrame(diagnostics)
    protocol_names = {"parent_inchikey": "compound_id"}
    protocols = protocols.rename(columns=protocol_names)
    if {"source_id", "compound_id"} <= set(protocols):
        out = out.merge(protocols, on=["source_id", "compound_id"], how="left")
    return out


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    fit, rna, receipt = load(args.bundle)
    train, validation = list(grouped(fit, 3))[args.fold]
    targets = json.loads(args.targets.read_text())
    fit_rna = rna.loc[rna.index.intersection(train.cell_id.unique())]
    selected, choices, raw = tune(train, fit_rna, targets)
    chosen = inner_chosen(raw, selected)
    chosen.to_parquet(args.out / "inner.parquet", index=False)
    model = fit_models(train, fit_rna, targets, selected)
    query = validation_roster(validation)
    values = model.predict(query, rna.loc[rna.index.intersection(validation.cell_id.unique())])
    forecasts = pd.concat([query[KEY].assign(method=method, predicted_value=forecast)
                           for method, forecast in values.items()], ignore_index=True)
    out = attach_native(forecasts, validation, args.fold)
    out.to_parquet(args.out / "dev_outer.parquet", index=False)
    joblib.dump(model, args.out / "models.joblib", compress=2)
    protocol = pd.read_csv(args.bundle / "audit/source_protocols.tsv", sep="\t")
    report_failures(validation, forecasts.loc[forecasts.method.eq("adaptive")], protocol).to_csv(
        args.out / "diagnostics.tsv", sep="\t", index=False)
    summary = summarize(out)
    publish_json(args.out / "receipt.json", {"fold": args.fold, "train_rows": len(train), "source_sha256": {
        key: val["sha256"] for key, val in receipt["artifacts"].items() if key.startswith("train/")},
        "selection": choices, "evaluation": summary.to_dict("records")})
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    from .source import DrugModels

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "targets", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--fold", choices=range(3), type=int, required=True)
    run(parser.parse_args())
