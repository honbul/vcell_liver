"""Fit and freeze pan-cancer combination predictors with blocked HCC queries."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from .context import verified
from .features import Library
from .models import (CLASSICAL, FAMILIES, NeuralInputs, fit_classical, fit_neural,
                     source_weights, targets)
from .splits import KEYS, OUTCOMES, record, validate_split


def training_measure(y, predictions, source):
    # Give Zip, Bliss and actual added inhibition equal weight on the
    # Sanger protocol. CSS is measured only by DrugComb and stays separate.
    sanger = np.char.startswith(source.astype(str), "Sanger:")
    output, uncertainty = {}, {}
    for j, name in enumerate(OUTCOMES):
        scope = ~sanger if name == "css" else sanger
        use = np.isfinite(y[:, j]) & scope
        if use.sum() < 10:
            raise ValueError(f"Independent selection endpoint too sparse for {name}")
        if np.unique(predictions[use, j]).size < 2:
            value = 0.
        else:
            value = float(spearmanr(y[use, j], predictions[use, j]).statistic)
        output[name] = value
        uncertainty[name] = float(np.quantile(np.abs(predictions[use, j] - y[use, j]), .95))
    return float(np.mean(list(output.values()))), output, uncertainty


def supported(rows, library):
    return rows[rows.cell_id.isin(library.rna_raw.index) & rows.compound_a.isin(library.ids)
                & rows.compound_b.isin(library.ids)].copy()


def means(train, y, predicted):
    support = train[["source", "pair"]].copy()
    for i, name in enumerate(OUTCOMES):
        support[name] = y[:, i]
    overall = np.nanmean(y, axis=0)
    by_domain = support.groupby("source")[OUTCOMES].mean()
    by_pair = support.groupby(["source", "pair"])[OUTCOMES].mean()
    domain = by_domain.reindex(predicted.source)[OUTCOMES].to_numpy()
    paired = by_pair.reindex(pd.MultiIndex.from_frame(predicted[["source", "pair"]]))[OUTCOMES].to_numpy()
    domain = np.where(np.isfinite(domain), domain, overall)
    paired = np.where(np.isfinite(paired), paired, domain)
    return domain, paired


def controls(fit, queries, y, view):
    mean, pair = means(fit, y, queries)
    no_add = np.zeros_like(mean)
    mono = view.mono
    a, b = mono.T
    combined = a + b - a * b
    # Non-interaction synergy is zero; the two efficacy axes are scored
    # only where both single-agent values are independently assayed.
    no_add[:, OUTCOMES.index("css")] = 100 * combined
    no_add[:, OUTCOMES.index("added_inhibition")] = 100 * (combined - np.maximum(a, b))
    targeted = np.column_stack([
        -view.dependency_left[:, 1] - view.dependency_right[:, 1]
    ] * len(OUTCOMES))
    return {"mean": mean, "pair_mean": pair, "mono_bliss_independent": no_add,
            "target_dependency_sum": targeted}


def save_predictions(name, rows, forecast):
    frame = pd.DataFrame(forecast, columns=OUTCOMES)
    frame["id"] = rows.id.to_numpy()
    frame["model"] = name
    return frame


def forecast(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    contract_path = a.panel / "receipt.json"
    contract = json.loads(contract_path.read_text())
    library = Library(a.context, a.embeddings, a.hallmark)
    eligible = {}
    for role in ("fit", "validation", "query"):
        raw = pd.read_parquet(verified(a.panel, role + ".parquet"))
        if role == "query" and set(raw.columns) != set(KEYS + ["pair"]):
            raise ValueError("Held-out combination predictors require an outcome-free query")
        eligible[role] = supported(raw, library)
        if len(eligible[role]) < 10:
            raise ValueError(f"No verified shared features for {role}")
    fit, valid, query = [eligible[r] for r in ("fit", "validation", "query")]
    held = set(contract["reserved_scaffolds"])
    axis = contract["axis"]
    validate_split(fit, valid, query, axis, held if axis == "ldo" else None)
    y, yv = targets(fit), targets(valid)
    w = source_weights(fit.source.to_numpy(), y)
    final_train = pd.concat([fit, valid], ignore_index=True)
    yt = targets(final_train)
    wt = source_weights(final_train.source.to_numpy(), yt)
    screen_rows, screen_rna = None, None
    if a.screen is not None:
        if axis != "lco_hcc":
            raise ValueError("Virtual experiments use only the liver-excluded training model")
        screen_rows = pd.read_parquet(verified(a.screen, "query.parquet"))
        screen_rna = pd.read_parquet(verified(a.screen, "rna.parquet"))
        if set(screen_rows.cell_id) != set(screen_rna.index) or "outcome" in screen_rows:
            raise ValueError("Virtual query must be expression-only before ranking")
    inner, outer = library.fold(fit), library.fold(final_train, screen_rna)
    screen_view = outer.view(screen_rows) if screen_rows is not None else None
    screened = []
    tf, tv = inner.view(fit), inner.view(valid)
    tr, tq = outer.view(final_train), outer.view(query)
    parts = [save_predictions(n, query, m) for n, m in controls(final_train, query, yt, tq).items()]
    torch.set_num_threads(4)
    opts, models_saved = {}, {}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as temp:
        stage = Path(temp)
        x = v = fx = q = x_screen = n_inner = n_outer = None
        for name in a.models:
            t0 = time.monotonic()
            classic = name in CLASSICAL
            if classic and x is None:
                x, v = tf.classical(inner), tv.classical(inner)
                fx, q = tr.classical(outer), tq.classical(outer)
                if screen_view is not None:
                    x_screen = screen_view.classical(outer)
            if not classic and n_inner is None:
                n_inner, n_outer = NeuralInputs(inner, a.device), NeuralInputs(outer, a.device)
                x = v = fx = q = x_screen = None
            scores = []
            for config in FAMILIES[name]:
                if classic:
                    valid_fit = fit_classical(name, config, x, y, w, a.seed)
                    provisional = valid_fit(v)
                else:
                    valid_fit = fit_neural(name, config, n_inner, tf, y, w, a.seed)
                    provisional = valid_fit.predict(n_inner, tv)
                value, metrics, uncertainty = training_measure(yv, provisional, valid.source.to_numpy())
                scores.append({"candidate": config, "mean_validation_spearman": value,
                               "residual_abs_q95": uncertainty, **metrics})
                print(f"{a.panel.name} {name} {config} validation={value:.4f}", flush=True)
            best = max(scores, key=lambda d: d["mean_validation_spearman"])
            config = best["candidate"]
            if classic:
                estimator = fit_classical(name, config, fx, yt, wt, a.seed)
                predictions = estimator(q)
            else:
                estimator = fit_neural(name, config, n_outer, tr, yt, wt, a.seed)
                predictions = estimator.predict(n_outer, tq)
                if name in {"cross_attention", "deepsynergy", "matchmaker"}:
                    path = stage / f"{name}.pt"
                    torch.save({"state_dict": estimator.model.state_dict(),
                                "center": torch.from_numpy(estimator.center),
                                "scale": torch.from_numpy(estimator.scale)}, path)
                    models_saved[name] = record(path)
            if screen_view is not None:
                virtual = estimator(x_screen) if classic else estimator.predict(n_outer, screen_view)
                if not np.isfinite(virtual).all():
                    raise ValueError(f"Nonfinite virtual combination predictions for {name}")
                screened.append(save_predictions(name, screen_rows, virtual))
            if not np.isfinite(predictions).all():
                raise ValueError(f"Nonfinite predicted outcomes for {name}")
            parts.append(save_predictions(name, query, predictions))
            opts[name] = {"validation": best, "all_candidates": scores,
                          "wall_seconds": time.monotonic() - t0}
        if screened:
            pd.concat(screened, ignore_index=True).to_parquet(stage / "screen_predictions.parquet", index=False)
        file = stage / "predictions.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(file, index=False)
        (stage / "receipt.json").write_text(json.dumps({
            "panel": record(contract_path), "context": record(a.context / "receipt.json"),
            "features": {str(kind): record(a.embeddings / kind / "receipt.json")
                         for kind in ("ecfp4", "molformer")},
            "hallmark": record(a.hallmark), "model_selection": opts, "neural_models": models_saved,
            "device": a.device, "seed": a.seed,
            "screen_registry": record(a.screen / "receipt.json") if a.screen is not None else None,
            "role_counts": {key: len(value) for key, value in eligible.items()},
            "original_query_rows": contract["test_rows"],
            "measured_independent_mono_for_both": int(np.isfinite(tq.mono).all(axis=1).sum()),
            "files": {f.name: record(f) for f in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "context", "embeddings", "hallmark", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--screen", type=Path)
    p.add_argument("--seed", type=int, default=2551)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--models", choices=tuple(FAMILIES), nargs="+", default=tuple(FAMILIES))
    a = p.parse_args()
    if len(a.models) != len(set(a.models)):
        p.error("Select distinct model families")
    forecast(a)


if __name__ == "__main__":
    main()
