"""Render the frozen malignant-cell resistance distributions and UMAPs."""
import argparse
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from umap import UMAP

from .cohorts import file_record
from .singlecell_inputs import files_receipt, load_embedding
from .source_panels import frozen


def figure(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = a.predictions / "receipt.json"
    embeddings_path = a.scgpt / "receipt.json"
    registry_path = a.singlecell / "receipt.json"
    scores_path = a.scores / "receipt.json"
    model, emb, raw, score = [json.loads(p.read_text())
                              for p in (prediction_path, embeddings_path, registry_path, scores_path)]
    if score["predictions"] != file_record(prediction_path) or model["inputs"]["scgpt"] != file_record(embeddings_path):
        raise ValueError("Plot inputs differ from the scored forecasts")
    if model["inputs"]["singlecell"] != file_record(registry_path):
        raise ValueError("Plot candidate registry differs from the forecast")
    forecasts = pd.concat([
        pd.read_parquet(frozen(a.predictions, f"fold-{fold}.parquet", model)) for fold in range(5)
    ], ignore_index=True)
    choices = ("scgpt_adversarial", "scgpt_mmd", "scgpt_mlp_no_adaptation")
    method = max(choices, key=lambda key: float(np.mean([
        fold["models"][key]["source_validation_spearman"] for fold in model["folds"].values()
    ])))
    predicted = forecasts[forecasts.drug.eq("Sorafenib") & forecasts.role.eq("patient")]
    source = pd.read_csv(frozen(a.scores, "sample_fractions.tsv", score), sep="\t")
    source = source[source.model.eq(method)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    rng = np.random.default_rng(a.seed)
    instances, rows = [], []
    for i, cohort in enumerate(("GSE149614", "GSE151530")):
        metadata = pd.read_parquet(frozen(a.singlecell, f"{cohort}/scoring_metadata.parquet", raw))
        malignant = predicted.merge(metadata[["id", "donor", "malignant_candidate", "source_sample"]],
                                     on=["id", "donor"], validate="1:1")
        malignant = malignant[malignant.malignant_candidate]
        counts = malignant.groupby(["donor", "source_sample"]).id.size()
        rank = counts[counts.ge(25)].groupby(level=0).agg(["size", "sum"])
        if len(rank) == 0:
            raise ValueError(f"No sufficiently sampled malignant donor in {cohort}")
        chosen = rank.sort_values(["size", "sum"], ascending=False).index[0]
        patient = malignant[malignant.donor.eq(chosen)].sort_values("id")
        if len(patient) > 3500:
            patient = patient.iloc[np.sort(rng.choice(len(patient), size=3500, replace=False))]
        context = load_embedding(frozen(a.scgpt, f"{cohort}.npz", emb)).loc[patient.id].to_numpy(np.float32)
        xy = UMAP(n_neighbors=20, min_dist=0.25, metric="cosine", random_state=a.seed,
                  n_jobs=1).fit_transform(context)
        membership = KMeans(n_clusters=4, random_state=a.seed, n_init=10).fit_predict(context)
        point = axes[0, i].scatter(xy[:, 0], xy[:, 1], c=patient[method], s=7, cmap="viridis",
                                   alpha=0.85, linewidths=0, rasterized=True)
        axes[0, i].set(title=f"{cohort} · {chosen}", xlabel="UMAP 1", ylabel="UMAP 2")
        fig.colorbar(point, ax=axes[0, i], label="Sorafenib resistance forecast")
        for state in np.unique(membership):
            selected = membership == state
            rows.append({"cohort": cohort, "donor": chosen, "state": int(state) + 1,
                         "cells_plotted": int(selected.sum()),
                         "median_forecast": float(np.median(patient[method].to_numpy()[selected]))})
        subplot = axes[1, i]
        series = source[source.cohort.eq(cohort) & source.cells.ge(20)]
        if cohort == "GSE149614":
            retain = [s for s in ("Tumor", "PVTT", "Lymph") if s in set(series.site)]
            values = [series.loc[series.site.eq(label), "high_resistance_fraction"].to_numpy() for label in retain]
        else:
            retain = ["Biopsy"]
            values = [series.high_resistance_fraction.to_numpy()]
        if all(len(v) for v in values):
            subplot.boxplot(values, labels=retain, showfliers=False)
            for j, values_at_site in enumerate(values, start=1):
                subplot.scatter(rng.uniform(j - 0.12, j + 0.12, len(values_at_site)), values_at_site,
                                s=12, color="#2b6d83", alpha=0.7)
        subplot.set(ylabel="High forecast fraction per biopsy", ylim=(-0.025, 1.025))
        instances.append({"cohort": cohort, "donor": chosen, "displayed_cells": len(patient)})
    fig.suptitle("Malignant cell expression states and held-donor sorafenib forecasts", fontsize=15)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        staged = Path(d)
        fig.savefig(staged / "malignant_resistance_umap.png", dpi=215)
        plt.close(fig)
        pd.DataFrame(rows).to_csv(staged / "displayed_transcriptional_states.tsv", sep="\t", index=False)
        (staged / "receipt.json").write_text(json.dumps({
            "predictions": file_record(prediction_path), "scoring": file_record(scores_path),
            "embedding": file_record(embeddings_path), "model": method, "displayed_donors": instances,
            "projection": "Descriptive per-donor UMAP of the frozen expression vectors after prediction and scoring",
            "files": files_receipt(staged),
        }, indent=2, sort_keys=True) + "\n")
        staged.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("predictions", "scores", "singlecell", "scgpt", "output"):
        p.add_argument("--" + arg, required=True, type=Path)
    p.add_argument("--seed", type=int, default=51226)
    figure(p.parse_args())


if __name__ == "__main__":
    main()
