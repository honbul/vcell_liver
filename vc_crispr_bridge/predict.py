"""Offline prediction entrypoint for HepG2 Perturb-seq transfer."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd

from .perturbseq import record


PANELS = ("gene_cold", "same_gene")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked(root, name, receipt):
    path = root / name
    if record(path) != receipt["files"][name]:
        raise ValueError(f"Integrity failure for {path}")
    return path


@dataclass
class Assay:
    meta: pd.DataFrame
    context: np.ndarray
    delta: np.ndarray | None = None


def assay(root, name, receipt, genes, observed):
    meta = pd.read_parquet(checked(root, f"{name}/identities.parquet", receipt))
    x = np.load(checked(root, f"{name}/control_context.npy", receipt), allow_pickle=False)
    y = np.load(checked(root, f"{name}/delta.npy", receipt), allow_pickle=False) if observed else None
    if len(meta) == 0 or not meta.id.is_unique or x.shape != (len(meta), len(genes)):
        raise ValueError(f"Invalid registered role {name}")
    if not observed and set(meta.columns) != {"id", "assay", "target_ensembl", "target_symbols"}:
        raise ValueError(f"Unexpected query covariates in {name}")
    if not np.isfinite(x).all() or (observed and (y.shape != x.shape or not np.isfinite(y).all())):
        raise ValueError(f"Invalid assay values in {name}")
    return Assay(meta, x, y)


def confirm_assays(training, validation, cold, seen):
    fit_genes = set(training.meta.target_ensembl)
    dev_genes = set(validation.meta.target_ensembl)
    cold_genes = set(cold.meta.target_ensembl)
    same_genes = set(seen.meta.target_ensembl)
    if fit_genes & dev_genes or cold_genes & (fit_genes | dev_genes) or not same_genes <= fit_genes:
        raise ValueError("Gene identities cross a cold boundary or same-gene support is missing")
    if cold_genes & same_genes or len(cold_genes) != len(cold.meta) or len(same_genes) != len(seen.meta):
        raise ValueError("Repeated query interventions")
    source_assays = set(training.meta.assay) | set(validation.meta.assay)
    if (set(cold.meta.assay) | set(seen.meta.assay)) & source_assays:
        raise ValueError("HepG2 assay is present in source fitting or tuning")


def predict(a):
    import torch
    from .gene_features import GeneFeatures
    from .models import (
        context_transform, copy_same_gene, equal_target_weights, feature_arrays, fit_neural,
        fit_reduced, fit_ridge, targeted_attenuation, validation_loss,
    )

    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    started = perf_counter()
    input_receipt = a.panel / "receipt.json"
    receipt = json.loads(input_receipt.read_text())
    genes = pd.Index(pd.read_csv(checked(a.panel, "genes.tsv", receipt), sep="\t").gene_id)
    if not genes.is_unique or len(genes) < 20:
        raise ValueError("Invalid expression gene axis")
    train = assay(a.panel, "fit", receipt, genes, observed=True)
    valid = assay(a.panel, "validation", receipt, genes, observed=True)
    queries = [assay(a.panel, "query_" + name, receipt, genes, observed=False) for name in PANELS]
    confirm_assays(train, valid, *queries)
    source_mean = np.average(train.delta, axis=0, weights=equal_target_weights(train.meta))
    scores, choices = {}, {}
    all_evaluation = [valid, *queries]

    def add(name, matrices, setting):
        evaluation, *outputs = matrices
        scores[name] = outputs
        choices[name] = {
            "source_validation_centered_rmse": validation_loss(
                evaluation, valid.delta, valid.meta.assay.to_numpy(),
            ),
            **setting,
        }
        print(f"{name}: source-validation centered RMSE {choices[name]['source_validation_centered_rmse']:.6f}",
              flush=True)

    add("no_change", [np.zeros_like(r.context) for r in all_evaluation], {})
    add("source_mean", [np.broadcast_to(source_mean, r.context.shape) for r in all_evaluation], {})
    add("same_gene_copy", copy_same_gene(train, all_evaluation, source_mean), {
        "unmatched_target_fallback": "source_mean",
    })
    coefficient, attenuated = targeted_attenuation(genes, train, valid, queries)
    add("target_expression_attenuation", attenuated, {"slope": coefficient})

    store = GeneFeatures(a.store, a.proteins, a.hgnc, a.links, a.aliases)
    roles = [train, *all_evaluation]
    contexts = context_transform(*(r.context for r in roles))
    combined = feature_arrays(store, roles, contexts, "combined", True)
    for name, model in (
        ("ridge", fit_ridge), ("elasticnet", lambda x, t, v: fit_reduced("elastic", x, t, v)),
        ("lightgbm", lambda x, t, v: fit_reduced("trees", x, t, v)),
    ):
        chosen = model(combined[:2], train, valid)
        add(name, [chosen.model.predict(x) for x in combined[1:]], chosen.setting)
    for name, representation, with_context in (
        ("conditional_mlp", "combined", True),
        ("mlp_no_context", "combined", False),
        ("mlp_esm2_only", "esm", True),
        ("mlp_string_only", "string", True),
        ("string_graph_gnn", "graph", True),
    ):
        arrays = feature_arrays(store, roles, contexts, representation, with_context)
        # The graph network projects the target and two weighted propagation
        # radii separately before the context-conditioned expression decoder.
        graph_block = store.one_hop.shape[1] if representation == "graph" else 0
        chosen = fit_neural(arrays[:2], train, valid, a.device, a.seed, a.epochs, graph_block)
        add(name, [chosen.model.predict(x) for x in arrays[1:]], chosen.setting)

    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        output = Path(directory)
        np.save(output / "source_mean.npy", source_mean, allow_pickle=False)
        pd.DataFrame({"gene_id": genes}).to_csv(output / "genes.tsv", sep="\t", index=False)
        for i, (name, role) in enumerate(zip(PANELS, queries)):
            rows = {"id": role.meta.id.to_numpy(dtype=str)}
            rows.update({m: np.asarray(p[i], dtype=np.float32) for m, p in scores.items()})
            if any(not np.isfinite(value).all() for key, value in rows.items() if key != "id"):
                raise ValueError(f"Nonfinite prediction on {name}")
            np.savez(output / f"{name}.npz", **rows)
        write_json(output / "receipt.json", {
            "panel": record(input_receipt), "gene_covariates": store.receipt,
            "models": choices, "device": torch.cuda.get_device_name() if a.device == "cuda" else "cpu",
            "stopping_rule": "Minimum source-only gene-cold assay macro centered RMSE",
            "fit_genes": train.meta.target_ensembl.nunique(), "validation_genes": valid.meta.target_ensembl.nunique(),
            "output_genes": len(genes), "elapsed_seconds": perf_counter() - started,
            "esm2_coverage": {name: int(store.encode(role.meta.target_ensembl, "esm")[:, -1].sum())
                              for name, role in zip(("fit", "validation", *PANELS), roles)},
            "string_coverage": {name: int(store.encode(role.meta.target_ensembl, "string")[:, -1].sum())
                               for name, role in zip(("fit", "validation", *PANELS), roles)},
            "files": {f.name: record(f) for f in sorted(output.iterdir())},
        })
        output.rename(a.output)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "chronos":
        from .chronos import arguments, predict as predict_chronos

        del sys.argv[1]
        predict_chronos(arguments())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["hepg2"], metavar="{hepg2,chronos}")
    for name in ("panel", "output", "store", "proteins", "hgnc", "links", "aliases"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=41026)
    parser.add_argument("--epochs", type=int, default=55)
    arguments = parser.parse_args()
    if arguments.epochs < 1:
        parser.error("--epochs must be positive")
    predict(arguments)


if __name__ == "__main__":
    main()
