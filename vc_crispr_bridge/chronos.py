"""DepMap Chronos prediction and detached assay scoring."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .gene_features import protein_by_symbol, verified_vectors
from .perturbseq import record
from .predict import PANELS, checked, write_json
from .regression import regression_metrics, train_scalar, validation_value


def validate_design(roles, models):
    train, valid, cold, same = roles
    donors = [set(role.donor) for role in (train, valid, cold)]
    if donors[0] & donors[1] or donors[0] & donors[2] or donors[1] & donors[2] or donors[2] != set(same.donor):
        raise ValueError("Chronos donor overlap or mismatched adult HCC panels")
    genes = [set(role.gene) for role in (train, valid, cold)]
    if genes[0] & genes[1] or genes[0] & genes[2] or genes[1] & genes[2] or genes[0] != set(same.gene):
        raise ValueError("Chronos gene crosses a cold boundary or same-gene panel is misaligned")
    for i, role in enumerate(roles):
        if role.duplicated(["model_id", "gene"]).any():
            raise ValueError("Repeated gene and model assay")
        chosen = models.iloc[role.cell_index.to_numpy()]
        if not np.array_equal(chosen.model_id.to_numpy(), role.model_id.to_numpy()):
            raise ValueError("RNA row index and DepMap identities differ")
        if not chosen.partition.eq("other" if i < 2 else "hcc").all():
            raise ValueError("Liver identity crosses the adult HCC transfer boundary")


def rna_context(rna, fit_indices):
    source = rna[fit_indices]
    variances = np.nanvar(source, axis=0)
    keep = np.argsort(np.nan_to_num(variances, nan=-np.inf))[-min(rna.shape[1], 2500):]
    values = rna[:, keep].copy()
    median = np.nan_to_num(np.nanmedian(values[fit_indices], axis=0), nan=0)
    rows, cols = np.where(~np.isfinite(values))
    values[rows, cols] = median[cols]
    return PCA(n_components=min(32, len(fit_indices) - 1, len(keep)), svd_solver="randomized",
               random_state=41026).fit(values[fit_indices]).transform(values).astype(np.float32)


def gene_latents(vectors, gene_map, roles):
    axis = pd.Index(sorted(set().union(*(set(role.gene) for role in roles))))
    symbol_axis = gene_map.set_index("ensembl_gene_id").symbol.reindex(axis)
    raw = vectors.reindex(symbol_axis)
    present = raw.notna().any(axis=1).to_numpy(dtype=np.float32)
    encoded = raw.fillna(0).to_numpy(dtype=np.float32)
    ids = axis.get_indexer(pd.Index(roles[0].gene.unique()))
    fit = encoded[ids]
    dims = min(24, encoded.shape[1], len(fit) - 1)
    reduction = PCA(n_components=dims, svd_solver="randomized", random_state=41026).fit(fit)
    result = np.column_stack((reduction.transform(encoded), present))
    return [result[axis.get_indexer(role.gene)] for role in roles]


def molecular_covariates(roles):
    train = roles[0][["rna", "cnv", "mut"]].to_numpy(np.float32)
    medians = np.nan_to_num(np.nanmedian(train, axis=0), nan=0)
    local = []
    for r in roles:
        x = r[["rna", "cnv", "mut"]].to_numpy(np.float32, copy=True)
        missing = ~np.isfinite(x)
        _, column = np.where(missing)
        x[missing] = medians[column]
        local.append(np.column_stack((x, missing.astype(np.float32))))
    return local


def predict(a):
    import torch

    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    json_path = a.panel / "receipt.json"
    receipt = json.loads(json_path.read_text())
    roles = [pd.read_parquet(checked(a.panel, f"{name}.parquet", receipt))
             for name in ("fit", "validation", *("query_" + p for p in PANELS))]
    allowed = {"model_id", "gene", "cell_index", "rna", "cnv", "mut", "donor"}
    if any(set(role.columns) != allowed for role in roles[2:]):
        raise ValueError("Outcome field or unregistered covariate in a query")
    models = pd.read_csv(checked(a.panel, "models.tsv", receipt), sep="\t")
    validate_design(roles, models)
    genes = pd.read_csv(checked(a.panel, "gene_map.tsv", receipt), sep="\t")
    rna = np.load(checked(a.panel, "rna.npy", receipt), allow_pickle=False)
    cells = rna_context(rna, roles[0].cell_index.unique())
    basal = [cells[r.cell_index.to_numpy()] for r in roles]
    local = molecular_covariates(roles)
    esm = gene_latents(protein_by_symbol(verified_vectors(a.store / "esm2"), a.proteins), genes, roles)
    graph = gene_latents(verified_vectors(a.store / "string"), genes, roles)
    train, valid, *query = roles
    y, vy = train.chronos.to_numpy(), valid.chronos.to_numpy()
    vg = valid.gene.to_numpy()
    model_predictions = [r[["model_id", "gene"]].copy() for r in query]
    metadata = {}

    def add(name, values, settings, value=None):
        for table, pred in zip(model_predictions, values):
            table[name] = pred.astype(np.float32)
        metadata[name] = {
            "settings": settings,
            "source_validation": {"within_gene_spearman": value[0], "rmse": -value[1]} if value else {},
        }
        print(name, metadata[name]["source_validation"], flush=True)

    mean = y.mean()
    add("mean", [np.full(len(q), mean) for q in query], {},
        validation_value(np.full(len(vy), mean), vy, vg))
    gene_means = train.groupby("gene", sort=False).chronos.mean()
    add("source_gene_mean", [q.gene.map(gene_means).fillna(mean).to_numpy() for q in query],
        {"unseen_gene_fallback": "overall_source_mean"},
        validation_value(np.full(len(vy), mean), vy, vg))
    perm = np.random.default_rng(a.seed).permutation(len(y))
    for name, estimator, blocks in (
        ("target_omics_marker", "ridge", [local]),
        ("ridge", "ridge", [basal, local, esm, graph]),
        ("elasticnet", "elasticnet", [basal, local, esm, graph]),
        ("lightgbm", "lightgbm", [basal, local, esm, graph]),
        ("conditional_mlp", "mlp", [basal, local, esm, graph]),
        ("mlp_no_esm2", "mlp", [basal, local, graph]),
        ("mlp_no_string", "mlp", [basal, local, esm]),
        ("mlp_no_cell_context", "mlp", [esm, graph]),
        ("mlp_no_target_omics", "mlp", [basal, esm, graph]),
    ):
        x = [np.column_stack(parts) for parts in zip(*blocks)]
        x[0] = x[0][perm]
        result = train_scalar(estimator, x, y[perm], vy, vg, a.device, a.seed,
                              modulus=2 if estimator == "lightgbm" else 1)
        add(name, result.predictions, result.settings, result.score)

    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        staged = Path(directory)
        for name, table in zip(PANELS, model_predictions):
            table.to_parquet(staged / f"{name}.parquet", index=False)
        write_json(staged / "receipt.json", {
            "panel": record(json_path),
            "protein_store": {m: record(a.store / m / "receipt.json") for m in ("esm2", "string")},
            "uniprot_crosswalk": record(a.proteins), "models": metadata, "seed": a.seed,
            "select_on": "source within-gene across-cell Spearman, with source RMSE tie-breaker",
            "device": torch.cuda.get_device_name() if a.device == "cuda" else "cpu",
            "files": {p.name: record(p) for p in sorted(staged.iterdir())},
        })
        staged.rename(a.output)


def score(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = a.panel / "receipt.json"
    registration = json.loads(manifest.read_text())
    frozen = a.predictions / "receipt.json"
    receipt = json.loads(frozen.read_text())
    if receipt["panel"] != record(manifest):
        raise ValueError("Chronos panel changed after prediction")
    rows = []
    for name in PANELS:
        observation = pd.read_parquet(checked(a.panel, f"labels_{name}.parquet", registration))
        prediction = pd.read_parquet(checked(a.predictions, f"{name}.parquet", receipt))
        joined = observation.merge(prediction, on=["model_id", "gene"], validate="1:1", how="left", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("Missing predictions for scored Chronos identities")
        for method in receipt["models"]:
            rows.append({"panel": name, "method": method, "models": joined.model_id.nunique(),
                         **regression_metrics(joined[method].to_numpy(), joined.chronos.to_numpy(),
                                              joined.gene.to_numpy())})
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        staged = Path(directory)
        table = staged / "chronos_metrics.tsv"
        pd.DataFrame(rows).to_csv(table, sep="\t", index=False, float_format="%.8g")
        write_json(staged / "receipt.json", {
            "panel": record(manifest), "predictions": record(frozen), "score_rows": len(rows),
            "constant_predictions": "Zero correlation in eligible variable-observation groups",
            "files": {table.name: record(table)},
        })
        staged.rename(a.output)


def arguments(score_mode=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    if score_mode:
        p.add_argument("--predictions", type=Path, required=True)
    else:
        p.add_argument("--store", type=Path, required=True)
        p.add_argument("--proteins", type=Path, required=True)
        p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        p.add_argument("--seed", type=int, default=41026)
    return p.parse_args()
