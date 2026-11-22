"""Apply read-only model releases and build cell-line reference domain projections."""

from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from .annotations import matrix, pathway_scores
from .cohort import digest, write_receipt


class VCRelease:
    """Adapter for the released vcell_liver functional contract."""

    def __init__(self, model_dir, upstream):
        # Read-only import of the release runtime. Bytecode stays disabled by the entry point.
        import sys

        upstream = Path(upstream).resolve()
        sys.path.insert(0, str(upstream / "src"))
        from vc_build.functional import load_model, transform_rna
        from vc_build.predict import _check_model_artifact

        self.transform = transform_rna
        contract = json.loads((model_dir / "contract.json").read_text())
        _check_model_artifact(model_dir, contract)
        self.model = load_model(model_dir)
        if contract["expression_scale"] != "log2(FPKM+1)":
            raise ValueError("A new native expression scale needs an explicit release adapter")
        self.contract = contract
        self.genes = list(self.model.bundle["rna"]["genes"])
        self.compounds = sorted(self.model.bundle["structures"])
        self.fit_cells = list(self.model.bundle["rna"]["fit_cells"])
        self.files = {p.name: digest(p) for p in sorted(model_dir.iterdir()) if p.is_file()}
        self.runtime = {p.name: digest(p) for p in sorted((upstream / "src/vc_build").glob("*.py"))}

    def predict(self, expression):
        rna = self.model.bundle["rna"]
        state, coverage = self.transform(rna, expression)
        distances = cdist(state, rna["coordinates"]).min(axis=1)
        below = coverage < rna["min_gene_fraction"]
        outside = distances > rna["distance_threshold"]
        candidates = [{"compound_id": compound} for compound in self.compounds]
        values = [
            np.full(len(candidates), np.nan) if missing else self.model.predict_many(row.dropna().to_dict(), candidates)
            for (_, row), missing in zip(expression.iterrows(), below, strict=True)
        ]
        predictions = pd.DataFrame(values, index=expression.index, columns=self.compounds)
        if self.contract["endpoint_direction"] == "lower":
            predictions = -predictions
        domain = pd.DataFrame(dict(gene_coverage=coverage, nearest_fit_distance=distances,
                                   insufficient_genes=below, ood=outside, abstain=outside | below),
                              index=expression.index)
        return predictions, domain


def load(path, upstream, adapter, cross):
    if adapter is None:
        if (path / "release.json").is_file():
            from .adapters import V0PlusRelease
            return V0PlusRelease(path, upstream, cross.parent)
        return VCRelease(path, upstream)
    module, factory = adapter.split(":", 1)
    return getattr(import_module(module), factory)(path, upstream)


def ranks(frame):
    values = rankdata(frame.to_numpy(), axis=1) / (frame.shape[1] + 1)
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def model_rank_bridge(query, training):
    """Map each tumor's observed within-sample ranks to the source quantile profile."""
    common = training.columns.intersection(query.columns)
    ref = training.loc[:, common].to_numpy()
    axis = np.linspace(0, 1, len(common))
    source = np.median(np.sort(ref, axis=1), axis=0)
    quantiles = ranks(query.loc[:, common])
    return pd.DataFrame(np.interp(quantiles, axis, source), columns=common, index=query.index)


def projection(train, query, family, seed, source_name):
    scaler = StandardScaler().fit(train)
    x = scaler.transform(train)
    y = scaler.transform(query)
    pca = PCA(n_components=min(16, x.shape[1], x.shape[0] - 1), svd_solver="full").fit(x)
    xp, yp = pca.transform(x), pca.transform(y)
    whitening = StandardScaler().fit(xp)
    xd, yd = whitening.transform(xp), whitening.transform(yp)
    fit_dist = cdist(xd, xd)
    fit_dist[family[:, None] == family[None, :]] = np.inf
    threshold = np.quantile(fit_dist.min(axis=1), .95)
    distance = cdist(yd, xd).min(axis=1)
    reducer = UMAP(n_neighbors=min(15, len(train) - 1), min_dist=.2, metric="euclidean",
                   random_state=seed, transform_seed=seed, n_jobs=1).fit(xd)
    xu, yu = reducer.embedding_, reducer.transform(yd)
    names = list(train.index) + list(query.index)
    coordinates = pd.DataFrame(np.c_[np.r_[xp[:, :2], yp[:, :2]], np.r_[xu, yu]], index=names,
                               columns=["PC1", "PC2", "UMAP1", "UMAP2"])
    coordinates["origin"] = [source_name] * len(train) + ["TCGA-LIHC"] * len(query)
    return coordinates, {
        "train_cells": len(train), "tumors": len(query), "input_features": train.shape[1],
        "reference_threshold": float(threshold), "in_domain": int((distance <= threshold).sum()),
        "distance_q05_q50_q95": np.quantile(distance, [.05, .5, .95]).tolist(),
        "pca_explained_first_two": pca.explained_variance_ratio_[:2].tolist(),
    }


def expression_domains(train, patient, tpm, family, genes, progeny, source_name):
    common = train.columns.intersection(patient.columns)
    train, patient, tpm = [matrix.loc[:, common] for matrix in (train, patient, tpm)]
    selected = sorted(set(genes) & set(common))
    modes = (
        ("native_fpkm", train[selected], patient[selected]),
        ("native_tpm_sensitivity", train[selected], tpm[selected]),
        ("within_sample_rank", ranks(train)[selected], ranks(patient)[selected]),
        ("fixed_progeny", pathway_scores(ranks(train), progeny)[0], pathway_scores(ranks(patient), progeny)[0]),
    )
    frames, metrics = [], {}
    for mode, source, target in modes:
        coords, metrics[mode] = projection(source, target, family, seed=19023, source_name=source_name)
        frames.append(coords.assign(preprocessing=mode))
    return pd.concat(frames), metrics, len(common)


def depmap_domain(fit, query, cross, model_registry, selected):
    meta = pd.read_csv(model_registry, usecols=["ModelID", "OncotreeCode", "Age", "PatientID"])
    age = pd.to_numeric(meta.Age, errors="coerce")
    hcc = meta.loc[meta.OncotreeCode.eq("HCC") & age.ge(18)].set_index("ModelID")
    rows = pd.read_parquet(cross / "rows.parquet", columns=["cell_id", "source_run"])
    ccle = pd.read_parquet(cross / "rna.parquet").loc[lambda data: data.index.isin(hcc.index)]
    common = fit.columns.intersection(query.columns).intersection(ccle.columns)
    chosen = sorted(set(selected) & set(common))
    if not len(ccle) or len(chosen) < 2:
        raise ValueError("The cross-source HCC expression axis has inadequate coverage")
    domains = {"LIMORE": ranks(fit[common])[chosen], "CCLE_HCC": ranks(ccle[common])[chosen],
               "TCGA-LIHC": ranks(query[common])[chosen]}
    for dataset in sorted(rows.source_run.unique()):
        sub = rows.loc[rows.source_run.eq(dataset), "cell_id"]
        domains[dataset] = domains["CCLE_HCC"].loc[lambda frame: frame.index.isin(sub)]
    coord = PCA(n_components=min(16, len(fit) - 1), svd_solver="full")
    scale = StandardScaler().fit(domains["LIMORE"])
    origin = coord.fit_transform(scale.transform(domains["LIMORE"]))
    return {name: {"profiles": len(frame), "common_genes": len(common),
                   "selected_genes": len(chosen),
                   "median_nearest_limore": float(np.median(cdist(
                       coord.transform(scale.transform(frame)), origin).min(axis=1))) if len(frame) else None}
            for name, frame in domains.items()}


def freeze(args):
    import sys

    sys.dont_write_bytecode = True
    participants, patient = matrix(args.cohort)
    annotation = pd.read_csv(args.cohort / "genes.tsv", sep="\t")
    raw_tpm = np.load(args.cohort / "tpm_unstranded.npy", allow_pickle=False)
    axis = pd.Index(annotation.gene_name).get_indexer_for(patient.columns)
    tpm = pd.DataFrame(np.log2(1 + raw_tpm[:, axis]), index=patient.index, columns=patient.columns)
    model = load(args.model, args.upstream, args.adapter, args.cross)
    reference_rows = pd.read_parquet(args.limore / "rows.parquet", columns=[
        "cell_id", "cell_group", "compound_id", "endpoint_value"]).rename(columns={"cell_group": "lineage_group"})
    fit_groups = reference_rows[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id")
    limore = pd.read_parquet(args.limore / "rna.parquet").loc[fit_groups.index]
    if model.contract["expression_scale"] == "log2(TPM+1)":
        source, families, scale_name, origin = model.source, model.groups, "native_tpm", "V0+ fit sources"
    else:
        source = limore.loc[model.fit_cells]
        families = fit_groups.loc[source.index, "lineage_group"].to_numpy()
        scale_name, origin = "native_fpkm", "LIMORE"
    native = tpm if scale_name == "native_tpm" else patient
    rank_bridge = model_rank_bridge(native, source)
    preds, domains = {}, {}
    for mode, query in ((scale_name, native), ("source_quantile_rank", rank_bridge)):
        preds["candidate", mode], domains[mode] = model.predict(query.reindex(columns=model.genes))
    releases = {"candidate": {"files": model.files, "runtime": model.runtime, "contract": model.contract}}
    if args.baseline:
        baseline = load(args.baseline, args.upstream, args.adapter, args.cross)
        if len(set(model.compounds) & set(baseline.compounds)) < 5:
            raise ValueError("The releases have too few shared compounds for paired comparisons")
        releases["baseline"] = {"files": baseline.files, "runtime": baseline.runtime, "contract": baseline.contract}
        baseline_is_tpm = baseline.contract["expression_scale"] == "log2(TPM+1)"
        baseline_native = tpm if baseline_is_tpm else patient
        baseline_source = baseline.source if baseline_is_tpm else limore.loc[baseline.fit_cells]
        baseline_bridge = model_rank_bridge(baseline_native, baseline_source)
        baseline_scale = "native_tpm" if baseline_is_tpm else "native_fpkm"
        for mode, query in ((baseline_scale, baseline_native), ("source_quantile_rank", baseline_bridge)):
            values, _ = baseline.predict(query.reindex(columns=baseline.genes))
            preds["baseline", mode] = values.loc[:, values.columns.intersection(model.compounds)]
    anchor = (model.anchor if hasattr(model, "anchor") else reference_rows.loc[
        reference_rows.cell_id.isin(model.fit_cells)].groupby("compound_id").endpoint_value.mean())
    direction = 1 if model.contract["endpoint_direction"] == "higher" else -1
    anchor = direction * anchor.reindex(model.compounds)
    preds["drug_mean", scale_name] = pd.DataFrame(np.tile(anchor, (len(patient), 1)), index=patient.index,
                                                       columns=model.compounds)
    focus = getattr(model, "focus_genes", model.genes)
    points, shifts, common = expression_domains(source, patient, tpm, families, focus, args.raw / "progeny.tsv", origin)
    registry = args.limore / "compound_registry.tsv"
    roster = pd.read_csv(registry, sep="\t").drop_duplicates("compound_id").set_index("compound_id")
    args.out.mkdir(parents=True, exist_ok=False)
    predict = pd.concat([frame.rename_axis(columns="compound_id").stack(future_stack=True).rename("value")
                         .reset_index().assign(arm=arm, preprocessing=mode)
                         for (arm, mode), frame in preds.items()], ignore_index=True)
    predict.to_parquet(args.out / "predictions.parquet", index=False)
    pd.concat([frame.assign(preprocessing=mode) for mode, frame in domains.items()]).to_parquet(args.out / "domain.parquet")
    points.to_parquet(args.out / "coordinates.parquet")
    roster.reindex(model.compounds)[["source_compound_label", "drug_target_source_labels"]].to_csv(
        args.out / "compounds.tsv", sep="\t")
    study = depmap_domain(limore if scale_name == "native_tpm" else source, tpm, args.cross, args.depmap, focus)
    expected = {"contract": releases, "tumors": len(participants), "compound_pool": len(model.compounds),
                "native_gene_overlap": len(set(patient) & set(model.genes)),
                "native_model_genes": len(model.genes), "common_v0_limore_tcga_protein_coding_genes": common,
                "projection_domains": shifts, "depmap_rank_domains": study,
                "sources": {name: digest(path) for name, path in {
                    "cohort": args.cohort / "receipt.json", "compound_registry": registry, "depmap_model": args.depmap,
                    "limore_rna": args.limore / "rna.parquet", "limore_rows": args.limore / "rows.parquet",
                    "cross_source": args.cross.parent / "receipt.json", "cross_rna": args.cross / "rna.parquet",
                    "cross_rows": args.cross / "rows.parquet", "progeny": args.raw / "progeny.tsv",
                    "forecast_code": Path(__file__)}.items()},
                "outputs": {p.name: digest(p) for p in sorted(args.out.iterdir())}}
    write_receipt(args.out / "freeze.json", expected)
