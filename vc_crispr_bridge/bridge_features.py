"""Independent molecular assays used by the LIMORE genetic bridge."""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from .chronos_prep import entrez_annotations, entrez_axis
from .design import summarize_assay, validate_source
from .gene_features import unambiguous_symbol_map, verified_vectors
from .perturbseq import record
from .predict import PANELS, checked


def read_dependency(effect_file, hgnc, targets, model_file, cell_registry):
    genes = entrez_annotations(hgnc)
    safe_symbols = unambiguous_symbol_map(hgnc)
    genes = genes[genes.ensembl_gene_id.isin(targets) & genes.ensembl_gene_id.isin(safe_symbols.index)]
    genes = genes[~genes.entrez_id.duplicated(keep=False)].set_index("entrez_id").ensembl_gene_id
    header = pd.read_csv(effect_file, nrows=0).columns
    column_index = entrez_axis(header)
    by_column = {column_index[entrez]: gene for entrez, gene in genes.items() if entrez in column_index}
    if not by_column:
        raise ValueError("No registered protein targets overlap DepMap Chronos")
    values = pd.read_csv(effect_file, index_col=0, usecols=[header[0], *by_column],
                         dtype={field: np.float32 for field in by_column})
    values = values.rename(columns=by_column)
    models = pd.read_csv(model_file, usecols=["ModelID", "PatientID", "OncotreeLineage"], dtype=str).set_index("ModelID")
    liver_donors = set(models.loc[models.OncotreeLineage.eq("Liver"), "PatientID"])
    controls = models[~models.PatientID.isin(liver_donors) & models.OncotreeLineage.notna()].index
    non_liver_prior = -values.reindex(controls).median(axis=0)
    cells = pd.read_csv(cell_registry, sep="\t",
                        usecols=["model_id", "depmap_model_id", "included"], dtype=str)
    cells = cells[cells.included.eq("True")].set_index("model_id", verify_integrity=True)
    measured = -values.reindex(cells.depmap_model_id)
    measured.index = cells.index
    return measured, non_liver_prior


def read_signatures(hepg2, predictions, gene_panels):
    manifest = json.loads((gene_panels / "receipt.json").read_text())
    axes = pd.Index(pd.read_csv(checked(gene_panels, "genes.tsv", manifest), sep="\t").gene_id)
    meta, values, _ = summarize_assay(validate_source(hepg2), axes)
    observed = pd.DataFrame(values, index=pd.Index(meta.target_ensembl), columns=axes)
    parts = []
    receipt = json.loads((predictions / "receipt.json").read_text())
    if receipt["panel"] != record(gene_panels / "receipt.json"):
        raise ValueError("HepG2 predicted signatures come from a different registered assay")
    for name in PANELS:
        with np.load(checked(predictions, f"{name}.npz", receipt), allow_pickle=False) as frozen:
            key = [name.split(":")[-1] for name in frozen["id"]]
            parts.append(pd.DataFrame(frozen["conditional_mlp"], columns=axes, index=key))
    predicted = pd.concat(parts)
    if not predicted.index.is_unique or set(predicted.index) != set(observed.index):
        raise ValueError("HepG2 signature forecast axes differ from registered profiles")
    return observed, predicted


def scaled_rows(values):
    center = values - values.mean(axis=1, keepdims=True)
    return center / np.maximum(np.linalg.norm(center, axis=1, keepdims=True), 1e-8)


def signature_table(values, mapping, fit_drugs, drugs, cells, symbol_map):
    pooled, fractions = [], []
    for drug in drugs:
        query = sorted(mapping[drug])
        candidate = values.reindex(query).dropna()
        pooled.append(candidate.mean(axis=0).to_numpy(np.float32) if len(candidate)
                      else np.zeros(values.shape[1], np.float32))
        fractions.append(len(candidate) / len(query))
    effects = pd.DataFrame(np.stack(pooled), index=drugs, columns=values.columns)
    projected = PCA(n_components=min(24, len(fit_drugs) - 1, len(effects.columns)), random_state=41124,
                    svd_solver="randomized").fit(effects.loc[fit_drugs]).transform(effects)
    drug_features = pd.DataFrame(np.column_stack((projected, fractions)), index=drugs)
    available = symbol_map.reindex(effects.columns).dropna()
    available = available[available.isin(cells.columns)]
    if len(available) < 100:
        raise ValueError("Too few common assay expression genes for genetic alignment")
    expression = cells[available.to_numpy()].to_numpy(np.float32)
    common_effect = effects[available.index].to_numpy(np.float32)
    alignment = scaled_rows(expression) @ scaled_rows(common_effect).T
    return drug_features, pd.DataFrame(alignment, index=cells.index, columns=drugs)


def dependence_summary(frame, mapping, measured, prior):
    values = {}
    for drug, group in frame.groupby("drug_id", sort=False):
        target_genes = sorted(mapping[drug])
        rows = group.cell_id
        raw = measured.reindex(index=rows, columns=target_genes).to_numpy()
        has = np.isfinite(raw)
        fallback = prior.reindex(target_genes).fillna(0).to_numpy()
        loss = np.where(has, raw, fallback)
        values[drug] = pd.DataFrame({
            "chronos_mean": loss.mean(axis=1), "chronos_max": loss.max(axis=1),
            "chronos_min": loss.min(axis=1), "chronos_spread": loss.std(axis=1),
            "chronos_measured_fraction": has.mean(axis=1),
            "chronos_missing": (~has.any(axis=1)).astype(float),
        }, index=group.index)
    return pd.concat(values.values()).loc[frame.index]


def pair_feature(records, per_drug, alignment):
    indices = alignment.index.get_indexer(records.cell_id)
    targets = alignment.columns.get_indexer(records.drug_id)
    if (indices < 0).any() or (targets < 0).any():
        raise ValueError("Context or drug order differs from the signature registry")
    rank = alignment.to_numpy()[indices, targets]
    return np.column_stack((per_drug.loc[records.drug_id].to_numpy(), rank))


def context_features(basal, train, roles):
    core = basal.to_numpy(dtype=np.float32)
    ranks = rankdata(core, axis=1, method="average").astype(np.float32) / core.shape[1]
    fitting_positions = basal.index.get_indexer(train.cell_id.unique())
    if (fitting_positions < 0).any():
        raise ValueError("No registered basal context for training row")
    variance = ranks[fitting_positions].var(axis=0)
    selected = np.lexsort((basal.columns, -variance))[:min(384, len(basal.columns))]
    reduced = pd.DataFrame(ranks[:, selected], index=basal.index)
    return [reduced.loc[role.cell_id].to_numpy(np.float32) for role in roles]


class BridgeFeatures:
    def __init__(self, args, targets):
        self.ecfp = verified_vectors(args.chemistry / "ecfp4")
        self.molf = verified_vectors(args.chemistry / "molformer")
        self.context = pd.read_parquet(args.context)
        if not self.context.index.is_unique or not self.context.columns.is_unique:
            raise ValueError("Ambiguous cell or gene identities in LIMORE RNA context")
        self.targets = targets.groupby("drug_id").gene.agg(set).to_dict()
        stable = target_union(self.targets)
        self.symbols = unambiguous_symbol_map(args.hgnc)
        self.dependency, self.prior = read_dependency(args.chronos, args.hgnc, stable,
                                                      args.depmap_models, args.cells)
        self.observed, self.predicted = read_signatures(args.hepg2, args.hepg2_predictions, args.gene_panels)

    def arrays(self, roles):
        sources = roles[0]
        fit_drugs = sorted(set(sources.drug_id))
        all_drugs = sorted(set().union(*(set(role.drug_id) for role in roles)))
        per_drug_o, rank_o = signature_table(self.observed, self.targets, fit_drugs,
                                             all_drugs, self.context, self.symbols)
        per_drug_p, rank_p = signature_table(self.predicted, self.targets, fit_drugs,
                                             all_drugs, self.context, self.symbols)
        dependency = [dependence_summary(role, self.targets, self.dependency, self.prior)
                      for role in roles]
        return {
            "ecfp4": [self.ecfp.loc[r.drug_id].to_numpy(np.float32) for r in roles],
            "molformer": [self.molf.loc[r.drug_id].to_numpy(np.float32) for r in roles],
            "basal": context_features(self.context, sources, roles),
            "dep": [m.to_numpy(np.float32) for m in dependency],
            "observed": [pair_feature(role, per_drug_o, rank_o) for role in roles],
            "predicted": [pair_feature(role, per_drug_p, rank_p) for role in roles],
        }, [m.chronos_measured_fraction.to_numpy() > 0 for m in dependency]


def target_union(mapping):
    return set().union(*mapping.values())
