"""Transfer LINCS compound-level profiles to previously unseen HepaRG parents."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA

from src.goal_tracks.heparg_models import Kernel, aggregate, design_gain, hill_direction
from src.r14_cross_organ.heparg_eval import GAINS, center

PROFILES = ("pan_near10", "pan_anydose", "hepatic_near10")
COVERAGE_PROFILES = (*PROFILES, "hepg2_near10", "phh_near10")
MODEL_NAMES = ("p3_joint_chemistry", "p3_lincs", "p3_chemistry_lincs", "p3_lincs_direct")
ALPHAS = (0.1, 1.0, 10.0)
MIXES = (0.25, 0.5)
RANKS = (16, 32)
def strata(query):
    return {
        "rank7": query.dose_rank.eq(7).to_numpy(),
        "rank8": query.dose_rank.eq(8).to_numpy(),
        "ge10uM": query.dose_uM.ge(10).to_numpy(),
    }


def subsets(query, prior):
    return {
        "all": np.ones(len(query), dtype=bool),
        "pan_profiled": prior.covered(query.parent_key, "pan_anydose"),
        "pan_unprofiled": ~prior.covered(query.parent_key, "pan_anydose"),
        "HepG2_profiled": prior.covered(query.parent_key, "hepg2_near10"),
        "PHH_profiled": prior.covered(query.parent_key, "phh_near10"),
    }


class SignatureBank:
    """Immutable profile vectors after exact gene matching and row normalization."""

    def __init__(self, path: Path, reference: Path, common: pd.Index, symbol_conflicts):
        self.path = path
        with np.load(path, allow_pickle=False) as data:
            self.keys = pd.Index(data["keys"].astype(str))
            symbols = pd.Index(data["symbols"].astype(str))
            entrez = np.asarray(data["entrez"])
            if not (self.keys.is_unique and len(symbols) == len(entrez) and len(np.unique(entrez)) == len(entrez)):
                raise ValueError("Invalid LINCS parent or landmark axis")
            gene = pd.read_csv(reference, sep="\t", usecols=["gene_symbol", "ensembl_id"], keep_default_na=False)
            gene = gene[gene.ensembl_id.str.fullmatch(r"ENSG\d{11}") & gene.gene_symbol.ne("")].drop_duplicates()
            valid = gene.loc[~gene.gene_symbol.duplicated(False) & ~gene.ensembl_id.duplicated(False)]
            mapping = valid.set_index("gene_symbol").ensembl_id.reindex(symbols)
            conflicts = np.isin(entrez, symbol_conflicts)
            selected = mapping.isin(common).to_numpy() & ~conflicts
            self.mapped = pd.DataFrame({
                "entrez_id": entrez,
                "lincs_symbol": symbols,
                "ensembl_id": mapping.to_numpy(),
                "symbol_conflict_between_lincs_releases": conflicts,
                "both_temposeq_assays": selected,
            })
            self.coverage = {}
            self.values = {}
            for profile in COVERAGE_PROFILES:
                if profile not in data and profile not in PROFILES:
                    raise ValueError(f"Required tissue profile missing: {profile}")
                values = np.asarray(data[profile][:, selected], dtype=np.float32)
                mask = np.asarray(data[f"has_{profile}"], dtype=bool)
                if len(values) != len(self.keys) or mask.shape != (len(self.keys),):
                    raise ValueError("Inconsistent LINCS profile axis")
                subset = values[mask]
                if not np.isfinite(subset).all():
                    raise ValueError(f"Nonfinite LINCS profile: {profile}")
                subset -= subset.mean(axis=1, keepdims=True)
                norm = np.linalg.norm(subset, axis=1, keepdims=True)
                mask = mask.copy()
                mask[np.flatnonzero(mask)[norm.ravel() <= 1e-10]] = False
                values = np.zeros_like(values, dtype=np.float32)
                values[mask] = (subset / np.maximum(norm, 1e-10))[norm.ravel() > 1e-10]
                self.values[profile] = values
                self.coverage[profile] = mask
        if int(selected.sum()) < 33:
            raise ValueError("The mapped LINCS and TempO-Seq landmark intersection is unexpectedly small")

    def profile_indices(self, parents):
        indices = self.keys.get_indexer(parents)
        if (indices < 0).any():
            raise ValueError("A verified HepaRG parent is absent from the LINCS register")
        return indices

    def covered(self, parents, profile):
        return self.coverage[profile][self.profile_indices(parents)]


def kernel_center(gram, cross):
    """Center a PSD kernel on fitted parents and apply the fit center to held parents."""
    col = gram.mean(axis=0)
    grand = col.mean()
    return gram - col[:, None] - col[None, :] + grand, cross - col[None, :] - cross.mean(axis=1, keepdims=True) + grand


def ridge_paths(gram, cross, target):
    centered = target - target.mean(axis=0, keepdims=True)
    diag = np.diag_indices(len(gram))
    for alpha in ALPHAS:
        reg = gram.copy()
        reg[diag] += alpha
        weights = cho_solve(cho_factor(reg, check_finite=False), centered, check_finite=False)
        yield alpha, cross @ weights


def direct_paths(x, z, target):
    dims = [n for n in RANKS if n <= min(len(x) - 1, x.shape[1])]
    if not dims:
        return
    pca = PCA(n_components=max(dims), svd_solver="randomized", random_state=42)
    fitted = pca.fit_transform(x)
    held = pca.transform(z)
    for dim in dims:
        scale = np.maximum(pca.singular_values_[:dim] / np.sqrt(len(x) - 1), 1e-8)
        a, b = fitted[:, :dim] / scale, held[:, :dim] / scale
        a, b = a / np.sqrt(dim), b / np.sqrt(dim)
        shared = a.T @ a
        diag = np.diag_indices(len(shared))
        centered = target - target.mean(axis=0, keepdims=True)
        for alpha in ALPHAS:
            reg = shared.copy()
            reg[diag] += alpha
            coeff = cho_solve(cho_factor(reg, check_finite=False), a.T @ centered, check_finite=False)
            yield dim, alpha, b @ coeff


def checked_fallback(chem, alternative, mask):
    combined = chem.copy()
    combined[mask] = alternative
    return combined


def candidates(query, templates, geometry, prior, ec50):
    """Apply the same Hill scale and pool centering to every arm in a dose stratum."""
    dose = design_gain(query, ec50)[:, None]
    fit, target = hill_direction(*templates, ec50)
    test, order = np.unique(query.parent_key, return_inverse=True)
    fit_keys = fit.parent_key.to_numpy()
    if set(test) & set(fit_keys):
        raise ValueError("A held HepaRG parent entered the external-prior training fit")
    gs, cs = geometry.design(fit_keys, test)
    chem = dict(ridge_paths(gs, cs, target))

    def finish(values):
        return center(query, (dose * values[order]).astype(np.float32))

    for alpha, raw in chem.items():
        yield "p3_joint_chemistry", (alpha,), finish(raw)
    i, j = prior.profile_indices(fit_keys), prior.profile_indices(test)
    for profile in PROFILES:
        train_has, query_has = prior.coverage[profile][i], prior.coverage[profile][j]
        if train_has.sum() < 17:
            # Keep the same grid defined in both inner folds when a profile is sparse.
            for alpha, raw in chem.items():
                yield "p3_lincs", (profile, alpha), finish(raw)
                for mix in MIXES:
                    yield "p3_chemistry_lincs", (profile, mix, alpha), finish(raw)
                for dim in RANKS:
                    yield "p3_lincs_direct", (profile, dim, alpha), finish(raw)
            continue
        if not query_has.any():
            for alpha, raw in chem.items():
                yield "p3_lincs", (profile, alpha), finish(raw)
                for mix in MIXES:
                    yield "p3_chemistry_lincs", (profile, mix, alpha), finish(raw)
                for dim in RANKS:
                    yield "p3_lincs_direct", (profile, dim, alpha), finish(raw)
            continue
        prof = prior.values[profile]
        a, b = prof[i[train_has]], prof[j[query_has]]
        ls, lc = kernel_center(a @ a.T, b @ a.T)
        for alpha, raw in ridge_paths(ls, lc, target[train_has]):
            yield "p3_lincs", (profile, alpha), finish(checked_fallback(chem[alpha], raw, query_has))
        a_all, b_all = prof[i], prof[j[query_has]]
        ls, lc = kernel_center(a_all @ a_all.T, b_all @ a_all.T)
        for mix in MIXES:
            # The masked LINCS cosine kernel stays PSD. Unsupported held parents
            # receive the unchanged raw structural prediction before pool centering.
            for alpha, raw in ridge_paths((1 - mix) * gs + mix * ls, (1 - mix) * cs[query_has] + mix * lc, target):
                yield "p3_chemistry_lincs", (profile, mix, alpha), finish(checked_fallback(chem[alpha], raw, query_has))
        fitted_ranks = set()
        for dim, alpha, raw in direct_paths(a, b, target[train_has]):
            fitted_ranks.add(dim)
            yield "p3_lincs_direct", (profile, dim, alpha), finish(checked_fallback(chem[alpha], raw, query_has))
        for dim in set(RANKS) - fitted_ranks:
            for alpha, raw in chem.items():
                yield "p3_lincs_direct", (profile, dim, alpha), finish(raw)


def pooled_scores(keep, observed, predictor):
    obs = observed[keep].astype(np.float64)
    ss = float(np.einsum("ij,ij->", obs, obs))
    if ss <= 0:
        raise ValueError("No evaluable HepaRG variation")
    errors = {}
    for arm, settings, values in predictor:
        pred = values[keep].astype(np.float64)
        pp = np.einsum("ij,ij->", pred, pred)
        py = np.einsum("ij,ij->", pred, obs)
        errors[arm, settings] = (ss - 2 * py * GAINS + pp * GAINS**2) / ss
    return errors


def pick(folds):
    grid = set(folds[0])
    if grid != set(folds[1]):
        raise ValueError("An inner validation fold dropped a registered model candidate")
    chosen = {}
    receipts = []
    for arm in MODEL_NAMES:
        options = []
        for key in (x for x in grid if x[0] == arm):
            trial = np.asarray([fold[key] for fold in folds])
            average = trial.mean(axis=0)
            for i, (g, loss) in enumerate(zip(GAINS, average)):
                options.append((float(loss), float(g), key[1]))
                receipts.append({"arm": arm, "settings": list(key[1]), "gain": float(g),
                                 "inner_0_r2": float(1 - trial[0, i]), "inner_1_r2": float(1 - trial[1, i])})
        # Round objective tolerance to 1e-8, then favor stronger regularization
        # via lower calibrated amplitude before the deterministic setting label.
        loss, gain, settings = min(options, key=lambda a: (round(a[0], 8), a[1], str(a[2])))
        chosen[arm] = {"settings": list(settings), "gain": gain, "inner_r2": 1 - loss}
    return chosen, receipts


def dose_templates(train, y):
    return aggregate(train, y, ["dataset", "parent_key", "dose_rank", "dose_uM", "max_uM"])
