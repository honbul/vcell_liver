"""Source-only features and scaffold-cross-fitted residual labels."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from nextgen_h2.assets import KEY, fold_ids
from nextgen_h2.run import META
from vc_build.external import SOURCE, native_sensitivity
from vc_build.structured_kernel import ranked_expression

from .kernel import Engine


def kernel_oof(engine, params, seed):
    groups = engine.reg.loc[engine.fit_drugs, "scaffold_block"]
    assignment = fold_ids(groups)
    frames = []
    for f in range(3):
        held = list(assignment.index[assignment.eq(f)])
        test = engine.fit[engine.fit[KEY].isin(held)]
        train = engine.fit[~engine.fit[KEY].isin(held)]
        sub = Engine(engine.data, engine.assets, train, test[META], engine.reg)
        prediction = sub.extract_rows(sub.solve(params, seed + f))
        by_assay = pd.DataFrame(sub.calibration).set_index(SOURCE)
        offsets = test.join(by_assay[["slope", "intercept"]], on=SOURCE, validate="many_to_one")
        native = native_sensitivity(test)
        y = native * offsets.slope + offsets.intercept
        frames.append(test[META].assign(oof=prediction, aligned=y, inner_crossfit=f))
    result = pd.concat(frames, ignore_index=True).set_index("row_id").reindex(engine.fit.row_id).reset_index()
    if (len(result) != len(engine.fit) or not result.row_id.is_unique or result.oof.isna().any()):
        raise ValueError("Scaffold-cross-fitted prediction keys differ from the fit cohort")
    return result


@dataclass
class Projector:
    admitted: np.ndarray
    location: np.ndarray
    spread: np.ndarray
    model: PCA | None

    @classmethod
    def fit(cls, frame, dim):
        admitted = np.isfinite(frame).mean(axis=0) > .75
        values = frame[:, admitted].astype(np.float32)
        center = np.nanmean(values, axis=0)
        spread = np.nanstd(values, axis=0).clip(.05)
        x = np.nan_to_num((values - center) / spread).clip(-8, 8)
        if dim and x.shape[1] > dim:
            model = PCA(n_components=min(dim, len(x) - 1), svd_solver="randomized", random_state=1947).fit(x)
        else:
            model = None
        return cls(admitted, center, spread, model)

    def apply(self, data):
        x = data[:, self.admitted].astype(np.float32)
        x = np.nan_to_num((x - self.location) / self.spread).clip(-8, 8)
        return self.model.transform(x).astype(np.float32) if self.model is not None else x


def transform(a, n, dim=0, *, floor=.75):
    ok = np.isfinite(a).mean(axis=1) >= floor
    fit = a[:n][ok[:n]]
    if len(fit) < 5:
        return np.zeros((len(a), dim or 1), np.float32), ok & False, None
    projector = Projector.fit(fit, dim)
    value = projector.apply(a)
    value[~ok] = 0.
    return value, ok, projector


def targets_for(data, compounds, genes):
    position = {g: i for i, g in enumerate(genes)}
    mapping = np.zeros((len(compounds), len(position)), np.float32)
    for i, key in enumerate(compounds):
        mapping[i, [position[g] for g in data.targets.get(key, ()) if g in position]] = 1.
    return mapping


def target_mean(values, active):
    weights = active / active.sum(axis=1, keepdims=True).clip(1)
    return np.nan_to_num(values) @ weights.T


def basal_paths(engine, base_ranks):
    data = engine.data
    projection = data.pathway
    pos = data.rna.columns.get_indexer(projection.index)
    common = pos >= 0
    x = base_ranks[:, pos[common]] @ projection.to_numpy(dtype=np.float32)[common]
    drug = data.profiles["paths_modz24"].reindex(engine.drugs).to_numpy(dtype=np.float32)
    c, _, _ = transform(x, engine.nfit_cells)
    d, mask, _ = transform(drug, engine.stop)
    return c, d, mask


class ResidualFeatures:
    def __init__(self, engine, params):
        data = engine.data
        self.cells = engine.all_cells
        self.drugs = engine.drugs
        self.cn, self.dn = engine.nfit_cells, engine.stop
        self.fit_cells = engine.fit_cells
        self.query_cells = engine.query_cells
        self.fit_drugs = engine.fit_drugs
        self.query_drugs = engine.query_drugs
        rna = data.expression(self.cells, engine.medians)
        ranked = ranked_expression(rna)
        seq, _, _ = transform(ranked, self.cn, 40)
        path, drug_path, drug_seen = basal_paths(engine, ranked)
        lincs = data.profiles["modz24"].reindex(engine.drugs).to_numpy(np.float32)
        ln, _, _ = transform(lincs, self.dn, 40)
        chemistry = engine.drug.explicit.cpu().numpy()
        ch, _, _ = transform(chemistry[:, :2052], self.dn, 48)
        tg, _, _ = transform(chemistry[:, 2052:], self.dn, 24)
        parts = []
        for view in ("mut", "cnv", "fusion", "sig"):
            frame = engine.assets.read("omics", view)
            x = frame.reindex(engine.all_cells).to_numpy(dtype=np.float32)
            z, seen, _ = transform(x, self.cn, 12)
            parts.append(np.c_[z, seen].astype(np.float32))
        omics = np.concatenate(parts, axis=1)
        self.cell = (seq, path, omics)
        self.drug = (ch, tg, ln)
        self.drug_seen = drug_seen
        for view in params.drug_views:
            frame = engine.view_data(view).reindex(engine.drugs)
            self.drug_seen = self.drug_seen | frame.notna().mean(axis=1).ge(.75).to_numpy()
        self.cell_path = path
        self.drug_path = drug_path
        self.target_pairs = self.target_features(engine, ranked)

    def target_features(self, engine, ranked):
        data = engine.data
        target_genes = sorted(set().union(*(data.targets.get(drug, ()) for drug in self.fit_drugs)))
        m = targets_for(data, self.drugs, target_genes)
        ix = data.rna.columns.get_indexer(target_genes)
        basis = np.zeros((len(self.cells), len(ix)), np.float32)
        common = ix >= 0
        basis[:, common] = ranked[:, ix[common]]
        effects = [target_mean(basis, m), target_mean(np.abs(basis), m)]
        for view in ("mut", "cnv"):
            tab = engine.assets.read("omics", view).reindex(index=self.cells, columns=target_genes)
            available = np.isfinite(tab.to_numpy())
            raw = np.nan_to_num(tab.to_numpy(np.float32))
            if view == "cnv":
                relative = np.log2(np.maximum(raw, .05))
                data_scale = np.where(available, relative, 0)
                effects.append(target_mean(np.abs(data_scale), m))
            else:
                data_scale = (raw > 0).astype(np.float32)
            effects.extend([target_mean(data_scale, m),
                            target_mean(available.astype(np.float32), m)])
        return tuple(effects)

    def coordinates(self, panel, fitting):
        if fitting:
            ci = self.fit_cells.get_indexer(panel.cell_id)
            di = self.fit_drugs.get_indexer(panel[KEY])
        else:
            ci = self.query_cells.get_indexer(panel.cell_id) + self.cn
            di = self.query_drugs.get_indexer(panel[KEY]) + self.dn
        if (ci < (0 if fitting else self.cn)).any() or (di < (0 if fitting else self.dn)).any():
            raise ValueError("Residual row contains unregistered coordinate")
        return ci, di

    def joint(self, ci, di, base):
        direct = [arr[ci, di] for arr in self.target_pairs]
        linked = self.cell_path[ci] * self.drug_path[di]
        return np.column_stack([base, self.drug_seen[di], linked, linked.mean(axis=1), *direct]).astype(np.float32)

    def tabular(self, panel, base, fitting):
        ci, di = self.coordinates(panel, fitting)
        return np.concatenate([self.joint(ci, di, base), *(x[ci] for x in self.cell),
                               *(x[di] for x in self.drug)], axis=1)

    def roles(self, panel, base, fitting):
        ci, di = self.coordinates(panel, fitting)
        return ci, di, self.joint(ci, di, base), self.drug_seen[di]
