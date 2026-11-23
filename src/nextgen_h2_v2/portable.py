"""Offline SMILES inference for the frozen drug-cold kernel on registered RNA cells."""
import argparse
import json
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.nn.functional import normalize

from nextgen_h2.assets import KEY, LINCS, canonical_chemistry
from nextgen_h2.replay import network_deny
from vc_build.external import _chemistry
from vc_build.io import sha256

from .kernel import Design, View


@dataclass
class MeasurementProjection:
    columns: list
    keep: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    pca: PCA | None
    whiten: bool

    @classmethod
    def fit(cls, frame, spec):
        raw = frame.to_numpy(dtype=np.float32).astype(np.float64)
        supports = np.isfinite(raw).mean(axis=1) >= .75
        values = raw[supports]
        accepted = np.isfinite(values).all(axis=0)
        mask = accepted.copy()
        complete = values[:, accepted]
        good = complete.std(axis=0) > 1e-6
        mask[accepted] = good
        train = values[:, mask]
        mean, spread = train.mean(axis=0), train.std(axis=0)
        pca = None
        if spec.transform in ("pca", "white"):
            normalized = ((train - mean) / spread).clip(-12, 12).astype(np.float32)
            count = min(spec.dimension, normalized.shape[1], normalized.shape[0] - 1)
            pca = PCA(n_components=count, svd_solver="randomized", random_state=9171,
                      iterated_power=4).fit(normalized)
        return cls(list(frame.columns), mask, mean, spread, pca, spec.transform == "white")

    def apply(self, values):
        x = np.asarray(values, np.float32).astype(np.float64)[:, self.keep]
        x = np.where(np.isfinite(x), x, self.center)
        result = ((x - self.center) / self.scale).clip(-12, 12).astype(np.float32)
        if self.pca is not None:
            result = self.pca.transform(result)
            if self.whiten:
                result /= np.sqrt(self.pca.explained_variance_ + .10)
        return normalize(torch.as_tensor(result), dim=1)


def cross_rbf(x, y, width):
    dist = (x.square().sum(1)[:, None] + y.square().sum(1)[None, :] - 2 * x @ y.T).clamp_min(0.)
    return torch.exp(-dist / width)


def view_cross(x, y, kind):
    if kind.startswith("cos"):
        v = x @ y.T
        self_score = x.square().sum(1)
        if kind != "cos":
            exponent = int(kind[-1])
            v = v.pow(exponent)
            self_score = self_score.pow(exponent)
        return v, self_score
    return cross_rbf(x, y, {"rbf_half": .5, "rbf": 1., "rbf_wide": 2.}[kind]), torch.ones(len(x))


@dataclass
class KernelRow:
    params: Design
    coef: torch.Tensor
    fit_tx: torch.Tensor
    fit_diag: torch.Tensor
    fit_lincs: torch.Tensor | None
    lincs_seen: torch.Tensor | None
    anchor_parameters: torch.Tensor | None

    def predict(self, chemistry, raw_lincs, cell_cross, matched, base_y):
        spec = self.params
        tani, target, props, id_gram, bit_present = chemistry
        gram = .1 + spec.tanimoto * tani + spec.target * target[0] + spec.properties * props + spec.identity * id_gram
        diag = .1 + spec.tanimoto * bit_present + spec.target * target[1] + spec.properties + spec.identity
        if raw_lincs is not None and self.fit_lincs is not None:
            view, self_diag = view_cross(raw_lincs, self.fit_lincs, spec.lincs.kernel)
            gram = gram + (spec.lincs.weight * view + spec.lincs.mask) * self.lincs_seen[None, :]
            diag = diag + spec.lincs.weight * self_diag + spec.lincs.mask
        if spec.normalization == "norm":
            gram = diag.rsqrt()[:, None] * gram * self.fit_diag.rsqrt()[None, :]
        ix = tani.argsort(dim=1, descending=True, stable=True)[:, :min(5, len(base_y))]
        taper = tani.gather(1, ix).pow(4)
        anchors = torch.where(taper.sum(1) > 0, (taper * base_y[ix]).sum(1) / taper.sum(1).clamp_min(1e-12),
                              base_y.mean())
        if spec.intercept == "krr":
            anchors = base_y.mean() + gram @ self.anchor_parameters
        elif spec.intercept == "neighbors":
            nearest, ix = torch.topk(gram.clip(min=0), min(int(spec.anchor_penalty), len(base_y)), dim=1)
            taper = nearest.pow(4)
            anchors = torch.where(taper.sum(1) > 0, (taper * base_y[ix]).sum(1) / taper.sum(1).clamp_min(1e-12),
                                  anchors)
        if matched >= 0:
            anchors[:] = base_y[matched]
        return anchors + (cell_cross @ self.coef) @ gram.T


@dataclass
class Portable:
    rna_basis: object
    rna: pd.DataFrame
    known_drugs: pd.Index
    catalogue: dict
    vocabulary: list
    bits: torch.Tensor
    props: torch.Tensor
    prop_center: torch.Tensor
    prop_scale: torch.Tensor
    anchor: torch.Tensor
    reference: KernelRow
    assay: KernelRow
    projector: MeasurementProjection
    gene_columns: list

    def features(self, smiles):
        parent, key, *_ = canonical_chemistry(smiles)
        label = pd.DataFrame({KEY: [key], "isomeric_smiles": [parent]})
        vector = torch.as_tensor(_chemistry(label, fp_size=2048).to_numpy(np.float32))
        bs, props = vector[:, :2048], vector[:, 2048:]
        props = ((props - self.prop_center) / self.prop_scale).clamp(-5, 5)
        dot = bs @ self.bits.T
        tani = dot / (bs.sum(1)[:, None] + self.bits.sum(1)[None, :] - dot).clamp_min(1)
        terms = set(self.catalogue.get(key, ()))
        tv = normalize(torch.tensor([[float(x in terms) for x in self.vocabulary]]), dim=1)
        cross = cross_rbf(.5 * props, .5 * self.props, 2.)
        same = torch.as_tensor(self.known_drugs == key, dtype=torch.float32)[None, :]
        return key, (tani, (tv @ self.reference.fit_tx.T, tv.square().sum(1)), cross, same, bs.sum(1) > 0)

    def predict(self, smiles, cells, view=None):
        key, chem = self.features(smiles)
        if not set(cells) <= set(self.rna.index):
            raise ValueError("A requested cell ID has no registered basal RNA")
        kernel = self.rna_basis.cross(self.rna.loc[cells])["full"]
        match = self.known_drugs.get_indexer([key])[0]
        if view is None:
            return key, self.reference.predict(chem, None, kernel, match, self.anchor).flatten().numpy()
        known = self.projector.apply(view)
        return key, self.assay.predict(chem, known, kernel, match, self.anchor).flatten().numpy()

    def signature(self, path):
        table = pd.read_csv(path, sep="\t")
        if set(table.columns) != {"feature", "value"} or not table.feature.is_unique:
            raise ValueError("LINCS TSV must contain unique feature and value columns")
        series = table.set_index("feature").value
        names = self.gene_columns if set(self.gene_columns) <= set(series.index) else self.projector.columns
        frame = pd.to_numeric(series.reindex(names), errors="coerce").to_numpy(np.float32)
        if np.isfinite(frame).mean() < .75:
            raise ValueError(f"Fewer than 75% of the required LINCS axes for {self.assay.params.lincs.recipe}")
        return frame[None, :]


def create_kit(engine, params, seed):
    params = replace(params, drug_views=(), cell_views=())
    chemical = Design(lincs=View("lincs", weight=0., mask=0.))
    registry = engine.fit_drugs
    reference = engine.data.features.loc[registry].to_numpy(np.float32)
    tensors = torch.as_tensor(reference, device=engine.device)
    bits, props = tensors[:, :2048], tensors[:, 2048:]
    pm, ps = props.mean(dim=0), props.std(dim=0, correction=0).clamp_min(.05)
    genes = sorted(set().union(*(engine.data.targets.get(x, ()) for x in registry)))
    target = torch.as_tensor([[float(x in set(engine.data.targets.get(drug, ()))) for x in genes]
                              for drug in registry])
    target = normalize(target, dim=1)
    anchors = torch.as_tensor(engine.anchor.reindex(registry).to_numpy(np.float32))
    values = engine.view_data(params.lincs).reindex(registry)
    projector = MeasurementProjection.fit(values, params.lincs)
    _, availability = engine.prepare(params.lincs, seed)
    key = ("drug", params.lincs.name, params.lincs.recipe, params.lincs.transform, params.lincs.dimension)
    fit_values = engine._features[key][0][:engine.stop].cpu()

    def fitted(policy, support=None):
        _, state = engine.solve(policy, seed, retain=True)
        expanded, _ = engine.drug_geometry(replace(policy, normalization="sum"), seed)
        diagonal = expanded.diagonal()[:engine.stop].cpu()
        adjust = None
        if policy.intercept == "krr":
            adjust = torch.linalg.solve(state["dfit"] + policy.anchor_penalty * torch.eye(len(registry)),
                                        anchors - anchors.mean())
        return KernelRow(policy, state["coef"], target, diagonal, support,
                         availability[:engine.stop].cpu() if support is not None else None, adjust)

    along = list(projector.columns)
    if params.lincs.recipe != "v1":
        axis, representation = params.lincs.recipe.split("_", 1)
        spec = pd.read_csv(LINCS / "geneinfo_beta.txt", sep="\t")
        admitted = {"landmark": spec.feature_space.eq("landmark"),
                "bing": spec.feature_space.isin(["landmark", "best inferred"]),
                "full": np.ones(len(spec), dtype=bool)}[axis]
        identifiers = spec.loc[admitted, "gene_symbol"].to_list()
        if len(identifiers) == len(along) and len(set(identifiers)) == len(identifiers):
            along = identifiers
    return Portable(engine.basis.cpu(), engine.data.expression(engine.all_cells.unique(), engine.medians), registry,
                    engine.data.targets, genes, bits.cpu(), ((props - pm) / ps).clamp(-5, 5).cpu(),
                    pm.cpu(), ps.cpu(), anchors, fitted(chemical), fitted(params, fit_values), projector, along)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--cell-id", nargs="+", required=True)
    parser.add_argument("--lincs-tsv", type=Path)
    args = parser.parse_args()
    sys.addaudithook(network_deny)
    if sha256(args.package) != args.sha256:
        raise ValueError("Portable model checksum mismatch")
    with args.package.open("rb") as handle:
        model = pickle.load(handle)
    supplied = model.signature(args.lincs_tsv) if args.lincs_tsv else None
    parent, score = model.predict(args.smiles, args.cell_id, supplied)
    print(json.dumps({"parent": parent, "mode": "Assay-assisted" if supplied is not None else "Structure-only",
                      "scope": "H2-R1 on registered basal RNA",
                      "predictions": dict(zip(args.cell_id, [float(x) for x in score]))}))


if __name__ == "__main__":
    main()
