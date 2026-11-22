"""Read the published V0+ package schema without mutating the upstream release."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .cohort import digest


class V0PlusRelease:
    """Use the package's frozen design, residual multiplier and compound registry."""

    def __init__(self, directory, upstream, train_bundle):
        path = Path(directory)
        sys.path.insert(0, str(Path(upstream).resolve() / "src"))
        from vc_build.v0plus_cv import panel_targets, training_inputs

        card = json.loads((path / "release.json").read_text())
        if (card["version"], card["schema_version"]) != ("V0+", 1):
            raise ValueError("Unsupported V0+ release")
        record = card["models"][card["default_model"]]
        target = Path(record["path"])
        if target.name != str(target):
            raise ValueError("V0+ checkpoints must be within the release directory")
        checkpoint = path / target
        registry = path / "compounds.tsv"
        if digest(checkpoint) != record["sha256"] or digest(registry) != card["drug_registry_sha256"]:
            raise ValueError("V0+ checkpoint or registry differs from the release")
        self.model = joblib.load(checkpoint)
        self.features = self.model.molecular.features if hasattr(self.model, "molecular") else self.model.features
        self.genes = list(self.features.genes)
        self.focus_genes = [self.genes[i] for i in self.features.indices]
        self.panel = pd.read_csv(registry, sep="\t", dtype=str).set_index("compound_id", verify_integrity=True).sort_index()
        self.compounds = self.panel.index.tolist()
        self.multiplier = float(record["multiplier"])
        rows, rna, _ = training_inputs(train_bundle)
        if card["source_rows"] != len(rows):
            raise ValueError("V0+ fit bundle and published source row count disagree")
        self.fit_cells = sorted(rows.cell_id.unique())
        self.source = rna.loc[self.fit_cells]
        self.groups = (rows[["cell_id", "lineage_group"]].drop_duplicates()
                       .set_index("cell_id", verify_integrity=True).loc[self.fit_cells].lineage_group.to_numpy())
        _, anchor, _, _ = panel_targets(rows)
        self.anchor = anchor.reindex(self.compounds)
        self.contract = {"model_version": "V0+", "expression_scale": "log2(TPM+1)", "endpoint_id": card["endpoint"],
                         "endpoint_direction": "higher", "native_domain_rule": False, "release": card}
        self.files = {name: digest(path / name) for name in ("release.json", target.name, "compounds.tsv")}
        self.runtime = {p.name: digest(p) for p in sorted((Path(upstream) / "src/vc_build").glob("*.py"))}

        state = self.features.transform(self.source.reindex(columns=self.genes))
        self.project = PCA(n_components=min(16, len(state) - 1), random_state=9021)
        pcs = self.project.fit_transform(state)
        self.whiten = StandardScaler().fit(pcs)
        self.coordinates = self.whiten.transform(pcs)
        distance = cdist(self.coordinates, self.coordinates)
        distance[self.groups[:, None] == self.groups[None, :]] = np.inf
        self.threshold = float(np.quantile(distance.min(axis=1), .95))
        self.contract["rna_distance_threshold"] = self.threshold

    def predict(self, frame):
        table = frame.reindex(columns=self.genes)
        coverage = table.notna().mean(axis=1)
        observed = table.iloc[:, self.features.indices].notna().mean(axis=1)
        support = coverage.ge(.85) & observed.ge(.85) & table.max(axis=1).gt(0)
        filled = table.fillna(pd.Series(self.features.raw_median, index=self.genes))
        state = self.features.transform(filled)
        distances = cdist(self.whiten.transform(self.project.transform(state)), self.coordinates).min(axis=1)
        outside = distances > self.threshold
        inputs = pd.DataFrame({"cell_id": table.index[support]}).merge(self.panel.reset_index(), how="cross")
        values = pd.DataFrame(np.nan, index=table.index, columns=self.compounds)
        if len(inputs):
            reference, delta = self.model.predict(inputs, filled.loc[support])
            score = np.asarray(reference) + self.multiplier * np.asarray(delta)
            if not np.isfinite(score).all():
                raise ValueError("V0+ produced nonfinite output on a supported tumor row")
            scored = inputs.assign(score=score).pivot(index="cell_id", columns="compound_id", values="score")
            values.loc[scored.index] = scored
        domain = pd.DataFrame(dict(gene_coverage=coverage, selected_gene_coverage=observed,
            nearest_fit_distance=distances, insufficient_genes=~support, ood=outside, abstain=outside | ~support),
            index=table.index)
        return values, domain
