"""Run a single I-track fold from a staged, predictor-only input bundle."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import sha256
from src.r15_lincs.data import table
from src.r15_lincs.design import chemical_features
from src.r15_lincs.estimators import _tanimoto
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_kernel_arms import curated_annotation_rows
from src.r19_plan.i_kernel_audit import phenotype_matrix, quantitative_rows
from src.r19_plan.i_predict import ViewInputs
from src.r19_plan.i_reference import predict_fold
from src.r19_plan.r18_access_plan import PLATE, keys
from src.r19_plan.r18_predict import PredictionBank, model_rows
from src.r19_plan.r18_preflight import checked_tables

SLICE = re.compile(r"f([0-2])-([ABC])")


class IModelBank(PredictionBank):
    """Reuse the guarded physical role bank on I-track prediction arrays only."""

    def __init__(self, staged, slice_id):
        match = SLICE.fullmatch(slice_id)
        if match is None:
            raise ValueError("Expected the registered I-track fold and rotation")
        fold, self.fit = match.groups()
        self.fold = int(fold)
        self.select = "BCA"["ABC".index(self.fit)]
        self.case, self.scenario = "I", "A"
        self.landmarks = staged.landmarks
        tables, arrays = staged.tables, staged.arrays
        self.query_axis = tables["query_features"].assign(fold=self.fold, role=self.fit)
        self.training = model_rows(tables["inner_and_outer_labels"])
        self.outer = model_rows(self.query_axis)
        self.check_splits()
        self.label_keys = keys(self.training)
        if not self.label_keys.is_unique or not keys(self.outer).is_unique:
            raise ValueError("I-track treatment well IDs must be unique")
        self.labels = checked_array(arrays["inner_and_outer_labels"],
                                    (len(self.training), len(self.landmarks)), "I-track fit labels")
        self.role_control = {}
        for name, role in (("fit_controls", self.fit), ("select_controls", self.select)):
            metadata = tables[name]
            values = checked_array(arrays[name], (len(metadata), len(self.landmarks)), role)
            if not metadata.role.eq(role).all():
                raise ValueError("An unexpected physical DMSO role reached I-track prediction")
            lookup = {}
            for plate, indices in metadata.groupby(PLATE).indices.items():
                if not np.all(values[indices] == values[indices[0]]):
                    raise ValueError("The cached DMSO mean changed within its registered plate and role")
                lookup[plate] = values[indices[0]]
            self.role_control[role] = lookup
        self.parents = pd.Index(sorted(set(self.training.parent) | set(self.outer.parent)))
        self.fingerprints = None


def approved_views(root, pin, parents):
    coverage_ref = pin / "coverage_receipt.json"
    coverage = checked_tables(root / "coverage", coverage_ref)
    recorded = table(root / "coverage" / "parent_availability.tsv")
    if recorded.parent.duplicated().any() or not parents.equals(pd.Index(recorded.parent).sort_values()):
        raise ValueError("The I-track response cohort differs from the pinned multimodal population")
    receipts = {"views": sha256(coverage_ref)}
    for name in ("I1", "I2", "I4"):
        ref = pin / f"{name}_receipt.json"
        checked_tables(root / name, ref)
        if sha256(ref) != coverage["feature_view_receipts"][name]:
            raise ValueError(f"The I-track source lineage changed: {name}")
        receipts[name] = sha256(ref)
    structure = chemical_features(parents)
    assays = root / "I1" / "bioassays.tsv.gz"
    inputs = ViewInputs(
        parents, _tanimoto(structure, structure),
        {name: phenotype_matrix(root / name / "features.npz", parents) for name in ("I2", "I4")},
        quantitative_rows(assays), curated_annotation_rows(assays),
    )
    return inputs, receipts


def run(args):
    from src.r19_plan.i_stage import read_prediction_bundle

    staged = read_prediction_bundle(args.bundle, args.split_reference, args.slice)
    bank = IModelBank(staged, args.slice)
    args.fold, args.role = bank.fold, bank.fit
    views, receipts = approved_views(args.views, args.view_references, bank.parents)
    return predict_fold(args, bank, views, receipts | staged.input_receipts | {"config": sha256(args.config)},
                        bank.query_axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", required=True)
    for name in ("bundle", "split-reference", "views", "view-references", "config", "assets", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
