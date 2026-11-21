"""Verify the complete I-track freeze before calling a separate score reader."""

from collections.abc import Mapping
from contextlib import ExitStack
import hashlib
import json

import numpy as np
import pandas as pd

from src.r19_plan.i_freeze import verify_run
from src.r19_plan.i_score import ViewCoverage, checked_array, checked_rows, coverage_arrays, score_i_track


class JoinedPredictions(Mapping):
    """Load and join one forecast arm at a time in the registered slice order."""

    def __init__(self, archives, lengths, genes):
        self.archives = archives
        self.arm_names = tuple(archives[0].files)
        self.lengths = lengths
        self.genes = len(genes)
        self.shape = (sum(lengths), self.genes)

    def __iter__(self):
        return iter(self.arm_names)

    def __len__(self):
        return len(self.arm_names)

    def __getitem__(self, arm):
        if arm not in self.arm_names:
            raise KeyError(arm)
        first = self.archives[0][arm]
        if first.dtype.kind != "f":
            raise ValueError(f"Nonfloating frozen forecast array: {arm}")
        result = np.empty(self.shape, dtype=first.dtype)
        start = 0
        for i, (archive, length) in enumerate(zip(self.archives, self.lengths, strict=True)):
            values = first if i == 0 else archive[arm]
            if values.shape != (length, self.genes) or values.dtype != result.dtype:
                raise ValueError(f"Frozen forecasts disagree on shape or dtype for {arm}")
            result[start:start + length] = values
            start += length
        return result


def frozen_registry(paths, registered_hashes):
    registry = None
    for path in paths:
        data = (path / "receipt.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != registered_hashes[path.name]:
            raise ValueError(f"I-track slice receipt changed after verification: {path.name}")
        views = json.loads(data)["views"]
        if (not isinstance(views, dict) or "B_R17" in views or views.get("I0") != []
                or any(not isinstance(names, list) or not all(isinstance(name, str) and name for name in names)
                       or len(names) != len(set(names)) or (arm != "I0" and not names)
                       for arm, names in views.items())):
            raise ValueError(f"Unexpected I-track arm/view registry: {path.name}")
        if registry is None:
            registry = views
        elif views != registry:
            raise ValueError(f"I-track arm/view registry differs across slices: {path.name}")
    return registry


def open_forecasts(stack, paths, filename, registered_arms):
    archives = []
    for path in paths:
        bundle = stack.enter_context(np.load(path / filename, allow_pickle=False))
        if len(bundle.files) != len(registered_arms) or set(bundle.files) != registered_arms:
            raise ValueError(f"Frozen {filename} arm inventory differs from receipt: {path.name}")
        archives.append(bundle)
    return archives


def merged_coverage(archives, registry, lengths):
    coverage = {}
    for arm, views in registry.items():
        blocks = [archive[arm] for archive in archives]
        if any(data.shape != (length,) or data.dtype.kind not in "iu" or (data < 0).any()
               or (data > len(views)).any() for data, length in zip(blocks, lengths, strict=True)):
            raise ValueError(f"Invalid frozen view counts: {arm}")
        if arm != "I0":
            counts = np.concatenate([data.astype(np.int64, copy=False) for data in blocks])
            coverage[arm] = ViewCoverage(len(views), counts)
    return coverage


def evaluate_i_run(freeze_path, expected_sha256, forecasts, queries, genes, protocol, *,
                   read_score_delta, fit_parents_by_fold):
    """Score the verified freeze using an independently supplied held-delta loader.

    Queries are independently registered DataFrames keyed by (fold, fit role).
    The callback receives a slice name and a copy of its verified ordered rows.
    It supplies that slice's held-treatment delta using the applicable score DMSO.
    """
    freeze = verify_run(freeze_path, expected_sha256, forecasts, queries, genes, protocol)
    order = [(fold, role) for fold in range(3) for role in "ABC"]
    slices = [f"f{fold}-{role}" for fold, role in order]
    paths = [forecasts / slice_id for slice_id in slices]
    parts = [queries[key] for key in order]
    lengths = [len(part) for part in parts]
    rows = checked_rows(pd.concat(parts, ignore_index=True), fit_parents_by_fold)
    registry = frozen_registry(paths, freeze["folds"])
    arms = set(registry) | {"B_R17"}
    with ExitStack() as stack:
        full = open_forecasts(stack, paths, "forecasts.npz", arms)
        kernel = open_forecasts(stack, paths, "kernel_only.npz", set(registry))
        counts = open_forecasts(stack, paths, "coverage.npz", set(registry))
        coverage = merged_coverage(counts, registry, lengths)
        coverage_arrays(rows, coverage, arms)

        observed = [
            checked_array(read_score_delta(slice_id, expected_rows.copy(deep=True)),
                          (length, len(genes)), f"held score delta {slice_id}")
            for slice_id, expected_rows, length in zip(slices, parts, lengths, strict=True)
        ]
        truth = np.concatenate(observed, axis=0)
        del observed
        return score_i_track(
            rows, truth, JoinedPredictions(full, lengths, genes), coverage,
            kernel_only=JoinedPredictions(kernel, lengths, genes),
            fit_parents_by_fold=fit_parents_by_fold,
        )
