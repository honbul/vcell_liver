"""Construct the matched-update additional-dose development comparison."""

from copy import copy
from functools import cached_property

import numpy as np
import pandas as pd

from src.r11_chemical_probe.dataset import Block, CONTROL, ENTITY, GROUP, LINE, group_weights, load_data
from src.r11_chemical_probe.models import Interaction, PoolMean

from .data import Design
from .network import packs
from .split import eligible_pools


def combine(a, b):
    rows = pd.concat((a.rows, b.rows), ignore_index=True)
    return Block(rows, np.concatenate((a.control, b.control)),
                 {k: np.concatenate((a.chemical[k], b.chemical[k])) for k in a.chemical},
                 np.concatenate((a.treated, b.treated)))


class DoseCache:
    def __init__(self, source):
        self.source = source

    @cached_property
    def treated(self):
        groups = set(self.source.fit_rows[GROUP])
        return load_data(self.source.assets / "doses/source", "treated",
                         mask=lambda rows: rows[GROUP].isin(groups).to_numpy())

    @cached_property
    def control(self):
        return load_data(self.source.assets / "doses/source", self.source.role)

    def block(self, fit):
        rows, matrix = self.treated
        controls, cv = self.control
        seen = set(fit.train.rows[GROUP])
        if seen.intersection(fit.query.rows[GROUP]):
            raise ValueError("Dose fit overlaps query compound groups")
        valid = rows[GROUP].isin(seen)
        rows = rows[valid].reset_index(drop=True)
        matrix = matrix[valid.to_numpy()]
        controls_ok = controls[["plate_id", LINE]].copy().assign(_known=True)
        use = rows.merge(controls_ok, on=["plate_id", LINE], how="left", validate="many_to_one")["_known"].eq(True)
        kept = rows.loc[use].assign(_position=np.flatnonzero(use))
        kept = eligible_pools(kept, fit.source.cfg["minimum_groups"])
        index = kept.pop("_position").to_numpy()
        block = self.source.assemble(kept, controls, cv[:, fit.panel].toarray().astype(np.float32),
                                     matrix[index][:, fit.panel].toarray().astype(np.float32))
        if not set(block.rows[GROUP]) <= seen:
            raise ValueError("An extra-dose compound entered from outside fit")
        return block


def dose_design(fit, augmented):
    max_rows = fit.cfg["batch_rows"]
    count = len(packs(fit.train.rows, max_rows))
    model = copy(fit)
    train, query = fit.train, fit.query
    if augmented:
        if not hasattr(fit.source, "dose_cache"):
            fit.source.dose_cache = DoseCache(fit.source)
        more = fit.source.dose_cache.block(fit)
        if len(more.rows) == 0:
            raise ValueError("No eligible additional-dose fit rows")
        train = combine(train, more)
        model.train = train
        # Distinct dose/plate pools have their own leave-group-out anchors.
        means = PoolMean(train.rows, train.delta)
        model.offset = means.predict(query.rows, fit.cfg["offset_shrinkage"])[0]
        m = means.predict(train.rows, fit.cfg["offset_shrinkage"], leave_group_out=True)[0]
        model.residual = (train.delta - m).astype(np.float32)
        model.weight = group_weights(train.rows)
        model.scale = max(1e-6, float(np.sqrt(np.dot(model.weight, np.square(model.residual).mean(axis=1))
                                             / model.weight.sum())))
    drug_unit = ~train.rows.duplicated(ENTITY)
    control_unit = ~train.rows.duplicated([CONTROL, LINE])
    drug_pca, drug_scale = Interaction._fit(train.chemical["E"][drug_unit], 32)
    ctx_pca, ctx_scale = Interaction._fit(train.control[control_unit], 16)
    output = []
    for block in (train, query):
        raw = drug_scale.transform(drug_pca.transform(block.chemical["E"]))
        dose = np.log10(block.rows.dose_uM.to_numpy() / fit.source.cfg["dose_uM"])
        drug = np.column_stack((raw, dose)).astype(np.float32)
        ctrl = ctx_scale.transform(ctx_pca.transform(block.control)).astype(np.float32)
        output.append((drug, ctrl))
    return model, Design(*output), count
