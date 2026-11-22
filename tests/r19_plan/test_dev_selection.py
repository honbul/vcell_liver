"""Check D1 and D2 policy differences using only synthetic well arrays."""

from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(not all(find_spec(name) for name in ("h5py", "rdkit")),
                                reason="requires the R17 LINCS estimators environment")


@pytest.fixture
def design():
    from src.r19_plan.dev_selection import (ComparisonDomain, PhysicalControls, ResponseAxis,
                                           ValidationTreatment)

    genes = (16, 22)
    rows = pd.DataFrame({"study": ["geo"] * 3, "inst_id": ["id1", "id2", "id3"],
                         "parent": ["p", "q", "q"], "rna_plate": ["p01", "p01", "p02"],
                         "rna_well": ["A01", "A02", "A01"]})
    wells = {phase: tuple((plate, f"{axis}{col:02d}") for plate in ("p01", "p02") for col in (1, 2))
             for phase, axis in (("fit", "B"), ("select", "C"), ("score", "D"))}
    domain = ComparisonDomain(rows, genes, frozenset({"inner-train-parent"}),
                              frozenset({"outer-held-parent"}), wells)
    axis = ResponseAxis(tuple(zip(rows.study, rows.inst_id, strict=True)), genes)
    validation = ValidationTreatment(axis, np.array([[1., 2.], [2., 1.], [1.5, 2.5]]))
    controls = {phase: PhysicalControls(wel, genes, np.zeros((len(wel), len(genes))))
                for phase, wel in wells.items()}
    return domain, validation, controls


def test_D2_uses_real_checkpoint_arrays_and_the_same_registered_loss(design):
    from src.r15_lincs.estimators import loss
    from src.r19_plan.dev_selection import Checkpoint, CheckpointBank, compare_d2

    domain, validation, controls = design
    truth = validation.values
    anchor = np.full_like(truth, 0.25)
    target = truth - anchor
    bank = CheckpointBank(validation.axis, anchor, (
        Checkpoint("strong", (0.001, 10), 4 * target),
        Checkpoint("moderate", (0.001, 5), .8 * target),
    ))
    results = compare_d2(domain, bank, validation, controls["select"])
    old, new = results["legacy_sequential"], results["joint"]
    assert (old.config_id, old.gain) == ("moderate", 1.)
    assert old.objective == loss(domain.validation_rows, truth, anchor + .8 * target)
    assert (new.config_id, new.gain, new.objective) == ("strong", .25, 0.)


def test_D1_changes_only_plate_control_means_on_the_same_bank(design):
    from src.r19_plan.dev_selection import Checkpoint, CheckpointBank, compare_d1

    domain, validation, controls = design
    validation = replace(validation, values=np.array([[2.25, 4.25]] * 3))
    anchor = np.full_like(validation.values, .25)
    bank = CheckpointBank(validation.axis, anchor, (Checkpoint("fixed", (5,), validation.values - anchor),))
    # Both plates retain two physical wells. Their distinct values share a mean.
    independent = replace(controls["select"], values=np.array([[1., 2.], [2., 4.], [2., 4.], [1., 2.]]))
    result = compare_d1(domain, bank, validation, controls["fit"], independent)
    assert (result["shared"].gain, result["shared"].objective) == (1., 0.)
    assert (result["independent"].gain, result["independent"].objective) == (.25, 0.)


def test_ties_use_legacy_native_parameters_and_DEV_gain_then_id(design):
    from src.r19_plan.dev_selection import Checkpoint, CheckpointBank, compare_d2

    domain, validation, controls = design
    truth = validation.values
    checkpoints = (Checkpoint("b", (2,), 2 * truth), Checkpoint("c", (1,), truth),
                   Checkpoint("a", (3,), 2 * truth), Checkpoint("d", (0,), truth))
    result = compare_d2(domain, CheckpointBank(validation.axis, np.zeros_like(truth), checkpoints),
                        validation, controls["select"])
    assert (result["legacy_sequential"].config_id, result["legacy_sequential"].gain) == ("d", 1.)
    assert (result["joint"].config_id, result["joint"].gain) == ("a", .5)


def test_validation_rejects_protected_roles_wrong_axes_and_scalar_predictions(design):
    from src.r19_plan.dev_selection import Checkpoint, CheckpointBank, compare_d1, compare_d2

    domain, validation, controls = design
    truth = validation.values
    bank = CheckpointBank(validation.axis, np.zeros_like(truth), (Checkpoint("raw", (), truth),))
    with pytest.raises(ValueError, match="permitted selection role"):
        compare_d2(domain, bank, validation, controls["score"])
    reversed_genes = replace(validation.axis, genes=validation.axis.genes[::-1])
    with pytest.raises(ValueError, match="row or gene order"):
        compare_d2(domain, bank, replace(validation, axis=reversed_genes), controls["select"])
    with pytest.raises(ValueError, match="well-by-gene"):
        compare_d2(domain, replace(bank, checkpoints=(Checkpoint("score", (), 0.3),)), validation, controls["select"])
    with pytest.raises(ValueError, match="well-by-gene"):
        compare_d2(domain, replace(bank, anchor=np.full_like(truth, np.nan)), validation, controls["select"])
    with pytest.raises(ValueError, match="parent sets"):
        compare_d2(replace(domain, outer_held_parents=frozenset({"p"})), bank, validation, controls["select"])
    overlapping = dict(domain.control_roles, score=(("P01", "c1"), *domain.control_roles["score"][1:]))
    with pytest.raises(ValueError, match="Physical wells overlap"):
        compare_d2(replace(domain, control_roles=overlapping), bank, validation, controls["select"])
    unequal = dict(domain.control_roles, score=domain.control_roles["score"][1:])
    with pytest.raises(ValueError, match="equal numbers"):
        compare_d1(replace(domain, control_roles=unequal), bank, validation, controls["fit"], controls["select"])
