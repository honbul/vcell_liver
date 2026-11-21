import pytest

from src.r19_plan.hcc_pilot_layout import layout, verify


def test_pilot_repeats_are_balanced_across_plates_and_positions():
    rows = layout()
    support = verify(rows)
    assert len(support) == 16
    assert support.treatment_edge.eq(12).all()
    assert support.vehicle_edge.eq(3).all()


def test_pilot_verifier_rejects_a_duplicate_physical_well():
    rows = layout()
    rows.loc[1, "well"] = rows.loc[0, "well"]
    with pytest.raises(ValueError, match="physical coordinates"):
        verify(rows)
