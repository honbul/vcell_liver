import pytest

from src.r19_plan.hcc_off_position_pilot import FIRST_SEED, SLOTS, draft_layout, verify_layout


def test_pilot_avoids_donor_coordinates_across_both_target_lines():
    forbidden = {slot: {"A01", "B03", "D09", "F04", "H12"} for slot in SLOTS}
    plate_map = draft_layout(forbidden, FIRST_SEED)
    checks = verify_layout(plate_map, forbidden)
    assert len(checks) == 16
    assert checks.treatments.sum() == 576
    assert checks.vehicles.sum() == 192

    row = plate_map.loc[plate_map.kind.eq("treated")].iloc[0]
    forbidden[row.parent_slot].add(row.well)
    with pytest.raises(ValueError, match="overlaps a registered donor index"):
        verify_layout(plate_map, forbidden)
