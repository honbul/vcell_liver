import math

import pytest

from src.r19_plan.hcc_confirm_design import pass_probability, required_parents, workload


@pytest.mark.parametrize("effect", (1., 2., 5.))
def test_integer_minimum_matches_forward_probability(effect):
    minimum = required_parents(42.95, .5, 24, effect)
    assert pass_probability(42.95, .5, 24, minimum, effect) >= .90
    assert pass_probability(42.95, .5, 24, minimum - 1, effect) < .90


def test_block_variance_can_make_any_parent_count_insufficient():
    assert math.isinf(required_parents(42.95, 1., 8, 1.))


@pytest.mark.parametrize("effect", (0., .4, .5))
def test_no_finite_parent_count_can_reliably_clear_an_effect_at_or_below_the_gate(effect):
    assert math.isinf(required_parents(4., 0., 24, effect))


def test_workload_retains_distinct_blocks_for_two_cell_lines():
    assert workload(96, 8) == (576, 16, 192)
    assert workload(1243, 24) == (7458, 156, 1872)
