import numpy as np
import pandas as pd
import pytest

from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE
from src.r19_plan.hcc_scaffold import LINE_ID, neighbors, scaffold_draw


def test_neighbor_records_tied_same_line_donors_only():
    train = pd.DataFrame({GROUP: [1, 2, 3], ENTITY: ["a", "b", "c"], LINE: [LINE_ID, LINE_ID, "other"]})
    query = pd.DataFrame({GROUP: [4], ENTITY: ["q"], LINE: [LINE_ID]})
    entities = pd.Index(["a", "b", "c", "q"])
    fingerprints = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0]])
    components = pd.Series([1, 2, 3, 1], index=[1, 2, 3, 4])
    result = neighbors(train, query, entities, fingerprints, components).iloc[0]
    assert result.max_tanimoto == .5
    assert result.nearest_fit_parents == "1|2"
    assert result.fit_scaffold_support
    with pytest.raises(ValueError, match="held parent"):
        neighbors(train, query.assign(**{GROUP: 1}), entities, fingerprints, components)


def test_scaffold_draw_reuses_the_same_weights_only_within_component():
    draw = scaffold_draw(pd.Index([2, 3, 10]), pd.Series([5, 7, 5], index=[2, 3, 10]))
    np.testing.assert_array_equal(draw[:, 0], draw[:, 2])
    assert not np.array_equal(draw[:, 0], draw[:, 1])
