import json

import numpy as np
import pandas as pd

from h3v2.fit_tree import train
from h3v2.grid import boundary_neighbors, initial
from h3v2.selection import branch_budgets
from test_h3v2_models import fixture


def test_matched_boundary_expands_and_brackets():
    config = initial("film")[2]
    fitted = initial("film")
    more = boundary_neighbors(config, fitted, 25)
    width = sorted(point["width"] for point in more if point["width"] != config["width"])
    assert width == [256, 1024]
    roundtrip = json.loads(json.dumps([*fitted, *more]))
    assert not boundary_neighbors(config, roundtrip, 25)
    wider = next(point for point in more if point["width"] == 1024)
    assert 2048 in {point["width"] for point in boundary_neighbors(wider, roundtrip, 25)}


def test_frozen_boosting_keeps_the_independent_inner_budgets():
    config = initial("hybrid")[1] | {"min_child": 2, "colsample": 1.0}
    rows = [{"training": {"branch_iterations": pair}} for pair in ([9, 2], [14, 4], [11, 5])]
    budget = branch_budgets(config, rows)
    assert budget == [11, 4]
    arrays = fixture()
    a, b = np.arange(15).repeat(4), np.tile(np.arange(4), 15)
    signal = arrays["rna"][a, 0] * arrays["bits"][b, 0] + 3 * arrays["bits"][b, 1]
    arrays.update(fit_cells=a, fit_drugs=b, query_cells=a[:8], query_drugs=b[:8], labels=signal, scale=1.0)
    fit = pd.DataFrame({"source_id":"synthetic", "endpoint_id":"example", "cell_id":a.astype(str),
                        "endpoint_value":signal, "endpoint_direction":"higher"})
    forecast, stages, _, stamp = train(arrays, {}, fit, fit.iloc[:8].drop(columns="endpoint_value"), None, config,
                                      81, train_base=np.zeros(len(fit)), val_base=np.zeros(8), fixed_epochs=budget)
    assert stamp["branch_iterations"] == budget
    assert [stage["objective"] for stage in stages] == ["l2", "lambdarank"]
    assert np.isfinite(forecast).all()


if __name__ == "__main__":
    test_matched_boundary_expands_and_brackets()
    test_frozen_boosting_keeps_the_independent_inner_budgets()
    print("Matched saturation neighbors and independent frozen L2 / LambdaRank budgets passed")
