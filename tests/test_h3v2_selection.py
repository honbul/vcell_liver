import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from h3v2.fit_tree import train
from h3v2 import selection
from h3v2.grid import boundary_neighbors, identify, initial
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


def test_early_neural_plan_matches_complete_inner_selection():
    stages = ["r1", "r2", "boundary-1"]

    def sweep(split, family, prior):
        assert prior == stages
        config = initial(family)[0]
        return [{"config": config, "id": identify(config), "utility": .5,
                 "point": {"ndcg_at_5": .4, "spearman": .6}, "max_budget_pressure_iteration": 12,
                 "folds": [{"receipt_sha256": str(i)} for i in range(3)]}]

    with TemporaryDirectory(dir=selection.ROOT / "tmp") as scratch:
        root = Path(scratch)
        with patch.multiple(selection, ROOT=root, SPLITS=("drug-0",)), \
                patch.object(selection, "evaluated", side_effect=sweep) as evaluated:
            (root / "locks").mkdir()
            (root / "locks/boundary-1.json").write_text("{}", encoding="utf-8")
            selection.register_neural_expansion(stages)
            assert [call.args[1] for call in evaluated.call_args_list] == list(selection.NEURAL_FAMILIES)
            plan = selection.read_neural_expansion("boundary-2")
            with patch.object(selection, "read_neural_expansion", return_value=plan | {"stages": ["r1"]}):
                try:
                    selection.select(stages)
                except ValueError as exc:
                    assert "differ from the complete inner selection" in str(exc)
                else:
                    raise AssertionError("A mismatched plan was accepted")
            assert not (root / "locks/boundary-1-selection.json").exists()
            selection.select(stages)
            registered = selection.read_json(root / "locks/boundary-2.json")["splits"]["drug-0"]
            assert {family: registered[family] for family in selection.NEURAL_FAMILIES} == plan["splits"]["drug-0"]


if __name__ == "__main__":
    test_matched_boundary_expands_and_brackets()
    test_frozen_boosting_keeps_the_independent_inner_budgets()
    test_early_neural_plan_matches_complete_inner_selection()
    print("Matched saturation neighbors and independent frozen L2 / LambdaRank budgets passed")
