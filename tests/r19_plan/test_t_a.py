from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r19_plan import t_a


def test_nested_calibration_keeps_pseudo_domain_and_parent_roles_disjoint(monkeypatch):
    cells = ("A375", "A549", "HT29", "MCF7")
    source = pd.DataFrame([dict(study="GSE92742", cell_id=cell, parent=f"p{i}", inst_id=f"{cell}-{i}")
                           for cell in cells for i in range(9)])
    target = pd.DataFrame([dict(study="GSE92742", cell_id="HEPG2", parent="held-parent", inst_id="held-well")])
    bank = SimpleNamespace(training=source, outer=target, target="HEPG2", fold=0,
                           landmarks=pd.DataFrame({"pr_gene_id": [1, 2]}))
    controls = pd.DataFrame({"cell_id": [*cells, "HEPG2"]})
    staged = SimpleNamespace(tables={"fit_controls": controls}, arrays={"fit_controls": np.zeros((5, 2))})
    calls = []

    def fixed(_bank, _opts, _assets, training, query):
        assert "HEPG2" not in set(training.cell_id)
        assert "held-parent" not in set(training.parent)
        assert set(training.cell_id).isdisjoint(query.cell_id)
        assert set(training.parent).isdisjoint(query.parent)
        calls.append((training.cell_id.nunique(), training.parent.nunique(), set(query.cell_id)))
        return np.full((len(query), 2), .5)

    monkeypatch.setattr(t_a, "fixed_prediction", fixed)
    monkeypatch.setattr(t_a, "effects", lambda *args: np.ones((len(source), 2)))
    monkeypatch.setattr(t_a, "source_folds", lambda rows, fold: rows.parent.str[1:].astype(int) % 3)
    monkeypatch.setattr(t_a, "rule_factors", lambda *args: {key: np.ones(2) for key in t_a.SETTINGS})
    monkeypatch.setattr(t_a, "config", lambda *_: {})
    guess, selected = t_a.nested_calibration(SimpleNamespace(config=None, assets=None), {}, {}, staged, bank)
    assert len(calls) == 61
    assert all(reps == 3 for _, reps, _ in calls[:48])
    assert all(domains == 3 and reps == 6 for domains, reps, _ in calls[48:60])
    assert calls[-1] == (4, 9, {"HEPG2"})
    assert selected["source_wells"] == len(source)
    np.testing.assert_array_equal(guess["T1_A"], guess["T3_A"])
