import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import scipy.sparse as sp

from src.r11_chemical_probe import dataset, metrics, prepare, report
from src.r11_chemical_probe.models import Interaction, PoolMean


def row_table():
    return pd.DataFrame({
        "physical_well_id": ["a", "b", "c", "d", "e"],
        "cell_line_id": ["line"] * 5,
        "plate_id": ["plate1"] * 5,
        "compound_group_id": [1, 1, 1, 2, 3],
        "compound_entity_id": ["A", "A", "B", "C", "D"],
        "dose_uM": [0.5] * 5,
        "time_h": [24] * 5,
        "control_well_id": ["ctrl"] * 5,
    })


def test_points_and_weighted_bootstrap():
    rows = row_table()
    contrast = metrics.Contrast(rows, minimum=3)
    observed = np.array([[0, 1], [1, 2], [100, 100], [2, 4], [3, 1]], dtype=float)
    zeros = np.zeros_like(observed)
    assert contrast.point(observed, observed) == 0
    assert contrast.point(observed + 12, observed) == 0
    assert contrast.point(zeros + 12, observed) == pytest.approx(contrast.point(zeros, observed))
    assert contrast.point(zeros, observed) > contrast.point(observed * 0.3, observed) > 0
    weights = np.array([[1, 1, 1], [0.5, 2, 3], [4, 0.01, 0.02]])
    error = observed * 0.3 - observed
    expected = []
    for w in weights:
        row_weights = np.array([w[0]/4, w[0]/4, w[0]/2, w[1], w[2]])
        centered = error - np.average(error, axis=0, weights=row_weights)
        expected.append(np.sqrt(np.mean(centered**2, axis=1)))
    np.testing.assert_allclose(contrast.bootstrap_values(observed * 0.3, observed, weights), expected, atol=1e-11)
    rank, top, n, _ = contrast.retrieval(zeros, observed)
    np.testing.assert_allclose(rank, 0.5)
    np.testing.assert_allclose(top, 1/n)
    for pearson, expected_reason in [(False, "NA_zero_norm"), (True, "NA_zero_variance")]:
        number, reason = metrics.correlations(zeros, observed, pearson=pearson)
        assert np.isnan(number).all()
        assert set(reason) == {expected_reason}


def test_group_exclusion_and_weights():
    rows = row_table()
    target = np.array([0, 0, 100, 20, 40], dtype=np.float32)[:, None]
    np.testing.assert_allclose(dataset.group_weights(rows), [0.25, 0.25, 0.5, 1.0, 1.0])
    baseline = PoolMean(rows, target)
    fit, _, _ = baseline.predict(rows, 2, leave_group_out=True)
    np.testing.assert_allclose(fit[:, 0], [30, 30, 30, 45, 35])
    query = rows.iloc[:1].assign(compound_group_id=999)
    held, _, _ = baseline.predict(query, 2)
    np.testing.assert_allclose(held[0, 0], (50 + 20 + 40) / 3)


def test_raw_alignment_and_full_library():
    kinds = dict(sample=["w"], plate=["plate1"], cell_line_id=["line"])
    prepare.set_scan({"w": (0, "plate1")}, {"line": 0}, np.array([4]), 10, 20)
    batch = pa.RecordBatch.from_pydict({**kinds, "genes": [[1, 3, 4]], "expressions": [[-2., 9., 1.]]})
    row, counts, cells, lib, rejected = prepare.aggregate_batch(batch)
    np.testing.assert_array_equal(row, [0])
    np.testing.assert_array_equal(cells, [1])
    np.testing.assert_array_equal(lib, [10])
    assert rejected == 0
    np.testing.assert_allclose(dataset.normalize(counts, lib).toarray(), [[np.log1p(1000)]])
    with pytest.raises(ValueError, match="Unaligned"):
        prepare.aggregate_batch(pa.RecordBatch.from_pydict(
            {**kinds, "genes": [[1, 3, 4]], "expressions": [[-2., 9.]]}
        ))
    with pytest.raises(ValueError, match="totals|retained"):
        dataset.normalize(sp.csr_matrix([[4]]), np.array([3]))


def test_join_realigns_order_and_rejects_wrong_ids():
    frame = row_table()
    np.testing.assert_array_equal(report.align(frame, frame.iloc[::-1]), [4, 3, 2, 1, 0])
    wrong = frame.assign(physical_well_id=list("abcdz"))
    with pytest.raises(ValueError, match="Missing score join key"):
        report.align(wrong, frame)
    with pytest.raises(ValueError, match="duplicate"):
        prepare.unique(frame.assign(physical_well_id="a"), dataset.KEY)


def test_split_and_transform_are_train_only():
    rows = pd.concat([row_table(), row_table().assign(
        physical_well_id=list("fghij"), control_well_id="ctrl2", cell_line_id="line2"
    )], ignore_index=True)
    rng = np.random.default_rng(42)
    chemical = rng.normal(size=(4, 7))[[0, 0, 1, 2, 3]*2]
    state = rng.normal(size=(2, 6)).repeat(5, axis=0)
    population = dataset.Block(rows, state, {"E": chemical})
    fit = population.take(rows.compound_group_id.lt(3).to_numpy())
    query = population.take(rows.compound_group_id.eq(3).to_numpy())
    dataset.check_split(fit, query)
    transform = Interaction(fit, "E", 5, 5)
    before = transform.transform(fit)
    changed = dataset.Block(query.rows, query.control * 1000, {"E": query.chemical["E"] * -100})
    transform.transform(changed)
    np.testing.assert_array_equal(transform.transform(fit), before)
    assert len(transform.drug[1].keep) <= 2
    with pytest.raises(ValueError, match="Leaking"):
        dataset.check_split(fit, fit)
    with pytest.raises(ValueError, match="unavailable"):
        _ = query.delta
