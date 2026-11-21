from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.dataset import Block, CONTROL, ENTITY, GROUP, KEY, LINE, group_weights
from src.r13_optimization.kernels import _features, _neighbor_weights, _tanimoto, candidates, predict


def _fixture():
    rng = np.random.default_rng(13)
    smiles = ("CCO", "CCN", "CCCO", "CCCN", "CC(=O)O", "CC(=O)N", "c1ccccc1", "c1ccncc1")
    atoms = rng.integers(0, 2, size=(len(smiles), 20)).astype(np.float32)
    latent = rng.normal(size=(len(smiles), 8)).astype(np.float32)
    controls = rng.normal(size=(3, 6)).astype(np.float32)
    effects = rng.normal(size=(20, 6)).astype(np.float32)
    rows = pd.DataFrame([
        {GROUP: g, ENTITY: f"d{g}", KEY[0]: f"w{g}", LINE: line, CONTROL: "vehicle",
         "plate_id": "p1", "dose_uM": 0.5, "time_h": 24, "smiles": smiles[g]}
        for g in range(len(smiles)) for line in range(len(controls))
    ])
    split = rows[GROUP] < 5
    groups, lines = rows[GROUP].to_numpy(), rows[LINE].to_numpy()
    block = Block(rows, controls[lines], {"E": atoms[groups], "C": latent[groups]})
    tr, qu = block.take(split), block.take(~split)
    residual = (atoms @ effects)[groups[split]] * 0.01
    offset = np.full_like(qu.control, 0.05)
    return SimpleNamespace(train=tr, query=qu, residual=residual, offset=offset,
                           weight=group_weights(tr.rows))


def test_train_only_kernel_map_matches_exact_tanimoto_gram():
    train = np.array([[1.0, 1, 0], [1, 0, 0], [0, 1, 0]])
    query = np.array([[1.0, 0, 1], [0, 0, 1]])
    gram = _tanimoto(train, train)
    cross = _tanimoto(query, train)
    train_features, query_features = _features(gram, cross)
    np.testing.assert_allclose(train_features @ train_features.T, gram, atol=1e-10)
    np.testing.assert_allclose(query_features @ train_features.T, cross, atol=1e-10)


def test_whole_compound_neighbors_tie_weights_have_no_identity_order_preference():
    similarity = np.array([[0.9, 0.5, 0.5, 0.5, 0.4]])
    np.testing.assert_allclose(_neighbor_weights(similarity, 2, 2), [[0.81, 1 / 12, 1 / 12, 1 / 12, 0]])


def test_family_predictions_use_no_query_treated_values():
    fit = _fixture()
    changed = _fixture()
    changed.query.treated = np.full_like(fit.query.control, 1000.0)
    names = set()
    for spec in candidates():
        assert spec["name"] not in names
        names.add(spec["name"])
        prediction = predict(fit, spec)
        assert prediction.shape == fit.offset.shape
        assert np.isfinite(prediction).all()
        np.testing.assert_array_equal(prediction, predict(changed, spec))
