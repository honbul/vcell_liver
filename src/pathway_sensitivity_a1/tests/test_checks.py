"""Minimum correctness checks of plan section 12. Run: pytest src/pathway_sensitivity_a1/tests -q

Synthetic fixtures always run; checks against the built registry/cache run when those files exist.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402
import metrics  # noqa: E402
import nested  # noqa: E402

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]


def toy(n_fam=30, p=40, n_drugs=4, seed=0):
    rs = np.random.RandomState(seed)
    fam = np.array([f"F{i:02d}" for i in range(n_fam)] + ["F00", "F01", "F01"])   # clones
    X = rs.normal(size=(len(fam), p))
    Y = X[:, :n_drugs] * 0.8 + rs.normal(size=(len(fam), n_drugs))
    fold = core.group_folds(fam, 5, seed=1)
    return X, Y, fam, fold


def run(X, Y, fam, fold, kind):
    out = np.full(Y.shape, np.nan)
    infos = {}
    for k, tr, te, pred, info in nested.outer_folds(X, Y, fam, fold, kind, ALPHAS, 3, lambda k: 50 + k, K=10):
        out[te] = pred
        infos[k] = (tr, info)
    return out, infos


# ---- family separation -------------------------------------------------------------------------
def test_family_never_in_train_and_test():
    X, Y, fam, fold = toy()
    for k in set(fold):
        assert not set(fam[fold == k]) & set(fam[fold != k])
    for seed in range(5):
        inner = core.group_folds(fam, 3, seed)
        for k in range(3):
            assert not set(fam[inner == k]) & set(fam[inner != k])


def test_committed_split_table():
    path = os.path.join(core.MANIFESTS, "splits_outer.tsv")
    if not os.path.exists(path):
        pytest.skip("split table not built")
    sp = pd.read_csv(path, sep="\t")
    reg = core.load_registry()
    elig = reg[(reg.study == "LIMORE") & reg.included]
    fam_of = dict(zip(reg.model_id, reg.family_id))
    for rep, t in sp.groupby("repeat"):
        assert sorted(t.model_id) == sorted(elig.model_id)          # every eligible model exactly once
        assert (t.family_id == t.model_id.map(fam_of)).all()
        assert t.groupby("family_id").outer_fold.nunique().max() == 1
        for k in t.outer_fold.unique():
            assert not set(t.family_id[t.outer_fold == k]) & set(t.family_id[t.outer_fold != k])
    assert not reg[~reg.included].model_id.isin(sp.model_id).any()   # excluded models never used


# ---- no test-label use -------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["A0", "A1", "A2"])
def test_test_labels_do_not_change_predictions(kind):
    X, Y, fam, fold = toy()
    base, info0 = run(X, Y, fam, fold, kind)
    rs = np.random.RandomState(9)
    for k in sorted(set(fold)):
        Y2 = Y.copy()
        Y2[fold == k] = rs.normal(size=Y2[fold == k].shape) * 50      # corrupt only this fold's test labels
        other, info1 = run(X, Y2, fam, fold, kind)
        assert np.allclose(base[fold == k], other[fold == k])       # predictions for that fold unchanged
        assert np.array_equal(info0[k][1]["alpha"], info1[k][1]["alpha"], equal_nan=True)


def test_test_rows_do_not_change_fitted_transform():
    X, Y, fam, fold = toy()
    tr, te = fold != 0, fold == 0
    kw = dict(kind="A2", alphas=ALPHAS, n_inner=3, inner_seed=3, K=10)
    together, _ = nested.fit_predict(X[tr], Y[tr], fam[tr], X[te], **kw)
    alone, _ = nested.fit_predict(X[tr], Y[tr], fam[tr], X[te][:1], **kw)
    assert np.allclose(together[:1], alone)                         # scaler/PCA are fitted on training rows only


def test_mean_baseline_uses_training_mean_only():
    X, Y, fam, fold = toy()
    pred, infos = run(X, Y, fam, fold, "A0")
    for k in set(fold):
        tr = fold != k
        fam_means = pd.DataFrame(Y[tr]).groupby(fam[tr]).mean().mean(axis=0).values   # family-weighted mean
        assert np.allclose(pred[fold == k], fam_means)
        assert np.allclose(infos[k][1]["mu"], fam_means)


def test_alpha_tie_goes_to_largest():
    X, Y, fam, fold = toy()
    Yc = np.ones_like(Y)                                             # constant label: every alpha ties
    _, info = nested.fit_predict(X, Yc, fam, X[:2], kind="A1", alphas=ALPHAS, n_inner=3, inner_seed=0, K=10)
    assert (info["alpha"] == max(ALPHAS)).all() and info["constant"].all()


def test_a2_dimension_rule():
    X, Y, fam, fold = toy()
    _, info = nested.fit_predict(X, Y, fam, X[:2], kind="A2", alphas=ALPHAS, n_inner=3, inner_seed=0, K=50)
    inner = core.group_folds(fam, 3, 0)
    n_min = min((inner != k).sum() for k in range(3))
    assert info["pca_dim"] == min(50, n_min - 1, X.shape[1])


# ---- metric formula ----------------------------------------------------------------------------
def test_metric_matches_hand_calculation():
    # one repeat, one drug, families A (two clones) and B; s_train = 2
    pred = pd.DataFrame(dict(repeat=0, outer_fold=0, model_id=["a1", "a2", "b1"], drug_id="D",
                             pred=[1.0, 1.0, 0.0], train_mean=0.0, s_train=2.0, alpha=1.0, constant=False))
    long = pd.DataFrame(dict(model_id=["b1", "a2", "a1"], drug_id="D", y=[4.0, 5.0, 3.0]))   # shuffled on purpose
    df = metrics.attach_truth(pred, long, dict(a1="A", a2="A", b1="B"))
    # normalised squared errors: a1 (3-1)^2/4 = 1, a2 (5-1)^2/4 = 4, b1 (4-0)^2/4 = 4
    # family A mean = 2.5, family B = 4 -> mean 3.25 -> sqrt
    F = metrics.error_cube(df, [0], ["D"], ["A", "B"])
    assert np.allclose(F, [[[2.5, 4.0]]])
    assert np.isclose(metrics.macro_nrmse(F), np.sqrt(3.25))
    tab = metrics.per_drug_table(df)
    assert np.isclose(tab.nrmse[0], np.sqrt(3.25))
    assert np.isclose(tab.rmse_raw[0], np.sqrt((10.0 + 16.0) / 2))  # raw: A mean (4+16)/2 = 10, B 16
    assert np.isclose(tab.mae_raw[0], (3.0 + 4.0) / 2)
    # bootstrap with identity weights reproduces the point estimate
    assert np.isclose(metrics.boot_macro(F, np.ones((1, 2)))[0], np.sqrt(3.25))
    assert np.isclose(metrics.boot_macro(F, np.array([[2.0, 0.0]]))[0], np.sqrt(2.5))


def test_repeat_mean_is_not_ensemble():
    F = np.array([[[1.0, 9.0]], [[4.0, 4.0]]])                        # two repeats
    assert np.isclose(metrics.macro_nrmse(F), (np.sqrt(5.0) + 2.0) / 2)


# ---- ids ---------------------------------------------------------------------------------------
def test_response_matrix_round_trip():
    long = pd.DataFrame(dict(model_id=["m2", "m1", "m1"], drug_id=["d1", "d2", "d1"], y=[1.0, 2.0, 3.0]))
    Y = core.response_matrix(long, ["m1", "m2"], ["d1", "d2"], "y")
    assert Y[0, 0] == 3.0 and Y[0, 1] == 2.0 and Y[1, 0] == 1.0 and np.isnan(Y[1, 1])
    Y2 = core.response_matrix(long.sample(frac=1, random_state=1), ["m2", "m1"], ["d2", "d1"], "y")
    assert Y2[1, 1] == 3.0 and Y2[1, 0] == 2.0 and Y2[0, 1] == 1.0


def test_model_ids_are_study_prefixed():
    assert core.model_id("LIMORE", "PLCPRF5") != core.model_id("LCCL", "PLC/PRF5")
    assert core.model_id("LCCL", "PLC/PRF5") == core.model_id("LCCL", "PLC.PRF5")


def test_refuse_overwrite(tmp_path):
    p = str(tmp_path / "x.txt")
    core.write_new(p, lambda t: open(t, "w").write("a"))
    with pytest.raises(FileExistsError):
        core.write_new(p, lambda t: open(t, "w").write("b"))


# ---- features ----------------------------------------------------------------------------------
def test_singscore_bounds_and_rank_ties():
    x = np.array([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
    r = core.min_rank(x)
    assert r.tolist() == [[1, 1, 3, 4, 5, 6]]
    top = core.singscore_up(np.array([[1.0, 2, 3, 4, 5, 6]]), np.array([4, 5]))
    bottom = core.singscore_up(np.array([[1.0, 2, 3, 4, 5, 6]]), np.array([0, 1]))
    assert np.isclose(top[0], 0.5) and np.isclose(bottom[0], -0.5)


def test_family_weights():
    w = core.family_weights(np.array(["a", "a", "b"]))
    assert w.tolist() == [0.5, 0.5, 1.0]


# ---- built data --------------------------------------------------------------------------------
def test_built_data_card():
    path = os.path.join(core.MANIFESTS, "data_card.json")
    if not os.path.exists(path):
        pytest.skip("data card not built")
    card = json.load(open(path))
    assert card["rna"]["limore_log2_max"] < 25 and card["rna"]["lccl_log2_max"] < 25   # no double log, one log
    assert card["rna"]["lccl_fpkm_max"] > 100
    assert card["endpoint"]["spearman_aa_ic50"] < 0 and card["endpoint"]["spearman_aa_emax"] < 0
    assert card["external_endpoint"]["spearman_auc_gi50"] > 0
    assert max(card["singscore_check"].values()) < 1e-6


def test_cache_alignment_and_study_separation():
    cfg = core.load_config()
    try:
        d = core.cache_dir(cfg)
    except FileNotFoundError:
        pytest.skip("raw files not present")
    if not os.path.exists(os.path.join(d, "limore__models.txt")):
        pytest.skip("cache not built")
    reg = core.load_registry()
    names = os.listdir(d)
    for stem in ("expr_log2.npy", "rank.npy", "models.txt", "hallmark_real.tsv", "response_long.tsv"):
        assert f"limore__{stem}" in names and f"lccl__{stem}" in names     # same content type, distinct files
    for study in ("LIMORE", "LCCL"):
        s = core.load_study(cfg, study)
        assert all(m.startswith(study + "__") for m in s.models)
        assert set(s.models) == set(reg.model_id[reg.study == study])
        sc = pd.read_csv(os.path.join(d, f"{study.lower()}__hallmark_real.tsv"), sep="\t", index_col=0)
        assert list(sc.index) == s.models                                   # row order equals model list
        assert np.allclose(core.relative_rank(s.expr), s.rank)
    axis = pd.read_csv(os.path.join(core.MANIFESTS, "gene_axis.tsv"), sep="\t")
    assert axis.ensembl_id.is_unique and axis.symbol.is_unique
    assert core.load_study(cfg, "LIMORE").expr.shape[1] == len(axis) == core.load_study(cfg, "LCCL").expr.shape[1]


def test_single_drug_matches_multi_drug():
    X, Y, fam, fold = toy()
    kw = dict(kind="A1", alphas=ALPHAS, n_inner=3, inner_seed=4, K=10)
    multi, im = nested.fit_predict(X[fold != 0], Y[fold != 0], fam[fold != 0], X[fold == 0], **kw)
    one, io = nested.fit_predict(X[fold != 0], Y[fold != 0][:, [2]], fam[fold != 0], X[fold == 0], **kw)
    assert np.allclose(multi[:, [2]], one) and im["alpha"][2] == io["alpha"][0]
