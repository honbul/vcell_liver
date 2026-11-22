"""Check DEV I paired kernels on synthetic drug profiles only."""

import numpy as np

from src.r19_plan.i_kernel_arms import (ANNOTATION, annotation_matrix,
                                       curated_annotation_rows, fit_optional_profiles,
                                       iter_kernel_arms, shuffle_views)


def example():
    fitting = np.array([True, True, True, True, False, False, False, False, False])
    query = np.array([False, False, False, False, True, True, True, True, False])
    x = np.array([[1., 3., 4., 8., 6., 7., 2., np.nan, 1.],
                  [0., 1., 3., 2., 7., 1., 0., np.nan, 5.]])
    structure = 0.7 * np.eye(len(fitting)) + 0.3
    profiles = {"I1": x.T.copy(), "I2": x[::-1].T.copy(), "I4": -x.T.copy(),
                ANNOTATION: np.where(np.isfinite(x.T), (x.T > 1.0).astype(float), np.nan)}
    profiles["I2"][4] = np.nan
    profiles["I4"][[1, 5]] = np.nan
    return structure, profiles, fitting, query


def test_registered_arms_are_psd_and_keep_matching_query_axes():
    structure, profiles, fitting, query = example()
    mapped = fit_optional_profiles(profiles, fitting, rank=2)
    arms = {arm.name: arm for arm in iter_kernel_arms(structure, mapped, fitting, query)}
    expected = {"I0", "I1_annotation_only", "I2_morphology_only"}
    for name in ("I1", "I2", "I4", "I3", "I3_all"):
        expected.update([name, f"{name}_mask_only", f"{name}_parent_shuffle"])
    for name, views in (("I3", ("I1", "I2")), ("I3_all", ("I1", "I2", "I4"))):
        expected.add(f"{name}_zero_fill_mask")
        expected.update(f"{name}_without_{view}" for view in views)
    assert set(arms) == expected
    assert arms["I3"].views == ("I1", "I2")
    for arm in arms.values():
        assert np.linalg.eigvalsh(arm.gram).min() >= -1e-10
        np.testing.assert_array_equal(arm.query_indices, np.flatnonzero(query))
        np.testing.assert_array_equal(arm.train_indices, np.flatnonzero(fitting))
        assert arm.train_gram.shape == (4, 4)
        assert arm.query_gram.shape == (4, 4)
    np.testing.assert_allclose(arms["I3_without_I1"].gram, arms["I2"].gram)
    np.testing.assert_allclose(arms["I3_all_without_I4"].gram, arms["I3"].gram)
    assert not np.allclose(arms["I3_zero_fill_mask"].gram, arms["I3"].gram)


def test_held_values_cannot_change_fit_grams_and_shuffling_stays_within_roles():
    structure, profiles, fitting, query = example()
    mapped = fit_optional_profiles(profiles, fitting, rank=2)
    before = {arm.name: arm for arm in iter_kernel_arms(structure, mapped, fitting, query)}
    held = ~fitting
    profiles["I1"][held] *= 100_000
    profiles["I2"][held] -= 1_000_000
    mapped = fit_optional_profiles(profiles, fitting, rank=2)
    after = {arm.name: arm for arm in iter_kernel_arms(structure, mapped, fitting, query)}
    for name in before:
        np.testing.assert_allclose(before[name].train_gram, after[name].train_gram)
    assert not np.allclose(before["I1"].query_gram, after["I1"].query_gram)

    values = np.arange(9, dtype=float)[:, None]
    present = np.array([True, True, False, True, True, False, True, True, True])
    shuffled, still_present = shuffle_views({"I1": (values, present)}, 4, seed=1517)["I1"]
    np.testing.assert_array_equal(still_present, present)
    np.testing.assert_array_equal(shuffled[~present], values[~present])
    for domain in (slice(None, 4), slice(4, None)):
        np.testing.assert_array_equal(np.sort(shuffled[domain], axis=0), np.sort(values[domain], axis=0))
    assert not np.array_equal(shuffled, values)


def test_unobserved_parents_get_the_fixed_I0_forecast():
    structure, profiles, fitting, query = example()
    mapped = fit_optional_profiles(profiles, fitting, rank=2)
    selected = next(arm for arm in iter_kernel_arms(structure, mapped, fitting, query)
                    if arm.name == "I3_all")
    np.testing.assert_equal(selected.query_view_count, [2, 2, 3, 0])
    prediction = np.zeros((query.sum(), 2))
    baseline = np.ones_like(prediction)
    combined = selected.apply_structure_fallback(prediction, baseline)
    np.testing.assert_equal(combined[:3], prediction[:3])
    np.testing.assert_equal(combined[3], baseline[3])


def test_annotation_vocabulary_uses_curated_actions_and_fit_support(tmp_path):
    assays = tmp_path / "bioassays.tsv"
    assays.write_text("parent\ttarget_chembl_id\tcurated_target_action_annotations\taction_type\tpchembl_value\n"
                      "p1\tt1\tAGONIST|ANTAGONIST\tMODULATOR\t5\n"
                      "p2\tt1\tAGONIST\tMODULATOR\t7\n"
                      "p3\tt1\tAGONIST\tMODULATOR\t9\n"
                      "p4\tt1\tANTAGONIST\tMODULATOR\t8\n"
                      "p5\tt1\t\tMODULATOR\t5\n")
    records = curated_annotation_rows(assays)
    matrix = annotation_matrix(records, ["p1", "p2", "p3", "p4", "p5"],
                               [True, True, False, False, False])
    np.testing.assert_equal(matrix[:, 0], [1., 1., 1., np.nan, np.nan])
    assert matrix.shape == (5, 1)
    mapped = fit_optional_profiles({ANNOTATION: matrix}, [True, True, False, False, False])
    values, present = mapped[ANNOTATION]
    np.testing.assert_array_equal(present, [True, True, True, False, False])
    np.testing.assert_array_equal(values[:, 0], [1., 1., 1., 0., 0.])
