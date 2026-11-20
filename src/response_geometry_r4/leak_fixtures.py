"""Leak fixtures: run before any model is trained. Each check raises on failure; results are written to a small JSON."""
import inspect, os
import numpy as np
import pandas as pd

from .common import MAN, public, dump
from .data import Store, TrainStats, reliability_weights
from .models import W0Native, W1DeltaAnchor, DeltaMLP
from .train import dev_subset


def refuses(f):
    try:
        f()
    except AssertionError:
        return True
    return False


def main():
    S = Store(); u = S.units
    res = {}
    split = pd.read_csv(os.path.join(MAN, "drug_split.tsv"), sep="\t").set_index("drug_group").role
    res["unit_role_equals_group_role"] = bool((u.drug_group.map(split) == u.role).all())
    res["one_role_per_group_across_cells"] = bool(u.groupby("drug_group").role.nunique().max() == 1)
    res["delta_refuses_test_units"] = refuses(lambda: S.delta(S.test[:3]))
    os.environ.pop("R4_FINAL_EVAL", None)
    res["test_delta_refused_outside_final_eval"] = refuses(lambda: S.test_delta(S.test[:3]))
    res["teacher_refuses_validation_fit"] = refuses(lambda: TrainStats(S, fit_idx=S.val[:100]))
    st = TrainStats(S)
    S2 = Store(); S2._open = S2._open.copy()
    S2._open[u.open_idx.to_numpy()[S2.val]] = np.random.RandomState(1).randn(len(S2.val), 978).astype(np.float32)
    st2 = TrainStats(S2)
    res["teacher_invariant_to_validation_values"] = bool(np.array_equal(st.U, st2.U) and np.array_equal(st.cell_mean, st2.cell_mean) and np.array_equal(st.scale, st2.scale))
    rr, known, _ = reliability_weights(S, st)
    res["reliability_only_training_units"] = bool(np.isnan(rr[S.val]).all() and np.isnan(rr[S.test]).all() and (u.half_idx.to_numpy()[u.role.to_numpy() != "train"] == -1).all())
    for cls in (W0Native, W1DeltaAnchor, DeltaMLP):
        res[f"predict_signature_{cls.__name__}"] = list(inspect.signature(cls.predict_delta).parameters) == ["self", "x0", "fp"]
    tr, va, cells, groups = dev_subset(S)
    res["dev_subset_train_only"] = bool((S.role[tr] == "train").all() and (S.role[va] == "validation").all())
    res["dev_subset_size"] = dict(train_units=int(len(tr)), val_units=int(len(va)), cells=cells, groups=len(groups))
    res["hcc_test_units_not_in_train"] = bool(not ((u.is_hcc == 1) & (u.role == "test")).to_numpy()[S.train].any())
    ok = all(v for k, v in res.items() if isinstance(v, bool))
    res["all_passed"] = ok
    dump(res, public("manifests", "response_geometry_r4", "leak_fixtures.json"))
    print(res)
    assert ok


if __name__ == "__main__":
    main()
