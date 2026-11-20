"""Leak fixtures (run once before the pilot; no test delta is read).
1. Store.delta refuses test units; Store.test_delta refuses without R7_FINAL_EVAL / committed freeze manifest.
2. TrainStats are unchanged when validation deltas are replaced by noise (fit on training units only).
3. A model's prediction for a unit does not change when every delta array is replaced by noise (inputs: X0, ECFP, KG only).
4. The KG build does not import the expression store.
"""
import json, os
import numpy as np
import torch

from .common import pman, dump
from .data import Store, TrainStats, Knowledge
from .models import R7Model
from .train import SeedBank, load_graph, tensors, predict


def main():
    S = Store(); res = {}
    try:
        S.delta(S.test[:3]); res["delta_refuses_test"] = False
    except AssertionError:
        res["delta_refuses_test"] = True
    os.environ.pop("R7_FINAL_EVAL", None)
    try:
        S.test_delta(S.test[:3]); res["test_delta_refuses_without_final_eval"] = False
    except AssertionError:
        res["test_delta_refuses_without_final_eval"] = True
    st1 = TrainStats(S)
    rng = np.random.RandomState(0)
    keep = S._open.copy()
    vo = S.units.open_idx.to_numpy()[S.val]
    S._open[vo] = rng.normal(size=S._open[vo].shape).astype(np.float32)
    st2 = TrainStats(S)
    res["train_stats_ignore_validation"] = bool(np.allclose(st1.cell_mean, st2.cell_mean) and np.allclose(st1.x0_mu, st2.x0_mu))
    S._open = keep
    K = Knowledge(len(S.groups))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    T = tensors(S, st1, dev)
    idx = S.val[:128]
    for arm in ["gated", "concat"]:
        m = R7Model(arm, load_graph(K)).to(dev)
        for p in m.parameters():          # non-zero output layers so the correction is active
            if p.dim() and p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.01)
        bank = SeedBank(K)
        a = predict(m, T, bank, idx, dev)
        a2 = predict(m, T, bank, idx, dev)                 # same inputs: GPU scatter-add run-to-run noise
        S._open = rng.normal(size=S._open.shape).astype(np.float32)
        b = predict(m, T, bank, idx, dev)
        S._open = keep
        res[f"{arm}_rerun_max_abs_diff"] = float(np.abs(a - a2).max())
        res[f"{arm}_delta_swap_max_abs_diff"] = float(np.abs(a - b).max())
        res[f"{arm}_prediction_independent_of_delta"] = bool(np.abs(a - b).max() < 1e-5)
        res[f"{arm}_correction_nonzero"] = bool(np.abs(a - (T["CM"][torch.as_tensor(S.cell_idx[idx], device=dev)].cpu().numpy())).max() > 0)
    src = open(os.path.join(os.path.dirname(__file__), "kg_build.py")).read()
    import re
    reads = re.findall(r'r4proc\(([^)]*)\)', src)
    res["kg_build_reads_only_metadata"] = ("from .data" not in src) and ("unit_delta" not in src) and ("unit_x0" not in src) and reads == ['"groups.tsv"']
    res["all_pass"] = all(v for k, v in res.items() if k != "all_pass" and isinstance(v, bool))
    dump(res, pman("leak_fixtures.json"))
    print(json.dumps(res, indent=1))
    assert res["all_pass"]


if __name__ == "__main__":
    main()
