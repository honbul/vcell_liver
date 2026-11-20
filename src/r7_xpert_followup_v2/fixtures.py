"""Synthetic fixtures for the calibrator (no study data). Checks: recovery of known a_c, b_c with small lambda; the fitted
objective never exceeds the identity's in-sample objective (identity is feasible at lambda -> 0 penalty for a=1, b=0);
identity arm returns its input; positive scales; shrinkage towards the global prior grows with lambda."""
import json
import numpy as np

from . import fit_calibrator as fc
from .common import pman, dump, stamp


def main():
    rng = np.random.RandomState(7)
    G, cells = 978, ["c0", "c1", "c2", "c3"]
    a_true = {"c0": 0.5, "c1": 1.0, "c2": 2.0, "c3": 1.5}
    b_true = {c: rng.normal(0, 0.2, G) for c in cells}
    rows = []
    P, Y, C, F = [], [], [], []
    for c in cells:
        n = 400
        p = rng.normal(0, 0.3, (n, G))
        y = a_true[c] * p + b_true[c] + rng.normal(0, 0.05, (n, G))
        P.append(p); Y.append(y); C += [c] * n; F += [f"f{c}"] * n
    P, Y, C, F = np.concatenate(P), np.concatenate(Y), np.array(C), np.array(F)
    hcc = np.zeros(len(P), bool)
    out = {}
    co = fc.fit("Csb", 1e-6, P, Y, C, F, hcc)
    out["recovery_a_max_abs_err"] = max(abs(co["cells"][c]["a"] - a_true[c]) for c in cells)
    out["recovery_b_max_rms_err"] = max(float(np.sqrt(((co["cells"][c]["b"] - b_true[c]) ** 2).mean())) for c in cells)
    for arm in ["Cg", "Cs", "Cb", "Csb"]:
        c_ = fc.fit(arm, 1e-6, P, Y, C, F, hcc)
        Z, _ = fc.apply(c_, P, C)
        out[f"{arm}_insample_mse_le_identity"] = bool(((Z - Y) ** 2).mean() <= ((P - Y) ** 2).mean() + 1e-12)
        out[f"{arm}_all_a_positive"] = bool(np.exp(c_["alpha0"]) > 0 and all(d["a"] > 0 for d in c_["cells"].values()))
    Z0, _ = fc.apply(fc.fit("C0", 1.0, P, Y, C, F, hcc), P, C)
    out["identity_returns_input"] = bool(np.array_equal(Z0, P))
    spread = []
    for lam in [0.01, 1, 100, 1e4]:
        c_ = fc.fit("Csb", lam, P, Y, C, F, hcc)
        spread.append(float(np.std([d["v"] for d in c_["cells"].values()])))
    out["shrinkage_spread_by_lambda"] = spread
    out["shrinkage_monotone"] = bool(all(spread[i] >= spread[i + 1] - 1e-12 for i in range(len(spread) - 1)))
    out["pass"] = bool(out["recovery_a_max_abs_err"] < 0.01 and out["recovery_b_max_rms_err"] < 0.01 and out["identity_returns_input"]
                       and out["shrinkage_monotone"] and all(v for k, v in out.items() if k.endswith(("_le_identity", "_positive"))))
    dump(dict(out, **stamp()), pman("calibrator_synthetic_fixtures.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
