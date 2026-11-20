"""R10 v3 P0: precision/power review of the primary contrast from existing source variance.

Input: v2 strata_primary_contrasts.tsv and primary_contrasts.tsv (contrast P6 = D11[cross] - W[cross], actual TV).
Approximation: SE ~ (hi - lo) / 3.92 from the drug bootstrap 95% interval; drug SD ~ SE * sqrt(n_drugs).
Power for n new drugs (normal approx., two-sided 95% CI upper bound < 0): Phi(|delta| / (SD / sqrt(n)) - 1.96).
Between-batch variance is not estimable from the source, so all numbers are optimistic lower bounds on the need.
"""
import argparse
import math

import pandas as pd

Z = 1.959964
ZB = 0.841621


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_candidates", default="4,10,20,38")
    a = ap.parse_args()
    s = pd.read_csv(f"{a.v2dir}/strata_primary_contrasts.tsv", sep="\t")
    p = pd.read_csv(f"{a.v2dir}/primary_contrasts.tsv", sep="\t")
    rows = []
    src = s[(s.id == "P6") & (s.episode == "E1000") & (s.stratum == "line")].copy()
    src["scope"] = "line=" + src["value"]
    pp = p[p.id == "P6"].copy()
    pp["scope"] = "all_lines(equal weight)"
    for _, r in pd.concat([pp, src], ignore_index=True).iterrows():
        se = (r.hi - r.lo) / (2 * Z)
        sd = se * math.sqrt(r.n_drugs)
        exposure = "development=head selection set (optimistic)" if r.role == "development" else "test=exposed (v1/v2 scored)"
        for delta_name, delta in [("observed_estimate", abs(r.est)), ("practical_threshold", 0.005)]:
            n80 = math.ceil(((Z + ZB) * sd / delta) ** 2) if delta > 0 else float("nan")
            row = dict(contrast="P6 D11[cross]-W[cross]", metric="comp_tv", episode=r.episode, role=r.role,
                       exposure=exposure, scope=r.scope, est=r.est, lo=r.lo, hi=r.hi, n_drugs_source=int(r.n_drugs),
                       se_approx=round(se, 5), drug_sd_approx=round(sd, 4), delta_basis=delta_name,
                       delta=round(delta, 5), n_drugs_for_80pct_power=n80)
            for n in [int(x) for x in a.n_candidates.split(",")]:
                row[f"power_n{n}"] = round(phi(delta / (sd / math.sqrt(n)) - Z), 3)
                row[f"ci95_halfwidth_n{n}"] = round(Z * sd / math.sqrt(n), 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, sep="\t", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
