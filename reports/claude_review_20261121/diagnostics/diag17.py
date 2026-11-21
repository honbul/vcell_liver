"""diag17 (Claude review, EXPLORATORY, not official).

R16's fair comparison chose each known VC's output gain on the inner validation
fold. That fold's truth uses the SAME DMSO role as the fit, while the outer
scoring uses the OPPOSITE role. Codex raised, without testing, that this could
reward predictions that track the fit role's control draw and so pick too large
a gain.

This tests that mechanism on data already frozen: for the OUTER rows (where the
predictions exist), score every model under both truths,
   truth_same = treated - control(fit role)      [same convention as inner selection]
   truth_opp  = treated - control(other role)    [the official scoring]
and report the gain each truth prefers, plus cos(estimate, control difference).

If the same-role truth prefers a larger gain, the inner selection is biased
upward and the gain-matched comparison in R16 handicaps the comparators.
"""
import json
import sys

import numpy as np
import pandas as pd

from src.r15_lincs.design import Bank, fold_id
from src.r15_lincs.estimators import GAINS, loss

R15 = "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r15_lincs/v1"
PROTOCOL = "reports/r15_lincs/PROTOCOL.md"
OUT = sys.argv[1]
VCS = ("ChemBERTa_R11", "FiLM_R12", "Concat_R12", "chemCPA_P0", "TranSiGen_Tahoe", "XPert_morgan",
       "PRnet-well-adaptation")


def outer_rows(bank, fold):
    rows = bank.target()
    assigned = rows.parent.map(lambda p: fold_id(p, "R15-outer", 3))
    extra = bank.auxiliary()
    extra = extra.loc[extra.parent.map(lambda p: fold_id(p, "R15-outer", 3)).eq(fold)]
    return pd.concat((rows.loc[assigned.eq(fold)], extra), ignore_index=True)


def main():
    report = {}
    for scenario in "AB":
        bank = Bank(f"{R15}/cache", PROTOCOL, scenario)
        acc = {}
        for fold in range(3):
            for role in "AB":
                rows = outer_rows(bank, fold)
                mask = (rows.cell_id.eq("HEPG2") & rows.study.eq("GSE92742") & rows.hours.eq(6)
                        & rows.quality & ~rows.prior_exact).to_numpy()
                keep = rows.loc[mask]
                other = "B" if role == "A" else "A"
                same = bank.response(rows, role)[mask]
                opp = bank.response(rows, other)[mask]
                delta = same-opp  # control(other) - control(role): pure vehicle noise
                root = f"{R15}/predict/{scenario}/f{fold}-{role}"
                with np.load(f"{root}/R15/forecast.npz", allow_pickle=False) as z:
                    mine = {k: z[k] for k in z.files}
                anchor = (mine["B_pos_cell"] if scenario == "B" else mine["B_pos"])[mask]
                est = {"R15": mine["R15"][mask]-anchor}
                for name in VCS:
                    with np.load(f"{root}/{name}/forecast.npz", allow_pickle=False) as z:
                        est[name] = z[name][mask]-anchor
                for name, e in est.items():
                    a = acc.setdefault(name, {"same": {g: [] for g in GAINS}, "opp": {g: [] for g in GAINS},
                                              "cos": [0., 0., 0.]})
                    for g in GAINS:
                        a["same"][g].append(loss(keep, same, anchor+g*e))
                        a["opp"][g].append(loss(keep, opp, anchor+g*e))
                    a["cos"][0] += float((e*delta).sum())
                    a["cos"][1] += float(np.square(e).sum())
                    a["cos"][2] += float(np.square(delta).sum())
                print(f"{scenario} f{fold}-{role}: {mask.sum()} rows", flush=True)
        out = {}
        print(f"\n=== scenario {scenario}: gain preferred by each truth convention")
        print(f"{'model':24s} {'same-role':>10s} {'opp-role':>10s}   cos(est, control diff)")
        for name, a in acc.items():
            g_same = min(GAINS, key=lambda g: (round(float(np.mean(a['same'][g])), 9), g))
            g_opp = min(GAINS, key=lambda g: (round(float(np.mean(a['opp'][g])), 9), g))
            c = a["cos"][0]/np.sqrt(a["cos"][1]*a["cos"][2])
            out[name] = dict(gain_same_role=g_same, gain_opposite_role=g_opp, cosine_with_control_diff=float(c),
                             loss_same={str(g): float(np.mean(v)) for g, v in a["same"].items()},
                             loss_opp={str(g): float(np.mean(v)) for g, v in a["opp"].items()})
            print(f"{name:24s} {g_same:10.3f} {g_opp:10.3f}   {c:+.4f}")
        report[scenario] = out
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
