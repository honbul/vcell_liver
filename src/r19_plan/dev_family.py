"""Apply the four-track DEV Holm adjustment to the fixed A/B primary slots."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json


FAMILY = (
    ("T.A", "T3", "T1", None, "No eligible A target variance design"),
    ("T.B", "T3", "T1", "t123/primary_stats/primary_p.tsv", ""),
    ("X.A", "X2", "X1", "x_a/score/paired.tsv", ""),
    ("X.B", "X2", "X1", None, "X3 donor weighting was a separate registered comparison"),
    ("S.A", "S2", "B_R17", "sl_a/score/paired.tsv", ""),
    ("S.B", "S2", "B_R17", "sl_b/score/paired.tsv", ""),
    ("L.A", "L1", "B_R17", "sl_a/score/paired.tsv", ""),
    ("L.B", "L1", "B_R17", "sl_b/score/paired.tsv", ""),
)
ALPHA = .01


def adjustment(p_values):
    index = np.argsort(p_values, kind="stable")
    padded = np.minimum(1., np.maximum.accumulate(p_values[index] * np.arange(len(index), 0, -1)))
    output = np.empty_like(padded)
    output[index] = padded
    return output


def primary(data, slot, arm, baseline, name):
    path = data/name
    stamp = read_json(path.with_name("receipt.json"))
    if sha256(path) != stamp["files"][path.name]:
        raise ValueError("A registered primary score table changed after scoring")
    rows = pd.read_csv(path, sep="\t")
    if slot != "T.B":
        rows = rows.loc[rows.primary.eq(True) & rows.arm.eq(arm) & rows.reference.eq(baseline)]
    order = rows.set_index("metric", verify_integrity=True)
    if set(order.index) != {"rmse", "r2_pooled"}:
        raise ValueError(f"{slot} requires both registered endpoints")
    p = order.one_sided_p.to_numpy()
    joint = order.joint_p_unadjusted.to_numpy()
    if not np.isfinite(p).all() or not np.all((p > 0) & (p <= 1)) or not np.allclose(joint, p.max()):
        raise ValueError(f"{slot} joint p must be the maximum of the two tested directions")
    return (float(order.loc["rmse", "one_sided_p"]), float(order.loc["r2_pooled", "one_sided_p"]), float(p.max()))


def run(data, out):
    if out.exists():
        raise FileExistsError(out)
    result, files = [], {}
    for slot, arm, baseline, name, reason in FAMILY:
        p_rmse, p_r2, p_joint = primary(data, slot, arm, baseline, name) if name else (None, None, None)
        if name:
            files[name] = sha256(data/name)
        result.append(dict(slot=slot, arm=arm, reference=baseline, status="TESTED" if name else "UNTESTED",
                           p_rmse=p_rmse, p_r2=p_r2, p_joint=p_joint, holm_input=1. if p_joint is None else p_joint,
                           source=name or "NA", reason=reason or "NA"))
    summary = pd.DataFrame(result)
    summary["p_holm"] = adjustment(summary.holm_input.to_numpy())
    summary["reject_at_0.01"] = summary.status.eq("TESTED") & summary.p_holm.le(ALPHA)
    out.mkdir(parents=True)
    summary.to_csv(out/"primary.tsv", sep="\t", index=False, na_rep="NA")
    write_json(out/"receipt.json", dict(alpha=ALPHA, n_slots=len(FAMILY), slots_tested=summary.status.eq("TESTED").sum().item(),
        untested_holm_input=1., input_files=files, code_sha256=sha256(__file__), primary_sha256=sha256(out/"primary.tsv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
