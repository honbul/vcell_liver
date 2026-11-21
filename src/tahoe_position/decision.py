"""Apply the prespecified Tahoe steps 2–3 decision in protocol order."""

import argparse
from pathlib import Path

from src.r11_chemical_probe.common import read_json, sha256, write_json


DOSE_GATES = (
    (0.5, "all", "keep_0.5_uM_all"),
    (0.5, "strong", "restrict_to_0.5_uM_strong"),
    (5.0, "all", "restrict_to_5_uM_all"),
)


def gate(option, groups, interval, cutoff):
    lower = interval[0] if interval is not None else None
    if groups < 10:
        reason = "insufficient_groups"
    elif lower is None:
        reason = "missing_interval"
    elif lower <= cutoff:
        reason = "lower_bound_does_not_exceed_cutoff"
    else:
        reason = "passed"
    return dict(option=option, groups=groups, ci95=interval, minimum_groups=10,
                required_lower_bound_exclusive=cutoff, reason=reason)


def evaluate(repeats, moa):
    same_dose = {(r["target_dose"], r["stratum"]): r for r in repeats if r["target_dose"] == r["source_dose"]}
    gates = []
    for dose, stratum, option in DOSE_GATES:
        result = same_dose[dose, stratum]
        gates.append(gate(option, result["groups"], result.get("gain025_improvement_ci95"), 1.0))
    gates.append(gate("choose_fine_moa_search", moa["groups"], moa.get("auc_ci95"), 0.55))
    decision = next((g["option"] for g in gates if g["reason"] == "passed"), "move_to_independent_position_replicates")
    return dict(decision=decision, gates=gates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "ceilings", "moa", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    ceilings, moa = read_json(args.ceilings), read_json(args.moa)
    protocol_hash = sha256(args.protocol)
    if {ceilings["protocol_sha256"], moa["protocol_sha256"]} != {protocol_hash}:
        raise ValueError("Decision inputs do not match the prescoring protocol")
    expected = ceilings["inputs"]["common_receipt_sha256"], ceilings["inputs"]["moa_sha256"]
    if (moa["common_receipt_sha256"], moa["moa_sha256"]) != expected:
        raise ValueError("Repeat and MoA results use different frozen targets or annotations")
    write_json(args.out, dict(**evaluate(ceilings["repeats"], moa["pooled"]), protocol_sha256=protocol_hash,
                              input_sha256={str(path): sha256(path) for path in (args.ceilings, args.moa)}))


if __name__ == "__main__":
    main()
