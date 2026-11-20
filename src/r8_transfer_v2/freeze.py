"""Freeze F2: source decisions of E1 are final; writes manifests/r8_transfer_v2/e1_source_freeze.json (server public copy) with the
final_inner constants, the code commit and the fixed HCC contrast list. HCC is read only after this file is committed."""
import argparse, datetime, json, os
from .common import pub_man, sha256, dump, git_state

PLAN = [("J3", "C1"), ("J3", "C1J1"), ("C1J1", "C1"), ("C1J1_full", "C1J1"), ("J2", "J1"), ("J1", "C0"), ("J2", "C1"), ("C0", "anchor_matched"), ("C1", "C0"),
        ("C1", "anchor_matched")]
MISS = [("J3", "C1"), ("J3", "C1J1"), ("C1J1", "C1"), ("J2", "C1"), ("J1", "C0")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-select", required=True)
    ap.add_argument("--decisions", required=True)
    a = ap.parse_args()
    cc = os.path.join(a.e1_select, "context_constants.json")
    c = json.load(open(cc))
    assert c["partition"] == "final_inner"
    pairs = [(f"{x}|planned", f"{y}|planned") for x, y in PLAN]
    for ct in ("missing_reselected", "missing_zero_overlap_fallback"):
        pairs += [(f"{x}|{ct}", f"{y}|{ct}") for x, y in MISS]
    pairs += [(f"{m}|missing_reselected", f"{m}|missing_zero_overlap_fallback") for m in ("J1", "C1J1", "J2", "J3")]
    pairs += [(f"{m}|missing_reselected", f"{m}|planned") for m in ("J1", "C1J1", "J2", "J3")]
    head, dirty = git_state()
    assert not dirty
    fz = dict(track="r8_transfer_v2", frozen_on=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), code_commit=head,
              context_constants=dict(path=os.path.abspath(cc), sha256=sha256(cc)), base_constants=dict(path=c["base_constants"], sha256=c["base_constants_sha256"]),
              source_decisions=dict(path=a.decisions, sha256=sha256(a.decisions)), hcc_contrasts=pairs,
              primary_hcc_contrasts=[["J3|planned", "C1|planned"], ["J3|planned", "C1J1|planned"], ["C1J1|planned", "C1|planned"]],
              rule="HCC is an already-exposed benchmark (studies B, R2, R8, R8 follow-up v1): frozen follow-up evaluation, one pass; no candidate is changed because of an HCC value",
              hcc_expression_values_read_in_v2_before_freeze="only for the P0 correction rows (anchor_matched / K-consistency replay with fixed or source-selected constants); no E1 arm scored on HCC")
    dump(fz, pub_man("e1_source_freeze.json"))
    print(json.dumps(fz, indent=1))


if __name__ == "__main__":
    main()
