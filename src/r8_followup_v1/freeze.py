"""Write the follow-up freeze manifest (frozen candidate, baselines, constants and nuisance inputs with SHA-256).
The manifest must be committed before final_eval.py can read the already-seen panels.

usage: python -m src.r8_followup_v1.freeze --candidate C1_iid --baselines a b c --constants <final_inner constants.json>
                                           --pairs <d1_plate_pairs.tsv> [--jconstants <final_inner branch_constants.json>] --code-commit <sha> --rule "<text>"
"""
import argparse, datetime, json, os

from .common import MAN, sha256, dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--extra-contrasts", nargs="*", default=[], help="a:b pairs")
    ap.add_argument("--constants", required=True)
    ap.add_argument("--jconstants", default="")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--rule", required=True)
    a = ap.parse_args()
    c = json.load(open(a.constants))
    assert c["partition"] == "final_inner"
    fz = dict(track="r8_followup_v1", frozen_on=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), code_commit=a.code_commit,
              candidate=dict(method=a.candidate, selection_rule=a.rule), baselines=a.baselines,
              extra_contrasts=[x.split(":") for x in a.extra_contrasts],
              constants=dict(path=os.path.abspath(a.constants), sha256=sha256(a.constants)), pairs=dict(path=os.path.abspath(a.pairs), sha256=sha256(a.pairs)),
              nuisance="global partition: original feature basis (eigen-signs of a refit differ only), covariance V0-V3 and kernel scales refit on all 115 source-train families",
              panels=dict(confirmation="9 confirmation families, support seeds 1-3, K = 8/16/32", matched="confirmation matched-plate E1/E2 (<= 6 pairs, seeds 1-3)",
                          hcc="HUH7 / JHH5 / JHH7, original frozen split lists (study B splits), split seeds 1-3, K = 8/16/32"),
              target_validation_labels=0, hcc_expression_values_read_in_this_followup_before_freeze=False,
              confirmation_values_read_in_this_followup_before_freeze=False,
              exposure="existing-benchmark follow-up evaluation: HCC scored in studies B, R2, R8; confirmation families scored in R8")
    if a.jconstants:
        j = json.load(open(a.jconstants))
        assert j["partition"] == "final_inner"
        fz["jconstants"] = dict(path=os.path.abspath(a.jconstants), sha256=sha256(a.jconstants))
    dump(fz, os.path.join(MAN, "followup_freeze.json"))
    print(json.dumps(fz, indent=1))


if __name__ == "__main__":
    main()
