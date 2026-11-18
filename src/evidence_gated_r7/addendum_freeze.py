"""Addendum A-2 freeze: XPert-ECFP checkpoints (sha256), reused frozen predictions, pre-specified contrasts."""
import json, os
from .common import runs, pman, dump, sha256


def main():
    xs = []
    for s in (42, 43, 44):
        rid = f"xpe_e100_s{s}"
        d = runs("xpert", rid, "expt")
        ck = [f for f in os.listdir(d) if f.endswith("_fold_early_stop.pth")][0]
        xs.append(dict(run_id=rid, checkpoint=os.path.join(d, ck), checkpoint_sha256=sha256(os.path.join(d, ck))))
    man = dict(frozen_commit=os.environ.get("R7_COMMIT"), main_freeze="a7ab705", xpert_ecfp=xs,
               reused_models=["chem_only", "B11", "xpert", "ecfp_x0_ridge"],
               contrasts=[["xpert_ecfp", "chem_only"], ["xpert_ecfp", "B11"], ["xpert_ecfp", "xpert"], ["xpert_ecfp", "ecfp_x0_ridge"]],
               note="post-freeze comparator added after the main test evaluation (DECISIONS A-2); R7 verdicts unchanged")
    dump(man, pman("addendum_freeze.json"))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
