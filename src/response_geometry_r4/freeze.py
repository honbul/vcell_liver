"""Freeze: record the selected configurations, run ids, checkpoint hashes and baseline selections before the test opens."""
import argparse, json, os
from .common import runs, public, dump, sha256, CFG, MAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help='JSON: {"name": ["arm/run_id", ...], ...}')
    ap.add_argument("--primary_contrasts", required=True, help='JSON list of [a, b] model-name pairs')
    a = ap.parse_args()
    models = json.loads(a.models)
    out = {}
    for name, lst in models.items():
        out[name] = []
        for ar in lst:
            arm, rid = ar.split("/")
            rec = json.load(open(runs(arm, rid, "record.json")))
            ck = runs(arm, rid, "best.pt")
            out[name].append(dict(arm=arm, run_id=rid, seed=rec["config"]["seed"], config=rec["config"], best=rec["best"], checkpoint=ck, checkpoint_sha256=sha256(ck)))
    man = dict(frozen_commit=os.environ.get("R4_COMMIT"), protocol_sha256=sha256(CFG),
               data_contract_sha256=sha256(os.path.join(MAN, "data_contract.json")),
               baseline_selection=json.load(open(os.path.join(MAN, "baseline_selection.json"))),
               models=out, contrasts=json.loads(a.primary_contrasts),
               note="test unit deltas (unit_delta_test.npy) have not been opened by any R4 job before this manifest is committed")
    dump(man, public("manifests", "response_geometry_r4", "freeze_manifest.json"))
    print(json.dumps({k: [r["run_id"] for r in v] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
