"""Freeze: selected configurations, run ids, checkpoint hashes, KG hashes, baseline selection and pre-specified contrasts,
written before the R7 final evaluation opens any test delta."""
import argparse, json, os
from .common import runs, pman, dump, sha256, CFG, MAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help='JSON {"name": ["kind/run_id", ...]}')
    ap.add_argument("--xpert", default="[]", help='JSON list of XPert run ids')
    ap.add_argument("--contrasts", required=True, help="JSON list of [a, b]")
    ap.add_argument("--stress", required=True, help="JSON list of model names for the annotation stress test")
    a = ap.parse_args()
    models = {}
    for name, lst in json.loads(a.models).items():
        models[name] = []
        for kr in lst:
            kind, rid = kr.split("/")
            rec = json.load(open(runs(kind, rid, "record.json")))
            ck = runs(kind, rid, "best.pt")
            models[name].append(dict(kind=kind, run_id=rid, seed=rec["config"]["seed"], config=rec["config"], best=rec["best"],
                                     checkpoint=ck, checkpoint_sha256=sha256(ck)))
    xp = []
    for rid in json.loads(a.xpert):
        d = runs("xpert", rid, "expt")
        ck = [f for f in os.listdir(d) if f.endswith("_fold_early_stop.pth")][0]
        xp.append(dict(run_id=rid, checkpoint=os.path.join(d, ck), checkpoint_sha256=sha256(os.path.join(d, ck))))
    kgc = json.load(open(os.path.join(MAN, "kg_card.json")))
    man = dict(frozen_commit=os.environ.get("R7_COMMIT"), protocol_sha256=sha256(CFG), kg_files=kgc["files"],
               baseline_selection=json.load(open(os.path.join(MAN, "baseline_selection.json"))),
               models=models, xpert=xp, contrasts=json.loads(a.contrasts), stress_models=json.loads(a.stress),
               stress_rule=dict(fractions=[0.0, 0.5, 1.0], unit="ChEMBL target entries (tid blocks)", seed=20261117),
               note="no R7 job has read a test delta before this manifest is committed; R4 opened the test once (job 25374)")
    dump(man, pman("freeze_manifest.json"))
    print(json.dumps({k: [r["run_id"] for r in v] for k, v in models.items()}, indent=1), xp)


if __name__ == "__main__":
    main()
