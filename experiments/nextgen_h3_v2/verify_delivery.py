"""Verify the pushed report package after the one conditional score has finished."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

REQUIRED = {
    "PLAN_LOCK.md","DATA_SUPPORT_AND_EXPOSURE.md","ROUND1_TO_2_FAMILY_TUNING.md","RESULTS.md",
    "H3V2_METRICS.tsv","H3V2_PAIRED_95CI.tsv","H3V2_FAMILY_LEARNING_CURVES.tsv","H3V2_PANCANCER_BY_TISSUE.tsv",
    "model_card.md","verification.json","score_receipt.json","SLURM_JOBS.tsv",
}
FAMILIES = {"gbdt","film","attention","gnn","hybrid"}
AXES = {"spearman":.01,"interaction_spearman":.02,"ndcg_at_5":.015}
COHORTS = {"GDSC1":39545,"LIMORE":3400,"LICOB":2888}


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def rows(file):
    with file.open() as handle:
        return list(csv.DictReader(handle,delimiter="\t"))


def verify(root):
    for name in REQUIRED:
        if not (root/name).is_file() or (root/name).stat().st_size==0:
            raise ValueError(f"Missing or empty required deliverable: {name}")
    verify=json.loads((root/"verification.json").read_text())
    receipt=json.loads((root/"score_receipt.json").read_text())
    lock=json.loads((root/"final_lock.json").read_text())
    if verify["outcomes_opened_by_independent_verification"] or verify["score_passes"]!=1 or receipt["score_passes"]!=1:
        raise ValueError("The audit cannot prove one-time score use")
    if verify["score_sha256"]!=digest(root/"score_receipt.json") or verify["frozen_sha256"]!=digest(root/"final_lock.json"):
        raise ValueError("Score/selection evidence differs from the published release")
    if verify["step0_sha256"]!=digest(root/"step0_verification.json"):
        raise ValueError("Feature parity evidence was not exported as verified")
    if verify["partition_manifest_sha256"]!=digest(root/"split_manifest.json"):
        raise ValueError("Feature support changed")
    for name,sha in verify["deliverables"].items():
        if digest(root/name)!=sha:
            raise ValueError(f"Verified artifact changed while exporting {name}")
    if {key:receipt["parts"][key] for key in COHORTS}!=COHORTS or lock["outer_labels_opened"]:
        raise ValueError("Registered technical cohort support or outcome handling changed")
    for label,line in zip(("가설","달라진 점","결과","판정","다음 결정"),(root/"RESULTS.md").read_text().splitlines()[:5]):
        if not line.startswith(f"- **{label}**"):
            raise ValueError("First five summary lines must precede the results tables")
    primary={(r["model"],r["seed"]) for r in rows(root/"H3V2_METRICS.tsv")
             if (r["scope"],r["screen"],r["support"],r["metric"])==("drug","macro","all","ndcg_at_5")}
    if {(model,seed) for model in FAMILIES|{"Champion"} for seed in ("11711","11729","11743","mean")} - primary:
        raise ValueError("At least one family or seed was lost before scoring")
    if {r["tissue"] for r in rows(root/"H3V2_PANCANCER_BY_TISSUE.tsv")}!=set(receipt["tissue_roster"]):
        raise ValueError("Incomplete tissue stratification")
    fits=rows(root/"H3V2_FAMILY_LEARNING_CURVES.tsv")
    if {(f,res) for f in FAMILIES for res in ("r1","r2")} - {(r["family"],r["round"]) for r in fits}:
        raise ValueError("A model family is missing measured two-round learning curves")
    checks=[r for r in rows(root/"H3V2_PAIRED_95CI.tsv") if r["baseline"]=="H2" and r["support"]=="all"]
    for model,outcome in receipt["verdicts"].items():
        evidence={(r["scope"],r["screen"],r["metric"]):r for r in checks if r["candidate"]==model}
        expected={}
        for axis,margin in AXES.items():
            here=evidence["drug","macro",axis]
            other=[evidence["drug","macro",name] for name in AXES if name!=axis]
            expected[axis]={
                "minimum_improvement_and_ci":float(here["delta"])>=margin and float(here["lower"])>0,
                "other_two_axes_protected":all(float(r["lower"])>=-.005 for r in other),
                "two_screens_improved":sum(float(evidence["drug",s,axis]["delta"])>0
                                           for s in ("CTRPv2","GDSC2","PRISM20Q2"))>=2,
                "gdsc1_three_axes_protected":all(float(evidence["GDSC1","macro",m]["lower"])>=-.005 for m in AXES),
            }
        if outcome["by_axis"]!=expected or (outcome["verdict"]=="GO-MODEL")!=any(all(x.values()) for x in expected.values()):
            raise ValueError(f"The report did not enforce the registered gates: {model}")
    print(f"Verified {len(REQUIRED)} required reports, five comparable model families and 28 tissue strata")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports",type=Path)
    verify(p.parse_args().reports)
