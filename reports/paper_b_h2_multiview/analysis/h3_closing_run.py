"""Execute the H3-v2 post-unsealing closing audit after both machine replays pass."""
from datetime import datetime, timezone
import sys

from h3_closing_inputs import *
from h3_closing_native import (
    BASELINES, EXTERNAL, H3_MODELS, a1, a1_panels, family_ndcg, local_roster, split_overlap,
)

STAGE = HOME / "tmp/h3v2_closing"
MACHINE_TABLES = {"h3":(H3 / "score/H3V2_METRICS.tsv",H3 / "score/score_receipt.json",H3_SCORE),
                  "e5":(E5 / "score/E5_METRICS.tsv",E5 / "score/score_receipt.json",E5_SCORE)}
COUNTS = {
    "H3V2_VS_H2V2_E5_METRICS.tsv": None,
    "H3V2_VS_H2V2_E5_PAIRED_95CI.tsv": None,
    "H3V2_HELD68_VS_H2_PAIRED_95CI.tsv": None,
    "H3V2_FAMILY980_SPLIT_OVERLAP.tsv": 15,
    "H3V2_FAMILY980_SENSITIVITY.tsv": None,
    "H3V2_PARTS_AND_NULL_AUDIT.tsv": None,
}


def gate():
    evidence = {}
    for key,(table,official,digest) in MACHINE_TABLES.items():
        replay = read(STAGE / f"{key}_replay.json")
        entry = replay["comparison"]
        assert replay["status"] == "PASS" and entry["max_abs_diff"] <= 1e-9
        old = read(official,digest)
        hashes = old["tables"] if key=="h3" else old["output_sha256"]
        assert replay["file"] == table.name and replay["input_sha256"][str(table)] == hashes[table.name]
        assert pin(table,hashes[table.name])
        evidence[table.name] = {"comparison":entry,"slurm":replay["slurm"],
                                "input_sha256":replay["input_sha256"]}
    return evidence


def save(table, name):
    path = STAGE / name
    tmp = path.with_suffix(".writing")
    table.to_csv(tmp,sep="\t",index=False,na_rep="NA",float_format="%.17g")
    tmp.replace(path)
    assert len(table)>0
    return len(table)


def budget(lock):
    ledger = read(H3 / "reports-final-delivery/inner_selection_ledger.json")
    exact_lock = read(H3 / "reports-final-delivery/final_lock.json")
    assert lock == exact_lock
    d = {"frozen_from_boundary_1":0,"arms":0,"arms_near_iteration_cap_94_percent":0,
         "unprobed_neighbors":sum(len(opts) for families in ledger["open_neighbors"].values()
                                   for opts in families.values())}
    for b in lock["splits"].values():
        for v in b["families"].values():
            cfg = v["config"]
            cap = cfg["trees"] if cfg["backend"]=="gbdt" else cfg["epochs"]
            d["arms"] += 1
            d["frozen_from_boundary_1"] += v["stage"]=="boundary-1"
            d["arms_near_iteration_cap_94_percent"] += v["max_budget_pressure_iteration"]>=.94*cap
    assert tuple(d.values()) == (72,75,13,719),d
    return d


def run(part):
    require_slurm()
    verified = gate()
    print("Both archived metric replays passed; beginning",part,flush=True)
    score, manifest, lock = h3_metadata()
    external = None if part=="held" else external_inputs()
    groups = registry()
    if part == "native":
        parts = ("drug-0","drug-1","drug-2","external")
    elif part == "held":
        parts = [s for s in manifest["split_support"] if not s.startswith(("drug-","cell-"))]
    else:
        parts = [s for s in manifest["split_support"] if s.startswith(("drug-","cell-"))] + ["external"]
    frames = {}
    for s in parts:
        frames[s] = h3_frame(s,score,manifest,lock,external[0] if s=="external" else None)
        print("Frozen forecast roster verified",s,len(frames[s]),flush=True)
    output, extra = {}, {}
    if part == "native":
        from h3_closing_held import cell_exclusion

        panels, parity, ext_groups = a1_panels(frames,external)
        extra["row_and_truth_parity"] = parity
        print("A1 exact row and truth parity",parity,flush=True)
        extra["boundary_audit"] = budget(lock)
        points, cis, units = a1(panels,groups,ext_groups)
        output["H3V2_VS_H2V2_E5_METRICS.tsv"] = save(points,"H3V2_VS_H2V2_E5_METRICS.tsv")
        output["H3V2_VS_H2V2_E5_PAIRED_95CI.tsv"] = save(cis,"H3V2_VS_H2V2_E5_PAIRED_95CI.tsv")
        for s in ("cell-0","cell-1","cell-2"):
            frames[s] = h3_frame(s,score,manifest,lock)
            print("Frozen forecast roster verified",s,len(frames[s]),flush=True)
        roster = local_roster()
        overlap, masks = split_overlap(frames,manifest,roster)
        output["H3V2_FAMILY980_SPLIT_OVERLAP.tsv"] = save(overlap,"H3V2_FAMILY980_SPLIT_OVERLAP.tsv")
        sensitive = []
        for scope,cell_units in units.items():
            baselines = tuple(k for k in BASELINES if k in set(cell_units.arm))
            sensitive.extend(family_ndcg(cell_units,roster,scope,baselines))
        sensitive.extend(cell_exclusion(frames,masks,groups,roster))
        output["family980_native.tsv"] = save(pd.DataFrame(sensitive),"family980_native.tsv")
    elif part == "held":
        from h3_closing_held import assemble, held_exclusion, held_points_and_ci

        old = assemble(frames)
        assert len(old)==45428
        support = {"observed_parents":int(old[KEY].nunique()),
                   "parents_by_screen":old.groupby("source_id")[KEY].nunique().to_dict()}
        print("Held/core68 observed support",support,flush=True)
        held = held_points_and_ci(old)
        output["H3V2_HELD68_VS_H2_PAIRED_95CI.tsv"] = save(held,"H3V2_HELD68_VS_H2_PAIRED_95CI.tsv")
        _, masks = split_overlap(frames,{"split_support":{s:manifest["split_support"][s] for s in parts}},local_roster())
        new = assemble(frames,masks=masks)
        output["family980_held.tsv"] = save(pd.DataFrame(held_exclusion(old,new)),"family980_held.tsv")
        extra["held68_scope"] = {"baseline":"Archived H3-v2 H2 kernel on each held/core68 split",
                                 "rows":len(old),"retained_rows_after_family_exclusion":len(new),**support,
                                 "E5_v2_and_E1_matched_predictions_available":False}
    else:
        from h3_closing_null import audit_null

        table, extra = audit_null(frames,lock,external,groups)
        output["H3V2_PARTS_AND_NULL_AUDIT.tsv"] = save(table,"H3V2_PARTS_AND_NULL_AUDIT.tsv")
    note = {"stage":part,"slurm":job_meta(),"finished_utc":datetime.now(timezone.utc).isoformat(),
            "rows":output,"output_sha256":{name:sha256(STAGE/name) for name in output},
            "input_sha256":INPUTS,"extra":extra}
    publish_json(STAGE/f"{part}_done.json",note)
    print("COMPLETED",part,output,flush=True)


if __name__=="__main__":
    run(sys.argv[1])
