"""Consume locked follow-up forecasts and produce the one-pass native outcomes ledger."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.run import META, read_json, subfold
from nextgen_h2.score import adult_licob, external_labels
from vc_build.external import native_sensitivity
from vc_build.io import publish_json, sha256

from .common import V1_ASSETS, native_rows, panel_hash, registered
from .e1 import E1_ARMS, arms, rank_invariance
from .e3 import ARMS, MODES, REFERENCES, column, read_contract
from .inference import DRAWS, measure, summarize

OBSERVED = "common_REF-EXCLUDED_LINCS"
E1_DIAGNOSTIC = "LIMORE-v2-lambda4"


def scope(panel):
    return {"rows": len(panel), "parents": panel[KEY].nunique(), "cells": panel.cell_id.nunique(),
            "lineages": panel.lineage_group.nunique()}


def paired(a, b):
    return f"{a} - {b}", {a: 1., b: -1.}


def e1_pairs(arms_expected):
    contrasts = [paired(a, b) for a,b in (
        ("F11","F00"), ("F10","F00"), ("F01","F00"), ("F11","F01"), ("F11","F10"),
        ("F01-only","F00"), ("F11","F01-only"), ("Cchem-int","F00"), ("F11","Cchem-int"),
        ("F01-only","Cchem-int"),
    )]
    contrasts.extend((
        ("kernel_ShAP",{"F10":.5,"F00":-.5,"F11":.5,"F01":-.5}),
        ("intercept_ShAP",{"F01":.5,"F00":-.5,"F11":.5,"F10":-.5}),
        ("factorial_nonadditivity",{"F11":1,"F00":1,"F10":-1,"F01":-1}),
    ))
    if E1_DIAGNOSTIC in arms_expected:
        contrasts += [paired(E1_DIAGNOSTIC, "F11"), paired(E1_DIAGNOSTIC,"F00")]
    return contrasts


def output(path, stage, results):
    points, intervals = ([r for points, _ in results for r in points], [r for _, cis in results for r in cis])
    if not points or not intervals:
        raise ValueError("Insufficient metric support across every cohort")
    p, d = pd.DataFrame(points), pd.DataFrame(intervals)
    p.to_csv(path/(stage+"_METRICS.tsv"), sep="\t", index=False, na_rep="NA")
    d.to_csv(path/(stage+"_PAIRED_95CI.tsv"), sep="\t", index=False, na_rep="NA")
    return p, d


def e1_validate(root):
    source, components = root/"e1-forecast",root/"e1-components"
    choice_path = root/"e1-chem-selection/receipt.json"
    choice = read_json(choice_path)
    assert choice["outer_outcome_rows_loaded"] == 0 and len(choice["candidates"]) == 9
    results, evidence = [], []
    for part in range(4):
        folder = source/str(part)
        proof = read_json(folder/"receipt.json")
        factors = read_json(components/str(part)/"receipt.json")
        if (proof["query_outcomes_loaded"] != 0 or factors["query_outcomes_loaded"] != 0
                or proof["chemistry_selection_sha256"] != sha256(choice_path)
                or proof["component_receipt_sha256"] != sha256(components/str(part)/"receipt.json")
                or max(factors["F00_max_abs_diff"], factors["F11_max_abs_diff"]) >= 1e-6):
            raise ValueError("E1 must first reproduce both originals and freeze the chemistry policy")
        frame = parquet_lock(folder/"forecast.parquet", proof["forecast_sha256"])
        assert panel_hash(frame) == factors["roster_sha256"] and len(frame) == factors["rows"]
        assert np.isfinite(frame[list(E1_ARMS)]).to_numpy().all()
        assert all(np.array_equal(frame[name], value) for name, value in arms(frame).items())
        results.append(frame)
        evidence.append(factors | {"forecast_sha256": proof["forecast_sha256"],
                                   "forecast_receipt_sha256": sha256(folder/"receipt.json")})
    assert [len(f) for f in results] == [218445,232324,220061,46213]
    return results, evidence, choice


def e1_scopes(query, groups, name):
    mixed = [(f, q.assign(zero=0.)) for f,q in query]
    diag = name == "LIMORE"
    candidates = E1_ARMS+(E1_DIAGNOSTIC,) if diag else E1_ARMS
    comps = {"F00":"r0", "F01":"r0", "F10":"r1", "F11":"r1",
             "F01-only":"zero", "Cchem-int":"rc"}
    results, support, verified = [], {}, 0
    for label, panels in (
        ("all", mixed), ("LINCS-observed", [(f,q[q.lincs_observed]) for f,q in mixed]),
        ("LINCS-unobserved", [(f,q[~q.lincs_observed]) for f,q in mixed]),
    ):
        supported = pd.concat([q for _,q in panels], ignore_index=True)
        support[label] = scope(supported)
        if supported.empty:
            continue
        for _, frame in panels:
            verified += rank_invariance(frame, {k: frame[k].to_numpy() for k in candidates})
        blocks = measure(panels, groups, candidates, components=comps)
        rs = summarize(blocks, {"cohort":name,"support":label}, {arm:{"arm":arm} for arm in candidates},
                       e1_pairs(candidates))
        results.append(rs)
        for row in rs[1]:
            if row["metric"] == "spearman" and row["contrast"] in ("F01 - F00","F11 - F10"):
                if any(row[n] != 0. for n in ("delta","lo95","hi95")):
                    raise ValueError("Common drug intercept changed a scored within-drug Spearman")
    return results, support, verified


def e1_score(root, out):
    panels, prediction_proof, chemistry = e1_validate(root)
    registry, _, _, assets = registered()
    out.mkdir(exist_ok=False, parents=True)
    publish_json(out/"e1_outcome_access.json",{"score_passes":1,"forecasts":prediction_proof})
    native = []
    for i, q in enumerate(panels[:3]):
        value = native_rows(subfold(registry,i)[1], labels=True)
        if not value[META].equals(q[META]):
            raise ValueError("Changed E1 native truth registration")
        native.append((i, q.assign(truth=native_sensitivity(value))))
    results, support, verified = {}, {}, 0

    def apply(name, block, groups):
        nonlocal verified
        results[name], support[name], count = e1_scopes(block,groups,name)
        verified += count
        print(name, support[name], flush=True)

    apply("DRP",native,registry.scaffold_block)
    truth, licence, ext_groups = external_labels(V1_ASSETS,assets)
    q = panels[3].merge(truth,on="row_id",how="left",validate="one_to_one")
    if set(q.row_id) != set(truth.row_id) or not np.isfinite(q.truth).all():
        raise ValueError("E1 external native outcome coverage changed")
    for name, block in q.groupby("source_id",sort=True):
        if name == "LICOB":
            block = block.drop(columns=E1_DIAGNOSTIC)
            offsets = block.groupby(KEY)[["i0","i1","ci"]].first()
            block = adult_licob(block,licence)
            for key, values in offsets.items():
                block[key] = block[KEY].map(values)
            block = block.assign(**arms(block))
            novel = block.loc[~block.in_source & block.remaining_unspecified_stereo.eq(0)]
            assert len(block.cell_id.unique()) == 38 and block[KEY].nunique() == 76 and novel[KEY].nunique() == 18
            apply("LICOB18",[(0,novel)],ext_groups)
        apply(name,[(0,block)],ext_groups)
    output(out,"E1",[x for cohort in results.values() for x in cohort])
    invariance = {"predictions":prediction_proof,"scored_rank_groups_checked":verified,
        "F01_minus_F00_within_drug_spearman":0.,"F11_minus_F10_within_drug_spearman":0.,
        "tolerance_excludes_cross_device_workspace_forward":1e-6,
        "chemistry_fit_only":chemistry, "source_presplit_unchanged":True,
        "cohort_support":support,"score_passes":1,"native_outcome_access_sha256":sha256(out/"e1_outcome_access.json")}
    publish_json(out/"e1_invariance_receipt.json",invariance)


def e3_validate(root):
    registration, profiles = root/"registration", root/"profiles"
    split, profile = read_contract(registration,profiles)
    paths = root/"e3-forecast"
    policy = root/"e3-selection"
    decision = read_json(policy/"receipt.json")
    if decision["outer_query_outcomes_loaded"] != 0 or decision["policy_sha256"] != sha256(policy/"policy.json"):
        raise ValueError("Double-cold policy not sealed")
    panels, proofs = [], []
    columns = [column(m,r,a) for m in MODES for r in REFERENCES for a in ARMS]
    for f, role in enumerate(split["outer"]):
        area = paths/str(f)
        proof = read_json(area/"receipt.json")
        if (proof["outer_query_outcomes_loaded"] != 0 or proof["nested_proof_sha256"] != sha256(policy/"receipt.json")
                or proof["profile_proof_sha256"] != sha256(profiles/"receipt.json")
                or proof["split_sha256"] != profile["split_sha256"]):
            raise ValueError("Double-cold forecast is not bound to the same release")
        frame = parquet_lock(area/"forecast.parquet", proof["forecast_sha256"])
        if (panel_hash(frame) != role["query_roster_sha256"] or proof["rows"] != role["query_rows"]
                or not np.isfinite(frame[columns]).to_numpy().all()):
            raise ValueError("Double-cold roster or frozen forecasts changed")
        panels.append(frame)
        proofs.append(proof | {"receipt_sha256":sha256(area/"receipt.json")})
    return split, profile, panels, proofs


def e3_policy(d):
    possible = d[(d.cohort.eq("DRP")) & (d.support.eq(OBSERVED)) & (d.context.eq("within_policy"))
                 & d.mode.eq("E3-Nested") & d.reference.eq("REF-EXCLUDED")]
    def effect(contrast, metric, unit):
        found = possible[possible.contrast.eq(contrast) & possible.metric.eq(metric) & possible.resampling.eq(unit)]
        return found.iloc[0] if len(found)==1 else None

    main = effect("H2-v2 - C1","spearman","scaffold_x_lineage")
    retrieval = effect("H2-v2 - C1","ndcg_at_5","lineage")
    c2 = effect("H2-v2 - C2","spearman","scaffold_x_lineage")
    c3 = effect("H2-v2 - C3","spearman","scaffold_x_lineage")
    if any(x is None or x.eligible_screen_folds != 9 for x in (main,retrieval,c2,c3)):
        return {"decision":"INFEASIBLE","evidence":"Fewer than nine eligible primary screen-folds"}
    gates = {"rho_delta_ge_002":bool(main.delta >= .02), "rho_crossed_ci_lower_positive":bool(main.lo95 > 0),
             "C2_crossed_ci_lower_positive":bool(c2.lo95 > 0), "C3_crossed_ci_lower_positive":bool(c3.lo95 > 0),
             "NDCG_lineage_ci_lower_ge_minus_001":bool(retrieval.lo95 >= -.01)}
    return {"decision":"GO-DOUBLE-COLD" if all(gates.values()) else "NO-GO-NEW-CELL",
            "gates":gates}


def e3_scopes(panels, groups, name):
    policy_pairs = []
    for mode in MODES:
        for ref in REFERENCES:
            tests = [("H2-v1","C1"), ("H2-v2","C1"), ("H2-v2","H2-v1"), ("H2-v2","C2"), ("H2-v2","C3")]
            for a,b in tests:
                policy_pairs.append((mode,ref,"within_policy",f"{a} - {b}",
                                     {column(mode,ref,a):1.,column(mode,ref,b):-1.}))
        for a in ("H2-v1","H2-v2"):
            policy_pairs.append((mode,"REF-EXCLUDED vs REF-INCLUDED","reference",f"{a}: EXCLUDED - INCLUDED",
                                 {column(mode,REFERENCES[0],a):1.,column(mode,REFERENCES[1],a):-1.}))
    for ref in REFERENCES:
        for a in ("H2-v1","H2-v2"):
            policy_pairs.append(("E3-Nested vs E3-Fixed",ref,"selection",f"{a}: Nested - Fixed",
                                 {column(MODES[1],ref,a):1.,column(MODES[0],ref,a):-1.}))
    registry = {column(mode,ref,a):{"mode":mode,"reference":ref,"arm":a}
                for mode in MODES for ref in REFERENCES for a in ARMS}
    results, proof = [], {}
    for label, subset in [("all",panels), (OBSERVED,[(f,r[r["REF-EXCLUDED/observed"]]) for f,r in panels])]:
        frame = pd.concat([r for _,r in subset],ignore_index=True)
        proof[label] = scope(frame)
        blocks = measure(subset,groups,registry,crossed=True)
        tags = {"cohort":name,"support":label}
        points, _ = summarize(blocks,tags,registry,[],crossed=True)
        intervals = []
        for mode,ref,context,contrast,w in policy_pairs:
            _, rows = summarize(blocks,tags | {"mode":mode,"reference":ref,"context":context},{},[(contrast,w)],
                                crossed=True)
            intervals.extend(rows)
        results.append((points,intervals))
        print(name,label,proof[label],flush=True)
    return results, proof


def e3_score(root, out):
    split, profile, predictions, proofs = e3_validate(root)
    registry, _, _, _ = registered()
    roster = parquet_lock(root/"registration/cell_roster.parquet",split["cell_roster_sha256"])
    out.mkdir(exist_ok=False,parents=True)
    access = out/"outcome_access.json"
    publish_json(access,{"score_passes":1,"forecasts":proofs})
    panels = []
    for f, (role,q) in enumerate(zip(split["outer"],predictions)):
        value = native_rows(role["query_drugs"],role["query_cells"],labels=True)
        if not value[META].equals(q[META]) or panel_hash(value) != role["query_roster_sha256"]:
            raise ValueError("E3 truth roster changed")
        panels.append((f,q.assign(truth=native_sensitivity(value),lineage_group=q.cell_id.map(roster.family),
                                  tissue=q.cell_id.map(roster.tissue))))
        print("E3 official outcome load",f,len(value),flush=True)
    hcc = [(f,v[v.tissue.eq("Liver")]) for f,v in panels]
    source, full_scope = e3_scopes(panels,registry.scaffold_block,"DRP")
    liver, liver_scope = e3_scopes(hcc,registry.scaffold_block,"Liver")
    p, d = output(out,"E3",source+liver)
    publish_json(out/"score_receipt.json",{"score_passes":1,"E3_Fixed_score_passes":1,"E3_Nested_score_passes":1,
        "native_outcome_access_sha256":sha256(access),"forecast_evidence":proofs,
        "split_sha256":profile["split_sha256"],"profiles_sha256":sha256(root/"profiles/receipt.json"),
        "source_support":full_scope,"Liver_support":liver_scope,"classification":e3_policy(d),
        "paired_draws":DRAWS,"interaction_interval":"Re-ranks fixed double-centered residuals within each paired lineage draw",
        "scoring_files":{n:sha256(out/n) for n in ("E3_METRICS.tsv","E3_PAIRED_95CI.tsv")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment",choices=("e1","e3"))
    parser.add_argument("root",type=Path)
    parser.add_argument("out",type=Path)
    args = parser.parse_args()
    {"e1":e1_score,"e3":e3_score}[args.experiment](args.root,args.out)
