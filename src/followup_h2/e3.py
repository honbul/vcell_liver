"""Select constrained H2 policies within joint holdouts and seal the two outer modes."""
import argparse
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.metrics import Evaluator, macro, stack
from nextgen_h2.model import Baseline, Fusion
from nextgen_h2.run import META, config_dict, read_json, spec_dict
from nextgen_h2_v2.kernel import V1, design
from vc_build.external import native_sensitivity
from vc_build.io import publish_json, sha256

from .common import ModzEngine, V1_RUN, context, device, native_rows, panel_hash
from .e1 import FIXED_V2

ARMS = ("C1", "H2-v1", "H2-v2", "C2", "C3")
MODES = ("E3-Fixed", "E3-Nested")
REFERENCES = ("REF-EXCLUDED", "REF-INCLUDED")
V1_GRID = tuple((w, p) for w in (.1, .25, .5) for p in (4., 16.))
ANCHORS = (("chem", 1.), ("neighbors", 5.), ("neighbors", 10.), ("krr", 1.))


def read_contract(registration, profile_dir):
    file = registration/"split_manifest.json"
    split, receipt = read_json(file), read_json(profile_dir/"receipt.json")
    if receipt["split_sha256"] != sha256(file) or receipt["source_outcome_columns_read"]:
        raise ValueError("LINCS profiles differ from the locked response-blind roles")
    return split, receipt


def reference_excluded(profile_dir, roles, role):
    evidence = roles[role["name"]]
    if (evidence["excluded_references"] != role["excluded_references"] or
            set(evidence["used_24h_references"]) & set(role["excluded_references"])):
        raise ValueError("Excluded lineage entered profile construction")
    d = parquet_lock(profile_dir/(role["name"]+".parquet"), evidence["profile_sha256"])
    return {"modz24": d["v1"], "landmark_mean": d["v2"]}


def role_rows(role, *, inner):
    fit = native_rows(role["fit_drugs"], role["fit_cells"], labels=True)
    query = native_rows(role["query_drugs"], role["query_cells"], labels=inner)
    if (panel_hash(fit) != role["fit_roster_sha256"] or panel_hash(query) != role["query_roster_sha256"]
            or set(fit[KEY]) & set(query[KEY]) or set(fit.lineage_group) & set(query.lineage_group)
            or set(fit.cell_id) & set(query.cell_id)):
        raise ValueError("Native response load altered the two held axes")
    return fit, query


def frozen_policies(f):
    v1 = read_json(V1_RUN/"champion.json")[str(f)]
    geometry = v1["champion"]
    settings = geometry["spec"]
    base = v1["base"]
    assert (settings["weight"], base["penalty"], base["style"]) == (.5, 4., "molecular")
    assert geometry["configs"] == [{"view":"lincs","recipe":"modz24","transform":"raw","dimension":0,"kernel":"cos"}]
    assert settings["method"] == "psd" and not settings["match"] and not settings["stress"]
    v2 = read_json(FIXED_V2)[str(f)]["R2b"]["design"]
    if v2["cell_views"] or v2["drug_views"] or v2["lincs"]["recipe"] not in ("v1", "landmark_mean"):
        raise ValueError("Follow-up permits only the final one-view 24h policy")
    return {"v1": {"base": base, "geometry": geometry}, "v2": v2}


def v1_predict(engine, record, seed):
    g = record["geometry"]
    model = engine.fusion(Baseline(**record["base"]), config_dict(g["configs"]), spec_dict(g["spec"]), seed=seed)
    return engine.extract(model)


def v2_parts(engine, params, seed):
    _, state = engine.solve(replace(params, intercept="chem"), seed=seed, retain=True)
    cross, coef, drug = (state[k].to(engine.device) for k in ("cell_cross", "coef", "dcross"))
    residual = (cross @ coef) @ drug.T
    fit_gram = state["dfit"].to(engine.device)
    mask = state["active"].to(engine.device)
    return residual, fit_gram, drug, mask


def v2_anchors(engine, params, pieces):
    residual, fit, drug, mask = pieces
    a = engine.new_intercept(params, fit, drug, [])
    warm = engine.drug.identity[engine.stop:, :engine.stop].any(dim=1)
    a = torch.where(warm, engine.intercept, a)
    prediction = torch.where(mask[None, :], a + residual, engine._c1)
    return engine.extract_rows(prediction)


def identifier(family, w, l, anchor="chem", parameter=1.):
    return f"{family}/{w:g}/{l:g}/{anchor}/{parameter:g}"


def candidates(base):
    for w, lam in V1_GRID:
        policy = {"base": base["v1"]["base"] | {"penalty": lam}, "geometry": {
            "spec": base["v1"]["geometry"]["spec"] | {"weight": w},
            "configs": base["v1"]["geometry"]["configs"],
        }}
        yield identifier("v1", w, lam), "v1", policy
    spec = design(base["v2"])
    for w in (.5*spec.lincs.weight, spec.lincs.weight, 2*spec.lincs.weight):
        for lam in (.5*spec.penalty, spec.penalty, 2*spec.penalty):
            for anchor, parameter in ANCHORS:
                model = replace(spec, lincs=replace(spec.lincs, weight=w), penalty=lam,
                                intercept=anchor, anchor_penalty=parameter)
                yield identifier("v2", w, lam, anchor, parameter), "v2", asdict(model)


def measure(registration, profiles, out, index):
    f, i = divmod(index, 3)
    on = device()
    split, receipt = read_contract(registration, profiles)
    role = split["inner"][index]
    assert role["name"] == f"{f}-{i}"
    fit, test = role_rows(role, inner=True)
    data, reg, _ = context(on)
    data.profiles = reference_excluded(profiles, receipt["roles"], role)
    q = test[META]
    engine = ModzEngine(data, None, fit, q, reg)
    held = q[KEY].isin(data.profiles["modz24"].index)
    evaluator = Evaluator(q.assign(truth=native_sensitivity(test)), reg.scaffold_block, i)
    trials = list(candidates(frozen_policies(f)))
    pieces = None
    prior = None
    pairs = []
    seed = 57091 + index
    for name, family, record in trials:
        if family == "v1":
            predicted = v1_predict(engine, record, seed)
        else:
            spec = design(record)
            tag = spec.lincs.weight, spec.penalty
            if prior != tag:
                pieces = v2_parts(engine, spec, seed)
                prior = tag
            predicted = v2_anchors(engine, spec, pieces)
        pairs.extend(evaluator.score(name, predicted, held))
    output = out / role["name"]
    output.mkdir(exist_ok=False, parents=True)
    d, c = stack(pairs)
    d.to_parquet(output/"drug.parquet", index=False)
    c.to_parquet(output/"cell.parquet", index=False)
    publish_json(output/"receipt.json", {
        "outer": f, "inner": i, "split_sha256": receipt["split_sha256"],
        "profiles_sha256": sha256(profiles/"receipt.json"), "fit_rows": len(fit),
        "inner_query_rows": len(q), "outer_query_outcomes_loaded": 0,
        "files": {name: sha256(output/name) for name in ("drug.parquet", "cell.parquet")},
    })
    print(role["name"], len(fit), len(q), len(trials), flush=True)


def select(registration, profiles, inner, out):
    _, plan = read_contract(registration, profiles)
    selected, points, proofs = {}, [], []
    for f in range(3):
        d, c = [], []
        for i in range(3):
            path = inner/f"{f}-{i}"
            evidence = read_json(path/"receipt.json")
            assert (evidence["outer"], evidence["inner"], evidence["outer_query_outcomes_loaded"]) == (f, i, 0)
            assert evidence["split_sha256"] == plan["split_sha256"]
            assert evidence["profiles_sha256"] == sha256(profiles/"receipt.json")
            proofs.append(sha256(path/"receipt.json"))
            for axis, rows in (("drug", d), ("cell", c)):
                name = axis+".parquet"
                rows.append(parquet_lock(path/name, evidence["files"][name]))
        drugs, cells = pd.concat(d), pd.concat(c)
        trials = list(candidates(frozen_policies(f)))
        rank = {}
        for name, family, _ in trials:
            rho = macro(drugs[drugs.arm.eq(name)], "spearman")
            ndcg = macro(cells[cells.arm.eq(name)], "ndcg_at_5")
            if not np.isfinite([rho, ndcg]).all():
                raise ValueError("A nested selection arm lacks eligible validation data")
            rank[name] = rho, ndcg
            points.append({"outer_fit": f, "arm": name, "family": family, "spearman": rho, "ndcg_at_5": ndcg})
        selected[str(f)] = {}
        for family in ("v1", "v2"):
            best = max((x for x in trials if x[1] == family), key=lambda row: rank[row[0]])
            selected[str(f)][family] = best[2]
            selected[str(f)][family+"_candidate"] = best[0]
    out.mkdir(exist_ok=False, parents=True)
    pd.DataFrame(points).to_csv(out/"points.tsv", sep="\t", index=False)
    publish_json(out/"policy.json", selected)
    publish_json(out/"receipt.json", {"outer_query_outcomes_loaded": 0,
        "split_sha256": plan["split_sha256"], "profile_sha256": sha256(profiles/"receipt.json"),
        "inner_receipts_sha256": proofs, "policy_sha256": sha256(out/"policy.json"),
        "points_sha256": sha256(out/"points.tsv"), "condition_for_selection": "REF-EXCLUDED"})


def column(mode, reference, arm):
    return f"{mode}/{reference}/{arm}"


def freeze(registration, profiles, decisions, out, fold):
    on = device()
    split, role_note = read_contract(registration, profiles)
    selection = read_json(decisions/"receipt.json")
    if (selection["outer_query_outcomes_loaded"] != 0 or selection["split_sha256"] != role_note["split_sha256"]
            or selection["policy_sha256"] != sha256(decisions/"policy.json")):
        raise ValueError("Selection and double-cold registration disagree")
    role = split["outer"][fold]
    fit, q = role_rows(role, inner=False)
    assert not {"endpoint_value","endpoint_direction"} & set(q)
    data, reg, _ = context(on)
    libraries = {"REF-INCLUDED": data.profiles, "REF-EXCLUDED": reference_excluded(profiles, role_note["roles"], role)}
    policies = {"E3-Fixed": frozen_policies(fold), "E3-Nested":read_json(decisions/"policy.json")[str(fold)]}
    panel = q.copy()
    base = None
    seed = 82201 + fold
    for reference in REFERENCES:
        data.profiles = libraries[reference]
        engine = ModzEngine(data, None, fit, q, reg)
        c1 = engine.extract_rows(engine._c1)
        if base is not None and not np.array_equal(base, c1):
            raise ValueError("C1 changed across LINCS reference masks")
        base = c1
        panel[reference+"/observed"] = q[KEY].isin(data.profiles["modz24"].index)
        for mode in MODES:
            cfg = policies[mode]
            panel[column(mode, reference, "C1")] = c1
            panel[column(mode, reference, "H2-v1")] = v1_predict(engine, cfg["v1"], seed)
            main = design(cfg["v2"])
            for arm, control in (("H2-v2", "real"), ("C2", "mask"), ("C3", "shuffle")):
                spec = replace(main, lincs=replace(main.lincs, control=control))
                panel[column(mode, reference, arm)] = engine.extract_rows(engine.solve(spec, seed=seed))
        print(fold, reference, int(panel[reference+"/observed"].sum()), flush=True)
    if (panel[REFERENCES[0]+"/observed"] & ~panel[REFERENCES[1]+"/observed"]).any():
        raise ValueError("Removing references cannot create observed support")
    dest = out/str(fold)
    dest.mkdir(exist_ok=False, parents=True)
    panel.to_parquet(dest/"forecast.parquet", index=False)
    publish_json(dest/"receipt.json", {"fold": fold, "fit_rows": len(fit), "rows": len(q),
        "outer_query_outcomes_loaded": 0, "split_sha256": role_note["split_sha256"],
        "query_roster_sha256": panel_hash(q), "nested_proof_sha256": sha256(decisions/"receipt.json"),
        "profile_proof_sha256": sha256(profiles/"receipt.json"), "forecast_sha256": sha256(dest/"forecast.parquet")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("measure", "select", "freeze"))
    parser.add_argument("registration", type=Path)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--upstream", type=Path)
    a = parser.parse_args()
    if a.stage == "select":
        select(a.registration, a.profiles, a.upstream, a.out)
    else:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if a.stage == "measure":
            measure(a.registration, a.profiles, a.out, index)
        else:
            freeze(a.registration, a.profiles, a.upstream, a.out, index)
