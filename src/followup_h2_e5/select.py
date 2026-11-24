"""Select every E5 structure against the two matched internal CV objectives."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from followup_h2.common import sha256
from followup_h2_e4.data import cpu_threads, utc_now
from followup_h2_e4.model import GRID, GRID_PARTS, LOSO, POOLED
from nextgen_h2.run import read_json
from vc_build.io import publish_json

from .grid import all_roles
from .kernel import BASELINE, STRUCTURES, TASK, TAUS
from .registration import E4, E4_SELECTED, PLAN, read_split, source_only

P1, REL, P2 = "P1-Minimax", "P1-RelMinimax", "P2-Decoupled"
SHARED, INSCREEN = "P3B-Shared", "P3B-InScreen"
COMBINED = "H2-v3-Combined"
MODEL_CASES = (P1, REL, P2, *STRUCTURES, COMBINED)
COLUMN_KEYS = ["family", "tau", "base", "shape", "weight", "penalty", "candidate"]


def sealed_e4_grid():
    file = E4 / "selection/e4_selection_receipt.json"
    note = read_json(file)
    assert sha256(file) == E4_SELECTED and not note["external_cohort_access"]
    scores = file.with_name("E4_INNER_GRID.tsv")
    assert sha256(scores) == note["grid_sha256"]
    d = pd.read_csv(scores, sep="\t")
    d = d.loc[d.stage.eq(1) & d.tier.eq("screen")]
    return d[["role", "screen", "base", "shape", "weight", "penalty", "candidate", "value", "n"]].assign(
        family=BASELINE, tau=0.)


def recorded_grids(preflight, work, split):
    registered = sha256(preflight / "split.json")
    frames = []
    for i, role in enumerate(all_roles(split)):
        for q, quadrant in enumerate(GRID_PARTS):
            shard = i*len(GRID_PARTS)+q
            folder = work/f"{shard:03d}"
            note = read_json(folder/"receipt.json")
            path = folder/"scores.tsv"
            assert note["registration_sha256"] == registered and not note["external_cohort_access"]
            assert note["role"] == role and note["grid_part"] == list(quadrant)
            assert note["scores_sha256"] == sha256(path)
            f = pd.read_csv(path, sep="\t")
            assert set(f.role) == {role} and len(f) == note["row_count"]
            assert np.isfinite(f.value).all() and np.isfinite(f.relative_residual).all()
            assert f.relative_residual.max() <= 5e-5 and f.n.min() > 0
            frames.append(f)
    results = pd.concat([*frames, sealed_e4_grid()], ignore_index=True)
    results = pd.concat([results, results[results.family.eq(BASELINE)].assign(family=TASK)], ignore_index=True)
    required = {(f, tau, c.key) for f in (BASELINE,*STRUCTURES)
                for tau in (TAUS if f == TASK else (0.,)) for c in GRID}
    for role, inner in results.groupby("role", sort=False):
        assert set(inner[["family","tau","candidate"]].itertuples(index=False,name=None)) == required,role
        assert not inner.duplicated(["family","tau","candidate","screen"]).any(),role
        assert set(inner.screen) == set(split["roles"][role]["query_screens"]),role
    return results


def objective(roles, pooled, loso):
    def mean(names):
        return (roles[roles.role.isin(names)].groupby([*COLUMN_KEYS,"screen"], sort=False).value.mean()
                .groupby(level=list(range(len(COLUMN_KEYS))),sort=False).mean())
    return pd.concat([mean(pooled).rename("pooled"),mean(loso).rename("loso")],axis=1).reset_index()


def regrets(frame):
    x = frame.copy().reset_index(drop=True)
    p_star, l_star = x.pooled.max(), x.loso.max()
    x["absolute_regret"] = np.maximum(p_star-x.pooled, l_star-x.loso)
    x["relative_regret"] = (np.maximum((p_star-x.pooled)/p_star, (l_star-x.loso)/l_star)
                            if p_star > 0. and l_star > 0. else np.nan)
    x["p_star"], x["l_star"] = p_star,l_star
    return x


def winner(frame, key):
    if not np.isfinite(frame[key]).all():
        raise ValueError(f"Cannot choose {key}: nonpositive best attainable relative score")
    options = frame[frame[key].le(frame[key].min()+.001)]
    pick = options.sort_values(["penalty",key,"order"],ascending=[False,True,True]).iloc[0]
    keep = [*COLUMN_KEYS, "pooled","loso","absolute_regret","relative_regret","p_star","l_star"]
    record = pick[keep].to_dict()
    if not np.isfinite(record["relative_regret"]):
        record["relative_regret"] = None
    return record


def select(preflight, work, out):
    source_only()
    split = read_split(preflight)
    score = recorded_grids(preflight, work, split)
    decisions, all_points = {}, []
    order = {c.key:i for i,c in enumerate(GRID)}
    family_order = {f:i for i,f in enumerate((BASELINE,*STRUCTURES))}
    for parent in sorted({v.rsplit("/",1)[0] for v in split["selections"]}):
        points = objective(score,split["selections"][f"{parent}/{POOLED}"],split["selections"][f"{parent}/{LOSO}"])
        assert len(points)==len(GRID)*(3+len(TAUS)) and np.isfinite(points[["pooled","loso"]]).all().all()
        points["order"] = points.family.map(family_order)*len(GRID)*len(TAUS) + points.tau.map(TAUS.index)*len(GRID) + points.candidate.map(order)
        base = regrets(points[points.family.eq(BASELINE)])
        p1 = winner(base,"absolute_regret")
        arms = {P1:p1, REL:winner(base,"relative_regret"),P2:p1.copy()}
        all_points.append(base.assign(outer=parent, comparison="P1"))
        for family in STRUCTURES:
            f = regrets(points[points.family.eq(family)])
            arms[family] = winner(f,"absolute_regret")
            all_points.append(f.assign(outer=parent,comparison=family))
        best = regrets(points[points.family.ne(BASELINE)])
        arms[COMBINED] = winner(best,"absolute_regret")
        all_points.append(best.assign(outer=parent,comparison="P3-Best"))
        decisions[parent] = arms
    out.mkdir(exist_ok=False,parents=True)
    table = out/"kernel_objectives.tsv"
    pd.concat(all_points,ignore_index=True).to_csv(table,index=False,sep="\t",na_rep="NA")
    publish_json(out/"kernels.json",{
        "external_cohort_access":0, "created_at_utc":utc_now(), "E4_internal_selection_sha256":E4_SELECTED,
        "registration_sha256":sha256(preflight/"split.json"), "plan_sha256":sha256(PLAN),
        "kernel_objectives_sha256":sha256(table), "choices":decisions, "expected_grid_roles":39,
        "geometry_candidates_each":160, "hierarchical_tau_points":list(TAUS),
    })
    for parent, arms in decisions.items():
        print(parent,[(a,item["candidate"],item["tau"]) for a,item in arms.items()],flush=True)


if __name__ == "__main__":
    cpu_threads()
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("preflight","work","out"):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    select(a.preflight,a.work,a.out)
