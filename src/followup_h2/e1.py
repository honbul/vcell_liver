"""Replay frozen H2 components, then fit the bounded chemical intercept counterfactual."""
import argparse
import os
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.metrics import Evaluator, macro, stack
from nextgen_h2.model import Fit
from nextgen_h2.replay import network_deny
from nextgen_h2.run import META, read_json, subfold
from nextgen_h2_v2 import freeze
from nextgen_h2_v2.kernel import V1, design
from vc_build.external import native_sensitivity
from vc_build.io import publish_json, sha256

from .common import ModzEngine, V1_RUN, V2, V2_FORECAST, context, device, native_rows, panel_hash

FIXED_V2 = V2 / "run-final-strict/champion.json"
CHEM_GRID = (("chem", 1.), *(("krr", a) for a in (.01, .10, 1., 10.)),
             *(("neighbors", n) for n in (3., 5., 10., 20.)))
E1_ARMS = ("F00", "F10", "F01", "F11", "F01-only", "Cchem-int")


class FrozenUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("__main__", "Bundle"):
            return freeze.Bundle
        return super().find_class(module, name)


def load_pinned(path, locked):
    if sha256(path) != locked:
        raise ValueError(f"Changed predictor {path}")
    with path.open("rb") as stream:
        return FrozenUnpickler(stream).load()


def v1_parts(model, raw, center):
    if model.shared_effect is not None or model.residual_coefficients is not None:
        raise ValueError("E1 registered v1 policy must be a separable single-view kernel")
    cross = raw - raw.mean(1, keepdim=True) - center + center.mean() if model.centered else raw
    intercept = model.intercept.broadcast_to((len(raw), len(model.intercept)))
    residual = cross @ model.coefficients
    if model.present is not None:
        i0, r0 = v1_parts(model.baseline, raw, center)
        intercept = torch.where(model.present[None], intercept, i0)
        residual = torch.where(model.present[None], residual, r0)
    return intercept, residual


def checked_delta(a, b):
    delta = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)), initial=0.))
    if not delta < 1e-6:
        raise ValueError(f"Frozen replay changed by {delta:.9g}")
    return delta


def arms(panel):
    i0, i1, r0, r1 = (panel[key].to_numpy(np.float64) for key in ("i0", "i1", "r0", "r1"))
    return dict(F00=i0+r0, F10=i0+r1, F01=i1+r0, F11=i1+r1, **{"F01-only": i1})


def rank_invariance(panel, predictions):
    panel = panel.reset_index(drop=True)
    checks = 0
    for idx in panel.groupby(["source_id", "endpoint_id", KEY], sort=False).indices.values():
        for a, b in (("F00", "F01"), ("F10", "F11")):
            if not np.array_equal(rankdata(predictions[a][idx]), rankdata(predictions[b][idx])):
                raise ValueError(f"Fixed-drug offset changed ranks for {a} vs {b}")
            checks += 1
    return checks


def replay(out, part):
    import sys
    sys.addaudithook(network_deny)
    on = device()
    torch.set_num_threads(5)
    old_dir = V1_RUN / "forecasts" / str(part)
    recent = V2_FORECAST / str(part)
    old_proof, new_proof = read_json(old_dir/"receipt.json"), read_json(recent/"receipt.json")
    assert old_proof["query_outcomes_loaded"] == new_proof["query_outcomes_loaded"] == 0
    old = parquet_lock(old_dir/"predictions.parquet", freeze.V1_SHA[part])
    new = parquet_lock(recent/"forecast.parquet", new_proof["files"]["forecast.parquet"])
    assert new[META].equals(old[META]) and np.array_equal(new["H2-v1"], old.champion)
    assert np.array_equal(new.C1, old.C1) and np.array_equal(new.lincs_observed, old.champion_observed)
    older = load_pinned(old_dir/"predictor.pkl", old_proof["predictor_sha256"])
    newer = load_pinned(recent/"bundle.pkl", new_proof["files"]["bundle.pkl"])
    assert newer.query.equals(new[META]) and newer.final == "H2-R2b"
    data, _, _ = context(torch.device("cpu"), external=part == 3)
    cells = pd.Index(sorted(old.cell_id.unique()))
    rna = data.rna.reindex(index=cells, columns=older["basis"].genes).fillna(older["medians"])
    raw = older["basis"].cross(rna)["full"]
    cr = cells.get_indexer(old.cell_id)
    dr = older["drug_ids"].get_indexer(old[KEY])
    i0, r0 = (v[cr, dr].numpy() for v in v1_parts(older["models"]["champion"], raw, older["center"]))
    ci, rc = (v[cr, dr].numpy() for v in v1_parts(older["models"]["C1"], raw, older["center"]))
    pinned_delta = checked_delta(ci.astype(float) + rc, old.C1)
    state, _ = newer.checkpoints[newer.final]
    if state["extra_drug"] is not None or state["cell_coverage"] is not None:
        raise ValueError("E1 registration expected the selected single-view v2 kernel")
    index = newer.selector
    anchor = state["anchor"].numpy().reshape(-1)[index[1]]
    cross, coeff, drug = (state[k].to(on) for k in ("cell_cross", "coef", "dcross"))
    residual = ((cross @ coeff) @ drug.T).cpu().numpy()[index]
    measured = new.lincs_observed.to_numpy()
    assert state["apply_c1"] and np.array_equal(state["active"].numpy()[index[1]], measured)
    i1 = np.where(measured, anchor, ci)
    r1 = np.where(measured, residual, rc)
    panel = new[META].assign(lincs_observed=measured, i0=i0, i1=i1, r0=r0, r1=r1, ci=ci, rc=rc)
    for key in ("i0", "i1", "ci"):
        if panel.groupby(KEY)[key].nunique().max() != 1:
            raise ValueError(f"{key} is not constant by exact parent")
    factorial = arms(panel)
    d0 = checked_delta(factorial["F00"], old.champion)
    d1 = checked_delta(factorial["F11"], new["H2-v2"])
    check = rank_invariance(panel, factorial)
    dest = out / str(part)
    dest.mkdir(exist_ok=False, parents=True)
    panel.to_parquet(dest/"components.parquet", index=False)
    publish_json(dest/"receipt.json", {
        "part": part, "rows": len(panel), "roster_sha256": panel_hash(panel),
        "components_sha256": sha256(dest/"components.parquet"), "query_outcomes_loaded": 0,
        "F00_max_abs_diff": d0, "F11_max_abs_diff": d1, "C1_max_abs_diff": pinned_delta,
        "aggregate_arithmetic": "FP64 sum of frozen FP32 components",
        "rank_groups_checked": check, "F01_minus_F00_within_drug_spearman": 0.,
        "F11_minus_F10_within_drug_spearman": 0., "network_allowed": False,
        "v1_package_sha256": old_proof["predictor_sha256"],
        "v2_package_sha256": new_proof["files"]["bundle.pkl"],
    })
    print(part, len(panel), d0, d1, check, flush=True)


def chem_anchors(engine, settings):
    chemical = engine.chemistry(V1.style)
    fit, query = chemical[:engine.stop, :engine.stop], chemical[engine.stop:, :engine.stop]
    y = torch.as_tensor(engine.anchor.reindex(engine.fit_drugs).to_numpy(np.float32), device=engine.device)
    same = engine.drug.identity[engine.stop:, :engine.stop]
    warm = same.any(1)
    method, parameter = settings
    if method == "chem":
        estimate = engine.intercept
    elif method == "krr":
        estimate = y.mean() + query @ torch.linalg.solve(
            fit + parameter * torch.eye(len(y), device=engine.device), y-y.mean())
    else:
        sim, ix = torch.topk(query.clip(min=0), int(parameter), dim=1, sorted=True)
        weights = sim.pow(4)
        estimate = (weights*y[ix]).sum(1)/weights.sum(1).clamp_min(1e-12)
    return torch.where(warm, engine.intercept, estimate)


def label(settings):
    return f"{settings[0]}/{settings[1]:g}"


def chem_measure(out, index):
    f, i = divmod(index, 3)
    on = device()
    data, reg, _ = context(on)
    fit_ids, test_ids = subfold(reg, f, i)
    fit, test = native_rows(fit_ids, labels=True), native_rows(test_ids, labels=True)
    q = test[META]
    engine = Fit(data, fit, q, registry=reg)
    baseline = engine.baseline(V1)
    residual = (engine.cross @ baseline.coefficients).cpu().numpy()[engine.selector]
    evaluate = Evaluator(q.assign(truth=native_sensitivity(test)), reg.scaffold_block, i)
    rows = []
    for settings in CHEM_GRID:
        anchor = chem_anchors(engine, settings).cpu().numpy()[engine.selector[1]]
        rows.extend(evaluate.score(label(settings), residual.astype(float) + anchor))
    dest = out/f"{f}-{i}"
    dest.mkdir(exist_ok=False, parents=True)
    _, cell = stack(rows)
    cell.to_parquet(dest/"cell.parquet", index=False)
    publish_json(dest/"receipt.json", {"outer": f, "inner": i, "fit_rows": len(fit), "inner_query_rows": len(q),
                  "outer_outcome_rows_loaded": 0, "candidate_budget": len(CHEM_GRID),
                  "cell_sha256": sha256(dest/"cell.parquet")})


def chem_select(inner, out):
    chosen, curve = {}, []
    for f in range(3):
        areas = [inner/f"{f}-{i}" for i in range(3)]
        receipts = [read_json(p/"receipt.json") for p in areas]
        assert all(x["outer_outcome_rows_loaded"] == 0 for x in receipts)
        cell = pd.concat([parquet_lock(p/"cell.parquet", x["cell_sha256"]) for p, x in zip(areas, receipts)])
        scores = {key: macro(cell[cell.arm == key], "ndcg_at_5") for key in map(label, CHEM_GRID)}
        chosen[str(f)] = max(CHEM_GRID, key=lambda cfg: scores[label(cfg)])
        curve.extend({"outer_fit": f, "candidate": name, "ndcg_at_5": point} for name, point in scores.items())
    values = list(chosen.values())
    chosen["3"] = max(values, key=values.count)
    out.mkdir(exist_ok=False, parents=True)
    pd.DataFrame(curve).to_csv(out/"points.tsv", index=False, sep="\t")
    publish_json(out/"receipt.json", {"parameters": chosen, "candidates": list(CHEM_GRID),
          "outer_outcome_rows_loaded": 0, "points_sha256": sha256(out/"points.tsv")})


def chem_freeze(parts, decision, out, part):
    on = device()
    external = part == 3
    data, reg, _ = context(on, external=external)
    location = parts/str(part)
    lock = read_json(location/"receipt.json")
    panel = parquet_lock(location/"components.parquet", lock["components_sha256"])
    assert lock["query_outcomes_loaded"] == 0
    fit_ids = reg.index if external else subfold(reg, part)[0]
    fit = native_rows(fit_ids, labels=True)
    e = Fit(data, fit, panel[META], registry=reg)
    selected = read_json(decision/"receipt.json")["parameters"][str(part)]
    chemistry = chem_anchors(e, selected).cpu().numpy()[e.selector[1]]
    combined = panel.assign(**arms(panel), **{"Cchem-int": panel.rc.to_numpy(np.float64)+chemistry})
    if external:
        limore = panel[panel.source_id.eq("LIMORE")]
        v2 = ModzEngine(data, None, fit, limore[META], reg)
        spec = design(read_json(FIXED_V2)["3"]["R2b"]["design"])
        if spec.drug_views or spec.cell_views:
            raise ValueError("The LIMORE lambda diagnostic must use the frozen one-view kernel")
        proposal = replace(spec, penalty=4.)
        forecast = v2.extract_rows(v2.solve(proposal, seed=75204))
        # Preserve the v2 anchor and the exact C1 fallback on natural LINCS missingness.
        old_base = v2.extract_rows(v2._c1)
        combined.loc[limore.index, "LIMORE-v2-lambda4"] = np.where(limore.lincs_observed, forecast, old_base)
    dest = out/str(part)
    dest.mkdir(exist_ok=False, parents=True)
    combined.to_parquet(dest/"forecast.parquet", index=False)
    publish_json(dest/"receipt.json", {"part": part, "rows": len(panel), "query_outcomes_loaded": 0,
                "component_receipt_sha256": sha256(location/"receipt.json"),
                "chemistry_selection_sha256": sha256(decision/"receipt.json"),
                "selected": selected, "LIMORE_diagnostic": "v2, only lambda set to 4" if external else None,
                "forecast_sha256": sha256(dest/"forecast.parquet")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("replay", "measure", "select", "freeze"))
    p.add_argument("out", type=Path)
    p.add_argument("--parts", type=Path)
    p.add_argument("--decision", type=Path)
    args = p.parse_args()
    if args.stage == "select":
        chem_select(args.parts, args.out)
    else:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if args.stage == "replay":
            replay(args.out, index)
        elif args.stage == "measure":
            chem_measure(args.out, index)
        else:
            chem_freeze(args.parts, args.decision, args.out, index)
