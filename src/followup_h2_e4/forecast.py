"""Seal all three tracks' forecasts before opening held outcomes in the scorer."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from followup_h2.common import V1_RUN, V2_FORECAST, context, panel_hash, parquet_lock, sha256
from followup_h2.e1 import arms, checked_delta
from followup_h2.e3 import frozen_policies, v1_predict
from nextgen_h2.assets import KEY, LIMORE
from nextgen_h2.replay import network_deny
from nextgen_h2.run import META, read_json
from nextgen_h2.score import adult_licob
from nextgen_h2_v2.freeze import V1_SHA, gram_receipt
from vc_build.io import publish_json

from .data import COLD, CPU, GDSC, PLAN, STORAGE, SUPPLEMENT, cpu_threads, internal_only, registry, source_rows, utc_now
from .model import CALIBRATED, CONTROLS, CONTROL_SEED, FULL, KERNEL_ONLY, LOSO, PENALTIES, POOLED, Candidate, Kernel, settings, v2_policy
from .selection import role_data, split_contract
from .split import role_id

E1_COMPONENTS = STORAGE / "followup_20261124/h2_e1e3/e1-components"
PERSONAL, V2_OFFSET, FULL_OFFSET = "__personal", "__v2_offset", "__full_offset"
STAGE_FIELDS = (PERSONAL, V2_OFFSET, FULL_OFFSET)


def selection_lock(preflight, selected):
    split = split_contract(preflight)
    path = selected / "e4_selection_receipt.json"
    proof = read_json(path)
    if (proof["plan_sha256"] != sha256(PLAN)
            or proof["split_sha256"] != sha256(preflight / "e4_split_manifest.json")
            or proof["grid_sha256"] != sha256(selected / "E4_INNER_GRID.tsv")
            or proof["external_cohort_access"] or proof["external_score_access"]
            or set(proof["policy"]) != set(split["selections"])
            or proof["kernel_candidates_per_method"] != 160 or proof["anchor_strategies_per_method"] != 16):
        raise ValueError("Forecasting requires all three sealed selection tracks")
    return split, proof, sha256(path)


def outcomes_locked():
    blocked = (GDSC, SUPPLEMENT, LIMORE / "test_labels")

    def guard(event, args):
        network_deny(event, args)
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(args[0])))
            if any(path.is_relative_to(asset) for asset in blocked):
                raise PermissionError(f"Outcome access belongs to the single official scorer: {path}")

    sys.addaudithook(guard)


def frozen_components(part):
    """Reassemble the E1 frozen components and compare to both original H2 forecasts."""
    directory = E1_COMPONENTS / str(part)
    e1 = read_json(directory / "receipt.json")
    pieces = parquet_lock(directory / "components.parquet", e1["components_sha256"])
    older = V1_RUN / "forecasts" / str(part)
    newer = V2_FORECAST / str(part)
    v2 = read_json(newer / "receipt.json")
    native = read_json(newer / "replay.json")
    v1score = parquet_lock(older / "predictions.parquet", V1_SHA[part], columns=META + ["champion", "C1"])
    v2hash = v2["files"]["forecast.parquet"]
    v2score = parquet_lock(newer / "forecast.parquet", v2hash, columns=META + ["H2-v2", "lincs_observed"])
    assert e1["query_outcomes_loaded"] == v2["query_outcomes_loaded"] == 0
    assert pieces[META].equals(v1score[META]) and pieces[META].equals(v2score[META])
    assert panel_hash(pieces) == e1["roster_sha256"]
    assert np.array_equal(pieces.lincs_observed, v2score.lincs_observed)
    assert native["forecast_sha256"] == v2hash and e1["v2_native_replay_sha256"] == sha256(newer / "replay.json")
    assert not native["network_allowed"] and native["max_abs_diff"] < 1e-6
    assert e1["v1_independent_forward_max_abs_diff"] < 1e-6
    scores = arms(pieces)
    replay = {
        "rows": len(pieces), "F00_max_abs_diff": checked_delta(scores["F00"], v1score.champion),
        "F11_max_abs_diff": checked_delta(scores["F11"], v2score["H2-v2"]),
        "C1_max_abs_diff": checked_delta(pieces.ci + pieces.rc, v1score.C1),
        "E1_component_receipt_sha256": sha256(directory / "receipt.json"),
        "F00_frozen_sha256": V1_SHA[part], "F11_frozen_sha256": v2hash,
        "native_F11_offline_replay_max_abs_diff": native["max_abs_diff"],
        "native_F00_independent_forward_max_abs_diff": e1["v1_independent_forward_max_abs_diff"],
    }
    return pieces.assign(C1=v1score.C1, **{"H2-v1": scores["F00"], "H2-v2": scores["F11"]}), replay


def selected_parts(engine, policy, control="real"):
    return engine.parts(Candidate(**policy["selected"]), policy["intercept"], control)


def check_selected_gram(gram, stop):
    evidence = gram_receipt(gram[:stop, :stop])
    if evidence["minimum_eigenvalue"] <= 0.:
        raise ValueError("Selected drug training Gram is not positive definite")
    return evidence


def adult_predictions(frame, licence):
    # Apply each constant offset after common donor-level residual averaging.
    data = adult_licob(frame.drop(columns=[KERNEL_ONLY, FULL]), licence)
    offsets = frame.groupby(KEY)[[V2_OFFSET, FULL_OFFSET]].first()
    for col in offsets:
        assert frame.groupby(KEY)[col].nunique().max() == 1
        data[col] = data[KEY].map(offsets[col])
    return data.assign(**{KERNEL_ONLY: data[PERSONAL] + data[V2_OFFSET], FULL: data[PERSONAL] + data[FULL_OFFSET]})


def baseline_panel(engine, pieces, personal, chosen, support):
    query = engine.query
    if not query[META].equals(pieces[META]):
        raise ValueError("Pinned baseline does not cover the exact registered query")
    assert np.array_equal(support[engine.selector[1]], pieces.lincs_observed)
    common = np.where(pieces.lincs_observed, personal[engine.selector], pieces.rc)
    v2_offset = pieces.i1.to_numpy(dtype=float)
    chosen_offset = chosen[engine.selector[1]]
    return query.assign(
        C1=pieces.C1, **{"H2-v1": pieces["H2-v1"], "H2-v2": pieces["H2-v2"],
                        KERNEL_ONLY: common + v2_offset, FULL: common + chosen_offset},
        lincs_observed=pieces.lincs_observed,
        **{PERSONAL: common, V2_OFFSET: v2_offset, FULL_OFFSET: chosen_offset},
    )


def freeze(preflight, selected, out, index):
    split, proof, selection_hash = selection_lock(preflight, selected)
    reg = registry()
    external = index == len(split["evaluations"])
    if not external:
        internal_only()
        role = split["evaluations"][index]
        data, fit, query = role_data(split["roles"][role], reg, inner=False)
        track, fold, _ = role.split("/")
        fold = int(fold)
    else:
        outcomes_locked()
        data, other_reg, query = context(CPU, external=True)
        assert other_reg.equals(reg)
        fit = source_rows(sorted(reg.index), sorted(split["roles"]["T2/0/outer"]["fit_screens"]), labels=True)
        query = query[META].sort_values("row_id", ignore_index=True)
        if panel_hash(fit) != split["source_roster_sha256"]:
            raise ValueError("Deployment fit differs from all registered source rows")
        track, fold = "T3", 3
    name = role_id(track, fold, LOSO)
    engine = Kernel(data, fit, query, reg, name)
    candidate = proof["policy"][name]
    personal, anchor, gram, support, solve = selected_parts(engine, candidate)
    gram_record = check_selected_gram(gram, engine.stop)
    replay, increment = None, {}
    if track == "T1":
        pool = proof["policy"][role_id(track, fold, POOLED)]
        z, u, pooled_gram, pooled_support, pooled_solve = selected_parts(engine, pool)
        assert np.array_equal(support, pooled_support)
        v1 = frozen_policies(3)["v1"]
        v2 = v2_policy(name)
        historical, historical_support = engine.geometry(v2)
        family = "lincs_neighbors" if v2.intercept == "neighbors" else "lincs_krr"
        old_anchor = engine.anchor_strategy((family, v2.anchor_penalty), historical, historical_support)
        panel = query.assign(**{"C1": engine.extract_rows(engine._c1),
                               "H2-v1": v1_predict(engine, v1, CONTROL_SEED),
                               LOSO: engine.extract_with(personal, anchor), POOLED: engine.extract_with(z, u),
                               FULL: engine.extract_with(personal, anchor),
                               KERNEL_ONLY: engine.extract_with(personal, old_anchor)},
                             lincs_observed=support[engine.selector[1]])
        increment = {"pooled_drug_gram": check_selected_gram(pooled_gram, engine.stop), "pooled_solve": pooled_solve}
    else:
        pieces, replay = frozen_components(fold)
        panel = baseline_panel(engine, pieces, personal, anchor, support)
        if track == "T2":
            for arm, ctrl in zip(CONTROLS, ("mask", "shuffle")):
                z, u, _, present, diagnostic = selected_parts(engine, candidate, ctrl)
                assert np.array_equal(support, present)
                baseline = np.where(panel.lincs_observed, z[engine.selector], pieces.rc)
                panel[arm] = baseline + u[engine.selector[1]]
                increment[arm] = diagnostic
        else:
            locked = Candidate(**candidate["selected"])
            for penalty in PENALTIES:
                point = replace(locked, penalty=penalty)
                if point == locked:
                    value, present = personal, support
                else:
                    value, _, present, _, _ = engine.solve_personal(settings(name, point))
                assert np.array_equal(support, present)
                panel[f"lambda/{penalty:g}"] = np.where(panel.lincs_observed, value[engine.selector], pieces.rc)
    dest = out / str(index)
    dest.mkdir(exist_ok=False, parents=True)
    file = dest / "predictions.parquet"
    panel.to_parquet(file, index=False)
    gram_path = dest / "selected_drug_gram.npy"
    np.save(gram_path, gram[:engine.stop, :engine.stop].numpy(), allow_pickle=False)
    publish_json(dest / "receipt.json", {
        "track": track, "fold": fold, "rows": len(panel), "fit_rows": len(fit),
        "created_at_utc": utc_now(), "selection_sha256": selection_hash, "query_outcomes_opened": 0,
        "roster_sha256": panel_hash(panel), "predictions_sha256": sha256(file), "role": name,
        "selected_drug_gram": gram_record | {"sha256": sha256(gram_path)}, "solve": solve, "frozen_replay": replay,
        "fit_calibration": engine.calibration, **increment,
    })
    print(track, fold, len(panel), gram_record, flush=True)


def read_forecast(folder, selection_hash):
    note = read_json(folder / "receipt.json")
    panel = parquet_lock(folder / "predictions.parquet", note["predictions_sha256"])
    if (note["selection_sha256"] != selection_hash or note["query_outcomes_opened"]
            or note["roster_sha256"] != panel_hash(panel) or note["rows"] != len(panel) or not panel.row_id.is_unique
            or note["selected_drug_gram"]["minimum_eigenvalue"] <= 0):
        raise ValueError("Forecast changed after freeze or fails the numerical contract")
    expected = ("C1", "H2-v1", LOSO, POOLED, KERNEL_ONLY, FULL) if note["track"] == "T1" else CALIBRATED
    if note["track"] == "T2":
        expected = (*expected, *CONTROLS)
    if not np.isfinite(panel[list(expected)]).all().all():
        raise ValueError("Nonfinite predictions on a sealed support")
    return panel, note


if __name__ == "__main__":
    cpu_threads()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preflight", type=Path)
    p.add_argument("selected", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    freeze(args.preflight, args.selected, args.out, int(os.environ["SLURM_ARRAY_TASK_ID"]))
