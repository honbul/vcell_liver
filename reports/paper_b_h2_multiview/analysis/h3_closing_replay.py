"""Machine-replay only. Rebuild every archived metric row without running official scorers."""
import sys

from h3_closing_inputs import *


def h3_replay():
    from h3v2 import core_score, score as hs
    from nextgen_h2.metrics import CELL, DRUG

    pin_modules(("h3v2.score", "h3v2.core_score", "h3v2.core_reference_metrics",
                 "nextgen_h2.metrics", "nextgen_h2.score", "vc_build.response_components"))
    score, manifest, lock = h3_metadata()
    labels, licence, external_groups = external_inputs()
    source_groups = registry()
    scored, held = [], []
    for split in (*hs.SPLITS, "external"):
        panel = h3_frame(split, score, manifest, lock, labels)
        cols = [k for k in panel if k not in (*META, "lincs", "truth")]
        if split == "external":
            for cohort, block in panel.groupby("source_id"):
                if cohort == "LICOB":
                    block = adult_licob(block.rename(columns={"lincs":"lincs_observed"}), licence)
                scored.append(hs.units(block, external_groups, 0, cohort, {"all":None}, cols))
        else:
            scope, fold = split.rsplit("-", 1)
            if scope in hs.SCREENS:
                held.append(panel.assign(fold=int(fold)))
            else:
                masks = {"all":None}
                if scope == "drug":
                    masks.update(lincs724=panel.lincs, structure462=~panel.lincs)
                scored.append(hs.units(panel, source_groups, int(fold), scope, masks, cols))
        print(f"H3 replay {split}: {len(panel)} rows", flush=True)

    # Original held scoring uses draws only for intervals. Point construction is identical.
    joint = pd.concat(held, ignore_index=True)
    cols = [k for k in joint if k not in (*META, "lincs", "truth", "fold")]
    panels = core_score.screen_scores(joint.rename(columns={"truth":"observed"}), cols, minimum=5)
    fields = {"within_drug_spearman":"spearman", "within_drug_pearson":"pearson",
              "precision_at_k":"precision_at_5", "residual_spearman":"interaction_spearman"}
    panels["macro"] = {m: ({metric: float(np.mean([p[m][0][metric] for p in panels.values()]))
                             for metric in core_score.ALL_METRICS}, {}) for m in cols}
    n = joint.lineage_group.nunique()
    points = []
    for screen, arms in panels.items():
        for arm, (point, _) in arms.items():
            model, seed = hs.tagged([arm])[arm]
            points.extend({"scope":"held", "support":"core68", "screen":screen, "tissue":"",
                           "model":model, "seed":seed, "metric":fields.get(k,k), "value":v,
                           "unit_count":n, "cluster_count":n} for k, v in point.items())
    d, c = map(pd.concat, zip(*scored))
    out = hs.mean_and_sd(pd.concat([hs.native_metric_points(d, DRUG), hs.native_metric_points(c, CELL),
                                   pd.DataFrame(points)], ignore_index=True))
    name = "H3V2_METRICS.tsv"
    keys = ["scope", "support", "model", "seed", "tissue", "screen", "metric"]
    return name, compare(out, H3 / "score" / name, keys, ["value", "seed_sd"], score["tables"][name])


def e5_replay():
    from followup_h2.inference import measure, summarize
    from followup_h2_e5.forecast import TRACK_ARMS
    from followup_h2_e5.registration import PLAN
    from followup_h2_e5.score import ANCHOR_ONLY, TRACK_TAG
    from trackb_replay_39118 import panels

    pin_modules(("followup_h2.inference", "followup_h2_e5.forecast", "followup_h2_e5.registration",
                 "followup_h2_e5.score", "followup_h2_e5.test_pipeline", "followup_h2_e4.forecast",
                 "followup_h2_e4.data", "trackb_replay_39118", "nextgen_h2.metrics", "nextgen_h2.score",
                 "vc_build.external"))
    score = read(E5 / "score/score_receipt.json", E5_SCORE)
    pin(E5 / "selection/e5_selection_receipt.json", score["selection_sha256"])
    for p in (E5 / "preflight/split.json", E5 / "selection/anchor_grid.tsv", Path(PLAN), TRAIN):
        pin(p)
    for i in range(7):
        path = E5 / "forecasts" / str(i)
        note = read(path / "receipt.json", score["forecast_receipt_sha256"][str(i)])
        pin(path / "predictions.parquet", note["predictions_sha256"])
    external_inputs()
    registry()
    out = []
    for track, cohort, parts, groups in panels():
        for scope in ("all", "lincs_observed", "lincs_unobserved"):
            selected = [(i, x if scope == "all" else x.loc[x.lincs_observed.eq(scope == "lincs_observed")])
                        for i, x in parts]
            pooled = pd.concat([x for _, x in selected], ignore_index=True)
            if pooled.empty:
                continue
            support = {"rows":len(pooled), "parents":pooled[KEY].nunique(), "cells":pooled.cell_id.nunique(),
                       "scaffolds":pooled[KEY].map(groups).fillna(pooled[KEY]).nunique(),
                       "lineages":pooled.lineage_group.nunique()}
            assert support == score["supports"][cohort][scope]
            tags = {"track":TRACK_TAG[track], "cohort":cohort, "support":scope,
                    "exposure":"previously_exposed_external_audit" if track == "T3" else "held_under_E5_nested"}
            cols = TRACK_ARMS[track]
            blocks = measure(selected, groups, cols, components=ANCHOR_ONLY)
            points, _ = summarize(blocks, tags, {a:{"arm":a} for a in cols}, [])
            for p in points:
                if p["screen"] == "macro":
                    p.update(support)
            out.extend(points)
        print(f"E5 replay {cohort} complete", flush=True)
    name = "E5_METRICS.tsv"
    keys = ["track", "cohort", "support", "screen", "fold", "metric", "arm"]
    return name, compare(out, E5 / "score" / name, keys, ["value"], score["output_sha256"][name])


if __name__ == "__main__":
    require_slurm()
    part = sys.argv[1]
    name, result = {"h3":h3_replay, "e5":e5_replay}[part]()
    status = "PASS" if result["max_abs_diff"] <= 1e-9 else "FAIL"
    evidence = {"status":status, "file":name, "comparison":result, "threshold":1e-9,
                "slurm":job_meta(), "input_sha256":INPUTS}
    target = HOME / "tmp/h3v2_closing" / f"{part}_replay.json"
    publish_json(target, evidence)
    print(status, result, flush=True)
    if status != "PASS":
        raise SystemExit(1)
