"""Paired native-panel audit and strengthened-family sensitivity from frozen forecasts."""
from collections import defaultdict

from scipy.stats import rankdata

from h3_closing_inputs import *
from h3v2.grid import FAMILIES, SEEDS
from nextgen_h2.metrics import CELL, DRUG, ScoreBlock, center_interactions, correlation

H3_MODELS = ("Champion", *FAMILIES)
ARMS = (*H3_MODELS, "Champion-ensemble")
FROZEN = [f"{a}__{s}" for a in H3_MODELS for s in SEEDS]
CANDIDATES = [*FROZEN, "Champion-ensemble"]
BASELINES = ("H2-v1", "H2-v2", "Cchem-int", "P3B-Shared", "H2-v3-Combined", "P3B-InScreen")
AXES = ("spearman", "interaction_spearman", "ndcg_at_5")
NATIVE = "native_minmax_NDCG_and_within_drug_interaction"
DRAWS = 2000
RANDOM = 318743
INTERNAL = ("Drug-Cold", "Cell-Cold")
EXTERNAL = ("GDSC1", "LIMORE", "LICOB", "LICOB18")
E5_EXTRA = ("C1", "H2-LOSO-Full", "P1-Minimax", "P1-RelMinimax", "P2-Decoupled",
            "P3A-ScreenDebiased", "P3C-EqualScreenWeight", "C2", "C3")


def load_e5(index, score):
    folder = E5 / "forecasts" / str(index)
    receipt = read(folder / "receipt.json", score["forecast_receipt_sha256"][str(index)])
    assert receipt["selection_sha256"] == score["selection_sha256"] and receipt["queried_outcomes"] == 0
    return read_frame(folder / "predictions.parquet", receipt["predictions_sha256"])


def join_forecasts(h3, e5, e1, truth, *, scope):
    e5, e1 = align(h3, e5), align(h3, e1)
    for frame in (e5, e1):
        np.testing.assert_array_equal(frame.lincs_observed, h3.lincs)
        np.testing.assert_array_equal(frame.row_id.map(truth), h3.truth)
    np.testing.assert_array_equal(h3.H2, e5["H2-v1"])
    np.testing.assert_array_equal(h3.C1, e5.C1)
    attach = {k:e5[k].to_numpy() for k in (*BASELINES, *E5_EXTRA, "__p1_personal")
              if k in e5 and k not in h3}
    attach.update({"Cchem-int":e1["Cchem-int"].to_numpy(), "rc":e1.rc.to_numpy()})
    parity = {"scope":scope, "raw_rows":len(h3), "parents":int(h3[KEY].nunique()),
              "truth_max_abs_diff":0.0, "lincs_identical":True, "H2_v1_and_C1_exact":True}
    return h3.assign(**attach).rename(columns={"lincs":"lincs_observed"}), parity


def a1_panels(frames, ext_data):
    from followup_h2_e4.data import OUTCOMES
    from followup_h2_e4.forecast import licob18
    from followup_h2_e5.score import aggregates
    from vc_build.external import native_sensitivity

    pin_modules(("followup_h2_e4.forecast", "followup_h2_e5.score", "followup_h2_e5.test_pipeline", "vc_build.external"))
    e5_score = read(E5 / "score/score_receipt.json", E5_SCORE)
    e1_audit = read(HOME / "tables/e1_e4_e5_paired_audit_receipt.json")
    cold = read(COLD / "receipt.json")
    source = read_frame(TRAIN, cold["files"]["train.parquet"], columns=[*META, *OUTCOMES])
    assert source.row_id.is_unique
    source_truth = pd.Series(native_sensitivity(source).to_numpy(), index=source.row_id)
    del source
    raw, proofs = {}, []
    labels, licence, ext_groups = ext_data
    ext_truth = labels.set_index("row_id").truth
    h3_cols = [k for k in frames["external"].rename(columns={"lincs":"lincs_observed"}) if k not in META]
    for i in range(4):
        split = f"drug-{i}" if i < 3 else "external"
        e5 = load_e5(i+3, e5_score)
        e1_path = E1 / "e1-forecast" / str(i)
        receipt = read(e1_path / "receipt.json")
        e1 = read_frame(e1_path / "forecast.parquet", receipt["forecast_sha256"])
        assert receipt["forecast_sha256"] in json.dumps(e1_audit), f"E1 role {i} absent from prior verified audit"
        raw[split], note = join_forecasts(frames[split], e5, e1, source_truth if i < 3 else ext_truth, scope=split)
        proofs.append(note)
    panels = {"Drug-Cold":[(i,raw[f"drug-{i}"]) for i in range(3)]}
    source_rows = pd.concat([q for _,q in panels["Drug-Cold"]], ignore_index=True)
    assert (len(source_rows), source_rows[KEY].nunique(), source_rows.row_id.nunique()) == (670830,1186,670830)
    assert source_rows.groupby(KEY).lincs_observed.nunique().eq(1).all()
    assert source_rows.drop_duplicates(KEY).lincs_observed.value_counts().to_dict() == {True:724, False:462}
    for scope, joint in raw["external"].groupby("source_id"):
        if scope == "LICOB":
            own_h3 = adult_licob(joint[[*META,*h3_cols]], licence)
            e5_licob = load_e5(6, e5_score).assign(truth=lambda p:p.row_id.map(ext_truth))
            e5_licob = aggregates(e5_licob.loc[e5_licob.source_id.eq("LICOB")], licence)
            joint = adult_licob(joint, licence)
            keys = ["source_id", "endpoint_id", "cell_id", KEY]
            for block in (own_h3, e5_licob):
                pd.testing.assert_frame_equal(joint[[*keys,"truth"]], block[[*keys,"truth"]], check_exact=True)
            subset = licob18(joint)
            pd.testing.assert_frame_equal(subset[[*keys,"truth"]], licob18(e5_licob)[[*keys,"truth"]], check_exact=True)
            panels["LICOB18"] = [(0,subset)]
        panels[scope] = [(0,joint)]
    for scope in EXTERNAL:
        part = panels[scope][0][1]
        n = e5_score["supports"][scope]["all"]
        assert (len(part), part[KEY].nunique(), part.cell_id.nunique()) == (n["rows"],n["parents"],n["cells"])
        proofs.append({"scope":scope,"scored_rows":len(part),"parents":int(part[KEY].nunique()),"truth_max_abs_diff":0.0})
    return panels, proofs, ext_groups


def contract(model, scope):
    if model in ("P3B-InScreen", "H2-v3-Combined") and scope == "Drug-Cold":
        return "known_screen_component_tau_f_s"
    if model in ("P3B-Shared", "H2-v3-Combined"):
        return "shared_component_no_query_screen_term"
    if model == "H2 held":
        return "H3 archived fold-refit current H2 kernel on held/core68"
    if model == "Cchem-int":
        return "chemistry_target_potency_with_E1_personal_component_for_interaction"
    if model == "Champion-ensemble":
        return "descriptive_row_forecast_ensemble"
    return "native_frozen_model"


def native_tables(parts, groups, scope, support, cols, *, components=None):
    components = components or {}
    drug, cell = [], []
    for fold, panel in parts:
        for screen, frame in panel.groupby("source_id", sort=True):
            layout = ScoreBlock(frame, np.arange(len(frame)), scope, fold, screen, groups, True)
            for arm in cols:
                d,c = layout.score(frame[arm].to_numpy(), arm, support)
                if arm in components:
                    personal = center_interactions(layout.wide(frame[components[arm]].to_numpy()))
                    axis = layout.drug_eligible
                    d["interaction_spearman"] = correlation(layout.residual_rank,
                            rankdata(personal, axis=0, nan_policy="omit"))[axis]
                    d["interaction_pearson"] = correlation(layout.residual,personal)[axis]
                drug.append(d)
                cell.append(c)
    return pd.concat(drug, ignore_index=True), pd.concat(cell, ignore_index=True)


def boot_points(table, names, *, draws=DRAWS):
    """Native block means with the registered Bayesian scaffold/lineage weights."""
    id_cols = ["screen", "fold", "unit", "cluster"]
    pivot = table.pivot(index=id_cols, columns="arm", values=list(names)).sort_index()
    if not np.isfinite(pivot.to_numpy()).all():
        raise ValueError("Unmatched panel units or non-finite native metric")
    cluster = pd.Index(sorted(table.cluster.unique()))
    weights = np.random.default_rng(RANDOM).exponential(1., size=(draws,len(cluster))) if draws else None
    by_screen, boot_screen = defaultdict(list), defaultdict(list)
    for (screen,_), block in pivot.groupby(level=["screen","fold"], sort=True):
        by_screen[screen].append(block.mean())
        if draws:
            grouped = block.groupby(level="cluster")
            count = grouped.size().reindex(cluster, fill_value=0).to_numpy()
            total = grouped.sum().reindex(cluster,fill_value=0).to_numpy()
            boot_screen[screen].append((weights @ total) / (weights @ count)[:,None])
    points = {s:pd.concat(v,axis=1).mean(axis=1) for s,v in by_screen.items()}
    points["macro"] = pd.concat(list(points.values()),axis=1).mean(axis=1)
    boot = {s:np.mean(v,axis=0) for s,v in boot_screen.items()}
    if draws:
        boot["macro"] = np.mean(list(boot.values()),axis=0)
        boot = {s:dict(zip(pivot.columns,v.T)) for s,v in boot.items()}
    return points, boot, len(cluster)


def paired_native(table, names, scope, support, counts, baselines, *, models=ARMS, only_macro=False):
    points, boots, clusters = boot_points(table,names)
    points_out, intervals = [], []
    for screen, p in points.items():
        if only_macro and screen != "macro":
            continue
        for metric in names:
            for arm in (*models,*baselines):
                arr = [p[metric,f"{arm}__{s}"] for s in SEEDS] if arm in H3_MODELS else [p[metric,arm]]
                tag = counts | {"scope":scope, "support":support, "screen":screen, "metric":metric, "model":arm,
                                "evaluator":NATIVE, "model_contract":contract(arm,scope)}
                entry = {"seed":"mean" if arm in H3_MODELS else "fixed", "value":float(np.mean(arr)),
                         "seed_sd":float(np.std(arr,ddof=1)) if arm in H3_MODELS else None}
                points_out.append(tag|entry)
                if arm in H3_MODELS:
                    points_out.extend(tag|{"seed":s,"value":v,"seed_sd":None} for s,v in zip(SEEDS,arr))
                if arm in baselines:
                    continue
                b = boots[screen]
                dist = np.mean([b[metric,f"{arm}__{s}"] for s in SEEDS],axis=0) if arm in H3_MODELS else b[metric,arm]
                for ref in baselines:
                    base = p[metric,ref]
                    delta = np.asarray(arr)-base
                    lo,hi = np.quantile(dist-b[metric,ref],[.025,.975])
                    intervals.append(tag|{"baseline":ref, "baseline_contract":contract(ref,scope),
                                     "delta":float(delta.mean()), "lo95":lo, "hi95":hi,
                                     "candidate_value":entry["value"],"baseline_value":base,
                                     "delta_seed_sd":entry["seed_sd"],
                                     **{f"delta_seed_{i}":d for i,d in enumerate(delta) if arm in H3_MODELS},
                                     "draws":DRAWS,"clusters":clusters,
                                     "cluster_axis":"scaffold" if metric in DRUG else "global_lineage"})
    return points_out, intervals


def a1(panels, source_groups, ext_groups):
    points, cis, base_units = [], [], {}
    for scope, folds in panels.items():
        groups = source_groups if scope=="Drug-Cold" else ext_groups
        baselines = tuple(k for k in BASELINES if k in folds[0][1])
        for support in ("all","lincs_obs","lincs_unobs"):
            parts = [(i,p if support=="all" else p.loc[p.lincs_observed.eq(support=="lincs_obs")]) for i,p in folds]
            pooled = pd.concat([p for _,p in parts], ignore_index=True)
            assert not pooled.empty
            counts = {"rows":len(pooled), "parents":pooled[KEY].nunique(), "cells":pooled.cell_id.nunique()}
            drug,cell = native_tables(parts, groups, scope, support, [*CANDIDATES,*baselines], components={"Cchem-int":"rc"})
            for table,names in ((drug,DRUG),(cell,CELL)):
                p,c = paired_native(table,names,scope,support,counts,baselines)
                points.extend(p)
                cis.extend(c)
            if scope in ("Drug-Cold","GDSC1") and support=="all":
                base_units[scope] = cell
            print(f"A1 {scope} {support} {counts}",flush=True)
    return pd.DataFrame(points),pd.DataFrame(cis),base_units


def local_roster():
    roster = read_frame(E1 / "registration/cell_roster.parquet", ROSTER)
    assert roster.index.is_unique and not roster.isna().any().any()
    assert (len(roster),roster.lineage_group.nunique(),roster.family.nunique()) == (1014,991,980)
    assert roster.groupby("lineage_group").family.nunique().eq(1).all()
    return roster


def roster_map(frame, roster):
    f = frame.cell_id.map(roster.family)
    l = frame.cell_id.map(roster.lineage_group)
    assert f.notna().all(), f"Missing cell IDs: {frame.loc[f.isna(),'cell_id'].unique()}"
    # The core68 roster retains its recorded original lineage when clustering its evaluator.
    return f,l


def split_overlap(frames, manifest, roster):
    rows, masks = [], {}
    for split, support in manifest["split_support"].items():
        fit = read_frame(H3 / "splits" / split / "fit.parquet",support["fit_sha256"],
                         columns=["cell_id","lineage_group"]).drop_duplicates()
        query = read_frame(H3 / "splits" / split / "query.parquet",support["query_sha256"],
                           columns=["row_id","cell_id","lineage_group"])
        if split in frames:
            align(query,frames[split],keys=["row_id","cell_id","lineage_group"])
        fit_family,fit_line = roster_map(fit,roster)
        query_family,query_line = roster_map(query,roster)
        common = sorted(set(fit_family)&set(query_family))
        lcross = sorted(set(fit_line)&set(query_line))
        original = sorted(set(fit.lineage_group)&set(query.lineage_group))
        masks[split] = ~query_family.isin(common).set_axis(query.row_id)
        details = [{"family":fam,"fit_cells":sorted(fit.loc[fit_family.eq(fam),"cell_id"].unique()),
                    "query_cells":sorted(query.loc[query_family.eq(fam),"cell_id"].unique())}
                   for fam in common]
        count = int(query_family.isin(common).sum())
        rows.append({"split":split,"n_train_cells":fit.cell_id.nunique(),"n_query_cells":query.cell_id.nunique(),
                     "lineage_overlap":len(lcross),"original_h3_lineage_overlap":len(original),
                     "lineages_991_crossing_json":json.dumps(lcross),
                     "family_overlap_count":len(common),"crossing_families_and_cells_json":json.dumps(details),
                     "query_rows":len(query),"query_rows_crossing":count,"query_rows_crossing_pct":100*count/len(query),
                     "fit_991_label_disagreement_cells":fit.loc[~fit.lineage_group.eq(fit_line),"cell_id"].nunique(),
                     "query_991_label_disagreement_cells":query.loc[~query.lineage_group.eq(query_line),"cell_id"].nunique()})
    return pd.DataFrame(rows), masks


def family_ndcg(native, roster, scope, baselines):
    rep = native.copy()
    rep["cluster"] = rep.unit.map(roster.family)
    assert rep.cluster.notna().all()
    both = [paired_native(x,("ndcg_at_5",),scope,"all",{},baselines,models=H3_MODELS,only_macro=True)[1]
            for x in (native,rep)]
    rows = []
    for old,new in zip(*both):
        assert abs(old["delta"]-new["delta"])<1e-12
        # Retain per-scenario interval endpoints so small CI-width shifts are auditable.
        rows.append(old|{"analysis":"family_cluster_bootstrap","after_cluster_axis":"980_family",
                         "before_value":old["delta"],"after_value":new["delta"],"change":new["delta"]-old["delta"],
                         "lo95_980":new["lo95"],"hi95_980":new["hi95"],"clusters_980":new["clusters"],
                         "ci_width_ratio":(new["hi95"]-new["lo95"])/(old["hi95"]-old["lo95"])
                                           if old["hi95"]>old["lo95"] else np.nan})
    return rows
