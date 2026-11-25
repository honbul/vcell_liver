"""Verify archived component targets and audit the native interaction zero tolerance."""
from collections import defaultdict

from scipy.stats import rankdata

from h3_closing_inputs import *
from h3_closing_native import NATIVE, contract
from nextgen_h2.metrics import ScoreBlock, center_interactions, correlation

TOL = 1e-9


def parts_inventory(lock):
    from h3v2.objectives import component_targets

    pin_modules(("h3v2.objectives","h3v2.fit_neural","h3v2.network"))
    picks = [(s,f,v) for s,b in lock["splits"].items() for f,v in b["families"].items()
             if v["config"].get("parts") is True]
    assert {(s,f) for s,f,_ in picks} == {("GDSC2-1","film"),("GDSC2-1","gnn"),
                                              ("PRISM20Q2-0","film"),("PRISM20Q2-0","gnn")}
    drug = [split for split in lock["splits"] if split.startswith("drug-")]
    drug_on = sum(lock["splits"][s]["families"][lock["splits"][s]["champion"]]["config"].get("parts",False)
                  for s in drug)
    ext = lock["external"]["operating_point"]["config"].get("parts",False)
    assert len(drug)==3 and drug_on==0 and not ext
    rows = [{"analysis":"parts_true","scope":s,"model":f,"parts":True,"config_id":v["id"],
             "is_split_champion":lock["splits"][s]["champion"]==f} for s,f,v in picks]
    rows.extend({"analysis":"parts_true_check","scope":scope,"model":"Champion","parts":False,
                 "expected_splits":n,"observed_on":0}
                for scope,n in (("Drug-Cold",3),("External",1)))

    # Drug-baseline residuals have exactly zero within-drug means. The unbalanced
    # mask causes the jointly estimated drug effect to reappear after fitting c.
    a = np.array([0,0,0,1,1,2,2])
    b = np.array([0,1,2,0,1,1,2])
    y = np.array([1.,4.,7.,3.,5.,10.,11.])
    baseline = (np.bincount(b,weights=y)/np.bincount(b))[b]
    c,resid = component_targets(a,b,y-baseline,3,3)
    error = y-(baseline+c[a]+resid)
    assert np.max(np.abs(error)) > 1e-4
    rows.append({"analysis":"unbalanced_parts_counterexample","scope":"synthetic","model":"component_targets",
                 "max_abs_diff":float(np.abs(error).max())})
    proof = {"inventory":[{"split":s,"family":f,"config_id":v["id"]} for s,f,v in picks],
             "drug_champion_parts_on":drug_on,"external_champion_parts_on":int(ext),
             "toy":{"cell":a.tolist(),"drug":b.tolist(),"y":y.tolist(),"baseline":baseline.tolist(),
                    "cell_component":c.tolist(),"residual":resid.tolist(),"reconstruction_error":error.tolist(),
                    "max_abs_reconstruction_error":float(np.abs(error).max())}}
    return rows,proof


def assess(y, p, active, residual_y=None, truth_ranks=None):
    ry = center_interactions(y) if residual_y is None else residual_y
    rp = center_interactions(p)
    clamp = lambda a: np.where(np.abs(a)<=TOL,0.,a)
    rx, px = clamp(ry),clamp(rp)
    axis = lambda x:rankdata(x,axis=0,nan_policy="omit")
    yr, yrc = truth_ranks if truth_ranks is not None else (axis(ry),axis(rx))
    current_r = correlation(yr,axis(rp))[active]
    clamped_r = correlation(yrc,axis(px))[active]
    current_p = correlation(ry,rp)[active]
    clamped_p = correlation(rx,px)[active]
    return (current_r,clamped_r,current_p,clamped_p), float(np.nanmax(np.abs(rp)))


def spread(p):
    hi = np.nanmax(p,axis=0)
    lo = np.nanmin(p,axis=0)
    return float(np.max((hi-lo)[np.isfinite(hi-lo)],initial=0.))


def synthetic(example):
    rng = np.random.default_rng(196511)
    masks = {"dense":np.ones((29,23),dtype=bool), "realistic_sparse_DRP_block":np.isfinite(example.truth)}
    rows = []
    for name, mask in masks.items():
        truth = np.where(mask, rng.random(mask.shape),np.nan)
        cell = np.linspace(-.9,.15,mask.shape[0])[:,None]
        drug = np.linspace(-.7,1.,mask.shape[1])[None,:]
        surfaces = {"constant":np.broadcast_to(.3,mask.shape),"drug_only":drug,
                    "cell_only":cell,"additive":cell+drug}
        for arm,p in surfaces.items():
            (r,r0,pr,pr0),size = assess(truth,np.where(mask,p,np.nan),mask.sum(axis=0)>=5)
            rows.append({"analysis":"synthetic_null","scope":name,"model":arm,"rows":int(mask.sum()),
                         "matrix_rows":mask.shape[0],"matrix_cols":mask.shape[1],
                         "max_abs_prediction_residual":size,
                         "interaction_spearman":float(r.mean()),"interaction_spearman_clamped":float(r0.mean()),
                         "interaction_pearson":float(pr.mean()),"interaction_pearson_clamped":float(pr0.mean()),
                         "max_abs_drug_rho":float(np.abs(r).max()),"max_abs_drug_rho_clamped":float(np.abs(r0).max())})
    return rows


def analytical_blocks(study, panels, groups, cols, supports, components=None):
    components = components or {}
    by_model, bounds = defaultdict(list), defaultdict(list)
    for support, sel in supports.items():
        for fold,frame in panels:
            block = frame if sel is None else frame.loc[frame[sel].eq(support=="lincs_obs")]
            for screen, part in block.groupby("source_id",sort=True):
                layout = ScoreBlock(part,np.arange(len(part)),study,fold,screen,groups,True)
                if not layout.drug_eligible.any():
                    continue
                cached = {}
                truth_ranks = (layout.residual_rank,rankdata(np.where(np.abs(layout.residual)<=TOL,0.,layout.residual),
                                                             axis=0,nan_policy="omit"))
                for col in cols:
                    key = components.get(col,col)
                    if key not in cached:
                        pred = layout.wide(part[key].to_numpy())
                        if not np.isfinite(pred[np.isfinite(layout.truth)]).all():
                            raise ValueError(f"Forecast support mismatch on {key} {screen} {fold}")
                        metrics,size = assess(layout.truth,pred,layout.drug_eligible,layout.residual,truth_ranks)
                        max_spread = spread(pred) if col=="drug_mean" else np.nan
                        cached[key] = np.array([float(x.mean()) for x in metrics]),size,max_spread
                    values,max_res, max_spread = cached[key]
                    by_model[support,screen,col].append(values)
                    if col=="drug_mean":
                        bounds[support,screen,col].append((max_res,max_spread))
    bag = defaultdict(list)
    for (support,screen,col), parts in by_model.items():
        bag[support,col].append(np.mean(parts,axis=0))
    for (support,col), parts in bag.items():
        by_model[support,"macro",col] = parts
        if col=="drug_mean":
            bounds[support,"macro",col] = [q for (u,s,m),b in list(bounds.items()) if (u,m)==(support,col)
                                           for q in b]
    return by_model,bounds


def clamp_table(study, scope, panels, groups, cols, supports, components=None):
    points,diagnostics = analytical_blocks(study,panels,groups,cols,supports,components)
    rows, group = [], defaultdict(list)
    for (support,screen,arm), values in points.items():
        r,rc,p,pc = np.mean(values,axis=0)
        tag = {"analysis":"clamp_model","study":study,"scope":scope,"support":support,
               "screen":screen,"model":arm,"evaluator":NATIVE,"model_contract":contract(arm,scope)}
        row = tag|{"interaction_spearman":r,"interaction_spearman_clamped":rc,"spearman_change":rc-r,
                   "interaction_pearson":p,"interaction_pearson_clamped":pc,"pearson_change":pc-p}
        if (support,screen,arm) in diagnostics:
            z0,z1 = zip(*diagnostics[support,screen,arm])
            row.update(max_abs_prediction_residual=max(z0),max_within_drug_prediction_spread=max(z1))
        rows.append(row)
        group[support,screen].append(row)
    for (support,screen), items in group.items():
        distinct = [p for p in items if p["model"] not in ("drug_mean","Champion-ensemble")]
        r_low,r_high = min(distinct,key=lambda x:x["spearman_change"]),max(distinct,key=lambda x:x["spearman_change"])
        p_low,p_high = min(distinct,key=lambda x:x["pearson_change"]),max(distinct,key=lambda x:x["pearson_change"])
        rows.append({"analysis":"max_nondegenerate_contrast_change","study":study,"scope":scope,
                     "support":support,"screen":screen,
                     "spearman_change":r_high["spearman_change"]-r_low["spearman_change"],
                     "pearson_change":p_high["pearson_change"]-p_low["pearson_change"],
                     "spearman_pair":f'{r_high["model"]} − {r_low["model"]}',
                     "pearson_pair":f'{p_high["model"]} − {p_low["model"]}'})
        null = next((p for p in items if p["model"]=="drug_mean"),None)
        if null:
            worst = max(distinct,key=lambda p:abs(p["spearman_change"]-null["spearman_change"]))
            rows.append({"analysis":"drug_mean_contrast_change","study":study,"scope":scope,
                         "support":support,"screen":screen,"model":worst["model"],"baseline":"drug_mean",
                         "spearman_change":worst["spearman_change"]-null["spearman_change"],
                         "pearson_change":worst["pearson_change"]-null["pearson_change"]})
    return rows


def audit_null(frames, lock, ext_data, source_groups):
    from followup_h2_e4.forecast import licob18
    from followup_h2_e5.forecast import TRACK_ARMS
    from followup_h2_e5.score import ANCHOR_ONLY
    from trackb_replay_39118 import panels as e5_panels

    rows,proof = parts_inventory(lock)
    panel = frames["drug-0"]
    sample = panel.loc[panel.source_id.eq("CTRPv2")].reset_index(drop=True)
    rows.extend(synthetic(ScoreBlock(sample,np.arange(len(sample)),"",0,"",source_groups,True)))
    labels,licence,ext_groups = ext_data
    raw_licob = frames["external"].loc[frames["external"].source_id.eq("LICOB")]
    collapsed_licob = adult_licob(raw_licob.rename(columns={"lincs":"lincs_observed"}),licence)
    for scope in ("Drug-Cold","Cell-Cold","GDSC1","LIMORE","LICOB","LICOB18"):
        if scope in ("Drug-Cold","Cell-Cold"):
            split = "drug" if scope=="Drug-Cold" else "cell"
            parts = [(i,frames[f"{split}-{i}"]) for i in range(3)]
            groups = source_groups
        else:
            part = frames["external"].loc[frames["external"].source_id.eq(scope)]
            if scope=="LICOB":
                part = collapsed_licob
            elif scope=="LICOB18":
                part = licob18(collapsed_licob)
            parts = [(0,part)]
            groups = ext_groups
        col_names = [k for k in parts[0][1] if k not in (*META,"lincs","lincs_observed","truth",
                                                       "in_source","remaining_unspecified_stereo")]
        flag = "lincs" if "lincs" in parts[0][1] else "lincs_observed"
        supports = {"all":None}
        if scope!="Cell-Cold":
            supports.update(lincs_obs=flag,lincs_unobs=flag)
        rows.extend(clamp_table("H3-v2",scope,parts,groups,col_names,supports))
        print("A3 clamp H3",scope,flush=True)
    max_raw = float(raw_licob.groupby(KEY).drug_mean.agg(lambda x:np.ptp(x.to_numpy())).max())
    max_collapsed = float(collapsed_licob.groupby(KEY).drug_mean.agg(lambda x:np.ptp(x.to_numpy())).max())
    delta_row = float(np.abs(collapsed_licob.drug_mean-collapsed_licob[KEY].map(raw_licob.groupby(KEY).drug_mean.first())).max())
    rows.append({"analysis":"drug_mean_LICOB_donor_aggregation","study":"H3-v2","scope":"LICOB",
                 "model":"drug_mean","raw_rows":len(raw_licob),"donor_rows":len(collapsed_licob),
                 "max_within_drug_prediction_spread_raw":max_raw,
                 "max_within_drug_prediction_spread":max_collapsed,"max_abs_aggregation_difference":delta_row})
    print("LICOB drug_mean raw/donor spread and donor deviation",max_raw,max_collapsed,delta_row,
          str(raw_licob.drug_mean.dtype),flush=True)
    donor64 = adult_licob(raw_licob.assign(drug_mean=raw_licob.drug_mean.astype("float64"))
                        .rename(columns={"lincs":"lincs_observed"}),licence)
    donor64 = donor64.assign(pure_drug_constant=donor64[KEY].map(raw_licob.groupby(KEY).drug_mean.first()))
    for mode, frame, col in (("archived_donor",collapsed_licob,"drug_mean"),
                             ("float64_donor",donor64,"drug_mean"),("pure_drug_constant",donor64,"pure_drug_constant")):
        layout = ScoreBlock(frame,np.arange(len(frame)),"LICOB",0,"LICOB",ext_groups,True)
        p = layout.wide(frame[col].to_numpy())
        (r,r0,pr,pr0), residual = assess(layout.truth,p,layout.drug_eligible,layout.residual)
        rows.append({"analysis":"LICOB_drug_mean_aggregation_diagnostic","study":"H3-v2","scope":"LICOB",
                     "model":"drug_mean","aggregation":mode,"prediction_dtype":str(frame[col].dtype),
                     "max_within_drug_prediction_spread":spread(p),"max_abs_prediction_residual":residual,
                     "interaction_spearman":float(r.mean()),"interaction_spearman_clamped":float(r0.mean()),
                     "interaction_pearson":float(pr.mean()),"interaction_pearson_clamped":float(pr0.mean())})

    for track,cohort,parts,groups in e5_panels():
        scope = "Drug-Cold" if cohort=="DRP" else cohort
        supports = {"all":None,"lincs_obs":"lincs_observed","lincs_unobs":"lincs_observed"}
        rows.extend(clamp_table("E5",scope,parts,groups,TRACK_ARMS[track],supports,ANCHOR_ONLY))
        print("A3 clamp E5",scope,flush=True)
    table = pd.DataFrame(rows)
    m = table.loc[table.analysis.eq("clamp_model") & ~table.model.isin(("drug_mean","Champion-ensemble"))]
    cross = []
    matched = m.loc[m.scope.isin(("Drug-Cold","GDSC1","LIMORE","LICOB","LICOB18"))]
    for (scope,support,screen), group in matched.groupby(["scope","support","screen"]):
        h = group.loc[group.study.eq("H3-v2")]
        e = group.loc[group.study.eq("E5")]
        if h.empty or e.empty:
            continue
        rec = {"analysis":"max_cross_study_contrast_change","study":"H3-v2 vs E5",
               "scope":scope,"support":support,"screen":screen}
        for metric in ("spearman","pearson"):
            field = metric+"_change"
            wide = np.abs(h[field].to_numpy()[:,None]-e[field].to_numpy()[None,:])
            i,j = np.unravel_index(np.argmax(wide),wide.shape)
            rec[field] = wide[i,j]
            rec[metric+"_pair"] = f'{h.iloc[i].model} − {e.iloc[j].model}'
        cross.append(rec)
    table = pd.concat([table,pd.DataFrame(cross)],ignore_index=True)
    valid = table.loc[table.analysis.isin(("max_nondegenerate_contrast_change","max_cross_study_contrast_change"))]
    proof["null_clamp"] = {"threshold":TOL,"comparison_threshold":1e-6,
                          "max_spearman_contrast_change_excluding_analytic_null":float(valid.spearman_change.max()),
                          "max_pearson_contrast_change_excluding_analytic_null":float(valid.pearson_change.max()),
                          "max_spearman_contrast_change_including_drug_mean":float(
                                  table.loc[table.analysis.eq("drug_mean_contrast_change"),"spearman_change"].abs().max()),
                          "drug_mean_LICOB_max_raw_drug_spread":max_raw,
                          "drug_mean_LICOB_max_donor_drug_spread":max_collapsed,
                          "drug_mean_LICOB_max_donor_aggregation_deviation":delta_row,
                          "drug_mean_LICOB_raw_dtype":str(raw_licob.drug_mean.dtype),
                          "drug_mean_LICOB_aggregation_comparison":table.loc[
                             table.analysis.eq("LICOB_drug_mean_aggregation_diagnostic"),
                             ["aggregation","prediction_dtype","max_within_drug_prediction_spread",
                              "max_abs_prediction_residual","interaction_spearman","interaction_spearman_clamped"]].to_dict("records")}
    return table,proof
