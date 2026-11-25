"""Core68 paired inference and row-exclusion sensitivity on archived split forecasts."""
from collections import defaultdict

from h3_closing_native import (
    ARMS, AXES, CANDIDATES, DRAWS, H3_MODELS, SEEDS, native_tables, boot_points, family_ndcg,
)
from h3_closing_inputs import *
from h3v2 import core_score
from vc_build.response_components import orthogonal_parts

CORE = "core68_rank_gain_NDCG_and_pooled_orthogonal_interaction"
HELD_BASE = "H2 held"
HELD_COLUMNS = [c for c in CANDIDATES if c != "Champion-ensemble"] + [HELD_BASE]
CORE_NAMES = {"within_drug_spearman":"spearman", "within_drug_pearson":"pearson",
              "precision_at_k":"precision_at_5", "residual_spearman":"interaction_spearman"}


def assemble(frames, *, masks=None):
    from h3v2.data import SCREENS

    parts = []
    for screen in SCREENS:
        for k in range(3):
            split = f"{screen}-{k}"
            part = frames[split]
            if masks is not None:
                part = part.loc[part.row_id.map(masks[split]).to_numpy()]
            parts.append(part.rename(columns={"truth":"observed", "H2":HELD_BASE}).assign(fold=k))
    return pd.concat(parts,ignore_index=True)


def core_residuals(panel):
    use = np.isfinite(panel)
    resid = orthogonal_parts(panel,use.astype(float))[2][use]
    tol = 1e-12 * max(1.,np.abs(panel[use]).max())
    return np.where(np.abs(resid)<=tol,0.,resid)


def residual_bootstrap(frame, cols, reweights):
    """Extend core_score to 2,000 interaction draws of the fixed residual surfaces.

    Each draw reranks the pooled observed residuals with lineage multiplicities.
    Centering nuisance terms and disk forecast values are frozen at the panel estimates.
    """
    out = defaultdict(lambda:defaultdict(list))
    lineages, weights = reweights
    for (screen,_,_), units, y, pred in core_score.blocks(frame,cols):
        yres = core_residuals(y)
        row, _ = np.nonzero(np.isfinite(y))
        weighted = weights[:,lineages.get_indexer(units)]
        pr = {arm:core_score.WeightedRanking(core_residuals(p)) for arm,p in pred.items()}
        yr = core_score.WeightedRanking(yres)
        for start in range(0,DRAWS,48):
            source = weighted[start:start+48]
            w = source[:,row]
            yt = yr.ranks(w)
            for arm,rank in pr.items():
                value = core_score.weighted_association(yt,rank.ranks(w),w)
                if start == 0:
                    out[screen][arm].append((np.empty(DRAWS),weighted.sum(axis=1)))
                out[screen][arm][-1][0][start:start+len(w)] = value
    result = {}
    for screen, models in out.items():
        result[screen] = {}
        for arm, pairs in models.items():
            values, weights = map(np.asarray,zip(*pairs))
            ok = np.isfinite(values)
            den = np.where(ok,weights,0.).sum(axis=0)
            result[screen][arm] = np.divide(np.where(ok,values*weights,0.).sum(axis=0),den,
                                             out=np.full(len(den),np.nan),where=den>0)
    return result


def held_points_and_ci(frame):
    pin_modules(("h3v2.core_score","h3v2.core_reference_metrics","vc_build.response_components"))
    draws = core_score.shared_draws(frame,DRAWS,14639)
    data = core_score.screen_scores(frame,HELD_COLUMNS,draws,minimum=5)
    extra = residual_bootstrap(frame,HELD_COLUMNS,draws)
    for screen, models in data.items():
        for arm,(_,b) in models.items():
            b["residual_spearman"] = extra[screen][arm]
    all_screens = tuple(data)
    metrics = tuple(next(iter(data[all_screens[0]].values()))[0])
    sampled = (*core_score.CORE_METRICS, "residual_spearman")
    data["macro"] = {arm: ({m:float(np.mean([data[s][arm][0][m] for s in all_screens])) for m in metrics},
                            {m:np.mean([data[s][arm][1][m] for s in all_screens],axis=0) for m in sampled})
                      for arm in HELD_COLUMNS}
    rows = []
    for screen, models in data.items():
        sub = frame if screen=="macro" else frame.loc[frame.source_id.eq(screen)]
        for candidate in H3_MODELS:
            members = [models[f"{candidate}__{s}"] for s in SEEDS]
            ref, boot_ref = models[HELD_BASE]
            for raw in metrics:
                point = np.asarray([p[raw] for p,_ in members])
                delta = point-ref[raw]
                lo,hi = (np.nan,np.nan)
                if raw in sampled:
                    effect = np.mean([b[raw] for _,b in members],axis=0)-boot_ref[raw]
                    assert np.isfinite(effect).all()
                    lo,hi = np.quantile(effect,[.025,.975])
                flavor = ("lineage_multinomial_fixed_pooled_residuals_reranked" if raw=="residual_spearman"
                          else "registered_core68_lineage_multinomial" if raw in sampled else "point_only")
                rows.append({"scope":"Held68","support":"core68","screen":screen,"model":candidate,"baseline":HELD_BASE,
                             "baseline_contract":"H3 archived fold-refit H2 kernel using held/core68 training split",
                             "metric":CORE_NAMES.get(raw,raw),"evaluator":CORE,"bootstrap":flavor,
                             "draws":DRAWS if raw in sampled else 0,"rows":len(sub),
                             "parents":sub[KEY].nunique(),"clusters":sub.lineage_group.nunique(),
                             "candidate_value":point.mean(),"baseline_value":ref[raw],"delta":delta.mean(),
                             "lo95":lo,"hi95":hi,"delta_seed_sd":point.std(ddof=1),
                             **{f"delta_seed_{i}":d for i,d in enumerate(delta)}})
    return pd.DataFrame(rows)


def exclusion_rows(before,after,scope,screen,metric,model,base,fold,rows_before,rows_after,evaluator):
    pairs = [("model_score",before[model],after[model]),("reference_score",before[base],after[base]),
             ("contrast",before[model]-before[base],after[model]-after[base])]
    return [{"analysis":"exclude_train_family_rows","scope":scope,"screen":screen,"fold":fold,
             "metric":metric,"model":model,"baseline":base,"quantity":kind,"evaluator":evaluator,
             "rows_before":rows_before,"rows_after":rows_after,
             "before_value":x,"after_value":y,"change":y-x,"sign_changed":bool(np.sign(x)!=np.sign(y))}
            for kind,x,y in pairs]


def held_exclusion(before, after):
    marks = [core_score.screen_scores(x,HELD_COLUMNS,minimum=5) for x in (before,after)]
    out = []
    for screen in (*marks[0],"macro"):
        panels = [(x if screen=="macro" else x.loc[x.source_id.eq(screen)]) for x in (before,after)]
        for raw in core_score.ALL_METRICS:
            values = [{arm:np.mean([v[arm][0][raw] for v in m.values()]) if screen=="macro" else m[screen][arm][0][raw]
                       for arm in HELD_COLUMNS} for m in marks]
            renamed = [{arm:np.mean([v[f"{arm}__{s}"] for s in SEEDS]) for arm in H3_MODELS}|{HELD_BASE:v[HELD_BASE]}
                       for v in values]
            for model in H3_MODELS:
                out.extend(exclusion_rows(*renamed,"Held68",screen,CORE_NAMES.get(raw,raw),model,HELD_BASE,
                                          "all",*map(len,panels),CORE))
    # Report the affected physical folds as well as the aggregated panels.
    for (screen,fold), original in before.groupby(["source_id","fold"]):
        kept = after.loc[after.source_id.eq(screen)&after.fold.eq(fold)]
        if len(original) != len(kept):
            marks_one = [core_score.screen_scores(x,HELD_COLUMNS,minimum=5)[screen] for x in (original,kept)]
            for raw in core_score.ALL_METRICS:
                for model in H3_MODELS:
                    values = [{model:np.mean([v[f"{model}__{s}"][0][raw] for s in SEEDS]),
                               HELD_BASE:v[HELD_BASE][0][raw]} for v in marks_one]
                    out.extend(exclusion_rows(*values,"Held68",screen,CORE_NAMES.get(raw,raw),model,HELD_BASE,
                                              fold,len(original),len(kept),CORE))
    return out


def cell_exclusion(frames, masks, groups, roster):
    from h3_closing_native import NATIVE
    from nextgen_h2.metrics import CELL, DRUG

    base = "H2"
    cols = [*CANDIDATES,base]
    raw = [(i,frames[f"cell-{i}"]) for i in range(3)]
    clean = [(i,x.loc[x.row_id.map(masks[f"cell-{i}"]).to_numpy()]) for i,x in raw]
    tables = [native_tables(parts,groups,"Cell-Cold","all",cols) for parts in (raw,clean)]
    out = []
    for axis, metrics in enumerate((DRUG,CELL)):
        pairs = [boot_points(t[axis],metrics,draws=0)[0] for t in tables]
        for screen in pairs[0]:
            sizes = [sum(len(p) if screen=="macro" else p.source_id.eq(screen).sum() for _,p in s)
                     for s in (raw,clean)]
            for metric in metrics:
                values = [{m:np.mean([p[screen][metric,f"{m}__{s}"] for s in SEEDS]) for m in H3_MODELS}
                          | {base:p[screen][metric,base]} for p in pairs]
                for m in H3_MODELS:
                    out.extend(exclusion_rows(*values,"Cell-Cold",screen,metric,m,base,"all",*sizes,NATIVE))
    out.extend(family_ndcg(tables[0][1],roster,"Cell-Cold",(base,)))
    return out
