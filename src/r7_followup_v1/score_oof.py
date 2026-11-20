"""D3 scoring on the source out-of-fold predictions (internal re-split evaluation of the ORIGINAL TRAIN drugs).

Every unit of the original train split is predicted once, by the models of the fold in which its drug group is held
(backbone, cell means, standardisation and corrections were fitted without it). The held deltas are opened only here.
Primary statistic (unchanged from the original study): per-unit delta RMSE over the 978 landmarks, averaged over
model seeds (GS: over 3 null mappings x 3 model seeds), then unit mean within cell-line family and equal mean over the
source (non-HCC) families. Differences are candidate - comparator; the drug-group bootstrap resamples all cell lines of
a drug group together and uses the same resamples for every contrast.
Core contrasts: GR - M and GR - GS (Bonferroni simultaneous intervals, Holm-adjusted p). Everything else is descriptive."""
import argparse, json, os, time
import numpy as np
import pandas as pd

from src.evidence_gated_r7.data import Store
from src.evidence_gated_r7.evaluate import unit_metrics
from .common import fruns, pman, ppred, dump, tag_cols, HCC, NULL_SEEDS, cfg
from .kgraph import load_kg, k_groups
from .metrics import retrieval_by_pool, chance
from .stats import FamBoot, contrast_row, interval, holm, boot_p, category, THR

KEYS = ["C", "M", "G0", "GR"] + [f"GS{n}" for n in NULL_SEEDS]
CORE = [("GR", "M"), ("GR", "GS")]
OTHER = [("GR", "C"), ("GR", "G0"), ("M", "C"), ("G0", "C"), ("GS", "C"), ("GS", "M"), ("G0", "M"), ("GS", "G0")] + [("GR", f"GS{n}") for n in NULL_SEEDS]
RET = ["legacy_midrank_mrr", "mrr_tie_expected", "top1_tie_expected", "top5_tie_expected", "legacy_unique_top1"]


def run_dir(key, ctx, seed):
    kind = "chem" if key == "C" else ("GS" if key.startswith("GS") else ("TDS" if key.startswith("TDS") else key))
    rid = f"{ctx}_s{seed}_{'chem' if key == 'C' else key}"
    return fruns(kind, rid), rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--keys", nargs="+", default=KEYS)
    ap.add_argument("--tag", default="oof", help="oof: official tables; anything else: written under manifests/.../<tag>/")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--extra_pairs", nargs="*", default=[], help="branch contrasts candidate:comparator (descriptive unless listed in the branch decision)")
    ap.add_argument("--branch_core", nargs="*", default=[], help="pre-registered branch contrasts candidate:comparator (own family: Bonferroni simultaneous interval + Holm)")
    a = ap.parse_args()
    official = a.tag == "oof"
    split_role = "source_out_of_fold(internal re-split of original train drugs)" if official else f"source_out_of_fold({a.tag})"
    S = Store()
    blocks = []
    for k in a.folds:
        ctx = f"fold{k}"
        held = np.load(os.path.join(run_dir("C", ctx, a.seeds[0])[0], "held_idx.npy"))
        assert (S.role[held] == "train").all()

        def load(key, s, ctx=ctx, held=held):
            d, rid = run_dir(key, ctx, s)
            rec = json.load(open(os.path.join(d, "record.json")))
            assert (np.load(os.path.join(d, "held_idx.npy")) == held).all()
            assert rec["config"]["ctx"] == ctx and rec["config"]["seed"] == s
            return np.load(os.path.join(d, "held_pred.npy")).astype(np.float64), rec, pd.read_csv(os.path.join(d, "history.tsv"), sep="\t"), rid
        blocks.append(dict(name=ctx, k=k, held=held, y=S.delta(held), load=load))
    tables(S, blocks, a.keys, a.seeds, a.tag, split_role, official, a.extra_pairs, a.n_boot, halves=True, branch_core=a.branch_core)


def tables(S, blocks, keys, seeds, tag, split_role, official, extra_pairs=(), n_boot=5000, halves=False, unit_file=None, branch_core=()):
    """All result tables from per-block predictions. blocks: dict(name, k, held, y, load(key, seed) -> (pred, record, history, run_id))."""
    import types
    a = types.SimpleNamespace(seeds=list(seeds), tag=tag, n_boot=n_boot, extra_pairs=list(extra_pairs), folds=[b["k"] for b in blocks])
    t0 = time.time()
    out = (lambda f: pman(f)) if official else (lambda f: pman(tag, f))
    klab, _ = k_groups(load_kg(os.environ["R7_KG"]))
    keys = list(keys)
    has_gs = all(f"GS{n}" in keys for n in NULL_SEEDS)
    has_tds = all(f"TDS{n}" in keys for n in NULL_SEEDS)
    pooled_of = lambda key: "TDS" if key.startswith("TDS") else ("GS" if key.startswith("GS") else None)
    # ---------------- per-run metrics
    frames, per = [], {}       # per[(key, seed)] = dict of per-unit arrays over the concatenated held units
    runrows = []
    k0_max = 0.0
    for blk in blocks:
        k, held, y = blk["k"], blk["held"], blk["y"]
        u = S.units.iloc[held][["unit_idx", "drug_group", "cell", "family", "is_hcc", "group_idx", "half_idx"]].reset_index(drop=True)
        u["fold"] = k; u["k_group"] = klab[u.group_idx.to_numpy()]
        frames.append(u)
        cells = u.cell.to_numpy(); src = u.is_hcc.to_numpy() == 0; ann = (u.k_group != "K0").to_numpy()
        fbk = FamBoot(u.drug_group.to_numpy(), u.family.to_numpy(), n_boot=2)
        for s in a.seeds:
            pc = None
            for key in keys:
                p, rec, hist, rid = blk["load"](key, s)
                if key == "C":
                    pc = p
                m = unit_metrics(p, y, cells)
                r = retrieval_by_pool(p, y, cells)
                assert np.allclose(r["legacy_midrank_mrr"], m.mrr.to_numpy()), "legacy retrieval differs from the original evaluator"
                corr = np.sqrt(((p - pc) ** 2).mean(1))
                if key != "C":
                    k0_max = max(k0_max, float(corr[~ann].max()))
                    assert float(corr[~ann].max()) < 1e-5, "an unannotated (K0) unit differs from the backbone prediction"
                e = dict(rmse=m.rmse.to_numpy(), centred_pearson=m.centred_pearson.to_numpy(), corr_rms=corr, n_candidates=r["n_candidates"],
                         tied=(r["n_tied"] > 1).astype(float), **{q: r[q] for q in RET})
                for q, v in e.items():
                    per.setdefault((key, s), {}).setdefault(q, []).append(v)
                runrows.append(dict(fold=k, model_seed=s, arm=pooled_of(key) or key, view_or_null_seed=int(key[len(pooled_of(key)):]) if pooled_of(key) else "",
                                    run_id=rid, best_step=rec["best"]["step"], sel_rmse_best=rec["best"]["score"], sel_rmse_last=float(hist.sel_rmse.iloc[-1]) if hist is not None else np.nan,
                                    held_source_rmse=fbk.stat(e["rmse"], src, boot=False)[0], held_source_annotated_rmse=fbk.stat(e["rmse"], src & ann, boot=False)[0],
                                    corr_rms_annotated=float(np.sqrt((corr[src & ann] ** 2).mean())), n_units=int(src.sum()), n_drugs=int(u.drug_group[src].nunique()),
                                    n_params_correction=rec["config"]["n_params_correction"], train_seconds=rec.get("train_seconds"), gpu_max_mem_gb=rec.get("gpu_max_mem_gb"),
                                    sec_per_1000_units=rec.get("inference", {}).get("sec_per_1000_units"), slurm_job=rec.get("slurm_job"), host=rec.get("host")))
    U = pd.concat(frames, ignore_index=True)
    per = {k_: {q: np.concatenate(v) for q, v in d.items()} for k_, d in per.items()}
    rr = pd.DataFrame(runrows)
    tag_cols(rr, split_role, metric_definition="family-balanced source-panel delta RMSE of the held fold; sel = inner-validation selection score").to_csv(out("capacity_attribution_by_run.tsv"), sep="\t", index=False)

    def avg(key, q="rmse", seeds=None, maps=None):
        ks = [f"{key}{n}" for n in (maps or NULL_SEEDS)] if key in ("GS", "TDS") else [key]
        return np.mean([per[(k_, s)][q] for k_ in ks for s in (seeds or a.seeds)], 0)

    arms = [k_ for k_ in keys] + (["GS"] if has_gs else []) + (["TDS"] if has_tds else [])
    src = (U.is_hcc == 0).to_numpy(); ann = (U.k_group != "K0").to_numpy()
    fb = FamBoot(U.drug_group.to_numpy(), U.family.to_numpy(), n_boot=a.n_boot)
    groups = {"all": np.ones(len(U), bool), "annotated": ann, "K0": (U.k_group == "K0").to_numpy(), "K1": (U.k_group == "K1").to_numpy(),
              "K2": (U.k_group == "K2").to_numpy(), "K3": (U.k_group == "K3").to_numpy(), "K1K2": U.k_group.isin(["K1", "K2"]).to_numpy()}
    scopes = {"pooled": np.ones(len(U), bool), **({b["name"]: (U.fold == b["k"]).to_numpy() for b in blocks} if len(blocks) > 1 else {})}
    agg = "seed-averaged unit RMSE -> unit mean within family -> equal mean over source families"
    # ---------------- arm table
    rows = []
    for sc, sm in scopes.items():
        for g, gm in groups.items():
            mk = sm & gm & src
            for arm in arms:
                v = fb.stat(avg(arm), mk, boot=False)[0]
                by_seed = [fb.stat(avg(arm, seeds=[s]), mk, boot=False)[0] for s in a.seeds]
                row = dict(scope=sc, k_group=g, arm=arm, rmse=v, rmse_sd_across_model_seeds=float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else np.nan,
                           rmse_by_model_seed="|".join("%.6f" % x for x in by_seed), centred_pearson=fb.stat(avg(arm, "centred_pearson"), mk, boot=False)[0],
                           corr_rms_mean_unit=fb.stat(avg(arm, "corr_rms"), mk, boot=False)[0], n_units=int(mk.sum()), n_drugs=int(U.drug_group[mk].nunique()),
                           n_families=int(U.family[mk].nunique()))
                if arm in ("GS", "TDS"):
                    bm = [fb.stat(avg(arm, maps=[n]), mk, boot=False)[0] for n in NULL_SEEDS]
                    row.update(rmse_sd_across_null_mappings=float(np.std(bm, ddof=1)), rmse_by_null_mapping="|".join("%.6f" % x for x in bm))
                rows.append(row)
    at = pd.DataFrame(rows)
    tag_cols(at, split_role, metric_definition=agg, model_seeds="|".join(map(str, a.seeds)), view_or_null_seed="|".join(map(str, NULL_SEEDS)),
             candidate_pool="").to_csv(out("capacity_attribution.tsv"), sep="\t", index=False)
    # ---------------- contrasts
    # core family 1 = D3 core contrasts; family 2 = pre-registered branch contrasts (DECISIONS.md); 0 = descriptive
    pairs = [(x, b, 1) for x, b in CORE] + [tuple(e.split(":")) + (2,) for e in branch_core] + [(x, b, 0) for x, b in OTHER] + [tuple(e.split(":")) + (0,) for e in a.extra_pairs]
    pairs = [(x, b, c) for x, b, c in pairs if x in arms and b in arms]
    fam_size = {c: sum(1 for _, _, c_ in pairs if c_ == c) for c in (1, 2)}
    crow = []
    for sc, sm in scopes.items():
        for g, gm in groups.items():
            mk = sm & gm & src
            if g == "K0":
                continue
            for x, b, core in pairs:
                d = avg(x) - avg(b)
                r = contrast_row(fb, d, mk, core_m=fam_size[core] if core else 1)
                by_seed = [fb.stat(avg(x, seeds=[s]) - avg(b, seeds=[s]), mk, boot=False)[0] for s in a.seeds]
                crow.append(dict(scope=sc, k_group=g, candidate=x, comparator=b, core=int(core), **r,
                                 diff_by_model_seed="|".join("%.6f" % v for v in by_seed), n_seeds_negative=int(sum(v < 0 for v in by_seed)),
                                 n_units=int(mk.sum()), n_drugs=int(U.drug_group[mk].nunique())))
    ct = pd.DataFrame(crow)
    ct["holm_p"] = np.nan
    for (sc, g, _), d in ct[ct.core > 0].groupby(["scope", "k_group", "core"]):
        ct.loc[d.index, "holm_p"] = holm(d.boot_p.to_numpy())
    fold_sign = ct[ct.scope != "pooled"].groupby(["k_group", "candidate", "comparator"])["diff"].apply(lambda s: int((s < 0).sum())).rename("n_folds_negative").reset_index()
    ct = ct.merge(fold_sign, on=["k_group", "candidate", "comparator"], how="left")
    tag_cols(ct, split_role, metric_definition="difference (candidate - comparator) of the " + agg + "; 95 % percentile interval of the paired drug-group bootstrap; core=1 (GR-M, GR-GS) and core=2 (pre-registered branch family): Bonferroni simultaneous interval (1 - 0.05/family size) and Holm p within the family",
             operating_threshold=THR, n_boot=a.n_boot).to_csv(out("capacity_attribution_contrasts.tsv"), sep="\t", index=False)
    # ---------------- retrieval: legacy vs tie-corrected
    rrow = []
    nc = per[(keys[0], a.seeds[0])]["n_candidates"]
    ch = {q: np.array([chance(int(n))[q] for n in nc]) for q in ["legacy_midrank_mrr", "mrr_tie_expected", "top1_tie_expected", "top5_tie_expected"]}
    for arm in arms:
        row = dict(arm=arm)
        for q in RET:
            row[q] = fb.stat(avg(arm, q), src, boot=False)[0]
        row["frac_queries_with_tie"] = fb.stat(avg(arm, "tied"), src, boot=False)[0]
        rrow.append(row)
    row = dict(arm="chance(all candidates tied)")
    for q, v in ch.items():
        row[q] = fb.stat(v, src, boot=False)[0]
    row["legacy_unique_top1"] = 0.0; row["frac_queries_with_tie"] = 1.0
    rrow.append(row)
    rt = pd.DataFrame(rrow)
    rt["mean_candidates_per_query"] = float(nc[src].mean()); rt["median_candidates_per_query"] = float(np.median(nc[src]))
    tag_cols(rt, split_role, metric_definition="family-balanced source mean of per-query values, model-seed averaged; legacy_midrank_mrr = 1/(mid-rank) of the original evaluator; *_tie_expected = expectation under random tie-breaking (Appendix B); the two definitions are not comparable with each other",
             candidate_pool="measured cell-centred responses of the held-fold units of the same cell line (identical for every arm)", n_units=int(src.sum())).to_csv(
        out("legacy_vs_tie_corrected_retrieval.tsv"), sep="\t", index=False)
    # ---------------- HCC panel (pre-specified lines; report only)
    hrow = []
    for h in HCC + ("hcc_mean",):
        for arm in arms:
            if h == "hcc_mean":
                v = float(np.mean([fb.unit_mean(avg(arm), (U.cell == x).to_numpy(), boot=False)[0] for x in HCC]))
            else:
                v = fb.unit_mean(avg(arm), (U.cell == h).to_numpy(), boot=False)[0]
            hrow.append(dict(kind="level", panel=h, candidate=arm, comparator="", value=v, n_units=int((U.cell == h).sum()) if h != "hcc_mean" else int(U.cell.isin(HCC).sum())))
        for x, b, core in pairs:
            d = avg(x) - avg(b)
            if h == "hcc_mean":
                res = [fb.unit_mean(d, (U.cell == z).to_numpy()) for z in HCC]
                point = float(np.mean([p_ for p_, _ in res])); bs = np.nanmean(np.stack([b_ for _, b_ in res]), 0)
            else:
                point, bs = fb.unit_mean(d, (U.cell == h).to_numpy())
            lo, hi = interval(bs)
            hrow.append(dict(kind="contrast", panel=h, candidate=x, comparator=b, value=point, lo95=lo, hi95=hi, category_ci95=category(point, lo, hi),
                             n_units=int((U.cell == h).sum()) if h != "hcc_mean" else int(U.cell.isin(HCC).sum())))
    tag_cols(pd.DataFrame(hrow), split_role, metric_definition="unit-mean delta RMSE per HCC line (original panel definition), hcc_mean = mean of the three lines; report only, never used for a choice").to_csv(
        out("hcc_panel.tsv"), sep="\t", index=False)
    # ---------------- correction size (label-free tertiles of the GR correction RMS among annotated source units)
    srow = []
    if "GR" in arms:
        cr = avg("GR", "corr_rms"); mk0 = src & ann
        cut = np.quantile(cr[mk0], [1 / 3, 2 / 3])
        tert = np.digitize(cr, cut)
        for t_, name in enumerate(["small", "middle", "large"]):
            mk = mk0 & (tert == t_)
            for x, b, core in pairs:
                if x != "GR":
                    continue
                srow.append(dict(gr_correction_tertile=name, corr_rms_min=float(cr[mk].min()), corr_rms_max=float(cr[mk].max()), candidate=x, comparator=b,
                                 **contrast_row(fb, avg(x) - avg(b), mk), n_units=int(mk.sum()), n_drugs=int(U.drug_group[mk].nunique())))
        tag_cols(pd.DataFrame(srow), split_role, metric_definition="tertiles of the seed-averaged per-unit RMS of the GR correction (label-free); contrasts as in capacity_attribution_contrasts").to_csv(
            out("correction_size_vs_error.tsv"), sep="\t", index=False)
    # ---------------- split-half positive control (replicate detection plates of held source units; diagnostic, not a predictor)
    hm = (U.half_idx >= 0).to_numpy() & src
    prow = []
    if halves and hm.sum() > 100:
        ha = np.load(os.path.join(os.environ["R4_PROC"], "half_delta_a.npy"), mmap_mode="r"); hb = np.load(os.path.join(os.environ["R4_PROC"], "half_delta_b.npy"), mmap_mode="r")
        hi_ = U.half_idx.to_numpy()[hm]
        A = np.asarray(ha[hi_], np.float64); B = np.asarray(hb[hi_], np.float64)
        Uh = U[hm].reset_index(drop=True)
        pool = (Uh.cell + "|" + Uh.fold.astype(str)).to_numpy()
        fbh = FamBoot(Uh.drug_group.to_numpy(), Uh.family.to_numpy(), n_boot=2)
        cand = {"replicate_half_A": A}
        yfull = S.delta(U.unit_idx.to_numpy()[hm])
        cm = np.zeros_like(B)
        for c in np.unique(pool):
            s_ = pool == c
            cm[s_] = B[s_].mean(0)                       # scoring-side pool mean of the target half (diagnostic reference only)
        cand["pool_mean_of_target_half(diagnostic)"] = cm
        # model predictions for the same units, seed 42-44 averaged at the metric level
        preds = {}
        for k in a.folds:
            ctx = f"fold{k}"; held = np.load(os.path.join(run_dir("C", ctx, a.seeds[0])[0], "held_idx.npy"))
            pos = pd.Series(np.arange(len(held)), index=held)
            sel_ = (Uh.fold == k).to_numpy()
            for key in keys:
                for s in a.seeds:
                    p = np.load(os.path.join(run_dir(key, ctx, s)[0], "held_pred.npy"), mmap_mode="r")
                    preds.setdefault((key, s), np.zeros_like(B))[sel_] = np.asarray(p[pos.loc[Uh.unit_idx.to_numpy()[sel_]].to_numpy()], np.float64)
        def score(P):
            rm = np.sqrt(((P - B) ** 2).mean(1)); r = retrieval_by_pool(P, B, pool)
            return dict(rmse_vs_half_B=fbh.stat(rm, None, boot=False)[0], mrr_tie_expected=fbh.stat(r["mrr_tie_expected"], None, boot=False)[0],
                        top1_tie_expected=fbh.stat(r["top1_tie_expected"], None, boot=False)[0], legacy_midrank_mrr=fbh.stat(r["legacy_midrank_mrr"], None, boot=False)[0])
        for name, P in cand.items():
            prow.append(dict(predictor=name, **score(P)))
        for key in keys:
            sc_ = [score(preds[(key, s)]) for s in a.seeds]
            prow.append(dict(predictor=f"model_{key}", **{q: float(np.mean([x[q] for x in sc_])) for q in sc_[0]}))
        nch = pd.Series(pool).map(pd.Series(pool).value_counts()).to_numpy()
        prow.append(dict(predictor="chance(all candidates tied)", rmse_vs_half_B=np.nan, mrr_tie_expected=fbh.stat(np.array([chance(int(n))["mrr_tie_expected"] for n in nch]), None, boot=False)[0],
                         top1_tie_expected=fbh.stat(1.0 / nch, None, boot=False)[0], legacy_midrank_mrr=fbh.stat(np.array([chance(int(n))["legacy_midrank_mrr"] for n in nch]), None, boot=False)[0]))
        prow.append(dict(predictor="half_B_vs_full_unit_delta(reference)", rmse_vs_half_B=fbh.stat(np.sqrt(((yfull - B) ** 2).mean(1)), None, boot=False)[0]))
        tag_cols(pd.DataFrame(prow), split_role, metric_definition="target = split-half B (detection plates alternately assigned) of held-fold source units with >= 2 detection plates; replicate_half_A uses post-treatment data of the same unit and is a measurement diagnostic, not a deployable predictor; halves share the unit's plates' controls only if on the same plate",
                 candidate_pool="half-B responses of the held units of the same cell line and fold", n_units=int(hm.sum()), n_drugs=int(Uh.drug_group.nunique())).to_csv(
            out("positive_control_split_half.tsv"), sep="\t", index=False)
    # ---------------- per-unit table (server; published only if small)
    ut = U[["unit_idx", "fold", "drug_group", "cell", "family", "is_hcc", "k_group"]].copy()
    for arm in arms:
        ut[f"rmse_{arm}"] = np.round(avg(arm), 5)
    if "GR" in arms:
        ut["corr_rms_GR"] = np.round(avg("GR", "corr_rms"), 5)
    ut.to_csv(ppred(unit_file or ("oof_unit_rmse.tsv.gz" if official else f"{a.tag}_unit_rmse.tsv.gz")), sep="\t", index=False)
    # ---------------- inputs of the pre-specified decision table (section 10), pooled source OOF
    g = lambda x, b, grp="all": ct[(ct.scope == "pooled") & (ct.k_group == grp) & (ct.candidate == x) & (ct.comparator == b)].iloc[0].to_dict() if ((ct.candidate == x) & (ct.comparator == b)).any() else None
    dec = dict(tag=a.tag, folds=a.folds, seeds=a.seeds, n_units_source=int(src.sum()), n_drugs=int(U.drug_group[src].nunique()),
               levels={arm: float(at[(at.scope == "pooled") & (at.k_group == "all") & (at.arm == arm)].rmse.iloc[0]) for arm in arms},
               contrasts={f"{x}-{b}": g(x, b) for x, b, _ in pairs}, contrasts_K1K2={f"{x}-{b}": g(x, b, "K1K2") for x, b, _ in pairs},
               contrasts_K3={f"{x}-{b}": g(x, b, "K3") for x, b, _ in pairs}, contrasts_annotated={f"{x}-{b}": g(x, b, "annotated") for x, b, _ in pairs},
               k0_max_abs_difference_to_backbone=k0_max, seconds=round(time.time() - t0), commit=os.environ.get("R7F_COMMIT"), job=os.environ.get("SLURM_JOB_ID"))
    if tag in ("oof", "plumbing") and has_gs and srow and all(k_ in arms for k_ in ("C", "M", "G0", "GR")):
        c = dec["contrasts"]; neg = lambda r, key="hi95": r[key] < 0; inc0 = lambda r, lo="lo95", hi="hi95": r[lo] <= 0 <= r[hi]
        k12, k3 = dec["contrasts_K1K2"], dec["contrasts_K3"]
        big = [r for r in srow if r["gr_correction_tertile"] == "large" and r["comparator"] == "C"][0]
        rule = dict(
            row1_GR_beats_M_and_GS=bool(neg(c["GR-M"], "hi_simultaneous") and neg(c["GR-GS"], "hi_simultaneous")),
            row2_GR_beats_C_but_not_M_G0_GS=bool(neg(c["GR-C"]) and inc0(c["GR-M"], "lo_simultaneous", "hi_simultaneous") and inc0(c["GR-GS"], "lo_simultaneous", "hi_simultaneous") and inc0(c["GR-G0"])),
            row3_GR_beats_GS_only=bool(neg(c["GR-GS"], "hi_simultaneous") and inc0(c["GR-M"], "lo_simultaneous", "hi_simultaneous")),
            row4_GR_worse_than_C_and_worse_where_correction_is_large=bool(c["GR-C"]["lo95"] > 0 and big["diff"] > 0),
            row5_K1K2_only=bool(neg(k12["GR-M"]) and neg(k12["GR-GS"]) and not (neg(k3["GR-M"]) and neg(k3["GR-GS"]))),
            row6_similar_and_wide=bool(inc0(c["GR-M"], "lo_simultaneous", "hi_simultaneous") and inc0(c["GR-GS"], "lo_simultaneous", "hi_simultaneous")
                                       and (c["GR-M"]["lo_simultaneous"] <= -THR or c["GR-GS"]["lo_simultaneous"] <= -THR)),
            core_meaningful_improvement_excluded=bool(c["GR-M"]["lo_simultaneous"] > -THR and c["GR-GS"]["lo_simultaneous"] > -THR))
        dec["decision_table_rows"] = rule
    dump(dec, out("d3_decision_inputs.json"))
    pd.set_option("display.width", 250)
    print(at[(at.scope == "pooled")][["k_group", "arm", "rmse", "rmse_sd_across_model_seeds", "corr_rms_mean_unit", "n_units", "n_drugs"]].round(5).to_string())
    print(ct[(ct.scope == "pooled")][["k_group", "candidate", "comparator", "diff", "lo95", "hi95", "lo_simultaneous", "hi_simultaneous", "holm_p", "category_ci95", "n_seeds_negative", "n_folds_negative"]].round(5).to_string())
    print(rt.round(4).to_string()); print(pd.DataFrame(prow).round(4).to_string())
    print(json.dumps(dec.get("decision_table_rows"), indent=1))


if __name__ == "__main__":
    main()
