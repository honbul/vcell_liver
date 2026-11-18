"""Follow-up branches chosen in BRANCH_DECISION.md (run only after that file is committed).

  select    inner partition: J1 / J2 (/ J3) constants (engine.select_j) and per-arm budget constants (kappa_b, lambda, eta)
  evaluate  outer partition: J0 / J1 / J2 (/ J3) on general and matched-plate E1 / E2 episodes; D16 vs REP8 budget episodes
  report    joint_model_results.tsv, support_budget.tsv
Source families only. Every J model receives the same query metadata (planned plate overlap); J1 is the same-metadata baseline.
"""
import argparse, json, math, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from src.replicate_calibrated_r8 import diag as D
from .common import CFG, orig_man, dump, new_run_dir, run_record, sha256, pub_man
from .partitions import Bank, INNER_OF
from . import engine as Eng
from . import fmethods as F
from . import stats as S
from .matched import matched_episodes, load_rows
from .budget import budget_episodes, count_eligible
from .report import as_sets, contrasts, summary

torch.set_grad_enabled(False)
KEEP = ("no_change", "same_drug_mean", "shrunk_mean", "C0_iid", "C0_V3", "C1_iid", "C1_V3", "J1", "J1_unknown_env", "J2", "J2_unknown_env", "J3", "J3_unknown_env")
ARM_K = {"D16": 16, "REP8": 8}


def log(*a):
    print(*a, flush=True)


def budget_preds(inp, K, kf, kbv, lam, eta, ctx):
    fs, fq = F.fstar(inp, kf)
    a = F.anchor_om(inp, kf, kbv, torch.eye(K, dtype=torch.float64))
    Kall = F.combine(F.kernel_blocks(ctx.feats(inp, fs, fq)), "all")
    return a["fq"] + a["b"], a["fq"] + a["b"] + eta * F.krr(Kall[0], Kall[1], a["w"], a["R"], lam, a["Om"])


def select(a, cfg, out):
    t0 = time.time()
    pairs = pd.read_csv(a.pairs, sep="\t")
    const = json.load(open(a.constants))
    assert const["partition"] == a.partition
    bank = Bank()
    ctx = Eng.Ctx(bank, a.partition)
    kf = const["kappa_f"]["16"]
    ctx.fit_cov(pairs, kf)
    ctx.fit_scales(kf)
    res = dict(partition=a.partition, base_constants=a.constants, base_constants_sha256=sha256(a.constants))
    res.update(Eng.select_j(ctx, const, cfg["grids"], with_j3=a.j3, log=log))
    if a.budget:
        rows, plates, RD = load_rows(os.environ["R8_PROC"], D.row_delta)
        X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
        KB = [Eng.kb(x) for x in cfg["grids"]["kappa_b"]]
        LAM, ETA = [float(x) for x in cfg["grids"]["lambda"]], [float(x) for x in cfg["grids"]["eta"]]
        r0, r1 = [], []
        for meta, eps in budget_episodes(ctx.st, rows, plates, RD, X0, cell_role="val", seeds=(1,), design=a.budget):
            for arm, (ep, cost) in eps.items():
                K = ARM_K[arm]
                inp = Eng.inputs(ep, K)
                b = dict(cell=meta["cell"], family=meta["family"], support_seed=f"{meta['plate_x']}|{meta['plate_y']}|{meta['plate_query']}", arm=arm, K=K)
                fs, fq = F.fstar(inp, const["kappa_f"][str(K)])
                E = inp["y_s"] - fs
                for kbv in KB:
                    r0.append(dict(b, kappa_b=kbv, delta_rmse=F.rmse(fq + E.mean(0, keepdim=True) * F.alpha(K, kbv), ep.y_q)))
        bc = {}
        for arm in ARM_K:
            c, _ = Eng._choose([r for r in r0 if r["arm"] == arm], ["kappa_b"], ["kappa_b"], [True])
            bc[arm] = dict(kappa_b=c[str(ARM_K[arm])]["kappa_b"])
        for meta, eps in budget_episodes(ctx.st, rows, plates, RD, X0, cell_role="val", seeds=(1,), design=a.budget):
            for arm, (ep, cost) in eps.items():
                K = ARM_K[arm]
                inp = Eng.inputs(ep, K)
                b = dict(cell=meta["cell"], family=meta["family"], support_seed=f"{meta['plate_x']}|{meta['plate_y']}|{meta['plate_query']}", arm=arm, K=K)
                kf_ = const["kappa_f"][str(K)]
                fs, fq = F.fstar(inp, kf_)
                aa = F.anchor_om(inp, kf_, bc[arm]["kappa_b"], torch.eye(K, dtype=torch.float64))
                Kall = F.combine(F.kernel_blocks(ctx.feats(inp, fs, fq)), "all")
                for lam in LAM:
                    corr = F.krr(Kall[0], Kall[1], aa["w"], aa["R"], lam, aa["Om"])
                    for eta in ETA:
                        r1.append(dict(b, lam=lam, eta=eta, delta_rmse=F.rmse(aa["fq"] + aa["b"] + eta * corr, ep.y_q)))
        for arm in ARM_K:
            c, _ = Eng._choose([r for r in r1 if r["arm"] == arm], ["lam", "eta"], ["eta", "lam"], [True, False])
            bc[arm].update(lam=c[str(ARM_K[arm])]["lam"], eta=c[str(ARM_K[arm])]["eta"], dev=c[str(ARM_K[arm])]["delta_rmse"])
        res["budget"] = dict(design=a.budget, arms=bc, n_selection_rows=len(r0) // max(len(KB), 1), n_selection_families=len({r["family"] for r in r0}))
        log("budget", bc)
    dump(res, os.path.join(out, "branch_constants.json"))
    dump(run_record(a.run_id, "branch_select", t0, partition=a.partition), os.path.join(out, "result.json"))


def evaluate(a, cfg, out):
    t0 = time.time()
    pairs = pd.read_csv(a.pairs, sep="\t")
    const = json.load(open(a.constants))
    jc = json.load(open(a.jconst))
    assert const["partition"] == INNER_OF[a.partition] == jc["partition"]
    fx = Eng.load_fixed(orig_man("anchor_selection.json"))
    bank = Bank()
    ctx = Eng.Ctx(bank, a.partition)
    Eng.fit_outer(ctx, pairs, const)
    st = ctx.st
    rows = []
    for ep in ctx.episodes("val", [1, 2, 3], None):
        truth = ep.y_q.numpy()
        for K in (8, 16, 32):
            if K > len(ep.g_s):
                continue
            for m, p, ex in Eng.method_preds(ctx, Eng.inputs(ep, K), K, const, fx, with_extras=False, jc=jc):
                if m in KEEP:
                    rows.append(dict(fold=a.partition, cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, method=m, **F.score(p.numpy(), truth), **ex))
    pd.DataFrame(rows).to_csv(os.path.join(out, "episodes.tsv.gz"), sep="\t", index=False, float_format="%.7g")
    log("general", f"{time.time() - t0:.0f}s")
    X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
    mrows, mplates, MRD = load_rows(os.environ["R8_PROC"], D.row_delta)
    rec = []
    for meta, e1, e2, pc in matched_episodes(st, mrows, mplates, MRD, X0, cell_role="val", max_pairs=2, seeds=(1,)):
        for block, ep in (("in_block", e1), ("other_block", e2)):
            for K in (8, 16, 32):
                if K > meta["kmax"]:
                    continue
                for m, p, ex in Eng.method_preds(ctx, Eng.inputs(ep, K), K, const, fx, with_extras=False, jc=jc):
                    if m in KEEP:
                        rec.append(dict(fold=a.partition, **meta, block=block, K=K, method=m, **F.score(p.numpy(), pc["truth"]), **ex))
    pd.DataFrame(rec).to_csv(os.path.join(out, "matched_episodes.tsv.gz"), sep="\t", index=False, float_format="%.7g")
    log("matched", f"{time.time() - t0:.0f}s")
    if "budget" in jc:
        bc = jc["budget"]
        nfam, ncell = count_eligible(st, mrows, "val")
        brec = []
        for meta, eps in budget_episodes(st, mrows, mplates, MRD, X0, cell_role="val", seeds=(1, 2, 3), design=bc["design"]):
            for arm, (ep, cost) in eps.items():
                K = ARM_K[arm]
                inp = Eng.inputs(ep, K)
                truth = ep.y_q.numpy()
                b = dict(fold=a.partition, **meta, arm=arm, K=K, **cost)
                kf = const["kappa_f"][str(K)]
                brec.append(dict(b, method="shrunk_mean", **F.score(F.fstar(inp, const["shrunk_kappa_f"][str(K)])[1].numpy(), truth)))
                c16 = const["c1"]["iid"]["16"]
                p0, p1 = budget_preds(inp, K, const["kappa_f"]["16"], const["kappa_b"]["iid"]["16"], c16["lam"], c16["eta"], ctx)
                brec.append(dict(b, method="C0_common", **F.score(p0.numpy(), truth)))
                brec.append(dict(b, method="C1_common", **F.score(p1.numpy(), truth)))
                ac = bc["arms"][arm]
                p0, p1 = budget_preds(inp, K, kf, ac["kappa_b"], ac["lam"], ac["eta"], ctx)
                brec.append(dict(b, method="C0_reselected", **F.score(p0.numpy(), truth)))
                brec.append(dict(b, method="C1_reselected", **F.score(p1.numpy(), truth)))
        pd.DataFrame(brec).to_csv(os.path.join(out, "budget_episodes.tsv.gz"), sep="\t", index=False, float_format="%.7g")
        log("budget", len(brec), "eligible families (approx.)", nfam, f"{time.time() - t0:.0f}s")
    dump(run_record(a.run_id, "branch_evaluate", t0, partition=a.partition, constants_sha256=sha256(a.constants), jconst_sha256=sha256(a.jconst)),
         os.path.join(out, "result.json"))


def report(a, cfg, out):
    gen = pd.concat([pd.read_csv(os.path.join(d, "episodes.tsv.gz"), sep="\t") for d in a.evaluate], ignore_index=True)
    mat = pd.concat([pd.read_csv(os.path.join(d, "matched_episodes.tsv.gz"), sep="\t") for d in a.evaluate], ignore_index=True)
    df = as_sets(gen, mat)
    pairs = [("J1", "C0_iid", "J1 vs J0-iid (same metadata baseline vs no query metadata)"), ("J2", "C0_iid", "J2 vs J0-iid"), ("J2", "J1", "J2 vs J1 (same metadata)"),
             ("J1", "C0_V3", "J1 vs J0-V3"), ("J2", "C0_V3", "J2 vs J0-V3"), ("J3", "J2", "J3 vs J2 (added per-drug kernel)"), ("J3", "C1_iid", "J3 vs C1-iid"),
             ("J3", "J1", "J3 vs J1"), ("C1_iid", "C0_iid", "C1-iid vs C0-iid (reference)"), ("J2_unknown_env", "C0_iid", "J2 unknown-environment mode vs J0-iid"),
             ("J1_unknown_env", "C0_iid", "J1 unknown-environment mode vs J0-iid"), ("J3_unknown_env", "C1_iid", "J3 unknown-environment mode vs C1-iid")]
    t = pd.concat([contrasts(df, pairs), contrasts(df, pairs, value="centred_rmse")], ignore_index=True)
    sm = summary(df, list(KEEP))
    sm.insert(0, "row", "summary")
    t.insert(0, "row", "contrast")
    cons = []
    for d in a.select:
        c = json.load(open(os.path.join(d, "branch_constants.json")))
        for name in ("j1", "j2", "j3"):
            for K, v in c.get(name, {}).items():
                cons.append(dict(row="selected_constants", partition=c["partition"], model=name, K=int(K), **{k: x for k, x in v.items()}))
    out_t = pd.concat([sm, t, pd.DataFrame(cons)], ignore_index=True)
    out_t["note"] = "J models receive the planned query-support plate overlap (prospective metadata); J1 = same-metadata simple baseline; J replaces b_S (no double correction)"
    out_t.to_csv(pub_man("joint_model_results.tsv"), sep="\t", index=False, float_format="%.6g")
    bud = [os.path.join(d, "budget_episodes.tsv.gz") for d in a.evaluate if os.path.exists(os.path.join(d, "budget_episodes.tsv.gz"))]
    if bud:
        b = pd.concat([pd.read_csv(x, sep="\t") for x in bud], ignore_index=True)
        b["support_seed"] = b.plate_x + "|" + b.plate_y + "|" + b.plate_query + "|" + b.support_seed.astype(str)
        fm = b.drop_duplicates("family").set_index("family").fold.to_dict()
        rows = []
        for v in ("delta_rmse", "centred_rmse", "centred_pearson", "mrr_tie_centred"):
            fam = S.family_table(b.assign(method=b.method + "|" + b.arm), v, by=("method",))
            for m in ("shrunk_mean", "C0_common", "C1_common", "C0_reselected", "C1_reselected"):
                for arm in ("D16", "REP8"):
                    h = fam[fam.method == f"{m}|{arm}"]
                    rows.append(dict(row="summary", method=m, arm=arm, metric=v, value=float(h[v].mean()), n_families=len(h)))
                c = S.paired(fam.assign(), f"{m}|D16", f"{m}|REP8", value=v, fold_of=fm)
                for r in c.to_dict("records"):
                    rows.append(dict(row="D16_minus_REP8", method=m, metric=v, **{k: r[k] for k in ("diff", "lo", "hi", "lo90", "hi90", "rel_diff", "n_families", "n_families_a_lower",
                                                                                                "fold_diffs", "n_folds_a_lower", "meets_0p5pct", "equivalent_0p5pct")}))
            for arm in ("D16", "REP8"):
                for x, y in (("C1_common", "C0_common"), ("C1_reselected", "C0_reselected"), ("C0_common", "shrunk_mean")):
                    c = S.paired(fam, f"{x}|{arm}", f"{y}|{arm}", value=v, fold_of=fm)
                    for r in c.to_dict("records"):
                        rows.append(dict(row=f"within_{arm}", method=f"{x}-{y}", metric=v, **{k: r[k] for k in ("diff", "lo", "hi", "n_families", "fold_diffs", "n_folds_a_lower")}))
        cost = b.drop_duplicates(["cell", "plate_x", "plate_y", "plate_query", "support_seed", "arm"]).groupby("arm")[["plate_rows", "unique_drugs", "treated_wells",
                                                                                                                       "support_plates", "control_wells_support_plates"]].mean()
        for arm, r in cost.iterrows():
            rows.append(dict(row="cost", arm=arm, **r.to_dict()))
        rows.append(dict(row="cohort", n_families=int(b.family.nunique()), n_cells=int(b.cell.nunique()), design=b.design.iloc[0],
                         same_compound_plate_fraction=float(b.drop_duplicates(["cell", "plate_x", "plate_y", "plate_query"]).same_compound_plate.mean())))
        bt = pd.DataFrame(rows)
        bt["note"] = "retrospective budget simulation on existing X-tag replicate plates; equal plate-row budget is not equal wet-lab cost"
        bt.to_csv(pub_man("support_budget.tsv"), sep="\t", index=False, float_format="%.6g")
    dump(dict(select=a.select, evaluate=a.evaluate), pub_man("branch_report_inputs.json"))
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "evaluate", "report"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--partition", default="")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--constants", default="")
    ap.add_argument("--jconst", default="")
    ap.add_argument("--j3", action="store_true")
    ap.add_argument("--budget", default="", choices=["", "third_plate", "ab_mean"])
    ap.add_argument("--select", nargs="*", default=[])
    ap.add_argument("--evaluate", nargs="*", default=[])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    out = new_run_dir("branch_" + a.stage, a.run_id)
    {"select": select, "evaluate": evaluate, "report": report}[a.stage](a, cfg, out)
