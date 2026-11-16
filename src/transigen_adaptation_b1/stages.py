"""Pre-freeze validation stage, freeze manifest, optional label-permutation control, final test.

  val      : every model and baseline scored on target_val; baseline hyper-parameters chosen; control gate decided
  control  : label-permutation control runs for the targets that pass the gate (plan section 9)
  freeze   : hashes of config, splits, selections and every checkpoint that the final test will read
  test     : opens target test once, writes predictions (server) and per-condition scores (repository)
"""
import argparse, json, os, subprocess
import numpy as np
import pandas as pd
import torch
import yaml

from . import tsg_adapter as T
from .adapt_target import ARMS, TargetSet, adapt_run, run_id
from .baselines import SourceBaselines
from .data import Data, MAN_DIR, FREEZE
from .evaluate import per_condition, summarise, magnitude_ratio, shuffled_retrieval_reference
from .predict import predictions, SEEDS, load_run

CFG = "configs/transigen_adaptation_b1/protocol.yaml"
REP = "reports/transigen_adaptation_b1"


def cells(cfg):
    """(target, split, k, is_main) for every scored configuration."""
    summ = json.load(open(os.path.join(MAN_DIR, "splits_summary.json")))["targets"]
    first = cfg["splits"]["target"]["split_seeds"][0]
    out = []
    for t, info in sorted(summ.items()):
        if info["status"] == "excluded":
            continue
        for s, si in info["splits"].items():
            out.append((t, int(s), si["k_main"], True))
            if int(s) == first:
                out += [(t, int(s), k, False) for k in si["k_curve"]]
    return out


def lrs(sel, t, s):
    return {arm: sel["selected"][f"{t}|{s}|{arm}"]["lr"] for arm in ARMS}     # k-curve reuses the k_main setting of its split


def score_all(cond, truth, P, extra):
    pcs, summ = [], []
    for (model, seed), pred in P.items():
        pc = per_condition(pred, truth)
        pc.insert(0, "drug_group", cond.drug_group.to_numpy())
        pc.insert(0, "pert_id", cond.pert_id.to_numpy())
        for k_, v in reversed(list(dict(extra, model=model, seed=seed).items())):
            pc.insert(0, k_, v)
        pcs.append(pc)
        mg, mglob = magnitude_ratio(pred, truth)
        sm, st5 = shuffled_retrieval_reference(pred, truth)
        summ.append(dict(extra, model=model, seed=seed, **summarise(pc), magnitude_ratio_gene_median=mg, magnitude_ratio_global=mglob,
                         shuffled_mrr=sm, shuffled_top5=st5))
    return pd.concat(pcs), pd.DataFrame(summ)


def val_stage(cfg):
    out_sel = os.path.join(MAN_DIR, "baseline_selection.json")
    assert not os.path.exists(out_sel), "validation stage already recorded"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, S = Data(), SourceBaselines()
    sel = json.load(open(os.path.join(MAN_DIR, "adapt_selection.json")))
    bsel, summs = {}, []
    for t, s, k, main in cells(cfg):
        cond, truth, P, bs, runs, info = predictions(D, S, t, s, k, "val", dev, lrs(sel, t, s), sel["max_updates"])
        bsel[f"{t}|{s}|{k}"] = bs
        _, sm = score_all(cond, truth, P, dict(target=t, split_seed=s, k=k, main=int(main), role="val", **info))
        summs.append(sm)
    val = pd.concat(summs)
    val.to_csv(os.path.join(MAN_DIR, "val_scores.tsv"), sep="\t", index=False)
    # control gate (protocol: label_permutation_control.gate), on k_main rows, seed mean then split mean
    m = val[val.main == 1].groupby(["target", "model", "split_seed"]).delta_rmse.mean().groupby(["target", "model"]).mean().unstack()
    gate = {t: dict(B2=float(r.B2_finetune), B0=float(r.B0), B1=float(r.B1_scratch), b0_mean_offset=float(r.b0_mean_offset),
                    passed=bool(r.B2_finetune < min(r.B0, r.B1_scratch, r.b0_mean_offset))) for t, r in m.iterrows()}
    json.dump(bsel, open(out_sel, "w"), indent=1)
    json.dump(gate, open(os.path.join(MAN_DIR, "control_gate.json"), "w"), indent=1)
    print(m.round(4).to_string())
    print(json.dumps(gate, indent=1))


def control_plan(cfg):
    gate = json.load(open(os.path.join(MAN_DIR, "control_gate.json")))
    sel = json.load(open(os.path.join(MAN_DIR, "adapt_selection.json")))
    s = cfg["splits"]["target"]["split_seeds"][0]
    summ = json.load(open(os.path.join(MAN_DIR, "splits_summary.json")))["targets"]
    plan = []
    for t, g in sorted(gate.items()):
        if not g["passed"]:
            continue
        k = summ[t]["splits"][str(s)]["k_main"]
        lr = sel["selected"][f"{t}|{s}|B2_finetune"]["lr"]
        real = load_run(run_id(t, s, k, "B2_finetune", lr, 42, sel["max_updates"]))
        for p in (1, 2, 3):
            plan.append(dict(target=t, split_seed=s, k=k, lr=lr, seed=42, perm_seed=p, fixed_steps=real["best_step"],
                             run_id=run_id(t, s, k, "B2_finetune", lr, 42, sel["max_updates"], f"_perm{p}")))
    return plan, sel


def control_stage(cfg):
    plan, sel = control_plan(cfg)
    if not plan:
        print("no target passed the gate: no label-permutation control is run")
        return
    dev = torch.device("cuda")
    D = Data()
    for e in plan:
        ts = TargetSet(D, e["target"], e["split_seed"], e["k"], dev)
        groups = D.rows.drug_group.iloc[ts.rows_a].to_numpy()
        ug = sorted(set(groups))
        pg = [ug[i] for i in np.random.RandomState(e["perm_seed"]).permutation(len(ug))]
        donor = dict(zip(ug, pg))                       # each adapt drug group receives another group's fingerprint
        first_row = {g: int(np.where(groups == g)[0][0]) for g in ug}
        fp_perm = [first_row[donor[g]] for g in groups]
        r = adapt_run(D, ts, "B2_finetune", e["lr"], e["seed"], cfg, dev, sel["max_updates"], tag=f"_perm{e['perm_seed']}",
                      fp_perm=fp_perm, fixed_steps=e["fixed_steps"])
        assert r["run_id"] == e["run_id"]
        json.dump(dict(donor_group=donor, moved=int(sum(a != b for a, b in donor.items()))),
                  open(os.path.join(os.path.dirname(r["checkpoint"]), "permutation.json"), "w"), indent=1)


def freeze_stage(cfg):
    assert not os.path.exists(FREEZE), "already frozen"
    sel = json.load(open(os.path.join(MAN_DIR, "adapt_selection.json")))
    src = json.load(open(os.path.join(MAN_DIR, "source_selection.json")))
    files = [CFG] + [os.path.join(MAN_DIR, f) for f in ["build_manifest.json", "splits_summary.json", "source_split.tsv", "source_selection.json",
                                                         "source_baselines.json", "adapt_selection.json", "baseline_selection.json",
                                                         "control_gate.json", "leakage_checks.json"]]
    summ = json.load(open(os.path.join(MAN_DIR, "splits_summary.json")))["targets"]
    files += [os.path.join(MAN_DIR, "splits", f"target_{t}_split{s}.tsv") for t, i in sorted(summ.items()) for s in i["splits"]]
    ck = {e["run_id"]: e["checkpoint_sha256"] for e in src["checkpoints"].values()}
    for t, s, k, main in cells(cfg):
        for arm, lr in lrs(sel, t, s).items():
            for seed in SEEDS:
                r = load_run(run_id(t, s, k, arm, lr, seed, sel["max_updates"]))
                ck[r["run_id"]] = r["checkpoint_sha256"]
    plan, _ = control_plan(cfg)
    for e in plan:
        ck[e["run_id"]] = load_run(e["run_id"])["checkpoint_sha256"]
    fz = dict(frozen=True, protocol_variant=cfg["protocol_variant"], code_commit=os.environ.get("TB_COMMIT", ""),
              max_updates=sel["max_updates"], files={f: T.sha256(f) for f in files}, checkpoints=ck,
              control_runs=[e["run_id"] for e in plan], scored_cells=[dict(target=t, split_seed=s, k=k, main=main) for t, s, k, main in cells(cfg)],
              statement="No target test X1 value has been read by any training, selection or scoring step up to this point.")
    json.dump(fz, open(FREEZE, "w"), indent=1)
    print(f"frozen: {len(files)} files, {len(ck)} checkpoints")


def test_stage(cfg):
    fz = json.load(open(FREEZE))
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    assert git("log", "-1", "--format=%H", "--", FREEZE), "the freeze manifest is not committed"
    assert git("status", "--porcelain", "--", FREEZE) == "", "the freeze manifest differs from the committed one"
    for f, h in fz["files"].items():
        assert T.sha256(f) == h, f"{f} changed after the freeze"
    out = os.path.join(os.environ["TB_RUNS"], "final", "final_v1")
    os.makedirs(out, exist_ok=False)
    os.environ["TB_FINAL_TEST"] = "1"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, S = Data(), SourceBaselines()
    sel = json.load(open(os.path.join(MAN_DIR, "adapt_selection.json")))
    bsel = json.load(open(os.path.join(MAN_DIR, "baseline_selection.json")))
    plan, _ = control_plan(cfg)
    pcs, summs, preds, used, rep = [], [], {}, set(), []
    for t, s, k, main in cells(cfg):
        extra = {f"B2_label_permuted{e['perm_seed']}": e["run_id"] for e in plan if (e["target"], e["split_seed"], e["k"]) == (t, s, k)}
        cond, truth, P, _, runs, info = predictions(D, S, t, s, k, "test", dev, lrs(sel, t, s), sel["max_updates"],
                                                    selection=bsel[f"{t}|{s}|{k}"], extra_runs=extra)
        for r in runs:
            assert fz["checkpoints"][r["run_id"]] == r["checkpoint_sha256"], f"{r['run_id']} is not the frozen checkpoint"
            used.add(r["run_id"])
        pc, sm = score_all(cond, truth, P, dict(target=t, split_seed=s, k=k, main=int(main), role="test", **info))
        pcs.append(pc)
        summs.append(sm)
        f = os.path.join(out, f"pred_{t}_split{s}_k{k}.npz")
        np.savez_compressed(f, truth=truth.astype(np.float32), pert_id=cond.pert_id.to_numpy().astype(str), drug_group=cond.drug_group.to_numpy().astype(str),
                            **{f"{m}__{sd}": p.astype(np.float32) for (m, sd), p in P.items()})
        preds[os.path.basename(f)] = dict(sha256=T.sha256(f), bytes=os.path.getsize(f), target=t, split_seed=s, k=k, n_conditions=len(cond))
        if main:      # replicate agreement of the measured response, where a condition has more than one plate
            rows_e = np.sort(D.target_rows(t, s, "test"))
            meta = D.rows.iloc[rows_e].reset_index(drop=True)
            d = D.target_x1(rows_e, t, s, "final_test") - np.asarray(D.x0[rows_e])
            rr = []
            for _, g in meta.groupby("pert_id"):
                if len(g) >= 2:
                    c = np.corrcoef(d[g.index.to_numpy()])
                    rr.append(c[np.triu_indices(len(g), 1)].mean())
            rep.append(dict(target=t, split_seed=s, conditions=int(meta.pert_id.nunique()), conditions_with_2plus_plates=len(rr),
                            mean_between_plate_delta_pearson=float(np.mean(rr)) if rr else float("nan")))
    missing = (set(fz["checkpoints"]) - used) - {e["run_id"] for e in json.load(open(os.path.join(MAN_DIR, "source_selection.json")))["checkpoints"].values()}
    assert not missing, f"frozen runs that were not scored: {sorted(missing)}"
    os.makedirs(os.path.join(REP, "tables"), exist_ok=True)
    pd.concat(pcs).to_csv(os.path.join(REP, "tables", "test_per_condition.tsv.gz"), sep="\t", index=False, float_format="%.6g")
    pd.concat(summs).to_csv(os.path.join(REP, "tables", "test_scores_by_run.tsv"), sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(rep).to_csv(os.path.join(REP, "tables", "replicate_agreement.tsv"), sep="\t", index=False, float_format="%.6g")
    json.dump(dict(dir=out, files=preds), open(os.path.join(MAN_DIR, "final_predictions.json"), "w"), indent=1)
    print(pd.concat(summs).query("main == 1").groupby(["target", "model"]).delta_rmse.mean().unstack(0).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["val", "control", "freeze", "test"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    dict(val=val_stage, control=control_stage, freeze=freeze_stage, test=test_stage)[a.stage](cfg)
