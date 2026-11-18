"""Existing-benchmark follow-up evaluation, RECORD ONLY, after the follow-up concluded negative.

Scores the frozen arms S, M and F1A (refit on all 209 TRAIN drugs with the Phase C / F1 settings,
seeds 42/43/44, the same view caches) ONCE on the 70 already-seen TEST drugs, next to reference
rows that need no training (cell-line mean, shrunk pool mean with the kappa chosen on source) and
the ORIGINAL R6 stored test predictions of B00 (original SD) and Z0 (original ridge).
Nothing is selected from these numbers; any later change is a new version, never an overwrite.

    --rehearsal   the same code on source fold 2 (TRAIN drugs only; Phase C / F1 fold-2 models and the
                  original R6 dev predictions). Writes to $R6F_ASSETS/scratch only. Used to check the
                  code before the freeze; it opens no TEST response.
Without --rehearsal the script refuses to run unless the freeze manifest exists, the frozen files are
unchanged since the frozen code commit, and no earlier scoring left its lock file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import common as C
from .baselines import MeanBaseline
from .phase_a import SNU423
from .scoring import DEF_TOTAL, DEF_WITHIN, Scorer
from .train_fixed import anchors, build_model, predict

FREEZE = Path("manifests/r6_followup_v1/freeze_manifest.json")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_freeze() -> dict:
    fm = json.loads(FREEZE.read_text())
    head = C.git_commit()
    code = fm["code_commit"]
    assert subprocess.run(["git", "merge-base", "--is-ancestor", code, head]).returncode == 0, "HEAD does not contain the frozen code commit"
    for f, h in fm["frozen_files"].items():
        assert sha(f) == h, f"frozen file changed: {f}"
    assert subprocess.run(["git", "diff", "--quiet", code, "HEAD", "--", *fm["frozen_files"]]).returncode == 0
    return fm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/r6_followup_v1/final_followup_eval.json")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    refs_spec = json.loads(Path("configs/r6_followup_v1/phase_a.json").read_text())
    kappa = json.loads(Path("manifests/r6_followup_v1/kappa_selection.json").read_text())["chosen_kappa"]
    PS = tuple(spec["ps"])
    d = C.load()
    if a.rehearsal:
        k = spec["rehearsal"]["fold"]
        fit, held = C.source_split(d, k)
        arm_spec, ref_tag, role = spec["rehearsal"]["arms"], "dev", f"rehearsal_source_fold{k}"
        out_dir = C.assets() / "scratch" / "final_eval_rehearsal"
        out_dir.mkdir(parents=True, exist_ok=True)
        rep_dir = pred_dir = man_dir = out_dir
        fm = None
    else:
        fm = check_freeze()
        fit, held = C.train_drugs(d), C.test_drugs(d)
        arm_spec, ref_tag, role = spec["arms"], "final", spec["split_role"]
        rep_dir, pred_dir, man_dir = Path("reports/r6_followup_v1"), Path("predictions/r6_followup_v1"), Path("manifests/r6_followup_v1")
        lock = C.assets() / "final_followup_eval.lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)       # score once: a second call fails here
        os.write(fd, json.dumps(dict(commit=C.git_commit(), job=os.environ.get("SLURM_JOB_ID"))).encode()); os.close(fd)
        assert not (rep_dir / f"{spec['out_prefix']}.tsv").exists()
    assert not (set(fit) & set(held))
    L, Dg = d.conditions(held)
    T = d.Y[L, Dg]
    pool = C.pool_of(d, L, Dg)
    snu = np.flatnonzero(d.lines.cell_line_id.to_numpy() == SNU423)
    masks = dict(snu423=(L == snu[0]))
    sc = Scorer(T, L, Dg, pool, None, masks, n_boot=a.n_boot, seed=0)
    inputs = dict(split_role=role, label=spec["label"], commit=C.git_commit(), n_conditions=int(len(L)), n_drugs=int(sc.nd),
                  n_snu423_conditions=int(masks["snu423"].sum()), kappa=kappa, runs={}, references={})

    # reference rows without training: fitted on `fit` drugs only
    Lf, Df = d.conditions(fit)
    mb = MeanBaseline(d.Y, Lf, Df, C.pool_of(d, Lf, Df), d.Y.shape[0], len(d.pools))
    keys = {}
    for name, kap in (("linemean", float("inf")), ("poolmean", float(kappa))):
        sc.add(name, [mb.predict(L, Dg, pool, kap)[0]])
        for p in PS:
            keys[(name, p)] = name
    # original R6 stored predictions (not retrained)
    root = C.orig_assets() / "runs" / ref_tag
    for name in spec["references"][ref_tag]:
        zs = [C.load_pred(root / rel, L, Dg) for rel in refs_spec[ref_tag][name]]
        inputs["references"][name] = [dict(path=rel, sha256=sha(root / rel)) for rel in refs_spec[ref_tag][name]]
        for p in PS:
            sc.add(f"{name}@{p}", [z[q] for z in zs for q in C.PKEYS[p]])
            keys[(name, p)] = f"{name}@{p}"
        print("reference", name, {p: round(sc.summary(keys[(name, p)])["total_rmse"], 6) for p in PS}, flush=True)
    # frozen arms
    dev = "cuda"
    views = [v for v in C.VIEWS if v[0] in PS]
    percond = {}
    for name, item in arm_spec.items():
        preds = {p: [] for p in PS}
        for s in spec["seeds"]:
            run = C.assets() / "runs" / item["stage"] / item["pattern"].format(seed=s)
            ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
            assert sorted(ck["fit_drugs"]) == sorted(fit) and ck["seed"] == s, run
            if not a.rehearsal:
                fz = spec["arms"][name]
                assert ck["setting"] == "final" and ck["arm"] == fz["arm"] and abs(ck["lr"] - fz["lr"]) < 1e-12, run
                assert ck["anchor_kappa"] == fz["anchor_kappa"] and float(ck["pair_lambda"]) == fz["pair_lambda"], run
            m = build_model(d, ck["arm"], ck["hp"]).to(dev)
            m.load_state_dict(ck["state_dict"])
            anc = anchors(d, ck["fit_drugs"], ck["anchor_kappa"]) if ck["anchor_kappa"] is not None else None
            L_, D_, pe = predict(m, d, held, dev, ck["y_scale"], views=views, anchor=anc)
            assert (L_ == L).all() and (D_ == Dg).all()
            if not a.rehearsal:
                rd = C.assets() / "final_followup_eval" / run.name
                rd.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(rd / "pred_test.npz", line=L, drug=Dg, **{f"p{p}_s{t}": v.astype(np.float16) for (p, t), v in pe.items()})
                inputs["runs"][run.name] = dict(ckpt_sha256=sha(run / "ckpt.pt"), pred_sha256=sha(rd / "pred_test.npz"))
            else:
                inputs["runs"][run.name] = dict(ckpt_sha256=sha(run / "ckpt.pt"))
            for p in PS:
                preds[p] += [pe[(p, t)] for (pp, t) in views if pp == p]
            del m
            torch.cuda.empty_cache()
        for p in PS:
            sc.add(f"{name}@{p}", preds[p])
            keys[(name, p)] = f"{name}@{p}"
        print("arm", name, {p: round(sc.summary(keys[(name, p)])["total_rmse"], 6) for p in PS}, flush=True)

    std = C.std_fields(split_role=role, label=spec["label"])
    rows, ret = [], []
    order = ["linemean", "poolmean", *spec["references"][ref_tag], *arm_spec]
    for name in order:
        for p in PS:
            for mk in ("all", "snu423"):
                rows.append(dict(std, row_type="arm", arm=name, comparator="", p=p, model_seed=",".join(map(str, spec["seeds"])), **sc.summary(keys[(name, p)], mk)))
            if keys[(name, p)] != name or p == PS[0]:
                for r in sc.retrieval_rows(keys[(name, p)]):
                    ret.append(dict(std, row_type="arm", arm=name, p=p, **r))
    for c, b in spec["pairs"]:
        for p in PS:
            for mk in ("all", "snu423"):
                pr = sc.pair(keys[(c, p)], keys[(b, p)], mk)
                pr.update(candidate=c, comparator=b)
                rows.append(dict(std, row_type="pair", arm=c, p=p, **pr))
            for r in sc.retrieval_pair(keys[(c, p)], keys[(b, p)]):
                r.update(candidate=c, comparator=b)
                ret.append(dict(std, row_type="pair", p=p, **r))
    out = pd.DataFrame(rows)
    out["metric_definition"] = f"total: {DEF_TOTAL} | within: {DEF_WITHIN}"
    out["bootstrap"] = f"paired, drug structure groups, {a.n_boot} replicates, pools re-centred per replicate"
    out["note"] = ("B00 = original R6 SD trained with fresh views (the follow-up arms used one fixed view cache), so S - B00 mixes "
                   "drug input with the view regime; Z0 = original ridge; linemean/poolmean fitted on the same drugs as the arms")
    rr = pd.DataFrame(ret)
    rr["metric_definition"] = "tie-corrected top1/top5/MRR by Pearson correlation of centred prediction with centred measured candidates; drug-macro; legacy_* = original R6 definitions"
    pfx = spec["out_prefix"] if not a.rehearsal else "rehearsal"
    out.to_csv(rep_dir / f"{pfx}.tsv", sep="\t", index=False, float_format="%.6f")
    out[out.subset == "snu423"].to_csv(rep_dir / f"{pfx}_snu423.tsv", sep="\t", index=False, float_format="%.6f")
    rr.to_csv(rep_dir / f"{pfx}_retrieval.tsv", sep="\t", index=False, float_format="%.6f")
    cid = d.lines.cell_line_id.to_numpy()
    cols = dict(split_role=role, cell_line_id=cid[L], drug_group=Dg, pool_id=pool, truth_rms=sc.truth_rms, truth_centred_rms=sc.truth_c_rms)
    for (name, p), key in keys.items():
        if key == name and p != PS[0]:
            continue
        sfx = name if key == name else f"{name}_p{p}"
        cols[f"{sfx}_total"] = sc.arms[key]["v"]; cols[f"{sfx}_within"] = sc.arms[key]["lw"]
    pd.DataFrame(cols).to_csv(pred_dir / f"{pfx}_per_condition.tsv.gz", sep="\t", index=False, float_format="%.5f", compression="gzip")
    if fm is not None:
        inputs["freeze_commit_manifest"] = dict(code_commit=fm["code_commit"], manifest_sha256=sha(FREEZE))
    C.dump_json(inputs, man_dir / f"{pfx}_inputs.json")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    for mk in ("all", "snu423"):
        o = out[out.subset == mk]
        print(o[o.row_type == "arm"][["arm", "p", "n_drugs", "n_units", "total_rmse", "total_lo", "total_hi", "within_rmse", "within_lo", "within_hi", "centred_magnitude_ratio"]].to_string())
        print(o[o.row_type == "pair"][["candidate", "comparator", "p", "total_diff", "total_lo", "total_hi", "total_rel", "total_category",
                                       "within_diff", "within_lo", "within_hi", "within_rel", "within_category"]].to_string())
    print(rr[rr.row_type == "arm"][["arm", "p", "candidate_pool", "top1", "top1_lo", "top1_hi", "chance_top1", "mrr", "legacy_rank_score", "legacy_unique_top1"]].to_string())


if __name__ == "__main__":
    main()
