"""R6-O: observation-budget training matched between N64-only (O0) and N-sampled (O1) training (plan section 3.4).

Arm M of the v1 follow-up only (mean log-normalised RNA of the control set -> MLP -> z, ECFP4 drug,
additive head with the frozen drug shrinkage; configs/r6_followup_v1/model.json, lr 3e-3 as in v1 D3).

    O0  every training set has N = 64 control cells
    O1  N in {16, 32, 64} per update (one N for the whole minibatch), uniform

Both arms use FRESH views and the identical random stream for everything except N:
    * the main generator draws pools, 64 distinct fit_input cells per set, the K drugs and the keep
      rate p in {1, 0.5, 0.25}, and thins all 64 cells, in exactly the same order in both arms;
    * O1 then keeps the first N of those 64 thinned cells (a nested subset of the same barcodes);
    * N comes from a SEPARATE generator (seed + 7_000_000), so drawing N changes no other draw;
    * dropout masks have a shape that does not depend on N (the encoder averages over cells first).
Updates, optimiser, schedule, epochs, sets per update and drugs per set are identical.

Settings: source folds only (fit = TRAIN drugs outside fold k, predict = fold k). No test drug is
predicted here. Predictions are written by obs_eval.py from the checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.depth_robust_r6 import data as D
from src.depth_robust_r6.count_view import set_features, thin_torch
from src.depth_robust_r6.train import Sampler
from src.r6_followup_v1 import common as C
from src.r6_followup_v1.train_fixed import build_model, param_report

N_CHOICES = (16, 32, 64)
N_SEED_OFFSET = 7_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-arm", required=True, choices=["O0", "O1"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stage", default="obs")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--hp", default="configs/r6_followup_v1/model.json")
    a = ap.parse_args()
    assert a.fold in C.FOLDS
    hp = json.loads(Path(a.hp).read_text())
    run = C.assets() / "runs" / a.stage / a.run_id
    run.mkdir(parents=True, exist_ok=False)                           # immutable run ids
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    ngen = torch.Generator(device="cpu"); ngen.manual_seed(a.seed + N_SEED_OFFSET)
    d = C.load()
    fit_drugs, eval_drugs = C.split_spec(d, "source", a.fold)
    assert not (set(fit_drugs) & set(eval_drugs)) and not (set(fit_drugs) & set(C.test_drugs(d)))
    Lf, Df = d.conditions(fit_drugs)
    y_scale = float(np.sqrt((d.Y[Lf, Df].astype(np.float64) ** 2).mean()))
    Y = torch.as_tensor(d.Y, device=dev); fp = torch.as_tensor(d.fp, device=dev)
    cnt = torch.as_tensor(d.cnt, device=dev); crest = torch.as_tensor(d.crest, device=dev)
    smp = Sampler(d, fit_drugs, dev, gen)
    ps = torch.tensor(D.PS, device=dev)
    model = build_model(d, "M", hp).to(dev)
    B, K = hp.get("sets_per_step", 16), hp.get("drugs_per_set", 12)
    shrink = float(hp.get("drug_shrink", 0.0))
    steps_per_epoch = math.ceil(smp.n_cond / (B * K))
    total = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=hp.get("weight_decay", 1e-4))
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / total))))
    print(json.dumps(dict(obs_arm=a.obs_arm, fold=a.fold, seed=a.seed, lr=a.lr, n_fit_cond=smp.n_cond, n_fit_drugs=len(fit_drugs),
                          steps_per_epoch=steps_per_epoch, epochs=a.epochs, y_scale=y_scale)), flush=True)
    t0 = time.time(); curve = []; n_hist = {n: 0 for n in N_CHOICES}
    torch.cuda.reset_peak_memory_stats()
    stream_check = []
    for ep in range(a.epochs):
        model.train()
        agg = {}
        for it in range(steps_per_epoch):
            cells, Lb, Db = smp.draw(B, K, D.EVAL_N)                  # 64 cells, same stream in both arms
            p = ps[torch.randint(len(ps), (B,), device=dev, generator=gen)]
            c, r = thin_torch(cnt[cells].float(), crest[cells].float(), p[:, None].expand(B, cells.shape[1]), gen)
            n_draw = N_CHOICES[int(torch.randint(len(N_CHOICES), (1,), generator=ngen))]   # drawn in BOTH arms (same ngen use)
            N = n_draw if a.obs_arm == "O1" else D.EVAL_N
            n_hist[N] = n_hist.get(N, 0) + 1
            c, r = c[:, :N], r[:, :N]
            if ep == 0 and it < 3:
                stream_check.append(dict(cells_sum=int(cells.sum()), drugs_sum=int(Db.sum()), p_sum=float(p.sum()), n=N))
            z, _ = model.state(set_features(c, r), B)
            g_part, h_part = model.delta_parts(z, fp[Db])
            o = h_part if g_part is None else g_part + h_part
            y = Y[Lb, Db] / y_scale
            parts = dict(supervised=torch.nn.functional.mse_loss(o, y))
            if shrink > 0:
                parts["drug_shrink"] = (h_part ** 2).mean()
            loss = parts["supervised"] + shrink * parts.get("drug_shrink", 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k_, v_ in parts.items():
                agg[k_] = agg.get(k_, 0.0) + float(v_.detach())
        curve.append(dict(epoch=ep + 1, seconds=round(time.time() - t0, 1), **{f"train_{k_}": v_ / steps_per_epoch for k_, v_ in agg.items()}))
        if (ep + 1) % 10 == 0:
            print(json.dumps(curve[-1]), flush=True)
    train_s = time.time() - t0
    torch.save(dict(state_dict=model.state_dict(), arm="M", obs_arm=a.obs_arm, setting="source", fold=a.fold, inner=False, hp=hp, lr=a.lr,
                    seed=a.seed, epochs=a.epochs, y_scale=y_scale, fit_drugs=list(fit_drugs), n_choices=list(N_CHOICES) if a.obs_arm == "O1" else [64]),
               run / "ckpt.pt")
    rec = dict(run_id=a.run_id, stage=a.stage, arm="M", obs_arm=a.obs_arm, setting="source", fold=a.fold, seed=a.seed, lr=a.lr, epochs=a.epochs,
               hp=hp, params=param_report(model, "M"), y_scale=y_scale, n_fit_conditions=smp.n_cond, n_fit_drugs=len(fit_drugs),
               n_eval_drugs=len(eval_drugs), updates=total, sets_per_update=B, drugs_per_set=K, views="fresh (64 cells drawn and thinned; O1 keeps first N)",
               n_update_histogram={str(k): v for k, v in n_hist.items()}, stream_check=stream_check,
               cells_processed=int(sum(k * v for k, v in n_hist.items()) * B),
               train_seconds=round(train_s, 1), peak_vram_MB=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
               gpu=torch.cuda.get_device_name(0), job=os.environ.get("SLURM_JOB_ID"), array=os.environ.get("SLURM_ARRAY_JOB_ID"),
               curve=curve, ckpt_sha256=D.sha256_file(run / "ckpt.pt"), commit=C.git_commit())
    C.dump_json(rec, run / "record.json")
    print(json.dumps({k: v_ for k, v_ in rec.items() if k not in ("curve", "hp")}), flush=True)


if __name__ == "__main__":
    main()
