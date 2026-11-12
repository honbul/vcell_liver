"""Train one (model, lr, weight decay, seed) run. Validation only -- this script never reads test.

The whole condition table fits in GPU memory, so there is no dataloader: each epoch shuffles the
valid training conditions and walks them in batches of 256. Selection is on overall validation
macro-ΔRMSE; SNU-423 is computed and stored but is never allowed to pick anything.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import evaluate as ev
from .data import Dataset
from .models import build, solve_dense_hidden


def lr_lambda(epoch: int, warmup: int, total: int, final_frac: float):
    if epoch < warmup:
        return (epoch + 1) / warmup
    t = (epoch - warmup) / max(total - warmup, 1)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def predict(model, ds, cond, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(cond), batch):
        b = slice(i, i + batch)
        p, _ = model(ds.basal[cond.line[b]], ds.fp[cond.drug[b]])
        out.append(p.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, ds.n_genes), np.float32)


@torch.no_grad()
def router_usage(model, ds, cond, batch: int = 2048):
    if model.kind == "A":
        return None
    model.eval()
    acc = []
    for i in range(0, len(cond), batch):
        b = slice(i, i + batch)
        _, pi = model(ds.basal[cond.line[b]], ds.fp[cond.drug[b]])
        acc.append(pi.float().cpu().numpy())
    pi = np.concatenate(acc)
    ent = -(pi * np.log(np.clip(pi, 1e-12, None))).sum(1)
    return dict(mean_weight=pi.mean(0).tolist(), max_weight_share=float(pi.max(1).mean()),
                entropy_mean=float(ent.mean()), entropy_max=float(np.log(pi.shape[1])),
                argmax_fraction=np.bincount(pi.argmax(1), minlength=pi.shape[1]).astype(float).tolist(),
                # how much the mixture actually moves with the input: a router whose weights barely
                # vary is behaving like B's single global vector, whatever its entropy says
                weight_sd_across_inputs=pi.std(0).tolist(),
                weight_sd_mean=float(pi.std(0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kind", required=True, choices=["A", "B", "C"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--final-lr-frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds = Dataset(args.data_dir, args.manifest_dir, device=dev)
    tr, va = ds.conditions("train"), ds.conditions("val")

    # Counting parameters must not change what anything is initialised to; `solve_dense_hidden`
    # runs in a forked RNG, and the shared modules are copied in from the one state for this seed
    # rather than drawn in place. `shared_fingerprint` is recorded so that two runs of different
    # kinds at the same seed can be checked for having started from identical shared weights.
    hidden, parity = (None, None)
    if args.kind == "A":
        hidden, parity = solve_dense_hidden(ds.n_genes, ds.n_bits)
    model = build(args.kind, ds.n_genes, ds.n_bits, dense_hidden=hidden, seed=args.seed).to(dev)
    shared_fp = model.shared_fingerprint()
    counts = model.param_counts()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: lr_lambda(e, args.warmup, args.max_epochs, args.final_lr_frac))

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run = Path(args.out_dir)
    run.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    n = len(tr)
    best = dict(val=float("inf"), epoch=-1)
    best_state = None
    # Score the untrained network once, as a diagnostic only. The first build reported the model as
    # peaking "at epoch 0" and read that as the network having barely moved from initialisation,
    # but epoch 0 in this loop is *after* a full pass over the training conditions. Without this
    # number there was nothing in the record to check that reading against. It is deliberately not
    # eligible for selection: early stopping still chooses among trained epochs only.
    val_at_init = ev.macro_delta_rmse(predict(model, ds, va), va.y.cpu().numpy(),
                                      va.line.cpu().numpy())
    history = [dict(epoch=-1, train_loss=float("nan"), val_macro_delta_rmse=val_at_init,
                    lr=opt.param_groups[0]["lr"])]
    t0 = time.time()

    for epoch in range(args.max_epochs):
        model.train()
        order = torch.randperm(n, generator=gen).to(dev)
        tot, wsum = 0.0, 0.0
        for i in range(0, n, args.batch):
            idx = order[i: i + args.batch]
            pred, _ = model(ds.basal[tr.line[idx]], ds.fp[tr.drug[idx]])
            se = ((pred - tr.y[idx]) ** 2).mean(1)
            w = tr.weight[idx]
            loss = (w * se).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += float(loss) * float(w.sum())
            wsum += float(w.sum())
        sched.step()

        vp = predict(model, ds, va)
        vmacro = ev.macro_delta_rmse(vp, va.y.cpu().numpy(), va.line.cpu().numpy())
        history.append(dict(epoch=epoch, train_loss=tot / wsum, val_macro_delta_rmse=vmacro,
                            lr=opt.param_groups[0]["lr"]))
        if vmacro < best["val"] - 1e-9:
            best = dict(val=vmacro, epoch=epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best["epoch"] >= args.patience:
            break

    model.load_state_dict(best_state)
    train_s = time.time() - t0
    vp = predict(model, ds, va)
    vy, vl = va.y.cpu().numpy(), va.line.cpu().numpy()
    metrics, _ = ev.all_metrics(vp, vy, vl, ds.line_names, ds.snu423)

    peak = {}
    if dev == "cuda":
        peak = dict(peak_alloc_MB=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_MB=torch.cuda.max_memory_reserved() / 2**20,
                    gpu=torch.cuda.get_device_name(0))
    # inference throughput on a fixed batch, same device, after training
    model.eval()
    with torch.no_grad():
        bb, ff = ds.basal[va.line[:256]], ds.fp[va.drug[:256]]
        for _ in range(5):
            model(bb, ff)
        if dev == "cuda":
            torch.cuda.synchronize()
        ti = time.time()
        for _ in range(50):
            model(bb, ff)
        if dev == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.time() - ti) / 50 * 1000

    rec = dict(kind=args.kind, lr=args.lr, wd=args.wd, seed=args.seed, tag=args.tag,
               params=counts, dense_hidden=hidden, parity=parity,
               shared_init_sha256=shared_fp,
               epochs_run=len(history) - 1, best_epoch=best["epoch"], val_macro_delta_rmse=best["val"],
               val_at_init=val_at_init,
               val_metrics=metrics, train_seconds=round(train_s, 1),
               infer_ms_per_256=round(infer_ms, 3), device=dev, **peak,
               n_train_conditions=n, n_val_conditions=len(va),
               n_lines=ds.n_lines, n_genes=ds.n_genes,
               router_usage_val=router_usage(model, ds, va),
               budget_limited=bool(len(history) - 1 >= args.max_epochs))
    torch.save(dict(state_dict=best_state, record=rec), run / "best.pt")
    (run / "record.json").write_text(json.dumps(rec, indent=1))
    np.save(run / "val_pred.npy", vp.astype(np.float32))
    with open(run / "history.tsv", "w") as f:
        f.write("epoch\ttrain_loss\tval_macro_delta_rmse\tlr\n")
        for h in history:
            f.write(f"{h['epoch']}\t{h['train_loss']:.6f}\t{h['val_macro_delta_rmse']:.6f}\t{h['lr']:.3e}\n")
    print(json.dumps({k: rec[k] for k in ["kind", "lr", "wd", "seed", "epochs_run", "best_epoch",
                                          "val_macro_delta_rmse", "train_seconds", "budget_limited"]}), flush=True)


if __name__ == "__main__":
    main()
