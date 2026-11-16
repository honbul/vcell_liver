"""Source pretraining of the public TranSiGen architecture on non-HCC cell lines (clean B0 source)."""
import argparse, json, os, time
import numpy as np
import torch
import yaml

from . import tsg_adapter as T
from .data import Data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default="configs/transigen_adaptation_b1/protocol.yaml")
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke runs only; study runs use the protocol value")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sp = cfg["source_pretraining"]
    assert a.lr in sp["lr_grid"] and a.seed in sp["seeds"]
    max_epochs = a.max_epochs or sp["max_epochs"]
    out = os.path.join(os.environ["TB_RUNS"], "source", a.run_id)
    os.makedirs(out, exist_ok=False)          # immutable run id: an existing directory is an error
    dev = torch.device("cuda")
    D = Data()
    tr, va = D.source_split()
    x1s = D.source_x1()
    didx = D.rows.drug_idx.to_numpy()
    X0 = torch.tensor(np.asarray(D.x0[: D.n_src]), device=dev)
    X1 = torch.tensor(np.asarray(x1s), device=dev)
    FP = torch.tensor(D.fp, dtype=torch.float32, device=dev)
    DI = torch.tensor(didx[: D.n_src], device=dev)
    tr_t, va_t = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
    va_delta = X1[va_t] - X0[va_t]

    model = T.build_model(a.seed, dev)
    init_hash = T.state_hash(model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=cfg["model"]["weight_decay"])
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    bs = sp["batch_size"]

    def val_rmse():
        p = T.predict_x1_mean_path(model, X0[va_t], FP[DI[va_t]])
        return torch.sqrt(((p - X0[va_t] - va_delta) ** 2).mean(1)).mean().item()

    log = open(os.path.join(out, "epochs.tsv"), "w")
    log.write("epoch\ttrain_loss_per_sample\tval_delta_rmse\tval_upstream_loss\tval_mse_pert\tseconds\n")
    v0 = val_rmse()
    log.write(f"0\tnan\t{v0:.6f}\tnan\tnan\t0\n")
    best, best_epoch, best_state = v0, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    stopped = "max_epochs"
    for ep in range(1, max_epochs + 1):
        perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=dev)]
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:                     # upstream skips single-sample batches (BatchNorm)
                continue
            tot += T.upstream_train_step(model, opt, X0[b], X1[b], FP[DI[b]])
            cnt += len(b)
        v = val_rmse()
        ul = T.upstream_eval_loss(model, X0[va_t], X1[va_t], FP[DI[va_t]], seed=a.seed)
        log.write(f"{ep}\t{tot / cnt:.4f}\t{v:.6f}\t{ul['loss']:.4f}\t{ul['mse_pert']:.4f}\t{time.time() - t0:.0f}\n")
        log.flush()
        if v < best:
            best, best_epoch = v, ep
            best_state = {k: t_.detach().clone() for k, t_ in model.state_dict().items()}
        if ep - best_epoch >= sp["patience_epochs"]:
            stopped = "patience"
            break
    log.close()
    model.load_state_dict(best_state)
    ck = os.path.join(out, "best.pt")
    T.save_checkpoint(model, ck, dict(run_id=a.run_id, lr=a.lr, seed=a.seed, best_epoch=best_epoch, kind="source"))
    res = dict(run_id=a.run_id, kind="source", lr=a.lr, seed=a.seed, epochs_run=ep, best_epoch=best_epoch, stopped=stopped,
               val_delta_rmse_best=best, val_delta_rmse_epoch0=v0,
               val_no_change_delta_rmse=torch.sqrt((va_delta ** 2).mean(1)).mean().item(),
               n_train_rows=int(len(tr)), n_val_rows=int(len(va)), train_cells=int(D.rows.cell.iloc[tr].nunique()),
               init_state_hash=init_hash, state_hash=T.state_hash(model), checkpoint=ck, checkpoint_sha256=T.sha256(ck),
               seconds=time.time() - t0, gpu=torch.cuda.get_device_name(0),
               gpu_peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
               commit=os.environ.get("TB_COMMIT", ""), slurm_job=os.environ.get("SLURM_JOB_ID", ""))
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
