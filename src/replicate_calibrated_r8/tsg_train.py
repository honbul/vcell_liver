"""Baseline: public TranSiGen architecture retrained on the R8 source composition (XC mixed pools and degenerate rows removed).

Same training loop and settings as study B's source pretraining (lr 3e-4 selected there; the other grid values diverged). Train rows:
source-train cells; early stopping on the rows of the confirmation cells (as in study B / R2). No HCC row is read.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import proc, dump, new_run_dir, run_record
from src.transigen_adaptation_b1 import tsg_adapter as T
from src.transigen_adaptation_b1.data import Data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/transigen_adaptation_b1/protocol.yaml"))
    sp = cfg["source_pretraining"]
    t0 = time.time()
    out = new_run_dir("tsg_source", a.run_id)
    dev = torch.device("cuda")
    torch.set_num_threads(1)
    D = Data(os.environ["TB_PROC"])
    rows = pd.read_csv(proc("rows.tsv"), sep="\t", usecols=["row_id", "cell"])
    cells = pd.read_csv(proc("cells.tsv"), sep="\t").set_index("cell").meta_role
    role = rows.cell.map(cells).to_numpy()
    tr, va = rows.row_id.to_numpy()[role == "train"], rows.row_id.to_numpy()[role == "val"]
    assert (tr < D.n_src).all() and (va < D.n_src).all() and not D.rows.cell.iloc[tr].str.startswith("XC.").any()
    X0 = torch.tensor(np.asarray(D.x0[: D.n_src]), device=dev)
    X1 = torch.tensor(np.asarray(D.source_x1()), device=dev)
    FP = torch.tensor(D.fp, dtype=torch.float32, device=dev)
    DI = torch.tensor(D.rows.drug_idx.to_numpy()[: D.n_src], device=dev)
    tr_t, va_t = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
    va_delta = X1[va_t] - X0[va_t]
    model = T.build_model(a.seed, dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=cfg["model"]["weight_decay"], fused=True)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    bs = sp["batch_size"]

    def val_rmse():
        p = T.predict_x1_mean_path(model, X0[va_t], FP[DI[va_t]])
        return torch.sqrt(((p - X0[va_t] - va_delta) ** 2).mean(1)).mean().item()

    log = open(os.path.join(out, "epochs.tsv"), "w")
    log.write("epoch\ttrain_loss_per_sample\tval_delta_rmse\tseconds\n")
    v0 = val_rmse()
    best, best_epoch, best_state = v0, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
    stopped = "max_epochs"
    for ep in range(1, sp["max_epochs"] + 1):
        perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=dev)]
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            l = T.upstream_train_step(model, opt, X0[b], X1[b], FP[DI[b]])
            tot += l
            cnt += len(b)
            if not np.isfinite(l):
                break
        if not np.isfinite(tot):
            stopped = "non_finite_loss"
            break
        v = val_rmse()
        v = v if np.isfinite(v) else float("inf")
        log.write(f"{ep}\t{tot / cnt:.4f}\t{v:.6f}\t{time.time() - t0:.0f}\n")
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
    T.save_checkpoint(model, ck, dict(run_id=a.run_id, lr=a.lr, seed=a.seed, best_epoch=best_epoch, kind="r8_source"))
    res = run_record(a.run_id, "tsg_source", t0, lr=a.lr, seed=a.seed, epochs_run=ep, best_epoch=best_epoch, stopped=stopped, val_delta_rmse_best=best,
                     val_delta_rmse_epoch0=v0, val_no_change_delta_rmse=torch.sqrt((va_delta ** 2).mean(1)).mean().item(), n_train_rows=int(len(tr)),
                     n_val_rows=int(len(va)), train_cells=int(D.rows.cell.iloc[tr].nunique()), checkpoint=ck, checkpoint_sha256=T.sha256(ck),
                     gpu=torch.cuda.get_device_name(0))
    dump(res, os.path.join(out, "result.json"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
