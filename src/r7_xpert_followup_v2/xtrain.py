"""XPert training trajectory with per-epoch source selectors (XPert env, one GPU). Private plan 2026-11-19 sections 4.1-4.4.

What is kept from the upstream run of the original R7 (GSanShui/XPert d53ff49, train_xpert.py, scaler path, include_cell_idx):
  model class and init (XPertNet + init_weights, upstream modules imported from the read-only working copy), config of
  the original R7 run (config_r7.yaml of xpr_/xpe_ runs; only file paths made absolute), Adam(lr 0.004, wd 1e-5),
  LambdaLR (1.0 < epoch 40, then 0.5), batch 128, AMP autocast + GradScaler, the L0 loss exactly as upstream train()
  (weights [0.2, 0.003, 0.2, 1]; epoch < 70: sqrt-of-batch-mean MSE terms + batch-mean CE/PCC; later: weighted sums),
  upstream validate() selection value loss4 = sum over rows of pcc_loss_sum per 128-row batch / n_rows (autocast).
What differs (recorded as deviations): data are GPU-resident tensors instead of the upstream DataLoader (same values:
  treated = X0 + delta, upstream _digitize(side='one') bins, the group's UniMol/HG or Morgan features, cell index,
  dose/time index of 10 uM / 24 h); minibatch order is a seeded torch.randperm per epoch; training never stops early.
  Instead every selector is evaluated each epoch on its own selection rows and its best checkpoint saved (strict
  improvement, first epoch on ties). Selector U ('upstream') emulates EarlyStopping(patience=50) exactly: after its
  counter reaches 50 its choice is frozen; because the trajectory does not depend on any selector this equals stopping.

Selectors (all on source-only rows unless named *_all_replay):
  U       : upstream loss4 on R4 validation units with is_hcc == 0 (corrected original rule), patience 50
  U_all   : upstream loss4 on every R4 validation unit (HCC included) = replay of the original rule (full scope only;
            impact measurement of the correction, never used for any new choice), patience 50
  R       : min  mean_family mean_units RMSE_978(delta_hat, delta)     (source validation, family-equal)
  P       : max  mean_family mean_units Pearson_gene(delta_hat, delta) (raw per-unit gene Pearson, NOT centred)
Recipes: L0 = upstream composite loss; L1 = mean_i mean_g (delta_hat - delta)^2 only, constant over epochs (auxiliary
heads kept in the graph with weight 0).
After training each selected checkpoint predicts (fp32, eval mode, treated slot zero-filled, as the original
xpert_predict) the trajectory's prediction sets."""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

from .common import runs, orig_xpert, xp_work, pman, dump, sha256, sha_array, stamp, CFG
from .folds import load_folds, unit_sets
from src.evidence_gated_r7.data import Store
from src.evidence_gated_r7.evaluate import family_mean

DOSE_IDX, TIME_IDX = 8, 2           # MyDataset: pert_dose 10.0 -> 8, pert_time 24 -> 2 (unused by the sdst model)


def upstream():
    sys.path.insert(0, os.environ["XP_UPSTREAM"])
    import yaml
    from models.model_XPert import XPertNet
    from utils import mse_loss_ls_sum, pcc_loss_sum, set_random_seed
    from datasets.MyDataset import _digitize
    return yaml, XPertNet, mse_loss_ls_sum, pcc_loss_sum, set_random_seed, _digitize


def build_config(yaml, variant):
    ref = "xpr_e100_s42" if variant == "official" else "xpe_e100_s42"
    cfg = yaml.safe_load(open(orig_xpert(ref, "config_r7.yaml")))
    cfg["model"]["ATTN"]["ppi_gene_vector_path"] = xp_work("processed_data", "r7_PPI_gene_vector_128d.npy")
    cfg["model"]["HG"]["drug_hg_pretrained_embed_path"] = xp_work("r7_drug_hg.npy" if variant == "official" else "r7_drug_hg_zero.npy")
    return cfg, orig_xpert(ref, "config_r7.yaml")


def model_args(variant, mode="train"):
    return argparse.Namespace(mode=mode, drug_feat="unimol" if variant == "official" else "morgan", dataset="l1000_r7sdst",
                              pretrained_mode="global", include_cell_idx=True, wo_HG=False, wo_atom=False, wo_atom_HG=False,
                              wo_unimol=False, wo_ppi=False, use_gene_pos_emed=False, output_attention=False, output_cls_embed=False)


def drug_table(variant):
    if variant == "official":
        return np.load(xp_work("r7_drug_unimol.npy")), xp_work("r7_drug_unimol.npy")
    d = np.load(xp_work("r7_drug_morgan1024.npy"), allow_pickle=True).item()
    return np.stack([d[g] for g in range(len(d))]).astype(np.float32), xp_work("r7_drug_morgan1024.npy")


class Units:
    """GPU-resident per-unit tensors (labels only for non-test units)."""

    def __init__(self, torch, S, idx, bins, digitize, dev, labels=True):
        self.idx = np.asarray(idx)
        x0 = S.x0[self.idx].astype(np.float32)
        self.ctl = torch.as_tensor(x0, device=dev)
        self.ctlb = torch.as_tensor(digitize(x0, bins, side="one"), device=dev)
        if labels:
            trt = x0 + S.delta(self.idx).astype(np.float32)
            self.trt = torch.as_tensor(trt, device=dev)
            self.trtb = torch.as_tensor(digitize(trt, bins, side="one"), device=dev)
        else:
            self.trt = torch.zeros_like(self.ctl); self.trtb = torch.zeros_like(self.ctlb)
        self.g = torch.as_tensor(S.group_idx[self.idx], dtype=torch.long, device=dev)
        self.c = torch.as_tensor(S.cell_idx[self.idx], dtype=torch.long, device=dev)
        self.n = len(self.idx)

    def batch(self, torch, F, j, zero_trt=False):
        n = len(j)
        trt = torch.zeros_like(self.ctl[j]) if zero_trt else self.trt[j]
        trtb = torch.zeros_like(self.ctlb[j]) if zero_trt else self.trtb[j]
        return (trt, self.ctl[j], trtb, self.ctlb[j], F[self.g[j]],
                torch.full((n,), DOSE_IDX, dtype=torch.long, device=self.ctl.device), torch.full((n,), TIME_IDX, dtype=torch.long, device=self.ctl.device),
                self.g[j], self.c[j], torch.zeros(n, dtype=torch.long, device=self.ctl.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="trajectory id in configs/r7_xpert_followup_v2/<manifest>")
    ap.add_argument("--manifest", default="trajectories.tsv")
    ap.add_argument("--epochs", type=int, default=None, help="override (pilot only)")
    ap.add_argument("--run-suffix", default="", help="pilot runs get their own run id")
    ap.add_argument("--check-v1", action="store_true", help="pilot: reproduce the original v1 checkpoint's prediction and logged loss4")
    a = ap.parse_args()
    import torch
    from torch.cuda.amp import GradScaler, autocast
    yaml, XPertNet, mse_sum, pcc_sum, set_random_seed, digitize = upstream()

    T = pd.read_csv(os.path.join(CFG, a.manifest), sep="\t", dtype={"outer": int, "inner": int, "seed": int, "epochs": int})
    t = T[T.traj_id == a.traj].iloc[0].to_dict()
    run_id = a.traj + a.run_suffix
    out = runs("traj", run_id)
    os.makedirs(out, exist_ok=False)                  # immutable run id
    epochs = a.epochs or int(t["epochs"])
    selectors = t["selectors"].split(",")
    t0 = time.time()

    S = Store()
    folds = load_folds()
    sets = unit_sets(S, folds, t["scope"], int(t["outer"]), int(t["inner"]))
    cfg, cfg_src = build_config(yaml, t["variant"])
    bins = np.quantile(np.array([cfg["dataset"]["min_value"], cfg["dataset"]["max_value"]]), np.linspace(0, 1, cfg["dataset"]["n_bins"] - 1))
    dev = torch.device("cuda")
    seed = int(t["seed"])
    set_random_seed(seed)
    torch.backends.cudnn.benchmark = True              # as upstream main() after set_random_seed
    Ftab, fpath = drug_table(t["variant"])
    F = torch.as_tensor(Ftab, device=dev); del Ftab
    args = model_args(t["variant"])
    import logging
    model = XPertNet(args, cfg, dev, logging.getLogger("xpert"))
    model.init_weights(); model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["train_lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0 if e < 40 else 0.5)
    scaler = GradScaler()
    CE = torch.nn.CrossEntropyLoss(reduction="sum")
    wa, wb, wc, wd = cfg["train"]["loss_weight"]
    init_epoch = cfg["train"]["init_epoch"]
    patience = cfg["train"]["patience"]
    bs = cfg["train"]["batch_size"]

    tr = Units(torch, S, sets["train"], bins, digitize, dev)
    dsrc = Units(torch, S, sets["select_src"], bins, digitize, dev)
    dall = Units(torch, S, sets["select_all"], bins, digitize, dev) if "U_all" in selectors else None
    fam_src = S.units.family.to_numpy()[sets["select_src"]]
    true_src = S.delta(sets["select_src"]).astype(np.float64)
    gen = torch.Generator().manual_seed(seed)

    def dev_pass(U):
        """upstream validate(): eval mode, autocast, 128-row batches in the given order; loss4 = sum pcc_loss_sum / n."""
        model.eval()
        tot4, n, preds = 0.0, 0, []
        with torch.no_grad():
            for i in range(0, U.n, bs):
                j = torch.arange(i, min(i + bs, U.n), device=dev)
                data = U.batch(torch, F, j)
                with autocast():
                    o = model(data)
                    l4 = pcc_sum(o[2], o[3] - o[4])
                tot4 += l4.item(); n += len(j)
                preds.append(o[2].float().cpu().numpy())
        return tot4 / n, np.concatenate(preds).astype(np.float64)

    best = {s: dict(score=None, epoch=None, counter=0, frozen=False) for s in selectors}

    def consider(sel, score, epoch, lower=True, emulate_patience=False, require_finite=False):
        b = best[sel]
        if b["frozen"]:
            return
        better = b["score"] is None or (score < b["score"] if lower else score > b["score"])
        if require_finite and not np.isfinite(score):     # v2 fix (2026-11-20): a diverged epoch is never a new best for R/P
            better = False
        if better:
            b.update(score=float(score), epoch=epoch, counter=0)
            torch.save({"epoch": epoch, "selector": sel, "score": float(score), "model_state_dict": model.state_dict()}, os.path.join(out, f"ckpt_{sel}.pt"))
        else:
            b["counter"] += 1
            if emulate_patience and b["counter"] >= patience:
                b["frozen"] = True; b["stopped_at_epoch"] = epoch

    curve = []
    for epoch in range(epochs):
        te = time.time()
        model.train()
        perm = torch.randperm(tr.n, generator=gen).to(dev)
        tl = np.zeros(5); ns = 0
        for i in range(0, tr.n, bs):
            j = perm[i:i + bs]
            data = tr.batch(torch, F, j)
            model.zero_grad()
            with autocast():
                trt_o, ctl_o, deg_o, trt_r, ctl_r, _, c_true, c_pred = model(data)
                deg_true = trt_r - ctl_r
                n = trt_r.shape[0]; ns += n
                l1 = mse_sum(trt_o, trt_r)
                l2 = CE(c_pred[0], c_true) + CE(c_pred[1], c_true)
                l3 = mse_sum(deg_o, deg_true)
                l4 = pcc_sum(deg_o, deg_true)
                weighted = l1 * wa + l2 * wb + l3 * wc + l4 * wd
                batch_weighted = torch.sqrt(l1 / n) * wa + (l2 / n) * wb + torch.sqrt(l3 / n) * wc + (l4 / n) * wd
                l1only = l3 / n
            if t["recipe"] == "L0":
                loss = batch_weighted if epoch < init_epoch else weighted
            else:
                loss = l1only
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += [weighted.item(), l1.item(), l2.item(), l3.item(), l4.item()]
        torch.cuda.synchronize(); ttrain = time.time() - te
        tv = time.time()
        u_src, p_src = dev_pass(dsrc)
        rmse = np.sqrt(((p_src - true_src) ** 2).mean(1))
        pc = p_src - p_src.mean(1, keepdims=True); tc = true_src - true_src.mean(1, keepdims=True)
        pear = (pc * tc).sum(1) / np.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1) + 1e-12)
        R = family_mean(rmse, fam_src); P = family_mean(pear, fam_src)
        if not np.isfinite(p_src).all():                     # v2 fix: NaN-skipping group sums turned a diverged epoch into R = 0.0
            R = P = float("nan")
        row = dict(epoch=epoch, lr=opt.param_groups[-1]["lr"], train_total=tl[0] / ns, train_loss1_rmse=np.sqrt(tl[1] / ns), train_loss2=tl[2] / ns,
                   train_loss3_rmse=np.sqrt(tl[3] / ns), train_loss4=tl[4] / ns, sel_U_src_loss4=u_src, sel_R_src_famrmse=R, sel_P_src_fampearson=P,
                   sel_U_src_unit_rmse=float(rmse.mean()))
        if "U" in selectors:
            consider("U", u_src, epoch, lower=True, emulate_patience=True)
        if "R" in selectors:
            consider("R", R, epoch, lower=True, require_finite=True)
        if "P" in selectors:
            consider("P", P, epoch, lower=False, require_finite=True)
        if dall is not None:
            u_all, _ = dev_pass(dall)
            row["sel_U_all_replay_loss4"] = u_all
            consider("U_all", u_all, epoch, lower=True, emulate_patience=True)
        sched.step()
        row.update(sec_train=round(ttrain, 1), sec_select=round(time.time() - tv, 1), max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
        curve.append(row)
        pd.DataFrame(curve).to_csv(os.path.join(out, "curve.tsv"), sep="\t", index=False)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    t_train = time.time() - t0

    # predictions of every selected checkpoint (fp32, eval mode, treated slot zero)
    pargs = model_args(t["variant"], mode="test")
    pm = XPertNet(pargs, cfg, dev, logging.getLogger("xpert")).to(dev)
    preds = {}

    @torch.no_grad()
    def predict(U, zero=True):
        pm.eval(); o = []
        for i in range(0, U.n, bs):
            j = torch.arange(i, min(i + bs, U.n), device=dev)
            o.append(pm(U.batch(torch, F, j, zero_trt=zero))[2].float().cpu().numpy())
        return np.concatenate(o)

    pred_sets = {k: Units(torch, S, sets[k], bins, digitize, dev, labels=False) for k in t["predict"].split(",")}
    treated_check = None
    for sel in selectors:
        ck = torch.load(os.path.join(out, f"ckpt_{sel}.pt"), map_location=dev)
        pm.load_state_dict(ck["model_state_dict"])
        for k, U in pred_sets.items():
            p = predict(U)
            f = os.path.join(out, f"pred_{sel}_{k}.npy"); np.save(f, p.astype(np.float32))
            preds[f"{sel}:{k}"] = dict(file=f, n=int(U.n), unit_idx_sha=sha_array(U.idx.astype(np.int64)), pred_sha256=sha256(f))
            if treated_check is None:          # the forward pass must not read the treated slot: noise vs zeros
                m = min(U.n, 512)
                U.trt = torch.randn_like(U.ctl) * 3 + 7
                U.trtb = torch.as_tensor(digitize(U.trt.cpu().numpy(), bins, side="one"), device=dev)
                with torch.no_grad():
                    q = np.concatenate([pm(U.batch(torch, F, torch.arange(i, min(i + bs, m), device=dev)))[2].float().cpu().numpy() for i in range(0, m, bs)])
                treated_check = float(np.abs(q - p[:m]).max())
                U.trt = torch.zeros_like(U.ctl); U.trtb = torch.zeros_like(U.ctlb)
    for k, U in pred_sets.items():
        np.save(os.path.join(out, f"unit_idx_{k}.npy"), U.idx.astype(np.int64))
    assert treated_check is not None and treated_check < 1e-4, f"prediction depends on the treated slot ({treated_check})"

    v1 = None
    if a.check_v1:
        ref = "xpr_e100_s42" if t["variant"] == "official" else "xpe_e100_s42"
        ck = torch.load(orig_xpert(ref, "expt", "r7_split_fold_early_stop.pth"), map_location=dev)
        pm.load_state_dict(ck["model_state_dict"])
        Uv = Units(torch, S, S.val, bins, digitize, dev, labels=True)
        p = predict(Uv)
        old = np.load(orig_xpert(ref, "pred_validation.npy"))
        model.load_state_dict(ck["model_state_dict"])
        l4_all, _ = dev_pass(Uv)
        logged = pd.read_csv(pman("orig_xpert_logged_curves.tsv"), sep="\t")
        lv = float(logged[(logged.run == ref) & (logged.epoch == int(ck["epoch"]))].logged_loss4_all_valid.iloc[0])
        v1 = dict(reference_run=ref, checkpoint_epoch=int(ck["epoch"]), pred_validation_max_abs_diff_vs_v1=float(np.abs(p - old).max()),
                  recomputed_loss4_all_valid=l4_all, logged_loss4_all_valid_3dp=lv, matches_3dp=abs(round(l4_all, 3) - lv) < 1e-9)
        print("v1 check", v1, flush=True)

    rec = dict(traj=t, run_id=run_id, epochs_run=epochs, selectors={s: {k: v for k, v in b.items()} for s, b in best.items()},
               n_units={k: int(len(v)) for k, v in sets.items()}, unit_idx_sha={k: sha_array(np.asarray(v, np.int64)) for k, v in sets.items()},
               predictions=preds, treated_slot_max_abs_diff=treated_check, v1_check=v1,
               config_source=cfg_src, drug_table=fpath, gpu=torch.cuda.get_device_name(0), seconds_train=round(t_train, 1),
               seconds_total=round(time.time() - t0, 1), max_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               mean_sec_train_epoch=float(np.mean([r["sec_train"] for r in curve])), mean_sec_select_epoch=float(np.mean([r["sec_select"] for r in curve])),
               torch=torch.__version__, **stamp())
    dump(rec, os.path.join(out, "record.json"))
    print(json.dumps({k: rec[k] for k in ["run_id", "selectors", "seconds_train", "mean_sec_train_epoch", "max_mem_gb", "gpu"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
