"""Stage B0 smoke test on the upstream example files only (never used for study results).

1. describe the example data;  2. load the public full checkpoint and run forward / loss / inference;
3. train a fresh ECFP4 model for a few epochs through the adapter.
The public checkpoint's training exposure to the HCC targets is unknown, so nothing here is a result.
"""
import argparse, json, os, pickle, sys
import h5py
import numpy as np
import torch

from . import tsg_adapter as T


def h5_dict(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f:
            v = np.asarray(f[k])
            out[k] = v.astype(str) if v.dtype.kind == "S" else v
    return out


def describe(d):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                    **({"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())} if v.dtype.kind == "f" else
                       {"n_unique": int(len(np.unique(v)))})) for k, v in d.items()}


def rowwise(pred_delta, true_delta):
    rmse = np.sqrt(((pred_delta - true_delta) ** 2).mean(1))
    a = pred_delta - pred_delta.mean(1, keepdims=True)
    b = true_delta - true_delta.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)
    return float(rmse.mean()), float(r.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.environ["TB_TSG_SRC"]
    ex = os.path.join(root, "data/LINCS2020/data_example")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = dict(upstream_commit=T.PIN, device=str(dev), torch=torch.__version__)
    files = {n: os.path.join(root, n) for n in [
        "data/LINCS2020/data_example/processed_data.h5", "data/LINCS2020/data_example/processed_data_id.h5",
        "data/LINCS2020/data_example/ECFP4_emb2048.pickle", "data/LINCS2020/data_example/KPGT_emb2304.pickle",
        "data/LINCS2020/processed_data_id.h5", "data/LINCS2020/idx2smi.pickle",
        "results/trained_models_164_cell_smiles_split/364039/feature_KPGT_init_pretrain_shRNA/best_model.pt"]}
    rep["files"] = {n: dict(sha256=T.sha256(p), bytes=os.path.getsize(p)) for n, p in files.items()}
    data = h5_dict(files["data/LINCS2020/data_example/processed_data.h5"])
    ids = h5_dict(files["data/LINCS2020/data_example/processed_data_id.h5"])
    full_ids = h5_dict(files["data/LINCS2020/processed_data_id.h5"])
    rep["example_processed_data"] = describe(data)
    rep["example_processed_data_id"] = describe(ids)
    rep["full_processed_data_id"] = describe(full_ids)
    rep["full_id_file_has_expression"] = bool(any(v.dtype.kind == "f" and v.ndim == 2 for v in full_ids.values()))
    cells = sorted(set(full_ids["cid"].tolist())) if "cid" in full_ids else []
    rep["full_id_file_cells"] = len(cells)
    rep["full_id_file_target_cells_present"] = [c for c in cells if c.upper().replace("-", "") in ("HUH7", "HUH751", "JHH5", "JHH7")]
    with open(files["data/LINCS2020/idx2smi.pickle"], "rb") as f:
        idx2smi = pickle.load(f)
    with open(files["data/LINCS2020/data_example/ECFP4_emb2048.pickle"], "rb") as f:
        ecfp = pickle.load(f)
    with open(files["data/LINCS2020/data_example/KPGT_emb2304.pickle"], "rb") as f:
        kpgt = pickle.load(f)
    e0 = np.asarray(next(iter(ecfp.values())))
    k0 = np.asarray(next(iter(kpgt.values())))
    rep["example_embeddings"] = dict(ecfp4=dict(n=len(ecfp), shape=list(e0.shape), dtype=str(e0.dtype), values=sorted(set(np.unique(e0).tolist()))[:5]),
                                     kpgt=dict(n=len(kpgt), shape=list(k0.shape), dtype=str(k0.dtype)))

    li = ids["LINCS_index"].astype(int)
    smi_idx = ids["canonical_smiles"].astype(int)
    x0 = torch.tensor(data["x1"][li], dtype=torch.float32, device=dev)   # upstream calls the control profile x1
    x1 = torch.tensor(data["x2"][li], dtype=torch.float32, device=dev)   # and the treated profile x2
    f_k = torch.tensor(np.stack([kpgt[idx2smi[i]] for i in smi_idx]), dtype=torch.float32, device=dev)
    f_e = torch.tensor(np.stack([ecfp[idx2smi[i]] for i in smi_idx]), dtype=torch.float32, device=dev)
    rep["example_pairs"] = dict(n=int(x0.shape[0]), cells=sorted(set(ids["cid"].tolist())), n_smiles=int(len(set(smi_idx.tolist()))))

    # ---- public checkpoint (whole-object pickle, needs the upstream module on the path)
    T.load_upstream_model_class()
    pub = torch.load(files["results/trained_models_164_cell_smiles_split/364039/feature_KPGT_init_pretrain_shRNA/best_model.pt"],
                     map_location="cpu", weights_only=False)
    pub.dev = dev
    pub.to(dev)
    rep["public_checkpoint"] = dict(cls=type(pub).__name__, features_dim=pub.features_dim, n_latent=pub.n_latent,
                                    n_params=int(sum(p.numel() for p in pub.parameters())), state_hash=T.state_hash(pub))
    loss = T.upstream_eval_loss(pub, x0, x1, f_k, seed=0)
    pm = T.predict_x1_mean_path(pub, x0, f_k)
    pm2 = T.predict_x1_mean_path(pub, x0, f_k)
    ps_a = T.predict_x1_sampled(pub, x0, f_k, seed=1)
    ps_b = T.predict_x1_sampled(pub, x0, f_k, seed=2)
    with torch.no_grad():
        pub.eval()
        rec = pub.decode_x2(pub.mu_z2(pub.encoder_x2(x1)))     # reconstruction: uses the measured X1
    d = (x1 - x0).cpu().numpy()
    g = lambda p: rowwise((p - x0).cpu().numpy(), d)
    rep["public_checkpoint"].update(
        upstream_loss_per_sample=loss, output_shape=list(pm.shape),
        mean_path_repeatable=bool(torch.equal(pm, pm2)), sampled_path_differs_between_seeds=bool(not torch.equal(ps_a, ps_b)),
        prediction_mean_path=dict(zip(["delta_rmse", "delta_pearson"], g(pm))),
        prediction_sampled_seed1=dict(zip(["delta_rmse", "delta_pearson"], g(ps_a))),
        reconstruction_from_X1_NOT_a_prediction=dict(zip(["delta_rmse", "delta_pearson"], g(rec))),
        no_change=dict(delta_rmse=float(np.sqrt((d ** 2).mean(1)).mean())),
        caveat="example rows may be training rows of this checkpoint; these numbers only show that the code path runs")

    # ---- fresh ECFP4 model through the adapter
    n = x0.shape[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    tr, va = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    tr, va = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
    m = T.build_model(42, dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=T.WEIGHT_DECAY)
    hist = []
    dv = (x1[va] - x0[va]).cpu().numpy()
    val = lambda: rowwise((T.predict_x1_mean_path(m, x0[va], f_e[va]) - x0[va]).cpu().numpy(), dv)[0]
    hist.append(dict(epoch=0, val_delta_rmse=val()))
    gen = torch.Generator(device="cpu").manual_seed(42)
    for ep in range(1, 201):
        p = tr[torch.randperm(len(tr), generator=gen)]
        tot, cnt = 0.0, 0
        for i in range(0, len(p), 16):
            b = p[i:i + 16]
            if len(b) < 2:
                continue
            tot += T.upstream_train_step(m, opt, x0[b], x1[b], f_e[b])
            cnt += len(b)
        if ep % 10 == 0 or ep == 1:
            hist.append(dict(epoch=ep, updates=ep * 5, train_loss_per_sample=tot / cnt, val_delta_rmse=val()))
    rep["fresh_ecfp4_model_on_example"] = dict(n_train=int(len(tr)), n_val=int(len(va)), batch_size=16, history=hist,
                                               n_params=int(sum(p.numel() for p in m.parameters())),
                                               val_no_change_delta_rmse=float(np.sqrt((dv ** 2).mean(1)).mean()))
    rep["upstream_dataset_module_imported"] = "dataset" in sys.modules
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
