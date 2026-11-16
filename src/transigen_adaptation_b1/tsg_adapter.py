"""Adapter around the pinned upstream TranSiGen model class.

Upstream `model.py` is imported unmodified. This module supplies what the study needs around it:
explicit data paths (the upstream dataset hard-codes the example files and is never imported),
the upstream training step, and an inference path that takes only X0 and the drug fingerprint.
"""
import hashlib, importlib, json, os, subprocess, sys
import numpy as np
import torch

PIN = "8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a"
PUBLIC_CFG = dict(n_genes=978, n_latent=100, n_en_hidden=[1200], n_de_hidden=[800],
                  features_dim=2048, features_embed_dim=[400], beta=0.1, dropout=0.1, init_w=True)
WEIGHT_DECAY = 1e-5


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def upstream_src():
    root = os.environ["TB_TSG_SRC"]
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == PIN, f"upstream checkout is {head}, expected {PIN}"
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    assert dirty == "", f"upstream checkout has local modifications:\n{dirty}"
    return os.path.join(root, "src")


def load_upstream_model_class():
    src = upstream_src()
    if src not in sys.path:
        sys.path.insert(0, src)
    mod = importlib.import_module("model")
    assert os.path.realpath(mod.__file__) == os.path.realpath(os.path.join(src, "model.py"))
    # The upstream dataset module reads ../data/LINCS2020/data_example/* whatever path is passed.
    assert "dataset" not in sys.modules, "upstream dataset.py must not be imported in study runs"
    return mod.TranSiGen


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(seed, device, cfg=None):
    """Random-initialised model in the public ECFP4 configuration."""
    cfg = dict(PUBLIC_CFG if cfg is None else cfg)
    TranSiGen = load_upstream_model_class()
    seed_all(seed)
    m = TranSiGen(device=device, path_model="UNUSED/", random_seed=seed, **cfg)
    return m.to(device)


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(model, path, meta):
    assert not os.path.exists(path), f"refusing to overwrite {path}"
    tmp = path + ".tmp"
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, meta=meta), tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device, seed=0, cfg=None):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = build_model(seed, device, cfg)
    m.load_state_dict(ck["state_dict"], strict=True)
    return m, ck["meta"]


def upstream_train_step(model, optimizer, x0, x1, feat):
    """One optimizer update, the same calls as upstream TranSiGen.train_model."""
    model.train()
    optimizer.zero_grad()
    x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, _ = model.forward(x0, feat)
    z2, mu2, logvar2 = model.encode_x2(x1)
    x2_rec = model.decode_x2(z2)
    loss = model.loss(x0, x1_rec, mu1, logvar1, x1, x2_rec, mu2, logvar2, x2_pert, mu_pred, logvar_pred)[0]
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def upstream_eval_loss(model, x0, x1, feat, seed, batch=4096):
    """Upstream validation loss per sample (latents are sampled, so the generator is seeded)."""
    model.eval()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tot = np.zeros(7)
    for i in range(0, x0.shape[0], batch):
        a, b, f = x0[i:i + batch], x1[i:i + batch], feat[i:i + batch]
        x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, _ = model.forward(a, f)
        z2, mu2, logvar2 = model.encode_x2(b)
        x2_rec = model.decode_x2(z2)
        ls = model.loss(a, x1_rec, mu1, logvar1, b, x2_rec, mu2, logvar2, x2_pert, mu_pred, logvar_pred)
        tot += np.array([v.item() for v in ls])
    names = ["loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert"]
    return dict(zip(names, tot / x0.shape[0]))


@torch.no_grad()
def predict_x1_mean_path(model, x0, feat, batch=8192):
    """Predicted treated profile from X0 and the fingerprint only, posterior means, eval mode.

    The signature has no X1 argument on purpose: the X1 encoder / reconstruction path cannot be
    reached from here.
    """
    model.eval()
    out = []
    for i in range(0, x0.shape[0], batch):
        a, f = x0[i:i + batch], feat[i:i + batch]
        mu1 = model.mu_z1(model.encoder_x1(a))
        fe = model.feat_embeddings(f)
        mu_pred = model.mu_z2Fz1(torch.cat([mu1, fe], 1))
        out.append(model.decoder_x2(mu_pred))
    return torch.cat(out, 0)


@torch.no_grad()
def predict_x1_sampled(model, x0, feat, seed):
    """Upstream forward() prediction (one latent sample), for the smoke test only."""
    model.eval()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return model.forward(x0, feat)[3]
