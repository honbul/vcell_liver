"""Fresh-fit Tahoe well-level adapters for pinned XPert, TranSiGen, and chemCPA."""

from argparse import Namespace
from functools import lru_cache
import importlib
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.r11_chemical_probe.common import sha256
from src.r11_chemical_probe.dataset import ENTITY, LINE, table

SEED = 42
EPOCHS = {"XPert_morgan": (3, 8, 16), "TranSiGen_Tahoe": (1, 5, 15, 30),
          "chemCPA_P0": (5, 15, 30)}
ROOT = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
XP = ROOT / "vcell_liver_evidence_gated_r7/assets"
TSG = ROOT / "vcell_liver_transigen_adaptation_b1"
CPA = ROOT / "vcell_liver_chemcpa_moe_v2"


def git_pin(path, prefix):
    commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    if not commit.startswith(prefix):
        raise ValueError(f"Unexpected upstream commit in {path}: {commit}")
    return commit


def tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device="cuda")


def modules_params(modules):
    return [p for module in modules for p in module.parameters()]


def batches(length, batch, rng=None):
    order = np.arange(length) if rng is None else rng.permutation(length)
    parts = list(np.array_split(order, max(1, int(np.ceil(length / batch)))))
    if len(parts) > 1 and len(parts[-1]) == 1:
        parts[-2] = np.concatenate(parts[-2:])
        parts.pop()
    return (tensor(p, torch.long) for p in parts)


@lru_cache(maxsize=4)
def ecfp_store(entities_file, length, chirality):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    entities = table(entities_file)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=length, includeChirality=chirality,
    )
    result = {}
    for entity, smiles in entities[[ENTITY, "smiles"]].itertuples(index=False, name=None):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid verified structure for {entity}")
        result[entity] = generator.GetFingerprintAsNumPy(mol)
    return result


def ecfp(fit, length, chirality):
    features = ecfp_store(fit.source.r11 / "features/entities.tsv", length, chirality)
    return tuple(tensor(np.stack([features[e] for e in block.rows[ENTITY]]))
                 for block in (fit.train, fit.query))


def run_path(train_step, predict, epochs, n, batch, *, validation=None, schedulers=()):
    rng = np.random.default_rng(SEED)
    scores = {}
    for epoch in range(1, max(epochs) + 1):
        for i in batches(n, batch, rng):
            train_step(i, epoch)
        for scheduler in schedulers:
            scheduler.step()
        if epoch in epochs and validation is not None:
            scores[epoch] = validation(predict())
    return scores if validation is not None else predict()


def xp_config(fit, directory, line_count):
    import yaml

    source = XP / "runs/xpert/xpe_e100_s42/config_r7.yaml"
    conf = yaml.safe_load(source.read_text())
    genes = table(fit.source.r11 / "metadata/genes.tsv").set_index("token_id")
    landmarks = table(TSG / "manifests/transigen_adaptation_b1/genes_landmark_978.tsv")
    r7 = XP / "xpert_work/processed_data/r7_PPI_gene_vector_128d.npy"
    ppi = np.load(r7, allow_pickle=False)
    if ppi.shape != (len(landmarks), 128):
        raise ValueError("Unverified R7 landmark PPI panel")
    lookup = dict(zip(landmarks.ensembl_id, ppi))
    ensembl = genes.loc[fit.panel, "ensembl_id"].tolist()
    vectors = np.stack([lookup.get(g, np.zeros(128, dtype=np.float32)) for g in ensembl])
    np.save(directory / "xpert-ppi.npy", vectors, allow_pickle=False)
    np.save(directory / "xpert-zero-hg.npy", np.zeros((1, 256), dtype=np.float32), allow_pickle=False)
    conf["dataset"]["gene_num"] = len(ensembl)
    conf["dataset"]["l1000"]["num_cell_id"] = line_count
    conf["model"]["ATTN"]["ppi_gene_vector_path"] = str(directory / "xpert-ppi.npy")
    conf["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(directory / "xpert-zero-hg.npy")
    return conf, dict(landmark_PPI=sum(g in lookup for g in ensembl), source_config_sha256=sha256(source),
                      landmark_PPI_sha256=sha256(r7))


def xpert(fit, work, epochs, validation):
    repo = XP / "external/XPert"
    pin = git_pin(repo, "d53ff49")
    sys.path.insert(0, str(repo))
    from models.model_XPert import XPertNet

    features = ecfp(fit, 1024, False)
    all_lines = sorted(set(fit.train.rows[LINE]) | set(fit.query.rows[LINE]))
    line_idx = {line: i for i, line in enumerate(all_lines)}
    conf, receipt = xp_config(fit, work, len(all_lines))
    args = Namespace(mode="train", drug_feat="morgan", dataset="l1000_r7sdst", pretrained_mode="global",
                     include_cell_idx=True, wo_HG=False, wo_atom=False, wo_atom_HG=False, wo_unimol=False,
                     wo_ppi=False, use_gene_pos_emed=False, output_attention=False, output_cls_embed=False)
    device = torch.device("cuda")
    model = XPertNet(args, conf, device, logging.getLogger("t2_xpert"))
    model.init_weights()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
    controls = tuple(tensor(block.control) for block in (fit.train, fit.query))
    treated = tensor(fit.train.treated)
    bins = np.linspace(conf["dataset"]["min_value"], conf["dataset"]["max_value"], conf["dataset"]["n_bins"] - 1)
    bins = tensor(bins)
    control_bins = tuple(torch.bucketize(v, bins, right=True) for v in controls)
    treated_bins = torch.bucketize(treated, bins, right=True)
    cell = tuple(tensor([line_idx[l] for l in block.rows[LINE]], torch.long) for block in (fit.train, fit.query))
    amp = torch.amp.GradScaler("cuda")

    def forward(j, mode):
        ctl = controls[mode][j]
        trt = treated[j] if mode == 0 else torch.zeros_like(ctl)
        ctl_bin = control_bins[mode][j]
        trt_bin = treated_bins[j] if mode == 0 else torch.zeros_like(ctl_bin)
        zeros = torch.zeros_like(cell[mode][j])
        with torch.autocast("cuda", dtype=torch.float16):
            return model((trt, ctl, trt_bin, ctl_bin, features[mode][j], zeros, zeros,
                          zeros, cell[mode][j], zeros))

    def train(j, epoch):
        model.train()
        opt.zero_grad(set_to_none=True)
        y1, _, d, truth, ctl, _, cls, logits = forward(j, 0)
        delta = truth - ctl
        cls_loss = sum(nn.functional.cross_entropy(z.float(), cls) for z in logits)
        dy = d.float()
        dy_center = dy - dy.mean(1, keepdim=True)
        delta_center = delta - delta.mean(1, keepdim=True)
        pearson = nn.functional.cosine_similarity(dy_center, delta_center, dim=1, eps=1e-8)
        loss = (0.2 * (y1.float() - truth).square().mean().clamp_min(1e-12).sqrt()
                + 0.003 * cls_loss + 0.2 * (dy - delta).square().mean().clamp_min(1e-12).sqrt()
                + 1 - pearson.mean())
        amp.scale(loss).backward()
        amp.step(opt)
        amp.update()

    @torch.inference_mode()
    def predict():
        model.eval()
        return torch.cat([forward(j, 1)[2].float() for j in batches(len(fit.query.rows), 24)]).cpu().numpy()

    return run_path(train, predict, epochs, len(fit.train.rows), 24, validation=validation), dict(
        receipt, upstream_commit=pin, batch=24, lr=0.004, architecture="R7 Morgan-1024 zero-HG, XPertNet",
        objective="R7 L0 early composite loss; short fixed epoch grid; fp32 loss reduction",
        inputs="Tahoe well pseudobulk, zero prior vector for genes not in landmark PPI axis")


def transigen(fit, work, epochs, validation):
    repo = TSG / "assets/external/TranSiGen"
    pin = git_pin(repo, "8ec2218e")
    sys.path.insert(0, str(repo / "src"))
    mod = importlib.import_module("model")
    if Path(mod.__file__).resolve() != (repo / "src/model.py").resolve():
        raise ValueError("Wrong TranSiGen model module")
    # B1 public architecture, resized to the common Tahoe panel and fit without a pretrained head.
    model = mod.TranSiGen(device="cuda", path_model=str(work), random_seed=SEED,
                         n_genes=len(fit.panel), n_latent=100, n_en_hidden=[1200], n_de_hidden=[800],
                         features_dim=2048, features_embed_dim=[400], beta=0.1, dropout=0.1, init_w=True).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    features = ecfp(fit, 2048, False)
    controls = tuple(tensor(block.control) for block in (fit.train, fit.query))
    treated = tensor(fit.train.treated)

    def train(j, epoch):
        model.train()
        opt.zero_grad(set_to_none=True)
        ctl, truth, f = controls[0][j], treated[j], features[0][j]
        ctl_rec, m1, v1, pert, mp, vp, _ = model(ctl, f)
        z2, m2, v2 = model.encode_x2(truth)
        loss = model.loss(ctl, ctl_rec, m1, v1, truth, model.decode_x2(z2), m2, v2, pert, mp, vp)[0]
        (loss / len(j)).backward()
        opt.step()

    @torch.inference_mode()
    def predict():
        model.eval()
        out = []
        for j in batches(len(fit.query.rows), 512):
            ctl, f = controls[1][j], features[1][j]
            latent = model.mu_z1(model.encoder_x1(ctl))
            drug = model.feat_embeddings(f)
            mean = model.mu_z2Fz1(torch.cat([latent, drug], dim=1))
            out.append(model.decoder_x2(mean) - ctl)
        return torch.cat(out).cpu().numpy()

    return run_path(train, predict, epochs, len(fit.train.rows), 128, validation=validation), dict(
        upstream_commit=pin, batch=128, lr=0.0003, architecture="B1 public TranSiGen resized to Tahoe panel",
        inputs="fold-fit Tahoe well mean logCP10k, ECFP4 2048 bits, no CMap checkpoint")


@lru_cache(maxsize=2)
def cpa_descriptors(entity_file):
    from descriptastorus.descriptors.rdDescriptors import RDKit2D

    generator = RDKit2D()
    table_rows = table(entity_file)
    feats = {}
    for entity, smiles in table_rows[[ENTITY, "smiles"]].itertuples(index=False, name=None):
        result = generator.process(smiles)
        if not result or not result[0] or not np.isfinite(result[1:]).all():
            raise ValueError(f"Invalid RDKit2D representation for {entity}")
        feats[entity] = np.asarray(result[1:], dtype=np.float32)
    return feats, len(next(iter(feats.values())))


def chemcpa(fit, work, epochs, validation):
    repo = CPA / "assets/external/chemCPA"
    pin = git_pin(repo, "43e830eb")
    sys.path.insert(0, str(repo))
    from chemCPA.model import ComPert
    import json

    hparams_file = CPA / "configs/chemcpa_moe_v2/hparams_official.json"
    hparams = json.loads(hparams_file.read_text())
    desc, raw_width = cpa_descriptors(fit.source.r11 / "features/entities.tsv")
    entities = sorted(set(fit.train.rows[ENTITY]) | set(fit.query.rows[ENTITY]))
    inds = {e: i for i, e in enumerate(entities)}
    fit_positions = np.array([inds[e] for e in sorted(fit.train.rows[ENTITY].unique())])
    all_x = np.stack([desc[e] for e in entities])
    all_sd = all_x[fit_positions].std(axis=0, ddof=1)
    keep = all_sd > 0.01
    scaled = (all_x[:, keep] - all_x[fit_positions][:, keep].mean(0)) / all_sd[keep]
    embedding = nn.Embedding.from_pretrained(tensor(scaled), freeze=True)
    line_id = {l: i for i, l in enumerate(sorted(set(fit.train.rows[LINE]) | set(fit.query.rows[LINE])))}
    model = ComPert(num_genes=len(fit.panel), num_drugs=len(entities), num_covariates=[len(line_id)],
                   device="cuda", seed=SEED, doser_type="amortized", decoder_activation="linear",
                   hparams=hparams, drug_embeddings=embedding, use_drugs_idx=True).cuda()
    groups = dict(autoencoder=modules_params((model.encoder, model.decoder, model.drug_embedding_encoder,
                                             model.covariates_embeddings)),
                  adversary=modules_params((model.adversary_drugs, model.adversary_covariates)),
                  dosers=list(model.dosers.parameters()))
    opt = {k: torch.optim.Adam(params, lr=hparams[f"{k}_lr"], weight_decay=hparams[f"{k}_wd"])
           for k, params in groups.items()}
    scheds = [torch.optim.lr_scheduler.StepLR(o, hparams["step_size_lr"], gamma=0.9) for o in opt.values()]
    cov = tuple(nn.functional.one_hot(tensor([line_id[l] for l in block.rows[LINE]], torch.long),
                                     len(line_id)).float() for block in (fit.train, fit.query))
    drugs = tuple(tensor([inds[e] for e in block.rows[ENTITY]], torch.long)
                  for block in (fit.train, fit.query))
    controls = tuple(tensor(block.control) for block in (fit.train, fit.query))
    train_target = tensor(fit.train.treated)
    dose = float(fit.source.dose)
    global_step = 0

    def forward(x, d, c, return_latent=False):
        return model.predict(x, drugs_idx=d, dosages=torch.full_like(d, dose, dtype=torch.float32),
                             covariates=[c], return_latent_basal=return_latent)

    def train(j, epoch):
        nonlocal global_step
        model.train()
        model.zero_grad(set_to_none=True)
        out, _, (basal, _, _) = forward(train_target[j], drugs[0][j], cov[0][j], True)
        mean, var = out.chunk(2, dim=1)
        drug_adv = model.adversary_drugs(basal)
        line_adv = model.adversary_covariates[0](basal)
        drug_loss = model.loss_adversary_drugs(drug_adv, nn.functional.one_hot(
            drugs[0][j], len(entities)).float())
        line_loss = nn.functional.cross_entropy(line_adv, cov[0][j].argmax(1))
        if global_step % hparams["adversary_steps"] == 0:
            penalty = sum(torch.autograd.grad(v.sum(), basal, create_graph=True)[0].square().mean()
                          for v in (drug_adv, line_adv))
            (drug_loss + line_loss + hparams["penalty_adversary"] * penalty).backward()
            opt["adversary"].step()
            global_step += 1
        else:
            recon = model.loss_autoencoder(mean, train_target[j], var)
            (recon - hparams["reg_adversary"] * drug_loss - hparams["reg_adversary_cov"] * line_loss).backward()
            for key in ("autoencoder", "dosers"):
                nn.utils.clip_grad_norm_(groups[key], 1.0)
                opt[key].step()
            global_step += 2

    @torch.inference_mode()
    def predict():
        model.eval()
        out = []
        for j in batches(len(fit.query.rows), 512):
            ctl = controls[1][j]
            rec, _ = forward(ctl, drugs[1][j], cov[1][j])
            out.append(rec[:, :len(fit.panel)] - ctl)
        return torch.cat(out).cpu().numpy()

    return run_path(train, predict, epochs, len(fit.train.rows), 256, validation=validation, schedulers=scheds), dict(
        upstream_commit=pin, hparams_sha256=sha256(hparams_file), batch=256,
        architecture="unmodified chemCPA ComPert P0 with amortized dose decoder",
        objective="upstream Lightning Gaussian NLL + drug and line adversarial steps, reconstructed on treated means",
        inputs="Tahoe well pseudobulk and upstream RDKit2D descriptors; fit-only variance and scaling",
        rdkit2d_raw_columns=raw_width, rdkit2d_fit_columns=int(keep.sum()))


def upstream(name):
    return {"XPert_morgan": xpert, "TranSiGen_Tahoe": transigen, "chemCPA_P0": chemcpa}[name]
