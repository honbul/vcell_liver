"""XPert inference wrapper (XPert env). Inputs: unit X0, the group's drug features, cell index. The treated profile slot
of the upstream forward() is filled with noise; a wrapper check asserts that the prediction does not change when that slot
changes (the forward pass must not read the treated RNA)."""
import argparse, json, logging, os, sys
import numpy as np
import torch

from .common import runs, dump, sha256, pman
from .data import Store

W = os.path.join(os.environ["R7_ASSETS"], "xpert_work")


def load(run):
    sys.path.insert(0, W); os.chdir(W)
    import yaml
    from models.model_XPert import XPertNet
    from datasets.MyDataset import _digitize
    cfg = yaml.safe_load(open(runs("xpert", run, "config_r7.yaml")))
    df = "morgan" if cfg["dataset"].get("drug_morgan_path") == "r7_drug_morgan1024.npy" else "unimol"
    args = argparse.Namespace(mode="test", drug_feat=df, dataset="l1000_r7sdst", pretrained_mode="global", include_cell_idx=True,
                              wo_HG=False, wo_atom=False, wo_atom_HG=False, wo_unimol=False, wo_ppi=False, use_gene_pos_emed=False,
                              output_attention=False, output_cls_embed=False)
    dev = torch.device("cuda")
    model = XPertNet(args, cfg, dev, logging.getLogger("xpert")).to(dev)
    ck = [f for f in os.listdir(runs("xpert", run, "expt")) if f.endswith("_fold_early_stop.pth")]
    assert len(ck) == 1, ck
    path = runs("xpert", run, "expt", ck[0])
    model.load_state_dict(torch.load(path, map_location=dev)["model_state_dict"])
    model.eval()
    bins = np.quantile(np.array([cfg["dataset"]["min_value"], cfg["dataset"]["max_value"]]), np.linspace(0, 1, cfg["dataset"]["n_bins"] - 1))
    return model, dev, bins, _digitize, path, df


@torch.no_grad()
def predict(model, dev, bins, digitize, S, idx, F, trt_fill="noise", bs=128, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        ctl = S.x0[j].astype(np.float32)
        trt = (rng.normal(7, 3, ctl.shape) if trt_fill == "noise" else np.zeros_like(ctl)).astype(np.float32)
        gi = S.group_idx[j]
        data = (torch.as_tensor(trt), torch.as_tensor(ctl), torch.as_tensor(digitize(trt, bins, side="one")), torch.as_tensor(digitize(ctl, bins, side="one")),
                torch.as_tensor(F[gi]), torch.zeros(len(j), dtype=torch.long), torch.zeros(len(j), dtype=torch.long),
                torch.as_tensor(gi, dtype=torch.long), torch.as_tensor(S.cell_idx[j], dtype=torch.long), torch.zeros(len(j), dtype=torch.long))
        o = model(data)
        out.append(o[2].float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    a = ap.parse_args()
    S = Store()
    root = os.getcwd()
    model, dev, bins, dig, ck, df = load(a.run)
    if df == "morgan":
        d = np.load(os.path.join(W, "r7_drug_morgan1024.npy"), allow_pickle=True).item()
        F = np.stack([d[g] for g in range(len(d))]).astype(np.float32)
    else:
        F = np.load(os.path.join(W, "r7_drug_unimol.npy"), mmap_mode="r")
    idx = S.val if a.split == "validation" else S.test
    p = predict(model, dev, bins, dig, S, idx, F, "noise", seed=0)
    chk = predict(model, dev, bins, dig, S, idx[:512], F, "zeros")
    diff = float(np.abs(chk - p[:512]).max())
    assert diff < 1e-4, f"prediction depends on the treated slot (max diff {diff})"
    f = runs("xpert", a.run, f"pred_{a.split}.npy"); np.save(f, p.astype(np.float32))
    rec = dict(run=a.run, split=a.split, checkpoint=ck, checkpoint_sha256=sha256(ck), n=int(len(idx)), treated_slot_max_abs_diff=diff, drug_feat=df,
               pred_sha256=sha256(f), job=os.environ.get("SLURM_JOB_ID"))
    os.chdir(root)
    dump(rec, pman("runs", f"xpert_{a.run}_pred_{a.split}.json"))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
