"""Source-only sanity diagnostic for the pretrained checkpoints (plan section 4.2: when a score is poor, check
errors, labels and the inference path first). Uses source rows only; no target row is read.

For each source seed: delta RMSE of the mean-path prediction on (a) a fixed sample of source-train rows (cell lines
seen in training) and (b) source-validation rows (held-out cell lines); the same with the upstream sampled forward();
and (b) again after removing, per held-out cell line, the mean residual vector of that cell line (how much of the
error is one constant offset per cell line)."""
import json, os
import numpy as np
import torch

from . import tsg_adapter as T
from .data import Data, MAN_DIR


def rmse(p, t):
    return float(np.sqrt(((p - t) ** 2).mean(1)).mean())


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Data()
    tr, va = D.source_split()
    tr = np.sort(np.random.RandomState(0).choice(tr, 20000, replace=False))
    x1 = D.source_x1()
    di = D.rows.drug_idx.to_numpy()
    sel = json.load(open(os.path.join(MAN_DIR, "source_selection.json")))
    out = {}
    for seed, e in sel["checkpoints"].items():
        m, _ = T.load_checkpoint(e["checkpoint"], dev, seed=int(seed))
        r = {}
        for name, idx in [("train_cells_sample", tr), ("heldout_cells", va)]:
            x0 = np.asarray(D.x0[idx])
            d = np.asarray(x1[idx]) - x0
            a, f = torch.tensor(x0, device=dev), torch.tensor(D.fp[di[idx]], dtype=torch.float32, device=dev)
            pm = T.predict_x1_mean_path(m, a, f).cpu().numpy() - x0
            ps = torch.cat([T.predict_x1_sampled(m, a[i:i + 8192], f[i:i + 8192], seed=1) for i in range(0, len(idx), 8192)]).cpu().numpy() - x0
            r[name] = dict(n_rows=int(len(idx)), no_change=rmse(np.zeros_like(d), d), mean_path=rmse(pm, d), sampled_forward=rmse(ps, d),
                           pred_rms=float(np.sqrt((pm ** 2).mean())), truth_rms=float(np.sqrt((d ** 2).mean())),
                           mean_rowwise_pearson=float(np.mean([np.corrcoef(pm[i], d[i])[0, 1] for i in range(0, len(idx), 7)])))
            if name == "heldout_cells":
                cells = D.rows.cell.iloc[idx].to_numpy()
                res = pm - d
                adj = pm.copy()
                for c in np.unique(cells):
                    k = cells == c
                    adj[k] -= res[k].mean(0)
                r[name]["mean_path_after_removing_per_cell_mean_residual"] = rmse(adj, d)
                r[name]["rms_of_per_cell_mean_residual"] = float(np.sqrt(np.mean([(res[cells == c].mean(0) ** 2).mean() for c in np.unique(cells)])))
        out[seed] = r
    json.dump(out, open(os.path.join(MAN_DIR, "source_diagnostic.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
