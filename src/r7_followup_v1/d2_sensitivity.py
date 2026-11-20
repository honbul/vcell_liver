"""D2: how much do the FIXED original checkpoints (modification-1 B11, seeds 42-44) use the target seed?

Inputs are SOURCE units only (original validation and a fixed subset of original train, non-HCC cell lines); no test
unit is touched. For the same drug / cell input three corrections are computed with q / status / m_available unchanged:
    r(S, q, G)        real targets
    r(empty, q, G)    seed list removed (status and m_available keep their real values)
    r(S_swap, q, G)   identity swapped by the fixed null mappings 142-144 of the original split (count / status kept)
This is an input-manipulation sensitivity of fixed models (possibly off-distribution inputs); it is not a retrained
attribution control and no causal statement about targets follows from it."""
import json, os, time
import numpy as np
import pandas as pd
import torch

from src.evidence_gated_r7.data import Store, Knowledge
from src.evidence_gated_r7.evaluate import unit_metrics
from .common import MAN, orig_runs, pman, dump, sha256, tag_cols, NULL_SEEDS
from .kgraph import load_kg, k_groups
from .models import FollowupModel
from .shuffle import load_mapping
from .stats import FamBoot, contrast_row
from .train import FoldStats, Bank, tensors, load_graph, forward_idx

CKPTS = {42: "m1_B11_s42", 43: "m1_B11_s43", 44: "m1_B11_s44"}


@torch.no_grad()
def run(model, T, bank, idx, dev, bs=64):
    out, corr, gs = [], [], []
    for i in range(0, len(idx), bs):
        d, r, a = forward_idx(model, T, bank, idx[i:i + bs], dev)
        out.append(d.cpu().numpy()); corr.append(r.cpu().numpy())
        gs.append(torch.stack([a.mean(1), a.var(1), (a < 0.05).float().mean(1), (a > 0.95).float().mean(1)], 1).cpu().numpy())
    return np.concatenate(out), np.concatenate(corr), np.concatenate(gs)


def rms(x):
    return float(np.sqrt((x.astype(np.float64) ** 2).mean())) if len(x) else np.nan


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    S = Store(); st = FoldStats(S, S.train); K = Knowledge(len(S.groups))
    klab, _ = k_groups(load_kg(os.environ["R7_KG"]))
    T = tensors(S, st, dev)
    man = json.load(open(os.path.join(MAN, "target_shuffle_manifest.json")))["mappings"]
    banks = dict(real=Bank(K, "real"), empty=Bank(K, "noseed"))
    for ns in NULL_SEEDS:
        banks[f"swap{ns}"] = Bank(K, "null", load_mapping(man[f"orig_null{ns}"]["path"]))
    val = S.val[S.source(S.val)]
    rng = np.random.RandomState(0)
    trs = S.train[S.source(S.train)]; trs = np.sort(rng.choice(trs, 8000, replace=False))
    sets = dict(validation_source=val, train_source_subset=trs)
    rows, unit = [], {}
    for seed, rid in CKPTS.items():
        ck = orig_runs("gated", rid, "best.pt")
        model = FollowupModel("GR", load_graph(K)).to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
        for sname, idx in sets.items():
            res = {b: run(model, T, bank, idx, dev) for b, bank in banks.items()}
            kg_ = klab[S.group_idx[idx]]
            ann = kg_ != "K0"
            pred, r, g = res["real"]; chem = pred - r
            r0 = res["empty"][1]
            assert float(np.abs(g - res["empty"][2]).max()) == 0.0, "gates changed with the seed list (they should depend on q and edge features only)"
            assert float(np.abs(r[~ann]).max()) == 0.0 and float(np.abs(r0[~ann]).max()) == 0.0
            for grp in ["annotated", "K1", "K2", "K3"]:
                s = ann if grp == "annotated" else (kg_ == grp)
                sw = [rms(r[s] - res[f"swap{ns}"][1][s]) for ns in NULL_SEEDS]
                cosv = (r[s] * r0[s]).sum(1) / (np.linalg.norm(r[s], axis=1) * np.linalg.norm(r0[s], axis=1) + 1e-12)
                rows.append(dict(checkpoint=rid, model_seed=seed, unit_set=sname, k_group=grp, n_units=int(s.sum()), n_drugs=int(len(set(S.group_idx[idx][s]))),
                                 rms_chem=rms(chem[s]), rms_r=rms(r[s]), rms_r_empty_seed=rms(r0[s]), rms_r_minus_empty=rms(r[s] - r0[s]),
                                 **{f"rms_r_minus_swap{ns}": v for ns, v in zip(NULL_SEEDS, sw)}, rms_r_minus_swap_mean=float(np.mean(sw)),
                                 ratio_minus_empty_to_r=rms(r[s] - r0[s]) / rms(r[s]), ratio_minus_swap_to_r=float(np.mean(sw)) / rms(r[s]),
                                 mean_cos_r_vs_r_empty=float(cosv.mean()),
                                 gate_mean=float(g[s, 0].mean()), gate_var_across_edges=float(g[s, 1].mean()), gate_var_across_samples=float(g[s, 0].var()),
                                 gate_frac_below_0p05=float(g[s, 2].mean()), gate_frac_above_0p95=float(g[s, 3].mean()),
                                 gate_max_abs_change_seed_removed=0.0))
            if sname == "validation_source":
                y = S.delta(idx); cells = S.units.cell.to_numpy()[idx]
                um = lambda p: unit_metrics(p, y, cells).rmse.to_numpy()
                unit.setdefault("chem", []).append(um(chem)); unit.setdefault("real", []).append(um(pred)); unit.setdefault("empty", []).append(um(res["empty"][0]))
                unit.setdefault("swap", []).append(np.mean([um(res[f"swap{ns}"][0]) for ns in NULL_SEEDS], 0))
    t = pd.DataFrame(rows)
    num = [c for c in t.columns if c not in ("checkpoint", "model_seed", "unit_set", "k_group")]
    mean = t.groupby(["unit_set", "k_group"], sort=False)[num].mean().reset_index()
    mean.insert(0, "checkpoint", "mean_of_3_seeds"); mean.insert(1, "model_seed", -1)
    t = pd.concat([t, mean], ignore_index=True)
    tag_cols(t, "source_development(original validation / train inputs; fixed checkpoints)",
             metric_definition="RMS over units x 978 genes of the correction r and of its change when only the seed list is removed / swapped; q, status, m_available unchanged").to_csv(
        pman("checkpoint_target_sensitivity.tsv"), sep="\t", index=False)
    # error on the original validation source panel (already-open development data), seed-averaged unit RMSE
    uv = S.units.iloc[val].reset_index(drop=True)
    kg_ = klab[S.group_idx[val]]
    fb = FamBoot(uv.drug_group.to_numpy(), uv.family.to_numpy())
    U = {k: np.mean(v, 0) for k, v in unit.items()}
    crow = []
    for grp in ["all", "annotated", "K1", "K2", "K3"]:
        s = None if grp == "all" else ((kg_ != "K0") if grp == "annotated" else (kg_ == grp))
        for name in U:
            crow.append(dict(kind="level", a=name, b="", k_group=grp, diff=fb.stat(U[name], s, boot=False)[0]))
        for a_, b_ in [("real", "chem"), ("empty", "chem"), ("swap", "chem"), ("real", "empty"), ("real", "swap")]:
            crow.append(dict(kind="contrast", a=a_, b=b_, k_group=grp, **contrast_row(fb, U[a_] - U[b_], s)))
    c = pd.DataFrame(crow).rename(columns={"diff": "value"})
    tag_cols(c, "source_development(original validation, labels already open; fixed checkpoints)",
             metric_definition="family-balanced source-panel delta RMSE (level) or candidate-comparator difference with 95 % paired drug-group bootstrap interval; seed-averaged unit RMSE; swap = mean over null mappings 142-144",
             n_units=int(len(val)), n_drugs=int(uv.drug_group.nunique())).to_csv(pman("checkpoint_target_sensitivity_error.tsv"), sep="\t", index=False)
    dump(dict(checkpoints={r: sha256(orig_runs("gated", r, "best.pt")) for r in CKPTS.values()}, seconds=round(time.time() - t0),
              commit=os.environ.get("R7F_COMMIT"), job=os.environ.get("SLURM_JOB_ID"), note="original checkpoints were read only"), pman("d2_record.json"))
    print(t[t.checkpoint == "mean_of_3_seeds"].round(5).to_string()); print(c.round(5).to_string())


if __name__ == "__main__":
    main()
