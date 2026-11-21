"""HepaRG paired group bootstrap and descriptive model-gain concentration audit."""

import numpy as np
import pandas as pd

from src.p123.p3_models import MODEL_NAMES, subsets
from src.r14_cross_organ.heparg_eval import SEED, center

DRAWS = 10000
REFERENCES = ("B0", "dose_shared_ridge", "p3_joint_chemistry")
DELETE_K = (5, 10, 20)


def component(truth, pred):
    value = np.asarray(truth, dtype=np.float64) - pred
    return np.einsum("ij,ij->i", value, value)


def estimate(rmse, sse, sst):
    return rmse.mean(axis=0), 1 - sse.sum(axis=0) / sst.sum()


def interval(draws):
    finite = np.isfinite(draws)
    if not finite.any():
        return [np.nan, np.nan]
    return np.quantile(draws[finite], [0.025, 0.975]).tolist()


def paired_summaries(parent, names, flags):
    n = len(parent)
    rmse = parent[[f"rmse_{a}" for a in names]].to_numpy()
    sse = parent[[f"sse_{a}" for a in names]].to_numpy()
    sst = parent.sst.to_numpy()
    p, r2 = estimate(rmse, sse, sst)
    rmse_draw = np.empty((DRAWS, len(names)), dtype=np.float64)
    r2_draw = np.empty_like(rmse_draw)
    rng = np.random.default_rng(SEED)
    for start in range(0, DRAWS, 128):
        block = slice(start, min(DRAWS, start + 128))
        draw = rng.integers(n, size=(block.stop - block.start, n))
        rmse_draw[block] = rmse[draw].mean(axis=1)
        r2_draw[block] = 1 - sse[draw].sum(axis=1) / sst[draw].sum(axis=1)[:, None]
    rows = []
    for i, arm in enumerate(names):
        common = {
            **flags, "arm": arm, "parents": n, "wells": int(parent.wells.sum()),
            "macro_rmse": float(p[i]), "r2_pooled": float(r2[i]),
            "macro_rmse_lower95": interval(rmse_draw[:, i])[0], "macro_rmse_upper95": interval(rmse_draw[:, i])[1],
            "r2_lower95": interval(r2_draw[:, i])[0], "r2_upper95": interval(r2_draw[:, i])[1],
        }
        for baseline in REFERENCES:
            j = names.index(baseline)
            gain_rmse = 100 * (1 - p[i] / p[j])
            gain_r2 = r2[i] - r2[j]
            mlo, mhi = interval(100 * (1 - rmse_draw[:, i] / rmse_draw[:, j]))
            rlo, rhi = interval(r2_draw[:, i] - r2_draw[:, j])
            rows.append({**common, "reference": baseline, "rmse_reduction_pct": gain_rmse,
                         "rmse_gain_lower95": mlo, "rmse_gain_upper95": mhi,
                         "delta_r2": gain_r2, "delta_r2_lower95": rlo, "delta_r2_upper95": rhi,
                         "rmse_improvement": bool(mlo > 0), "r2_improvement": bool(rlo > 0),
                         "both_improve": bool(mlo > 0 and rlo > 0)})
        if "repeat_x0.25" in names:
            j = names.index("repeat_x0.25")
            base = names.index("B0")
            delta = p[base] - p[j]
            gap = rmse_draw[:, base] - rmse_draw[:, j]
            glo, ghi = interval(gap)
            rdelta = r2[j] - r2[base]
            rgap = r2_draw[:, j] - r2_draw[:, base]
            rglo, rghi = interval(rgap)
            mlo, mhi = interval((rmse_draw[:, base] - rmse_draw[:, i]) / gap * 100) if glo > 0 else (np.nan, np.nan)
            rlo, rhi = interval((r2_draw[:, i] - r2_draw[:, base]) / rgap * 100) if rglo > 0 else (np.nan, np.nan)
            rows.append({
                **common, "reference": "repeat_x0.25", "repeat_rmse_gap": delta,
                "repeat_rmse_gap_lower95": glo, "repeat_rmse_gap_upper95": ghi,
                "repeat_r2_gap": rdelta, "repeat_r2_gap_lower95": rglo, "repeat_r2_gap_upper95": rghi,
                "repeat_rmse_reach_pct": (p[base] - p[i]) / delta * 100 if delta > 0 else np.nan,
                "repeat_rmse_reach_lower95": mlo, "repeat_rmse_reach_upper95": mhi,
                "repeat_r2_reach_pct": (r2[i] - r2[base]) / rdelta * 100 if rdelta > 0 else np.nan,
                "repeat_r2_reach_lower95": rlo, "repeat_r2_reach_upper95": rhi,
            })
    return rows


def parent_table(frame, truth, predictions):
    n_genes = truth.shape[1]
    sst = component(truth, 0)
    table = pd.DataFrame({
        "parent_key": frame.parent_key.to_numpy(),
        "wells": np.ones(len(frame), dtype=np.int64),
        "sst": sst,
        "rmse_B0": np.sqrt(sst / n_genes),
        "sse_B0": sst,
    })
    for name, p in predictions.items():
        error = component(truth, p)
        table[f"rmse_{name}"] = np.sqrt(error / n_genes)
        table[f"sse_{name}"] = error
    aggs = {p: ("mean" if p.startswith("rmse") else "sum") for p in table if p != "parent_key"}
    return table.groupby("parent_key", sort=True).agg(aggs)


def dropped_metrics(parts, groups, arms):
    bank = []
    for frame, truth, preds, _ in parts:
        retained = frame.loc[~frame.parent_key.isin(groups)]
        supported = retained.groupby(["plate", "dose_rank"]).parent_key.transform("nunique").ge(3)
        retained = retained.loc[supported]
        if len(retained):
            ind = retained.index.to_numpy()
            centered = center(retained, truth[ind].copy())
            arms_out = {name: center(retained, preds[name][ind].copy()) for name in arms if name != "B0"}
            bank.append(parent_table(retained, centered, arms_out))
    if not bank:
        return {"retained_parents": 0, "retained_wells": 0}
    table = pd.concat(bank, verify_integrity=True)
    total = table.sst.sum()
    values = {"retained_parents": len(table), "retained_wells": int(table.wells.sum())}
    for name, arm in zip(("model", "ref"), arms):
        values[f"rmse_{name}"] = table[f"rmse_{arm}"].mean()
        values[f"r2_{name}"] = 1 - table[f"sse_{arm}"].sum() / total if total > 0 else np.nan
    values["rmse_delta"] = values["rmse_model"] - values["rmse_ref"]
    values["delta_r2"] = values["r2_model"] - values["r2_ref"]
    return values


def concentration(parts, parent, arm, ref, metadata):
    records = []
    out = pd.DataFrame({
        "parent_key": parent.index,
        "sst": parent.sst.to_numpy(),
        "wells": parent.wells.to_numpy(),
        "sse_gain": (parent[f"sse_{ref}"] - parent[f"sse_{arm}"]).to_numpy(),
        "macro_rmse_gain": (parent[f"rmse_{ref}"] - parent[f"rmse_{arm}"]).to_numpy(),
    })
    out = out.sort_values(["sse_gain", "parent_key"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank_by_sse_gain", np.arange(1, len(out) + 1))
    out = out.assign(**metadata, arm=arm, reference=ref)
    total, pos = out.sse_gain.sum(), out.sse_gain.clip(lower=0).sum()
    rmse_total = out.macro_rmse_gain.sum()
    records.append({
        **metadata, "arm": arm, "reference": ref,
        "drop_top_k": 0, "dropped_parents": 0, "retained_parents": len(parent), "retained_wells": int(parent.wells.sum()),
        "rmse_model": parent[f"rmse_{arm}"].mean(), "rmse_ref": parent[f"rmse_{ref}"].mean(),
        "r2_model": 1 - parent[f"sse_{arm}"].sum() / parent.sst.sum(),
        "r2_ref": 1 - parent[f"sse_{ref}"].sum() / parent.sst.sum(),
        "rmse_delta": -rmse_total / len(parent), "delta_r2": total / parent.sst.sum(),
        "net_sse_gain": total, "positive_sse_gain": pos, "top_positive_sse_gain": 0,
    })
    for k in DELETE_K:
        top = out.loc[out.sse_gain.gt(0)].head(k)
        contribution = top.sse_gain.sum()
        rmse_gain = top.macro_rmse_gain.sum()
        records.append({
            **metadata, "arm": arm, "reference": ref, "drop_top_k": k, "dropped_parents": len(top),
            "net_sse_gain": total, "positive_sse_gain": pos,
            "top_positive_sse_gain": contribution,
            "top_share_of_positive_gain_pct": 100 * contribution / pos if pos > 0 else np.nan,
            "top_share_of_net_gain_pct": 100 * contribution / total if total > 0 else np.nan,
            "top_share_of_net_rmse_gain_pct": 100 * rmse_gain / rmse_total if rmse_total > 0 else np.nan,
            "top_parent_keys": "|".join(top.parent_key),
            **dropped_metrics(parts, set(top.parent_key), (arm, ref)),
        })
    return records, out


def evaluate_stratum(dataset, label, by_fold, prior):
    all_parts, metrics, concentrations, contributions = [], [], [], []
    scope_parts = {}
    for frame, truth, pred, repeats in by_fold:
        scopes = subsets(frame, prior)
        original = parent_table(frame, truth, pred)
        all_parts.append(original)
        for scope, keep in {**scopes, "repeat_supported": frame.repeat_supported.to_numpy()}.items():
            if not keep.any():
                continue
            if scope == "all":
                table = original
            else:
                p = {name: value[keep] for name, value in pred.items()}
                if scope == "repeat_supported":
                    p["repeat_x0.25"] = repeats[keep]
                table = parent_table(frame.loc[keep], truth[keep], p)
            scope_parts.setdefault(scope, []).append(table)
    for scope, blocks in scope_parts.items():
        parent = pd.concat(blocks, verify_integrity=True)
        arms = [n.removeprefix("rmse_") for n in parent if n.startswith("rmse_")]
        flags = {"dataset": dataset, "stratum": label, "scope": scope, "primary": label in ("rank7", "rank8") and scope == "all"}
        metrics.extend(paired_summaries(parent, arms, flags))
    all_parent = pd.concat(all_parts, verify_integrity=True)
    names = [n.removeprefix("rmse_") for n in all_parent if n.startswith("rmse_")]
    for arm in names:
        if arm == "B0":
            continue
        refs = ("B0",)
        if arm in MODEL_NAMES:
            refs += ("dose_shared_ridge",)
        if arm in MODEL_NAMES and arm != "p3_joint_chemistry":
            refs += ("p3_joint_chemistry",)
        for ref in refs:
            details, group = concentration(by_fold, all_parent, arm, ref, {"dataset": dataset, "stratum": label, "scope": "all"})
            concentrations.extend(details)
            contributions.append(group)
    return pd.DataFrame(metrics), pd.DataFrame(concentrations), pd.concat(contributions, ignore_index=True)
