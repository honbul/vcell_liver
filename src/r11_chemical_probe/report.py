"""Score frozen role predictions once and store reproducible paired summaries."""

import numpy as np
import pandas as pd

from src.r6_followup_v1.metrics import cond_rmse

from .common import arguments, config, read_json, sha256, write_json
from .dataset import GROUP, KEY, LINE, POOL, load_data, table, group_weights
from .freeze import verify
from .metrics import Contrast, correlations


def macro(rows, values, mask=None, draws=None, inverse=None):
    if mask is None:
        mask = np.ones(len(rows), dtype=bool)
    active = np.flatnonzero(mask)
    if not len(active):
        return None
    weight = group_weights(rows.iloc[active])
    if draws is not None:
        weight = weight * draws[:, inverse[active]]
    return (values[..., active] * weight).sum(axis=-1) / weight.sum(axis=-1)


def point_summary(meta, values, scope, mask):
    base = dict(scope=scope, n_rows=int(mask.sum()), n_groups=meta.loc[mask, GROUP].nunique(),
                n_wells=meta.loc[mask, KEY[0]].nunique())
    for name, vals in values.items():
        available = mask & np.isfinite(vals)
        base[name] = macro(meta, vals, available)
    base["n_contrast_rows"] = int((mask & np.isfinite(values["contrast_rmse"])).sum())
    base["n_contrast_groups"] = meta.loc[mask & np.isfinite(values["contrast_rmse"]), GROUP].nunique()
    return base


def align(left, right, right_key=KEY):
    index = pd.MultiIndex.from_frame(right[right_key])
    align_to = pd.MultiIndex.from_frame(left[right_key])
    if not index.is_unique or not align_to.is_unique:
        raise ValueError(f"Non-unique scoring join: {right_key}")
    result = index.get_indexer(align_to)
    if (result < 0).any():
        raise ValueError("Missing score join key")
    return result


def final_score(cfg, assets, out):
    spec = verify(assets, cfg)
    score_root = assets / "predictions"
    score_specs, row_tables = {}, {}
    for role in ("A", "B"):
        folder = score_root / role
        receipt = read_json(folder / "receipt.json")
        if receipt["freeze_sha256"] != sha256(assets / "freeze.json"):
            raise ValueError("Prediction used another freeze")
        for file in ("rows.tsv", "vectors.npz"):
            if sha256(folder / file) != receipt["files"][file]:
                raise ValueError(f"{file} changed since inference")
        score_specs[role] = dict(folder=folder, receipt=receipt)
        row_tables[role] = table(folder / "rows.tsv")
    rows = row_tables["A"].merge(row_tables["B"][KEY], on=KEY, validate="one_to_one",
                                 how="inner", sort=False).reset_index(drop=True)
    eligible_rows, measured = load_data(assets / "final", "treated")
    selected = align(rows, eligible_rows)
    absolute = measured[selected].toarray()
    panel = table(assets / "development" / "final_panel.tsv").token_id.to_numpy()
    metric = Contrast(rows, cfg["evaluation"]["minimum_groups"])
    is_primary = np.isin(np.arange(len(rows)), metric.active)
    rng = np.random.default_rng(cfg["evaluation"]["seed"])
    draws = rng.exponential(size=(cfg["evaluation"]["bootstrap_replicates"], len(metric.group_ids)))
    draws /= draws.mean(axis=1, keepdims=True)
    contrasts = [("C-B*", {r: ("C", spec["roles"][r]["baseline"]) for r in ("A", "B")}, True)]
    contrasts += [("C-E", {r: ("C", "E") for r in ("A", "B")}, False)]
    structure = spec["E3_roles"]
    if structure:
        contrasts += [("F-B†", {r: ("F", spec["roles"][r]["structure_baseline"]) for r in structure}, len(structure) == 2),
                      ("F-M", {r: ("F", "M") for r in structure}, False)]
    boot, values, details = {}, {}, []
    sc_meta = table(assets / "features" / "test_similarity.tsv")
    eligible_sc = (~sc_meta.scaffold_seen & sc_meta.scaffold.ne("<acyclic>")).groupby(sc_meta[GROUP]).all()
    scaffold_mask = rows[GROUP].isin(eligible_sc[eligible_sc].index).to_numpy()
    scaffold_rows = rows[scaffold_mask].reset_index(drop=True)
    scaffold_metric = None
    if scaffold_rows.groupby(POOL)[GROUP].nunique().ge(cfg["evaluation"]["minimum_groups"]).any():
        scaffold_metric = Contrast(scaffold_rows, cfg["evaluation"]["minimum_groups"])
    scaffold_points = {}
    groups = {"all": np.ones(len(rows), dtype=bool)}
    groups.update({name: rows[LINE].eq(cid).to_numpy() for cid, name in cfg["data"]["liver"].items()})
    plates = sorted(rows.plate_id.unique())
    groups.update({plate: rows.plate_id.eq(plate).to_numpy() for plate in plates})
    groups.update({f"omit_{plate}": rows.plate_id.ne(plate).to_numpy() for plate in plates})
    out.mkdir(parents=True, exist_ok=False)
    for role in ("A", "B"):
        position = align(rows, row_tables[role])
        joined = row_tables[role].iloc[position]
        if not np.array_equal(rows[GROUP], joined[GROUP]) or not np.array_equal(rows.plate_id, joined.plate_id):
            raise ValueError("Role row IDs refer to mismatched compounds or plates")
        opposite = "A" if role == "B" else "B"
        ctrl_rows, ctrl = load_data(assets / "source", role, panel)
        score_rows, score_ctrl = load_data(assets / "source", opposite, panel)
        ctrl_key = ["plate_id", LINE]
        input_ref = align_pool(rows, ctrl_rows, ctrl_key)
        score_ref = align_pool(rows, score_rows, ctrl_key)
        if not np.array_equal(ctrl_rows.iloc[input_ref][KEY[0]], joined.control_well_id):
            raise ValueError("Saved inference controls differ from indicated role")
        if (ctrl_rows.iloc[input_ref][KEY[0]].to_numpy() == score_rows.iloc[score_ref][KEY[0]].to_numpy()).any():
            raise ValueError("Scoring and feature control wells overlap")
        observed = absolute - score_ctrl[score_ref].toarray()
        ctrl_vectors = ctrl[input_ref].toarray()
        boot_arms = {a for _, pairs, _ in contrasts for a in pairs.get(role, ())}
        observed_norm = cond_rmse(np.zeros_like(observed), observed)
        observed_contrast = metric.points(np.zeros_like(observed), observed)[2]
        with np.load(score_specs[role]["folder"] / "vectors.npz") as file:
            for arm in file.files:
                seed_vectors = file[arm][:, position]
                point, boot_values, sc_points = [], [], []
                for idx, prediction in enumerate(seed_vectors):
                    _, _, contrast, delta = metric.points(prediction, observed)
                    rank, top, candidates, _ = metric.retrieval(prediction, observed)
                    pearson, pearson_why = correlations(prediction, observed, pearson=True)
                    cosine, cosine_why = correlations(prediction, observed)
                    seed_point = dict(contrast_rmse=contrast, delta_rmse=delta, pds=rank,
                                      retrieval_top1=top, retrieval_candidates=candidates.astype(float),
                                      absolute_rmse=cond_rmse(ctrl_vectors + prediction, absolute),
                                      observed_delta_rms=observed_norm,
                                      predicted_delta_rms=cond_rmse(prediction, np.zeros_like(prediction)),
                                      observed_contrast_rms=observed_contrast,
                                      predicted_contrast_rms=metric.points(prediction, np.zeros_like(prediction))[2],
                                      pearson=pearson, cosine=cosine)
                    if scaffold_metric is not None:
                        sc_points.append(scaffold_metric.points(prediction[scaffold_mask], observed[scaffold_mask])[:2])
                    seed_label = spec["seeds"][idx] if len(seed_vectors) > 1 else "deterministic"
                    details.append(pd.DataFrame({**seed_point, "role": role, "arm": arm, "seed": seed_label,
                                                 "pearson_reason": pearson_why, "cosine_reason": cosine_why,
                                                 "row_index": np.arange(len(rows))}))
                    point.append(seed_point)
                    if arm in boot_arms:
                        boot_values.append(metric.bootstrap_values(prediction, observed, draws))
                values[role, arm] = {key: nanmean([p[key] for p in point]) for key in point[0]}
                if boot_values:
                    boot[role, arm] = np.mean(boot_values, axis=0)
                if sc_points:
                    scaffold_points[role, arm] = np.mean(sc_points, axis=0)
        np.savez_compressed(out / f"truth_{role}.npz", delta=observed,
                            input_control=ctrl_vectors, measured_absolute=absolute)

    summaries = []
    for (role, arm), record in values.items():
        summaries += [dict(role=role, arm=arm, **point_summary(rows, record, scope, mask))
                      for scope, mask in groups.items()]
    for arm in sorted(set(k[1] for k in values if k[0] == "A") & set(k[1] for k in values if k[0] == "B")):
        combined = {name: nanmean([vals[name] for vals in (values["A", arm], values["B", arm])])
                    for name in values["A", arm]}
        summaries += [dict(role="A+B", arm=arm, **point_summary(rows, combined, scope, mask))
                      for scope, mask in groups.items()]
    pd.DataFrame(summaries).to_csv(out / "arm_metrics.tsv", sep="\t", index=False, na_rep="NA")
    paired, boot_store = [], {}
    influence = []
    for label, pairs, formal in contrasts:
        level = spec["official_ci"] if formal else spec["attribution_ci"]
        delta_pt = np.mean([values[r, a]["contrast_rmse"] - values[r, b]["contrast_rmse"] for r, (a, b) in pairs.items()], axis=0)
        delta_bs = np.mean([boot[r, a] - boot[r, b] for r, (a, b) in pairs.items()], axis=0)
        comparator = np.mean([values[r, b]["contrast_rmse"] for r, (_, b) in pairs.items()], axis=0)
        for scope, mask in groups.items():
            core = mask & is_primary
            boot_mean = macro(rows, delta_bs, core, draws, metric.inverse)
            if boot_mean is None:
                continue
            point = macro(rows, delta_pt, core)
            other = macro(rows, comparator, core)
            lo, hi = np.quantile(boot_mean, [(1-level)/2, (1+level)/2])
            boot_store[f"{label}|{scope}"] = boot_mean
            paired.append(dict(contrast=label, scope=scope, roles="+".join(pairs), arm_map=str(pairs),
                               formal=bool(formal and scope == "all"), n_groups=rows.loc[core, GROUP].nunique(),
                               n_rows=int(core.sum()), difference=point, relative=point/other,
                               ci_level=level, low=lo, high=hi, numeric_failures=int((~np.isfinite(boot_mean)).sum())))
        for group in metric.group_ids:
            row_mask = rows[GROUP].eq(group).to_numpy()
            point = macro(rows, delta_pt, is_primary & row_mask)
            if point is not None:
                influence.append(dict(contrast=label, compound_group_id=group, difference=point))
    pd.DataFrame(paired).to_csv(out / "paired.tsv", sep="\t", index=False)
    if scaffold_metric is not None:
        n_sc = scaffold_rows.iloc[scaffold_metric.active][GROUP].nunique()
        stress = [dict(role=role, arm=arm, contrast_rmse=point[0], delta_rmse=point[1],
                       n_groups=n_sc, n_rows=len(scaffold_metric.active), n_pools=len(scaffold_metric.pools))
                  for (role, arm), point in scaffold_points.items()]
        pd.DataFrame(stress).to_csv(out / "unseen_scaffold_stress.tsv", sep="\t", index=False)
    pd.DataFrame(influence).to_csv(out / "group_effect.tsv", sep="\t", index=False)
    np.savez_compressed(out / "bootstrap.npz", weights=draws, group_ids=metric.group_ids, **boot_store)
    rows["eligible_contrast"] = is_primary
    rows.to_csv(out / "scoring_rows.tsv", sep="\t", index=False)
    pd.concat(details).to_csv(out / "per_prediction_metrics.tsv.gz", sep="\t", index=False, na_rep="NA",
                             compression={"method": "gzip", "mtime": 0})
    write_json(out / "receipt.json", dict(
        freeze_sha256=sha256(assets / "freeze.json"),
        predictions={role: sha256(spec["folder"] / "receipt.json") for role, spec in score_specs.items()},
        scoring_rows=len(rows), candidate_rows=len(eligible_rows),
        role_rows={r: len(w) for r, w in row_tables.items()},
        eligible_pools=len(metric.pools), rows_per_pool=[len(idx) for idx in metric.pools],
        ci_scope="observed plates, controls, fitted models; paired positive Exp(1) group-bootstrap percentile approximation",
        files={p.name: sha256(p) for p in sorted(out.iterdir())}
    ))
    print(pd.DataFrame(paired).query("scope == 'all'").to_string(index=False), flush=True)


def align_pool(rows, reference, keys):
    return align(rows[keys].drop_duplicates(), reference, keys)[
        pd.MultiIndex.from_frame(rows[keys].drop_duplicates()).get_indexer(pd.MultiIndex.from_frame(rows[keys]))
    ]


def nanmean(array):
    values = np.stack(array)
    count = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def main():
    parser = arguments(__doc__)
    args = parser.parse_args()
    final_score(config(args.config), args.assets, args.assets / "metrics")


if __name__ == "__main__":
    main()
