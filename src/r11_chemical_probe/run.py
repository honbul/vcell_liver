"""Run role-isolated OOF development or predict once from a frozen specification."""

import importlib.metadata
import os
import time

import numpy as np
import pandas as pd

from .common import arguments, config, read_json, sha256, write_json
from .dataset import GROUP, KEY, check_split, role_blocks, select_panel, table
from .diagnostics import context_recurrence, gate, permuted_drugs
from .metrics import Contrast
from .models import FullInput, Interaction, PoolMean, WeightedRidge


BASELINES = ("B0", "B1_line", "B1_pool", "B2_linear", "E", "B2_mlp")
ORDER = {arm: i for i, arm in enumerate(BASELINES + ("C", "M", "F"))}
ANCHOR = 2.0
SEEDS = (42, 43, 44)


def runtime():
    versions = {}
    for pkg in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "pyarrow"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return dict(packages=versions, slurm=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
                python=os.sys.version.split()[0], backend="CPU BLAS / deterministic Torch when used")


def record_item(records, vector, truth, metric, arm, value, duration, **details):
    primary, delta, _, _ = metric.points(vector, truth)
    records.append(dict(arm=arm, setting=value, contrast_rmse=primary, delta_rmse=delta,
                        seconds=round(duration, 3), **details))


def fold_means(grid, key):
    def weighted(frame):
        return pd.Series({
            name: np.average(frame[name], weights=frame["contrast_groups"])
            for name in ("contrast_rmse", "delta_rmse")
        })
    return grid.groupby(key, sort=False).apply(weighted, include_groups=False)


def choose(grid, arm):
    records = fold_means(grid[grid.arm.eq(arm)], "setting")
    best = records.contrast_rmse.min()
    near = records[records.contrast_rmse <= best + 1e-10]
    return str(near.delta_rmse.idxmin()) if arm == "B1_pool" else str(near.index[0])


def best_arm(grid, arms):
    means = fold_means(grid[grid.arm.isin(arms) & grid.selected], "arm").contrast_rmse
    best = means.min()
    return min((arm for arm, score in means.items() if score <= best + 1e-10), key=ORDER.__getitem__)


def designs(cfg, fit, query, arms):
    transforms = {
        a: Interaction(fit, a, cfg["features"]["drug_dim"], cfg["features"]["control_dim"])
        for a in ("E", "C") if a in arms or (a == "C" and set(arms) & {"M", "F"})
    }
    for arm, transform in transforms.items():
        if arm in arms:
            yield arm, transform, transform.transform(fit), transform.transform(query)
    if "B2_linear" in arms or "B2_mlp" in arms:
        full = FullInput(fit)
        yield "B2_linear", full, full.transform(fit), full.transform(query)


def offsets(fit, query):
    check_split(fit, query)
    means = PoolMean(fit.rows, fit.delta)
    fit_offset, fit_levels, _ = means.predict(fit.rows, ANCHOR, leave_group_out=True)
    query_offset, query_levels, _ = means.predict(query.rows, ANCHOR)
    fallback = dict(fit=pd.Series(fit_levels).value_counts().to_dict(),
                    query=pd.Series(query_levels).value_counts().to_dict())
    return means, query_offset, fit.delta - fit_offset, fallback


def fold_fit(cfg, full, fold, use_mlp):
    fit = full.take(full.rows.fold.ne(fold).to_numpy())
    query = full.take(full.rows.fold.eq(fold).to_numpy())
    means, query_offset, residual, fallback = offsets(fit, query)
    truth = query.delta
    metric = Contrast(query.rows, cfg["evaluation"]["minimum_groups"])
    records = []
    predictions = {}
    n = len(truth)
    record_item(records, np.zeros_like(truth), truth, metric, "B0", "fixed", 0)
    record_item(records, means.predict(query.rows, float("inf"))[0], truth, metric, "B1_line", "fixed", 0)
    for kappa in cfg["model"]["shrinkage"]:
        value = means.predict(query.rows, float(kappa))[0]
        record_item(records, value, truth, metric, "B1_pool", str(kappa), 0)
    shuffled, pen = permuted_drugs(query, cfg["evaluation"]["seed"] + int(fold))
    for arm, transform, x, z in designs(cfg, fit, query, ("E", "C", "B2_linear")):
        start = time.monotonic()
        solver = WeightedRidge(x, residual, fit.rows)
        setup = time.monotonic() - start
        params = dict(design_cols=x.shape[1])
        if arm != "B2_linear":
            params.update(drug_rank=len(transform.drug[1].keep), control_rank=len(transform.control[1].keep))
        for alpha in cfg["model"]["alpha"]:
            start = time.monotonic()
            coef = solver.coefficients(alpha)
            prediction = query_offset + solver.predict(z, coef)
            record_item(records, prediction, truth, metric, arm, str(alpha), time.monotonic() - start,
                        fit_setup_seconds=round(setup, 3), **params)
            if arm in ("C", "E"):
                predictions[(arm, str(alpha))] = prediction
            if arm == "C":
                perm = query_offset + solver.predict(transform.transform(query, shuffled), coef)
                record_item(records, perm, truth, metric, "C_permuted", str(alpha), 0, **pen)
        if arm == "B2_linear" and use_mlp:
            nn_grid("B2_mlp", (x,), (z,), fit, residual, truth, query_offset, metric, cfg, records)
    for item in records:
        item.update(fold=int(fold), n_fit=len(fit.rows), n_query=n,
                    n_contrast=len(metric.active), contrast_groups=query.rows.iloc[metric.active][GROUP].nunique(),
                    query_groups=query.rows[GROUP].nunique())
    return records, dict(rows=query.rows, predictions=predictions, truth=truth, fallback=fallback)


def nn_grid(arm, x, z, fit, residual, truth, query_offset, metric, cfg, records):
    from .network import train_nn

    for lr in cfg["model"]["mlp_lr"]:
        start = time.monotonic()
        pred, _, metadata = train_nn(
            arm, x, residual, fit.rows, z, lr, cfg["model"], SEEDS[0],
            score=lambda p: metric.point(query_offset + p, truth)
        )
        record_item(records, query_offset + pred, truth, metric, arm, str(lr), time.monotonic() - start, **metadata)


def film_fold(cfg, full, fold):
    fit = full.take(full.rows.fold.ne(fold).to_numpy())
    query = full.take(full.rows.fold.eq(fold).to_numpy())
    _, anchor, residual, _ = offsets(fit, query)
    transform = Interaction(fit, "C", cfg["features"]["drug_dim"], cfg["features"]["control_dim"])
    x, z = transform.blocks(fit), transform.blocks(query)
    metric = Contrast(query.rows, cfg["evaluation"]["minimum_groups"])
    records = []
    for arm in ("M", "F"):
        nn_grid(arm, x, z, fit, residual, query.delta, anchor, metric, cfg, records)
    for item in records:
        item["fold"] = int(fold)
        item["contrast_groups"] = query.rows.iloc[metric.active][GROUP].nunique()
    params = {record["arm"]: record["params"] for record in records}
    if abs(params["M"] / params["F"] - 1) > 0.10:
        raise ValueError("FiLM and concat parameter counts differ by over 10%")
    return records


def calculate_panels(cfg, assets, stage):
    root = assets / ("pilot_panels" if stage == "pilot" else "development")
    root.mkdir(parents=True, exist_ok=False)
    stage_rows = table(assets / stage / "treated_rows.tsv")
    folds = sorted(stage_rows.fold.unique())
    if len(folds) != (2 if stage == "pilot" else 3) or min(folds) < 0:
        raise ValueError(f"Unexpected source folds: {folds}")
    for fold in folds:
        select_panel(assets, stage, fold, cfg["data"]["panel_size"]).to_csv(root / f"fold{fold}_panel.tsv", sep="\t", index=False)
    select_panel(assets, stage, None, cfg["data"]["panel_size"]).to_csv(root / "final_panel.tsv", sep="\t", index=False)
    write_json(root / "panels.json", {file.name: sha256(file) for file in sorted(root.glob("*_panel.tsv"))})


def development(cfg, assets, stage, role):
    start = time.monotonic()
    root, raw = (assets / "pilot_panels", "pilot") if stage == "pilot" else (assets / "development", "source")
    panels = read_json(root / "panels.json")
    folds = sorted(int(name[4]) for name in panels if name.startswith("fold"))
    out = root / role
    out.mkdir(exist_ok=False)
    grid_parts = []
    hold = {}

    def blocks(fold):
        name = f"fold{fold}_panel.tsv"
        if sha256(root / name) != panels[name]:
            raise ValueError("Fold gene panel hash changed")
        return role_blocks(assets, raw, role, table(root / name).token_id.to_numpy())

    for fold in folds:
        rows, hold[str(fold)] = fold_fit(cfg, blocks(fold), fold, stage != "pilot")
        grid_parts.extend(rows)
        print(f"{stage}/{role}: fold {fold} complete", flush=True)
    grid = pd.DataFrame(grid_parts)
    selection = {arm: choose(grid, arm) for arm in grid.arm.unique() if arm != "C_permuted"}
    selection["C_permuted"] = selection["C"]
    grid["selected"] = [setting == selection[arm] for arm, setting in zip(grid.arm, grid.setting)]
    baseline = best_arm(grid, BASELINES)
    context = {}
    for num, item in hold.items():
        pred = item["predictions"][("C", selection["C"])]
        context[num] = context_recurrence(item["rows"], item["truth"] - pred, cfg["data"]["liver"],
                                          cfg["evaluation"]["seed"] + int(num))
        item["rows"][KEY].to_csv(out / f"fold{num}_predicted_rows.tsv", sep="\t", index=False)
        np.savez_compressed(out / f"fold{num}_selected.npz", C=pred, observed=item["truth"],
                            E=item["predictions"][("E", selection["E"])])
    admission = gate(dict(baseline=baseline), grid, context, folds)
    admission["enter"] = admission["enter"] and stage != "pilot"
    structure_baseline = None
    if admission["enter"]:
        extra = pd.DataFrame([r for fold in folds for r in film_fold(cfg, blocks(fold), fold)])
        for arm in ("M", "F"):
            selection[arm] = choose(extra, arm)
        extra["selected"] = [setting == selection[arm] for arm, setting in zip(extra.arm, extra.setting)]
        grid = pd.concat([grid, extra], ignore_index=True)
        structure_baseline = best_arm(grid, (baseline, "C", "E", "M"))
    epochs = {}
    for arm in ("B2_mlp", "M", "F"):
        use = grid[grid.arm.eq(arm) & grid.selected]
        if not use.empty:
            epochs[arm] = int(use.epoch.median())
    grid.to_csv(out / "grid.tsv", sep="\t", index=False)
    write_json(out / "selection.json", dict(
        role=role, chosen=selection, epochs=epochs, baseline=baseline, structure_baseline=structure_baseline,
        e3=admission, context=context, stage=stage, seconds=round(time.monotonic()-start, 2),
        offset_fallback={num: item["fallback"] for num, item in hold.items()}, runtime=runtime(),
        grid_sha256=sha256(out / "grid.tsv")
    ))
    print(fold_means(grid[grid.selected], "arm").to_string(), flush=True)


def final_predict(cfg, assets, role):
    from .freeze import verify

    frozen = verify(assets, cfg)
    plan = frozen["roles"][role]
    root = assets / "development"
    panel = table(root / "final_panel.tsv").token_id.to_numpy()
    fit, query = role_blocks(assets, "source", role, panel, final=True)
    start = time.monotonic()
    means, base, residual, fallback = offsets(fit, query)
    out = assets / "predictions" / role
    out.mkdir(parents=True, exist_ok=False)
    n_rows, n_genes = query.control.shape
    arms = ["B0", "B1_line", "B1_pool", "B2_linear", "B2_mlp", "E", "C"]
    if plan["e3"]["enter"]:
        arms += ["M", "F"]
    chosen = plan["chosen"]
    pred = dict(B0=np.zeros((1, n_rows, n_genes), dtype=np.float32),
                B1_line=means.predict(query.rows, float("inf"))[0][None],
                B1_pool=means.predict(query.rows, float(chosen["B1_pool"]))[0][None])
    timing, nn_metadata = {}, {}
    for arm, _, x, z in designs(cfg, fit, query, arms):
        tick = time.monotonic()
        solver = WeightedRidge(x, residual, fit.rows)
        coef = solver.coefficients(float(chosen[arm]))
        pred[arm] = (base + solver.predict(z, coef))[None]
        np.savez_compressed(out / f"{arm}_model.npz", coefficients=coef, x_mean=solver.xmean, y_mean=solver.ymean)
        timing[arm] = time.monotonic() - tick
        if arm == "B2_linear":
            inputs = {"B2_mlp": ((x,), (z,))}
    if {"M", "F"} <= set(arms):
        transform = Interaction(fit, "C", cfg["features"]["drug_dim"], cfg["features"]["control_dim"])
        state = (transform.blocks(fit), transform.blocks(query))
        inputs.update(M=state, F=state)
    from .network import torch, train_nn

    for arm, (x, z) in inputs.items():
        tick = time.monotonic()
        pred[arm], nn_metadata[arm] = [], []
        for seed in SEEDS:
            value, state, meta = train_nn(arm, x, residual, fit.rows, z, float(chosen[arm]),
                                          cfg["model"], seed, epochs=plan["epochs"][arm])
            pred[arm].append(base + value)
            nn_metadata[arm].append(meta)
            torch.save(state, out / f"{arm}_seed{seed}.pt")
        pred[arm] = np.stack(pred[arm])
        timing[arm] = time.monotonic() - tick
    if set(pred) != set(arms) or any(not np.isfinite(x).all() for x in pred.values()):
        raise ValueError("Incomplete or non-finite frozen predictions")
    query.rows.to_csv(out / "rows.tsv", sep="\t", index=False)
    np.savez_compressed(out / "vectors.npz", **pred)
    write_json(out / "receipt.json", dict(role=role, freeze_sha256=sha256(assets / "freeze.json"), arms=arms,
               n_fit=len(fit.rows), n_query=n_rows, fallback=fallback, timing=timing, network=nn_metadata,
               runtime=runtime(), seconds=round(time.monotonic()-start, 2),
               files={p.name: sha256(p) for p in sorted(out.iterdir())}))


def main():
    parser = arguments(__doc__)
    parser.add_argument("--stage", choices=["pilot-panels", "panels", "pilot", "development", "final"], required=True)
    parser.add_argument("--role", choices=["A", "B"])
    args = parser.parse_args()
    cfg = config(args.config)
    if args.stage in ("panels", "pilot-panels"):
        calculate_panels(cfg, args.assets, "pilot" if args.stage == "pilot-panels" else "source")
    elif args.role and args.stage == "final":
        final_predict(cfg, args.assets, args.role)
    elif args.role:
        development(cfg, args.assets, args.stage, args.role)
    else:
        parser.error("--role required for model stages")


if __name__ == "__main__":
    main()
