"""Select a portable A549 candidate from alias-disjoint Tahoe folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, LINE, table
from src.r11_chemical_probe.report import align
from src.r12_multiaxis.data import Fit, Source, select_panel
from src.r12_multiaxis.split import eligible_pools
from src.r12_multiaxis.stage import load_staged

from .external import DATASET, allowed_training_rows, read_manifest, verify_dose_rows
from .objectives import GainScores
from .tune import GAINS, checked_forecast, implementation, specs_for


FAMILIES = ("kernels", "transfer", "neural", "dose_transfer")
LINE_ID = "CVCL_0023"


def portable(spec):
    return spec.get("chemistry") not in {"chemberta", "cls", "mixed"} and spec.get("chemical") not in {
        "chembert", "dual",
    }


def evaluate(args):
    read_manifest(args.manifest)
    cfg = config(args.config)
    allowed = set(allowed_training_rows(args.assets, args.manifest)[GROUP])
    fold, role = divmod(args.task, 2)
    role = "AB"[role]
    source = Source(cfg, args.assets, fold, role)
    tr = source.fit_rows.loc[source.fit_rows[GROUP].isin(allowed)].reset_index(drop=True)
    query = source.query_rows.loc[source.query_rows[LINE].eq(LINE_ID) & source.query_rows[GROUP].isin(allowed)]
    query = eligible_pools(query, cfg["minimum_groups"])
    if set(tr[GROUP]) & set(query[GROUP]) or query.empty:
        raise ValueError("Alias-clean Tahoe A549 fold lacks disjoint contrast pools")
    if args.family == "dose_transfer":
        verify_dose_rows(args.assets, args.manifest, set(tr[GROUP]))
    source.fit_counts = source.fit_counts[align(tr, source.fit_rows)]
    source.fit_rows, source.query_rows = tr, query
    genes = table(args.manifest / "genes.tsv").sort_values("token_id").token_id.to_numpy()
    counts = source.fit_counts[:, genes]
    padded = sparse.hstack((sparse.csr_matrix((len(tr), 3)), counts), format="csr")
    panel = genes[select_panel(padded, cfg["panel_size"]) - 3]
    fit = Fit(source, tr, query, panel=panel, observe_query=False)
    if fit.query.treated is not None:
        raise ValueError("Tahoe held-out subset was attached to the forecast")
    stage = args.assets / "staged"
    staged = read_json(stage / "receipt.json")
    measured_rows, values = load_staged(stage / f"outer{fold}", "score", staged["genes"])
    measured = values[align(fit.query.rows, measured_rows)][:, panel].toarray()
    scores = GainScores(fit.query.rows, measured - fit.query.control, fit.offset, cfg["minimum_groups"])
    module = implementation(args.family)
    specs = [spec for spec in specs_for(module, fit) if portable(spec)]
    records = []
    for spec, prediction in checked_forecast(module, fit, specs):
        contrast, absolute = scores.points(prediction - fit.offset, GAINS)
        records.extend(dict(
            family=args.family, task=args.task, role=role, fold=fold, name=spec["name"], gain=gain,
            contrast=ctr, absolute=abs_rmse, b0=contrast[0],
            weight=scores.metric.primary_weight.sum(), specification=json.dumps(spec, sort_keys=True),
        ) for gain, ctr, abs_rmse in zip(GAINS, contrast, absolute))
    args.out.mkdir(parents=True, exist_ok=False)
    grid = args.out / "grid.tsv"
    pd.DataFrame(records).to_csv(grid, index=False, sep="\t")
    write_json(args.out / "receipt.json", dict(
        task=args.task, family=args.family, family_sha256=sha256(module.__file__),
        selector_sha256=sha256(__file__), manifest_sha256=sha256(args.manifest / "receipt.json"),
        partition_sha256=source.split["manifest_sha256"], grid_sha256=sha256(grid),
        source_groups=int(tr[GROUP].nunique()), development_groups=int(query[GROUP].nunique()),
        wells=len(query), n_specs=len(specs), treated_external_outcomes_accessed=False,
    ))


def lock(args):
    manifest = sha256(args.manifest / "receipt.json")
    evidence = {}
    records = []
    for family in FAMILIES:
        for task in range(6):
            source = args.runs / family / f"task{task}"
            grid, receipt = source / "grid.tsv", source / "receipt.json"
            rec = read_json(receipt)
            if (rec["task"], rec["family"], rec["manifest_sha256"], rec["grid_sha256"]) != (
                task, family, manifest, sha256(grid),
            ) or rec["selector_sha256"] != sha256(__file__):
                raise ValueError(f"Alias-clean validation artifact differs: {source}")
            evidence[str(receipt)] = sha256(receipt)
            evidence[str(grid)] = sha256(grid)
            records.append(table(grid))
    combined = pd.concat(records, ignore_index=True)
    source = combined.loc[combined.gain.gt(0)]
    ranked = []
    for (family, name, gain), group in source.groupby(["family", "name", "gain"]):
        if set(group.task) != set(range(6)) or len(group) != 6 or group.specification.nunique() != 1:
            raise ValueError("Development candidate did not finish all six matched folds")
        diff = group.contrast - group.b0
        by_fold = np.array([np.average(diff.loc[ix], weights=group.weight.loc[ix])
                            for ix in group.groupby("fold").groups.values()])
        ranked.append(dict(family=family, name=name, gain=gain, specification=json.loads(group.specification.iloc[0]),
                           effect=float(np.average(diff, weights=group.weight)),
                           improving_folds=int((by_fold < -1e-8).sum())))
    def rank(entry):
        return -entry["improving_folds"], entry["effect"], entry["gain"], entry["name"]

    winners = {family: min((v for v in ranked if v["family"] == family), key=rank) for family in FAMILIES}
    primary = min(winners, key=lambda family: rank(winners[family]))
    rule = ("Exclude every GSE306429-linked group before any Tahoe fit or validation. Select each family's "
            "positive residual gain by number of improving A549 outer folds averaged across A/B, followed "
            "by weighted contrast improvement versus B0, smaller gain, and candidate name. The best family "
            "under the same rule is the fixed external challenger. The other family scores are exploratory.")
    if args.recipe.exists():
        raise FileExistsError(args.recipe)
    write_json(args.recipe, dict(
        dataset=DATASET, line="A549", dose_uM=1.0, time_h=24.0, manifest_sha256=manifest,
        selection_rule=rule, tahoe_receipts=evidence, primary=primary, arms=winners,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("evaluate")
    dev.add_argument("--config", type=Path, required=True)
    dev.add_argument("--assets", type=Path, required=True)
    dev.add_argument("--manifest", type=Path, required=True)
    dev.add_argument("--task", type=int, required=True, choices=range(6))
    dev.add_argument("--family", choices=FAMILIES, required=True)
    dev.add_argument("--out", type=Path, required=True)
    seal = commands.add_parser("lock")
    seal.add_argument("--manifest", type=Path, required=True)
    seal.add_argument("--runs", type=Path, required=True)
    seal.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    (evaluate if args.command == "evaluate" else lock)(args)


if __name__ == "__main__":
    main()
