"""Tune model, residual gain and simple anchor using only inner development folds."""

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP
from src.r11_chemical_probe.report import align
from src.r12_multiaxis.data import Fit, Source

from .objectives import GainScores


GAINS = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
FAMILIES = ("control", "mechanism", "kernels", "transfer", "neural")
POOL_STRENGTHS = [0, 2, 8, 32]


def implementation(family):
    return importlib.import_module(f"src.r13_optimization.{family}")


def specs_for(module, fit):
    specs = module.candidates()
    if len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Candidate names must be unique within a family")
    if hasattr(module, "available"):
        specs = [s for s in specs if module.available(fit, s)]
    if not specs:
        raise ValueError(f"No registered candidate is usable for {module.__name__}")
    return specs


def forecast(module, fit, specs):
    if hasattr(module, "predict_many"):
        yield from module.predict_many(fit, specs)
    else:
        for spec in specs:
            yield spec, module.predict(fit, spec)


def checked_forecast(module, fit, specs):
    wanted = {s["name"]: s for s in specs}
    seen = set()
    for spec, prediction in forecast(module, fit, specs):
        name = spec["name"]
        if name in seen or name not in wanted or spec != wanted[name]:
            raise ValueError("Forecast must emit requested specifications exactly once")
        seen.add(name)
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != fit.offset.shape or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid predictions from {module.__name__} candidate {name}")
        yield spec, prediction
    if seen != set(wanted):
        raise ValueError(f"Incomplete prediction flow from {module.__name__}")


def location(cfg, assets, fold, role, protocol):
    if protocol == "compound":
        return Source(cfg, assets, fold, role)
    return Source(cfg, assets, fold, role, split_dir="scaffold", stage_dir="staged-scaffold")


@dataclass(frozen=True)
class InternalScores:
    fold: int
    role: str
    inner: int
    fit: Fit
    scope: GainScores

    def evaluate(self, module):
        fit = self.fit
        is_control = module.__name__.endswith(".control")
        gains = np.array([1.0]) if is_control else GAINS
        specs = specs_for(module, fit)
        for spec, pred in checked_forecast(module, fit, specs):
            adjustment = pred - fit.offset
            ctr, absolute = self.scope.points(adjustment, gains)
            coverage, cosine = self.scope.diagnostics(adjustment)
            for gain, contrast, abs_rmse in zip(gains, ctr, absolute):
                yield dict(
                    fold=self.fold, role=self.role, inner=self.inner, name=spec["name"],
                    specification=json.dumps(spec, sort_keys=True, allow_nan=False),
                    gain=float(gain), contrast=float(contrast), absolute=float(abs_rmse),
                    contrast_weight=float(self.scope.metric.primary_weight.sum()),
                    absolute_weight=float(self.scope.metric.all_weight.sum()),
                    n_groups=int(fit.query.rows[GROUP].nunique()),
                    observed_alignment_coverage=coverage, observed_centered_cosine=cosine,
                )


def internal_data(source, inner):
    tr, val = source.inner(inner)
    fit = Fit(source, tr, val, observe_query=False)
    if fit.query.treated is not None or set(tr[GROUP]).intersection(val[GROUP]):
        raise ValueError("Unsafe inner fit/query construction")
    # All perturbation outcomes available here live in the outer-fit stage. The
    # query object passed to estimators contains rows, chemistry, and own controls.
    # Opposite-control subtraction cancels exactly under the within-pool contrast.
    query = source.fit_counts[align(fit.query.rows, source.fit_rows)][:, fit.panel].toarray()
    scores = GainScores(fit.query.rows, query - fit.query.control, fit.offset,
                        source.cfg["minimum_groups"])
    return InternalScores(source.outer, source.role, inner, fit, scores)


def select(rows):
    table = pd.DataFrame(rows)
    if table.duplicated(["inner", "name", "gain"]).any():
        raise ValueError("Duplicated inner fit result")
    n_folds = table.inner.nunique()
    ranked = []
    for pair, g in table.groupby(["name", "gain"], sort=True):
        if len(g) != n_folds or g.specification.nunique() != 1:
            continue
        ranked.append(dict(
            name=pair[0], gain=pair[1], specification=json.loads(g.specification.iloc[0]),
            contrast=float(np.average(g.contrast, weights=g.contrast_weight)),
            absolute=float(np.average(g.absolute, weights=g.absolute_weight)),
            contrast_by_fold=dict(zip(map(str, g.inner), g.contrast)),
            absolute_by_fold=dict(zip(map(str, g.inner), g.absolute)),
        ))
    if not ranked:
        raise ValueError("No candidates with all inner folds")
    min_ctr = min(r["contrast"] for r in ranked)
    best = min(
        (r for r in ranked if r["contrast"] <= min_ctr + 1e-8),
        key=lambda r: (r["gain"], r["absolute"], r["name"]),
    )
    challengers = [r for r in ranked if r["gain"] > 0 and not r["specification"].get("permute")]
    best_ctr = min(
        (r for r in challengers if r["contrast"] <= min(x["contrast"] for x in challengers) + 1e-8),
        key=lambda r: (r["gain"], r["absolute"], r["name"]),
    )
    best_abs = min(challengers, key=lambda r: (r["absolute"], r["gain"], r["name"]))
    diagnostic = min((r for r in ranked if r["specification"].get("permute") and r["gain"] > 0),
                     key=lambda r: r["contrast"], default=None)
    return dict(best=best, best_nonzero=best_ctr, best_absolute=best_abs, permutation_diagnostic=diagnostic,
                n_specs=int(table.name.nunique()), n_inner_metrics=len(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True, choices=range(6))
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--protocol", choices=["compound", "scaffold"], default="compound")
    args = parser.parse_args()
    fold, role = args.task // 2, "AB"[args.task % 2]
    cfg = config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    src = location(cfg, args.assets, fold, role, args.protocol)
    module = implementation(args.family)
    started = time.monotonic()
    results = []
    for inner in sorted(set(src.inner_labels.values())):
        data = internal_data(src, inner)
        rows = list(data.evaluate(module))
        results.extend(rows)
        print(f"{args.family} {args.protocol} {fold}/{role}/{inner}: {len(rows)} scored inner settings", flush=True)
        del data
    pd.DataFrame(results).to_csv(args.out / "grid.tsv", sep="\t", index=False)
    outcome = select(results)
    outcome.update(protocol=args.protocol, family=args.family, fold=fold, role=role,
                   config_sha256=sha256(args.config), partition_sha256=src.split["manifest_sha256"],
                   code={str(p): sha256(p) for p in [Path(__file__).relative_to(Path.cwd()),
                                                    Path(module.__file__).relative_to(Path.cwd()),
                                                    Path("src/r13_optimization/objectives.py")]},
                   grid_sha256=sha256(args.out / "grid.tsv"),
                   wall_seconds=time.monotonic() - started)
    write_json(args.out / "selection.json", outcome)
    print(json.dumps({key: outcome[key] for key in ["best", "best_nonzero", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
