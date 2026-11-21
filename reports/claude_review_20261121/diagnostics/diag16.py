"""diag16 (Claude review, EXPLORATORY, not official).

Question: R15 passed its pre-registered gate against known VCs fitted from scratch,
but R15's estimate had an inner-fold output gain while the known VCs had none, and
several known VCs show a higher direction cosine than R15 with a worse RMSE.
Does R15 still beat the known VCs when each of them is given an output gain from
the same grid?

Gain choice here: (1) cross-fold - chosen on the two other outer folds, applied to
the third (a single scalar per model; uses outer outcomes of other folds, so this
is a favourable-to-comparator approximation of an inner-fold choice);
(2) oracle - best gain on the scored rows themselves (upper bound for comparators).
Also explores simple two-model blends as possible next candidates.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.r15_lincs.design import Bank, fold_id

RUN = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r15_lincs/v1")
PROTOCOL = Path("reports/r15_lincs/PROTOCOL.md")
OUT = Path(sys.argv[1])
GAINS = (0., .125, .25, .5, .75, 1.)
VCS = ("ChemBERTa_R11", "FiLM_R12", "Concat_R12", "chemCPA_P0", "TranSiGen_Tahoe", "XPert_morgan",
       "PRnet-well-adaptation")
BLEND = ("FiLM_R12", "Concat_R12", "XPert_morgan")
NB = 10000


def outer_rows(bank, fold):
    rows = bank.target()
    assigned = rows.parent.map(lambda p: fold_id(p, "R15-outer", 3))
    outer = rows.loc[assigned.eq(fold)]
    extra = bank.auxiliary()
    extra = extra.loc[extra.parent.map(lambda p: fold_id(p, "R15-outer", 3)).eq(fold)]
    return pd.concat((outer, extra), ignore_index=True)


def load(path):
    with np.load(path/"forecast.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def units(parent, truth, pred):
    err = np.square((truth-pred).astype(np.float64))
    return pd.DataFrame({"parent": parent, "rows": 1, "rmse_sum": np.sqrt(err.mean(1)), "sse": err.sum(1)})


def collect(scenario):
    bank = Bank(RUN/"cache", PROTOCOL, scenario)
    store, base = {}, []
    for fold in range(3):
        for role in "AB":
            rows = outer_rows(bank, fold)
            root = RUN/"predict"/scenario/f"f{fold}-{role}"
            mine = load(root/"R15")
            listed = pd.read_csv(root/"R15"/"rows.tsv", sep="\t", keep_default_na=False)
            assert len(listed) == len(rows) and (listed["index"].to_numpy() == rows["index"].to_numpy()).all()
            mask = (rows.cell_id.eq("HEPG2") & rows.study.eq("GSE92742") & rows.hours.eq(6)
                    & rows.quality & ~rows.prior_exact).to_numpy()
            other = "B" if role == "A" else "A"
            truth = bank.response(rows, other)[mask]
            parent = rows.parent.to_numpy()[mask]
            anchor = (mine["B_pos_cell"] if scenario == "B" else mine["B_pos"])[mask]
            est = {"R15": mine["R15"][mask]-anchor}
            for name in VCS:
                est[name] = load(root/name)[name][mask]-anchor
            sst = np.square(truth.astype(np.float64))
            base.append(pd.DataFrame({"parent": parent, "fold": fold, "rows": 1,
                                      "b0_sum": np.sqrt(sst.mean(1)), "sst": sst.sum(1)}))

            def add(key, pred):
                store.setdefault(key, []).append(units(parent, truth, pred))

            add(("B0",), np.zeros_like(truth))
            add(("anchor",), anchor)
            add(("R15",), anchor+est["R15"])
            for g in GAINS:
                add(("R15x", g), anchor+g*est["R15"]/1.0)
                for name in VCS:
                    add((name, g), anchor+g*est[name])
                if scenario == "B":
                    add(("B_cell", g), g*mine["B_cell"][mask])
            for name in BLEND:
                for a in GAINS:
                    for b in GAINS:
                        add(("blend", name, a, b), anchor+a*est["R15"]+b*est[name])
            # direction of the estimate beyond the anchor, energy weighted
            resid = truth-anchor
            for name in ("R15", *VCS):
                num = float((resid*est[name]).sum())
                den = float(np.sqrt(np.square(resid).sum()*np.square(est[name]).sum()))
                store.setdefault(("cos", name), []).append((num, den, float(np.square(est[name]).sum()),
                                                            float(np.square(resid).sum())))
            print(f"loaded {scenario} f{fold}-{role}: {mask.sum()} primary rows", flush=True)
    base = pd.concat(base, ignore_index=True).groupby("parent").agg(
        fold=("fold", "first"), rows=("rows", "sum"), b0_sum=("b0_sum", "sum"), sst=("sst", "sum"))
    arms = {}
    for key, blocks in store.items():
        if key[0] == "cos":
            continue
        t = pd.concat(blocks, ignore_index=True).groupby("parent").sum().loc[base.index]
        arms[key] = np.column_stack((t.rmse_sum/t.rows, t.sse))
    cos = {k[1]: v for k, v in store.items() if k[0] == "cos"}
    return base, arms, cos


def objective(base, arm, idx):
    b0 = (base.b0_sum/base.rows).to_numpy()[idx].mean()
    return .5*arm[idx, 0].mean()/b0+.5*arm[idx, 1].sum()/base.sst.to_numpy()[idx].sum()


def crossfold(base, arms, keys):
    """Choose one key per outer fold on the other two folds; splice the per-parent units."""
    fold = base.fold.to_numpy()
    out = np.empty_like(arms[keys[0]])
    chosen = {}
    for f in range(3):
        others = np.flatnonzero(fold != f)
        best = min(keys, key=lambda k: (round(objective(base, arms[k], others), 9), str(k)))
        out[fold == f] = arms[best][fold == f]
        chosen[f] = best
    return out, chosen


def point(base, arm):
    return dict(rmse=float(arm[:, 0].mean()), r2=float(1-arm[:, 1].sum()/base.sst.sum()))


def main():
    report = {}
    for scenario in "AB":
        base, arms, cos = collect(scenario)
        n = len(base)
        sst = base.sst.to_numpy()
        final, picks = {"B0": arms[("B0",)], "anchor": arms[("anchor",)], "R15": arms[("R15",)]}, {}
        for name in VCS:
            final[f"{name}|raw"] = arms[(name, 1.)]
            final[f"{name}|cv"], picks[f"{name}|cv"] = crossfold(base, arms, [(name, g) for g in GAINS])
            best = min(GAINS, key=lambda g: objective(base, arms[(name, g)], np.arange(n)))
            final[f"{name}|oracle"], picks[f"{name}|oracle"] = arms[(name, best)], best
        final["R15|cv-regain"], picks["R15|cv-regain"] = crossfold(base, arms, [("R15x", g) for g in GAINS])
        if scenario == "B":
            final["B_cell|raw"] = arms[("B_cell", 1.)]
            final["B_cell|cv"], picks["B_cell|cv"] = crossfold(base, arms, [("B_cell", g) for g in GAINS])
        for name in BLEND:
            keys = [("blend", name, a, b) for a in GAINS for b in GAINS]
            final[f"blend-R15+{name}|cv"], picks[f"blend-R15+{name}|cv"] = crossfold(base, arms, keys)
        names = list(final)
        rng = np.random.default_rng(160116)
        rm = np.column_stack([final[a][:, 0] for a in names])
        se = np.column_stack([final[a][:, 1] for a in names])
        boots = np.empty((2, NB, len(names)))
        for at in range(0, NB, 128):
            end = min(at+128, NB)
            pick = rng.integers(n, size=(end-at, n))
            boots[0, at:end] = rm[pick].mean(1)
            boots[1, at:end] = 1-se[pick].sum(1)/sst[pick].sum(1)[:, None]
        col = {a: i for i, a in enumerate(names)}

        def paired(a, refs, label):
            rows = {}
            for i, metric in enumerate(("rmse", "r2")):
                ref = (np.minimum.reduce if i == 0 else np.maximum.reduce)([boots[i, :, col[r]] for r in refs])
                pts = [point(base, final[r])[metric] for r in refs]
                bestpt = min(pts) if i == 0 else max(pts)
                d = boots[i, :, col[a]]-ref
                lo, hi = np.quantile(d, (.005, .995))
                rows[metric] = dict(diff=point(base, final[a])[metric]-bestpt, low99=float(lo), high99=float(hi),
                                    passes=bool(hi < 0 if i == 0 else lo > 0),
                                    best_ref=refs[int(np.argmin(pts) if i == 0 else np.argmax(pts))])
            return {"arm": a, "vs": label, **rows}

        cv = [f"{v}|cv" for v in VCS]
        orc = [f"{v}|oracle" for v in VCS]
        raw = [f"{v}|raw" for v in VCS]
        tests = [paired("R15", ["B0"], "B0"), paired("R15", raw, "best raw VC"),
                 paired("R15", cv, "best cross-fold-gain VC"), paired("R15", orc, "best oracle-gain VC")]
        for v in ("FiLM_R12", "Concat_R12", "XPert_morgan", "ChemBERTa_R11"):
            tests.append(paired(f"{v}|cv", ["B0"], "B0"))
        for name in BLEND:
            b = f"blend-R15+{name}|cv"
            tests += [paired(b, ["R15"], "R15"), paired(b, cv, "best cross-fold-gain VC")]
        if scenario == "B":
            tests += [paired("R15", ["B_cell|cv"], "B_cell cross-fold gain"),
                      paired("FiLM_R12|cv", ["B_cell|cv"], "B_cell cross-fold gain")]
        report[scenario] = dict(
            parents=n, points={a: point(base, final[a]) for a in names},
            picks={k: (v if not isinstance(v, dict) else {str(f): list(map(str, x)) for f, x in v.items()})
                   for k, v in picks.items()},
            residual_cosine={k: dict(cosine=sum(x[0] for x in v)/np.sqrt(sum(x[2] for x in v)*sum(x[3] for x in v)),
                                     norm_ratio=float(np.sqrt(sum(x[2] for x in v)/sum(x[3] for x in v))))
                             for k, v in cos.items()},
            tests=tests)
        print(f"\n=== scenario {scenario}: {n} parents")
        for a in names:
            p = point(base, final[a])
            print(f"{a:34s} rmse={p['rmse']:.5f} r2={p['r2']:+.5f}  {picks.get(a, '')}")
        print("--- energy-weighted cosine of estimate with (truth - anchor), and norm ratio")
        for k, v in report[scenario]["residual_cosine"].items():
            print(f"{k:24s} cos={v['cosine']:+.4f} norm_ratio={v['norm_ratio']:.3f}")
        print("--- paired (arm - reference), 99% parent bootstrap")
        for t in tests:
            print(f"{t['arm']:30s} vs {t['vs']:28s} dRMSE={t['rmse']['diff']:+.5f} [{t['rmse']['low99']:+.5f},{t['rmse']['high99']:+.5f}] "
                  f"{'P' if t['rmse']['passes'] else '-'}  dR2={t['r2']['diff']:+.5f} [{t['r2']['low99']:+.5f},{t['r2']['high99']:+.5f}] "
                  f"{'P' if t['r2']['passes'] else '-'}  ref={t['rmse']['best_ref']}/{t['r2']['best_ref']}")
    OUT.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
