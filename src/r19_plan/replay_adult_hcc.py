"""Reproduce matched adult HCC scores from compact parent-level statistics."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ARMS = ("calibrated_shrink", "eight_token_State", "XPert_morgan", "PRnet-well-adaptation", "chemCPA_P0",
        "TranSiGen_Tahoe")
DRAWS = 10_000
SEED = 150104
NUMERIC = ("arm_value", "ref_value", "difference", "low99", "high99", "relative_gain", "relative_low99", "relative_high99")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def tabular(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def replay(path):
    rows = tabular(path)
    if [r["ordinal"] for r in rows] != [str(i+1) for i in range(len(rows))]:
        raise ValueError("The frozen bootstrap parent order changed")
    wells = np.asarray([int(r["wells"]) for r in rows])
    if np.any(wells < 1):
        raise ValueError("Each parent requires at least one physical treatment well")
    rms, sse = (np.asarray([[float(r[f"{arm}_{metric}"]) for arm in ARMS] for r in rows])
                for metric in ("rmse", "sse"))
    energy = np.asarray([float(r["B0_sse"]) for r in rows])
    if (not all(np.isfinite(x).all() and np.all(x>=0) for x in (rms, sse, energy)) or np.any(energy<=0)):
        raise ValueError("Invalid absolute or squared error moments")

    rng = np.random.default_rng(SEED)
    chosen = rng.integers(len(rows), size=(DRAWS,len(rows)))
    sampled_rmse = rms[chosen].mean(axis=1)
    sampled_r2 = 1-sse[chosen].sum(axis=1)/energy[chosen].sum(axis=1,keepdims=True)
    estimates, frequency = [], []
    for name, point, sampled, choose in (
        ("rmse",rms.mean(axis=0),sampled_rmse,np.argmin),
        ("r2_pooled",1-sse.sum(axis=0)/energy.sum(),sampled_r2,np.argmax),
    ):
        best_point = 1+choose(point[1:])
        draw_idx = 1+choose(sampled[:,1:],axis=1)
        ref = sampled[np.arange(DRAWS),draw_idx]
        lo,hi = np.quantile(sampled[:,0]-ref,(.005,.995))
        if name == "rmse":
            rel = 1-point[0]/point[best_point]
            rl,rh = np.quantile(1-sampled[:,0]/ref,(.005,.995))
        else:
            rel = rl = rh = np.nan
        estimates.append(dict(scope="all",parents=len(rows),wells=int(wells.sum()),arm=ARMS[0],
            reference="best_VC",metric=name,best_at_point=ARMS[best_point],arm_value=point[0],ref_value=point[best_point],
            difference=point[0]-point[best_point],low99=lo,high99=hi,relative_gain=rel,relative_low99=rl,relative_high99=rh))
        frequency.extend(dict(metric=name,model=arm,wins=int(np.count_nonzero(draw_idx==i)))
                         for i,arm in enumerate(ARMS[1:],1))
    return estimates,frequency


def verify(estimates, frequencies, expected_estimates, expected_frequency):
    for computed, expected in zip(estimates,expected_estimates,strict=True):
        for key in ("scope","parents","wells","arm","reference","metric","best_at_point"):
            if str(computed[key]) != expected[key]:
                raise ValueError(f"Score metadata differs for {computed['metric']}: {key}")
        np.testing.assert_allclose([computed[x] for x in NUMERIC],
            [float(expected[x]) if expected[x]!="NA" else np.nan for x in NUMERIC],rtol=0,atol=1e-12,equal_nan=True)
    def index(rows):
        return {(r["metric"],r["model"]):int(r["wins"]) for r in rows}
    if index(frequencies)!=index(expected_frequency):
        raise ValueError("Best-adaptation frequency differs from the score ledger")
    for row in estimates:
        if sum(r["wins"] for r in frequencies if r["metric"]==row["metric"])!=DRAWS:
            raise ValueError("Winner counts do not cover the registered number of resamples")


def export_tables(destination, tables):
    destination.mkdir(exist_ok=False)
    for name, rows in tables.items():
        with (destination/(name+".tsv")).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def run(packet, output):
    manifest=json.loads((packet/"manifest.json").read_text())
    if manifest["arms"]!=list(ARMS) or manifest["seed"]!=SEED or manifest["draws"]!=DRAWS:
        raise ValueError("The replay recipe differs from the frozen comparison")
    if set(manifest["datasets"]) != {"JHH5_absolute","JHH5_centered","JHH7_absolute","JHH7_centered"}:
        raise ValueError("The expected four benchmark source sets are required")
    result=defaultdict(list)
    for name, record in sorted(manifest["datasets"].items()):
        folder=packet/name
        if set(record["files"])!={"parent_moments.tsv","expected.tsv","expected_frequency.tsv"}:
            raise ValueError(f"Unexpected replay files: {name}")
        for f,h in record["files"].items():
            if digest(folder/f)!=h:
                raise ValueError(f"Replay file failed its SHA-256 check: {name}/{f}")
        estimates, frequency=replay(folder/"parent_moments.tsv")
        verify(estimates,frequency,tabular(folder/"expected.tsv"),tabular(folder/"expected_frequency.tsv"))
        result["replayed_scores"].extend(dict(dataset=name,**row) for row in estimates)
        result["replayed_winners"].extend(dict(dataset=name,**row) for row in frequency)
        print(f"{name}: {estimates[0]['parents']} parents, {estimates[0]['wells']} wells, verified")
    export_tables(output,result)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--packet",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    run(args.packet,args.out)
