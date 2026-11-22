"""Export anonymized score moments from the already-scored adult HCC comparison."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.i_score import parent_summaries, squared_rows
from src.r19_plan.jhh_off_position_interaction import binding
from src.r19_plan.jhh_off_position_robustness import frozen_vectors
from src.r19_plan.replay_adult_hcc import ARMS, replay, tabular, verify
from src.r19_plan.t4_position import contrast


def expected(data, cell, scope, frozen_path):
    native = data/(cell.lower()+"_off_position_native")/"score"
    source = native if scope=="absolute" else data/"jhh_off_position_contrast"/cell
    receipt=read_json(source/"receipt.json")
    if (receipt.get("freeze_sha256") or receipt.get("original_freeze_sha256")) != sha256(frozen_path):
        raise ValueError("The expected score lineage is detached from the pinned benchmark freeze")
    points_name="best.tsv" if scope=="absolute" else "centered_best.tsv"
    frequency_name="winner_frequency.tsv" if scope=="absolute" else "strongest_frequency.tsv"
    paths=[source/name for name in (points_name,frequency_name)]
    for path in paths:
        if sha256(path)!=receipt["files"][path.name]:
            raise ValueError(f"Expected score summaries changed: {path.name}")
    return paths, sha256(source/"receipt.json")


def write_moments(path, rows, outcome, predictions):
    energy=squared_rows(outcome)
    moments=[parent_summaries(rows,outcome.shape[1],energy,squared_rows(outcome,predictions[arm]),energy)
             for arm in ARMS]
    parents=rows[["parent","inst_id"]].drop_duplicates().groupby("parent",sort=True).size().to_numpy()
    columns=["ordinal","wells","B0_sse",*[f"{a}_{metric}" for a in ARMS for metric in ("rmse","sse")]]
    with path.open("w",newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle,delimiter="\t",lineterminator="\n")
        writer.writerow(columns)
        for i,wells in enumerate(parents):
            writer.writerow((i+1,wells,moments[0][2][i],*[v for r,s,_ in moments for v in (r[i,0],s[i,0])]))


def export(args):
    reference=binding(args,args.cell)
    rows,y,arrays=frozen_vectors(reference)
    yc, pc, _ = contrast(rows,y,arrays,arms=ARMS)
    for name,outcome,predictions in (("absolute",y,arrays),("centered",yc,pc)):
        table_paths,original_score=expected(args.reports/"data",args.cell,name,reference.native_pin)
        folder=args.out/(args.cell+"_"+name)
        if folder.exists():
            raise FileExistsError(folder)
        with TemporaryDirectory(dir=args.out,prefix=".replay-") as tmp:
            stage=Path(tmp)
            measures=stage/"parent_moments.tsv"
            write_moments(measures,rows,outcome,predictions)
            estimates,frequency=replay(measures)
            verify(estimates,frequency,*(tabular(p) for p in table_paths))
            for filename,src in zip(("expected.tsv","expected_frequency.tsv"), table_paths, strict=True):
                (stage/filename).write_bytes(src.read_bytes())
            write_json(stage/"receipt.json",dict(mode="internal_review",cell=args.cell,scope=folder.name,
                protocol_sha256=sha256(args.protocol), original_freeze_sha256=sha256(reference.native_pin),
                original_score_receipt_sha256=original_score,score_verified=True,
                files={p.name:sha256(p) for p in stage.iterdir()}))
            stage.rename(folder)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cell",choices=("JHH5","JHH7"),required=True)
    for name in ("server","reports","protocol","out"):
        cli.add_argument("--"+name,type=Path,required=True)
    export(cli.parse_args())
