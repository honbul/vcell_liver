"""Audit historical response exposure and generate pre-assay pilot structure features."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import axes
from src.goal_tracks.t1_score import STRATUM_LINES
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, LINE
from src.r13_optimization.lincs_transfer import structural_keys
from src.r15_lincs.design import chemical_features
from src.r15_lincs.forecast import LEGACY_CPU, LEGACY_GPU
from src.r16_lincs.confirmation import embed
from src.r19_plan.t2_control_support import tahoe_support


PILOT_CELL_IDS = {"SNU-423": STRATUM_LINES["SNU-423"][0], "HUH7": "CVCL_0336"}


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Historical or pilot artifact changed: {path}")
    return path


def read_tsv(path, **kwargs):
    return pd.read_csv(path, sep="\t", **kwargs)


def overlap(slots, parents):
    training = set()
    for smiles in set(parents):
        keys, _ = structural_keys(smiles)
        if not keys:
            raise ValueError(f"Unresolved structure in the historical fit universe: {smiles}")
        training.update(keys)
    return [slot for slot, keys in slots.items() if keys & training]


def r17_history(base, reports, slots):
    run = base / "vcell_liver_r17_lincs/v1"
    score = read_json(reports / "r17_lincs/data/score/receipt.json")
    freeze_path = verify(run / "freeze.json", score["freeze_sha256"])
    frozen = read_json(freeze_path)
    rows = []
    for fold in range(3):
        for role in "ABC":
            slice_name = f"f{fold}-{role}"
            items = frozen["slices"][f"A-{slice_name}"]
            if set(items) != {"cpu", *LEGACY_GPU}:
                raise ValueError("Incomplete R17 A score freeze")
            for bundle, digest in items.items():
                source = run / "predict/A" / slice_name / bundle
                stamp = read_json(verify(source / "receipt.json", digest))
                selection = stamp["selection"]
                if (stamp["scenario"], stamp["fold"], stamp["role"], selection["fit"]) != ("A", fold, role, role):
                    raise ValueError("R17 historical slice changed")
                fit, select = set(selection["fit_groups"]), set(selection["selection_groups"])
                held = set(read_tsv(verify(source / "rows.tsv", stamp["files"]["rows.tsv"]),
                                    usecols=[GROUP])[GROUP])
                if not fit or not select or not select < fit or fit & held:
                    raise ValueError("R17 historical parent separation is invalid")
                fit_hit, select_hit = overlap(slots, fit), overlap(slots, select)
                arms = ("blend_FiLM", *LEGACY_CPU) if bundle == "cpu" else (bundle,)
                for arm in arms:
                    if arm not in selection["settings"]:
                        raise ValueError(f"R17 predictive arm missing from audited settings: {arm}")
                    rows.append(dict(history="R17.A", arm=arm, minimum_cells=None, fold=fold, role=role,
                                     fit_parents=len(fit), select_parents=len(select), fit_cells="HEPG2",
                                     snu423_fit_parents=0, huh7_fit_parents=0, fit_hits=len(fit_hit),
                                     select_hits=len(select_hit), fit_slots="|".join(fit_hit),
                                     select_slots="|".join(select_hit)))
    return rows, score["freeze_sha256"]


def t2_history(base, reports, slots):
    path = reports / "goal_tracks/data/known_vc/d5"
    t2 = base / "vcell_liver_goal_tracks/v1"
    score = read_json(path / "receipt.json")
    freeze_path = verify(t2 / "t2/d5/freeze.json", score["freeze_sha256"])
    frozen = read_json(freeze_path)
    stage = t2 / "t1-cache"
    cache = read_json(verify(stage / "receipt.json", frozen["stage_sha256"]))
    models = frozen["eligible_known"]
    if len(models) != 10 or len(frozen["tasks"]) != 12 or frozen["dose"] != 5.:
        raise ValueError("The recorded 5 µM T2 comparison changed")
    rows = []
    for fold in range(3):
        relative = f"f{fold}/fit_rows.tsv"
        registered = read_tsv(verify(stage / relative, cache["files"][relative]))
        for minimum in (20, 200):
            fit = registered.loc[axes(registered, 5., minimum)]
            fit_hit = overlap(slots, fit.smiles)
            cell_support = fit.groupby(LINE)[GROUP].nunique()
            for role in "AB":
                name = f"d5-n{minimum}-f{fold}-{role}"
                expected = frozen["tasks"][name]
                for arm, root in models.items():
                    record = read_json(verify(Path(root) / name / "receipt.json", expected[arm]))
                    if ((record["dose"], record["minimum_cells"], record["fold"], record["role"],
                         record["stage_sha256"], record["models"]) !=
                            (5., minimum, fold, role, frozen["stage_sha256"], [arm])):
                        raise ValueError(f"T2 provenance disagrees with the registered fit metadata: {name}, {arm}")
                    # T2 internal selection uses subsets of this same registered outer-fit universe.
                    rows.append(dict(history="T2.5uM", arm=arm, minimum_cells=minimum, fold=fold, role=role,
                                     fit_parents=fit[GROUP].nunique(), select_parents=None,
                                     fit_cells="|".join(sorted(cell_support.index)),
                                     snu423_fit_parents=int(cell_support.get(PILOT_CELL_IDS["SNU-423"], 0)),
                                     huh7_fit_parents=int(cell_support.get(PILOT_CELL_IDS["HUH7"], 0)),
                                     fit_hits=len(fit_hit), select_hits=None, fit_slots="|".join(fit_hit),
                                     select_slots=""))
    controls, _ = tahoe_support(stage, freeze_path)
    return rows, score["freeze_sha256"], controls


def audit(args, slots, stage):
    keys = {}
    for slot, smiles in zip(slots.parent_slot, slots.parent, strict=True):
        keys[slot], _ = structural_keys(smiles)
        if not keys[slot]:
            raise ValueError(f"Unresolved structure for {slot}")
    r17, first = r17_history(args.base, args.reports, keys)
    t2, second, controls = t2_history(args.base, args.reports, keys)
    frame = pd.DataFrame(r17 + t2).astype({"minimum_cells": "Int64", "select_parents": "Int64", "select_hits": "Int64"})
    frame.to_csv(stage / "model_exposure.tsv", sep="\t", index=False, na_rep="NA")
    pd.DataFrame(controls).to_csv(stage / "tahoe_fit_controls_corrected.tsv", sep="\t", index=False)
    return dict(r17_freeze_sha256=first, t2_freeze_sha256=second, model_slice_rows=len(frame),
                pilot_cellosaurus_ids=PILOT_CELL_IDS, tahoe_control_code_sha256=sha256(tahoe_support.__code__.co_filename))


def features(args, slots, stage):
    root = args.base / "vcell_liver_r15_lincs/v1"
    reference = args.reports / "r11_chemical_probe/manifest/feature_receipt.json"
    prior = read_json(args.reports / "r15_lincs/data/model-inputs/features/receipt.json")
    gene_map = verify(root / "model-inputs/metadata/genes.tsv", prior["metadata_sha256"])
    cache_ref = read_json(args.reports / "r15_lincs/data/cache/GSE92742/receipt.json")
    landmarks = verify(root / "cache/GSE92742/landmarks.tsv", cache_ref["files"]["landmarks.tsv"])
    parents = slots.parent.to_numpy(dtype=str)
    with TemporaryDirectory(dir=args.out.parent) as scratch:
        metadata = Path(scratch)
        slots[["parent"]].to_csv(metadata / "chemicals.tsv", sep="\t", index=False)
        (metadata / "landmarks.tsv").write_bytes(landmarks.read_bytes())
        write_json(metadata / "receipt.json", {"files": {p.name: sha256(p) for p in metadata.iterdir()}})
        view = metadata / "gene-map.tsv"
        read_tsv(gene_map).rename(columns={"token_id": "gene_id"}).to_csv(view, sep="\t", index=False)
        embed(SimpleNamespace(cache=metadata, out=stage, reference=reference,
                              checkpoint=root / "chemberta", gene_map=view))
    verify(stage / "metadata/genes.tsv", prior["metadata_sha256"])
    with np.load(stage / "features/chemberta.npz", allow_pickle=False) as output:
        if (not np.array_equal(parents, output["parents"]) or output["embedding"].shape != (len(slots), 384)
                or not np.isfinite(output["embedding"]).all()):
            raise ValueError("ChemBERTa pilot feature support differs from the fixed axis")
    ecfp = chemical_features(parents)
    if ecfp.shape != (len(parents), 2048) or not np.isin(ecfp, (0, 1)).all() or (ecfp.sum(axis=1) == 0).any():
        raise ValueError("Pilot ECFP is missing or malformed")
    np.savez_compressed(stage / "features/ecfp4.npz", parents=parents, ecfp=ecfp)
    slots[["parent_slot"]].assign(ChemBERTa=True, ECFP4=True).to_csv(
        stage / "structure_support.tsv", sep="\t", index=False)
    return dict(encoder_reference_sha256=sha256(reference), landmark_map_sha256=sha256(gene_map),
                chemberta_dimension=384, ecfp_dimension=2048, token_truncation=0)


def run(args):
    stamp = read_json(args.reports / "r19_plan/data/hcc_pilot_chemistry/audit/receipt.json")
    draft = args.reports / "r19_plan/data/hcc_pilot_chemistry/audit/draft_slots.tsv"
    slots = read_tsv(verify(draft, stamp["files"][draft.name]), usecols=["parent_slot", "parent"])
    if (len(slots) != 96 or slots.parent_slot.tolist() != [f"P{i:03d}" for i in range(1, 97)]
            or not slots.parent.is_unique or slots.isna().any().any()):
        raise ValueError("The draft pilot parent axis changed")
    if args.out.exists():
        raise FileExistsError(args.out)
    with TemporaryDirectory(prefix=".hcc-pilot-model-", dir=args.out.parent) as temporary:
        stage = Path(temporary) / args.mode
        stage.mkdir()
        audit_fields = {"audit": audit, "features": features}[args.mode](args, slots, stage)
        write_json(stage / "receipt.json", dict(
            draft_sha256=sha256(draft), protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
            mode=args.mode, parents=len(slots), trained_response_model=False, expression_read=False,
            files={str(p.relative_to(stage)): sha256(p) for p in stage.rglob("*") if p.is_file()}, **audit_fields))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "features"))
    for name in ("base", "reports", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
