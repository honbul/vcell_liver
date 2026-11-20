"""Turn the fixed single-cell sample into the AnnData that chemCPA's own loader reads.

Nothing here invents a new data contract: the cells, the gene panel, the drug groups and the
209/70/70 drug split all come straight from the v2 pass and the v1 manifests. This only arranges
them the way `chemCPA.data.load_dataset_splits` expects.

Split mapping.  chemCPA knows three splits and this study needs three:

    chemCPA "train"  <-  the 209 training drug groups' treated cells, plus most control cells
    chemCPA "test"   <-  the 70 *validation* drug groups' treated cells, plus the rest of the
                         controls.  Everything that selects a setting reads this and only this.
    chemCPA "ood"    <-  the 70 *test* drug groups' treated cells.  Opened once, at the end.

So "test" in a chemCPA log means this study's validation split. The names are chemCPA's; the
discipline is the study's.

Control cells.  DMSO cells carry no drug response, so dividing them between the train and
validation splits leaks nothing about a held-out drug. The division is by a hash of the barcode,
which makes it reproducible and independent of row order.

Differentially expressed genes.  chemCPA's `uns[degs_key]` exists to score `R2_mean_de`. Computing
DE genes for a held-out drug requires that drug's held-out responses, so the entry here is present
with empty gene lists: the mask is all zeros, the field is well formed, and no held-out truth is
read. This study's metrics are computed by its own evaluator, not by `evaluate_r2`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def bhash(s: str) -> float:
    """A stable value in [0, 1) from a string. blake2b, not Python's salted hash."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") / 2.0 ** 64


def sanitise(name: str) -> str:
    """chemCPA splits drug names on '+' and builds pert_category by joining on '_'."""
    return name.replace("+", "-").replace("_", "-").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="output of build_v2.py")
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--v1-manifest-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--control-to-val", type=float, default=0.3)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    cells = pd.read_csv(raw / "cells_meta.tsv", sep="\t")
    X = np.load(raw / "cells_X.npy")
    assert len(cells) == len(X), (len(cells), len(X))

    wells = pd.read_csv(Path(args.prep_dir) / "wells.tsv", sep="\t")
    lines = pd.read_csv(Path(args.prep_dir) / "lines.tsv", sep="\t")
    panel = pd.read_csv(Path(args.v1_manifest_dir) / "genes.tsv", sep="\t").sort_values("panel_index")
    splits = pd.read_csv(Path(args.v1_manifest_dir) / "splits.tsv", sep="\t")
    aliases = pd.read_csv(Path(args.v1_manifest_dir) / "drug_aliases.tsv", sep="\t")

    # well -> drug group (the stereochemistry-free parent SMILES the v1 split is defined on).
    # `drug_aliases.tsv` stores each group's drug names as a Python list literal in `drugs`.
    import ast
    name2group, group2label = {}, {}
    for _, r in aliases.iterrows():
        names = [str(x) for x in ast.literal_eval(r.drugs)]
        for n in names:
            name2group[n] = str(r.group)
        group2label[str(r.group)] = sanitise(sorted(names)[0])
    group2split = dict(zip(splits.group_smiles.astype(str), splits.split.astype(str)))

    w = wells.set_index("sample")
    line_name = dict(zip(lines.cell_line_id, lines.cell_name))

    is_dmso = ~cells["sample"].isin(w.index)
    drug_name = np.where(is_dmso, "DMSO",
                         cells["sample"].map(lambda s: w.drug.get(s, "")).astype(str))
    group = pd.Series(drug_name).map(lambda d: name2group.get(d, ""))
    label = group.map(lambda g: group2label.get(g, ""))
    split_of = group.map(lambda g: group2split.get(g, ""))

    keep = is_dmso.to_numpy() | (split_of != "").to_numpy()
    dropped = int((~keep).sum())
    cells, X = cells[keep].reset_index(drop=True), X[keep]
    is_dmso, label, split_of = is_dmso[keep].to_numpy(), label[keep].to_numpy(), split_of[keep].to_numpy()
    group = group[keep].to_numpy()

    cell_type = cells.cell_line_id.map(line_name).astype(str).to_numpy()
    condition = np.where(is_dmso, "DMSO", label)
    dose = np.where(is_dmso, 0.0, 0.5)

    # chemCPA split names; see the module docstring for the mapping
    chem_split = np.where(split_of == "train", "train",
                 np.where(split_of == "val", "test",
                 np.where(split_of == "test", "ood", "")))
    ctrl_to_val = np.array([bhash("ctrl|" + b) < args.control_to_val for b in cells.barcode])
    chem_split = np.where(is_dmso, np.where(ctrl_to_val, "test", "train"), chem_split)
    assert (chem_split != "").all()

    smiles = np.where(is_dmso, "CS(C)=O", group)      # DMSO's own structure for the control rows
    pert_cat = np.char.add(np.char.add(np.char.add(cell_type.astype(str), "_"),
                                       condition.astype(str)), np.char.add("_", dose.astype(str)))
    assert all(len(k.split("_")) == 3 for k in pert_cat[:1000]), "pert_category must be 3 fields"

    obs = pd.DataFrame(dict(
        condition=pd.Categorical(condition),
        dose=dose.astype(np.float32),
        cell_type=pd.Categorical(cell_type),
        SMILES=pd.Categorical(smiles),
        cov_drug_dose_name=pd.Categorical(pert_cat),
        split=pd.Categorical(chem_split),
        control=np.where(is_dmso, 1, 0).astype(np.int64),
        plate=pd.Categorical(cells.plate.astype(str)),
        sample=pd.Categorical(cells["sample"].astype(str)),
        cell_line_id=pd.Categorical(cells.cell_line_id.astype(str)),
        barcode=cells.barcode.astype(str).to_numpy(),
        library_size=cells.library_size.astype(np.float32).to_numpy()))

    var = pd.DataFrame(index=panel.gene_symbol.astype(str).to_numpy())
    var["ensembl_id"] = panel.ensembl_id.astype(str).to_numpy()
    var["token_id"] = panel.token_id.to_numpy()

    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    adata.uns["all_DEGs"] = {str(k): np.array([], dtype=object)
                             for k in pd.unique(pert_cat)}
    adata.uns["v2_contract"] = json.dumps(dict(
        expression="log1p(1e4 * count / library_size), per cell",
        panel_genes=int(len(var)), dose_uM=0.5,
        split_map={"train": "train drug groups", "test": "VALIDATION drug groups",
                   "ood": "TEST drug groups"},
        degs="empty by design: computing them for a held-out drug needs its held-out response"))
    adata.write_h5ad(args.out)

    man = dict(
        n_cells=int(adata.n_obs), n_genes=int(adata.n_vars),
        n_cells_dropped_no_group=dropped,
        by_split={k: int(v) for k, v in pd.Series(chem_split).value_counts().items()},
        n_control_cells=int((obs.control == 1).sum()),
        control_to_val_fraction=args.control_to_val,
        n_drug_labels=int(pd.Series(condition[~is_dmso]).nunique()),
        n_cell_types=int(pd.Series(cell_type).nunique()),
        drugs_per_split={s: int(pd.Series(condition[(chem_split == s) & ~is_dmso]).nunique())
                         for s in ["train", "test", "ood"]},
        cells_per_split_treated={s: int(((chem_split == s) & ~is_dmso).sum())
                                 for s in ["train", "test", "ood"]},
        sha256_X=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        out=str(args.out))
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1), flush=True)

    # the property every holdout claim rests on
    treated_groups = {s: set(group[(chem_split == s) & ~is_dmso]) for s in ["train", "test", "ood"]}
    for a, b in [("train", "test"), ("train", "ood"), ("test", "ood")]:
        inter = treated_groups[a] & treated_groups[b]
        assert not inter, f"{a} and {b} share drug groups: {sorted(inter)[:5]}"
    print("drug groups disjoint across train / test / ood: ok", flush=True)


if __name__ == "__main__":
    main()
