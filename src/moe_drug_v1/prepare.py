"""Build the frozen (cell line, drug) condition table for the conditional-expert experiment.

Reads only the already-aggregated R3 prep assets (well x cell-line pseudobulk and plate-matched DMSO);
the 100M-cell raw mirror is never re-read. Reuse is limited to aggregation, the Delta definition and
ID linkage -- no previously fitted coefficient, panel or hyperparameter enters here.

Truth (plan s4.3, identical to r3_prep.py):
    E[l, w, g]      = log1p(mean over cells of CP10k)                      -- already in delta.npy
    Delta[l, w, g]  = E[l, w, g] - E[l, plate(w) DMSO pool, g]             -- already in delta.npy
    y[l, drug, g]   = equal-weight mean of Delta over that drug's valid wells

Outputs (assets/moe_drug_v1/data + manifests/moe_drug_v1):
    Y.npy      float32 [n_lines, n_groups, n_panel]  NaN where the condition is not valid
    valid.npy  bool    [n_lines, n_groups]
    basal.npy  float32 [n_lines, n_panel]            standardized pooled-DMSO profile
    fp.npy     uint8   [n_groups, 2048]              Morgan radius 2
    plate.npy  int16   [n_lines, n_groups]           modal plate index of the averaged wells (diagnostics only)
    manifests: sources, splits, drug_aliases, genes, coverage, transform
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# plan s4.5 / configs/moe_drug_v1/protocol.yaml -- fixed before any response was read
TECH_PREFIXES = ("MT-", "RNA5S", "RNA18S", "RNA28S", "RNA45S", "RNA5-8S", "RN7SK", "RN7SL")
TECH_SYMBOLS = {"RMRP", "RPPH1"}
SNU423 = "CVCL_0366"


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


def canonical_parent(smiles: str):
    """Largest organic fragment, uncharged, canonical SMILES, with and without stereochemistry.

    The grouping unit has to be at least as coarse as what the model's input can distinguish. A
    radius-2 Morgan fingerprint is not chirality-aware, so two records of one compound that differ
    only in whether the stereocentres were annotated produce the *same* input vector; splitting them
    would put an identical input in train and in the evaluation set. Grouping therefore uses the
    stereochemistry-free key, and the stereo-aware form is kept only for the record.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
    except Exception:
        return None
    if m is None or m.GetNumAtoms() == 0:
        return None
    stereo = Chem.MolToSmiles(m, canonical=True)
    flat = Chem.Mol(m)
    Chem.RemoveStereochemistry(flat)
    return stereo, Chem.MolToSmiles(flat, canonical=True)


def morgan(smiles: str, n_bits: int, radius: int):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.frombuffer(bytes(gen.GetFingerprintAsNumPy(m).astype(np.uint8)), dtype=np.uint8).copy()


def assign_split(keys: list[str], strata: list[str], fracs, seed: int):
    """Plate-stratified 60/20/20 over drug groups using metadata only (plan s5.2 step 3).

    Within each plate stratum the groups are shuffled by the fixed seed and cut at the running
    fraction boundaries, so each stratum contributes the same proportion to every split.
    """
    rng = np.random.default_rng(seed)
    out = {}
    log = []
    for s in sorted(set(strata)):
        idx = [i for i, t in enumerate(strata) if t == s]
        order = rng.permutation(len(idx))
        n = len(idx)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)
        for rank, j in enumerate(order):
            k = keys[idx[j]]
            out[k] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
        log.append(dict(stratum=s, n=n, train=n_tr, val=n_va, test=n - n_tr - n_va))
    return out, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True, help="r3 prep assets (read-only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--dose", type=float, default=0.5)
    ap.add_argument("--min-cells-treated", type=int, default=50)
    ap.add_argument("--min-cells-dmso", type=int, default=100)
    ap.add_argument("--n-panel", type=int, default=2000)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--split-seed", type=int, default=20261112)
    args = ap.parse_args()
    t0 = time.time()
    P, O, M = Path(args.prep_dir), Path(args.out_dir), Path(args.manifest_dir)
    O.mkdir(parents=True, exist_ok=True)
    M.mkdir(parents=True, exist_ok=True)

    import rdkit

    wells = pd.read_csv(P / "wells.tsv", sep="\t")
    lines = pd.read_csv(P / "lines.tsv", sep="\t")
    genes = pd.read_csv(P / "genes.tsv", sep="\t")
    dmso_wells = pd.read_csv(P / "dmso_wells.tsv", sep="\t")
    plates = sorted(dmso_wells.plate.unique())            # same order r3_prep.py used
    pidx = {p: i for i, p in enumerate(plates)}
    dmso_plate_n = np.load(P / "dmso_plate_ncells.npy")   # [n_plates, n_lines]
    line_prof = np.load(P / "dmso_line_profile.npy")      # [n_lines, n_cand_genes]
    ncells = np.load(P / "ncells.npy")                    # [n_lines, n_wells]
    delta = np.load(P / "delta.npy", mmap_mode="r")       # [n_lines, n_wells, n_cand_genes]
    nL, nW, nG = delta.shape
    assert len(lines) == nL and len(wells) == nW and len(genes) == nG

    # ---- 1. dose filter + drug structure grouping (metadata only) -------------------------------
    sel_dose = wells.index[wells.dose_uM == args.dose].to_numpy()
    w = wells.loc[sel_dose].copy()
    drop_no_smiles = sorted(w.loc[w.canonical_smiles.isna(), "drug"].unique())
    w = w[w.canonical_smiles.notna()].copy()

    parents, bad_parse = {}, []
    for s in sorted(w.canonical_smiles.unique()):
        p = canonical_parent(s)
        if p is None:
            bad_parse.append(s)
        else:
            parents[s] = p
    w = w[w.canonical_smiles.isin(parents)].copy()
    w["parent_smiles"] = w.canonical_smiles.map(lambda s: parents[s][0])
    w["flat_smiles"] = w.canonical_smiles.map(lambda s: parents[s][1])
    sel = w.index.to_numpy()          # rows of the original wells table that survive every filter

    # A group = one canonical parent structure as the fingerprint sees it, i.e. stereochemistry-free.
    # PubChem CID is cross-checked, never used to split a structure that is already identical, and a
    # CID clash across two groups is reported rather than merged.
    w["group"] = w.flat_smiles
    stereo_merges = {k: sorted(set(v)) for k, v in w.groupby("flat_smiles").parent_smiles.agg(set).items()
                     if len(v) > 1}
    cid_map = (w.dropna(subset=["pubchem_cid"]).groupby("group").pubchem_cid
               .agg(lambda s: sorted(set(int(x) for x in s))))
    cid_conflict = {g: c for g, c in cid_map.items() if len(c) > 1}
    cid_to_groups = {}
    for g, cs in cid_map.items():
        for c in cs:
            cid_to_groups.setdefault(c, []).append(g)
    cid_shared = {int(c): gs for c, gs in cid_to_groups.items() if len(gs) > 1}

    groups = sorted(w.group.unique())
    gi = {g: i for i, g in enumerate(groups)}
    nD = len(groups)
    alias = (w.groupby("group")
             .agg(drugs=("drug", lambda s: sorted(set(s))),
                  n_wells=("sample", "size"),
                  plates=("plate", lambda s: sorted(set(s))),
                  cids=("pubchem_cid", lambda s: sorted({int(x) for x in s if pd.notna(x)})),
                  stereo_parents=("parent_smiles", lambda s: sorted(set(s))),
                  smiles_variants=("canonical_smiles", lambda s: sorted(set(s))))
             .reset_index())
    alias["n_drug_names"] = alias.drugs.str.len()

    # ---- 2. validity mask and equal-weight averaging ---------------------------------------------
    well_plate_i = np.array([pidx[p] for p in w.plate])
    dmso_ok = dmso_plate_n[well_plate_i] >= args.min_cells_dmso        # [n_sel_wells, n_lines]
    cells_ok = ncells[:, sel].T >= args.min_cells_treated              # [n_sel_wells, n_lines]
    ok = (dmso_ok & cells_ok).T                                        # [n_lines, n_sel_wells]

    Yfull = np.zeros((nL, nD, nG), np.float32)
    valid = np.zeros((nL, nD), bool)
    plate_of = np.full((nL, nD), -1, np.int16)
    gcol = np.array([gi[g] for g in w.group])
    for d in range(nD):
        cols = np.where(gcol == d)[0]
        wrows = sel[cols]
        blk = np.asarray(delta[:, wrows, :], np.float32)               # [n_lines, k, n_genes]
        m = ok[:, cols] & np.isfinite(blk).all(2)
        cnt = m.sum(1)
        blk = np.where(m[:, :, None], blk, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Yfull[:, d] = blk.sum(1) / np.maximum(cnt, 1)[:, None]
        valid[:, d] = cnt > 0
        for l in np.where(cnt > 0)[0]:
            plate_of[l, d] = Counter(well_plate_i[cols[m[l]]]).most_common(1)[0][0]
    Yfull[~valid] = np.nan

    # ---- 3. split on metadata (plate stratum = modal plate of the group's wells) -----------------
    strata = [Counter(w.plate[gcol == gi[g]]).most_common(1)[0][0] for g in groups]
    split_map, split_log = assign_split(groups, strata, (0.6, 0.2, 0.2), args.split_seed)
    split = np.array([split_map[g] for g in groups])
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    # ---- 4. gene panel from TRAIN responses only -------------------------------------------------
    sym = genes.gene_symbol.fillna("").to_numpy()
    tech = np.array([s.startswith(TECH_PREFIXES) or s in TECH_SYMBOLS for s in sym])
    Ytr = Yfull[:, tr, :]
    mtr = valid[:, tr]
    n_ok = mtr.sum()
    s1 = np.where(mtr[:, :, None], Ytr, 0.0).sum((0, 1))
    s2 = np.where(mtr[:, :, None], Ytr.astype(np.float64) ** 2, 0.0).sum((0, 1))
    var = s2 / n_ok - (s1 / n_ok) ** 2
    var_masked = np.where(tech, -np.inf, var)
    panel = np.sort(np.argsort(-var_masked, kind="stable")[: args.n_panel])
    assert not tech[panel].any()

    Y = np.ascontiguousarray(Yfull[:, :, panel])
    del Yfull

    # ---- 5. inputs -------------------------------------------------------------------------------
    basal_raw = line_prof[:, panel].astype(np.float32)
    bmu = basal_raw.mean(0)
    bsd = basal_raw.std(0)
    bsd_floor = np.maximum(bsd, 1e-6)
    basal = ((basal_raw - bmu) / bsd_floor).astype(np.float32)

    fp = np.zeros((nD, args.n_bits), np.uint8)
    fp_fail = []
    for g in groups:
        v = morgan(g, args.n_bits, args.radius)
        if v is None:
            fp_fail.append(g)
        else:
            fp[gi[g]] = v
    assert not fp_fail, f"parent SMILES failed fingerprinting: {fp_fail}"

    # ---- 6. coverage -----------------------------------------------------------------------------
    cov = pd.DataFrame(dict(
        cell_line_id=lines.cell_line_id, cell_name=lines.cell_name, organ=lines.organ,
        train=valid[:, tr].sum(1), val=valid[:, va].sum(1), test=valid[:, te].sum(1)))
    cov["keep"] = (cov.train >= 100) & (cov.val >= 30) & (cov.test >= 30)

    # ---- 7. write --------------------------------------------------------------------------------
    np.save(O / "Y.npy", Y)
    np.save(O / "valid.npy", valid)
    np.save(O / "basal.npy", basal)
    np.save(O / "fp.npy", fp)
    np.save(O / "plate.npy", plate_of)
    np.save(O / "basal_mean.npy", bmu)
    np.save(O / "basal_sd.npy", bsd)

    pd.DataFrame(dict(group_index=range(nD), group_smiles=groups, split=split, plate_stratum=strata,
                      n_valid_lines=valid.sum(0))).to_csv(M / "splits.tsv", sep="\t", index=False)
    alias.assign(split=[split_map[g] for g in alias.group]).to_csv(M / "drug_aliases.tsv", sep="\t", index=False)
    genes.iloc[panel].assign(panel_index=range(len(panel)), train_variance=var[panel]) \
        .to_csv(M / "genes.tsv", sep="\t", index=False)
    cov.to_csv(M / "coverage.tsv", sep="\t", index=False)
    pd.DataFrame(dict(cell_line_id=lines.cell_line_id, line_index=range(nL))).to_csv(
        M / "lines.tsv", sep="\t", index=False)

    snu = int(np.where(lines.cell_line_id == SNU423)[0][0]) if (lines.cell_line_id == SNU423).any() else None
    sources = dict(
        prep_dir=str(P),
        prep_meta=json.loads((P / "prep_meta.json").read_text()),
        files={f: dict(sha256_first_64MB=sha256(P / f, 64 << 20), bytes=(P / f).stat().st_size)
               for f in ["delta.npy", "ncells.npy", "dmso_line_profile.npy", "dmso_plate_ncells.npy",
                         "wells.tsv", "lines.tsv", "genes.tsv", "dmso_wells.tsv"]},
        rdkit_version=rdkit.__version__, numpy_version=np.__version__, pandas_version=pd.__version__,
        python=platform.python_version(),
        commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    )
    (M / "sources.json").write_text(json.dumps(sources, indent=1))

    transform = dict(
        dose_uM=args.dose, min_cells_treated=args.min_cells_treated, min_cells_dmso=args.min_cells_dmso,
        truth="mean over valid wells of (log1p(mean CP10k treated) - log1p(mean CP10k plate+line DMSO pool))",
        replicate_rule="equal weight per valid well",
        basal="pooled DMSO profile of the line, z-scored per gene over the 50 lines",
        basal_sd_floor=1e-6, n_genes_below_sd_floor=int((bsd < 1e-6).sum()),
        fingerprint=dict(kind="morgan", radius=args.radius, n_bits=args.n_bits, rdkit=rdkit.__version__),
        technical_exclusion=dict(prefixes=list(TECH_PREFIXES), symbols=sorted(TECH_SYMBOLS),
                                 n_excluded=int(tech.sum())),
        panel=dict(n=len(panel), selection="top variance of train-split responses only",
                   n_train_conditions_used=int(n_ok)),
        split=dict(seed=args.split_seed, fractions=[0.6, 0.2, 0.2], unit="canonical parent SMILES",
                   stratified_by="modal plate of the group's wells", per_stratum=split_log),
        drugs=dict(n_wells_at_dose=len(sel_dose), n_wells_kept=int(len(w)), n_groups=nD,
                   dropped_missing_smiles=drop_no_smiles, dropped_unparsable_smiles=bad_parse,
                   groups_with_multiple_drug_names=int((alias.n_drug_names > 1).sum()),
                   stereo_variants_merged=stereo_merges,
                   cid_conflicts_within_group={k: v for k, v in cid_conflict.items()},
                   cid_shared_across_groups=cid_shared),
        conditions=dict(n_lines=nL, n_valid_conditions=int(valid.sum()),
                        n_possible=int(nL * nD), frac_valid=float(valid.mean())),
        snu423_line_index=snu,
        elapsed_s=round(time.time() - t0, 1),
    )
    (M / "transform.json").write_text(json.dumps(transform, indent=1))
    print(json.dumps({k: v for k, v in transform.items() if k != "split"}, indent=1), flush=True)
    print("split per stratum:", json.dumps(split_log), flush=True)
    print(cov.to_string(index=False), flush=True)
    if snu is not None:
        print(f"SNU-423 valid drugs  train={cov.train[snu]} val={cov.val[snu]} test={cov.test[snu]}", flush=True)


if __name__ == "__main__":
    main()
