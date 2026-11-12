"""The v2 (cell line, drug) condition table: same split, same groups, new within-well aggregation.

Everything that could be a free parameter is taken from the frozen v1 manifests rather than decided
again here: the 349 structure groups, the 209/70/70 drug split, the 2,000-gene panel, the cell line
order, the 0.5 uM dose, the 50-cell and 100-cell inclusion thresholds. The one thing that changes is
how a well becomes a number:

    v1   E = log1p( 1e4 * sum_cells(count) / sum_cells(library) )     log of the pooled mean
    v2   E = mean_cells( log1p( 1e4 * count / library ) )             mean of the per-cell logs

and therefore every downstream score changes, which is why the baselines and the models are all
re-scored here rather than carried over.

Output has the same shapes and file names as the v1 data directory, so `src/moe_drug_v1`'s Dataset,
evaluator and baselines read it without modification.

One honest difference beyond the transform: the per-line basal DMSO profile is now pooled over the
five plates that carry 0.5 uM wells, because the other nine plates' shards were not read. A DMSO
well is only ever a control for its own plate, so nothing in the truth changes; the *input* basal
profile is pooled over fewer wells than v1's was.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def morgan(smiles: str, n_bits: int, radius: int):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.frombuffer(bytes(gen.GetFingerprintAsNumPy(m).astype(np.uint8)), dtype=np.uint8).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--v1-manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--min-cells-treated", type=int, default=50)
    ap.add_argument("--min-cells-dmso", type=int, default=100)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    a = ap.parse_args()

    raw, O, M = Path(a.raw_dir), Path(a.out_dir), Path(a.manifest_dir)
    O.mkdir(parents=True, exist_ok=True)
    M.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(raw / "groups.tsv", sep="\t")
    logsum = np.load(raw / "logsum.npy")                 # [n_groups, 11132]
    ncells = np.load(raw / "ncells.npy")
    meas = pd.read_csv(Path(a.prep_dir) / "genes.tsv", sep="\t")
    panel = pd.read_csv(Path(a.v1_manifest_dir) / "genes.tsv", sep="\t").sort_values("panel_index")
    tok2col = np.full(63000, -1, np.int64)
    tok2col[meas.token_id.to_numpy()] = np.arange(len(meas))
    pcol = tok2col[panel.token_id.to_numpy()]
    assert (pcol >= 0).all()

    lines = pd.read_csv(Path(a.v1_manifest_dir) / "lines.tsv", sep="\t").sort_values("line_index")
    line_ids = lines.cell_line_id.tolist()
    li = {l: i for i, l in enumerate(line_ids)}
    nL = len(line_ids)

    splits = pd.read_csv(Path(a.v1_manifest_dir) / "splits.tsv", sep="\t").sort_values("group_index")
    groups = splits.group_smiles.astype(str).tolist()
    gi = {s: i for i, s in enumerate(groups)}
    nD = len(groups)

    aliases = pd.read_csv(Path(a.v1_manifest_dir) / "drug_aliases.tsv", sep="\t")
    import ast
    name2group = {}
    for _, r in aliases.iterrows():
        for n in ast.literal_eval(r.drugs):
            name2group[str(n)] = str(r.group)

    wells = pd.read_csv(Path(a.prep_dir) / "wells.tsv", sep="\t").set_index("sample")
    dmso_samples = set(pd.read_csv(Path(a.prep_dir) / "dmso_wells.tsv", sep="\t")["sample"])

    # ---- per (sample, line) mean of the per-cell logs ---------------------------------------
    g["row"] = np.arange(len(g))
    g["line_index"] = g.cell_line_id.map(li)
    assert g.line_index.notna().all()
    mean_log = np.zeros_like(logsum)
    nz = ncells > 0
    mean_log[nz] = logsum[nz] / ncells[nz][:, None]

    plates = sorted(g.plate.unique())
    pidx = {p: i for i, p in enumerate(plates)}
    nP = len(plates)

    # ---- plate DMSO pools, per (plate, line): pooled over that plate's DMSO wells -------------
    dm = g[g["sample"].isin(dmso_samples)]
    dmso_sum = np.zeros((nP, nL, logsum.shape[1]), np.float64)
    dmso_n = np.zeros((nP, nL), np.int64)
    for _, r in dm.iterrows():
        dmso_sum[pidx[r.plate], int(r.line_index)] += logsum[int(r.row)]
        dmso_n[pidx[r.plate], int(r.line_index)] += int(ncells[int(r.row)])
    dmso_mean = np.zeros_like(dmso_sum)
    ok = dmso_n > 0
    dmso_mean[ok] = dmso_sum[ok] / dmso_n[ok][:, None]

    # ---- per-line basal: the DMSO pool over every plate read ---------------------------------
    line_sum = dmso_sum.sum(0)
    line_n = dmso_n.sum(0)
    line_prof = np.zeros_like(line_sum)
    line_prof[line_n > 0] = line_sum[line_n > 0] / line_n[line_n > 0][:, None]

    # ---- treated wells -> conditions ----------------------------------------------------------
    tw = g[~g["sample"].isin(dmso_samples)].copy()
    tw["drug"] = tw["sample"].map(lambda s: str(wells.drug.get(s, "")))
    tw["group"] = tw.drug.map(lambda d: name2group.get(d, ""))
    unmapped = sorted(set(tw.loc[tw.group == "", "drug"]))
    tw = tw[tw.group != ""].copy()
    tw["group_index"] = tw.group.map(gi)
    assert tw.group_index.notna().all()

    nG = len(pcol)
    Y = np.full((nL, nD, nG), np.nan, np.float32)
    valid = np.zeros((nL, nD), bool)
    plate_of = np.full((nL, nD), -1, np.int16)
    n_wells_used = np.zeros((nL, nD), np.int16)

    tw["plate_index"] = tw.plate.map(pidx)
    rows_ = tw.row.to_numpy()
    lidx_ = tw.line_index.to_numpy().astype(int)
    didx_ = tw.group_index.to_numpy().astype(int)
    pidx_ = tw.plate_index.to_numpy().astype(int)
    keep = (ncells[rows_] >= a.min_cells_treated) & (dmso_n[pidx_, lidx_] >= a.min_cells_dmso)

    acc = np.zeros((nL, nD, nG), np.float64)
    plate_votes: dict = {}
    for r, l, d, p_, k in zip(rows_, lidx_, didx_, pidx_, keep):
        if not k:
            continue
        acc[l, d] += mean_log[r][pcol] - dmso_mean[p_, l][pcol]
        n_wells_used[l, d] += 1
        plate_votes.setdefault((l, d), []).append(p_)
    valid = n_wells_used > 0
    Y = np.full((nL, nD, nG), np.nan, np.float32)
    Y[valid] = (acc[valid] / n_wells_used[valid][:, None]).astype(np.float32)
    for (l, d), v in plate_votes.items():
        plate_of[l, d] = Counter(v).most_common(1)[0][0]

    # ---- inputs, identical in form to v1 -------------------------------------------------------
    basal_raw = line_prof[:, pcol].astype(np.float32)
    bmu, bsd = basal_raw.mean(0), basal_raw.std(0)
    basal = ((basal_raw - bmu) / np.maximum(bsd, 1e-6)).astype(np.float32)

    fp = np.zeros((nD, a.n_bits), np.uint8)
    for s, i in gi.items():
        v = morgan(s, a.n_bits, a.radius)
        assert v is not None, f"fingerprint failed for {s}"
        fp[i] = v

    np.save(O / "Y.npy", Y)
    np.save(O / "valid.npy", valid)
    np.save(O / "basal.npy", basal)
    np.save(O / "fp.npy", fp)
    np.save(O / "plate.npy", plate_of)
    np.save(O / "basal_mean.npy", bmu)
    np.save(O / "basal_sd.npy", bsd)

    # manifests: the split, lines and genes are copied through unchanged so the v1 loader works
    for f in ["splits.tsv", "lines.tsv", "genes.tsv", "drug_aliases.tsv"]:
        (M / f).write_text((Path(a.v1_manifest_dir) / f).read_text())

    cov = pd.DataFrame(dict(
        cell_line_id=line_ids,
        n_valid_train=[int(valid[l, splits.split.to_numpy() == "train"].sum()) for l in range(nL)],
        n_valid_val=[int(valid[l, splits.split.to_numpy() == "val"].sum()) for l in range(nL)],
        n_valid_test=[int(valid[l, splits.split.to_numpy() == "test"].sum()) for l in range(nL)]))
    cov.to_csv(M / "coverage.tsv", sep="\t", index=False)

    tr = splits.split.to_numpy() == "train"
    rec = dict(
        truth="mean over cells of log1p(1e4 * count / library), treated minus same-plate DMSO pool "
              "of the same cell line, replicate wells equally weighted",
        v1_truth="log1p(1e4 * sum count / sum library), same subtraction",
        n_lines=nL, n_drug_groups=nD, n_panel_genes=nG,
        n_valid_conditions=int(valid.sum()),
        n_valid_train=int(valid[:, tr].sum()),
        n_valid_val=int(valid[:, splits.split.to_numpy() == "val"].sum()),
        n_valid_test=int(valid[:, splits.split.to_numpy() == "test"].sum()),
        lines_with_zero_coverage=[line_ids[l] for l in range(nL) if not valid[l].any()],
        min_cells_treated=a.min_cells_treated, min_cells_dmso=a.min_cells_dmso,
        unmapped_drug_names=unmapped,
        basal_pooled_over_plates=plates,
        truth_rms_against_zero=float(np.sqrt(np.nanmean(Y[valid] ** 2))),
        sha256_Y=hashlib.sha256(np.ascontiguousarray(Y).tobytes()).hexdigest(),
        panel_from="manifests/moe_drug_v1/genes.tsv (selected on v1 train responses only)",
        split_from="manifests/moe_drug_v1/splits.tsv (seed 20261112)")
    (M / "transform.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
