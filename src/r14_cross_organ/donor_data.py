"""Align independently sourced perturbation units for the R14 execution."""

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy import sparse
from scipy.io import mmread

from src.r11_chemical_probe.common import sha256

PROTOCOL = "09774371b5965afbee825e37f955756e90b84700985605a01d1f6c6230c7f63c"
SCI_LINES = dict(A549="CVCL_0023", K562="CVCL_0004", MCF7="CVCL_0031")
MIX_LIVER = {
    "HEP3B217_LIVER": "CVCL_0326", "JHH5_LIVER": "CVCL_0364",
    "LI7_LIVER": "CVCL_3840", "SNU761_LIVER": "CVCL_5089",
}
DATA_COLUMNS = [
    "parent_key", "compound", "smiles", "cell_line_id", "dose_uM", "time_h", "plate_id", "physical_well_id", "n_cells", "scope",
]


@lru_cache(maxsize=None)
def identity(smiles):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "", ""
        try:
            mol = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
            Chem.RemoveStereochemistry(mol)
            key = Chem.MolToInchiKey(mol).split("-")[0]
            canon = Chem.MolToSmiles(mol, canonical=True)
        except (RuntimeError, ValueError):
            return "", ""
    return (key, canon) if len(key) == 14 else ("", "")


def tab(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def ensemble(genes):
    ids = pd.Series(genes, dtype=str).str.split(".").str[0]
    good = ids.str.fullmatch(r"ENSG\d{11}") & ~ids.duplicated(keep=False)
    ix = np.flatnonzero(good)
    return ix[np.argsort(ids.iloc[ix])], np.sort(ids.iloc[ix].to_numpy())


def logcp(counts):
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1)
    norm = np.log1p(counts * (1e4 / np.maximum(1, totals))[:, None])
    return norm.astype(np.float32), totals


def write_output(root, rows, genes, deltas, controls, source_files, details):
    if (set(DATA_COLUMNS) - set(rows.columns) or rows.parent_key.eq("").any()
            or rows.duplicated(["physical_well_id", "cell_line_id"]).any()
            or len(set(genes)) != len(genes) or list(genes) != sorted(genes)):
        raise ValueError("Invalid or duplicate output manifest entry")
    condition = (len(rows), len(genes))
    if deltas.shape != condition or controls.shape != condition or not np.isfinite(deltas).all() or not np.isfinite(controls).all():
        raise ValueError("Response/control and manifest dimensions differ")
    root.mkdir(parents=True, exist_ok=False)
    rows.to_csv(root / "rows.tsv", sep="\t", index=False)
    pd.DataFrame(dict(ensembl_id=genes)).to_csv(root / "genes.tsv", sep="\t", index=False)
    for name, data in (("delta", deltas), ("control", controls)):
        np.save(root / f"{name}.npy", data.astype("float32", copy=False), allow_pickle=False)
    receipt = dict(
        protocol_sha256=PROTOCOL, input_sha256={str(p): sha256(p) for p in source_files},
        files={p.name: sha256(p) for p in sorted(root.iterdir())},
        rows=len(rows), genes=len(genes),
        parents=rows.parent_key.nunique(), details=details,
    )
    with (root / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sciplex(args):
    source = args.base / "vcell_liver_r10_dose_composition/assets/cache/r10_v1"
    wells = tab(source / "wells.tsv").set_index("Combo", verify_integrity=True)
    gene_meta = tab(source / "genes_human.tsv")
    cols, genes = ensemble(gene_meta.id)
    rows, deltas, controls = [], [], []
    files = [source / "wells.tsv", source / "genes_human.tsv"]
    for line, cvcl in SCI_LINES.items():
        line_rows = wells.loc[wells.line.eq(line)].copy()
        cell_path, count_path = source / f"obs_{line}.parquet", source / f"counts_{line}.npz"
        obs = pd.read_parquet(cell_path)
        if not obs.cell.is_unique:
            raise ValueError(f"Nonunique R10 cell IDs for {line}")
        indices = line_rows.index.get_indexer(obs.Combo)
        if (indices < 0).any() or not np.array_equal(np.bincount(indices, minlength=len(line_rows)), line_rows.n_qc):
            raise ValueError(f"R10 cells are not aligned to the physical wells for {line}")
        counts = sparse.load_npz(count_path)
        if counts.shape != (len(obs), len(gene_meta)) or not np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), obs.n_umi_human):
            raise ValueError(f"R10 count row order differs for {line}")
        membership = sparse.csr_matrix((np.ones(len(obs), dtype="int32"), (indices, np.arange(len(obs)))),
                                       shape=(len(line_rows), len(obs)))
        log, total = logcp((membership @ counts).toarray())
        log = log[:, cols]
        files += [cell_path, count_path]
        for plate, ix in line_rows.groupby("culture_plate", sort=True).indices.items():
            plate_rows = line_rows.iloc[ix]
            veh = plate_rows.vehicle.to_numpy()
            ctrl = ix[veh & (total[ix] > 0)]
            if len(ctrl) < 2:
                continue
            average = log[ctrl].mean(axis=0)
            for i in ix[(~veh) & plate_rows.n_qc.ge(20).to_numpy() & (total[ix] > 0)]:
                row = line_rows.iloc[i]
                key, smiles = identity(row.SMILES)
                if not key or row.time_point != 24 or row.dose <= 0:
                    continue
                rows.append(dict(parent_key=key, compound=row.product_name, smiles=smiles, cell_line_id=cvcl,
                                 dose_uM=float(row.dose / 1000), time_h=24, plate_id=f"SciPlex3/{plate}",
                                 physical_well_id=f"SciPlex3/{line_rows.index[i]}", n_cells=int(row.n_qc),
                                 scope="donor", replicate=row.replicate))
                deltas.append(log[i] - average)
                controls.append(average)
    write_output(args.output, pd.DataFrame(rows), genes, np.stack(deltas), np.stack(controls), files,
                 dict(unit="physical parsed Combo well", qc="R10 archived cell QC, min20 cells, min2 plate vehicles",
                      response="log1p CP10k, same plate vehicle mean", normalized_column_totals="all human counts"))


def namekey(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def released_chemistry(args):
    base = args.base / "vcell_liver_r13_optimization/assets/external-gse306429/tahoe-drugs.tsv"
    cmap = Path("/mnt/S1/sdata/pubdata/LINCS/LINCS2020/compoundinfo_beta.txt")
    sci = args.base / "vcell_liver_r10_dose_composition/assets/cache/r10_v1/wells.tsv"
    choices = defaultdict(dict)
    frames = [
        (tab(base), "drug", "canonical_smiles"),
        (tab(cmap), "cmap_name", "canonical_smiles"),
        (tab(sci).loc[lambda x: ~x.vehicle], "product_name", "SMILES"),
    ]
    for (frame, name, smi), source in zip(frames, (base, cmap, sci)):
        for label, raw in set(zip(frame[name], frame[smi])):
            key, canonical = identity(raw)
            if key:
                choices[namekey(label)][key] = (canonical, source.name)
    out = {name: (key, canonical, src) for name, options in choices.items() if len(options) == 1
           for key, (canonical, src) in options.items()}
    return out, [base, cmap, sci]


def mix_design(directory):
    try:
        import openpyxl
    except ImportError:
        sys.path.append("/usr/local/lib/python3.10/dist-packages")
        import openpyxl
    path = directory / "Supplementary Tables.xlsx"
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    design = {namekey(row[0]): (float(row[1]), row[0]) for row in list(book["Supplementary Table 3"].values)[1:]
              if row[0] and isinstance(row[1], (float, int))}
    book.close()
    return path, design


def mix_profile(archive):
    with ZipFile(archive) as bundle:
        files = {Path(name).name: name for name in bundle.namelist() if not name.endswith("/")}
        with bundle.open(files["genes.tsv"]) as stream:
            feature = pd.read_csv(stream, sep="\t", header=None, dtype=str)
        with bundle.open(files["classifications.csv"]) as stream:
            assignments = pd.read_csv(stream, keep_default_na=False)
        with bundle.open(files["barcodes.tsv"]) as stream:
            barcodes = pd.read_csv(stream, sep="\t", header=None, dtype=str)[0]
        ix = pd.Index(barcodes).get_indexer(assignments.barcode)
        if (ix < 0).any() or not assignments.barcode.is_unique:
            raise ValueError(f"MIX-seq barcode assignment differs from matrix: {archive.name}")
        valid = assignments.cell_quality.eq("normal") & assignments.singlet_ID.ne("")
        assignment = assignments.loc[valid, "singlet_ID"].astype(str)
        codes, lines = pd.factorize(assignment, sort=True)
        membership = sparse.csr_matrix((np.ones(len(codes), dtype="int32"), (codes, ix[valid])),
                                       shape=(len(lines), len(barcodes)))
        with bundle.open(files["matrix.mtx"]) as stream:
            counts = mmread(stream).tocsr()
        if counts.shape != (len(feature), len(barcodes)):
            raise ValueError(f"MIX-seq matrix/metadata sizes disagree: {archive.name}")
        expression, totals = logcp((membership @ counts.T).toarray())
        return list(lines), np.bincount(codes), expression, totals, feature[0]


def mixseq(args):
    directory = args.base / "vcell_liver_r14_cross_organ/raw/mix_seq"
    sources = sorted(directory.glob("*.zip"))
    raw_receipt = directory / "receipt.json"
    chemistry, source_files = released_chemistry(args)
    design_path, design = mix_design(directory)
    picks, outlines = {}, {}
    for archive in sources:
        prefix, exp = re.fullmatch(r"(.*)_expt(3|10)", archive.stem).groups()
        compound = prefix.removesuffix("_24hr")
        if compound == "DMSO":
            picks[archive] = ("", compound, "", exp, 0.0, "release DMSO")
            continue
        key = namekey(compound)
        # Source archive uses BRD3379 abbreviated in the same supplied release table.
        annotation = design.get(key)
        if annotation is None:
            outlines[archive.name] = "dose absent from publication design table"
            continue
        if key not in chemistry:
            outlines[archive.name] = "no unique released structure by name"
            continue
        parent, smiles, src = chemistry[key]
        picks[archive] = (parent, annotation[1], smiles, exp, annotation[0], src)
    by_exp = {}
    frames = []
    for archive in sorted(picks, key=lambda p: (picks[p][1] != "DMSO", p.name)):
        key, compound, smiles, exp, dose, src = picks[archive]
        lines, count, val, totals, genes = mix_profile(archive)
        take, ids = ensemble(genes)
        if compound == "DMSO":
            by_exp[exp] = dict(zip(lines, zip(val[:, take], count, totals)))
            expected = ids if not frames else expected
            continue
        if not np.array_equal(ids, expected):
            raise ValueError("MIX-seq archives differ in Ensembl axis")
        ctrl = by_exp[exp]
        result = []
        for i, line in enumerate(lines):
            if line not in ctrl or count[i] < 10 or ctrl[line][1] < 10 or totals[i] <= 0 or ctrl[line][2] <= 0:
                continue
            if line.endswith("_LIVER") and line not in MIX_LIVER:
                raise ValueError(f"Unrecognized liver line: {line}")
            lineage = MIX_LIVER.get(line, f"MIXSEQ/{line}")
            v = ctrl[line][0]
            row = dict(parent_key=key, compound=compound, smiles=smiles, cell_line_id=lineage,
                       dose_uM=dose, time_h=24, plate_id=f"MIXseq/experiment{exp}",
                       physical_well_id=f"MIXseq/{archive.stem}", n_cells=int(count[i]),
                       scope="external" if line in MIX_LIVER else "donor",
                       release_archive=archive.name, chemistry_reference=src,
                       control=f"MIXseq/experiment{exp}/DMSO")
            result.append((row, val[i, take] - v, v))
        frames.extend(result)
        print(archive.name, len(result), flush=True)
    rows, values, controls = zip(*frames)
    write_output(args.output, pd.DataFrame(rows), expected, np.stack(values), np.stack(controls),
                 [*sources, raw_receipt, design_path, *source_files],
                 dict(excluded=outlines, unit="one treatment archive in a numbered pool experiment",
                      qc="published normal singlets; 10 minimum treatment/control cells per line",
                      response="log1p CP10k, experiment matched DMSO pseudobulk"))


def gse306429(args):
    from src.r13_optimization.external import COUNT_LAYER, OBS, VEHICLE_OBS, read_count_rows, read_manifest, select_query

    base = args.base / "vcell_liver_r13_optimization/assets/external-gse306429"
    meta = base / "manifest-v2"
    receipt = read_manifest(meta)
    matrices = Path(receipt["download"]["path"])
    if sha256(matrices) != receipt["download"]["sha256"]:
        raise ValueError("GSE306429 count source differs from metadata receipt")
    candidates, query = select_query(meta, receipt, "HepG2", 1, 24, 3)
    source = tab(meta / "genes.tsv")
    selected, genes = ensemble(source.ensembl_id)
    assay_shape = receipt["matrix_descriptors_only"][COUNT_LAYER]["shape"]
    indices = np.r_[query[OBS], query[VEHICLE_OBS]]
    normalized = read_count_rows(matrices, assay_shape, indices, source.external_col.iloc[selected].to_numpy())
    actual, control = np.split(normalized, 2)
    chemistry = [identity(s) for s in query.smiles]
    keys, smile = zip(*chemistry)
    rows = query.assign(parent_key=keys, compound=query.compound_name, smiles=smile,
                        n_cells=0, scope="external")
    columns = DATA_COLUMNS + ["control_role", "control_well_id"]
    write_output(args.output, rows[columns], genes, actual - control, control,
                 [meta / f"{name}.tsv" for name in receipt["output_sha256"]] + [matrices, meta / "receipt.json"],
                 dict(candidate_rows=len(candidates), unknown_cell_counts="0 indicates absent metadata",
                      unit="published physical treatment well", response="log1p CP10k minus same library and replicate DMSO"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["sciplex3", "mixseq", "gse306429"])
    for name in ("base", "protocol", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if sha256(args.protocol) != PROTOCOL:
        raise ValueError("Execution protocol differs from prescoring commit")
    {"sciplex3": sciplex, "mixseq": mixseq, "gse306429": gse306429}[args.kind](args)


if __name__ == "__main__":
    main()
