"""Freeze Tahoe metadata and aggregate raw counts in independently staged source and final caches."""

import ast
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem
import scipy.sparse as sp

from src.moe_drug_v1.prepare import canonical_parent

from .common import arguments, config, read_json, sha256, write_json


SPLIT = Path("manifests/depth_robust_r6/drug_split.json")
WELLS = Path("manifests/anchored_distribution_r3/wells.tsv")
COLS = ["sample", "plate", "cell_line_id", "genes", "expressions"]
_SCAN = {}


def check_revision(root, relative, revision):
    metadata = root / ".cache/huggingface/download" / (str(relative) + ".metadata")
    lines = metadata.read_text().splitlines()
    if lines[0] != revision:
        raise ValueError(f"Unexpected mirror revision for {relative}: {lines[0]}")
    return lines[1]


def unique(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate join keys: {keys}")


def metadata(cfg, out):
    data = cfg["data"]
    root = Path(data["root"])
    revision = data["revision"]
    inputs = [Path("README.md")]
    tables = {}
    for key in ("sample", "drug", "gene", "cell_line"):
        path = Path(f"metadata/{key}_metadata.parquet")
        inputs.append(path)
        tables[key] = pq.read_table(root / path).to_pandas()
    provenance = {str(p): dict(etag=check_revision(root, p, revision), sha256=sha256(root / p)) for p in inputs}

    samples = tables["sample"]
    unique(samples, ["sample"])
    if set(samples.plate) != {f"plate{i}" for i in range(1, 15)}:
        raise ValueError("Canonical plate mapping differs from the documented 1–14 labels")
    drugs = tables["drug"][["drug", "canonical_smiles", "pubchem_cid"]]
    unique(drugs, ["drug"])
    panel = tables["gene"].sort_values("token_id")
    unique(panel, ["token_id"])
    if not np.array_equal(panel.token_id, np.arange(3, len(panel) + 3)):
        raise ValueError("Unrecognized mapped gene token range")

    split = read_json(SPLIT)
    parent_to_group = {s: int(g) for g, s in split["group_smiles"].items()}
    split_by_group = {int(g): role for role, groups in split["groups"].items() for g in groups}
    folds = {int(g): int(f) for g, f in split["dev_fold"].items()}
    drugs = drugs.copy()
    drugs["parent"] = drugs.canonical_smiles.map(
        lambda s: canonical_parent(s) if isinstance(s, str) and s else None
    )
    drugs["compound_group_id"] = drugs.parent.map(lambda p: parent_to_group.get(p[1], -1) if p else -1)
    drugs["compound_entity_id"] = drugs.drug
    drugs["smiles"] = drugs.canonical_smiles.map(
        lambda s: Chem.MolToSmiles(m) if isinstance(s, str) and s and (m := Chem.MolFromSmiles(s)) is not None else ""
    )
    joined = samples.merge(drugs[["drug", "compound_group_id", "compound_entity_id", "smiles", "pubchem_cid"]],
                           on="drug", how="left", validate="many_to_one")
    joined["compound_group_id"] = joined.compound_group_id.fillna(-1).astype(int)
    joined["split"] = joined.compound_group_id.map(split_by_group).fillna("outside_r6")
    joined["fold"] = joined.compound_group_id.map(folds).fillna(-1).astype(int)
    joined["dose_uM"] = joined.drugname_drugconc.map(dose)
    joined["time_h"] = data["time_h"]
    assigned = joined[joined.compound_group_id.ge(0)]
    for column in ("smiles", "pubchem_cid"):
        known = assigned[assigned[column].notna() & assigned[column].astype(str).ne("")]
        if known.groupby(column).compound_group_id.nunique().gt(1).any():
            raise ValueError(f"Known {column} connects historical compound groups; split must be rebuilt")

    legacy = pd.read_csv(WELLS, sep="\t").drop_duplicates(["sample", "plate", "drug"])
    unique(legacy, ["sample"])
    for record in legacy.merge(joined, on="sample", how="left", validate="one_to_one").itertuples():
        if record.plate_x != record.plate_y or (
            record.kind == "treated" and (record.drug_x != record.compound_group_id or record.dose_uM != data["dose_uM"])
        ) or (record.kind == "ctrl" and record.drug_y != "DMSO_TF"):
            raise ValueError(f"R6 well mapping conflicts with raw metadata: {record.sample}")

    eligible = joined.dose_uM.eq(data["dose_uM"]) & joined.compound_group_id.ge(0)
    plates = sorted(joined.loc[eligible, "plate"].unique())
    controls = joined[joined.drug.eq("DMSO_TF") & joined.plate.isin(plates)].sort_values("sample").copy()
    if set(controls.groupby("plate").size()) != {2} or set(controls.plate) != set(plates):
        raise ValueError("Expected two physically separate DMSO wells on each treatment plate")
    controls["control_role"] = controls.groupby("plate").cumcount().map({0: "A", 1: "B"})
    controls["split"] = "control"
    treated = joined[eligible].assign(control_role="treated")
    # Record excluded sample metadata without opening any response.
    excluded = joined[joined.dose_uM.eq(data["dose_uM"]) & ~eligible]
    use = pd.concat([treated, controls], ignore_index=True).rename(columns={"sample": "physical_well_id", "plate": "plate_id"})
    use["quarantined"] = use.plate_id.eq(data["quarantine"])
    use = use[["physical_well_id", "plate_id", "compound_entity_id", "compound_group_id", "dose_uM",
               "time_h", "control_role", "split", "fold", "smiles", "quarantined"]].sort_values("physical_well_id")
    unique(use, ["physical_well_id"])
    cell_lines = pd.read_csv("manifests/chemcpa_moe_v2/lines.tsv", sep="\t").cell_line_id.tolist()
    if not set(cell_lines) <= set(tables["cell_line"].Cell_ID_Cellosaur) or not set(data["liver"]) <= set(cell_lines):
        raise ValueError("Cell Line IDs disagree with metadata")

    out.mkdir(parents=True, exist_ok=False)
    use.to_csv(out / "samples.tsv", sep="\t", index=False)
    panel.to_csv(out / "genes.tsv", sep="\t", index=False)
    pd.Series(cell_lines, name="cell_line_id").to_csv(out / "lines.tsv", sep="\t", index=False)
    excluded[["sample", "plate", "drug", "split"]].to_csv(out / "outside_r6.tsv", sep="\t", index=False)
    counts = use.groupby(["split", "plate_id", "quarantined"]).agg(
        wells=("physical_well_id", "nunique"), groups=("compound_group_id", lambda x: x[x.ge(0)].nunique())
    )
    counts.to_csv(out / "sample_coverage.tsv", sep="\t")
    write_json(out / "receipt.json", dict(revision=revision, inputs=provenance,
               r6_split_sha256=sha256(SPLIT), r3_wells_sha256=sha256(WELLS),
               quarantine_reason="external_report_unverified_detail", canonical_plate_numbering="1_to_14",
               time_source="https://doi.org/10.1016/j.cell.2026.08.035",
               historical_validation="kept unused as in R6", outside_r6_at_primary_dose=len(excluded),
               control_well_rule="sort two DMSO_TF physical sample IDs lexically per plate",
               n_cell_lines=len(cell_lines), units="sample is one physical well; barcode identifies a cell"))
    print(counts.to_string(), flush=True)


def dose(value):
    records = ast.literal_eval(value)
    if len(records) != 1 or records[0][2] != "uM":
        return None
    return float(records[0][1])


def sample_bounds(path):
    fp = pq.ParquetFile(path)
    meta = fp.metadata
    column = next(i for i in range(meta.num_columns) if meta.schema.column(i).path == "sample")
    return path, [(rg, meta.row_group(rg).column(column).statistics.min,
                   meta.row_group(rg).column(column).statistics.max) for rg in range(meta.num_row_groups)]


def set_scan(samples, lines, panel, width, batch_cells):
    _SCAN.update(samples=samples, lines=lines, panel=panel, width=width, batch_cells=batch_cells,
                 sample_names=pa.array(list(samples)), line_names=pa.array(list(lines)))


def aggregate_batch(batch):
    mask = pc.and_(pc.is_in(batch.column("sample"), value_set=_SCAN["sample_names"]),
                   pc.is_in(batch.column("cell_line_id"), value_set=_SCAN["line_names"]))
    batch = batch.filter(mask)
    if not len(batch):
        return None
    samples = batch.column("sample").to_pylist()
    plates = batch.column("plate").to_pylist()
    lines = batch.column("cell_line_id").to_pylist()
    row_ids = []
    for sample, plate, line in zip(samples, plates, lines):
        first, declared_plate = _SCAN["samples"][sample]
        if plate != declared_plate:
            raise ValueError(f"Sample to plate mismatch: {sample}: {plate}, {declared_plate}")
        row_ids.append(first + _SCAN["lines"][line])

    genes = batch.column("genes")
    expr = batch.column("expressions")
    go = genes.offsets.to_numpy().astype(np.int64)
    eo = expr.offsets.to_numpy().astype(np.int64)
    go -= go[0]
    eo -= eo[0]
    if not np.array_equal(go, eo) or np.any(np.diff(go) < 2):
        raise ValueError("Unaligned gene and count lists or empty gene vectors")
    tokens = genes.flatten().to_numpy()
    counts = expr.flatten().to_numpy()
    markers = go[:-1]
    if not np.all(tokens[markers] == 1):
        raise ValueError("Expected the verified leading CLS marker at paired position zero")
    keep = np.ones(len(tokens), dtype=bool)
    keep[markers] = False
    values = counts[keep]
    indices = tokens[keep]
    width = _SCAN["width"]
    if np.any(indices < 3) or np.any(indices >= width):
        raise ValueError("Invalid or unmapped gene token")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.rint(values)):
        raise ValueError("Non-finite, negative, or non-integer count")
    cell = sp.csr_matrix((values.astype(np.int64), indices, go - np.arange(len(go))),
                         shape=(len(batch), width))
    library = np.asarray(cell.sum(axis=1)).ravel()
    use = library > 0
    groups, inverse = np.unique(np.asarray(row_ids)[use], return_inverse=True)
    selector = sp.csr_matrix((np.ones(use.sum(), dtype=np.int64), (inverse, np.flatnonzero(use))),
                             shape=(len(groups), len(batch)))
    selected = cell if _SCAN["panel"] is None else cell[:, _SCAN["panel"]]
    return groups, selector @ selected, np.bincount(inverse), selector @ library, int((~use).sum())


def read_shard(task):
    path, affected = task
    fp = pq.ParquetFile(path)
    result = {}
    dropped = 0
    for batch in fp.iter_batches(batch_size=_SCAN["batch_cells"], row_groups=affected, columns=COLS,
                                 use_threads=False):
        aggregated = aggregate_batch(batch)
        if aggregated is None:
            continue
        groups, counts, n_cells, totals, lost = aggregated
        dropped += lost
        for i, row in enumerate(groups):
            current = (counts[i], int(n_cells[i]), int(totals[i]))
            if row in result:
                old = result[row]
                current = (old[0] + current[0], old[1] + current[1], old[2] + current[2])
            result[row] = current
    return result, dropped


def counts(cfg, assets, stage, output):
    data = cfg["data"]
    root = Path(data["root"])
    meta = assets / "metadata"
    if stage == "final":
        from .freeze import verify

        frozen = verify(assets, cfg)
        if frozen["sample_manifest_sha256"] != sha256(meta / "samples.tsv"):
            raise ValueError("Frozen sample manifest changed")
        panel_path = assets / "development" / "final_panel.tsv"
        if frozen["panel_sha256"] != sha256(panel_path):
            raise ValueError("Frozen panel changed")
        panel = pd.read_csv(panel_path, sep="\t").token_id.to_numpy()
    else:
        panel = None

    samples = pd.read_csv(meta / "samples.tsv", sep="\t", keep_default_na=False)
    controls = samples.split.eq("control")
    roles = {"pilot": "train", "source": "train", "final": "test"}
    use = samples[~samples.quarantined & (samples.split.eq(roles[stage]) | (controls & (stage != "final")))]
    if stage == "pilot":
        eligible = use[use.split.eq("train")]
        plate = eligible.groupby("plate_id").compound_group_id.nunique().idxmax()
        pool = eligible[eligible.plate_id.eq(plate)]
        # Two folds with at least three groups per side, chosen from metadata only.
        ranked = pool.groupby("fold").compound_group_id.nunique().sort_values(ascending=False)
        groups = []
        for fold in ranked.index[:2]:
            groups += sorted(pool.loc[pool.fold.eq(fold), "compound_group_id"].unique())[:4]
        use = use[use.plate_id.eq(plate) & (use.compound_group_id.isin(groups) | use.split.eq("control"))]
    use = use.reset_index(drop=True)
    if use.empty:
        raise ValueError(f"No eligible metadata wells for {stage}")
    lines = pd.read_csv(meta / "lines.tsv", sep="\t").cell_line_id.tolist()
    grid = use.merge(pd.DataFrame({"cell_line_id": lines}), how="cross")
    n_genes = len(pd.read_csv(meta / "genes.tsv", sep="\t")) + 3
    well_index = {r.physical_well_id: (i * len(lines), r.plate_id) for i, r in enumerate(use.itertuples())}
    line_index = {line: i for i, line in enumerate(lines)}
    shards = sorted((root / "data").glob("train-*.parquet"))
    if not shards:
        raise ValueError("No Tahoe expression shards present")

    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    selected, shard_receipts = [], []
    with ThreadPoolExecutor(max_workers=data["workers"]) as executor:
        for path, ranges in executor.map(sample_bounds, shards):
            active = [row for row, low, high in ranges if any(low <= s <= high for s in well_index)]
            if active:
                etag = check_revision(root, path.relative_to(root), data["revision"])
                selected.append((path, active))
                shard_receipts.append(dict(shard=path.name, etag=etag, bytes=path.stat().st_size, row_groups=len(active)))
    print(f"{stage}: {len(use)} wells, {len(selected)}/{len(shards)} shards, footer scan {time.monotonic()-start:.1f}s", flush=True)
    pd.DataFrame(shard_receipts).to_csv(output / "shards.tsv", sep="\t", index=False)
    rows = {}
    cell_counts = np.zeros(len(grid), dtype=np.int64)
    library_counts = np.zeros(len(grid), dtype=np.int64)
    dropped = 0
    with ProcessPoolExecutor(max_workers=data["workers"], initializer=set_scan,
                             initargs=(well_index, line_index, panel, n_genes, data["batch_cells"])) as executor:
        # Limit completed results buffered in the parent when an early shard is slow.
        for offset in range(0, len(selected), data["workers"] * 3):
            for res, lost in executor.map(read_shard, selected[offset:offset + data["workers"] * 3]):
                dropped += lost
                for i, (vector, n, total) in res.items():
                    rows[i] = rows[i] + vector if i in rows else vector
                    cell_counts[i] += n
                    library_counts[i] += total
            if offset % (data["workers"] * 21) == 0:
                print(f"{stage}: {min(offset + data['workers'] * 3, len(selected))}/{len(selected)} shards, "
                      f"{cell_counts.sum():,} cells, {time.monotonic()-start:.1f}s", flush=True)
    width = n_genes if panel is None else len(panel)
    empty = sp.csr_matrix((1, width), dtype=np.int64)
    matrix = sp.vstack([rows.get(i, empty) for i in range(len(grid))], format="csr")
    if not rows:
        raise ValueError("No valid cells on planned well×line rows")
    summed = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(summed > library_counts) or (panel is None and not np.array_equal(summed, library_counts)):
        raise ValueError("Stored gene counts did not match validated full-gene library totals")
    grid["n_cells"] = cell_counts
    grid["mapped_library"] = library_counts
    grid["eligible_cell_count"] = cell_counts >= data["minimum_cells"]
    unique(grid, ["physical_well_id", "cell_line_id"])
    files = ["shards.tsv"]
    for role, group in grid.groupby("control_role", sort=True):
        row_file, count_file = f"{role}_rows.tsv", f"{role}_counts.npz"
        group.to_csv(output / row_file, sep="\t", index=False)
        sp.save_npz(output / count_file, matrix[group.index], compressed=False)
        files.extend([row_file, count_file])
    write_json(output / "receipt.json", dict(stage=stage, n_grid=len(grid), n_cells=int(cell_counts.sum()),
               n_eligible=int(grid.eligible_cell_count.sum()), n_genes=width, n_zero_library_cells=dropped,
               elapsed_seconds=round(time.monotonic()-start, 2), files={
                   file: sha256(output / file) for file in files}))


def main():
    parser = arguments(__doc__)
    parser.add_argument("--stage", choices=["metadata", "pilot", "source", "final"], required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.stage == "metadata":
        metadata(cfg, args.assets / "metadata")
    else:
        counts(cfg, args.assets, args.stage, args.assets / args.stage)


if __name__ == "__main__":
    main()
