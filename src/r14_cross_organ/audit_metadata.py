"""Link assay metadata and count columns without opening expression values."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

import openpyxl

from .acquire import checksums, read_json


def write_table(path, rows):
    if not rows:
        raise ValueError(f"No metadata rows for {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sciplex(raw, out):
    source = next(raw.glob("*hashTable_metadata.txt.gz"))
    with gzip.open(source, "rt") as stream:
        samples = list(csv.DictReader(stream, delimiter="\t"))
    if len({x["Combo"] for x in samples}) != len(samples):
        raise ValueError("Sci-Plex released hash IDs are not unique")
    rows = [x for x in samples if x["time_point"] == "24"]
    write_table(out / "sciplex3_wells_24h.tsv", rows)
    by_line = {}
    for line in sorted({x["cell_type"] for x in rows}):
        all_wells = [x for x in rows if x["cell_type"] == line]
        held = [x for x in all_wells if x["vehicle"] == "FALSE"]
        rep = defaultdict(set)
        for record in held:
            rep[(record["catalog_number"], record["dose"])].add(record["replicate"])
        by_line[line] = dict(wells=len(all_wells), vehicle_wells=len(all_wells) - len(held),
                             catalog_compounds=len({x["catalog_number"] for x in held}),
                             dose_labels=sorted({x["dose"] for x in held}, key=float),
                             compound_dose_pairs_in_both_replicates=sum(v == {"rep1", "rep2"} for v in rep.values()))
    return dict(time_counts=dict(Counter(x["time_point"] for x in samples)), design_by_line=by_line)


def geo_samples(path):
    columns = {"title": "!Sample_title = ", "batch": "!Sample_characteristics_ch1 = batch: "}
    rows = []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                rows.append(dict(accession=line.removeprefix("^SAMPLE = ")))
            for name, prefix in columns.items():
                if line.startswith(prefix):
                    rows[-1][name] = line.removeprefix(prefix)
    return rows


def heparg_single_cell(raw, out):
    rows = []
    for sample in geo_samples(next(raw.glob("*_family.soft.gz"))):
        chemical, dose, identity, time, label = sample["title"].split("|")
        rows.append(dict(accession=sample["accession"], batch=sample["batch"], chemical=chemical,
                         dose=dose, identity=identity, time=time, reported_sample_label=label))
    write_table(out / "heparg_single_cell_samples.tsv", rows)
    return dict(samples=len(rows), batches=dict(Counter(x["batch"] for x in rows)),
                treated_chemicals=len({x["chemical"] for x in rows if x["chemical"] != "Dimethyl sulfoxide"}),
                dmso_by_batch=dict(Counter(x["batch"] for x in rows if x["chemical"] == "Dimethyl sulfoxide")))


def spreadsheet_rows(book):
    rows = iter(book["Metadata"].values)
    header = next(row for row in rows if row[0] == "*library name")
    positions = {field: header.index(field) for field in set(header) - {None}}
    for row in rows:
        if row[2] == "RNA-Seq" and row[0]:
            yield {name: row[index] for name, index in positions.items()}


def index_bulk(book, accession):
    records = []
    for row in spreadsheet_rows(book):
        library = row["*library name"]
        if accession == "GSE284321":
            chemical, concentration, identity = (row[key] for key in
                                                 ("chemical name", "chemical concentration", "chemical sample ID"))
            plate, batch, timepoint = row["assay plate"], "", row["exposure time"]
            file = accession + "_" + row["processed data file"]
        else:
            key = re.fullmatch(r"(.+?)\s+([0-9.eE+-]+uM)(?:\s*\|\s*(.+))?", str(row["description"]).strip())
            chemical, concentration, identity = key.groups() if key else ("", "", "")
            plate, batch, timepoint = library.rsplit("_", 1)[0], row["batch"], "24h"
            file = "GSE296254_tox21-counts-1.tsv.gz"
        records.append(dict(library=library, plate=plate, batch=batch, time=timepoint,
                            chemical=chemical, concentration=concentration, identity=identity,
                            treatment=row["treatment"], count_file=file))
    return records


def heparg_bulk(raw, out, accession):
    workbook = next(raw.glob("*metadata*.xlsx"))
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    records = index_bulk(book, accession)
    files = defaultdict(set)
    for row in records:
        files[row["count_file"]].add(row["library"])
    if sum(map(len, files.values())) != len(records):
        raise ValueError(f"Duplicated metadata sample IDs for {accession}")
    published = {str(a): str(b).lower() for a, b in book["MD5 Checksums"].values if a and b}
    matrix_only, metadata_only, verified, matrix_columns = 0, 0, 0, 0
    for name, expected in sorted(files.items()):
        file = raw / name
        if name.removeprefix(accession + "_") in published:
            if checksums(file)[1] != published[name.removeprefix(accession + "_")]:
                raise ValueError(f"Publisher MD5 mismatch for {name}")
            verified += 1
        with gzip.open(file, "rt") as stream:
            header = next(csv.reader(stream, delimiter="," if name.endswith(".csv.gz") else "\t"))
        gene_columns = ["probe_id"] if accession == "GSE284321" else ["entrez_id", "probe_id", "gene"]
        if header[:len(gene_columns)] != gene_columns:
            raise ValueError(f"Unexpected gene columns in {name}")
        sample_ids = header[len(gene_columns):]
        count_ids = set(sample_ids)
        if len(count_ids) != len(sample_ids):
            raise ValueError(f"Duplicate matrix sample IDs in {name}")
        matrix_columns += len(count_ids)
        matrix_only += len(count_ids - expected)
        metadata_only += len(expected - count_ids)
        for row in records:
            if row["count_file"] == name:
                row["in_count_header"] = row["library"] in count_ids
    book.close()
    write_table(out / f"{accession}_bulk_sample_index.tsv", records)
    joined = [x for x in records if x["in_count_header"]]
    vehicles = [x for x in joined if str(x["chemical"]).casefold() in {"dimethyl sulfoxide", "dmso"}
                or str(x["treatment"]).casefold() == "vehicle control"]
    vehicle_libraries = {x["library"] for x in vehicles}
    vehicle_plates = {x["plate"] for x in vehicles}
    candidates = [x for x in joined if x["time"] == "24h" and x["chemical"] not in ("", "NA", None)
                  and x["library"] not in vehicle_libraries]
    return dict(metadata_sample_rows=len(records), count_columns=matrix_columns, unmatched_count_columns=matrix_only,
                unmatched_metadata_rows=metadata_only, count_files=len(files), publisher_md5_verified=verified,
                joined_vehicle_rows=len(vehicles), joined_24h_candidate_names=len({x["chemical"] for x in candidates}),
                joined_24h_candidates_on_vehicle_plate=sum(x["plate"] in vehicle_plates for x in candidates))


def mix_seq(raw, out):
    book = openpyxl.load_workbook(raw / "Supplementary Tables.xlsx", read_only=True, data_only=True)
    sheet_names = list(book.sheetnames)
    lines = {exp: [] for exp in ("3", "10")}
    for name, depmap, label in list(book["Supplementary Table 1"].values)[1:]:
        expts = {v.strip() for v in str(label).split(",")}
        for exp in expts.intersection(lines):
            if name and depmap:
                lines[exp].append(dict(ccle_name=name, depmap_id=depmap))
    conditions = defaultdict(list)
    for row in list(book["Supplementary Table 2"].values)[1:]:
        exp, drug, time, n_lines, cells = row[:5]
        if str(exp) in lines and time == 24:
            conditions[str(exp)].append(dict(perturbation=drug, assayed_lines=n_lines, reported_cells=cells))
    book.close()
    rows, liver_support = [], []
    for file in sorted(raw.glob("*.zip")):
        exp = file.stem.rsplit("expt", 1)[1]
        liver = {v["ccle_name"] for v in lines[exp] if v["ccle_name"].endswith("_LIVER")}
        with ZipFile(file) as bundle:
            rows.extend(dict(archive=file.name, entry=entry.filename, uncompressed_bytes=entry.file_size)
                        for entry in bundle.infolist() if not entry.is_dir())
            assignments = next(name for name in bundle.namelist() if name.endswith("/classifications.csv"))
            with bundle.open(assignments) as stream:
                observed = Counter((row["singlet_ID"], row["DepMap_ID"], row["cell_quality"])
                                   for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"))
                                   if row["singlet_ID"] in liver)
            liver_support.extend(dict(experiment=exp, archive=file.name, ccle_name=line, archived_depmap_id=model,
                                      author_cell_quality=quality, cells=count)
                                 for (line, model, quality), count in sorted(observed.items()))
    write_table(out / "mix_seq_archive_index.tsv", rows)
    write_table(out / "mix_seq_cell_lines.tsv", [dict(experiment=exp, **line)
                                                for exp, entries in lines.items() for line in entries])
    write_table(out / "mix_seq_liver_assignments.tsv", liver_support)
    return dict(archives=len({r["archive"] for r in rows}), archive_entries=len(rows), design_tables=sheet_names,
                lines_per_experiment={exp: len(entries) for exp, entries in lines.items()},
                liver_lines={exp: [line for line in entries if line["ccle_name"].endswith("_LIVER")]
                             for exp, entries in lines.items()}, conditions_24h=conditions)


def shared_bulk_probes(raw, tox21, tahoe, out):
    first_plate = raw / "GSE284321_heparg_pg1_count_data.csv.gz"
    with gzip.open(first_plate, "rt") as stream:
        next(stream)
        probes = {line.partition(",")[0].strip('"') for line in stream}
    with tahoe.open() as stream:
        genes = defaultdict(set)
        for row in csv.DictReader(stream, delimiter="\t"):
            genes[row["gene_symbol"]].add(row["ensembl_id"])
    matched = []
    with gzip.open(tox21 / "GSE296254_tox21-counts-1.tsv.gz", "rt") as stream:
        next(stream)
        for line in stream:
            entrez, probe, symbol, _ = line.split("\t", maxsplit=3)
            ensembl = genes.get(symbol, set()) - {"", "NA"}
            if probe in probes and len(ensembl) == 1:
                matched.append(dict(probe_id=probe, entrez_id=entrez, gene_symbol=symbol,
                                    ensembl_id=next(iter(ensembl))))
    pairs = Counter(row["probe_id"] for row in matched)
    unique = [row for row in matched if pairs[row["probe_id"]] == 1]
    write_table(out / "heparg_bulk_preview_probe_map.tsv", unique)
    return dict(heparg_bulk_plate_group=first_plate.name, plate_group_unique_probes=len(probes),
                exact_shared_probes_with_unique_tahoe_gene=len(unique),
                mapped_tahoe_ensembl_ids=len({row["ensembl_id"] for row in unique}),
                tahoe_metadata_sha256=checksums(tahoe)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tox21", type=Path, required=True)
    parser.add_argument("--tahoe-genes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results = {"sciplex3": sciplex(args.raw / "sciplex3", args.out),
               "heparg_single_cell": heparg_single_cell(args.raw / "heparg_single_cell", args.out),
               "GSE284321": heparg_bulk(args.raw / "heparg_bulk", args.out, "GSE284321"),
               "GSE296254": heparg_bulk(args.tox21, args.out, "GSE296254"),
               "mix_seq": mix_seq(args.raw / "mix_seq", args.out),
               "heparg_bulk_shared_probe_preview": shared_bulk_probes(
                   args.raw / "heparg_bulk", args.tox21, args.tahoe_genes, args.out)}
    receipts = sorted(args.raw.glob("*/receipt.json"))
    if len(receipts) != 4 or any(read_json(p)["count_values_read"] for p in receipts):
        raise ValueError("Expected the four complete download receipts")
    result = dict(sources=results, input_receipts={str(p): checksums(p)[0] for p in receipts},
                  metadata_files={p.name: checksums(p)[0] for p in args.out.glob("*.tsv")},
                  count_values_read=False)
    with (args.out / "summary.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
