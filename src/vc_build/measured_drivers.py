"""Register measured coding mutations and absolute copy-number events for transfer."""

import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.reader.strings import read_string_table

from .context_features import COPY, DRIVER_SHA256, MUTATION, load_drivers
from .function_data import alias_token, new_output
from .io import publish_json, sha256
from .v0plus_licob import ASSET_HASHES, checked_registration
from .v0plus_priors import gene_columns
from .yang_hcc_external import require_hash

INPUT_HASHES = {
    "limore_s2": "173d13d3ddc5d1cb3dc9268b44d4fff194a5f98ca04da8ea2ea4493b58842116",
    "limore_s3": DRIVER_SHA256,
    "licob_node": ASSET_HASHES["node_archive"],
    "depmap_mutations": "ba2a16d408d9220615fb08396c8d3879cb7e3e7ad97d3ce891f3851792b80326",
    "depmap_copy": "5f530c797bf12bce68cccc3e763b29491c368bbbced8971733bed807a13b8c13",
}
CODING_EFFECT = r"\|(?:missense_variant|nonsense|stop_gained|stop_lost|frameshift_variant|inframe_deletion|inframe_insertion|splice_donor_variant|splice_acceptor_variant)(?:,|$)"
LIMORE_EFFECTS = {
    "nonsynonymous SNV", "nonsense", "stoploss", "frameshift insertion", "frameshift deletion",
    "splicing", "nonframeshift insertion", "nonframeshift deletion",
}
LICOB_EFFECTS = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
}


def events(cells, assayed, positives):
    """Keep an absent assay distinct from a measured profile with no qualifying calls."""
    frame = pd.DataFrame(np.nan, index=pd.Index(cells, name="cell_id"), columns=sorted(MUTATION), dtype=np.float32)
    frame.loc[frame.index.intersection(assayed)] = 0.0
    for cell, gene in positives:
        frame.loc[cell, gene] = 1.0
    return frame.add_prefix("coding_")


def depmap_coding(path, cells):
    present, positives = set(), set()
    columns = ["ModelID", "IsDefaultEntryForModel", "HugoSymbol", "MolecularConsequence"]
    for frame in pd.read_csv(path, usecols=columns, chunksize=100_000):
        selected = frame[frame.ModelID.isin(cells) & frame.IsDefaultEntryForModel.eq("Yes")]
        present.update(selected.ModelID)
        calls = selected[selected.HugoSymbol.isin(MUTATION) & selected.MolecularConsequence.str.contains(
            CODING_EFFECT, na=False, regex=True,
        )]
        positives.update(zip(calls.ModelID, calls.HugoSymbol))
    return events(cells, present, positives)


def depmap_copy(path, cells):
    columns = gene_columns(path, set(COPY))
    absolute = pd.read_csv(path, usecols=["Unnamed: 0", *columns], index_col="Unnamed: 0").rename(columns=columns)
    absolute = absolute.reindex(index=cells, columns=sorted(COPY)).astype(np.float32)
    if (absolute < 0).any().any():
        raise ValueError("DepMap 24Q4 absolute copy numbers must be nonnegative")
    gain = absolute.gt(4).where(absolute.notna()).add_prefix("gain_")
    loss = absolute.eq(0).where(absolute.notna()).add_prefix("loss_")
    return pd.concat([gain, loss], axis=1).astype(np.float32)


def limore_coding(path, roster):
    with path.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=True)
        sheet = book["Somatic mutations"]
        rows = sheet.iter_rows(min_row=2, values_only=True)
        header = next(rows)
        if tuple(header[:7]) != ("Sequencing", "Chromosome", "Start", "Ref", "Alt", "Gene", "Variant type"):
            raise ValueError("LIMORE S2 variant schema changed")
        aliases = {alias_token(value): i for i, value in enumerate(header[7:], 7) if isinstance(value, str)}
        if len(aliases) != 81:
            raise ValueError("Unexpected LIMORE S2 WGS roster")
        cells = [(cell, aliases[alias_token(label)]) for cell, label in roster[["cell_id", "source_cell_label"]]
                 .drop_duplicates().itertuples(index=False, name=None)]
        positives = set()
        for row in rows:
            gene, effect = row[5:7]
            if gene in MUTATION and effect in LIMORE_EFFECTS:
                for cell, column in cells:
                    if not isinstance(row[column], int) or row[column] < 0:
                        raise ValueError("Expected LIMORE S2 nonnegative call codes")
                    if row[column] > 0:
                        positives.add((cell, gene))
        book.close()
    ids = sorted(cell for cell, _ in cells)
    if len(ids) != len(set(ids)):
        raise ValueError("LIMORE registered model aliases are ambiguous")
    return events(ids, ids, positives)


def sheet_rows(archive, member, strings):
    """Read a single CRC-checked worksheet, retaining real column addresses for sparse cells."""
    with archive.open(member) as stream:
        nodes = ElementTree.iterparse(stream, events=("start", "end"))
        container = None
        for event, element in nodes:
            if event == "start" and element.tag.endswith("}sheetData"):
                container = element
            if event != "end" or not element.tag.endswith("}row"):
                continue
            values = {}
            number = int(element.attrib["r"])
            for cell in element:
                value = cell.findtext("{*}v")
                if value is not None:
                    column = cell.attrib["r"].removesuffix(str(number))
                    values[column] = strings[int(value)] if cell.get("t") == "s" else value
            yield number, values
            if container is not None:
                container.remove(element)
        # ZipExtFile verifies CRC when the worksheet is consumed through EOF.


def licob_coding(path, models):
    observed, positives, copy_models = set(), set(), set()
    sheets = {name: f"xl/worksheets/sheet{index}.xml" for name, index in {"mutations": 5, "copy": 6}.items()}
    evidence = {}
    with ZipFile(path) as archive:
        with archive.open("xl/sharedStrings.xml") as stream:
            strings = read_string_table(stream)
        for kind, member in sheets.items():
            info = archive.getinfo(member)
            evidence[kind] = {"member": member, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
            for number, row in sheet_rows(archive, member, strings):
                if number == 2:
                    expected = {"A": "Hugo_Symbol", "F": "Variant_Classification", "K": "Tumor_Sample_Barcode"}
                    if kind == "mutations" and any(row.get(col) != name for col, name in expected.items()):
                        raise ValueError("Unexpected LICOB D worksheet schema")
                    if kind == "copy":
                        if row.get("A") != "Hugo_Symbol" or row.get("B") != "Gene ID":
                            raise ValueError("Unexpected LICOB E worksheet schema")
                        copy_models = set(row.values()) & set(models)
                if kind == "mutations" and number > 2 and row.get("K") in models:
                    observed.add(row["K"])
                    if row.get("A") in MUTATION and row.get("F") in LICOB_EFFECTS:
                        positives.add((row["K"], row["A"]))
    return events(models, observed, positives), evidence | {"copy_header_models": len(copy_models)}


def counts(frame):
    return {name: {"measured": int(column.notna().sum()), "positive": int(column.sum())}
            for name, column in frame.items()}


def prepare(options, train, limore, licob):
    checked_registration(options.licob)
    for attribute, digest in INPUT_HASHES.items():
        require_hash(getattr(options, attribute), digest)
    fit_cells = sorted(train.cell_id.unique())
    limore_cells = sorted(limore.cell_id.unique())
    licob_cells = sorted(licob.cell_id.unique())
    if len(set(fit_cells + limore_cells + licob_cells)) != len(fit_cells + limore_cells + licob_cells):
        raise ValueError("Measured driver study identifiers overlap")
    lc_seq, licob_evidence = licob_coding(options.licob_node, licob_cells)
    lim_copy, missing = load_drivers(options.limore_s3, limore)
    if missing or set(lim_copy.index) != set(limore_cells):
        raise ValueError("LIMORE S3 copy calls must cover the registered 50 models")
    data = {
        "source_coding": depmap_coding(options.depmap_mutations, fit_cells),
        "source_copy": depmap_copy(options.depmap_copy, fit_cells),
        "limore_coding": limore_coding(options.limore_s2, limore),
        "limore_copy": lim_copy.loc[:, lambda table: ~table.columns.str.startswith("seq_")].astype(np.float32),
        "licob_coding": lc_seq,
    }
    with new_output(options.out) as temp:
        artifacts = {}
        for name, frame in data.items():
            path = temp / f"{name}.parquet"
            frame.to_parquet(path)
            artifacts[path.name] = sha256(path)
        publish_json(temp / "receipt.json", {
            "sources": INPUT_HASHES, "artifacts": artifacts, "licob_recoverable_genomics": licob_evidence,
            "support": {key: {"models": len(frame), "assayed": int(frame.notna().any(axis=1).sum()),
                              "features": counts(frame)} for key, frame in data.items()},
        })


def read(directory):
    path = Path(directory) / "receipt.json"
    document = json.loads(path.read_text())
    names = ("source_coding", "source_copy", "limore_coding", "limore_copy", "licob_coding")
    if document["sources"] != INPUT_HASHES or set(document["artifacts"]) != {f"{n}.parquet" for n in names}:
        raise ValueError("Measured driver input registration changed")
    tables = {}
    for name in names:
        filename = name + ".parquet"
        require_hash(path.parent / filename, document["artifacts"][filename])
        tables[name] = pd.read_parquet(path.parent / filename)
    return tables, sha256(path)
