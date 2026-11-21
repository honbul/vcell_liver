"""Inventory published liver repeat designs from sample metadata."""

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


ROOT = Path("/mnt/S1/sdata")
PUBLIC = ROOT / "pubdata"
BASE = ROOT / "workbench/Virtual_Cell/gssong"
R14 = BASE / "vcell_liver_r14_cross_organ"
INDEX = R14 / "metadata/v4"
PARTITION = BASE / "vcell_liver_r12_multiaxis/assets/partition"
GEO = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
EPA = "https://www.epa.gov/comptox-tools/downloadable-computational-toxicology-data"
TG = "https://dbarchive.biosciencedbc.jp/en/open-tggates/"
LICENSE_EPA = "Public GEO; EPA computational toxicology open-data policy; no separate accession-specific license located"


def digest(path):
    with path.open("rb") as file:
        d = hashlib.sha256()
        for block in iter(lambda: file.read(1 << 20), b""):
            d.update(block)
        return d.hexdigest()


@lru_cache(maxsize=None)
def parent(smiles):
    if not smiles or smiles in {"-666", "NA", "nan"}:
        return ""
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return ""
        molecule = rdMolStandardize.FragmentParent(molecule)
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        return Chem.MolToInchiKey(molecule).partition("-")[0]


def name_key(name):
    name = re.sub(r"(?:[ ,(-]+(?:hydrochloride|dihydrochloride|dihydrate|monohydrate|hydrate|"
                  r"sodium salt|potassium salt|ditosylate|d-tartrate)\)?)+$", "", name.casefold().strip())
    return re.sub(r"[^a-z0-9]", "", name)


def tab(path, **kwargs):
    return pd.read_csv(path, sep="\t", **kwargs)


class Inventory:
    def __init__(self):
        self.inputs = set()
        self.rows, self.overlaps = [], []
        source = PARTITION / "source_rows.tsv"
        split = PARTITION / "split_manifest.tsv"
        self.inputs.update((source, split))
        groups = set(tab(split, usecols=["compound_group_id"]).compound_group_id)
        if len(groups) != 203:
            raise ValueError(f"Expected 203 Tahoe groups, found {len(groups)}")
        catalog = tab(source, usecols=["compound_group_id", "compound_entity_id", "smiles"]).drop_duplicates()
        catalog = catalog.loc[catalog.compound_group_id.isin(groups)]
        self.keys = defaultdict(set)
        self.names = defaultdict(set)
        self.group_names = defaultdict(set)
        for row in catalog.itertuples():
            self.keys[parent(str(row.smiles))].add(row.compound_group_id)
            self.names[name_key(row.compound_entity_id)].add(row.compound_group_id)
            self.group_names[row.compound_group_id].add(row.compound_entity_id)
        self.keys.pop("", None)
        self.names.pop("", None)
        self.repeat_rows = []

    def repeat_counts(self, dataset, frame, condition, unit, plate=None, position=None, batch=None):
        cols = list(dict.fromkeys([unit] + [key for key in [plate, position, batch] if key]))
        agg = frame.groupby(condition, observed=True)[cols].nunique()
        key = dict(dataset=dataset)
        repeats = agg[unit].ge(2)
        key.update(conditions=len(agg), conditions_with_multiple_units=int(repeats.sum()),
                   compounds_with_multiple_units=agg.loc[repeats].reset_index().chemical.nunique())
        if plate:
            across = agg[plate].ge(2)
            key.update(conditions_across_plates=int(across.sum()),
                       compounds_across_plates=agg.loc[across].reset_index().chemical.nunique())
            if position:
                across_both = across & agg[position].ge(2)
                key.update(conditions_across_plate_and_position=int(across_both.sum()),
                           compounds_across_plate_and_position=agg.loc[across_both].reset_index().chemical.nunique())
        if batch:
            key.update(conditions_across_batch=int(agg[batch].ge(2).sum()))
        self.repeat_rows.append(key)
        return key

    def overlap(self, label, chemicals, scope):
        chemicals = chemicals[["chemical", "smiles"]].drop_duplicates().fillna("")
        structure, names = set(), set()
        for name, smiles in chemicals.itertuples(index=False, name=None):
            key = parent(smiles)
            matches = self.keys.get(key, set()) if key else self.names.get(name_key(name), set())
            if len(matches) != 1:
                continue
            group = next(iter(matches))
            method = "parent_connectivity" if key else "unique_normalized_name"
            (structure if key else names).add(group)
            self.overlaps.append(dict(dataset=label, scope=scope, chemical=name, tahoe_group=group, method=method,
                                      tahoe_names=" | ".join(sorted(self.group_names[group]))))
        return len(structure), len(names - structure)

    def add(self, label, biology, chemicals, timing, design, genes, license_note, source_urls, repeat_chemicals=None):
        structure, names = self.overlap(label, chemicals, "all")
        repeated = ("", "") if repeat_chemicals is None else self.overlap(label, repeat_chemicals, "cross_plate_position")
        self.rows.append(dict(dataset=label, liver_system=biology, chemical_labels=chemicals.chemical.nunique(),
                              concentration_and_time=timing, repeat_evidence=design,
                              measured_features=genes, tahoe_groups_structure=structure,
                              tahoe_groups_additional_name=names, tahoe_groups_any=structure + names,
                              tahoe_repeat_groups_structure=repeated[0], tahoe_repeat_groups_additional_name=repeated[1],
                              data_license=license_note,
                              primary_sources=source_urls))


def across_plates_and_positions(frame, condition):
    matched = frame.groupby(condition, observed=True)[["plate", "position"]].transform("nunique").ge(2).all(axis=1)
    return frame.loc[matched]


def lincs(inv):
    for release in ("GSE92742", "GSE70138"):
        folder = PUBLIC / "LINCS" / release
        metadata = next(folder.glob("*inst_info*"))
        perturbagens = sorted(folder.glob("*pert_info*.txt*"))[-1]
        if release == "GSE70138":
            perturbagens = next(folder.glob("*pert_info_2017*"))
        inv.inputs.update((metadata, perturbagens))
        names = {"rna_plate": "plate", "rna_well": "position", "det_plate": "plate", "det_well": "position"}
        records = []
        for chunk in tab(metadata, chunksize=200000, dtype=str):
            records.append(chunk.loc[chunk.cell_id.isin(["HEPG2", "HUH7", "PHH"])].rename(columns=names))
        h = pd.concat(records, ignore_index=True)
        h = h.loc[(h.pert_type == "trt_cp") & (h.pert_dose_unit != "-666") & (h.pert_time_unit != "-666")].copy()
        h["pert_dose"] = pd.to_numeric(h.pert_dose, errors="coerce")
        h["pert_time"] = pd.to_numeric(h.pert_time, errors="coerce")
        h = h.dropna(subset=["pert_dose", "pert_time"]).loc[lambda x: x.pert_dose.ge(0) & x.pert_time.ge(0)]
        h["unit"] = h.plate + ":" + h.position
        vehicle = pd.concat(records, ignore_index=True).loc[lambda x: x.pert_type == "ctl_vehicle"]
        h["vehicle_on_plate"] = h.plate.isin(vehicle.plate)
        p = tab(perturbagens, dtype=str).drop_duplicates("pert_id")
        h = h.merge(p[["pert_id", "canonical_smiles"]], how="left", on="pert_id", validate="many_to_one")
        h = h.rename(columns={"pert_iname": "chemical", "canonical_smiles": "smiles"})
        for line in ("HEPG2", "HUH7", "PHH"):
            label = f"{release} Level 4 {line}"
            frame = h.loc[h.cell_id.eq(line)]
            counts = inv.repeat_counts(label, frame, ["pert_id", "chemical", "pert_dose", "pert_dose_unit",
                                                        "pert_time", "pert_time_unit"], "unit", "plate", "position")
            for unit, f in frame.groupby("pert_dose_unit"):
                display_unit = "µM" if unit.casefold() == "um" else unit
                ranges = f"{f.pert_dose.min():g}–{f.pert_dose.max():g} {display_unit}"
                counts[f"dose_range_{unit}"] = ranges
            times = [f"{'/'.join(f'{v:g}' for v in sorted(x.pert_time.unique()))} {u}"
                     for u, x in frame.groupby("pert_time_unit")]
            counts["vehicle_matched_rows"] = int(frame.vehicle_on_plate.sum())
            counts["instance_rows"] = len(frame)
            frame_24 = frame.loc[frame.pert_time.eq(24) & frame.pert_time_unit.eq("h") & frame.vehicle_on_plate]
            condition_24 = ["pert_id", "chemical", "pert_dose", "pert_dose_unit"]
            inv.repeat_counts(f"{label} 24h with vehicle on plate", frame_24, condition_24, "unit", "plate", "position")
            design = (f"{counts['conditions_with_multiple_units']} repeated conditions; "
                      f"{counts['conditions_across_plate_and_position']} on distinct release plates and positions. "
                      "Plate and well are recorded; separate culture prep is unverified.")
            dose_time = "; ".join([", ".join(v for k, v in counts.items() if k.startswith("dose_range_")), ", ".join(times)])
            inv.add(label, line, frame, dose_time if len(frame) else "No eligible chemical instances in this release",
                    design, "978 measured landmarks; 12,328 Level 4 features with inference",
                    "Public GEO; accession-specific reuse license unverified", GEO + release + " | https://clue.io/GEO-guide",
                    across_plates_and_positions(frame_24, condition_24))
            for name, smiles in frame[["chemical", "smiles"]].drop_duplicates().fillna("").itertuples(index=False):
                if parent(smiles) in inv.keys and name_key(name):
                    inv.names[name_key(name)].update(inv.keys[parent(smiles)])


def heparg_bulk(inv):
    for acc in ("GSE284321", "GSE296254"):
        path = INDEX / f"{acc}_bulk_sample_index.tsv"
        inv.inputs.add(path)
        df = tab(path, dtype=str).fillna("")
        control = df.chemical.str.casefold().isin(["dmso", "dimethyl sulfoxide"]) | df.treatment.str.casefold().eq("vehicle control")
        vehicle_plates = set(df.loc[control, "plate"])
        treated = df.loc[~control & df.time.eq("24h") & df.chemical.ne("") & df.in_count_header.eq("True")].copy()
        treated = treated.loc[treated.plate.isin(vehicle_plates)]
        treated["position"] = treated.library.str.rsplit("_", n=1).str[-1]
        treated["dose"] = pd.to_numeric(treated.concentration.str.extract(r"^([0-9.eE+-]+)\s*[uµ]M$")[0],
                                       errors="coerce")
        treated["culture"] = treated.batch.str.extract(r"culture:(\S+)")[0]
        condition = ["chemical", "dose", "time"]
        counts = inv.repeat_counts(acc, treated, condition, "library", "plate", "position", "culture")
        counts["rows_with_dose_uM"] = int(treated.dose.notna().sum())
        counts["treated_rows_on_vehicle_plates"] = len(treated)
        counts["range_uM"] = [float(treated.dose.min()), float(treated.dose.max())]
        genes = ("22,537 probe IDs in R14 first plate-group index; 15,125 unique Tahoe genes in the unambiguous map"
                 if acc == "GSE284321" else "~20,000 genes (hWTv2, GEO description)")
        culture = (f"{counts['conditions_across_batch']} span distinct recorded culture labels."
                   if treated.culture.notna().any() else "Culture labels unavailable in the sample index.")
        design = (f"{counts['conditions_with_multiple_units']} exact nominal-dose repeated conditions; "
                  f"{counts['conditions_across_plate_and_position']} across different assay plates and positions. {culture}")
        treated = treated.assign(smiles="")
        inv.add(acc, "Differentiated HepaRG, bulk TempO-Seq", treated,
                f"{counts['range_uM'][0]:g}–{counts['range_uM'][1]:g} µM (observed); 24 h",
                design, genes, LICENSE_EPA, GEO + acc + " | " + EPA, across_plates_and_positions(treated, condition))


def other(inv, supplements):
    path = INDEX / "mix_seq_liver_assignments.tsv"
    inv.inputs.add(path)
    mix = tab(path).loc[lambda d: d.author_cell_quality.eq("normal")].copy()
    mix["chemical"] = mix.archive.str.replace(r"(_24hr)?_expt\d+\.zip$", "", regex=True)
    mix = mix.loc[mix.chemical.ne("DMSO")]
    inv.repeat_counts("MIX-seq acquired 24h liver expts 3+10", mix, ["ccle_name", "chemical"], "archive")
    mix_extra = supplements["mix_seq"]
    mix_min, mix_max = mix_extra["acquired_range_uM"]
    mismatch = ", ".join(mix_extra["unlinked_dose_labels"])
    inv.add("MIX-seq acquired expts 3+10", "Hep3B, JHH-5, Li-7, SNU-761", mix.assign(smiles=""),
            f"{mix_min:g}–{mix_max:g} µM; 24 h; dose unlinked for {mismatch}", "Trametinib is shared across acquired "
            f"experiments with disjoint hepatic lines. Full supplementary design has {mix_extra['repeated_liver_experiment_conditions']} "
            "cross-experiment conditions matching a liver line, chemical, dose and time.",
            "32,738 annotated features in two checked archives (10x whole-transcriptome)",
            "CC BY 4.0", "https://doi.org/10.6084/m9.figshare.10298696.v3 | https://doi.org/10.1038/s41467-020-17440-w")

    path = PUBLIC / "TGGATEs/attributes/open_tggates_cel_file_attribute.zip"
    inv.inputs.add(path)
    with ZipFile(path) as bundle, bundle.open("open_tggates_cel_file_attribute.csv") as stream:
        tg = pd.read_csv(stream, dtype=str, encoding="cp932", encoding_errors="replace")
    tg = tg.loc[tg.SPECIES.eq("Human") & tg.EXP_TEST_TYPE.eq("in vitro") & tg.DOSE_LEVEL.ne("Control")].copy()
    tg = tg.rename(columns={"COMPOUND_NAME": "chemical"})
    counts = inv.repeat_counts("Open TG-GATEs PHH", tg, ["chemical", "DOSE", "DOSE_UNIT", "SACRIFICE_PERIOD"],
                               "BARCODE", batch="EXP_ID")
    inv.add("Open TG-GATEs PHH", "Primary human hepatocytes, bulk microarray", tg.assign(smiles=""),
            "Dose-tier low/medium/high; 2, 8, 24 h",
            f"{counts['conditions_with_multiple_units']} conditions have multiple array barcodes; release design "
            "has paired assay samples. Physical well position unavailable in attribute table.",
            "54,675 Affymetrix U133 Plus 2.0 probe sets (>20,000 genes)",
            "CC BY-SA 2.1 Japan + NBDC additional terms",
            TG + "desc.html | " + TG + "lic.html | https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-798")

    path = INDEX / "heparg_single_cell_samples.tsv"
    inv.inputs.add(path)
    sc = tab(path)
    sc["panel_label"] = sc.reported_sample_label.str.extract(r"^(S\d+)_")[0]
    controls = sc.chemical.str.casefold().eq("dimethyl sulfoxide")
    treated = sc.loc[~controls]
    counts = inv.repeat_counts("GSE299113 GEO labels", treated, ["chemical", "dose", "time"], "accession", batch="batch")
    counts["vehicle_rows_by_batch"] = str(sc.loc[controls].groupby("batch").size().to_dict())
    heparg = supplements["heparg"]
    inv.add("GSE299113", "HepaRG single-cell TempO-LINC", treated.assign(smiles=""),
            "Seven chemicals × three agent-specific doses; 24 h",
            f"Of {heparg['positive_dose_conditions']} positive-dose conditions, {heparg['conditions_with_multiple_exposure_ids']} "
            "have multiple H5AD EPA sample IDs despite multiple GEO accessions per condition. DMSO in EPA0001 only.",
            f"{heparg['targeted_probes']:,} targeted probes (H5AD var); mean 4,837 detected genes per cell (paper)",
            "Public GEO; accession-specific reuse license unverified",
            GEO + "GSE299113 | https://doi.org/10.1016/j.namjnl.2025.100046")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--supplements", type=Path, required=True)
    args = parser.parse_args()
    inv = Inventory()
    lincs(inv)
    heparg_bulk(inv)
    supplements = json.loads(args.supplements.read_text())
    inv.inputs.update((args.supplements, R14 / "raw/mix_seq/receipt.json", R14 / "raw/heparg_single_cell/receipt.json",
                       R14 / "raw/mix_seq/Supplementary Tables.xlsx"))
    other(inv, supplements)
    args.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(inv.rows).to_csv(args.out / "liver_repeat_inventory.tsv", sep="\t", index=False)
    pd.DataFrame(inv.repeat_rows).fillna("").to_csv(args.out / "metadata_repeat_counts.tsv", sep="\t", index=False)
    pd.DataFrame(inv.overlaps).drop_duplicates().to_csv(args.out / "tahoe_overlap.tsv", sep="\t", index=False)
    receipt = dict(protocol_sha256=digest(args.protocol), expression_read=False, rdkit=rdBase.rdkitVersion,
                   input_sha256={str(path): digest(path) for path in sorted(inv.inputs)},
                   output_sha256={f.name: digest(f) for f in sorted(args.out.iterdir())},
                   overlap_notes="Parent structure match when released SMILES is valid; otherwise unique normalized "
                   "name (including LINCS aliases verified against Tahoe parent structure). Name matches are preliminary. "
                   "Cross-plate-and-position scope restricts LINCS to 24h with vehicle on the same released plate.")
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
