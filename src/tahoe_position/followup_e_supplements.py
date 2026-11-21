"""Read published MIX-seq design and the HepaRG single-cell metadata groups."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import h5py
import openpyxl


def mix_seq(raw):
    book = openpyxl.load_workbook(raw / "Supplementary Tables.xlsx", read_only=True, data_only=True)
    lines = defaultdict(set)
    for name, _, experiments in list(book["Supplementary Table 1"].values)[1:]:
        if name and name.endswith("_LIVER"):
            for experiment in map(str.strip, str(experiments).split(",")):
                lines[experiment].add(name)
    drugs = {row[0]: row[1] for row in list(book["Supplementary Table 3"].values)[1:]}
    drugs["BRD3379"] = drugs["BRD-3379"]
    experiments = defaultdict(set)
    acquired_doses = set()
    unlinked = set()
    for experiment, name, hour, *_ in list(book["Supplementary Table 2"].values)[1:]:
        dose = drugs.get(name)
        if name and name not in {"DMSO", "Untreated"} and not name.startswith("sg"):
            for line in lines[str(experiment)]:
                experiments[line, name, dose, hour].add(str(experiment))
            if str(experiment) in {"3", "10"} and hour == 24:
                (unlinked if dose is None else acquired_doses).add(name if dose is None else dose)
    book.close()
    probes = {}
    for archive in ("Dabrafenib_24hr_expt3.zip", "Afatinib_expt10.zip"):
        with ZipFile(raw / archive) as bundle:
            with bundle.open(next(name for name in bundle.namelist() if name.endswith("/genes.tsv"))) as stream:
                probes[archive] = sum(1 for _ in stream)
    return dict(repeated_liver_experiment_conditions=sum(len(v) > 1 for v in experiments.values()),
                acquired_range_uM=[min(acquired_doses), max(acquired_doses)],
                unlinked_dose_labels=sorted(unlinked), checked_gene_catalog_sizes=probes)


def heparg(raw):
    path = next(raw.glob("*.h5ad"))
    with h5py.File(path, "r") as infile:
        obs = infile["obs"]

        def categories(label):
            col = obs[label]
            return [value.decode() for value in col["categories"][:][col["codes"][:]]]

        sample_by_condition, dmso = defaultdict(set), defaultdict(set)
        for chemical, run, dose, sample, time in zip(
            categories("chem_name"), categories("biospyder_run"),
            obs["conc"][:], obs["epa_sample_id"][:], obs["timeh"][:],
        ):
            if dose > 0:
                sample_by_condition[chemical, dose, time].add(int(sample))
            elif chemical == "Dimethyl sulfoxide":
                dmso[run].add(int(sample))
        gene_count = len(infile["var/_index"])
    return dict(positive_dose_conditions=len(sample_by_condition),
                conditions_with_multiple_exposure_ids=sum(len(v) > 1 for v in sample_by_condition.values()),
                dmso_exposure_ids_by_run={k: sorted(v) for k, v in dmso.items()}, targeted_probes=gene_count,
                input_file=path.name, access="H5AD obs/var only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = dict(mix_seq=mix_seq(args.raw / "mix_seq"), heparg=heparg(args.raw / "heparg_single_cell"))
    with args.out.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
