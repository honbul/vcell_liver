"""Evaluate RNA-to-target dependency transfer on public tissue-held-out organoids."""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook

from .evaluate import score_across_cells
from .expanded_panel_cv import RECEIPTS
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .pdo_transfer_cv import device_cuda, domain_kernels
from .v0plus_cv import _rna, molecular_inputs
from .v0plus_features import rna_values
from .v0plus_score import crossed_patient_drug_ci
from .yang_hcc_external import require_hash

META = "7e83b7551c9c272de0bc0455ea9a0e7803a8043d15d9cfedf5a1c3e609a2a6e2"
FITNESS = "7d44232fef08a39750861037886a1924f9780043685f5e78e1f08682c566c6e2"
CHRONOS = "fa44b6a7cefe8748aaa2bef21bd49aa105f7ebfe819510da3022266d88f24d51"
MODELS = "9dbb9de8805696c1345816ab07edd23fb4fd95e117739f3c5c3b1cf062c1233b"
TARGETS = "7e1f65e1aa32b8a99f2de4b9d5b9c9a0e94a786d786534c278e1c8239d9f475e"
TISSUES = ("Colorectal", "Oesophageal")
FILES = ("rna.parquet", "roster.parquet", "dependency_2d.parquet", "roster_2d.parquet")
REFERENCE = ("target_rna", "depmap_2d", "sanger_shuffled")
PRIMARY = "equal_2d_3d"
ARMS = (*REFERENCE, "sanger_3d", PRIMARY)
RANK_TOLERANCE = 0.0
MIN_HELD_DONORS = 30


def worksheet(archive, table, sheet):
    data = archive.read(f"supplementary_table_{table}_revision.xlsx")
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        rows = workbook[sheet].values
        columns = next(rows)
        return pd.DataFrame(rows, columns=columns)
    finally:
        workbook.close()


def unique_symbols(frame, symbol):
    return frame[frame[symbol].notna() & ~frame[symbol].duplicated(keep=False)].set_index(symbol)


def panel_from_core(core, target_file):
    receipt = core / "receipt.json"
    require_hash(receipt, RECEIPTS["core"])
    require_hash(target_file, TARGETS)
    artifacts = json.loads(receipt.read_text())["artifacts"]
    for relative in ("train/rows.parquet", "train/rna.parquet"):
        require_hash(core / relative, artifacts[relative]["sha256"])
    compounds = pd.read_parquet(core / "train/rows.parquet", columns=["compound_id"]).compound_id.unique()
    all_targets = json.loads(target_file.read_text())
    selected = sorted({gene for compound in compounds for gene in all_targets.get(compound, ())})
    return selected, _rna(core / "train/rna.parquet")


def organoid_inputs(meta, fitness, targets):
    with ZipFile(meta) as source:
        measured = worksheet(source, 2, "data_availability")
        raw_rna = worksheet(source, 3, "RNAseq_TPM")
    measured = measured[measured.sample_ID.notna()]
    rna = unique_symbols(raw_rna, "gene").drop(columns="ENSEMBL_ID").T
    blocked = set(measured.loc[measured.overlap_Broad_DepMap.eq("Yes"), "individual_ID"])
    roster = measured[measured.CRISPR_available.eq("Yes") & measured.sample_ID.isin(rna.index)].copy()
    retained = roster[~roster.individual_ID.isin(blocked)].copy()
    with ZipFile(fitness) as source:
        (member,) = [name for name in source.namelist() if "_fold_change_values_" in name]
        with source.open(member) as stream:
            reader = csv.reader(io.TextIOWrapper(stream), delimiter="\t")
            names, model, lab, qc, gene_header = (next(reader) for _ in range(5))
            if [row[0] for row in (names, model, lab, qc, gene_header)] != [
                    "model_name", "model_id", "source", "qc_pass", "gene_id"]:
                raise ValueError("Unexpected Sanger fitness header")
            if len(names) != 165 or set(lab[3:]) != {"Sanger"} or set(qc[3:]) != {"TRUE"}:
                raise ValueError("Unexpected Sanger QC model roster")
            rows = [row[1:] for row in reader if row[1] in targets]
    values = pd.DataFrame(rows, columns=["symbol", "ensembl", *names[3:]])
    dependency = -unique_symbols(values, "symbol").drop(columns="ensembl").replace(["", "NA"], np.nan).astype(float).T
    retained = retained[retained.sanger_ID.isin(dependency.index)]
    if not retained.sample_ID.is_unique or not retained.sanger_ID.is_unique:
        raise ValueError("An organoid model maps to multiple RNA or knockout rows")
    mapping = retained.set_index("sanger_ID").sample_ID
    dependency = dependency.loc[mapping.index].rename(index=mapping.to_dict()).sort_index()
    fields = {"individual_ID": "donor", "primary_tumour_type": "tissue"}
    metadata = retained.set_index("sample_ID").rename(columns=fields)[list(fields.values())].sort_index()
    if metadata.donor.isna().any() or metadata.groupby("donor").tissue.nunique().gt(1).any():
        raise ValueError("Organoid donor linkage lacks a unique tissue")
    return rna.loc[dependency.index].astype(np.float32), dependency, metadata, len(roster) - len(retained)


def depmap_inputs(path, model_path, rna, targets):
    require_hash(path, CHRONOS)
    require_hash(model_path, MODELS)
    models = pd.read_csv(model_path, usecols=["ModelID", "PatientID", "OncotreeLineage", "ModelType", "HCMIID"],
                         low_memory=False).set_index("ModelID", verify_integrity=True)
    header = pd.read_csv(path, nrows=0).columns
    mapping = {col: col.rsplit(" (", 1)[0] for col in header[1:] if col.rsplit(" (", 1)[0] in targets}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("DepMap target symbols are ambiguous")
    dependency = -pd.read_csv(path, usecols=[header[0], *mapping]).set_index(header[0]).rename(columns=mapping)
    keep = rna.index.intersection(dependency.index).intersection(models.index)
    roster = models.loc[keep]
    roster = roster[roster.ModelType.eq("Cell Line") & roster.HCMIID.isna() & roster.OncotreeLineage.notna()]
    ids = roster.index
    donors = roster.PatientID.fillna(ids.to_series())
    # Aggregate only linked cultures from the same biological donor.
    if roster.assign(donor=donors).groupby("donor").OncotreeLineage.nunique().gt(1).any():
        raise ValueError("Conflicting DepMap tissue labels within donor")
    roster = pd.DataFrame({"donor": donors, "tissue": roster.OncotreeLineage})
    return rna.loc[ids], dependency.loc[ids], roster


def register(args):
    require_hash(args.supplement, META)
    require_hash(args.fitness, FITNESS)
    targets, rna_2d = panel_from_core(args.core, args.targets)
    rna, labels, roster, blocked = organoid_inputs(args.supplement, args.fitness, targets)
    rna_2d, labels_2d, roster_2d = depmap_inputs(args.chronos, args.models, rna_2d, targets)
    genes = rna.columns.intersection(rna_2d.columns)
    eligible = sorted(set(labels.columns) & set(labels_2d.columns) & set(genes))
    donors = roster.groupby("tissue").donor.nunique().to_dict()
    if len(genes) < 12_000 or len(eligible) < 50 or min(donors[t] for t in TISSUES) < MIN_HELD_DONORS:
        raise ValueError("Minimum common RNA, target or donor support was not met")
    rna = pd.concat([rna_2d[genes], rna[genes]], verify_integrity=True)
    with new_output(args.out) as out:
        for name, frame in zip(FILES, (rna, roster, labels_2d[eligible], roster_2d)):
            frame.to_parquet(out / name)
        for name, group in roster.groupby("tissue", sort=True):
            labels.loc[group.index, eligible].to_parquet(out / f"{name}.parquet")
        files = {file.name: sha256(file) for file in sorted(out.glob("*.parquet"))}
        publish_json(out / "receipt.json", {
            "sanger_metadata_sha256": META, "sanger_fitness_sha256": FITNESS, "chronos_sha256": CHRONOS,
            "depmap_models_sha256": MODELS, "targets_sha256": TARGETS, "source_core_sha256": RECEIPTS["core"],
            "blocked_organoids": blocked, "organoids": roster.tissue.value_counts().to_dict(),
            "organoid_donors": donors, "depmap_donors": roster_2d.donor.nunique(), "genes": len(genes),
            "target_panel": eligible, "files": files,
        })


def verified(path, selected):
    receipt = json.loads((path / "receipt.json").read_text())
    if tuple(receipt[name] for name in ("sanger_metadata_sha256", "sanger_fitness_sha256", "chronos_sha256",
                                        "targets_sha256")) != (META, FITNESS, CHRONOS, TARGETS):
        raise ValueError("Unexpected organoid dependency registration")
    for filename in selected:
        require_hash(path / filename, receipt["files"][filename])
    return [pd.read_parquet(path / filename) for filename in selected], sha256(path / "receipt.json")


def donor_inputs(profile, values, roster):
    donor = roster.donor
    gene_values = values.groupby(donor).mean()
    expression = profile.loc[roster.index].groupby(donor).mean().loc[gene_values.index]
    tissue = roster.groupby("donor").tissue.first().loc[gene_values.index]
    return expression, gene_values, tissue


def contrasts(values, tissue):
    ranks = values.groupby(tissue).rank(method="average", pct=True)
    center = ranks.groupby(tissue).transform("mean")
    spread = ranks.groupby(tissue).transform("std")
    count = values.groupby(tissue).transform("count")
    return ((ranks - center) / spread).where(count.ge(5) & spread.gt(0))


def fit_profiles(rna, train, values, held, edges, targets, device):
    frame = pd.concat([rna.loc[train], held], verify_integrity=True)
    a, b = domain_kernels(frame, train, held.index, edges, targets, device=device)
    estimated = []
    for key in values:
        observed = values[key].dropna()
        locations = torch.as_tensor(train.get_indexer(observed.index), device=device)
        gram = a[locations[:, None], locations]
        y = torch.as_tensor(observed.to_numpy(), device=device, dtype=gram.dtype)
        coeff = torch.linalg.solve(gram + 4 * torch.eye(len(gram), device=device), y)
        estimated.append((b[:, locations] @ coeff).detach().cpu().numpy())
    return pd.DataFrame(dict(zip(values, estimated)), index=held.index)


def predict(args):
    tissue = args.tissue
    labels = tuple(f"{name}.parquet" for name in ("Colorectal", "Oesophageal", "Ovarian", "Pancreatic", "Gastric")
                   if name != tissue)
    (rna, info, two, info_2d, *three_parts), receipt = verified(args.registered, (*FILES, *labels))
    organoids = pd.concat(three_parts, verify_integrity=True)
    held = info[info.tissue.eq(tissue)]
    fit = info.loc[organoids.index]
    if set(held.donor) & set(fit.donor):
        raise ValueError("An organoid donor crosses the held tissue")
    # Keep two donors with the same label from colliding across different catalogues.
    held_profiles = rna.loc[held.index].groupby(held.donor).mean().rename(index=lambda x: f"SANGER:{x}")
    x3, y3, tissue3 = donor_inputs(rna, organoids, fit)
    x2, y2, tissue2 = donor_inputs(rna, two, info_2d)
    a, b = contrasts(y2, tissue2), contrasts(y3, tissue3)
    genes = a.count().ge(100) & b.count().ge(20)
    target_panel = genes.index[genes]
    if len(target_panel) < 40:
        raise ValueError(f"Held tissue {tissue} has too few fitted targets")
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    require_hash(args.targets, TARGETS)
    raw = rna_values(held_profiles, rank=True)
    direct = pd.DataFrame(raw, columns=held_profiles.columns, index=held_profiles.index)[target_panel]
    device = device_cuda()
    two = fit_profiles(x2, x2.index, a[target_panel], held_profiles, edges, targets, device)
    three = fit_profiles(x3, x3.index, b[target_panel], held_profiles, edges, targets, device)
    rng = np.random.default_rng(62093)
    shuffled = b[target_panel].copy()
    for _, subset in tissue3.groupby(tissue3):
        order = rng.permutation(subset.index.to_numpy())
        shuffled.loc[order] = b.loc[np.roll(order, 1), target_panel].to_numpy()
    null = fit_profiles(x3, x3.index, shuffled, held_profiles, edges, targets, device)
    joint = dict(zip(ARMS, (direct, two, null, three, .5 * (two + three))))
    tables = []
    for name, values in joint.items():
        long = values.rename_axis("cell_id").reset_index().melt("cell_id", var_name="compound_id",
                                                                value_name="predicted_value")
        tables.append(long.assign(method=name))
    rows = pd.concat(tables, ignore_index=True).assign(track=tissue, fold=0, lineage_group=lambda f: f.cell_id,
                                                       row_id=lambda f: f.cell_id + "|" + f.compound_id)
    freeze_table(rows, args.out, provenance={
        "registration_sha256": receipt, "held_tissue": tissue, "held_donors": len(held_profiles),
        "fit_3d_donors_by_tissue": tissue3.value_counts().to_dict(), "fit_2d_donors": len(x2),
        "target_panel": target_panel.tolist(), "features": priors,
    })


def score(args):
    result, hashes = {}, {}
    for name in TISSUES:
        (roster, observed), registered = verified(args.registered, ("roster.parquet", f"{name}.parquet"))
        prediction, seal = verify_forecast(args.forecasts / f"{name}.parquet")
        meta = seal["provenance"]
        if (meta["registration_sha256"], meta["held_tissue"]) != (registered, name):
            raise ValueError("Forecast registration differs from the measured tissue")
        measured = observed.groupby(roster.donor).mean()[meta["target_panel"]]
        measured.index = pd.Index([f"SANGER:{i}" for i in measured.index], name="cell_id")
        truth = measured.reset_index().melt("cell_id", var_name="compound_id", value_name="endpoint_value")
        joined = prediction.merge(truth, on=["cell_id", "compound_id"], validate="many_to_one").dropna()
        if joined.lineage_group.nunique() != meta["held_donors"] or len(prediction) != len(truth) * len(ARMS):
            raise ValueError("Mismatched frozen donor or target support")
        comparisons = [(PRIMARY, baseline) for baseline in REFERENCE]
        per_gene, _ = score_across_cells(joined, min_lineages=MIN_HELD_DONORS)
        usable = per_gene.groupby("compound_id").pearson.count()
        per_gene = per_gene[per_gene.compound_id.isin(usable.index[usable.eq(len(ARMS))])]
        support = joined[joined.compound_id.isin(per_gene.compound_id)]
        intervals = {reference: crossed_patient_drug_ci(support, PRIMARY, reference, repeats=1000)
                     for reference in REFERENCE}
        gates = {key: len(ci) == 2 and all(value[0] > RANK_TOLERANCE for value in ci.values())
                 for key, ci in intervals.items()}
        result[name] = {"donors": len(measured), "targets": per_gene.compound_id.nunique(), "gates": gates,
                        "scores": per_gene.groupby("method")[["spearman", "pearson"]].mean().to_dict("index"),
                        "ci95": intervals, "per_gene": per_gene.to_dict("records")}
        hashes[name] = seal["sha256"]
    passed = all(r["targets"] >= 40 and all(r["gates"].values()) for r in result.values())
    publish_json(args.out, {"tissues": result, "forecast_sha256": hashes, "pass": passed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "register": ("supplement", "fitness", "chronos", "models", "core", "targets"),
        "predict": ("registered", "priors", "hallmark", "targets"),
        "score": ("registered", "forecasts"),
    }
    for stage, arguments in fields.items():
        command = stages.add_parser(stage)
        for name in (*arguments, "out"):
            command.add_argument("--" + name, type=Path, required=True)
        if stage == "predict":
            command.add_argument("--tissue", choices=TISSUES, required=True)
    args = parser.parse_args()
    {"register": register, "predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
