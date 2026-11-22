"""Register and rank virtual combinations in low-response liver models."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .context import verified
from .identity import exact_token
from .splits import pair_key, record


ANCHORS = ("Sorafenib", "Lenvatinib")
ENDPOINTS = ("zip", "bliss", "added_inhibition")


def exact_approval(prepared):
    chemistry = pd.read_parquet(verified(prepared, "dili_inputs.parquet"))
    truth = pd.read_parquet(verified(prepared, "dili_labels.parquet"))
    rows = chemistry.merge(truth, on="source_drug", validate="one_to_one")
    same = rows.groupby("drug_id").concern.agg(lambda k: len(set(k)) == 1)
    return rows.loc[rows.drug_id.map(same), ["drug_id", "concern"]].drop_duplicates().rename(columns={"drug_id": "id"})


def screen_compounds(context, approved):
    local = pd.read_csv(verified(context, "molecules.tsv"), sep="\t")
    metadata = pd.read_csv(verified(context, "drug_metadata.tsv"), sep="\t", dtype=str)
    lookup = metadata.assign(clean=metadata.dname.map(exact_token))
    anchor_ids = {}
    for name in ANCHORS:
        ids = lookup.loc[lookup.clean.eq(exact_token(name)), "id"].unique()
        if len(ids) != 1 or ids[0] not in set(local.id):
            raise ValueError(f"No unique molecular identity for {name}")
        anchor_ids[name] = ids[0]
    low = approved[approved.concern.isin(["No", "Less"]) & approved.id.isin(local.id)
                   & ~approved.id.isin(anchor_ids.values())].copy()
    names = metadata.assign(upper=metadata.dname.str.upper().str.startswith("CHEMBL")).sort_values(
        ["upper", "dname"], kind="stable"
    ).drop_duplicates("id").set_index("id").dname
    low["candidate"] = low.id.map(names).fillna(low.id)
    return anchor_ids, low


def limore_profiles(rna_file, assay, receipt_path):
    digest = json.loads(receipt_path.read_text())
    if digest.get("artifacts", {}).get(assay.name) != record(assay):
        raise ValueError("LIMORE phenotype differs from its completed source receipt")
    rna = pd.read_parquet(rna_file)
    measured = pd.read_parquet(assay, columns=["source_compound_label", "endpoint_value", "cell_id", "cell_group"])
    sorafenib = measured[measured.source_compound_label.map(exact_token).eq("SORAFENIB")]
    sorafenib = sorafenib.sort_values("cell_id").drop_duplicates("cell_group")
    low = sorafenib.endpoint_value.quantile(.25)
    ranks = sorafenib[sorafenib.endpoint_value.le(low) & sorafenib.cell_id.isin(rna.index)]
    ids = "LIMORE:" + ranks.cell_id.astype(str)
    rna = rna.loc[ranks.cell_id].set_axis(ids)
    case = pd.DataFrame({"cell_id": ids.to_numpy(), "lineage": ranks.cell_group.to_numpy(), "domain": "LIMORE"})
    return rna, case, float(low)


def organoid_profiles(root):
    rna = pd.read_parquet(verified(root, "organoids/rna.parquet"))
    profiles = pd.read_parquet(verified(root, "organoids/query.parquet"))
    if not profiles.id.is_unique or profiles.donor.nunique() != 4 or set(rna.index) != set(profiles.id):
        raise ValueError("Unexpected paired organoid cohort")
    return rna, profiles.rename(columns={"id": "cell_id", "donor": "lineage"}).assign(domain="GSE182593")


def register(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    anchor_ids, compounds = screen_compounds(a.context, exact_approval(a.dili))
    org, org_keys = organoid_profiles(a.organoids)
    models, model_keys, q25 = limore_profiles(a.limore_rna, a.limore_assays, a.limore_receipt)
    expression = pd.concat([org, models], axis=0, join="inner")
    roster = pd.concat([org_keys, model_keys], ignore_index=True)
    if len(compounds) < 20 or len(model_keys) < 10 or len(org_keys) != 24:
        raise ValueError("Too little independently approved chemical or HCC profile support")
    chunks = []
    for anchor, drug in anchor_ids.items():
        for row in compounds.itertuples(index=False):
            if drug.split("-")[0] == row.id.split("-")[0]:
                continue
            out = roster.assign(anchor=anchor, candidate=row.candidate, approved_compound=row.id,
                                compound_a=drug, compound_b=row.id, dili=row.concern)
            out["pair"] = pair_key(out.compound_a.str.split("-").str[0], out.compound_b.str.split("-").str[0])
            out["source"] = "Sanger:virtual"
            out["id"] = out.anchor + ":" + out.approved_compound + ":" + out.cell_id
            chunks.append(out)
    candidates = pd.concat(chunks, ignore_index=True)
    if not candidates.id.is_unique:
        raise ValueError("Screen candidate roster contains duplicate pairs")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        stage = Path(d)
        candidates.to_parquet(stage / "query.parquet", index=False)
        expression.to_parquet(stage / "rna.parquet")
        (stage / "receipt.json").write_text(json.dumps({
            "inputs": {name: record(path / "receipt.json") for name, path in vars(a).items()
                       if isinstance(path, Path) and path != a.output and path.is_dir()},
            "limore_rna": record(a.limore_rna), "limore_inputs": record(a.limore_receipt),
            "rows": len(candidates), "organoid_profiles": len(org_keys), "limore_lineages": len(model_keys),
            "candidate_compounds": len(compounds), "limore_q25": q25,
            "independent_dili_policy": "Exact chemically registered FDA DILIrank v2 No or Less concern",
            "files": {p.name: record(p) for p in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def rank(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    frozen = json.loads((a.predictions / "receipt.json").read_text())
    candidate_file = verified(a.predictions, "screen_predictions.parquet")
    if frozen.get("screen_registry") != record(a.query / "receipt.json"):
        raise ValueError("Screening roster does not match the frozen inputs")
    query = pd.read_parquet(verified(a.query, "query.parquet"))
    pred = pd.read_parquet(candidate_file)
    labels = pd.read_parquet(verified(a.organoids, "organoids/labels.parquet"))
    name, selected = max(frozen["model_selection"].items(),
                         key=lambda m: sum(m[1]["validation"][e] for e in ENDPOINTS))
    validation = selected["validation"]
    cases = query.merge(pred[pred.model.eq(name)], on="id", validate="one_to_one")
    if len(cases) != len(query) or cases.id.duplicated().any():
        raise ValueError("Missing frozen candidate predictions")
    cases = cases.merge(labels.rename(columns={"id": "cell_id", "outcome": "resistant"}),
                        on="cell_id", how="left", validate="many_to_one")
    chosen = cases[cases.domain.eq("GSE182593")].copy()
    by_organoid = chosen.groupby(["anchor", "approved_compound", "lineage", "resistant"])[list(ENDPOINTS)].mean()
    resistant = by_organoid.xs(1, level="resistant")
    sensitive = by_organoid.xs(0, level="resistant")
    limore = cases[cases.domain.eq("LIMORE")].groupby(["anchor", "approved_compound"])[list(ENDPOINTS)].mean()
    rng = np.random.default_rng(a.seed)
    output = []
    for key, donor in resistant.groupby(level=["anchor", "approved_compound"], sort=True):
        reg = query[query.anchor.eq(key[0]) & query.approved_compound.eq(key[1])].iloc[0]
        row = {"anchor": key[0], "molecule": key[1], "candidate": reg.candidate, "dili_concern": reg.dili,
               "organoid_donors": len(donor), "model": name}
        x_pos, x_neg = donor.to_numpy(), sensitive.loc[donor.index].to_numpy()
        if len(x_pos) != 4:
            raise ValueError("Every hypothesis needs all four paired organoid donors")
        draw = rng.integers(len(x_pos), size=(a.bootstrap, len(x_pos)))
        means = x_pos.mean(axis=0)
        intervals = np.quantile(x_pos[draw].mean(axis=1), [.025, .975], axis=0)
        for j, endpoint in enumerate(ENDPOINTS):
            center = means[j]
            scale = validation["residual_abs_q95"][endpoint]
            row.update({
                endpoint: center, endpoint + "_donor_low": intervals[0, j], endpoint + "_donor_high": intervals[1, j],
                endpoint + "_source_error_radius95": scale, endpoint + "_limore": float(limore.loc[key, endpoint]),
                endpoint + "_resistant_minus_sensitive": (x_pos - x_neg).mean(axis=0)[j],
            })
        row["both_positive_predicted"] = bool(min(row["bliss"], row["added_inhibition"]) > 0)
        row["priority"] = min(row["bliss"], row["added_inhibition"])
        output.append(row)
    ranking = pd.DataFrame(output).sort_values(["both_positive_predicted", "priority"], ascending=False, kind="stable")
    eligible = ranking[ranking.both_positive_predicted].head(20)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        stage = Path(d)
        files = {"ranked_virtual_combinations.tsv": ranking, "top20.tsv": eligible}
        for filename, frame in files.items():
            frame.to_csv(stage / filename, sep="\t", index=False)
        (stage / "receipt.json").write_text(json.dumps({
            "predictions": record(a.predictions / "receipt.json"), "screen_registry": record(a.query / "receipt.json"),
            "organoid_phenotypes": record(verified(a.organoids, "organoids/labels.parquet")),
            "model": name, "bootstrap": a.bootstrap, "n_eligible": int(ranking.both_positive_predicted.sum()),
            "tested_candidates": len(ranking), "top_rows": len(eligible),
            "scope": "In silico research prioritization, no combination treatment outcomes",
            "files": {p.name: record(p) for p in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    reg, evaluate = modes.add_parser("register"), modes.add_parser("rank")
    for name in ("context", "dili", "organoids", "limore-rna", "limore-assays", "limore-receipt", "output"):
        reg.add_argument("--" + name, type=Path, required=True)
    for name in ("predictions", "query", "organoids", "output"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=3000)
    evaluate.add_argument("--seed", type=int, default=654)
    a = p.parse_args()
    (register if a.mode == "register" else rank)(a)


if __name__ == "__main__":
    main()
