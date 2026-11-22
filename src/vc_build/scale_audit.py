"""Compare endpoint alignment under grouped, source-only validation."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from .evaluate import score, score_across_cells
from .external import KEY, SOURCE, _rna, native_sensitivity, train_rankers
from .io import publish_json, read_table, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.bundle / "receipt.json").read_text())
    for name in ("rows", "rna"):
        relative = f"train/{name}.parquet"
        if sha256(args.bundle / relative) != receipt["artifacts"][relative]["sha256"]:
            raise ValueError(f"Training input differs from its receipt: {relative}")
    frame = read_table(args.bundle / "train/rows.parquet").rename(
        columns={"record_id": "row_id", "cell_group": "lineage_group"}
    )
    rna = _rna(args.bundle / "train/rna.parquet")
    audit = []
    validations = []
    keys = ["cell_id", "lineage_group", "compound_id"]
    for fold, (fit_indices, validation_indices) in enumerate(
        GroupKFold(n_splits=3).split(frame, groups=frame.lineage_group)
    ):
        fit, validation = (frame.iloc[indices] for indices in (fit_indices, validation_indices))
        query = validation[[*keys, "isomeric_smiles"]].drop_duplicates()
        query["row_id"] = query.cell_id + "|" + query.compound_id
        query = query[query.compound_id.isin(fit.compound_id)]
        actual = validation.assign(endpoint_value=native_sensitivity(validation)).drop(columns="row_id")
        fit_rna = rna.loc[sorted(fit.cell_id.unique())]
        query_rna = rna.loc[sorted(query.cell_id.unique())]
        for historical in (True, False):
            forecast, _, transforms = train_rankers(
                fit, query, fit_rna, query_rna, {}, historical_rank_control=historical
            )
            forecast = forecast[forecast.track.eq("X_C_LIMORE")].drop(columns=["track", "fold"])
            prefix = "historical" if historical else "aligned"
            forecast["method"] = prefix + "/" + forecast.method
            joined = forecast.merge(actual[[*keys, *SOURCE, "endpoint_value"]], on=keys, validate="many_to_many")
            joined["track"] = joined.source_id + "/" + joined.endpoint_id
            validations.append(joined.assign(fold=fold)[[*KEY, "track", "fold", "method", "predicted_value", "endpoint_value"]])
            if not historical:
                audit.append({"fold": fold, "sources": transforms,
                              "validation_lineages": int(actual.lineage_group.nunique())})
    scored = pd.concat(validations, ignore_index=True)
    compare = [("aligned/" + method, "historical/" + method)
               for method in ("drug_mean", "rna_additive", "direct_structure_rna")]
    cells, _ = score(scored, endpoint_direction="higher", rank_only=True, comparisons=compare)
    drugs, contrasts = score_across_cells(scored, comparisons=compare)
    metrics = (drugs.groupby(["track", "method"])[["spearman", "pearson"]].mean()
               .reset_index().to_dict(orient="records"))
    publish_json(args.out, {"calibration": audit, "within_cell": cells,
                            "within_drug": metrics, "within_drug_contrasts": contrasts})


if __name__ == "__main__":
    main()
