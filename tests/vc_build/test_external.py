import numpy as np
import pandas as pd
import pytest

from vc_build.evaluate import across_cell_metrics
from vc_build.external import align_endpoints, train_rankers


def test_external_fit_only_curves_and_prediction_roster():
    rng = np.random.default_rng(14822)
    structures = ["CC", "CCC", "CCCC", "CCO", "CCN", "CNC"]
    rows = [dict(
        row_id=f"{cell}-{drug}", cell_id=cell, lineage_group=cell, compound_id=str(drug),
        isomeric_smiles=smiles, endpoint_id="source_AUC", source_id="source",
        endpoint_direction="lower", endpoint_value=1 - drug / len(structures) + noise,
    ) for cell, noise in zip(["f0", "f1", "f2", "f3"], [0., .1, -.1, .05])
            for drug, smiles in enumerate(structures)]
    fit = pd.DataFrame(rows)
    query = fit[fit.cell_id == "f0"].drop(columns=["endpoint_value"]).assign(cell_id="q", lineage_group="q")
    query["row_id"] = "q-" + query.compound_id
    rna = pd.DataFrame(rng.random((5, 60)), index=["f0", "f1", "f2", "f3", "q"],
                       columns=[f"gene{i}" for i in range(60)])
    forecast, source_values, report = train_rankers(fit, query, rna.iloc[:4], rna.iloc[-1:], {"0": ["gene1"]})
    assert len(forecast) == 4 * len(query)
    assert report[0]["training_rows"] == len(fit)
    assert np.isfinite(forecast.predicted_value).all()
    assert np.all(np.diff(source_values.drug_mean) > 0)
    with pytest.raises(ValueError, match="received target outcomes"):
        train_rankers(fit, query.assign(endpoint_value=0), rna.iloc[:4], rna.iloc[-1:], {})
    with pytest.raises(ValueError, match="overlap in lineage_group"):
        train_rankers(fit, query.assign(lineage_group="f0"), rna.iloc[:4], rna.iloc[-1:], {})


def test_alignment_recovers_scale_without_effect_from_source_unique_drugs():
    ref = pd.DataFrame([
        {"cell_id": f"c{i}", "compound_id": f"d{j}", "endpoint_value": i + j,
         "source_id": "GDSC2", "endpoint_id": "ref", "endpoint_direction": "higher"}
        for i in range(30) for j in range(3)
    ])
    src = ref.assign(source_id="other", endpoint_id="other", endpoint_value=(ref.endpoint_value - 2) / 3)
    exclusive = src.iloc[:30].assign(compound_id="different-panel", endpoint_value=1e4)
    aligned, transforms = align_endpoints(pd.concat([ref, src, exclusive], ignore_index=True))
    other = transforms[1]
    assert other["slope"] == pytest.approx(3)
    assert other["intercept"] == pytest.approx(2)
    assert other["anchor_pairs"] == 90
    assert np.allclose(aligned.iloc[len(ref):len(ref) + len(src)].model_value, ref.endpoint_value)


def test_across_cell_uses_independent_lineages_and_drug_mean_is_zero():
    group = pd.DataFrame({
        "lineage_group": ["a", "a", "b", "c", "d", "e"],
        "endpoint_value": [0., 1., 2., 3., 4., 5.],
        "predicted_value": [0., 1., 2., 3., 4., 5.],
    })
    result = across_cell_metrics(group)
    assert result["lineages"] == 5
    assert result["spearman"] == pytest.approx(1)
    assert result["pearson"] == pytest.approx(1)
    assert across_cell_metrics(group.assign(predicted_value=0))["spearman"] == 0
