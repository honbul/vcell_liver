import numpy as np
from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.run import read_json
from h3v2.data import ASSETS, ROOT, SCREENS, SPLITS, V1, read_data
from h3v2.final_fit import exact_join, external_query, old_h2
from h3v2.grid import SEEDS
from h3v2.kernel import read_bank
from vc_build.io import publish_json, verify_forecast

parts = {}
assets = read_json(ASSETS / "data_receipt.json")
coverage = parquet_lock(ASSETS / "coverage.parquet", assets["files"]["coverage.parquet"])
for split in [*SPLITS, "external"]:
    if split == "external":
        query = external_query()
        mode, fold = "external", 3
    else:
        query = read_data(split, "query")
        mode, number = split.rsplit("-", 1)
        fold = int(number)
    part = {"query_rows": len(query)}
    if mode in ("drug", "external"):
        h2, digest = old_h2(fold)
        aligned = exact_join(query, h2)
        lincs = query[KEY].map(coverage.lincs_modz24).fillna(False).to_numpy()
        assert np.array_equal(lincs, aligned.champion_observed.to_numpy())
        part["published_h2_sha256"] = digest
    else:
        bank = read_bank(split, "outer")
        assert set(query.row_id) == set(bank.index)
        assert np.isfinite(bank[["C1", "H2"]]).all().all()
    if mode in SCREENS:
        path = V1 / "outer-forecast" / split / "forecast.parquet"
        forecasts, proof = verify_forecast(path)
        wide = forecasts.pivot(index="row_id", columns="method", values="predicted_value")
        assert set(("structured", "drug_ridge", "drug_mean", *(f"G2__{seed}" for seed in SEEDS))) <= set(wide)
        assert np.isfinite(exact_join(query, wide)).all().all()
        part["published_h3v1_sha256"] = proof["sha256"]
    parts[split] = part
assert not (ROOT / "score" / "consumed.json").exists()
publish_json(ROOT / "reference_roster_verification.json", {"partitions": parts, "outcomes_loaded": False})
print("Verified all sixteen exact row rosters for C1, H2 and H3-v1 references without outcomes", flush=True)
