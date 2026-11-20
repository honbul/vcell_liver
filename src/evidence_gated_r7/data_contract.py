"""DATA_CONTRACT.json: the R4 unit files R7 read (re-hashed and checked against the committed R4 contract), roles, axes."""
import json, os
import pandas as pd

from .common import r4proc, tb, pman, dump, sha256, R4_MAN, HCC


def main():
    r4 = json.load(open(os.path.join(R4_MAN, "data_contract.json")))
    files = {}
    for f in ["units.tsv", "groups.tsv", "unit_x0.npy", "unit_delta_open.npy", "unit_delta_test.npy"]:
        h = sha256(r4proc(f))
        files[f] = dict(sha256=h, matches_r4_contract=h == r4["files"][f]["sha256"],
                        read_by="final_eval only (R7_FINAL_EVAL, committed freeze)" if f == "unit_delta_test.npy" else "R7 jobs")
    u = pd.read_csv(r4proc("units.tsv"), sep="\t")
    out = dict(track="evidence_gated_r7", source="R4 processed units (read-only)", r4_contract_sha256=sha256(os.path.join(R4_MAN, "data_contract.json")),
               files=files, all_match=all(v["matches_r4_contract"] for v in files.values()),
               platform="CMap LINCS 2020 Level 3, single small molecules, 10 uM, 24 h", genes=978, gene_axis_sha256=sha256(tb("genes.tsv")),
               ecfp4_sha256=sha256(tb("ecfp4.npy")), unit="(cell, drug_group)",
               units_by_role=u.role.value_counts().to_dict(), groups_by_role=u.drop_duplicates("drug_group").role.value_counts().to_dict(),
               cells=int(u.cell.nunique()), families=int(u.family.nunique()),
               hcc_units_by_role={h: u[u.cell == h].role.value_counts().to_dict() for h in HCC},
               split="R4 global drug-group split 70/15/15, seed 20261117 (manifests/response_geometry_r4/drug_split.tsv)")
    dump(out, pman("DATA_CONTRACT.json"))
    print(json.dumps(out, indent=1))
    assert out["all_match"]


if __name__ == "__main__":
    main()
