"""Register scaffold-cold and independently target-cold LIMORE tests."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import pandas as pd

from .perturbseq import record
from .predict import checked, write_json


def target_union(drugs, targets):
    return set().union(*(targets[key] for key in set(drugs)))


def eligible_rows(rows, targets, references, check_targets):
    groups = rows.scaffold.isin(references.scaffold) | rows.connectivity.isin(references.connectivity)
    if check_targets:
        seen = target_union(references.drug_id, targets)
        groups |= rows.drug_id.map(lambda key: bool(targets[key] & seen))
    return rows[~groups]


def verify_roles(train, query, targets, strict):
    for field in ("id", "scaffold", "connectivity", "drug_id"):
        if not set(train[field]).isdisjoint(query[field]):
            raise ValueError(f"Drug-cold identity overlap: {field}")
    overlap = target_union(train.drug_id, targets) & target_union(query.drug_id, targets)
    if overlap and strict:
        raise ValueError("Registered target IDs crossed a target-cold boundary")
    return len(overlap)


def validation_partition(rows, targets, seed, strict):
    keys = sorted(rows.scaffold.unique(), key=lambda key: sha256(
        f"track4-bridge-{seed}:{key}".encode(),
    ).digest())
    for offset in range(5):
        valid = rows[rows.scaffold.isin(keys[offset::5])]
        fit = eligible_rows(rows, targets, valid, strict)
        if valid.drug_id.nunique() >= 3 and fit.drug_id.nunique() >= 7:
            return fit, valid
    raise ValueError(f"Insufficient independent target/scaffold groups for tuning ({rows.drug_id.nunique()} drugs)")


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    target_receipt = a.targets / "receipt.json"
    target_claims = pd.read_csv(checked(a.targets, "targets.tsv", json.loads(target_receipt.read_text())), sep="\t")
    targets = target_claims.groupby("drug_id").gene.agg(set).to_dict()
    summaries = {}
    manifests = {}
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        staged = Path(directory)
        target_claims.to_csv(staged / "targets.tsv", sep="\t", index=False)
        for fold in range(5):
            base = a.panels / f"limore-drug-cold-{fold}"
            base_receipt = base / "receipt.json"
            source = json.loads(base_receipt.read_text())
            manifests[str(fold)] = record(base_receipt)
            train, query, truth = [pd.read_parquet(checked(base, f"{role}.parquet", source))
                                   for role in ("train", "query", "outcomes")]
            train = train[train.drug_id.isin(targets)]
            query = query[query.drug_id.isin(targets)]
            if "outcome" in query or not set(query.id).issubset(truth.id):
                raise ValueError("Invalid external assay covariate/label boundary")
            for strict, panel in ((False, "scaffold"), (True, "unseen_target")):
                key = f"{panel}-{fold}"
                outer = eligible_rows(train, targets, query, strict)
                overlap = verify_roles(outer, query, targets, strict)
                fit, valid = validation_partition(outer, targets, a.seed + fold, strict)
                valid_overlap = verify_roles(fit, valid, targets, strict)
                if min(fit.drug_id.nunique(), valid.drug_id.nunique(), query.drug_id.nunique()) < 3:
                    raise ValueError(f"Insufficient independent drugs for {key}")
                output = staged / key
                output.mkdir()
                roles = {"fit": fit, "validation": valid, "query": query,
                         "labels": truth[truth.id.isin(query.id)]}
                for role, rows in roles.items():
                    rows.to_parquet(output / f"{role}.parquet", index=False)
                testing_drugs = sorted(set(query.drug_id))
                seen_genes = target_union(outer.drug_id, targets)
                observed = {k: len(targets[k] & seen_genes) for k in testing_drugs}
                summary = {
                    "fold": fold, "strict_target_cold": strict, "outer_training_targets_shared": overlap,
                    "fit_validation_targets_shared": valid_overlap, "fit_drugs": fit.drug_id.nunique(),
                    "validation_drugs": valid.drug_id.nunique(), "test_drugs": query.drug_id.nunique(),
                    "outer_drugs_removed": train.drug_id.nunique() - outer.drug_id.nunique(),
                    "test_drugs_with_any_seen_target": sum(v > 0 for v in observed.values()),
                    "query_rows": len(query), "seed": a.seed + fold,
                }
                write_json(output / "receipt.json", {
                    **summary, "source_panel": record(base_receipt), "target_map": record(staged / "targets.tsv"),
                    "files": {x.name: record(x) for x in sorted(output.iterdir())},
                })
                summaries[key] = summary
        write_json(staged / "receipt.json", {
            "source_panels": manifests, "target_registry": record(target_receipt),
            "panels": summaries, "files": {str(x.relative_to(staged)): record(x) for x in sorted(staged.rglob("*"))
                                           if x.is_file()},
        })
        staged.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("panels", "targets", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--seed", type=int, default=41124)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
