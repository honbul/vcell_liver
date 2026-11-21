"""LINCS Level 3 response priors from compounds disjoint from the Tahoe catalog."""

import argparse
import hashlib
from collections import defaultdict
import json
import os
import re
import unicodedata
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.linalg import eigh

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import config, sha256, write_json

from .transfer import TransferFit, mean_units, unit_norm


SOURCE = "CMap LINCS2020 Level 3, 10 µM nominal nearest dose, 24 h, landmarks"
PHASE1_CROSSWALK = "GSE92742_Broad_LINCS_pert_info.txt.gz"
PHASE1_SHA512 = (
    "cdba99a4a6204ecdd26adb2338a418e60ac3da9e18d6d75214dea87c3cf06e951f9"
    "356e876f581dd7d6081014a428a11b6ad93b68486954cddbae5e29177530b"
)
METADATA_MISSING = {"", "-666", "nan", "na", "null", "none", "restricted", "unavailable"}


def clean(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.casefold() in METADATA_MISSING else value


def aliases(value):
    return (v for item in re.split(r"[|;]", clean(value)) if (v := clean(item)))


def normalized_name(name):
    return re.sub("[^a-z0-9]", "", unicodedata.normalize("NFKC", name).casefold())


@lru_cache(maxsize=32768)
def structural_keys(smiles):
    if not smiles:
        return set(), ""
    with rdBase.BlockLogs():
        parent = canonical_parent(smiles)
        if not parent:
            return set(), ""
        flat = parent[1]
        mol = Chem.MolFromSmiles(flat)
        if mol is None:
            return set(), ""
        key = Chem.MolToInchiKey(mol)
        try:
            tautomer = Chem.MolToSmiles(rdMolStandardize.TautomerParent(mol), canonical=True)
        except (RuntimeError, ValueError):
            tautomer = flat
    result = {("parent", flat), ("tautomer", tautomer)}
    if key:
        result.add(("connectivity", key.split("-")[0]))
    return result, tautomer


def identifiers(names, smiles="", inchis="", cid=""):
    structures, parent = structural_keys(clean(smiles))
    keys = structures | {("name", x) for n in names if (x := normalized_name(n))}
    keys.update(("connectivity", k.upper().split("-")[0]) for k in aliases(inchis)
                if re.fullmatch(r"[A-Za-z]{14}-[A-Za-z]{10}-[A-Za-z]", k))
    if re.fullmatch(r"[1-9]\d*(?:\.0+)?", clean(cid)):
        keys.add(("cid", clean(cid).split(".")[0]))
    return keys, parent


def exclusion_audit(catalog, metadata, donors, phase1=None):
    """Expand exact identity hits through shared keys and preexisting LINCS alias groups."""
    forbidden = set()
    for row in catalog.itertuples(index=False):
        keys, _ = identifiers([row.drug], row.canonical_smiles, cid=row.pubchem_cid)
        forbidden.update(keys)

    keys_by_id, parent_by_id, codes_by_key = defaultdict(set), defaultdict(set), defaultdict(set)
    cid_field = next((c for c in ("pubchem_cid", "pubchem_id", "cid") if c in metadata), None)

    def register(code, names, smiles, inchis, cid=""):
        keys, parent = identifiers(names, smiles, inchis, cid)
        keys_by_id[code].update(keys)
        if parent:
            parent_by_id[code].add(parent)
        for key in keys:
            codes_by_key[key].add(code)

    for row in metadata.itertuples(index=False):
        register(row.pert_id, [row.pert_id, row.cmap_name, *aliases(row.compound_aliases)],
                 row.canonical_smiles, row.inchi_key, getattr(row, cid_field, "") if cid_field else "")
    if phase1 is not None:
        for row in phase1.itertuples(index=False):
            # Legacy annotations establish identity links; chemistry comes from the 2020 records.
            register(row.pert_id, [row.pert_id, row.pert_iname], "", row.inchi_key, row.pubchem_cid)
    by_group = defaultdict(set)
    for r in donors.itertuples(index=False):
        register(r.pert_id, [r.pert_id, *aliases(r.cmap_name)], r.smiles_used, r.inchi_keys)
        by_group[r.drug_group].add(r.pert_id)

    source_group = donors.set_index("pert_id").drug_group.to_dict()
    blocked = {}
    pending = set()
    for key in sorted(forbidden):
        for code in codes_by_key.get(key, ()):
            blocked.setdefault(code, []).append(key[0])
            pending.add(code)
    visited_keys = forbidden.copy()
    while pending:
        code = pending.pop()
        candidates = {other for key in keys_by_id[code] - visited_keys for other in codes_by_key[key]}
        visited_keys.update(keys_by_id[code])
        if code in source_group:
            candidates.update(by_group[source_group[code]])
        for other in candidates - blocked.keys():
            blocked[other] = ["transitive_alias"]
            pending.add(other)

    ambiguous = {g for g, ids in by_group.items() if any(len(parent_by_id[code]) != 1 for code in ids)}
    receipt = []
    for row in donors.itertuples(index=False):
        code = row.pert_id
        flags = set(blocked.get(code, []))
        if row.drug_group in ambiguous:
            flags.add("ambiguous_or_unresolved_group")
        parents = sorted(parent_by_id[code])
        cids = sorted(v for method, v in keys_by_id[code] if method == "cid")
        receipt.append(dict(pert_id=code, drug_group=row.drug_group, retained=not flags,
                            exclusion=";".join(sorted(flags)), parents="|".join(parents), cids="|".join(cids)))
    audit = pd.DataFrame(receipt)
    retained = audit.loc[audit.retained, "pert_id"]
    if any(keys_by_id[code] & visited_keys for code in retained):
        raise ValueError("Retained LINCS donor intersects the closed Tahoe identity exclusion")
    return audit, {
        "direct_lincs2020_cid_field": cid_field,
        "phase1_cid_crosswalk_compounds": 0 if phase1 is None else len(phase1),
        "tahoe_catalog_cids": sum(k[0] == "cid" for k in forbidden),
        "donor_with_direct_cid": int(audit.cids.ne("").sum()),
        "donor_retained_with_direct_cid": int(audit.loc[audit.retained, "cids"].ne("").sum()),
        "donor_direct_cid_matches": sum("cid" in reasons for code, reasons in blocked.items() if code in source_group),
        "tahoe_direct_cid_matches": sum(k[0] == "cid" and bool(codes_by_key[k]) for k in forbidden),
        "tahoe_exclusion_keys": len(forbidden),
        "closed_exclusion_keys": len(visited_keys),
    }


def phase1_crosswalk(path):
    if hashlib.sha512(path.read_bytes()).hexdigest() != PHASE1_SHA512:
        raise ValueError(f"Phase I CID crosswalk differs from GEO receipt for {PHASE1_CROSSWALK}")
    columns = ["pert_id", "pert_iname", "pert_type", "inchi_key", "pubchem_cid"]
    metadata = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, compression="gzip", usecols=columns)
    return metadata.loc[metadata.pert_type.eq("trt_cp")]


def gene_map(donor, target):
    left = donor.assign(donor_col=np.arange(len(donor)))
    keep = left.ensembl_id.str.fullmatch(r"ENSG\d+") & ~left.ensembl_id.duplicated(keep=False)
    right_ok = target.ensembl_id.str.fullmatch(r"ENSG\d+") & ~target.ensembl_id.duplicated(keep=False)
    return left.loc[keep].merge(target.loc[right_ok, ["token_id", "ensembl_id"]],
                                on="ensembl_id", validate="one_to_one").sort_values("donor_col")


def verify_cache(cache, manifest):
    needed = ["rows.tsv", "drugs.tsv", "genes.tsv", "x0_all.npy", "x1_source.npy"]
    receipt = json.loads(manifest.read_text())
    for name in needed:
        expected = receipt["files"][name]
        path = cache / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"LINCS source cache differs from B1 receipt for {name}")
    return {name: receipt["files"][name]["sha256"] for name in needed}


def enumerated_keys(rows, cols):
    keys = rows[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)
    code = pd.MultiIndex.from_frame(keys).get_indexer(pd.MultiIndex.from_frame(rows[cols]))
    return keys, code


def aggregate_donor(rows, control, treated, cols):
    """Balanced other-compound signatures with detection-plate effects removed."""
    kept_rows = []
    deltas = []
    for start in range(0, len(rows), 8192):
        block = rows.iloc[start:start + 8192]
        row_ids = block.row_id.to_numpy()
        a, b = control[row_ids], treated[row_ids]
        eligible = (a.mean(axis=1) >= 1.0) & (b.mean(axis=1) >= 1.0)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("LINCS donor source has non-finite measurements")
        kept_rows.append(block.loc[eligible])
        deltas.append(np.ascontiguousarray(b[eligible][:, cols] - a[eligible][:, cols]))
    if not deltas or not any(len(block) for block in deltas):
        raise ValueError("No eligible identity-disjoint LINCS donor measurements")
    data = np.concatenate(deltas, axis=0)
    known = pd.concat(kept_rows, ignore_index=True)
    plate_keys, index = enumerated_keys(known, ["cell", "det_plate", "drug_group"])
    plate_group, _ = mean_units(data, index, len(plate_keys))
    plate_cols, plate_ix = enumerated_keys(plate_keys, ["cell", "det_plate"])
    plate_mean, count = mean_units(plate_group, plate_ix, len(plate_cols))
    chosen = count[plate_ix] >= 3
    centered = plate_group[chosen] - plate_mean[plate_ix[chosen]]
    cell_keys, cell_ix = enumerated_keys(plate_keys.loc[chosen], ["cell", "drug_group"])
    cell_group, _ = mean_units(centered, cell_ix, len(cell_keys))
    groups, group_index = np.unique(cell_keys.drug_group.to_numpy(dtype=str), return_inverse=True)
    signatures, support = mean_units(cell_group, group_index, len(groups))
    # Stable cell partitions provide heterogeneous views, with each cell contributing once.
    view = cell_keys.cell.map({cell: i % 2 for i, cell in enumerate(sorted(cell_keys.cell.unique()))}).to_numpy()
    views = [mean_units(cell_group[view == i], group_index[view == i], len(groups)) for i in (0, 1)]
    counts = {"known_rows": len(known), "plate_condition_groups": int(chosen.sum()),
              "cell_condition_groups": len(cell_keys), "donor_genes": len(cols),
              "retained_donor_cell_lines": int(cell_keys.cell.nunique()), "donor_signature_groups": len(groups)}
    return groups, signatures, support, views, counts


def fingerprints(parents):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return unit_norm(np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)) for smiles in parents]))


def group_fingerprints(audit, groups):
    selected = audit.loc[audit.retained & audit.drug_group.isin(groups), ["drug_group", "parents"]].drop_duplicates()
    code = pd.Index(groups).get_indexer(selected.drug_group)
    descriptors, _ = mean_units(fingerprints(selected.parents), code, len(groups))
    return descriptors


def prepare(cache, manifest, tmeta, tgenes, trows, lmeta, phase1_path, output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = pd.read_csv(lmeta, sep="\t", dtype=str, keep_default_na=False)
    catalog = pq.read_table(tmeta, columns=["drug", "canonical_smiles", "pubchem_cid"]).to_pandas()
    protected = pd.read_csv(trows, sep="\t", usecols=["compound_entity_id", "smiles"]).drop_duplicates()
    if catalog.drug.duplicated().any() or not set(protected.compound_entity_id) <= set(catalog.drug):
        raise ValueError("Tahoe primary groups were not fully covered by the drug catalog")
    scope = protected.merge(catalog, left_on="compound_entity_id", right_on="drug", validate="many_to_one")
    for row in scope.itertuples(index=False):
        # The feature-staged SMILES and catalog must identify the same standardized parent.
        staged, original = structural_keys(clean(row.smiles))[0], structural_keys(clean(row.canonical_smiles))[0]
        if not (staged & original):
            raise ValueError(f"Unresolved chemistry conflict for Tahoe staged entity {row.compound_entity_id}")
    donors = pd.read_csv(cache / "drugs.tsv", sep="\t", dtype=str, keep_default_na=False)
    if donors.pert_id.duplicated().any() or donors.drug_group.eq("").any():
        raise ValueError("Incomplete source compound grouping")
    crosswalk = phase1_crosswalk(phase1_path)
    audit, identity = exclusion_audit(catalog, metadata, donors, crosswalk)
    audit.to_csv(output / "identity.tsv", sep="\t", index=False)
    genes = pd.read_csv(cache / "genes.tsv", sep="\t", dtype=str, keep_default_na=False)
    targets = pd.read_csv(tgenes, sep="\t", keep_default_na=False)
    matching = gene_map(genes, targets)
    if len(matching) < 10:
        raise ValueError("Too few exact Ensembl IDs referenced by both assays")
    matching.to_csv(output / "genes.tsv", sep="\t", index=False)
    retained = audit.loc[audit.retained, "pert_id"]
    scope_entities = sorted(set(protected.compound_entity_id))
    provenance = dict(identity=identity, source=SOURCE, tahoe_catalog_compounds=len(catalog),
                      staged_catalog_coverage=len(scope_entities), protected_entities=scope_entities,
                      donor_rows_metadata=len(donors), metadata_excluded_pert_ids=int((~audit.retained).sum()),
                      eligible_donor_pert_ids=len(retained), aligned_genes=len(matching),
                      source_files=verify_cache(cache, manifest), lincs_metadata_sha256=sha256(lmeta),
                      phase1_cid_metadata=PHASE1_CROSSWALK, phase1_cid_metadata_sha512=PHASE1_SHA512,
                      tahoe_catalog_sha256=sha256(tmeta), staged_scope_sha256=sha256(trows))
    write_json(output / "identity_receipt.json", provenance)
    print(json.dumps({k: provenance[k] for k in ("identity", "metadata_excluded_pert_ids",
                                                "eligible_donor_pert_ids", "aligned_genes")}), flush=True)
    rows = pd.read_csv(cache / "rows.tsv", sep="\t")
    source = rows.loc[rows.role.eq("source")]
    x0 = np.load(cache / "x0_all.npy", mmap_mode="r")
    x1 = np.load(cache / "x1_source.npy", mmap_mode="r")
    if (x0.shape != (len(rows), len(genes)) or x1.shape != (len(source), len(genes))
            or not np.array_equal(source.row_id, np.arange(len(source)))):
        raise ValueError("LINCS cache row/gene manifest does not align with measurements")
    subset = source.loc[source.pert_id.isin(retained)]
    groups, signatures, support, views, counts = aggregate_donor(subset, x0, x1, matching.donor_col.to_numpy())
    np.savez(output / "other_compound_signatures.npz", groups=groups, signature=signatures, support=support,
             view0=views[0][0], view0_n=views[0][1], view1=views[1][0], view1_n=views[1][1],
             ecfp=group_fingerprints(audit, groups), token_id=matching.token_id.to_numpy(dtype=np.int64))
    write_json(output / "signature_receipt.json", dict(**counts, donor_source_rows=len(source),
               other_compound_rows=len(subset), artifact_sha256=sha256(output / "other_compound_signatures.npz"),
               identity_receipt_sha256=sha256(output / "identity_receipt.json")))
    print(json.dumps(counts), flush=True)


def candidates():
    specs = []
    for ridge in (3, 30, 300):
        specs.append(dict(name=f"lx_local_a{ridge}", ridge=ridge, prior="local", rank=0, fraction=0.0))
        for prior, rank, split_weight in product(("pca", "cross"), (16, 64), (0.5, 1.0)):
            specs.append(dict(name=f"lx_{prior}{rank}_w{split_weight}_a{ridge}", ridge=ridge,
                              prior=prior, rank=rank, fraction=split_weight))
    for ridge, rank in product((3, 30, 300), (16, 64, 0)):
        specs.append(dict(name=f"lx_direct_rank{rank}_a{ridge}", ridge=ridge,
                          prior="direct", rank=rank, fraction=1.0))
    return specs


class LINCSFit:
    def __init__(self, fit, root):
        from src.r11_chemical_probe.dataset import ENTITY

        self.fit = fit
        self.local = TransferFit(fit)
        identity = root / "identity_receipt.json"
        receipt = json.loads((root / "signature_receipt.json").read_text())
        asset = root / "other_compound_signatures.npz"
        if sha256(asset) != receipt["artifact_sha256"] or sha256(identity) != receipt["identity_receipt_sha256"]:
            raise ValueError("Donor signature receipt no longer matches")
        covered = set(json.loads(identity.read_text())["protected_entities"])
        if not (set(fit.train.rows[ENTITY]) | set(fit.query.rows[ENTITY])) <= covered:
            raise ValueError("This donor asset did not audit every entity in the destination fit and query")
        with np.load(asset, allow_pickle=False) as data:
            positions = pd.Index(data["token_id"]).get_indexer(fit.panel)
            self.cols = np.flatnonzero(positions >= 0)
            if len(self.cols) < 10:
                raise ValueError("Too few independently matched landmark genes")
            pos = positions[self.cols]
            self.signature = data["signature"][:, pos]
            self.ecfp = data["ecfp"]
            keep = (data["view0_n"] > 0) & (data["view1_n"] > 0)
            self.left = data["view0"][keep][:, pos]
            self.right = data["view1"][keep][:, pos]
        self.bases = {}
        self.local_outputs = {}

    def basis(self, prior):
        if prior not in self.bases:
            if prior == "pca":
                signal = self.signature - self.signature.mean(axis=0)
                covariance = signal.T @ signal
            else:
                left = self.left - self.left.mean(axis=0)
                right = self.right - self.right.mean(axis=0)
                covariance = left.T @ right
            if not np.isfinite(covariance).all():
                raise ValueError("Invalid LINCS response covariance")
            values, vectors = eigh((covariance + covariance.T) * 0.5, check_finite=False)
            positive = values > max(1e-8, float(values.max()) * 1e-6)
            self.bases[prior] = vectors[:, positive][:, ::-1].astype(np.float32)
        return self.bases[prior]

    @cached_property
    def donor_ridge(self):
        # Center on the LINCS source, then use the same Euclidean ECFP geometry in Tahoe.
        x = self.ecfp - self.ecfp.mean(axis=0)
        y = self.signature - self.signature.mean(axis=0)
        values, basis = eigh(x.T @ x, check_finite=False)
        return np.maximum(values, 0), basis, basis.T @ (x.T @ y)

    @cached_property
    def query_ecfp(self):
        smiles = self.fit.query.rows.iloc[self.local.query_entity_first]["smiles"]
        parents = [structural_keys(clean(value))[1] for value in smiles]
        if not all(parents):
            raise ValueError("Source and query ECFP require resolvable parent structures")
        return fingerprints(parents) - self.ecfp.mean(axis=0)

    def predict(self, spec):
        prior, alpha, rank = spec["prior"], spec["ridge"], spec["rank"]
        if prior == "direct":
            eig, directions, target = self.donor_ridge
            coef = directions @ (target / (eig[:, None] + alpha))
            p = self.query_ecfp @ coef
            if rank:
                space = self.basis("cross")[:, :rank]
                p = (p @ space) @ space.T
            outcome = self.fit.offset.copy()
            outcome[:, self.cols] += p[self.local.query_entity_inverse]
            return outcome
        if alpha not in self.local_outputs:
            self.local_outputs[alpha] = self.local.predict(dict(chemistry="ecfp", alpha=alpha, gene_basis="full",
                                                               line_basis="none", line_rank=0, line_gain=0.0,
                                                               pooling="mean", plate_adjust=False))
        parent = self.local_outputs[alpha]
        if prior == "local":
            return parent
        space = self.basis(prior)[:, :rank]
        residual = parent[:, self.cols] - self.fit.offset[:, self.cols]
        projected = (residual @ space) @ space.T
        output = parent.copy()
        output[:, self.cols] += spec["fraction"] * (projected - residual)
        return output


def predict(fit, spec):
    if not hasattr(fit, "_r13_lincs"):
        root = Path(os.environ["R13_LINCS_ASSET"])
        fit._r13_lincs = LINCSFit(fit, root)
    return fit._r13_lincs.predict(spec)


def pilot(cfg_path, assets, donor, out, task, inner_fold):
    from . import tune

    out.mkdir(parents=True, exist_ok=False)
    os.environ["R13_LINCS_ASSET"] = str(donor)
    fold, role = task // 2, "AB"[task % 2]
    source = tune.location(config(cfg_path), assets, fold, role, "compound")
    data = tune.internal_data(source, inner_fold)

    class Family:
        candidates = staticmethod(candidates)
        predict = staticmethod(predict)

    rows = list(data.evaluate(Family))
    pd.DataFrame(rows).to_csv(out / "inner_grid.tsv", sep="\t", index=False)
    selected = tune.select(rows)
    selected.update(exposure="post-R13 exploratory, only role-local inner development outcome scored",
                    task=task, inner_fold=inner_fold,
                    lincs_mapped_panel_genes=len(data.fit._r13_lincs.cols),
                    lincs_asset_receipt_sha256=sha256(donor / "signature_receipt.json"))
    write_json(out / "selection.json", selected)
    print(json.dumps({k: selected[k] for k in ("best", "best_nonzero", "n_specs", "lincs_mapped_panel_genes")}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for key in ("cache", "manifest", "tmeta", "tgenes", "trows", "lmeta", "phase1", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p = commands.add_parser("pilot")
    for key in ("config", "assets", "donor", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--task", type=int, choices=range(6), default=0)
    p.add_argument("--inner-fold", type=int, choices=(0, 1), default=0)
    a = parser.parse_args()
    if a.command == "prepare":
        prepare(a.cache, a.manifest, a.tmeta, a.tgenes, a.trows, a.lmeta, a.phase1, a.out)
    else:
        pilot(a.config, a.assets, a.donor, a.out, a.task, a.inner_fold)


if __name__ == "__main__":
    main()
