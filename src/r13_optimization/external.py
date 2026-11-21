"""Freeze external drug predictions from vehicle counts before reading treated outcomes."""

import argparse
from collections.abc import Mapping
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import (
    Block, CONTROL, ENTITY, GROUP, KEY, LINE, POOL, assemble, check_split, load_data, table,
)
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align
from src.r12_multiaxis.data import Fit, select_panel

from .objectives import GainScores
from .score import draw_contrast
from .tune import checked_forecast, implementation


DATASET = "GSE306429"
COUNT_LAYER = "layers/counts"
ACCEPTED_FAMILIES = {"kernels", "transfer", "neural", "dose_transfer"}
OBS = "obs_row"
VEHICLE_OBS = "control_obs_row"
CONTROL_ROLE = "control_role"
PARENT = "compound_parent"


def verify_hashes(paths):
    for path, checksum in paths.items():
        if sha256(path) != checksum:
            raise ValueError(f"Input differs from its recorded SHA-256: {path}")


def read_manifest(root):
    rec = read_json(root / "receipt.json")
    if rec["dataset"] != DATASET or rec["treated_expression_accessed"] is not False:
        raise ValueError("Expected a metadata-only GSE306429 receipt")
    verify_hashes({root / f"{name}.tsv": value for name, value in rec["output_sha256"].items()})
    return rec


def boolean(values):
    flags = values.astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"Invalid boolean metadata column {values.name}")
    return flags.eq("true")


def select_query(root, rec, line, dose, time, minimum):
    treatments = table(root / "treatment_rows.tsv")
    vehicles = table(root / "control_rows.tsv")
    identity = rec["assay_lines"][line]["cellosaurus_id"]
    select = (treatments.cell_id.eq(line) & treatments.cellosaurus_id.eq(identity)
              & treatments.dose_uM.eq(dose) & treatments.timepoint_hr.eq(time)
              & boolean(treatments.eligible_paired_roles))
    rows = treatments.loc[select].copy()
    if rows.empty:
        raise ValueError("The requested cohort has no identity-resolved, replicate-paired metadata")
    if not rows.identity_status.eq("resolved").all() or rows[PARENT].eq("").any():
        raise ValueError("Eligible cohort contains unresolved molecules")
    matched = vehicles.set_index(OBS, verify_integrity=True).loc[rows[VEHICLE_OBS].to_numpy()]
    for name in ("cell_id", "plate", "library_id", "replicate", CONTROL_ROLE):
        if not np.array_equal(matched[name], rows[name]):
            raise ValueError(f"Vehicle shares neither the expected library nor replicate: {name}")
    if (not matched.compound_name.str.casefold().eq("dmso").all()
            or not np.array_equal(matched[KEY[0]], rows[CONTROL])):
        raise ValueError("Eligibility was based on a mismatched physical vehicle")
    rows[LINE] = identity
    rows["time_h"] = rows.timepoint_hr
    rows["plate_id"] = DATASET + "/" + rows.plate.astype(str) + "/" + rows.library_id.astype(str)
    rows[KEY[0]] = DATASET + "/" + rows[KEY[0]].astype(str)
    rows[CONTROL] = DATASET + "/" + rows[CONTROL].astype(str)
    rows[ENTITY] = DATASET + "/" + rows.compound_name.astype(str)
    rows["smiles"] = rows[PARENT]
    parents = {p: -1 - i for i, p in enumerate(sorted(rows[PARENT].unique()))}
    rows[GROUP] = rows[PARENT].map(parents)
    if rows[OBS].duplicated().any() or rows.duplicated(KEY).any():
        raise ValueError("The cohort contains duplicate evaluated wells")
    valid = rows.groupby(POOL, sort=False)[GROUP].transform("nunique").ge(minimum)
    query = rows.loc[valid].sort_values(OBS).reset_index(drop=True)
    if set(query[CONTROL_ROLE]) != {"A", "B"}:
        raise ValueError("Both physical replicate roles require eligible within-library pools")
    if query.groupby("plate_id")[CONTROL].nunique().gt(1).any():
        raise ValueError("One analysis pool must have a shared physical DMSO control")
    if set(query[OBS]) & set(query[VEHICLE_OBS]):
        raise ValueError("A held-out treatment is recorded as a feature control")
    return rows, query


def cohort_summary(all_rows, query):
    roles = {}
    for role, rows in query.groupby(CONTROL_ROLE):
        roles[role] = dict(groups=int(rows[GROUP].nunique()), wells=len(rows),
                           library_pools=int(rows["plate_id"].nunique()),
                           physical_plates=sorted(rows.plate.unique()))
    paired = set(query.loc[query[CONTROL_ROLE].eq("A"), GROUP]) & set(
        query.loc[query[CONTROL_ROLE].eq("B"), GROUP])
    plates = {
        plate: dict(role=rows[CONTROL_ROLE].iloc[0], wells=len(rows), groups=int(rows[GROUP].nunique()),
                    library_pools=int(rows.plate_id.nunique()))
        for plate, rows in query.groupby("plate")
    }
    return dict(metadata_resolved_wells=len(all_rows), eligible_wells=len(query),
                eligible_groups=int(query[GROUP].nunique()), both_roles_eligible_groups=len(paired),
                by_role=roles, by_physical_plate=plates)


def allowed_training_rows(assets, manifest):
    rows = table(assets / "partition/source_rows.tsv")
    group_mask = table(manifest / "held_in_groups.tsv").set_index(GROUP, verify_integrity=True)
    if set(rows[GROUP]) != set(group_mask.index):
        raise ValueError("The alias-exclusion manifest must cover the full Tahoe R12 development partition")
    kept = rows.loc[rows[GROUP].isin(group_mask.index[boolean(group_mask.eligible_for_external_model_fit)])]
    if kept.empty or kept.duplicated(KEY).any():
        raise ValueError("No unique alias-disjoint training rows")
    verify_disjoint_identity(kept, manifest)
    return kept.reset_index(drop=True)


def verify_disjoint_identity(rows, manifest):
    aliases = table(manifest / "compound_identity.tsv")
    parents = {p for entry in aliases.candidate_parents for p in entry.split("|") if p}
    parents.update(aliases[PARENT].loc[aliases[PARENT].ne("")])
    excluded = table(manifest / "excluded_tahoe_compounds.tsv")
    if set(rows[ENTITY]) & set(excluded.drug):
        raise ValueError("A held-out assay alias remains among the Tahoe training names")
    fitted_parents = set()
    for smile in rows.smiles.unique():
        parent = canonical_parent(smile)
        if parent is None:
            raise ValueError("Missing chemical identity on a fitted Tahoe molecule")
        fitted_parents.add(parent[1])
    if fitted_parents & parents:
        raise ValueError("An external molecular parent remains among the fitted Tahoe aliases")


def load_train_data(rows, manifest, cfg):
    r11 = Path(cfg["r11_assets"])
    genes = table(manifest / "genes.tsv").sort_values("token_id").reset_index(drop=True)
    for identifier in ("token_id", "gene_symbol", "external_col", "ensembl_id"):
        if genes[identifier].duplicated().any():
            raise ValueError(f"Expected one-to-one gene mapping for {identifier}")
    reference = table(r11 / "metadata/genes.tsv").set_index("token_id", verify_integrity=True)
    for name in ("gene_symbol", "ensembl_id"):
        if not np.array_equal(reference.loc[genes.token_id, name], genes[name]):
            raise ValueError(f"Gene coordinate differs from the Tahoe metadata: {name}")
    wanted = pd.MultiIndex.from_frame(rows[KEY])
    ref, counts = load_data(
        r11 / "source", "treated", panel=genes.token_id.to_numpy(),
        mask=lambda v: pd.MultiIndex.from_frame(v[KEY]).isin(wanted),
    )
    index = align(rows, ref)
    if not np.array_equal(rows[GROUP], ref.iloc[index][GROUP]):
        raise ValueError("Tahoe outcome alignment changed compound groups")
    raw = counts[index]
    # Reuse the standard variance ranking on the compact column axis. Its
    # first three entries are reserved for Tahoe's padding tokens.
    import scipy.sparse as sp

    padded = sp.hstack((sp.csr_matrix((len(rows), 3)), raw), format="csr")
    ranked = select_panel(padded, cfg["panel_size"]) - 3
    return genes.iloc[ranked].reset_index(drop=True), raw[:, ranked].toarray().astype(np.float32)


def read_count_rows(path, expected_shape, indices, gene_columns):
    import h5py

    unique, index = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
    if unique.min() < 0 or unique.max() >= expected_shape[0]:
        raise ValueError("Metadata maps outside the immutable count matrix")
    with h5py.File(path, "r") as handle:
        layer = handle[COUNT_LAYER]
        if tuple(layer.shape) != tuple(expected_shape):
            raise ValueError("Count layer shape differs from the released dataset receipt")
        data = layer[unique, :].astype(np.float64)
    totals = data.sum(axis=1)
    if not np.isfinite(data).all() or np.any(data < 0) or np.any(totals <= 0):
        raise ValueError("Invalid released count rows")
    normalized = np.log1p(data[:, gene_columns] * (10000 / totals[:, None]))
    return normalized[index].astype(np.float32)


def molecular_inputs(query, r11, checkpoint):
    # Wells may repeat an identity across physical plates. Encode each exact
    # metadata entity once and align the immutable vector by the entity ID.
    entities = query[[ENTITY, "smiles"]].drop_duplicates().set_index(ENTITY, verify_integrity=True)
    with np.load(r11 / "features/features.npz") as cache:
        bits = cache["E"].shape[1]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits, includeChirality=False)
    molecules = [Chem.MolFromSmiles(smile) for smile in entities.smiles]
    if any(m is None for m in molecules):
        raise ValueError("External fit requires all prospectively eligible molecules to parse")
    values = {"E": np.stack([generator.GetFingerprintAsNumPy(m) for m in molecules])}
    if checkpoint is not None:
        values["C"] = frozen_cls(entities.smiles.tolist(), r11, checkpoint)
    positions = entities.index.get_indexer(query[ENTITY])
    return {k: v[positions] for k, v in values.items()}


def released_chemistry(r11):
    features = r11 / "features/features.npz"
    receipt = read_json(features.parent / "receipt.json")
    verify_hashes({features: receipt["outputs"][features.name]})
    with np.load(features) as file:
        return {key: file[key] for key in ("entity", "E", "C")}


def verify_disjoint_fingerprints(train_rows, chemical, query):
    indices = pd.Index(chemical["entity"]).get_indexer(train_rows[ENTITY].unique())
    if (indices < 0).any():
        raise ValueError("A fitted Tahoe chemical entity has no released fingerprint")
    reference = chemical["E"][indices].astype(np.float32)
    queries = query["E"].astype(np.float32)
    if np.any(2 * (queries @ reference.T) == queries.sum(axis=1)[:, None] + reference.sum(axis=1)):
        raise ValueError("An exact released ECFP vector duplicates an external evaluation molecule")


def frozen_cls(smiles, r11, checkpoint):
    import torch
    from transformers import AutoModel, AutoTokenizer

    rec = read_json(r11 / "features/receipt.json")
    verify_hashes({checkpoint / name: value for name, value in rec["checkpoint_files"].items()})
    model, loading = AutoModel.from_pretrained(
        checkpoint, local_files_only=True, output_loading_info=True, add_pooling_layer=False,
    )
    if (loading["missing_keys"] or loading["mismatched_keys"]
            or set(loading["unexpected_keys"]) != set(rec["unused_regression_head_keys"])):
        raise ValueError("Frozen released chemical checkpoint does not match the Tahoe feature contract")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        batch = tokenizer(smiles, padding=True, truncation=False, return_tensors="pt")
        length = model.config.max_position_embeddings - tokenizer.pad_token_id - 1
        if batch.input_ids.shape[1] > length or not torch.all(batch.input_ids[:, 0] == tokenizer.cls_token_id):
            raise ValueError("External CLS tokenization altered the released feature contract")
        return model(**batch).last_hidden_state[:, 0].cpu().numpy()


class ExternalSource:
    """Expose Tahoe fit data and one physical external vehicle role to R12 Fit."""

    optional_cells = None

    def __init__(self, cfg, assets, role, train_rows, train_counts, query, qfeat, controls, panel, chemistry):
        self.cfg, self.assets, self.role = cfg, assets, role
        self.r11 = Path(cfg["r11_assets"])
        self.fit_rows, self.fit_counts = train_rows, train_counts
        self.query_rows, self.query_controls, self.query_features = query, controls, qfeat
        self.panel = panel
        self.chemical = chemistry
        self.control_rows, self.control_counts = load_data(self.r11 / "source", role, panel=panel)

    def assemble(self, rows, control_rows, controls, observed):
        own = f"control_{self.role}"
        expected = rows[own].to_numpy() if own in rows else None
        clean = rows.drop(columns=["control_A", "control_B"], errors="ignore")
        result = assemble(clean, control_rows, controls, self.chemical, observed)
        if not result.rows[KEY].equals(rows[KEY].reset_index(drop=True)):
            raise ValueError("Tahoe own-control join changed fit row identities")
        if expected is not None and not np.array_equal(result.rows[CONTROL], expected):
            raise ValueError("Tahoe matched vehicle differs from the recorded role")
        return result

    def block(self, fit, query, panel=None, observe_query=False):
        if observe_query or panel is not None:
            raise ValueError("External source only supports one fixed panel with sealed query outcomes")
        if fit is not self.fit_rows or query is not self.query_rows:
            raise ValueError("External adapter cannot use alternative fit groups or rows")
        training = self.assemble(fit, self.control_rows, self.control_counts.toarray(), self.fit_counts)
        heldout = Block(query.copy(), self.query_controls, self.query_features)
        check_split(training, heldout)
        return training, heldout, self.panel

    def control_block(self, panel):
        if not np.array_equal(panel, self.panel):
            raise ValueError("Unexpected alternate gene panel")
        existing = self.control_rows.rename(columns={KEY[0]: CONTROL})
        role_rows = self.query_rows[[CONTROL, LINE, "plate_id"]]
        accept = ~role_rows.duplicated([CONTROL, LINE])
        roles = pd.concat((existing, role_rows.loc[accept]), ignore_index=True)
        values = np.concatenate((self.control_counts.toarray(), self.query_controls[accept]))
        return roles, values


def validate_recipe(recipe, args):
    want = (DATASET, args.line, args.dose, args.time, sha256(args.manifest / "receipt.json"))
    seen = tuple(recipe[k] for k in ("dataset", "line", "dose_uM", "time_h", "manifest_sha256"))
    if want != seen:
        raise ValueError("Recipe targets another cohort or metadata receipt")
    if not recipe["selection_rule"].strip() or not recipe["tahoe_receipts"]:
        raise ValueError("A frozen recipe must link the Tahoe-only selection evidence")
    if recipe["primary"] not in recipe["arms"]:
        raise ValueError("One primary challenger must be designated in the frozen recipe")
    for label, arm in recipe["arms"].items():
        if (label in ("B0", "anchor") or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label)
                or arm["family"] not in ACCEPTED_FAMILIES
                or not isinstance(arm["specification"], Mapping)
                or arm["specification"].get("permute")
                or not np.isfinite(arm["gain"]) or arm["gain"] <= 0):
            raise ValueError(f"Invalid externally frozen candidate {label}")
        if arm["specification"] not in implementation(arm["family"]).candidates():
            raise ValueError(f"External arm {label} differs from its registered Tahoe specification")
    verify_hashes(recipe["tahoe_receipts"])


def verify_dose_rows(assets, manifest, groups):
    raw = table(assets / "doses/source/treated_rows.tsv")
    verify_disjoint_identity(raw.loc[raw[GROUP].isin(groups)], manifest)


def forecast(args, rec, recipe, rows, cfg):
    validate_recipe(recipe, args)
    verify_hashes({args.h5: rec["download"]["sha256"]})
    train_rows = allowed_training_rows(args.assets, args.manifest)
    if any(arm["family"] == "dose_transfer" for arm in recipe["arms"].values()):
        verify_dose_rows(args.assets, args.manifest, set(train_rows[GROUP]))
    r11 = Path(cfg["r11_assets"])
    chemistry = released_chemistry(r11)
    features = molecular_inputs(rows, r11, args.chemical_checkpoint)
    verify_disjoint_fingerprints(train_rows, chemistry, features)
    genes, train_counts = load_train_data(train_rows, args.manifest, cfg)
    shape = rec["matrix_descriptors_only"][COUNT_LAYER]["shape"]
    controls = read_count_rows(args.h5, shape, rows[VEHICLE_OBS], genes.external_col.to_numpy())
    results = {"B0": np.zeros_like(controls), "anchor": np.empty_like(controls)}
    results.update({name: np.empty_like(controls) for name in recipe["arms"]})
    for role in ("A", "B"):
        mask = rows[CONTROL_ROLE].eq(role).to_numpy()
        local = rows.loc[mask].reset_index(drop=True)
        source = ExternalSource(
            cfg, args.assets, role, train_rows, train_counts, local,
            {k: v[mask] for k, v in features.items()}, controls[mask], genes.token_id.to_numpy(),
            chemistry,
        )
        fit = Fit(source, source.fit_rows, source.query_rows, observe_query=False)
        if fit.query.treated is not None:
            raise ValueError("The external query has treated values")
        results["anchor"][mask] = fit.offset
        for label, arm in recipe["arms"].items():
            module = implementation(arm["family"])
            spec = arm["specification"]
            if hasattr(module, "available") and not module.available(fit, spec):
                raise ValueError(f"Prospectively frozen features absent for {label}")
            ((_, prediction),) = checked_forecast(module, fit, [spec])
            results[label][mask] = fit.offset + arm["gain"] * (prediction - fit.offset)

    args.out.mkdir(parents=True, exist_ok=False)
    paths = []
    for label, values in {"vehicle": controls, **results}.items():
        name = f"{label}.npy"
        np.save(args.out / name, values, allow_pickle=False)
        paths.append(name)
    for name, frame in (("genes.tsv", genes), ("rows.tsv", rows),
                        ("tahoe_fit_identity.tsv", train_rows[KEY + [GROUP, ENTITY, "smiles"]])):
        frame.to_csv(args.out / name, index=False, sep="\t")
        paths.append(name)
    active = {arm["family"] for arm in recipe["arms"].values()} | {"external", "score", "tune", "objectives"}
    code = {str(p): sha256(p) for p in Path("src/r13_optimization").glob("*.py") if p.stem in active}
    for raw in ("src/r12_multiaxis/data.py", "src/r11_chemical_probe/dataset.py",
                "src/r11_chemical_probe/metrics.py", "src/r11_chemical_probe/models.py"):
        code[raw] = sha256(raw)
    write_json(args.out / "freeze.json", dict(
        dataset=DATASET, recipe_sha256=sha256(args.recipe), config_sha256=sha256(args.config),
        manifest_sha256=sha256(args.manifest / "receipt.json"), h5_sha256=rec["download"]["sha256"],
        files={name: sha256(args.out / name) for name in paths}, code=code, arms=list(results),
        fit_groups=int(train_rows[GROUP].nunique()), fit_wells=int(train_rows[KEY[0]].nunique()),
        normalization="log1p(count * 10000 / all released assay-gene counts)",
        vehicle_only_rows=sorted(map(int, rows[VEHICLE_OBS].unique())), treated_expression_accessed=False,
    ))


def score_frozen(args, rec, recipe, rows, cfg):
    frozen = read_json(args.out / "freeze.json")
    if frozen["treated_expression_accessed"] is not False:
        raise ValueError("Prediction freeze receipt did not attest to label-free inference")
    validate_recipe(recipe, args)
    verify_hashes({
        args.recipe: frozen["recipe_sha256"], args.config: frozen["config_sha256"],
        args.manifest / "receipt.json": frozen["manifest_sha256"], args.h5: frozen["h5_sha256"],
        **{args.out / name: value for name, value in frozen["files"].items()},
        **frozen["code"],
    })
    expected = {"B0", "anchor", *recipe["arms"]}
    if set(frozen["arms"]) != expected:
        raise ValueError("Frozen predictions differ from the predeclared arm set")
    frozen_rows = table(args.out / "rows.tsv")
    if not frozen_rows[KEY + [OBS, VEHICLE_OBS, GROUP]].equals(rows[KEY + [OBS, VEHICLE_OBS, GROUP]]):
        raise ValueError("Frozen prediction rows differ from the prospectively eligible cohort")
    if sorted(map(int, rows[VEHICLE_OBS].unique())) != frozen["vehicle_only_rows"]:
        raise ValueError("Vehicle source rows differ from the prediction receipt")
    genes = table(args.out / "genes.tsv")
    vehicle = np.load(args.out / "vehicle.npy", allow_pickle=False)
    predictions = {name: np.load(args.out / f"{name}.npy", allow_pickle=False) for name in sorted(expected)}
    if any(matrix.shape != (len(rows), len(genes)) or not np.isfinite(matrix).all()
           for matrix in (vehicle, *predictions.values())):
        raise ValueError("Invalid frozen arm or vehicle dimensions")
    if np.count_nonzero(predictions["B0"]):
        raise ValueError("No-change reference contains a treatment effect")
    # Allocate one immutable scoring location before opening external outcomes.
    args.scores.mkdir(parents=True, exist_ok=False)
    shape = rec["matrix_descriptors_only"][COUNT_LAYER]["shape"]
    treated = read_count_rows(args.h5, shape, rows[OBS], genes.external_col.to_numpy())
    measured = treated - vehicle
    # A shared within-library DMSO vector cancels from the centered contrast.
    groups = sorted(rows[GROUP].unique())
    settings = cfg["evaluation"]
    draws = np.random.default_rng(settings["seed"]).exponential(size=(settings["replicates"], len(groups)))
    draws /= draws.mean(axis=1, keepdims=True)
    scopes = [("combined", "", np.ones(len(rows), dtype=bool))]
    scopes.extend((role, "", rows[CONTROL_ROLE].eq(role).to_numpy()) for role in ("A", "B"))
    for plate, plate_rows in rows.groupby("plate"):
        role = plate_rows[CONTROL_ROLE].unique()
        if len(role) != 1:
            raise ValueError("Physical plate crosses replicate roles")
        scopes.append((role[0], plate, rows.plate.eq(plate).to_numpy()))
    output = []
    for role, plate, mask in scopes:
        scope = rows.loc[mask].reset_index(drop=True)
        metric = Contrast(scope, cfg["minimum_groups"])
        local = measured[mask]
        group_index = pd.Index(groups).get_indexer(scope[GROUP])
        denominator = draws @ np.bincount(group_index, weights=metric.primary_weight, minlength=len(groups))
        reference = draw_contrast(metric, predictions["B0"][mask], local, draws, group_index)
        base = metric.point(predictions["B0"][mask], local)
        diagnostics = GainScores(scope, local, np.zeros_like(local), cfg["minimum_groups"])
        for name, pred in predictions.items():
            prediction = pred[mask]
            point, absolute, _, _ = metric.points(prediction, local)
            effect = (draw_contrast(metric, prediction, local, draws, group_index) - reference) / denominator
            confidence = 0.99 if name == recipe["primary"] and role == "combined" else 0.95
            low, high = np.quantile(effect, [(1 - confidence) / 2, (1 + confidence) / 2])
            alignment = diagnostics.diagnostics(prediction)
            output.append(dict(arm=name, replicate_role=role, physical_plate=plate, groups=scope[GROUP].nunique(),
                               wells=len(scope), primary=name == recipe["primary"] and role == "combined",
                               contrast_rmse=point, absolute_rmse_descriptive=absolute,
                               contrast_delta_vs_B0=point - base, confidence=confidence,
                               low=float(low), high=float(high),
                               alignment_coverage=alignment[0], centered_cosine=alignment[1]))
    pd.DataFrame(output).to_csv(args.scores / "metrics.tsv", sep="\t", index=False)
    write_json(args.scores / "receipt.json", dict(
        dataset=DATASET, line=args.line, dose_uM=args.dose, time_h=args.time,
        prediction_freeze_sha256=sha256(args.out / "freeze.json"),
        metric_sha256=sha256(args.scores / "metrics.tsv"), replicate_rule=rec["control_rule"],
        primary=recipe["primary"], comparator="B0", bootstrap_draws=len(draws),
        outcome_rows_read=sorted(map(int, rows[OBS])), treated_expression_accessed=True,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "predict", "score"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--chemical-checkpoint", type=Path)
    parser.add_argument("--line", default="A549")
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=24.0)
    args = parser.parse_args()
    cfg = config(args.config)
    rec = read_manifest(args.manifest)
    all_rows, rows = select_query(args.manifest, rec, args.line, args.dose, args.time, cfg["minimum_groups"])
    if args.action == "plan":
        train = allowed_training_rows(args.assets, args.manifest)
        r11 = Path(cfg["r11_assets"])
        verify_disjoint_fingerprints(train, released_chemistry(r11), molecular_inputs(rows, r11, None))
        summary = cohort_summary(all_rows, rows)
        summary.update(line=args.line, dose_uM=args.dose, time_h=args.time, manifest_sha256=sha256(
            args.manifest / "receipt.json"), tahoe_fit_groups=int(train[GROUP].nunique()),
            exact_ecfp_overlap=False,
            treated_expression_accessed=False)
        args.out.mkdir(parents=True, exist_ok=False)
        rows.to_csv(args.out / "eligible_rows.tsv", sep="\t", index=False)
        write_json(args.out / "cohort.json", summary)
    elif args.recipe is None:
        parser.error("A prospectively locked --recipe is required")
    elif args.action == "predict":
        forecast(args, rec, read_json(args.recipe), rows, cfg)
    elif args.scores is None:
        parser.error("Score requires a new --scores directory")
    else:
        score_frozen(args, rec, read_json(args.recipe), rows, cfg)


if __name__ == "__main__":
    main()
