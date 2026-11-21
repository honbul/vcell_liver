"""Apply the frozen R17 predictor to a registered, score-free R18 input slice."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, ENTITY, GROUP, KEY, LINE
from src.r15_lincs.data import table
from src.r15_lincs.design import chemical_features, fold_id
from src.r15_lincs.forecast import LEGACY_CPU, LEGACY_GPU, PROPOSED, StructureMap, position
from src.r16_lincs.predict import CELL_GAIN, GAIN_SUFFIX, HEADS
from src.r17_lincs import predict as r17
from src.r19_plan.r18_access_plan import PLATE, keys
from src.r19_plan.r18_input_bundle import EXPRESSION_TABLES, PLAN_TABLES, check_predictor_scope
from src.r19_plan.r18_score_contract import check_slice_id, frozen_arms, role_names


MODELS = ("cpu", *LEGACY_GPU)
BUNDLE_FILES = {f"{name}.tsv" for name in PLAN_TABLES} | {"landmarks.tsv", "expression.npz"}
PART_FILES = {"forecast.npz", "selection.tsv"}


def verify_files(root, receipt, expected):
    if ({item.name for item in root.iterdir()} != expected | {"receipt.json"}
        or set(receipt["files"]) != expected):
        raise ValueError(f"Unexpected R18 file inventory: {root}")
    for name, digest in receipt["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"R18 file changed after registration: {root / name}")


def model_rows(rows):
    """Map the locked physical metadata to the existing R15/R17 fit interface."""
    return rows.assign(**{
        GROUP: rows.parent,
        ENTITY: rows.parent,
        LINE: rows.cell_id,
        KEY[0]: rows.study + ":" + rows.inst_id,
        CONTROL: rows.study + ":" + rows.rna_plate,
        "plate_id": rows.study + ":" + rows.rna_plate,
        "hours": pd.to_numeric(rows.hours, errors="raise"),
        "dose": pd.to_numeric(rows.dose, errors="raise"),
        "time_h": pd.to_numeric(rows.hours, errors="raise"),
        "dose_uM": pd.to_numeric(rows.dose, errors="raise"),
    })


def vehicle_means(rows, values, columns):
    """Mean control per physical plate, with a separate entry per donor half if present."""
    return {key: values[idx].mean(axis=0) for key, idx in rows.groupby(columns).indices.items()}


def donor_means(rows, values, controls, control_values):
    """Reproduce R15's well, cell, and release aggregation for registered source donors."""
    halves = vehicle_means(controls, control_values, ["study", "plate", "half"])
    plate_control = {key: (halves[(*key, "1")] + halves[(*key, "2")]) * .5
                     for key in keys(rows, ["study", "plate"]).unique()}
    response = values - np.stack([plate_control[key] for key in keys(rows, ["study", "plate"])])
    meta = rows.assign(hours=pd.to_numeric(rows.hours, errors="raise"))
    line_groups = meta.groupby(["study", "parent", "hours", "cell_id"]).indices
    lines = np.stack([response[idx].mean(axis=0) for idx in line_groups.values()])
    metadata = pd.DataFrame(line_groups.keys(), columns=["study", "parent", "hours", "cell_id"])
    release_groups = metadata.groupby(["study", "parent", "hours"]).indices
    releases = np.stack([lines[idx].mean(axis=0) for idx in release_groups.values()])
    metadata = pd.DataFrame(release_groups.keys(), columns=["study", "parent", "hours"])
    return {key: releases[idx].mean(axis=0)
            for key, idx in metadata.groupby(["parent", "hours"]).indices.items()}


class PredictionBank:
    """R17 data interface with no path that can address a score-role expression row."""

    def __init__(self, root, access_reference, d0_reference, slice_id):
        self.case, self.scenario, fold, self.fit = check_slice_id(slice_id)
        self.fold = int(fold)
        _, self.select, _ = role_names(self.fit)
        self.root = root
        self.slice = slice_id
        receipt = read_json(root / "receipt.json")
        access = read_json(access_reference)
        registered = access["files"][slice_id]["predict"]
        if (receipt["slice"] != slice_id or receipt["access_sha256"] != sha256(access_reference)
            or not access["metadata_only"] or access["expression_read"]
            or set(registered) != {f"{name}.tsv" for name in PLAN_TABLES}
            or receipt["predict_plan_sha256"] != registered
            or any(receipt["files"][name] != digest for name, digest in registered.items())):
            raise ValueError("R18 predictor bundle differs from its locked prediction input")
        verify_files(root, receipt, BUNDLE_FILES)
        self.receipt_sha256 = sha256(root / "receipt.json")
        self.predict_plan_sha256 = registered
        gene_reference = read_json(d0_reference)
        if set(gene_reference["gene_axis_sha256"].values()) != {receipt["files"]["landmarks.tsv"]}:
            raise ValueError("R18 predictor bundle changes the R17 landmark order")
        self.landmarks = pd.read_csv(root / "landmarks.tsv", sep="\t", dtype={"pr_gene_id": int})
        genes = len(self.landmarks)
        if genes != gene_reference["landmarks"] or not self.landmarks.pr_gene_id.is_unique:
            raise ValueError("R18 predictor bundle changes the R17 landmark ID support")

        rows = {name: table(root / f"{name}.tsv") for name in PLAN_TABLES}
        check_predictor_scope(rows, self.case, self.scenario, self.fit)
        self.training = model_rows(rows["inner_and_outer_labels"])
        self.outer = model_rows(rows["query_features"])
        self.check_splits()
        self.label_keys = keys(self.training)
        self.role_control = {}
        with np.load(root / "expression.npz", allow_pickle=False) as unpacked:
            shapes = {name: [len(rows[name]), genes] for name in EXPRESSION_TABLES}
            if (receipt["arrays"] != shapes or len(unpacked.files) != len(shapes)
                or set(unpacked.files) != set(shapes)):
                raise ValueError("Unexpected R18 predictor expression sections")
            values = {}
            for name, shape in shapes.items():
                array = unpacked[name]
                if (list(array.shape) != shape or array.dtype != np.float32 or not np.isfinite(array).all()):
                    raise ValueError(f"Invalid R18 predictor expression section: {name}")
                values[name] = array
        self.labels = values["inner_and_outer_labels"]
        for name, role in (("fit_controls", self.fit), ("select_controls", self.select)):
            self.role_control[role] = vehicle_means(rows[name], values[name], PLATE)
        self.prior = {}
        if self.scenario == "B":
            self.prior = donor_means(rows["B_donor_inputs"], values["B_donor_inputs"],
                                     rows["B_donor_controls"], values["B_donor_controls"])
        self.parents = pd.Index(sorted(set(self.training.parent) | set(self.outer.parent)))
        self.fingerprints = None

    def check_splits(self):
        training, outer = self.training, self.outer
        test_fold = outer.parent.map(lambda p: fold_id(p, "R15-outer", 3))
        train_fold = training.parent.map(lambda p: fold_id(p, "R15-outer", 3))
        valid = training.parent.map(lambda p: fold_id(p, f"R15-inner-{self.fold}", 5)).eq(0)
        if (not test_fold.eq(self.fold).all() or train_fold.eq(self.fold).any()
            or not training.inner_phase.eq("select").eq(valid).all()):
            raise ValueError("R18 split metadata disagrees with the R17 parent fold rule")
        test_line = "HUH7" if self.case.startswith("C3_") else "HEPG2"
        fit_lines = {"HEPG2", "HUH7"} if self.case.endswith("HUH7_added") else {"HEPG2"}
        if set(outer.cell_id) != {test_line} or set(training.cell_id) != fit_lines:
            raise ValueError("An R18 target cell differs from the registered design")

    def split(self):
        return tuple(rows.reset_index(drop=True) for rows in (
            self.training.loc[self.training.inner_phase.eq("inner_fit")],
            self.training.loc[self.training.inner_phase.eq("select")], self.training, self.outer,
        ))

    def empty(self):
        return np.empty((0, len(self.landmarks)), dtype=np.float32)

    def response(self, rows, role):
        loc = self.label_keys.get_indexer(keys(rows))
        if (loc < 0).any():
            raise ValueError("R18 predictor cannot access a query treatment response")
        if role == self.select and not self.training.inner_phase.iloc[loc].eq("select").all():
            raise ValueError("R18 select truth can only use the registered validation treatments")
        return self.labels[loc] - self.control(rows, role)

    def control(self, rows, role):
        if role not in self.role_control:
            raise ValueError("R18 predictor cannot access a score-role control")
        available = self.role_control[role]
        if role == self.select:
            selected = self.training.loc[self.training.inner_phase.eq("select")]
            if not keys(rows).isin(keys(selected)).all():
                raise ValueError("R18 select controls are available for validation treatments only")
        requested = keys(rows, PLATE)
        if not requested.isin(available).all():
            raise ValueError("R18 predictor plate lacks its registered DMSO role")
        return np.stack([available[key] for key in requested]) if len(rows) else self.empty()

    def chemistry(self, rows):
        indices = self.parents.get_indexer(rows.parent)
        if (indices < 0).any():
            raise ValueError("R18 predictor parent is outside its registered structures")
        if self.fingerprints is None:
            self.fingerprints = chemical_features(self.parents)
        return self.fingerprints[indices]

    def transfer(self, rows):
        if self.scenario == "A":
            return np.zeros((len(rows), len(self.landmarks)), dtype=np.float32)
        return np.stack([self.prior[key] for key in zip(rows.parent, rows.hours)]) if len(rows) else self.empty()


def cohort_position(bank, train, query, role):
    """Use the R17 position estimator within the registered release and cell contexts."""
    raw = bank.response(train, role)
    train_pos = np.zeros_like(raw)
    query_pos = np.zeros((len(query), len(bank.landmarks)), dtype=np.float32)
    context = ["study", "cell_id"]
    requested = query.groupby(context).indices
    for key, loc in train.groupby(context).indices.items():
        qloc = requested.get(key, np.empty(0, dtype=np.intp))
        y, pos_in, pos_out = position(bank, train.iloc[loc], query.iloc[qloc], role)
        if not np.array_equal(y, raw[loc]):
            raise ValueError("R18 position estimates diverge from the registered response axis")
        train_pos[loc], query_pos[qloc] = pos_in, pos_out
    return raw, train_pos, query_pos


def model_arms(scenario, model):
    gpu_arms = {name + GAIN_SUFFIX for name in LEGACY_GPU}
    return frozen_arms(scenario) - gpu_arms if model == "cpu" else {model + GAIN_SUFFIX}


def model_selections(model):
    return {PROPOSED, CELL_GAIN, *LEGACY_CPU, *HEADS} if model == "cpu" else {model}


def validate_forecasts(arrays, shape, arms):
    if set(arrays) != arms or any(values.shape != shape or values.dtype.kind != "f"
                                 or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("R18 predictor changed a registered arm, row, or gene axis")
    if "B0" in arrays and arrays["B0"].any():
        raise ValueError("R18 B0 must be zero change")


def asset_fingerprint(args, bank):
    """Require pinned structure assets to cover the slice rather than dropping arms."""
    features = args.assets / "features"
    receipt = read_json(args.asset_reference)
    if (sha256(args.assets / "receipt.json") != sha256(args.asset_reference)
            or receipt["r18_expression_read"] or receipt["fitting"]):
        raise ValueError("The R18 comparator input is not the pinned structure-only package")
    asset_files = {"features/chemberta.npz", "features/entities.tsv", "features/receipt.json", "metadata/genes.tsv"}
    if set(receipt["files"]) != asset_files or any(
        sha256(args.assets / name) != digest for name, digest in receipt["files"].items()
    ):
        raise ValueError("An R18 comparator structure or landmark artifact changed")
    paths = {"config": args.config, "entities": features / "entities.tsv"}
    entities = table(paths["entities"])
    matched = entities.loc[entities[ENTITY].isin(bank.parents)]
    if (entities[ENTITY].duplicated().any() or not bank.parents.isin(entities[ENTITY]).all()
        or not matched[ENTITY].eq(matched.smiles).all()):
        raise ValueError("R18 comparator structure file lacks canonical parent coverage")
    if args.model == "cpu":
        StructureMap(args.assets).pos(bank.parents)
        paths["chemberta_receipt"] = features / "receipt.json"
    elif args.model == "XPert_morgan":
        paths["genes"] = args.assets / "metadata" / "genes.tsv"
        if not bank.landmarks.pr_gene_id.astype(str).isin(table(paths["genes"]).token_id).all():
            raise ValueError("R18 XPert landmark gene mapping is incomplete")
    elif args.model == "PRnet-well-adaptation":
        paths["prnet_model"] = args.prnet / "models" / "PRnet.py"
    return {name: sha256(path) for name, path in paths.items()} | {"structure_package": sha256(args.asset_reference)}


def provenance(args, bank):
    return dict(slice=bank.slice, protocol_sha256=sha256(args.protocol),
                predict_plan_sha256=bank.predict_plan_sha256, input_bundle_sha256=bank.receipt_sha256,
                predictor_sha256=sha256(__file__), r17_predictor_sha256=sha256(r17.__file__))


def predict(args, bank):
    path = args.parts / bank.slice / args.model
    if path.exists():
        raise FileExistsError(path)
    assets = asset_fingerprint(args, bank)
    args.role, args.scenario, args.fold = bank.fit, bank.scenario, bank.fold
    arrays, selected = r17.fit_slice(args, bank, *bank.split(), get_position=cohort_position)
    shape = (len(bank.outer), len(bank.landmarks))
    validate_forecasts(arrays, shape, model_arms(bank.scenario, args.model))
    roles = [selected[key] for key in ("fit", "select", "score")]
    validation = bank.training.loc[bank.training.inner_phase.eq("select"), "parent"]
    if (tuple(roles) != role_names(bank.fit) or selected["model"] != args.model
        or selected["fit_groups"] != sorted(set(bank.training.parent))
        or selected["selection_groups"] != sorted(set(validation))
        or set(selected["settings"]) != model_selections(args.model)):
        raise ValueError("R18 predictor returned unregistered fit or selection groups")
    selection = pd.DataFrame([
        dict(slice=bank.slice, model=args.model, arm=arm, fit=roles[0], select=roles[1], score=roles[2],
             position_gain=selected["position_gain"], **{
                 field: json.dumps(value) if isinstance(value, (tuple, list, dict)) else value
                 for field, value in settings.items()})
        for arm, settings in selected["settings"].items()
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".r18-fit-", dir=path.parent) as tmp:
        stage = Path(tmp)
        selection.to_csv(stage / "selection.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecast.npz", **arrays)
        write_json(stage / "receipt.json", dict(provenance(args, bank), model=args.model, assets=assets,
            files={name: sha256(stage / name) for name in sorted(PART_FILES)}))
        stage.rename(path)


def merge(args, bank):
    path = args.out / bank.slice
    if path.exists():
        raise FileExistsError(path)
    stamp = provenance(args, bank)
    shape = (len(bank.outer), len(bank.landmarks))
    arrays, settings, part_hashes = {}, [], {}
    position_gains, asset_versions = set(), set()
    for model in MODELS:
        part = args.parts / bank.slice / model
        receipt = read_json(part / "receipt.json")
        if receipt["model"] != model or any(receipt[key] != val for key, val in stamp.items()):
            raise ValueError(f"R18 predictor part belongs to a different run: {part}")
        verify_files(part, receipt, PART_FILES)
        selection = table(part / "selection.tsv")
        if (set(selection.arm) != model_selections(model) or selection.arm.duplicated().any()
            or not selection.model.eq(model).all()
            or not selection.slice.eq(bank.slice).all()
            or any(not selection[name].eq(role).all() for name, role in zip(
                ("fit", "select", "score"), role_names(bank.fit), strict=True))):
            raise ValueError(f"R18 selection receipt differs from the registered slice: {part}")
        settings.append(selection)
        position_gains.update(selection.position_gain)
        assets = receipt["assets"]
        asset_versions.add((assets.get("structure_package"), assets.get("config")))
        with np.load(part / "forecast.npz", allow_pickle=False) as forecast:
            if len(set(forecast.files)) != len(forecast.files):
                raise ValueError(f"R18 predictor part repeats an arm: {part}")
            part_arrays = dict(forecast)
        validate_forecasts(part_arrays, shape, model_arms(bank.scenario, model))
        arrays.update(part_arrays)
        part_hashes[model] = sha256(part / "receipt.json")
    validate_forecasts(arrays, shape, frozen_arms(bank.scenario))
    if len(position_gains) != 1:
        raise ValueError("R18 comparator parts disagree about the shared position gain")
    if len(asset_versions) != 1:
        raise ValueError("R18 comparator parts used different structure packages or fit configurations")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".r18-merge-", dir=path.parent) as tmp:
        stage = Path(tmp)
        copyfile(bank.root / "query_features.tsv", stage / "rows.tsv")
        copyfile(bank.root / "landmarks.tsv", stage / "landmarks.tsv")
        pd.concat(settings, ignore_index=True).fillna("").to_csv(stage / "selection.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecast.npz", **arrays)
        write_json(stage / "receipt.json", dict(stamp, predict_parts_sha256=part_hashes,
            files={name: sha256(stage / name) for name in ("rows.tsv", "landmarks.tsv", *sorted(PART_FILES))}))
        stage.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "merge"))
    parser.add_argument("--slice", required=True)
    parser.add_argument("--model", choices=MODELS)
    for name in ("bundle", "access-reference", "d0-reference", "protocol", "parts"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("config", "assets", "asset-reference", "prnet", "out"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.action == "predict" and any(item is None for item in
        (args.model, args.config, args.assets, args.asset_reference)
    ):
        parser.error("predict requires --model, --config, --assets, and --asset-reference")
    if args.action == "predict" and args.model == "PRnet-well-adaptation" and args.prnet is None:
        parser.error("PRnet requires --prnet")
    if args.action == "merge" and args.out is None:
        parser.error("merge requires --out")
    bank = PredictionBank(args.bundle, args.access_reference, args.d0_reference, args.slice)
    (predict if args.action == "predict" else merge)(args, bank)


if __name__ == "__main__":
    main()
