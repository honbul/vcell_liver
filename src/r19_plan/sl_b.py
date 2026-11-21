"""Compare S/L residual arms around a common exact-dose B donor and R17 anchor."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.metrics import Contrast
from src.r15_lincs.design import fold_id
from src.r15_lincs.estimators import parent_average
from src.r15_lincs.forecast import choose_proposed, grid_chemical, position
from src.r16_lincs.predict import gain_choice
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_score import checked_rows
from src.r19_plan.r18_access_plan import PLATE, keys
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.r18_predict import PredictionBank, model_rows, vehicle_means
from src.r19_plan.sl_predict import FULL_ARMS, SETTINGS, PairedResult, choose_variant, fit_mean, predictor_code, sl_fit
from src.r19_plan.sl_workflow import compare
from src.r19_plan.t123_workflow import indexed_means, responses, uniform
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes, read_plan, reference, slice_axes
from src.r19_plan.x_workflow import score_delta

TARGET = "HEPG2"
DONORS = {"A375", "A549", "HT29", "MCF7"}
FULL_B = FULL_ARMS | {"B_cell", "B_blend_anchor", "B_cell_parent_shuffle"}
ARRAYS = {"full": FULL_B, "neural_only": set(SETTINGS)}
CONFIG_SHA256 = "dcbcd7f77734d3e19f8231628f079f08a6a9a8fe372dd9f6f61c76f1951c8394"


def access(args):
    ref = reference(args.pin)
    if ref["parents"] != 493 or set(ref["cells"]) != DONORS | {TARGET}:
        raise ValueError("S/L B requires the matched X cell and parent registry")
    if sha256(args.config) != CONFIG_SHA256:
        raise ValueError("S/L B must use the same training configuration as S/L A")
    return ref


def inventory(args, ref, key):
    target, _, fold, fit, select, _ = slice_axes(key, ref["cells"])
    if target != TARGET:
        raise ValueError("S/L B scores only the registered HEPG2 context")
    _, _, main = read_plan(args.access, args.pin, key, "predict")
    query = main["query"]
    source_context = min(DONORS)
    _, _, train_plan = read_plan(args.access, args.pin, f"{source_context}-f{fold}-{fit}", "predict")
    train = train_plan["labels"].loc[train_plan["labels"].cell_id.eq(TARGET)].copy()
    nested = train.parent.map(lambda p: fold_id(p, f"R15-inner-{fold}", 5) == 0)
    train["inner_phase"] = nested.map({True: "select", False: "inner_fit"})
    combined_fit = pd.concat((main["fit_controls"], train_plan["fit_controls"]), ignore_index=True)
    fit_dmsos = combined_fit.loc[combined_fit.cell_id.eq(TARGET)].drop_duplicates()
    _, _, rotated = read_plan(args.access, args.pin, f"{source_context}-f{fold}-{select}", "predict")
    select_dmsos = rotated["fit_controls"].loc[rotated["fit_controls"].rna_plate.isin(train.loc[nested].rna_plate)]
    control_axis(fit_dmsos, pd.concat((train, query)), fit)
    control_axis(select_dmsos, train.loc[nested], select)

    source, vehicles = [main["labels"]], [main["fit_controls"].loc[main["fit_controls"].cell_id.isin(DONORS)]]
    for donor in sorted(DONORS):
        donor_key = f"{donor}-f{fold}-{fit}"
        _, _, observed = read_plan(args.access, args.pin, donor_key, "score")
        _, _, plan = read_plan(args.access, args.pin, donor_key, "predict")
        donor_rows = observed["labels"]
        if (not donor_rows.cell_id.eq(donor).all() or set(donor_rows.parent) != set(query.parent)
                or not donor_rows.drop(columns="level3_row_index").equals(plan["query"])):
            raise ValueError("Held B donor treatment metadata disagree with their previously registered query")
        controls = plan["fit_controls"]
        controls = controls.loc[controls.cell_id.eq(donor) & controls.rna_plate.isin(donor_rows.rna_plate)]
        control_axis(controls, donor_rows, fit)
        source.append(donor_rows)
        vehicles.append(controls)
    donors = pd.concat(source, ignore_index=True)
    donor_dmsos = pd.concat(vehicles, ignore_index=True).drop_duplicates()
    control_axis(donor_dmsos, donors, fit)

    parents = set(train.parent) | set(query.parent)
    if (len(parents) != ref["parents"] or set(train.parent) & set(query.parent)
            or "level3_row_index" in query or "level3_row_index" not in train
            or not train.cell_id.eq(TARGET).all() or not query.cell_id.eq(TARGET).all()
            or not all(set(rows.parent) == parents for _, rows in donors.groupby("cell_id"))
            or set(donors.cell_id) != DONORS
            or any(not rows.hours.eq("6").all() or not rows.dose.eq("10").all() for rows in (train, query, donors))):
        raise ValueError("S/L B parent, target, donor, time or exact concentration support changed")
    check_unique(pd.concat((train, query, donors, fit_dmsos, select_dmsos, donor_dmsos), ignore_index=True))
    return query, (train, donors, fit_dmsos, select_dmsos, donor_dmsos)


class DonorBank(PredictionBank):
    """Use only allowed HEPG2 training contrasts and complete nonhepatic B donors."""

    def __init__(self, query, tables, raw, genes, fold, fit, select):
        train, donors, fit_dmsos, select_dmsos, donor_dmsos = tables
        labels, donor_values, fit_values, select_values, donor_controls = np.split(
            raw, np.cumsum([len(rows) for rows in tables[:-1]]))
        self.case, self.scenario = "SL", "B"
        self.fold, self.fit, self.select = fold, fit, select
        self.landmarks = genes
        self.training, self.outer = model_rows(train), model_rows(query)
        self.check_splits()
        self.parents = pd.Index(sorted(set(train.parent) | set(query.parent)))
        self.fingerprints = None
        self.label_keys, self.labels = keys(self.training), labels
        self.role_control = {fit: vehicle_means(fit_dmsos, fit_values, PLATE),
                             select: vehicle_means(select_dmsos, select_values, PLATE)}
        donor_delta = responses(donors, donor_values, donor_dmsos, donor_controls)
        aggregate = uniform(indexed_means(donors, donor_delta), self.parents, sorted(DONORS))
        self.prior = dict(zip(((p, 6) for p in self.parents), aggregate, strict=True))


def permute_donors(rows, value):
    _, first, expand = np.unique(rows.parent, return_index=True, return_inverse=True)
    permutation = np.random.default_rng(1501).permutation(len(first))
    return value[first[permutation]][expand]


def fit_paired(args, bank, measured):
    inner, select, train, held = bank.split()
    truth = bank.response(select, bank.select)
    y, fit_pos, select_pos = position(bank, inner, select, bank.fit)
    raw, train_pos, held_pos = position(bank, train, held, bank.fit)
    zeros = np.zeros_like(truth)
    position_gain = gain_choice(select, truth, select_pos, zeros)
    select_donor, held_donor = bank.transfer(select), bank.transfer(held)
    donor_gain = gain_choice(select, truth, select_donor, zeros)
    (rule, kernel_gain), objective = choose_proposed(
        bank, inner, select, y, truth, fit_pos, select_pos, position_gain)
    select_pos, held_pos = position_gain * select_pos, position_gain * held_pos
    anchors = select_pos + donor_gain * select_donor, held_pos + donor_gain * held_donor
    inner_y = y - position_gain * fit_pos - bank.transfer(inner)
    outer_y = raw - position_gain * train_pos - bank.transfer(train)
    kernels = (grid_chemical(bank, inner, select, inner_y, [rule])[rule],
               grid_chemical(bank, train, held, outer_y, [rule])[rule])
    inner_fit = sl_fit(args, bank, inner, select, inner_y, measured)
    outer_fit = sl_fit(args, bank, train, held, outer_y, measured)
    arms = {name: choose_variant(args, name, inner_fit, outer_fit, select, truth, anchors, kernels)
            for name in SETTINGS}
    arms["L0"] = arms["B_R17"] = arms["S0"]
    for control in ("drug_shuffle", "context_constant", "context_shuffle"):
        arms[f"S0_{control}"] = choose_variant(args, "S0", inner_fit, outer_fit, select, truth, anchors, kernels,
                                               control=control)
    arms["B_context"] = choose_variant(args, "S0", inner_fit, outer_fit, select, truth, anchors,
                                      tuple(np.zeros_like(k) for k in kernels), control="drug_constant")
    zero = np.zeros_like(held_pos)
    baseline = {
        "B0": (zero, {}),
        "B_pos": (held_pos, dict(gain=position_gain)),
        "B_cell": (donor_gain * held_donor, dict(gain=donor_gain)),
        "B_cell_parent_shuffle": (donor_gain * permute_donors(held, held_donor), dict(gain=donor_gain, seed=1501)),
        "B_blend_anchor": (anchors[1], dict(position_gain=position_gain, donor_gain=donor_gain)),
        "B_R15": (held_pos + held_donor + kernel_gain * kernels[1], dict(rule=rule, gain=kernel_gain, objective=objective)),
    }
    for arm, inner_delta, outer_delta, offsets in (
        ("B_mean", y, raw, (zeros, zero)),
        ("B_resmean", y-position_gain*fit_pos-donor_gain*bank.transfer(inner),
         raw-position_gain*train_pos-donor_gain*bank.transfer(train), anchors),
    ):
        gain, objective = fit_mean(inner, inner_delta, truth, select, offsets[0])
        effect = parent_average(train, outer_delta).mean(axis=0).astype(np.float32)
        baseline[arm] = offsets[1] + gain * effect, dict(gain=gain, objective=objective)
    arms.update({name: PairedResult(values, values, info) for name, (values, info) in baseline.items()})
    return arms


def lineage(args):
    here = Path(__file__)
    return dict(access_sha256=sha256(args.pin), protocol_sha256=sha256(args.protocol),
                config_sha256=sha256(args.config), code_sha256=predictor_code() |
                {name: sha256(here.with_name(name)) for name in (here.name, "sl_workflow.py", "t123_workflow.py")})


def predict(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    ref = access(args)
    query, tables = inventory(args, ref, args.slice)
    genes, columns = matrix_axes(args.access, ref)
    _, _, fold, fit, select, _ = slice_axes(args.slice, ref["cells"])
    raw = extract_matrix(ref["matrix"], pd.concat(tables, ignore_index=True), columns, genes.pr_gene_id.to_numpy())
    bank = DonorBank(query, tables, raw, genes, fold, fit, select)
    train, donors, fit_dmsos, select_dmsos, donor_dmsos = tables
    dose = train.set_index("inst_id", verify_integrity=True).dose
    arms = fit_paired(args, bank, dose)
    if set(arms) != FULL_B:
        raise ValueError("S/L B variant inventory changed")
    shape = len(query), len(genes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".sl-b-predict-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage/"full.npz", **{n: checked_array(a.full, shape, n) for n, a in arms.items()})
        np.savez_compressed(stage/"neural_only.npz", **{
            n: checked_array(arms[n].neural_only, shape, n) for n in SETTINGS})
        write_json(stage/"selection.json", {n: a.selection for n, a in arms.items()})
        support = dict(held_parents=query.parent.nunique(), held_wells=len(query), held_plates=query.rna_plate.nunique(),
                       train_parents=train.parent.nunique(), train_wells=len(train), donor_wells=len(donors),
                       fit_dmsos=len(fit_dmsos), select_dmsos=len(select_dmsos), donor_fit_dmsos=len(donor_dmsos))
        write_json(stage/"receipt.json", dict(lineage(args), slice=args.slice, support=support,
            query_sha256=ref["slices"][args.slice]["predict"]["query.tsv"],
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)
    return sha256(args.out/"receipt.json")


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("S/L B scoring requires a new score path and the independent freeze digest")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    ref = access(args)
    relevant = sorted(key for key in ref["slices"] if key.startswith(TARGET+"-"))
    genes, columns = matrix_axes(args.access, ref)
    inventory, digests, choices, support = {}, {}, [], []
    arrays = {kind: {arm: [] for arm in names} for kind, names in ARRAYS.items()}
    source = lineage(args)
    for key in relevant:
        _, _, plan = read_plan(args.access, args.pin, key, "predict")
        query = inventory[key] = plan["query"]
        path = args.forecasts/key
        receipt = read_json(path/"receipt.json")
        if (any(receipt[k] != h for k, h in source.items()) or receipt["slice"] != key
                or receipt["query_sha256"] != ref["slices"][key]["predict"]["query.tsv"]
                or set(receipt["files"]) != {"full.npz", "neural_only.npz", "selection.json"}
                or any(sha256(path/n) != h for n, h in receipt["files"].items())):
            raise ValueError("S/L B predictions or their metadata lineage disagree with the frozen contract")
        digests[key] = sha256(path/"receipt.json")
        choices.append(dict(slice=key, arms=read_json(path/"selection.json")))
        support.append(dict(slice=key, **receipt["support"]))
        for kind, names in ARRAYS.items():
            with np.load(path/(kind+".npz"), allow_pickle=False) as file:
                if set(file.files) != names:
                    raise ValueError("S/L B forecast arm inventory changed")
                for arm in names:
                    values = checked_array(file[arm], (len(query), len(genes)), arm)
                    if scoring:
                        arrays[kind][arm].append(values)
                if kind == "full" and any(not np.array_equal(file[arm], file["B_R17"]) for arm in ("S0", "L0")):
                    raise ValueError("S/L B S0, L0 and B_R17 must share one prediction")
    rows = checked_rows(pd.concat([query.assign(fold=slice_axes(k, ref["cells"])[2],
                                       role=slice_axes(k, ref["cells"])[3]) for k, query in inventory.items()],
                                  ignore_index=True), None)
    if rows.parent.nunique() != ref["parents"] or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("S/L B forecasts lost a parent or DMSO rotation")
    stamp = dict(source, slices=digests)
    if not scoring:
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if read_json(args.freeze) != stamp:
        raise ValueError("S/L B arrays changed after freeze")
    truth = np.concatenate([score_delta(args, key, query, genes, columns) for key, query in inventory.items()])
    model = model_rows(rows)
    model["plate_id"] = model.plate_id + ":" + model.fold.astype(str) + ":" + model.role
    contrast = Contrast(model, minimum=2)
    paired, diagnostics, robustness = [], [], []
    for kind, models in arrays.items():
        effects, technical, strata = compare(rows, truth, {arm: np.concatenate(models[arm])
                                                          for arm in sorted(models)}, contrast, kind)
        paired.extend(effects)
        diagnostics.extend(technical)
        robustness.extend(strata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".sl-b-score-") as tmp:
        stage = Path(tmp)
        for name, data in (("paired", paired), ("diagnostics", diagnostics), ("sensitivity", robustness), ("support", support)):
            pd.DataFrame(data).to_csv(stage/(name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"selection.json", choices)
        write_json(stage/"receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
            parents=rows.parent.nunique(), wells=rows.inst_id.nunique(), plates=rows.rna_plate.nunique(),
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)
    return sha256(args.out/"receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "freeze", "score"))
    for name in ("access", "pin", "protocol", "config", "assets", "out", "forecasts", "freeze"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--slice")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print((predict if args.action == "predict" else evaluate)(args))
