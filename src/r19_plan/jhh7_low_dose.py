"""Transfer low-dose calibration from JHH-5 to unscored JHH-7 treatments."""

import argparse
from hashlib import sha256 as hash_text
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r17_lincs.inventory import in_footprint, register_footprint
from src.r19_plan.hcc_lincs_24h_access import CONTEXT, assigned_controls
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.i_preflight import read_pinned, write_plan
from src.r19_plan.i_score import checked_array, checked_rows
from src.r19_plan.i_stage import parse_landmarks, pinned_bytes
from src.r19_plan.jhh5_donor_attribution import FIX, MEAN, SHUFFLE, STATE, shrink_consensus, wrong_parent
from src.r19_plan.jhh7_donor_transfer import ARMS, SCALED, controls_for, read_plan
from src.r19_plan.jhh_dose_bridge import audited
from src.r19_plan.r18_input_bundle import indices
from src.r19_plan.r18_matrix_index import metadata_index
from src.r19_plan.t4_position import compare
from src.r19_plan.t4_state import state_forecast
from src.r19_plan.t4_workflow import scalar
from src.r19_plan.x_inputs import check_unique


DOSES = ("1", "0.1")
FIT_DOSE = "1.11111"
KEYS = ["actual_dose_uM", "parent"]
PRIMARY = (STATE, "B0", SHUFFLE)
PAIRS = tuple((SCALED, ref) for ref in (*PRIMARY, MEAN, FIX)) + ((STATE, "B0"), (FIX, "B0"))


def access(args):
    low, high = audited(args.audit, args.audit_pin), audited(args.high_audit, args.high_pin)
    old = high["targets"].loc[lambda r: r.cell_id.eq("JHH7")]
    targets = low["targets"].loc[lambda r: r.cell_id.eq("JHH7") & r.actual_dose_uM.isin(DOSES)]
    targets = targets.loc[~in_footprint(targets, register_footprint(old))].copy()
    all_fit = low["targets"].loc[lambda r: r.cell_id.eq("JHH5") & r.actual_dose_uM.eq(FIT_DOSE)]
    fit = all_fit.loc[~all_fit.parent.isin(targets.parent) & ~all_fit.scaffold.isin(targets.scaffold)
                      & ~all_fit.rna_plate.isin(targets.rna_plate)]
    treatments = pd.concat((targets, fit), ignore_index=True)
    source = low["donors"].loc[lambda r: r.view.eq("nonhepatic") & ~r.rna_plate.isin(treatments.rna_plate)]
    source = source.merge(treatments[KEYS].drop_duplicates(), on=KEYS, validate="many_to_one")
    support = source.groupby(KEYS).agg(cells=("cell_id", "nunique"), plates=("rna_plate", "nunique"))
    allowed = support.index[support.cells.ge(8) & support.plates.ge(8)]
    targets = targets.loc[pd.MultiIndex.from_frame(targets[KEYS]).isin(allowed)]
    fit = fit.loc[pd.MultiIndex.from_frame(fit[KEYS]).isin(allowed)]
    treatments = pd.concat((targets, fit), ignore_index=True)
    source = source.merge(treatments[KEYS].drop_duplicates(), on=KEYS, validate="many_to_one")
    contexts = pd.concat((treatments[CONTEXT], source[CONTEXT]), ignore_index=True).drop_duplicates()
    controls = assigned_controls(low["controls"].merge(contexts, on=CONTEXT, validate="many_to_one"))
    by_dose = {dose: dict(parents=int(rows.parent.nunique()), wells=len(rows), plates=int(rows.rna_plate.nunique()),
                    prior_target_parents=int(rows.loc[rows.parent.isin(old.parent)].parent.nunique()))
               for dose, rows in targets.groupby("actual_dose_uM")}
    if (set(by_dose) != set(DOSES) or min(row["parents"] for row in by_dose.values()) < 90
            or fit.parent.nunique() < 100 or targets.rna_plate.nunique() != 1
            or not controls.groupby(CONTEXT).role.nunique().eq(3).all()):
        raise ValueError(f"The registered JHH-7 technical support gate did not pass: {by_dose}")
    if min(targets.groupby(["actual_dose_uM", "fold", "rna_plate"]).parent.nunique()) < 3:
        raise ValueError("A within-dose wrong-parent pool has insufficient support")
    details = dict(by_dose=by_dose, fit_parents=int(fit.parent.nunique()), fit_wells=len(fit),
        high_audit_sha256=sha256(args.high_pin), target_dmsos=int(controls.cell_id.eq("JHH7").sum()),
        previous_10uM_target_dmsos=int(controls.loc[controls.cell_id.eq("JHH7"), "previous_10uM_control"].eq("True").sum()),
        target_parent_overlap=int(len(set(targets.loc[targets.actual_dose_uM.eq(DOSES[0]), "parent"])
                                     & set(targets.loc[targets.actual_dose_uM.eq(DOSES[1]), "parent"]))),
        prior_target_plates=int(targets.loc[targets.rna_plate.isin(old.rna_plate)].rna_plate.nunique()))
    register(args, targets, fit, source, controls, details)


def register(args, target, fit, source, controls, details):
    treatments = pd.concat((target, fit, source), ignore_index=True)
    check_unique(pd.concat((treatments, controls), ignore_index=True))

    standard = read_json(args.cmap_pin)
    landmarks = parse_landmarks(pinned_bytes(args.landmarks, standard["landmark_axis_sha256"]))
    genes = landmarks.pr_gene_id.to_numpy()
    matrices = standard["matrix_records"]
    t, trt_genes, trt = metadata_index(Path(matrices["treated"]["path"]), treatments, genes)
    controls, ctl_genes, ctl = metadata_index(Path(matrices["controls"]["path"]), controls, genes)
    if matrices != dict(treated=trt, controls=ctl):
        raise ValueError("The physical Level 3 metadata does not match CMap")
    target, fit = (t.loc[t.cell_id.eq(cell)].copy() for cell in ("JHH7", "JHH5"))
    source = t.loc[~t.cell_id.isin(("JHH5", "JHH7"))].copy()
    locked = args.run/"access"
    if locked.exists():
        raise FileExistsError(locked)
    with TemporaryDirectory(dir=args.run, prefix=".jhh7-input-") as folder:
        out = Path(folder)
        splits, stats = {}, []
        for i, role in enumerate("ABC"):
            outcome_role = "ABC"[(i+2)%3]
            fit_dmsos = controls_for(controls, fit, role)
            donor_dmsos = controls_for(controls, source, role)
            outcome_dmsos = controls_for(controls, target, outcome_role)
            if set(target.inst_id) & set(pd.concat((fit, source, donor_dmsos, fit_dmsos)).inst_id):
                raise ValueError("JHH-7 outcome crossed the predictor boundary")
            query = target.drop(columns="level3_row_index")
            splits[role] = dict(predict=write_plan(out/"predict"/role, {"fit.tsv":fit,"fit_controls.tsv":fit_dmsos,
                "source.tsv":source,"source_controls.tsv":donor_dmsos,"query.tsv":query}),
                score=write_plan(out/"score"/role, {"query.tsv":target,"controls.tsv":outcome_dmsos}))
            stats.append(dict(role=role, score_role=outcome_role, score_wells=len(target),
                score_dmsos=len(outcome_dmsos), fit_wells=len(fit), fit_parents=fit.parent.nunique(), fit_dmsos=len(fit_dmsos),
                donor_wells=len(source), donor_cells=source.cell_id.nunique(), donor_plates=source.rna_plate.nunique()))
        landmarks.to_csv(out/"landmarks.tsv",sep="\t",index=False)
        trt_genes.to_csv(out/"treated_genes.tsv",sep="\t",index=False)
        ctl_genes.to_csv(out/"controls_genes.tsv",sep="\t",index=False)
        pd.DataFrame(stats).to_csv(out/"support.tsv",sep="\t",index=False)
        write_json(out/"receipt.json", dict(protocol_sha256=sha256(args.protocol), audit_sha256=sha256(args.audit_pin),
            code_sha256=sha256(__file__), metadata_only=True, expression_read=False,
            roles=splits, matrices=matrices, files={p.name:sha256(p) for p in out.glob("*.tsv")}, **details))
        out.rename(locked)


def provenance(args):
    access = args.run/"access"
    ref = read_json(access/"receipt.json")
    stamp = dict(access_sha256=sha256(access/"receipt.json"), code_sha256=sha256(__file__), protocol_sha256=sha256(args.protocol))
    if (stamp["access_sha256"] != sha256(args.access_pin) or not ref["metadata_only"] or ref["expression_read"]
            or any(ref[k] != v for k,v in stamp.items() if k!="access_sha256")):
        raise ValueError("The metadata-only JHH-7 authority changed")
    genes = parse_landmarks(pinned_bytes(access/"landmarks.tsv",ref["files"]["landmarks.tsv"])).pr_gene_id.to_numpy()
    cols = {}
    for kind in ("treated","controls"):
        file = kind+"_genes.tsv"
        table = read_pinned(access,file,ref["files"][file])
        if not np.array_equal(indices(table.gene_id),genes):
            raise ValueError("JHH-7 and donor landmark axes disagree")
        cols[kind] = indices(table.level3_gene_index)
    return ref, genes, cols, stamp


def predict(args, ref, genes, cols, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    block = read_plan(args,ref,"predict",args.role)
    fit, query, source = (block[n] for n in ("fit","query","source"))
    if (set(fit.scaffold) & set(query.scaffold) or set(fit.parent) & set(query.parent) or "level3_row_index" in query
            or not query.cell_id.eq("JHH7").all() or not fit.cell_id.eq("JHH5").all()
            or not source.view.eq("nonhepatic").all() or set(fit.rna_plate) & set(query.rna_plate)):
        raise ValueError("JHH-7 target, its chemotypes, or target Level 3 indices intersect fitting")
    values = effect(source, block["source_controls"], ref, genes, cols, args.role)
    keys = KEYS
    plates, pvals = group_means(values, source, [*keys, "cell_id", "rna_plate"])
    cells, cvals = group_means(pvals, plates, [*keys, "cell_id"])
    registry, dmean = group_means(cvals, cells, keys)
    profiles = [cvals[ix] for ix in cells.groupby(keys, sort=True).indices.values()]
    shrunk = shrink_consensus(profiles,[len(a) for a in profiles],dmean)
    fit_rows, truth = group_means(effect(fit,block["fit_controls"],ref,genes,cols,args.role),fit,["parent"])
    wrong = np.empty(len(query),dtype=object)
    for (*_, fold, plate), ix in query.groupby([*keys[:-1], "fold", "rna_plate"]).indices.items():
        wrong[ix] = wrong_parent(query.iloc[ix],fold,plate)
    fit_dose, = fit.actual_dose_uM.unique()
    if fit_dose != FIT_DOSE or set(query.actual_dose_uM) != set(DOSES):
        raise ValueError("Forecast tables have changed the declared concentrations")
    fit_rows = fit_rows.assign(actual_dose_uM=fit_dose)
    donor_axis = pd.MultiIndex.from_frame(registry[keys])
    fit_index, test, wrong_index = (donor_axis.get_indexer(pd.MultiIndex.from_frame(meta[keys]))
                                    for meta in (fit_rows, query, query.assign(parent=wrong)))
    if any((ix<0).any() for ix in (fit_index,test,wrong_index)) or (wrong_index==test).any():
        raise ValueError("Same input parent identifiers changed")
    x = dmean[fit_index]
    full = dmean[test]
    gain = scalar(fit_rows,x,truth)
    predicted = {"B0":np.zeros_like(full),MEAN:full,FIX:shrunk[test],SCALED:gain*full,SHUFFLE:dmean[wrong_index],
        STATE:state_forecast(load_network(args.state),(fit_rows,x,truth),full)}
    predicted[SHUFFLE] = gain * predicted[SHUFFLE]
    dest = args.run/"forecasts"/args.role
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=dest.parent,prefix=".forecast-") as temp:
        stage = Path(temp)
        matrix = stage/"predictions.npz"
        np.savez_compressed(matrix,**{k:checked_array(v,(len(query),len(genes)),k).astype("float32") for k,v in predicted.items()})
        write_json(stage/"receipt.json",dict(stamp,role=args.role,file_sha256=sha256(matrix),fit_parents=len(fit_rows),
            scalar_gain=gain, state_upstream=UPSTREAM,
            state_model_sha256=sha256(args.state/"src/state/tx/models/state_transition.py"),
            wrong_parent_mapping_sha256=hash_text("\n".join(wrong).encode()).hexdigest()))
        stage.rename(dest)


def finish(args, ref, genes, cols, stamp):
    scoring = args.phase == "score"
    freeze = args.run / "freeze.json"
    if scoring and (sha256(freeze) != args.digest or (args.run / "score").exists()):
        raise ValueError("JHH-7 low-dose scoring requires its forecast freeze")
    if not scoring and freeze.exists():
        raise FileExistsError(freeze)
    queries, forecasts, receipts, mappings = {}, {}, {}, set()
    fits = []
    for role in "ABC":
        path = args.run / "forecasts" / role
        receipt = read_json(path / "receipt.json")
        actual = path / "predictions.npz"
        if (receipt["role"] != role or sha256(actual) != receipt["file_sha256"]
            or any(receipt[k] != v for k, v in stamp.items()) or receipt["fit_parents"] != ref["fit_parents"]):
            raise ValueError("A low-dose JHH-7 forecast differs from authorization metadata")
        name = "query.tsv"
        q = read_pinned(args.run / "access" / "predict" / role, name, ref["roles"][role]["predict"][name])
        queries[role] = q
        with np.load(actual, allow_pickle=False) as data:
            arrays = {arm: checked_array(data[arm], (len(q), len(genes)), arm) for arm in ARMS}
        if scoring:
            forecasts[role] = arrays
        receipts[role] = sha256(path / "receipt.json")
        mappings.add(receipt["wrong_parent_mapping_sha256"])
        fits.append(dict(role=role, fit_parents=receipt["fit_parents"], scalar_gain=receipt["scalar_gain"]))
    rows = checked_rows(pd.concat([q.assign(role=role) for role, q in queries.items()], ignore_index=True), None)
    if len(mappings) != 1 or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("Well roles or wrong-parent correspondence differ across forecasts")
    for dose, support in ref["by_dose"].items():
        subset = rows.loc[rows.actual_dose_uM.eq(dose)]
        if (subset.inst_id.nunique(), subset.parent.nunique(), subset.rna_plate.nunique()) != (
                support["wells"], support["parents"], support["plates"]):
            raise ValueError("The dose-specific score axes differ from the metadata lock")
    manifest = dict(stamp, roles=receipts, by_dose=ref["by_dose"], fit_parents=ref["fit_parents"])
    if not scoring:
        write_json(freeze, manifest)
        return
    if read_json(freeze) != manifest:
        raise ValueError("Low-dose point predictions changed after freezing")
    outcomes = []
    for role, query in queries.items():
        plan = read_plan(args, ref, "score", role)
        target = plan["query"]
        if not target.drop(columns="level3_row_index").equals(query):
            raise ValueError("The scorers received a different JHH-7 treatment order")
        score_role = "ABC"[("ABC".index(role) + 2) % 3]
        outcomes.append(effect(target, plan["controls"], ref, genes, cols, score_role))
    truth = np.concatenate(outcomes)
    forecasts = {arm: np.concatenate([values[arm] for values in forecasts.values()]) for arm in ARMS}
    results = []
    for dose in DOSES:
        keep = rows.actual_dose_uM.eq(dose)
        results.extend(compare(rows.loc[keep].reset_index(drop=True), truth[keep],
                               {arm: values[keep] for arm, values in forecasts.items()}, PAIRS, dose))
    with TemporaryDirectory(dir=args.run, prefix=".jhh7-low-score-") as temporary:
        stage = Path(temporary)
        for name, table in (("paired", results), ("fits", fits)):
            pd.DataFrame(table).to_csv(stage / (name + ".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(manifest, freeze_sha256=args.digest,
                                                  files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(args.run / "score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("access", "predict", "freeze", "score"))
    for key in ("audit", "audit-pin", "high-audit", "high-pin", "cmap-pin", "landmarks", "protocol", "run", "access-pin", "state"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--role", choices=tuple("ABC"))
    parser.add_argument("--digest")
    args = parser.parse_args()
    if args.phase == "access":
        access(args)
    elif args.phase == "predict":
        predict(args, *provenance(args))
    else:
        finish(args, *provenance(args))
