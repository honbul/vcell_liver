"""Audit physical position before selecting the R15 prediction design."""

import argparse
from collections import Counter, defaultdict
import itertools
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json

from .data import EXCLUDE_LINES, LIVER, STUDIES, asset, chemicals, genes, instances, landmark_rows, signatures, table


def eligible(rows):
    return rows.pert_type.eq("trt_cp") & rows.dose.between(5, 20)


def repeat_layout(rows):
    totals = Counter()
    for _, group in rows.groupby(["parent", "hours", "dose"]):
        if group.rna_plate.nunique() < 2:
            continue
        values = list(group.groupby("rna_plate").rna_well.value_counts().groupby(level=0))
        total, same = 0, 0
        for (_, a), (_, b) in itertools.combinations(values, 2):
            total += a.sum() * b.sum()
            same += a.droplevel(0).mul(b.droplevel(0), fill_value=0).sum()
        totals.update(parents=1, pairs=int(total), same_well=int(same))
    return dict(totals)


def physical_effect(rows, values, level):
    rows = rows.copy().reset_index(drop=True)
    vehicle = rows.pert_type.eq("ctl_vehicle").to_numpy()
    ctrl = rows.loc[vehicle].groupby("rna_plate").indices
    vehicles = values[vehicle]
    keep = (~vehicle) & rows.rna_plate.isin(ctrl).to_numpy()
    if not keep.any():
        return None
    obs = values[keep].astype(np.float64)
    treated = rows.loc[keep].reset_index(drop=True)
    for plate, idx in treated.groupby("rna_plate").indices.items():
        if level != 5:
            obs[idx] -= vehicles[ctrl[plate]].mean(axis=0)
    keep = treated.groupby("rna_plate").parent.transform("nunique").ge(3)
    return treated.loc[keep].reset_index(drop=True), obs[keep]


def position_effect(rows, response, position):
    loc = pd.Series(position).groupby(position, sort=False).indices
    plate = rows.rna_plate.to_numpy()
    actual = rows.footprint.to_list() if "footprint" in rows else [frozenset([v]) for v in plate]
    fold = rows.parent.map(lambda v: crc32(v.encode()) % 3).to_numpy()
    forecast = np.zeros_like(response)
    support = np.zeros(len(rows), dtype=bool)
    partitions = rows.groupby("rna_plate").indices
    donor = {}
    for k in range(3):
        train = np.zeros_like(response)
        for idx in partitions.values():
            fitted = idx[fold[idx] != k]
            if len(fitted):
                train[fitted] = response[fitted] - response[fitted].mean(axis=0)
        donor[k] = train
    for matches in loc.values():
        for i in matches:
            use = [j for j in matches if fold[i] != fold[j] and plate[i] != plate[j] and not actual[i].intersection(actual[j])]
            if use:
                forecast[i] = donor[fold[i]][use].mean(axis=0)
                support[i] = True
    for idx in partitions.values():
        forecast[idx] -= forecast[idx].mean(axis=0)
    return forecast, support


def position_scores(rows, obs):
    if len(rows) == 0:
        return []
    pos = rows.rna_well.to_numpy()
    prediction, support = position_effect(rows, obs, pos)
    rng = np.random.default_rng(150803)
    shuffled = pos.copy()
    for indices in rows.groupby("rna_plate").indices.values():
        shuffled[indices] = rng.permutation(shuffled[indices])
    null, null_support = position_effect(rows, obs, shuffled)
    obs = obs.copy()
    for indices in rows.groupby("rna_plate").indices.values():
        obs[indices] -= obs[indices].mean(axis=0)
    parent = pd.Index(sorted(rows.parent.unique())).get_indexer(rows.parent)
    groups = int(parent.max())+1
    count = np.bincount(parent)
    b0 = np.sqrt(np.square(obs).mean(axis=1))
    base = np.bincount(parent, b0)/count
    draw = rng.integers(groups, size=(3000, groups))
    results = []
    for kind, value, available in (("same_well", prediction, support), ("well_shuffle", null, null_support)):
        for gain in (.25, 1.):
            residual = obs - gain*value
            rmse = np.sqrt(np.square(residual).mean(axis=1))
            point = np.bincount(parent, rmse)/count
            improvement = 100*(1.-point[draw].mean(axis=1)/base[draw].mean(axis=1))
            passed = np.isfinite(improvement)
            if not passed.any():
                continue
            low, high = np.percentile(improvement[passed], [.5, 99.5])
            results.append(dict(kind=kind, gain=gain, rows=len(rows), parents=groups,
                                plates=rows.rna_plate.nunique(), supported=int(available.sum()),
                                b0_rmse=base.mean(), rmse=point.mean(), gain_pct=100*(1.-point.mean()/base.mean()),
                                gain_low99=low, gain_high99=high,
                                r2=1.-np.square(residual).sum()/np.square(obs).sum()))
    return results


def vehicle_pairs(rows, values):
    choose = rows.pert_type.eq("ctl_vehicle")
    controls = rows.loc[choose].reset_index(drop=True)
    values = values[choose]
    vectors = defaultdict(list)
    rng = np.random.default_rng(151)
    shuffled = defaultdict(list)
    for plate, ix in controls.groupby("rna_plate").indices.items():
        own = controls.iloc[ix]
        unique = own.rna_well.value_counts()
        good = [i for i in ix if unique.loc[controls.rna_well[i]] == 1]
        coords = controls.rna_well.iloc[good].to_numpy()
        try:
            v = values[good]
        except IndexError:
            continue
        ordered = np.argsort(coords)
        coords, v = coords[ordered], v[ordered]
        random_values = v[rng.permutation(len(v))]
        for a, b in itertools.combinations(range(len(coords)), 2):
            key = (coords[a], coords[b])
            diff = v[a]-v[b]
            diff_shuffled = random_values[a]-random_values[b]
            if np.linalg.norm(diff) > 1e-12 and np.linalg.norm(diff_shuffled) > 1e-12:
                vectors[key].append(diff/np.linalg.norm(diff))
                shuffled[key].append(diff_shuffled/np.linalg.norm(diff_shuffled))
    def summarize(bank):
        count, inner, coordinates = 0, 0., 0
        for vs in bank.values():
            if len(vs) > 1:
                arr = np.stack(vs).astype(np.float64)
                coordinates += 1
                count += len(vs)*(len(vs)-1)//2
                inner += (np.square(arr.sum(axis=0)).sum() - np.square(arr).sum())/2
        return dict(plate_pairs=count, well_pair_types=coordinates, mean_cosine=(inner/count if count else None))
    return {"matched": summarize(vectors), "within_plate_shuffle": summarize(shuffled),
            "vehicle_wells": len(controls), "plates": controls.rna_plate.nunique()}


def modz_locations(sig, inst):
    registry = inst.set_index("inst_id", verify_integrity=True)
    coordinate, footprints, all_plates = [], [], []
    for record in sig.itertuples(index=False):
        known = registry.reindex(record.distil_id.split("|")).dropna(subset=["rna_well", "rna_plate"])
        known = known.loc[known.rna_well.str.fullmatch(r"[A-Z]\d\d")]
        coordinates = set(known.rna_well)
        coordinate.append(next(iter(coordinates)) if len(coordinates) == 1 else "")
        footprints.append(frozenset(known.rna_plate))
        all_plates.append(len(known))
    return sig.assign(rna_plate=sig.sig_id.str.split(":", n=1).str[0], rna_well=coordinate,
                      footprint=footprints, mapped_replicates=all_plates)


def audit(args):
    out = args.out / args.study
    out.mkdir(parents=True, exist_ok=True)
    sig = signatures(args.root, args.study)
    inst = instances(args.root, args.study)
    chem = chemicals(args.root, args.study)
    land = genes(args.root, args.study)
    metric = table(asset(args.root, args.study, "sig_metrics"))
    cols = ["sig_id", "distil_cc_q75", "distil_nsample", "is_exemplar", "tas"]
    sig = sig.merge(metric[[c for c in cols if c in metric]], on="sig_id", how="left", validate="one_to_one")
    if "is_exemplar" not in sig:
        sig["is_exemplar"] = ""
    parent = chem.set_index("pert_id").parent
    for x in (inst, sig):
        x["parent"] = x.pert_id.map(parent).fillna(x.pert_id)
    meta = []
    for cell in LIVER:
        for hours in (0, 6, 24):
            chosen = sig.pert_type.eq("trt_cp") & sig.cell_id.eq(cell)
            if hours:
                chosen &= sig.hours.eq(hours)
            s = sig.loc[chosen]
            near = s.loc[s.dose.between(5, 20)]
            cross = sig.loc[sig.pert_type.eq("trt_cp") & ~sig.cell_id.isin(EXCLUDE_LINES) & sig.pert_id.isin(s.pert_id)]
            q = pd.to_numeric(near.distil_cc_q75, errors="coerce").ge(.2) & pd.to_numeric(near.distil_nsample, errors="coerce").ge(2)
            meta.append(dict(cell=cell, hours=hours, signatures=len(s), pert_ids=s.pert_id.nunique(),
                             parents=s.parent.nunique(), near10_parents=near.parent.nunique(),
                             validated_structure=near.loc[~near.parent.str.startswith("UNRESOLVED:"), "parent"].nunique(),
                             quality_parent=near.loc[q,"parent"].nunique(), transfer_any=cross.pert_id.nunique(),
                             transfer_same_time=cross.loc[cross.hours.eq(hours)&cross.dose.between(5,20), "pert_id"].nunique()))
    posrows, vehicles, layouts, modz = [], [], [], []
    h = inst.loc[inst.cell_id.eq("HEPG2") & inst.rna_well.str.fullmatch(r"[A-Z]\d\d") & inst.hours.isin([6,24])].copy()
    h = h.loc[eligible(h) | h.pert_type.eq("ctl_vehicle")].reset_index(drop=True)
    headers = {"inst_info":asset(args.root,args.study,"inst_info"),"sig_info":asset(args.root,args.study,"sig_info"),
               "gene_info":asset(args.root,args.study,"gene_info"),"pert_info":asset(args.root,args.study,"pert_info"),
               "sig_metrics":asset(args.root,args.study,"sig_metrics")}
    for hour in (6,24):
        tr = h.loc[h.hours.eq(hour) & eligible(h)].reset_index(drop=True)
        layouts.append(dict(hours=hour, **repeat_layout(tr)))
    for level in (3,4,5):
        matches = list((args.root/args.study).glob(f"*Level{level}*.gctx"))
        if len(matches) != 1:
            raise ValueError(f"Unexpected raw file registry {matches}")
        path = matches[0]
        headers[f"level{level}"] = path
        if level == 5:
            s = sig.loc[sig.cell_id.eq("HEPG2") & sig.hours.isin([6,24]) & (eligible(sig)|sig.pert_type.eq("ctl_vehicle"))]
            located = modz_locations(s, inst.loc[inst.cell_id.eq("HEPG2")])
            for hour, subset in located.groupby("hours"):
                targets = subset.loc[eligible(subset)]
                modz.append(dict(hours=hour, signatures=len(targets), single_coordinate=int(targets.rna_well.ne("").sum()),
                                 all_replicates_linked=int((targets.mapped_replicates == targets.distil_id.str.count(r"\|")+1).sum())))
            rows = located.loc[located.rna_well.ne("")].reset_index(drop=True)
            ids = rows.sig_id
        else:
            rows, ids = h, h.inst_id
        if not len(rows):
            continue
        responses = landmark_rows(path, ids, land.pr_gene_id)
        for hour in (6,24):
            take = rows.hours.eq(hour).to_numpy()
            if not take.any():
                continue
            group, values = rows.loc[take].reset_index(drop=True), responses[take]
            control = vehicle_pairs(group, values)
            vehicles.append(dict(level=level, hours=hour, **control))
            case = physical_effect(group, values, level)
            if case is not None:
                posrows.extend(dict(level=level, hours=hour, **r) for r in position_scores(*case))
        print(f"{args.study} Level {level} audited", flush=True)
    pd.DataFrame(meta).to_csv(out/"inventory.tsv",sep="\t",index=False)
    pd.DataFrame(posrows).to_csv(out/"position.tsv",sep="\t",index=False)
    pd.DataFrame(layouts).to_csv(out/"layout.tsv",sep="\t",index=False)
    pd.DataFrame(modz).to_csv(out/"modz.tsv",sep="\t",index=False)
    write_json(out/"vehicle.json",vehicles)
    chem.to_csv(out/"identities.tsv",sep="\t",index=False)
    write_json(out/"receipt.json", {"study":args.study,"method_sha256":sha256(__file__),
                                    "inputs":{n:{"path":str(p),"bytes":p.stat().st_size,
                                                 **({"sha256":sha256(p)} if not n.startswith("level") else {})}
                                              for n,p in headers.items()},
                                    "files":{p.name:sha256(p) for p in sorted(out.iterdir())}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--study",choices=STUDIES,required=True)
    parser.add_argument("--out",type=Path,required=True)
    audit(parser.parse_args())
