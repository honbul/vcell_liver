"""Score matched forecasts after verifying the historical freeze and all prediction receipts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vc_fm_bench.artifacts import digest, new_directory, verify_record, write_json
from .data import SCOPES, TRACK6, Registered, legacy_rosters, read
from .models import metrics

METRICS = ("auroc", "auprc", "balanced_accuracy", "brier", "ece")


def verify_common(run, reg):
    history = run / "historical/freeze.json"
    policy = read(history)
    if policy["split_sha256"] != reg.split_hash or policy["external_labels_loaded"]:
        raise ValueError("Missing verified historical model selection")
    expected = digest(history)
    parts, files = {}, {}
    for scope in SCOPES:
        base = run / "forecasts" / scope
        r = read(base / "receipt.json")
        if r["history_freeze"] != expected or r["label_file_opened"] or r["connectivity_overlap"]:
            raise ValueError(f"Invalid registered forecast provenance: {scope}")
        part = base / "forecast.parquet"
        verify_record(part, r["forecast"])
        f = pd.read_parquet(part)
        q = reg.query(scope)
        if not (f.index.equals(q.index) and np.isfinite(f.astype(float).to_numpy()).all()):
            raise ValueError(f"Unequal prediction denominator: {scope}")
        parts[scope] = f.join(q[["connectivity"]], validate="one_to_one").assign(fold=scope)
        files[scope] = dict(receipt=digest(base / "receipt.json"), forecast=r["forecast"]["sha256"])
    return policy, expected, parts, files


def denominator(reg, parts):
    saved = legacy_rosters()
    source = TRACK6 / "prepared/track6-v2"
    prep = read(source / "receipt.json")
    frames = {}
    for key in ("dili_inputs", "dili_labels"):
        p = source / f"{key}.parquet"
        verify_record(p, prep["files"][p.name])
        frames[key] = pd.read_parquet(p).set_index("source_drug", verify_integrity=True)
    chemical = frames["dili_inputs"].join(frames["dili_labels"], validate="one_to_one")
    reports = []
    observations = []
    for kind, old_stem in (("murcko", "dili-scaffold-"), ("release", "dili-release-increment")):
        raw, old, missing = saved[kind]
        scopes = [s for s in SCOPES if s.startswith(kind + "-")]
        actual = pd.concat([parts[s] for s in scopes], verify_integrity=True)
        compared = pd.concat([reg.labels(s) for s in scopes], verify_integrity=True)
        archived_labels = []
        for old_scope in ([old_stem + str(i) for i in range(5)] if kind == "murcko" else [old_stem]):
            d = TRACK6 / "prepared/track6-panels-v2" / old_scope
            p = d / "outcomes.parquet"
            verify_record(p, read(d / "receipt.json")["files"][p.name])
            archived_labels.append(pd.read_parquet(p).set_index("id"))
        prior_y = pd.concat(archived_labels, verify_integrity=True).outcome
        matched = old.index.intersection(compared.index)
        if not np.array_equal(prior_y.loc[matched], compared.loc[matched, "label"]):
            raise ValueError(f"Binary labels disagree on matched record IDs: {kind}")
        participant = chemical.loc[chemical.concern.isin(["Most", "Less", "No"])].copy()
        if kind == "release":
            participant = participant.loc[participant.release_increment]
        participant["binary"] = participant.concern.isin(["Most", "Less"])
        group = participant.groupby("drug_id")
        for key in actual.index.symmetric_difference(old.index).sort_values():
            cause = []
            if key not in actual.index:
                cause.append("legacy_only_canonicalization_or_binary_eligibility")
            elif key not in participant.index:
                cause.append("not_prepared_by_track6_structure_eligibility")
            elif key not in raw.index:
                drug = participant.at[key, "drug_id"]
                same = group.get_group(drug)
                if same.binary.nunique() > 1:
                    cause.append("track6_charge_parent_conflicting_binary_labels")
                elif len(same) > 1:
                    cause.append("track6_charge_parent_duplicate_record")
                else:
                    cause.append("track6_other_preparation_exclusion")
            else:
                cause.extend(f"legacy_missing_{m}" for m, ids in missing.items() if key in ids)
                if not cause:
                    cause.append("track6_other_coverage_exclusion")
            same_ids = ""
            if key in participant.index:
                same_ids = ",".join(sorted(group.get_group(participant.at[key, "drug_id"]).index.difference([key])))
            slot = actual.at[key, "fold"] if key in actual.index else "outside"
            reports.append(dict(split=kind, fda_id=key, common_target=key in actual.index,
                                archived_panel=key in raw.index, archived_scored=key in old.index,
                                registered_fold=slot, parent_siblings=same_ids or "none",
                                audit_reason=";".join(cause)))
        observations.append(dict(split=kind, registered=len(actual), legacy_prepared=len(raw), legacy_scored=len(old),
                                 matched=len(matched), matching_binary=len(matched),
                                 legacy_only=len(old.index.difference(actual.index))))
    return pd.DataFrame(reports), observations


def cluster_weights(groups, n, seed):
    _, inverse = np.unique(groups, return_inverse=True)
    total = inverse.max() + 1
    rng = np.random.default_rng(seed)
    return rng.multinomial(total, np.repeat(1. / total, total), size=n)[:, inverse].astype(np.int16)


def weighted_scores(y, prob, decision, weights):
    order = np.argsort(-prob, kind="stable")
    sorted_p, sorted_y = prob[order], y[order]
    endpoints = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(y) - 1]
    positive = y == 1
    negative = ~positive
    bins = np.clip((prob * 10).astype(int), 0, 9)
    residuals = y - prob
    output = []
    for block in np.array_split(weights, 30):
        sorted_w = block[:, order]
        n = block.sum(axis=1)
        hits = np.cumsum(sorted_w * sorted_y, axis=1)[:, endpoints]
        cum = np.cumsum(sorted_w, axis=1)[:, endpoints]
        with np.errstate(invalid="ignore", divide="ignore"):
            tpr = hits / hits[:, -1:]
            fp = cum - hits
            fpr = fp / fp[:, -1:]
            auc = (.5 * (tpr + np.c_[np.zeros(len(block)), tpr[:, :-1]])
                   * np.diff(fpr, axis=1, prepend=0)).sum(axis=1)
            precision = np.divide(hits, cum, out=np.zeros_like(hits, dtype=float), where=cum > 0)
            ap = (precision * np.diff(tpr, axis=1, prepend=0)).sum(axis=1)
            ba = .5 * (block[:, positive & decision].sum(axis=1) / block[:, positive].sum(axis=1)
                       + block[:, negative & ~decision].sum(axis=1) / block[:, negative].sum(axis=1))
            brier = (block * np.square(residuals)).sum(axis=1) / n
            ece = np.sum([np.abs((block[:, bins == i] * residuals[bins == i]).sum(axis=1))
                          for i in np.unique(bins)], axis=0) / n
        output.append(np.stack([auc, ap, ba, brier, ece], axis=1))
    return np.concatenate(output)


def evaluate(part, kind, chosen):
    arms = sorted(k.rsplit("/", 1)[0] for k in part if k.endswith("/p"))
    outcome = part.label.to_numpy(dtype=np.int64)
    weights = cluster_weights(part.connectivity.to_numpy(), 10000, 20261123)
    reports, intervals, pairs = {}, {}, []
    point, means = {}, {}
    valid_count = None
    for name in arms:
        prob = part[name + "/p"].to_numpy(dtype=float)
        decisions = part[name + "/decision"].to_numpy(dtype=bool)
        point[name] = metrics(outcome, prob, decisions)
        fold_values = [metrics(item.label.to_numpy(dtype=np.int64), item[name + "/p"].to_numpy(dtype=float),
                               item[name + "/decision"].to_numpy(dtype=bool))
                       for _, item in part.groupby("fold")]
        means[name] = {k: float(np.mean([r[k] for r in fold_values])) for k in METRICS}
        values = weighted_scores(outcome, prob, decisions, weights)
        valid = np.isfinite(values).all(axis=1)
        valid_count = int(valid.sum())
        reports[name] = values[valid]
        # Independently verify the grouped resampling metric implementation.
        control = weighted_scores(outcome, prob, decisions, np.ones((30, len(part)), dtype=np.int16))[0]
        if not np.allclose(control, [point[name][k] for k in METRICS], atol=1e-10):
            raise ValueError("Vectorized paired bootstrap metrics differ from unweighted reference")
        intervals[name] = np.quantile(reports[name], [.005, .995], axis=0)
    for name in arms:
        delta = reports[chosen] - reports[name]
        low, high = np.quantile(delta, [.005,.995], axis=0)
        for i, key in enumerate(METRICS):
            pairs.append(dict(split=kind, fixed_model=chosen, comparator=name, metric=key,
                              delta=point[chosen][key]-point[name][key], lo99=low[i], hi99=high[i], draws=valid_count))
    rows = [
        dict(split=kind, model=name, n=len(part), metric=key, pooled=point[name][key],
             fold_mean=means[name][key], lo99=intervals[name][0,i], hi99=intervals[name][1,i])
        for name in arms for i,key in enumerate(METRICS)
    ]
    return rows, pairs, valid_count


def score(args):
    reg = Registered()
    policy, history, parts, files = verify_common(args.run, reg)
    with new_directory(args.out) as temp:
        write_json(temp / "prediction_freeze.json", dict(history_freeze=history, scopes=files))
        diff, support = denominator(reg, parts)
        diff.to_csv(temp / "DILI_DENOMINATOR_DIFF.tsv", sep="\t", index=False)
        point, comparison, counts = [], [], {}
        for kind, expected in (("murcko", 887), ("butina", 887), ("release", 155)):
            frames = [part.join(reg.labels(scope)[["label"]], validate="one_to_one")
                      for scope, part in parts.items() if scope.startswith(kind + "-")]
            observations = pd.concat(frames, verify_integrity=True)
            if len(observations) != expected or observations.label.isna().any():
                raise ValueError(f"Unexpected shared denominator: {kind}")
            rows, pairs, draws = evaluate(observations, kind, policy["deployment"])
            print(kind, len(observations), draws, flush=True)
            point.extend(rows)
            comparison.extend(pairs)
            counts[kind] = dict(rows=len(observations), positives=int(observations.label.sum()), valid_draws=draws)
        pd.DataFrame(point).to_csv(temp / "dili_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(comparison).to_csv(temp / "dili_paired99.tsv", sep="\t", index=False)
        write_json(temp / "score_receipt.json", dict(
            history_freeze=history, prediction_freeze=digest(temp / "prediction_freeze.json"),
            denominators=counts, legacy_denominators=support, final_model=policy["deployment"],
            scoring_passes=1, bootstrap_unit="InChIKey connectivity", metric_ece_bins=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    score(parser.parse_args())
