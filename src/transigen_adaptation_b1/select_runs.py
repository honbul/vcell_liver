"""Selections that use validation data only: source learning rate, source checkpoints, adaptation
learning rates, the one-time budget extension check, and the run ledger."""
import argparse, glob, json, os
import pandas as pd
import yaml

from .adapt_target import run_id, ARMS
from .data import MAN_DIR
from .tsg_adapter import sha256


def src_id(lr, seed):
    return f"source_lr{lr:g}_seed{seed}_v1"


def load(kind, rid):
    p = os.path.join(os.environ["TB_RUNS"], kind, rid, "result.json")
    assert os.path.exists(p), f"expected run is missing: {rid}"
    return json.load(open(p))


def source_lr(cfg):
    sp = cfg["source_pretraining"]
    runs = [load("source", src_id(lr, sp["search_seed"])) for lr in sp["lr_grid"]]
    best = min(runs, key=lambda r: (r["val_delta_rmse_best"], r["lr"]))
    out = dict(selected_lr=best["lr"], search_seed=sp["search_seed"], metric=sp["selection_metric"],
               candidates=[{k: r[k] for k in ("run_id", "lr", "best_epoch", "epochs_run", "stopped", "val_delta_rmse_best",
                                               "val_delta_rmse_epoch0", "val_no_change_delta_rmse", "seconds")} for r in runs])
    json.dump(out, open(os.path.join(MAN_DIR, "source_lr_selection.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def source_final(cfg):
    sp = cfg["source_pretraining"]
    lr = json.load(open(os.path.join(MAN_DIR, "source_lr_selection.json")))["selected_lr"]
    ck = {}
    for seed in sp["seeds"]:
        r = load("source", src_id(lr, seed))
        assert sha256(r["checkpoint"]) == r["checkpoint_sha256"]
        ck[str(seed)] = {k: r[k] for k in ("run_id", "lr", "seed", "best_epoch", "epochs_run", "stopped", "val_delta_rmse_best",
                                           "val_no_change_delta_rmse", "checkpoint", "checkpoint_sha256", "state_hash", "seconds")}
    out = dict(selected_lr=lr, checkpoints=ck)
    json.dump(out, open(os.path.join(MAN_DIR, "source_selection.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def adapt(cfg):
    ad = cfg["adaptation"]
    bf = os.path.join(MAN_DIR, "adapt_budget.json")
    extended = os.path.exists(bf)
    max_updates = json.load(open(bf))["max_updates"] if extended else ad["max_updates"]
    summ = json.load(open(os.path.join(MAN_DIR, "splits_summary.json")))["targets"]
    selected, cand = {}, []
    for tgt, info in sorted(summ.items()):
        if info["status"] == "excluded":
            continue
        for split, si in info["splits"].items():
            for arm in ARMS:
                rs = [load("adapt", run_id(tgt, int(split), si["k_main"], arm, lr, ad["search_seed"], max_updates))
                      for lr in ad["arms"][arm]["lr_grid"]]
                best = min(rs, key=lambda r: (r["val_delta_rmse_best"], r["lr"]))
                selected[f"{tgt}|{split}|{arm}"] = dict(lr=best["lr"], run_id=best["run_id"], val_delta_rmse=best["val_delta_rmse_best"],
                                                       best_step=best["best_step"], best_at_max=best["best_at_max"])
                for r in rs:
                    cand.append({k: r[k] for k in ("run_id", "target", "split_seed", "k", "arm", "lr", "best_step", "stopped",
                                                   "val_delta_rmse_step0", "val_delta_rmse_best", "best_at_max")} | dict(selected=r["run_id"] == best["run_id"]))
    frac = {arm: sum(v["best_at_max"] for k, v in selected.items() if k.endswith(arm)) / max(1, sum(k.endswith(arm) for k in selected))
            for arm in ARMS}
    trigger = any(f >= 1 / 3 for f in frac.values())
    out = dict(max_updates=max_updates, selected=selected, fraction_best_at_max=frac, extension_triggered=bool(trigger and not extended),
               already_extended=extended)
    pd.DataFrame(cand).to_csv(os.path.join(MAN_DIR, f"adapt_search_u{max_updates}.tsv"), sep="\t", index=False)
    if trigger and not extended:
        json.dump(dict(max_updates=2 * ad["max_updates"], reason="extension_rule", fraction_best_at_max=frac,
                       previous_search_table=f"adapt_search_u{max_updates}.tsv"), open(bf, "w"), indent=1)
        print("EXTENSION TRIGGERED: repeat the search under the doubled budget, then select again")
    else:
        json.dump(out, open(os.path.join(MAN_DIR, "adapt_selection.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def ledger(cfg):
    rows = []
    for p in sorted(glob.glob(os.path.join(os.environ["TB_RUNS"], "*", "*", "result.json"))):
        r = json.load(open(p))
        rows.append({k: r.get(k, "") for k in ("run_id", "kind", "target", "split_seed", "k", "arm", "lr", "seed", "max_updates",
                                               "best_step", "best_epoch", "epochs_run", "steps_run", "stopped", "val_delta_rmse_step0",
                                               "val_delta_rmse_epoch0", "val_delta_rmse_best", "n_adapt_rows", "n_adapt_conditions",
                                               "n_val_conditions", "adapt_rows_hash", "val_rows_hash", "state_hash", "checkpoint_sha256",
                                               "checkpoint", "control", "seconds", "gpu", "commit", "slurm_job")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(MAN_DIR, "run_ledger.tsv"), sep="\t", index=False)
    stray = [d for d in glob.glob(os.path.join(os.environ["TB_RUNS"], "*", "*")) if os.path.isdir(d) and not os.path.exists(os.path.join(d, "result.json"))
             and not os.path.basename(os.path.dirname(d)).startswith(("baselines", "final"))]
    print(f"{len(df)} finished runs; {len(stray)} run directories without a result: {stray}")
    print(f"gpu seconds total: {pd.to_numeric(df.seconds).sum():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["source_lr", "source_final", "adapt", "ledger"])
    ap.add_argument("--config", default="configs/transigen_adaptation_b1/protocol.yaml")
    a = ap.parse_args()
    globals()[a.what](yaml.safe_load(open(a.config)))
