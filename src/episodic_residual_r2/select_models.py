"""Source-side selection bookkeeping for learned models. Reads run results only (meta-validation scores)."""
import argparse, json, os
import pandas as pd

from .common import MAN, runs, public, dump


def load(run_id):
    return json.load(open(runs("meta", run_id, "result.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "final"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run ids")
    a = ap.parse_args()
    rs = [load(r) for r in a.runs.split(",")]
    tab = pd.DataFrame([dict(run_id=r["run_id"], lr=r["lr"], seed=r["seed"], stopped=r["stopped"], best_step=r["best_step"], steps_run=r["steps_run"],
                             val_k8=r["val_delta_rmse"].get("normal|8"), val_k16=r["val_delta_rmse"].get("normal|16"), val_k32=r["val_delta_rmse"].get("normal|32"),
                             lam=r["lam"], n_parameters=r["n_parameters"], seconds=r["seconds"]) for r in rs])
    print(tab.to_string())
    path = public("manifests", "episodic_residual_r2", "model_selection.json")
    committed = os.path.join(MAN, "model_selection.json")
    cur = json.load(open(path)) if os.path.exists(path) else (json.load(open(committed)) if os.path.exists(committed) else dict(search={}, final={}))
    if a.mode == "search":
        best = tab.sort_values(["val_k16", "lr"]).iloc[0]
        cur["search"][a.name] = dict(rule="lowest meta-validation K=16 query delta RMSE (family-equal mean); ties -> smaller lr", candidates=tab.to_dict("records"),
                                     selected_run=best.run_id, selected_lr=float(best.lr), mean_offset_reference=rs[0]["val_mean_offset"])
        print("selected", best.run_id)
    else:
        assert len({(r["lr"], json.dumps(r["variant"], sort_keys=True), r["model"]) for r in rs}) == 1, "final seeds must share one configuration"
        cur["final"][a.name] = {str(r["seed"]): dict(run_id=r["run_id"], checkpoint=r["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"], lr=r["lr"],
                                                     variant=r["variant"], best_step=r["best_step"], val=r["val_delta_rmse"], lam=r["lam"]) for r in rs}
    dump(cur, path)


if __name__ == "__main__":
    main()
