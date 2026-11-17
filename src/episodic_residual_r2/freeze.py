"""Write the freeze manifest: configuration, code commit, every source-side selection, checkpoint hashes and the
fixed support / query lists per (target, split_seed). Reads split metadata only; no HCC expression value."""
import argparse, json, os, subprocess
import pandas as pd
import yaml

from .common import CFG, MAN, B_MAN, public, sha256, dump
from .suite import current_selection
from . import tsg as G

TARGETS = ("HUH7", "JHH5", "JHH7")


def support_and_query(target, split):
    path = os.path.join(B_MAN, "splits", f"target_{target}_split{split}.tsv")
    t = pd.read_csv(path, sep="\t")
    pool = t[t.role == "adapt_pool"].sort_values("adapt_rank")
    return dict(split_file=path, split_file_sha256=sha256(path), support_order=list(pool.drug_group.iloc[:32]), adapt_pool_size=int(len(pool)),
                query_test=list(t.drug_group[t.role == "test"]), query_val_secondary=list(t.drug_group[t.role == "val"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-method", required=True, help="name in model_selection.json['final'] that is the frozen proposal")
    ap.add_argument("--attribution", default="", help="comma-separated extra learned names evaluated as attribution/controls")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    assert dirty == "", "working tree has uncommitted changes"
    for f in ("baseline_lambda_selection.json", "tsg_ft_selection.json", "model_selection.json", "data_contract.json", "source_families.tsv"):
        assert os.path.exists(os.path.join(MAN, f)), f"{f} is not committed"
    sel = current_selection(committed_only=True)
    names = [a.final_method, "generic_cnp"] + [n for n in a.attribution.split(",") if n]
    assert all(n in sel["learned"] for n in names), (names, list(sel["learned"]))
    learned = {n: sel["learned"][n] for n in names}
    for n, seeds in learned.items():
        assert sorted(seeds) == [str(s) for s in cfg["proposed"]["seeds"]]
        for e in seeds.values():
            assert sha256(e["checkpoint"]) == e["checkpoint_sha256"]
    fb = os.path.join(os.environ["R2_PROC"], "feature_basis.npz")
    ev = cfg["evaluation"]["hcc"]
    fz = dict(track="episodic_residual_r2", frozen_on=subprocess.run(["date", "-Is"], capture_output=True, text=True).stdout.strip(),
              code_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              config=dict(path=CFG, sha256=sha256(CFG)), final_method=a.final_method, attribution_methods=[n for n in a.attribution.split(",") if n], note=a.note,
              selection=dict(lambdas=sel["lambda"], tsg_ft=sel["tsg_ft"], learned=learned,
                             modes={a.final_method: ["zero_residual", "shuffle_residual"]}),
              transigen_checkpoints={str(s): dict(run_id=e["run_id"], checkpoint=e["checkpoint"], checkpoint_sha256=e["checkpoint_sha256"]) for s, e in G.checkpoints().items()},
              feature_basis=dict(path=fb, sha256=sha256(fb)), data_contract_sha256=sha256(os.path.join(MAN, "data_contract.json")),
              source_families_sha256=sha256(os.path.join(MAN, "source_families.tsv")),
              k_values=cfg["episodes"]["k_values"], k_main=cfg["episodes"]["k_main"], model_seeds=ev["model_seeds"], split_seeds=ev["split_seeds"],
              target_validation_labels=0, bootstrap=ev["bootstrap"],
              targets={t: {str(s): support_and_query(t, s) for s in ev["split_seeds"]} for t in TARGETS},
              hcc_expression_values_read_before_freeze_in_this_study=False,
              exposure="The HCC data and these test splits were scored in the prior study (d67bf68). This is a post-freeze evaluation on a reused public benchmark.")
    for t in TARGETS:
        for s, e in fz["targets"][t].items():
            assert not (set(e["support_order"]) & set(e["query_test"])) and not (set(e["support_order"]) & set(e["query_val_secondary"]))
    out = public("manifests", "episodic_residual_r2", "freeze_manifest.json")
    assert not os.path.exists(out) and not os.path.exists(os.path.join(MAN, "freeze_manifest.json")), "already frozen"
    dump(fz, out)
    print(json.dumps({k: v for k, v in fz.items() if k != "targets"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
