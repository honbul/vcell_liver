"""Estimate LICOB sample-size sensitivity on the existing inner validation donors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h1_anchor.inputs import fold_order, json_write
from nextgen_h1_anchor.metrics import Retrieval
from vc_build.io import sha256

from .anchors import nested_contexts
from .boost import region
from .data import drug_features, source, training_context, within_drug
from .freeze import FAMILIES
from .search import cache_read, eligible, fit_model


def probe(round1, final, out, device):
    out.mkdir()
    rows, hashes = [], {}
    for task in (f"LICOB-{j}" for j in range(3)):
        study, dev, meta = source(task)
        drug = drug_features(meta)
        states, policy, _, _, _ = cache_read(round1 / task)
        entries = {}
        for family in FAMILIES:
            file = final / task / family / "result.json"
            stamp = json.loads(file.read_text())
            if stamp["outer_labels_opened"] or any(m["remaining_numerical_boundary_trials"]
                                                   for m in stamp["modes"].values()):
                raise ValueError("The sample-size probe requires completed inner searches")
            entries[family] = stamp["modes"]
            hashes[f"{task}/{family}"] = sha256(file)
        for mode in ("anchor", "dual"):
            contenders = sorted(FAMILIES, key=lambda f: entries[f][mode]["selected"]["objective"], reverse=True)[:2]
            for fold, (reference, _, orders, fit, val) in enumerate(states):
                order = orders[policy[mode]]
                sequence = fold_order(dev["groups"][fit], f"h1v2-sample-{task}-{fold}")
                vt = np.isin(np.arange(len(dev["y"])), val)
                q = region(order, 5, len(meta))
                observed = eligible(reference, order, q, 5)
                for fraction in (.50, .75, 1.):
                    n = max(8, int(np.ceil(fraction * len(fit))))
                    use = np.isin(dev["groups"], sequence[:n])
                    ctx = training_context(study, dev, use, vt, drug)
                    sub = nested_contexts(study, dev, use, drug)
                    for family in contenders:
                        config = entries[family][mode]["selected"]["config"]
                        model = fit_model(ctx, sub, order, mode == "dual", config, device, {})
                        score = model.predict(ctx.valid[:, order[:5]], 5)[observed][:, q]
                        truth = ctx.valid[observed][:, q]
                        metric = np.nanmean(Retrieval(truth)(score), axis=0)
                        rows.append(dict(task=task, inner_fold=fold, modality=mode, family=family,
                                         training_fraction=fraction, fit_groups=int(use.sum()),
                                         val_groups=int(observed.sum()), q=len(q), ndcg=float(metric[0]),
                                         precision=float(metric[1]), rho=within_drug(truth, score)))
            print(task, mode, "sample size diagnostics complete", flush=True)
    file = out / "H1V2_LICOB_SAMPLE_SIZE.tsv"
    pd.DataFrame(rows).to_csv(file, sep="\t", index=False, float_format="%.9g")
    json_write(out / "size_probe_receipt.json", {"rows": len(rows), "outer_labels_opened": False,
                                                 "table_sha256": sha256(file), "family_receipts": hashes,
                                                 "design":"fixed fold reserve and recipe, nested 50/75/100 percent inner-fit subsets"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("round1", "final", "out"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--device", default="cpu")
    probe(**vars(parser.parse_args()))
