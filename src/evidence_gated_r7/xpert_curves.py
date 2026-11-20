"""Extract the XPert validation selection metric (upstream loss4 = 1 - mean delta PCC) per epoch from the training logs.
Evidence for whether the 100-epoch cap (D-3) was binding. Metadata from logs only."""
import glob, json, os, re
import numpy as np
import pandas as pd

from .common import pman, dump

RUNS = ["xpr_e100_s42", "xpr_e100_s43", "xpr_e100_s44", "xpe_e100_s42", "xpe_e100_s43", "xpe_e100_s44"]
PAT = re.compile(r"Valid Total Loss: [\d.]+, Loss1: [\d.]+, Loss2: [\d.-]+, Loss3: [\d.]+, Loss4: ([\d.]+)")


def main():
    root = os.path.join(os.environ["R7_ASSETS"], "xpert_runs_work")
    rows, summ = [], {}
    for r in RUNS:
        logs = glob.glob(os.path.join(root, r, "experiment", "l1000", "*", "*.log"))
        assert len(logs) == 1, (r, logs)
        v = [float(m.group(1)) for m in PAT.finditer(open(logs[0]).read())]
        for e, x in enumerate(v):
            rows.append(dict(run=r, epoch=e, val_selection_loss=x))
        v = np.array(v)
        tail = v[40:]
        summ[r] = dict(epochs=len(v), best_epoch=int(v.argmin()), best=float(v.min()),
                       at_epoch_40=float(v[40]), plateau_min_from_40=float(tail.min()), plateau_max_from_40=float(tail.max()),
                       improvement_after_40=float(v[40] - v.min()), plateau_range_from_40=float(tail.max() - tail.min()),
                       slope_per_epoch_last_40=float(np.polyfit(np.arange(len(v[-40:])), v[-40:], 1)[0]))
    pd.DataFrame(rows).to_csv(pman("xpert_val_curves.tsv"), sep="\t", index=False)
    dump(dict(metric="upstream validation selection loss (1 - mean delta Pearson), lower is better",
              per_run=summ,
              reading="the improvement after epoch 40 is within the plateau's own oscillation range, and the slope over the last 40 epochs is ~0",
              caveat="evidence that the 100-epoch cap was not materially binding; it does not prove that 2500 epochs would not differ"),
         pman("xpert_cap_check.json"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
