"""Addendum A-2 inputs: XPert 'morgan' drug features = the R7 ECFP4 2048 bits folded to 1024 (OR of halves), keyed by R4
group index; a zero HG embedding. Metadata only."""
import json, os
import numpy as np

from .common import tb, pman, dump, sha256
from .data import Store

W = os.path.join(os.environ["R7_ASSETS"], "xpert_work")


def main():
    S = Store()
    fp = S.group_fp > 0
    fold = (fp[:, :1024] | fp[:, 1024:]).astype(np.float32)
    feat = {int(g): fold[g] for g in range(len(fold))}
    np.save(os.path.join(W, "r7_drug_morgan1024.npy"), feat, allow_pickle=True)
    np.save(os.path.join(W, "r7_drug_hg_zero.npy"), np.zeros((len(fold), 256), np.float32))
    rec = dict(groups=len(fold), bits_in=2048, bits_out=1024, fold="OR of bits [0:1024] and [1024:2048]",
               mean_on_bits=float(fold.sum(1).mean()), hg="zeros (8288 x 256)",
               sha256={f: sha256(os.path.join(W, f)) for f in ["r7_drug_morgan1024.npy", "r7_drug_hg_zero.npy"]})
    dump(rec, pman("addendum_xpert_ecfp_inputs.json"))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
