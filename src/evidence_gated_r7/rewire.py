"""Degree- and channel-preserving rewired graph (attribution control): within each relation channel, destination endpoints
are permuted (out-degree and in-degree per channel kept); self-loops / duplicates are re-drawn up to 50 times then dropped."""
import json
import numpy as np

from .common import kg, pman, dump, sha256

SEED = 20261118


def main():
    ei = np.load(kg("edge_index.npy")); rel = np.load(kg("edge_rel.npy"))
    rng = np.random.RandomState(SEED)
    new = ei.copy()
    for r in np.unique(rel):
        k = np.where(rel == r)[0]
        src, dst = ei[0, k], ei[1, k].copy()
        for _ in range(50):
            rng.shuffle(dst)
            pairs = src.astype(np.int64) * 10 ** 6 + dst
            bad = (src == dst)
            _, first = np.unique(pairs, return_index=True)
            dup = np.ones(len(k), bool); dup[first] = False
            if not (bad | dup).any():
                break
        new[1, k] = dst
    pairs_old = set(map(tuple, ei.T.tolist())); pairs_new = list(map(tuple, new.T.tolist()))
    kept = np.mean([p in pairs_old for p in pairs_new])
    np.save(kg("edge_index_rewired.npy"), new)
    dump(dict(seed=SEED, edges=int(ei.shape[1]), self_loops=int((new[0] == new[1]).sum()), frac_edges_unchanged=float(kept),
              sha256=sha256(kg("edge_index_rewired.npy"))), pman("rewired_graph.json"))
    print("rewired", kept)


if __name__ == "__main__":
    main()
