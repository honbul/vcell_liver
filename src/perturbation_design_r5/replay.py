"""Sequential-reveal replay. The whole selection sequence is run first (policy sees only revealed values);
query truth is opened afterwards, only to score the snapshots at each budget K."""
import argparse, json, os, pickle, time
from multiprocessing import get_context
import numpy as np
import pandas as pd

from .common import protocol, runs, public, dump_json, nproc, commit, sha_hex, MAN
from .data import Store, make_episode, target_cells
from .models import EpisodeKernels, Readout
from .oracle import Oracle, Truth, score
from .acquisition import Context, POLICIES

HYPER = os.path.join(MAN, "hyper.json")


def load_hyper(path=HYPER):
    with open(path) as f:
        hp = json.load(f)
    if hp.get("learned_loss_model_path") and os.path.exists(hp["learned_loss_model_path"]):
        with open(hp["learned_loss_model_path"], "rb") as f:
            hp["learned_loss_model"] = pickle.load(f)
    return hp


def run_selection(st, ep, policy, hp, budgets, override=None, record_rounds=False):
    """Returns the revealed sequence (indices), revealed values, snapshots {K: n revealed}, cost and wall time."""
    oracle = Oracle(st, ep, override)
    ek = EpisodeKernels(ep, st.bases)
    S = [int(i) for i in ep.initial]
    Y = [oracle.reveal(i) for i in S]
    kmax = min(max(budgets), len(ep.pool))
    wall = 0.0
    rnd = 0
    rounds = []
    while len(S) < kmax:
        Ya = np.stack(Y)
        Z = ek.to_z(ep, np.array(S), Ya)
        ctx = Context(ep, ek, hp, S, Ya, Z, rnd, int(sha_hex(f"rnd|{ep.cell}|{ep.seed}|{rnd}|{policy}")[:8], 16))
        t = time.perf_counter()
        b = POLICIES[policy](ctx)
        wall += time.perf_counter() - t
        b = b[:kmax - len(S)]
        assert len(set(b)) == len(b) and not (set(b) & set(S)), "policy returned an invalid batch"
        if record_rounds:
            rounds.append(dict(round=rnd, K_before=len(S), chosen=[ep.groups[i] for i in b]))
        for i in b:
            Y.append(oracle.reveal(i))
            S.append(int(i))
        rnd += 1
    return dict(S=S, Y=np.stack(Y), ek=ek, cost=oracle.cost(), ledger=oracle.ledger, wall=wall, rounds=rounds)


def score_snapshots(st, ep, sel, readouts, budgets, truth=None):
    truth = truth or Truth(st, ep)
    out, per_unit = [], []
    for K in budgets:
        if K > len(sel["S"]):
            continue
        S = np.array(sel["S"][:K])
        for name, ro in readouts.items():
            pred = ro.predict(ep, sel["ek"], S, sel["Y"][:K], ep.query)
            sc, rm = score(pred, truth)
            out.append(dict(K=K, readout=name, **sc))
            if name == "main":
                qg = ep.groups[ep.query][truth.mask]
                per_unit.append(pd.DataFrame(dict(K=K, group=qg, rmse=rm)))
    return out, (pd.concat(per_unit) if per_unit else None)


def auc(ks, errs):
    ks = np.asarray(ks, float)
    errs = np.asarray(errs, float)
    if len(ks) < 2:
        return float("nan")
    return float(np.trapz(errs, ks) / (ks[-1] - ks[0]))


def readouts_from(hp):
    r = hp["readout"]
    return {"main": Readout(r["alpha"], r["lam"], "main"), "mean_offset": Readout(0, 1, "mean_offset")}


# ------------------------------------------------------------------ parallel driver
_G = {}


def _init(role, stress):
    _G["st"] = Store(allow_target=(role == "hcc"))
    _G["pr"] = protocol()
    _G["hp"] = load_hyper()
    _G["stress"] = stress


def _job(args):
    role, cell, family, seed, policy = args
    st, pr, hp = _G["st"], _G["pr"], _G["hp"]
    budgets = pr["episodes"]["budgets_K"]
    ep = make_episode(st, role, cell, family, seed, stress=_G["stress"], pr=pr)
    sel = run_selection(st, ep, policy, hp, budgets, record_rounds=True)
    rows, pu = score_snapshots(st, ep, sel, readouts_from(hp), budgets)
    meta = dict(role=role, cell=cell, family=family, seed=seed, policy=policy, n_groups=len(ep.groups), n_query=int(len(ep.query)),
                n_pool=int(len(ep.pool)), wall_s=round(sel["wall"], 3), **sel["cost"])
    for r in rows:
        r.update({k: meta[k] for k in ("role", "cell", "family", "seed", "policy")})
    if pu is not None:
        for k in ("cell", "family", "seed", "policy"):
            pu[k] = meta[k]
    seq = dict(meta, sequence=[ep.groups[i] for i in sel["S"]], rounds=sel["rounds"])
    return rows, pu, seq


def replay(role, policies, seeds, tag, stress=False, cells=None):
    st = Store(allow_target=(role == "hcc"))
    pr = protocol()
    tc = target_cells(st, role, pr)
    if cells:
        tc = [c for c in tc if c[0] in cells]
    jobs = [(role, c, f, s, p) for (c, f) in tc for s in seeds for p in policies]
    out_dir = os.path.join(runs(), tag)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(nproc(), initializer=_init, initargs=(role, stress)) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    rows = pd.DataFrame([r for rr, _, _ in res for r in rr])
    pu = pd.concat([p for _, p, _ in res if p is not None])
    seqs = [s for _, _, s in res]
    rows.to_csv(os.path.join(out_dir, "scores.tsv"), sep="\t", index=False)
    pu.to_csv(os.path.join(out_dir, "per_unit.tsv.gz"), sep="\t", index=False)
    with open(os.path.join(out_dir, "sequences.jsonl"), "w") as f:
        for s in seqs:
            f.write(json.dumps(s, default=str) + "\n")
    info = dict(tag=tag, role=role, policies=policies, seeds=seeds, stress=stress, cells=[c for c, _ in tc], commit=commit(),
                seconds=round(time.time() - t0, 1), n_jobs=len(jobs))
    dump_json(info, os.path.join(out_dir, "info.json"))
    return rows, pu, seqs, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["dev", "meta_val"])
    ap.add_argument("--policies", required=True)
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stress", action="store_true")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    rows, pu, seqs, info = replay(a.role, a.policies.split(","), list(range(lo, hi + 1)), a.tag, a.stress)
    from .evaluate import summarize_run
    summarize_run(a.tag)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
