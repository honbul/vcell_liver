"""Development-only chemical correspondence and remaining liver-context diagnostics."""

import itertools

import numpy as np

from .dataset import ENTITY, GROUP, LINE


def permuted_drugs(block, seed):
    """Permute entire compound groups with matching entity-to-plate patterns."""
    rows = block.rows[[GROUP, ENTITY, "plate_id"]].drop_duplicates()
    strata = {}
    for group, frame in rows.groupby(GROUP, sort=True):
        pairs = [(tuple(sorted(e.plate_id)), name) for name, e in frame.groupby(ENTITY, sort=True)]
        pairs.sort()
        signature = tuple(plates for plates, _ in pairs)
        strata.setdefault(signature, []).append((group, [name for _, name in pairs]))
    rng = np.random.default_rng(seed)
    mapping = {}
    active = 0
    for entries in strata.values():
        n = len(entries)
        if n < 2:
            continue
        active += n
        order = rng.permutation(n)
        for source, target in zip(order, np.roll(order, 1)):
            mapping.update(zip(entries[source][1], entries[target][1]))
    indices = {name: i for i, name in enumerate(block.rows[ENTITY])}
    permuted = [indices[mapping.get(entity, entity)] for entity in block.rows[ENTITY]]
    return block.chemical["C"][permuted], dict(
        groups_permuted=active, groups_unpermutable=rows[GROUP].nunique() - active,
        eligible_stratum_sizes=sorted(len(entries) for entries in strata.values() if len(entries) > 1),
        unpermutable_strata=sum(len(entries) < 2 for entries in strata.values())
    )


def context_recurrence(rows, residual, liver, seed):
    """Compare between-line OOF residual contrasts for repeated groups across plates."""
    a, b = liver
    strata = {}
    for g, idx in rows[rows[LINE].isin(liver)].groupby(GROUP).groups.items():
        frame = rows.loc[idx]
        # A missing or aliased entity must not be taken for an exact repeat.
        for entity, e_idx in frame.groupby(ENTITY).groups.items():
            plates = {}
            for plate, p_idx in rows.loc[e_idx].groupby("plate_id").groups.items():
                line = rows.loc[p_idx, LINE]
                if set(line) == {a, b}:
                    plates[plate] = residual[line.index[line.eq(a)]].mean(0) - residual[line.index[line.eq(b)]].mean(0)
            for p, q in itertools.combinations(sorted(plates), 2):
                strata.setdefault((p, q), {}).setdefault(g, []).append((plates[p], plates[q]))
    rng = np.random.default_rng(seed)
    sources = []
    for pair, groups in sorted(strata.items()):
        if len(groups) < 3:
            continue
        pairs = np.stack([np.mean(matches, axis=0) for matches in groups.values()])
        norm = np.linalg.norm(pairs, axis=2, keepdims=True)
        pairs = pairs / np.maximum(norm, 1e-12)
        similar = pairs[:, 0] @ pairs[:, 1].T
        observed = float(np.diag(similar).mean())
        decoy = np.mean([similar[np.arange(len(groups)), rng.permutation(len(groups))].mean() for _ in range(199)])
        sources.append(dict(plate_pair=list(pair), n_groups=len(groups), matched_cosine=observed,
                            shuffled_cosine=float(decoy), excess=observed - float(decoy)))
    return dict(strata=sources, positive=bool(sources) and all(s["excess"] > 0 for s in sources))


def gate(selected, grid, context, folds):
    """Open only with replicated chemical gain and repeatable residual context signal."""
    chemical = grid[grid.selected].pivot(index="fold", columns="arm", values="contrast_rmse")
    best = selected["baseline"]
    direction = ((chemical["C"] < chemical["E"]) & (chemical["C"] < chemical[best]) &
                 (chemical["C"] < chemical["C_permuted"]))
    residual_folds = [int(f) for f in folds if context[str(f)]["positive"]]
    return dict(enter=bool(direction.all() and len(residual_folds) >= 2),
                all_folds_chemical_direction=bool(direction.all()),
                chemical_direction_folds=[int(f) for f in direction.index[direction]],
                residual_positive_folds=residual_folds,
                resource_rule="two CPU fits per family on existing allocation; no new GPU allocation")
