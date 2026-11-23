"""Select the strongest simple comparator within the new fixed inner fits."""
import numpy as np

from nextgen_h1_anchor.metrics import Retrieval
from nextgen_h1_anchor.models import Fit
from nextgen_h1_anchor.tune import baseline_grid

from .boost import region
from .search import eligible


def inner_choices(states, policy, has_rna):
    grid = baseline_grid(has_rna)
    scores = {arm: [] for arm, items in grid.items() if items}
    for arm, opts in grid.items():
        for config in opts:
            folds = []
            for ctx, _, orders, _, _ in states:
                order = orders[policy]
                q = region(order, 5, len(ctx.mean))
                good = eligible(ctx, order, q, 5)
                model = Fit(ctx.train, ctx.mean, ctx.sd, order, rna_fit=ctx.bvalid, rna_oof=ctx.bfit)
                prediction = model.predict(ctx.valid[good][:, order[:5]], 5, config,
                                           rna_index=np.flatnonzero(good))[:, q]
                y = ctx.valid[good][:, q]
                measures = Retrieval(y)(prediction)
                folds.append((float(np.nanmean(measures[:, 0])), -float(np.nanmean(np.abs(y - prediction)))))
            ndcg, neg_mae = np.mean(folds, axis=0)
            metrics = round(float(ndcg), 12), float(neg_mae)
            scores[arm].append((metrics, config))
    winners = {arm: max(candidates, key=lambda e: e[0]) for arm, candidates in scores.items()}
    strongest = max(winners, key=lambda arm: (winners[arm][0][0], -int(arm[-1])))
    return {"configs": {arm: cfg for arm, (_, cfg) in winners.items()}, "strongest": strongest,
            "inner_ndcg": {arm: float(perf[0]) for arm, (perf, _) in winners.items()}}
