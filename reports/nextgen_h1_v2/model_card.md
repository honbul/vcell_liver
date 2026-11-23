# H1-v2 model card

## Intended operation

H1-v2 predicts drug sensitivity on the registered, unmeasured query compounds in cancer cell-line and organoid experiments. The two prespecified algorithms are H1-v2-Dual-Best and H1-v2-AnchorOnly-Best. Each uses nested fit conditions to select one family per fit fold. The archived freeze enumerates the selected estimator, hyperparameters, feature support and reference set. Tier 4 uses a LIMORE source atlas without a local LICOB fit and uses no RNA.

The operating anchor budget is K=5. Accepted budgets are 0, 1, 3, 5, 8 and 12. The first five budgets share the v1 Q8 drug set. K=12 uses the separately registered Q12. A fit-selected ordering changes which compounds are assayed at K=1/3/5. Q8 support stays fixed.

## Runtime

Use a local, hash-checked G1 package with installed scientific Python dependencies. Run the CPU CLI from the H1-v2 worktree or the signed source snapshot.

```bash
PYTHONPATH=src python -m nextgen_h1_v2.cli \
  --model /mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h1_v2/frozen-a/LICOB-0 \
  --anchors anchors.tsv --k 5 --arm H1-v2-AnchorOnly-Best --out predictions.tsv
```

The TSV has `compound_id` and `response` columns. Use the exact prefix printed in the relevant package manifest. For K=0, use a header-only TSV. LICOB local values use negative native S6 AUC. LIMORE uses native Activity Area. Pan-cancer studies use negative fitted AUC. Tier 4 uses 1 minus AUC on the target and the registered fitted source scale. The CLI rejects missing, duplicate, nonfinite and out-of-envelope inputs. It returns an ordered query list, scores and ranks.

For H1-v2-Dual-Best add `--rna ranked_rna.npy`. The array must contain the single-sample rank-normalized values on the original registered fit gene axis. Rank normalization matches v1, `(average_rank - 0.5) / number_of_genes - 0.5`. The CPU CLI derives its RNA structure-kernel prediction and PCA projection from the registered fit rows.

## Comparability and monitoring

The main evidence uses previously exposed v1 cohorts and fixed independent lineage or donor folds. Cases outside the complete v1 A8 comparison cohort lack matched evidence. K=12 adds eligibility requirements and reduces Q, so its scores are reported on separate support. The B3 arm uses the corrected shrinkage shift. The strongest v2 simple comparator is selected by inner NDCG@5, and the archived v1 strongest comparator is marked separately.

Offline replay checks both champion roles against archived batch predictions and exercises the TSV command and invalid-input handling. RNA shuffle preserves anchor observations. Anchor shuffle preserves each entire measured packet and RNA. For the 96-well audit, the simulated anchor perturbation is an endpoint-level sensitivity analysis.
