# Manuscript claims and supporting evidence

| Claim | Status | Evidence and scope |
|---|---|---|
| Shared vehicle means changed validation and gain selection | Development evidence | Matched physical control count, independent roles and paired comparisons on the existing GSE92742 development support. [D1 and D2](../D12_RESULTS.md) |
| Earlier same-drug transcriptional measurements improved later HEPG2 prediction | Prespecified within-assay test passed | Exact exposure, held canonical parent and independent later plate, 132 compounds. The matched-dose wrong-parent and fit-mean controls were added after the original T4 result. [T4](../T4_RESULTS.md) · [Attribution](../T4_ATTRIBUTION_RESULTS.md) |
| The temporal model was superior to a directly measured different-plate 24 h replicate | Unresolved | The prespecified 99% paired RMSE and R² intervals both encompass zero on common support. [T4](../T4_RESULTS.md) |
| Standard R17 residual blend improved over a matched nonhepatic donor anchor | Development evidence | 493 parents and 2,324 HEPG2 wells. Same measured donor inputs across the comparison. Structural features in its neural branch had no demonstrated isolated advantage over shuffle. [S/L B](../SL_B_RESULTS.md) |
| Changing the head, loss or phenotype features delivered a practical performance gain | Gate unmet | Canonical-parent-paired A and B ablations and missing-view fallback tests. Low-rank heads reduced operations in S/L. The I5 kernel effect was a small replay gain. [S/L A](../SL_RESULTS.md) · [S/L B](../SL_B_RESULTS.md) · [I5](../I5_RESULTS.md) |
| The within-LINCS temporal effect generalized across assay or cell types | Unresolved | Frozen directions had mean cosine from −0.017 to +0.024 across the two previously exposed exact-condition HepaRG sets, with only 3 and 5 parents. Both cellular context and assay changed. [HepaRG](../T4_CROSS_ASSAY_RESULTS.md) |
| The method outperforms existing virtual cells in adult HCC | Unverified | A matched external comparison with adult HCC and predeclared benchmarks is needed. R18 C1 was evaluated under R16 before reservation. R18 C3 is a small HUH7 technical support set. R18 itself currently has metadata, synthetic processing tests and structure inputs. [External confirmation scope](../R18_CONFIRMATION_PROTOCOL.md) |

The public human HCC studies examined in a targeted follow-up primarily support one- or two-compound comparisons. The audit separates registered sequencing entries from source specimens and records inconsistent sample assignments. [Public HCC support audit](../HCC_PUBLIC_SUPPORT.md)

The final manuscript requires separate evidence for the adult HCC claim, documented data and code access, and author-confirmed contributions and declarations. The development draft can be read alongside the outcome files. [Submission guidance](https://www.nature.com/srep/author-instructions/submission-guidelines)

## Regenerate the figure

Use Python with Matplotlib and the repository's PyYAML dependency. The script checks the score table hashes against their receipts and uses the recorded relative RMSE and cosine values.

```bash
python -m src.r19_plan.t4_evidence_figure \
  --data reports/r19_plan/data \
  --output reports/r19_plan/manuscript/figure1.pdf
```
