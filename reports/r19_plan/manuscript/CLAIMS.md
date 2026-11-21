# Manuscript claims and supporting evidence

| Claim | Status | Evidence and scope |
|---|---|---|
| Shared vehicle means changed validation and gain selection | Development evidence | Matched physical control count, independent roles and paired comparisons on the existing GSE92742 development support. [D1 and D2](../D12_RESULTS.md) |
| Earlier same-drug transcriptional measurements improved later HEPG2 prediction | Prespecified within-assay test passed | Exact exposure, held canonical parent and independent later plate, 132 compounds. The matched-dose wrong-parent and fit-mean controls were added after the original T4 result. [T4](../T4_RESULTS.md) · [Attribution](../T4_ATTRIBUTION_RESULTS.md) |
| The temporal model was superior to a directly measured different-plate 24 h replicate | Calibrated repeat outperformed the 6 h model | The original unscaled comparison was unresolved in HEPG2. Fitting a scalar on different 24 h plates from fitting parents yielded a better 24 h replicate comparator in both HEPG2 and PHH. [Calibrated repeat](../T4_REPEAT_RESULTS.md) |
| Standard R17 residual blend improved over a matched nonhepatic donor anchor | Development evidence | 493 parents and 2,324 HEPG2 wells. Same measured donor inputs across the comparison. Structural features in its neural branch had no demonstrated isolated advantage over shuffle. [S/L B](../SL_B_RESULTS.md) |
| Independently measured basal RNA improved context conditioning with the same measured RNA donors | Gate unmet | The matched B comparison showed higher RMSE and lower pooled R² for basal RNA than identity/OOV on five held-cell contexts. [B context](../XB_RESULTS.md) |
| A gene-wise vehicle variance correction improved over a scalar across L1000 contexts | Practical gate unmet | The structure-only follow-up showed small harm relative to a scalar. A separate donor-informed comparison yielded a gain below the registered practical threshold. [A calibration](../TA_RESULTS.md) · [B calibration](../T123_RESULTS.md) |
| Changing the head, loss or phenotype features delivered a practical performance gain | Gate unmet | Canonical-parent-paired A and B ablations and missing-view fallback tests. Low-rank heads reduced operations in S/L. The I5 kernel effect was a small replay gain. [S/L A](../SL_RESULTS.md) · [S/L B](../SL_B_RESULTS.md) · [I5](../I5_RESULTS.md) |
| The within-LINCS temporal effect generalized across assay or cell types | Unresolved | Frozen directions had mean cosine from −0.017 to +0.024 across the two previously exposed exact-condition HepaRG sets, with only 3 and 5 parents. Both cellular context and assay changed. [HepaRG](../T4_CROSS_ASSAY_RESULTS.md) |
| A temporal correction also helped within primary human hepatocytes | Limited to scalar versus zero response | On 67 held parents, a fit-only scalar improved over B0. A fixed rank-16 operator did not improve both endpoints over the scalar. The operators were fitted and scored within PHH. [PHH](../T4_PHH_RESULTS.md) |
| The method outperforms existing virtual cells in adult HCC | Unverified | A matched external comparison with adult HCC and predeclared benchmarks is needed. R18 C1 was evaluated under R16 before reservation. R18 C3 is a small HUH7 technical support set. R18 itself currently has metadata, synthetic processing tests and structure inputs. [External confirmation scope](../R18_CONFIRMATION_PROTOCOL.md) |
| SNU-423 high-dose analogue copying demonstrated a broad chemical advantage | Unverified | Six of 203 parents supplied 90.8% of positive SSE savings. The original R² point of +8.11 points versus the best comparator had a scaffold-component sensitivity interval of −0.64 to +15.37 points. [Tahoe scaffold sensitivity](../HCC_SCAFFOLD_RESULTS.md) |
| Same-drug nonhepatic RNA rescued structurally distant SNU-423 prediction | Gate unmet | The fixed rank-4 map increased RMSE against the matched scalar and RNA-augmented State on 94 held parents with Tanimoto below 0.7. Most source measurements were 10 µM and all target measurements were 5 µM. [External donor analysis](../SNU423_DONOR_RESULTS.md) |
| The Tahoe SNU-423 repeats establish drug-response reliability across well positions | Insufficient support | The exact-entity comparison had two different-position parents per dose and 50 same-position parents at 5 µM. No target well had both donor types. [Repeat analysis](../TAHOE_HCC_REPEAT_RESULTS.md) |

The public human HCC studies examined in a targeted follow-up primarily support one- or two-compound comparisons. The audit separates registered sequencing entries from source specimens and records inconsistent sample assignments. [Public HCC support audit](../HCC_PUBLIC_SUPPORT.md)

The final manuscript requires separate evidence for the adult HCC claim, documented data and code access, and author-confirmed contributions and declarations. The development draft can be read alongside the outcome files. [Submission guidance](https://www.nature.com/srep/author-instructions/submission-guidelines)

## Regenerate the figure

Use Python with Matplotlib and the repository's PyYAML dependency. The script checks the score table hashes against their receipts and uses the recorded relative RMSE and cosine values.

```bash
python -m src.r19_plan.t4_evidence_figure \
  --data reports/r19_plan/data \
  --output reports/r19_plan/manuscript/figure1.pdf
```
