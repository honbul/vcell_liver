# Manuscript draft (v0.1, 2026-11-21)

Working draft written by the Claude Code review session (v0.1, before P4 and R15). All numbers are taken from frozen, pre-registered evaluations (Codex tracks T1–T3 and P1–P3); exploratory diagnostics are excluded from the Results and appear only where explicitly labelled.

Target: Scientific Reports (Article). Working title below.

---

## Title

**Well position confounds unseen-drug benchmarks in Tahoe-100M: a position-controlled evaluation of virtual cell models in liver-cancer lines**

Alternative, if reviewers prefer a positive framing:
*Controlling a plate-position nuisance changes which perturbation-response model wins on liver-cancer cell lines*

## Abstract (draft, ~200 words)

Virtual cell models aim to predict the transcriptional response of a cell line to a compound it has never seen. We evaluated ten published or re-implemented perturbation-response models on liver-cancer lines (SNU-423, HepG2/C3A) in the Tahoe-100M drug screen under a single frozen protocol with compound-group cross-validation. None of the ten reduced contrast RMSE below a zero-response baseline at any of three doses. We then show that Tahoe-100M carries a reproducible well-position effect: 332 of 380 compounds occupy the same within-plate position on every plate, the difference between the two DMSO control wells reproduces across plates (median cosine 0.54), and a predictor built only from other compounds at the same position beats the zero-response baseline (−2.64% contrast RMSE at 5 µM in liver lines, 99% CI excluding zero). A procedure that uses this position signal outperformed all ten models (−5.4% versus the best, 99% CI [−0.0087, −0.0034]). However, when the same position estimate was supplied to the published models, the advantage disappeared (99% CI included zero), and gains were concentrated in five compound groups. Drug-specific prediction beyond position passed its pre-registered gate only in HepaRG hepatocytes at the highest tested concentrations, using external LINCS L1000 signatures of the same compound.

## Key messages

1. Published virtual cell models do not beat a zero-response baseline on unseen drugs in liver-cancer lines under a matched, frozen protocol.
2. Tahoe-100M contains a well-position nuisance of the same magnitude as the drug-reproducibility signal; benchmarks that ignore it reward models for the wrong reason.
3. Controlling position is necessary before any claim of drug-specific prediction; our own position-aware model wins only because of position.
4. The one setting where drug-specific prediction passed a pre-registered gate was HepaRG at high concentration with external same-compound signatures — pointing to where the signal actually lives.

---

## 1. Introduction

Outline:
- Virtual cell / perturbation-response prediction: goal of predicting unseen compound effects; recent models (chemCPA, CPA, PRnet, XPert, TranSiGen, State, foundation-model contexts scGPT/UCE).
- Liver cancer motivation: HCC drug response, limited perturbation data on hepatic lines.
- Benchmarking problem: most reports evaluate on seen compounds, or on splits that do not isolate chemistry; baselines are often weak or absent.
- Independent corroboration of the position issue: arXiv 2611.01604 (Huang, 2026-11-01) reports that cell lines sharing a well in Tahoe-100M are not independent observations, with control pairs from other plates predicting at cosine 0.47. We arrived at the same conclusion independently and quantify its effect on model ranking.
- Contributions (list the four key messages).

## 2. Results

### 2.1 Evaluation protocol

Tahoe-100M, 24 h wells, 203 canonical parent compound groups, 2,000 variance-selected genes fitted per dose on training rows only, log1p CP10k pseudobulk per well × cell line. The prediction target is the *contrast*: the well × line expression vector after subtracting the mean of its pool (plate × cell line × dose). Three doses (0.05, 0.5, 5 µM); minimum 20 cells per row (200 as sensitivity). Three-fold cross-validation over compound groups; evaluation groups are removed from training at **all** doses, plates and cell lines. Primary readouts: compound-group-equal contrast RMSE, and pooled explained variance R²_pooled = 1 − ΣSSE/ΣSST. Uncertainty from 10,000 compound-group bootstrap resamples with pool centres recomputed each replicate. Protocol was committed before scoring; predictions were frozen (SHA-256) and scored by a separate job.

Baselines: **B0** (zero contrast, i.e. "no response"), **B_abs** (absolute-level anchor), **B_pos** (mean response of training compounds at the same within-plate position on other plates, same cell line, same dose), **B_pos-all** (same, pooled across doses).

### 2.2 Ten virtual cell models do not beat a zero-response baseline

*Table 1.* 5 µM, liver lines, ≥20 cells, contrast RMSE (B0 = 0.108676):

| Model | contrast RMSE | vs B0 | direction cosine |
|---|---|---|---|
| UCE_C8 | 0.109437 | +0.70% | 0.000 |
| Concat_R12 | 0.109463 | +0.72% | −0.000 |
| scGPT_C7 | 0.109465 | +0.73% | 0.001 |
| FiLM_R12 | 0.109714 | +0.95% | 0.011 |
| chemCPA_P0 | 0.110907 | +2.05% | 0.043 |
| XPert_morgan | 0.114028 | +4.92% | 0.014 |
| TranSiGen_Tahoe | 0.114322 | +5.20% | −0.001 |
| State-Morgan (well adaptation) | 0.115934 | +6.68% | 0.005 |
| ChemBERTa_R11 | 0.120739 | +11.10% | 0.022 |
| PRnet (well adaptation) | 0.123549 | +13.69% | 0.012 |

All ten are worse than predicting no response. Direction cosines are near zero. The same ordering holds at 0.5 µM and at the 200-cell threshold. The public State ST-Tahoe checkpoint was excluded from quantitative comparison because exposure of the evaluation compounds in its training split could not be ruled out; a structure-conditioned State adaptation and PRnet were re-fit from scratch on training folds and are included.

*Point to make in the text:* this is not a failure to reproduce the original papers. Those papers evaluate under different splits, panels and targets. The claim is narrower and testable: under a protocol where the evaluation compound is unseen at every dose and the target is the within-pool contrast, these architectures carry no usable unseen-drug signal in these two liver lines.

### 2.3 A reproducible well-position effect

- 332 of 380 Tahoe compounds occupy the same within-plate position (rank of `smp_` index) on every plate where they appear. "Same compound, different plate" is therefore confounded with "same position".
- The contrast between the two DMSO control wells (positions 95 and 96) reproduces across plates: median cosine 0.536 over 78 plate pairs.
- B_pos, built only from *other* compounds at the same position, beats B0: at 0.5 µM, 0.090610 vs 0.092792, −2.35%, 99% CI [−0.002961, −0.001119]. Three position-label permutations selected scale zero, i.e. the effect is tied to the label, not to the estimator.
- At 5 µM in liver lines: B_pos 0.107500 (−1.08%), B_pos-all 0.105804 (−2.64%), both relative to B0 = 0.108676.
- Structure: the effect reproduces across drug-set-disjoint plate halves at cosine ≈0.45; different positions ≈0.00. No adjacency, row, column or edge structure in either 8×12 orientation; the leading principal components explain 15/13/10%. Regressing out cell count and sequencing depth does not remove it. *Interpretation (ours, not tested):* this behaves like a per-well, barcode-associated effect rather than culture-plate geometry. The public release provides a first-barcode index that matches the `smp_` rank on all 1,344 wells, but no A1–H12 culture map, so physical-coordinate validation requires the original plate design.

*Consequence:* "same compound measured on another plate or at another dose" is not an independent replicate. Earlier cross-plate repeatability estimates in this and other work overstate drug-specific signal.

### 2.4 A position-aware procedure beats all ten models — for the wrong reason

Proposed model **v1** = trimmed position component + nearest structural sibling residual. The trim drops donor rows whose L2 norm exceeds a threshold (1.25/1.5/2.0× the cell-line donor median, chosen in inner folds) to limit contamination of the position estimate by strong responders sharing that position in other drug sets. The chemistry arm copies the position-corrected response of the nearest training compound above a Tanimoto threshold, or averages structure/target/MoA sibling families; family and scales were chosen in inner folds only.

*Table 2.* 5 µM, liver lines, ≥20 cells:

| Arm | contrast RMSE | vs B0 | R²_pooled |
|---|---|---|---|
| B0 | 0.108676 | — | 0 |
| B_pos-all | 0.105804 | −2.64% | 3.39% |
| v0 (B_pos-all + model residual) | 0.105506 | −2.92% | 4.77% |
| trimmed position alone | 0.104251 | −4.07% | 4.87% |
| **v1** | **0.103588** | **−4.68%** | **8.96%** |

*Table 3.* Pre-registered gates (99% CI of the paired difference; RMSE upper bound < 0 and ΔR² lower bound > 0 required):

| Comparison | ΔRMSE [99% CI] | ΔR²_pooled [99% CI] | Verdict |
|---|---|---|---|
| v1 − B0 | −0.005087 [−0.008971, −0.002587] | +8.96 pp [+3.34, +18.95] | **pass** |
| v1 − best known VC | −0.005849 [−0.008684, −0.003383] | +8.89 pp [+4.00, +15.54] | **pass** |
| v1 − trimmed position | −0.000663 [−0.004757, +0.001388] | +4.09 pp [−0.99, +14.05] | fail |
| v1 − best *position-matched* known VC | −0.000709 [−0.004099, +0.001333] | +3.57 pp [−1.02, +9.38] | fail |

The fourth row is the decisive one. We gave each of the ten published models the same trimmed position estimate (as an additive term with scales chosen in inner folds), and additionally re-fit five of them (ChemBERTa, FiLM, concat, scGPT-context, UCE-context) against position-corrected training targets. Against the best of these fifteen position-matched comparators, v1's advantage is not distinguishable from zero. The win in row 2 is therefore attributable to position information, not to chemistry.

Trimming itself was a real improvement over the plain position mean: ΔRMSE −0.001553 [−0.002172, −0.000766] in liver lines, −0.001224 [−0.001819, −0.000497] across all lines.

### 2.5 The residual gain is concentrated in five compound groups

Of the positive squared-error reduction of v1 over the trimmed position baseline at 5 µM, the top 5 / 10 / 20 compound groups account for 82.6% / 87.4% / 91.9% in liver lines (87.0% / 89.9% / 93.5% across all lines). The top five are idarubicin, doxorubicin/epirubicin, triamcinolone, dexamethasone and tetracycline — two anthracyclines and two glucocorticoids, i.e. families where a close structural sibling is present in the training folds and the response is large.

Removing those five groups and recomputing pool centres reverses the ordering: v1 0.102773 vs trimmed position 0.101711 in liver lines; R²_pooled 3.99% vs 5.14%. This exclusion analysis is post hoc and outcome-dependent (the groups were ranked by observed error reduction), and is reported as a sensitivity analysis, not as a gate.

*Honest reading:* structural-sibling transfer works where a near-identical compound with a large, stereotyped response is available. It does not generalise across the compound panel.

### 2.6 External LINCS signatures do not help in Tahoe, but do in HepaRG at high concentration

**Tahoe.** 144 of 203 compound groups (70.9%) matched a 24 h LINCS L1000 Level 5 consensus signature after de-salting and InChIKey/structure-verified matching (290 structure-verified assignments; 14 name candidates rejected on structure conflict). Adding a signature-cosine kernel to the structure kernel at the same position baseline did not pass its gate: all 203 groups ΔRMSE +0.000111 [−0.000752, +0.000779], ΔR² +0.62 pp [−0.68, +2.33]; covered 144 groups ΔRMSE +0.000214 [−0.000940, +0.001135], ΔR² +0.85 pp [−1.02, +3.19].

**HepaRG.** Two bulk TempO-Seq studies (GSE296254, GSE284321), joint compound-group folds, dose-rank 7–8 as primary. 827 of 978 LINCS landmarks mapped uniquely to both TempO-Seq assays; 162–164 intersected each fitted panel. Structure + LINCS passed both metrics against B0 in both studies, and against the dose-shared ridge in GSE296254:

| Study / dose rank | vs B0 ΔRMSE% [95% CI] | vs B0 ΔR² pp | vs ridge ΔRMSE% | vs ridge ΔR² pp |
|---|---|---|---|---|
| GSE296254 rank 7 (289 groups) | +0.648 [0.251, 1.057] | +1.718 [0.862, 2.592] | +0.402 [0.041, 0.778] | +1.067 [0.285, 1.907] |
| GSE296254 rank 8 (262 groups) | +1.874 [1.333, 2.417] | +4.183 [3.002, 5.368] | +1.378 [0.884, 1.887] | +3.152 [2.066, 4.269] |
| GSE284321 rank 7 (1,127) | +0.257 [0.028, 0.514] | +1.298 [0.430, 2.259] | +0.091 [−0.073, 0.275] | +0.753 [0.146, 1.427] |
| GSE284321 rank 8 (1,125) | +0.483 [0.223, 0.765] | +1.690 [0.910, 2.629] | +0.108 [−0.067, 0.297] | +0.614 [0.110, 1.226] |

Crucially, in GSE296254 rank 8 the gain over a structure-only model of identical design was confined to the 201 groups with an observed external signature (−1.292% RMSE [0.765, 1.845]); the 61 groups without one were slightly *worse* (−0.246% [−0.420, −0.073] in the wrong direction). Against an independent-culture, different-position replicate reference, the model closes 27.0% [20.1, 34.3] of the B0-to-replicate gap on RMSE.

*Caveat to state prominently:* a position-only baseline in GSE296254 rank 8 explains 13.8% of variance versus 4.2% for structure + LINCS. The HepaRG layout confound is characterised but not eliminated; the reported effect is measured against the study's own pool contrast.

### 2.7 How far is any of this from a true replicate?

Plate 14 repeats plate 6 at 5 µM at the same positions. Using the plate-6 replicate response (×0.25) as a predictor improves over B_pos-all by 0.009694 (all lines) / 0.011118 (liver). v0 reaches 5.57% [−7.61, 20.02] of that gap in liver lines. In other words, even a same-position replicate of the same compound leaves an order of magnitude more headroom than any model recovers. Forty plate-6↔14 pairs outside the development cohort were cached and left unopened for one-shot external confirmation.

## 3. Discussion

Points to make:
- Benchmark design, not architecture, decided the ranking here. Any Tahoe-based unseen-drug benchmark should report a position baseline and a position-corrected target.
- Our own proposed model is subject to the same criticism; we report the position-matched comparison that removes its advantage. We argue this is the comparison the field should adopt.
- Where drug-specific signal survives: high concentration, large-response compounds, and settings where an external measurement of the same compound exists. This suggests the near-term useful "virtual cell" is a transfer/lookup system over measured compounds rather than a de novo chemistry-to-response model.
- Limitations: two liver-cancer lines; pseudobulk targets; the position effect's physical cause is not identified; the HepaRG layout confound is not fully resolved; exclusion analyses are post hoc.
- What would change the conclusion: an independent-position replicate design in Tahoe; confirmation on the reserved 40 pairs and on compounds outside the 203-group panel.

## 4. Methods

(Sections to write from the frozen protocol documents; all content exists.)
- Data and preprocessing (Tahoe-100M; HepaRG GSE296254/GSE284321; LINCS GSE92742/GSE70138 Level 5 MODZ).
- Compound-group definition (canonical parent, de-salting, InChIKey), splits, dose handling.
- Target definition, gene panel fitting, pool centring, DMSO role assignment.
- Baselines and the proposed model; inner-fold selection grids.
- Comparator models: sources, licences, commit hashes, input adaptations (Table S1 — already drafted in `reports/goal_tracks/T2_COMPARATORS.md` and `PUBLIC_COMPARATORS.md`).
- Metrics, bootstrap, pre-registration and freezing procedure; code and spec commit hashes.

## 5. Data and code availability

- Tahoe-100M: public (tahoebio/Tahoe-100M). HepaRG: GSE296254, GSE284321. LINCS: GSE92742, GSE70138.
- Analysis code, frozen protocols and scoring receipts: research branch `research/r14-cross-organ-data`; spec `18163c8`, T1–T3 results `7b89aea`, P1–P3 results `b6485bf`.
- Per-figure numeric tables: `reports/goal_tracks/data/`, `reports/p123/data/`.

---

## Figures (planned)

- **Fig. 1** Protocol schematic: pools, splits, freezing, the two metrics.
- **Fig. 2** The position effect. (a) compound-position occupancy; (b) DMSO A−B cross-plate cosine distribution; (c) B_pos vs B0 by dose and stratum; (d) position vector reproducibility across drug-set-disjoint plate halves vs different positions.
- **Fig. 3** Model comparison at 5 µM in liver lines: ten published models, baselines, v0, trimmed position, v1 — with and without position matching. The two panels side by side are the paper's core argument.
- **Fig. 4** Concentration of the residual gain: per-group squared-error reduction waterfall; top-k exclusion curves for both metrics.
- **Fig. 5** HepaRG: dose-rank response, covered vs uncovered groups, replicate-ceiling attainment.
- **Supp.** 0.05/0.5 µM tables, 200-cell sensitivity, per-cell-line results, LINCS matching audit, job ledger and hashes.

## Open items before submission

1. **P4 (done, failed):** on the line-specific response target (row minus the well mean across cell lines) the position baseline nearly vanished (R² 3.39% → 0.63% in liver lines), but none of the three drug arms passed versus B0 or versus the best known VC at 5 µM (sibling copy vs best known VC: ΔRMSE −0.000506 [−0.002299, +0.000787], ΔR² +3.22 pp [−0.65, +6.54]). A section reporting this fair, position-free comparison should be added to Results; it supports message 3 and does not provide a positive liver-VC claim. See `reports/p4/RESULTS.md`.
1b. **R15 (started):** liver-cancer VC on LINCS HEPG2 (3,927 compounds; structure-only and cross-cell-line transfer scenarios), with exposure and plate-position audits first and external confirmation on the Tahoe liver-line benchmark. See `requests/CLAUDE_REQUEST_R15_LINCS_LIVER_VC_2026-11-21.md`.
2. One-shot confirmation on the reserved 40 plate-6↔14 pairs and/or compounds outside the 203-group panel — **requires explicit user approval to open**; plan written in `reports/p123/CONFIRMATION_PLAN.md`.
3. Literature scan for prior reports of plate-position/batch artefacts in large perturbation screens (not yet done in this session; needs a model without the current safeguard block, or Codex).
4. Decide author list, affiliations, and whether to cite arXiv 2611.01604 as concurrent independent work (recommended).
