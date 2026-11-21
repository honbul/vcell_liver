# Time-matched measurements improve prediction of hepatic chemical responses within L1000

Development manuscript. Intended scope and outstanding evidence are recorded in the [claims audit](CLAIMS.md).

## Abstract

Predictive models of chemical perturbation can combine molecular structure with measured responses from related experimental settings. We tested which inputs improved expression prediction after separating vehicle wells used for training, validation and scoring. In HEPG2 L1000 data, a low-rank model transferred measured 6-hour responses to independent 24-hour wells for 132 held-out compounds and 247 exact-concentration conditions. Parent-macro RMSE fell by 7.33% relative to a source-fitted scalar model (99% interval 5.06–9.58%). Replacing the measured input with a different compound at the same concentration attenuated this effect. On a separate 493-compound development comparison, a structure-kernel and FiLM blend reduced RMSE by 2.05% relative to a matched nonhepatic donor average. Alternative output structures and training losses did not reach the prespecified threshold for practical improvement. Across exact-concentration TempO-Seq comparisons in HepaRG, mean cosine was near zero in samples of three and five compounds. The results support time transfer when a matched measurement is available in the same assay. Performance in independent adult hepatocellular carcinoma remains unresolved.

## Introduction

Transcriptomic measurements can connect gene-expression changes across drugs and biological contexts. Predicting a response before the experiment requires a precise specification of measurements available at inference. A molecular structure, a measured response in another cell line, and a measurement at an earlier time give access to different information. Their performance can also depend on whether vehicle variation is shared between model fitting and evaluation. We distinguish these input conditions and test time, cell context and architecture changes under independent vehicle roles. [L1000 resource study](https://doi.org/10.1016/j.cell.2017.10.049)

## Results

### Control independence changed the model comparison

With the physical vehicle sample counts held equal between protocols, separating the fitting and selection roles reduced the selected residual gain in each of the six nonzero-gain comparator families. The relative ranking of the models depended on whether selection used an independent vehicle group. Across the legacy model families, joint selection of the training checkpoint and residual gain also affected the matched score. These comparisons motivated the common fitting, selection and scoring roles used below. [Role and selection comparisons](../D12_RESULTS.md)

### A paired temporal model improved within-assay prediction

Of the exact-concentration matched HEPG2 conditions, 247 across 132 chemical parents had independent RNA plates at both 6 and 24 hours and at least two RNA plates at each time. We assigned one 24-hour scoring plate to each condition and excluded the entire plate from fitting, selection and direct 24-hour donor aggregation in that comparison. On 268 physical scoring wells, the temporal model lowered parent-macro RMSE from 0.97896 for the scalar model to 0.90722 and increased pooled R² from 11.64% to 25.96%. The paired R² change was 14.32 percentage points (99% interval 10.42–18.06). The comparison with a direct measurement on a different 24-hour plate included zero in both intervals. [Temporal analysis](../T4_RESULTS.md)

Two subsequent controls used the original fitted settings and reproduced the frozen predictions for the exact 6-hour donor. For all 132 compounds, RMSE was 7.25% lower than predicting the fitted 24-hour population mean (99% interval 5.26–9.28%). Where a different held-out compound was available at the same dose, the exact compound input lowered RMSE by 16.11% relative to the input substitution (89 parents, 99% interval 12.48–19.92%). These controls quantify the contribution of measured compound-specific response in the same assay. [Input attribution](../T4_ATTRIBUTION_RESULTS.md)

In a post hoc influence analysis, the temporal model reduced RMSE relative to the scalar comparator in 95 of 132 parents. Removing the 20 largest positive parent-level RMSE contributions left a 3.95% relative RMSE gain. Independently reweighting parents and physical plates gave relative RMSE gains of 5.05–9.75% across the central 99% of weighted comparisons. The different-plate 24-hour donor comparison again included zero. [Parent and plate sensitivity](../ROBUSTNESS_RESULTS.md)

### Additional model changes produced smaller effects

In the matched 10 µM, 6-hour benchmark with 493 parents and 2,324 HEPG2 wells, the full structure-kernel and FiLM reference reduced RMSE from 0.95987 for the donor anchor to 0.94022. The paired absolute RMSE difference was −0.01965 (99% interval −0.02245 to −0.01687), and pooled R² increased by 3.27 percentage points (99% interval 2.71–3.83). A separate fit-only mean residual did not close this gap. Each of six output or loss changes had an upper 99% bound below 0.5% for relative RMSE gain. The low-rank output used fewer dense multiply-accumulate operations and had an RMSE gain of 0.026%. With donor and kernel held fixed, shuffling structure input in the neural branch had similar prediction scores. This control limits claims about the amount of additional chemical identity learned by that branch. [Donor-matched ablations](../SL_B_RESULTS.md)

On the separate HEPG2 development set where no query drug expression measurement was supplied, Cell Painting and nonhepatic viability features gave a small kernel gain. On the 345 parents with both views, the relative RMSE improvement against the frozen R17 reference was 0.200%. On all 1,862 parents with a missing-view fallback, it was 0.046%. Adding the same measured features to the FiLM input did not improve both metrics relative to the shared kernel or its mask and parent-permutation controls. [Phenotypic view ablation](../I5_RESULTS.md)

### Directional transfer was weak across the small matched HepaRG sets

Exact concentration and parent matching yielded five compounds and 21 physical wells in GSE284321 and three compounds and nine wells in GSE296254. On their corresponding independent L1000 observations, the 6-to-24-hour prediction had parent-macro cosine of 0.247 and 0.220. On the identical 827-gene axis in the respective HepaRG datasets, cosine was 0.024 and −0.017. Direct 24-hour L1000 donors also had near-zero mean cosine in HepaRG. The two comparisons jointly change assay and cell context. Their small supports limit population inference. [Matched cross-assay comparison](../T4_CROSS_ASSAY_RESULTS.md) · [Figure 1](figure1.pdf)

## Methods

### Physical units and information available at inference

LINCS samples were indexed by release, cell type, canonical compound parent and RNA plate–well coordinate. We required measured concentrations and validated exact dose–time pairs against original instance and signature-quality metadata for the temporal comparison. The model using structure and the model supplied with actual same-compound expression were evaluated separately. In the matched nonhepatic transfer comparison, each donor cell line contributed its mean fit-vehicle contrast, and the four cell lines contributed equally. For validation and scoring, independent vehicle wells formed the reference contrast. Physical wells were kept together across reuse of the same well and rotations of the three vehicle roles. Data reserved for the separately registered confirmation study were excluded from development inputs. [Temporal protocol](../T4_PROTOCOL.md) · [Matched donor protocol](../SL_B_PROTOCOL.md)

### Temporal model and comparators

The 6-hour vector was centered with the source mean and projected onto principal directions learned only from fitting conditions. A ridge map estimated the 24-hour vector from those coordinates, and the fitting mean was added to the output. Each parent had equal total fitting weight. Rank (4, 16 or 32) and ridge penalty (0.1 or 1) were selected from internally held parents and disjoint vehicle means. The early identity vector, a source-fitted nonnegative scalar and a different-plate 24-hour donor provided comparators. Exact-condition controls retained the same trained operator. For transfer to HepaRG we used the registered one-to-one Entrez–Ensembl panel, preserved the frozen L1000 predictions, and computed cosine against independent plate vehicle subtractions in TempO-Seq. [Time model and scoring contract](../T4_PROTOCOL.md) · [Cross-assay protocol](../T4_HEPARG_PROTOCOL.md)

### Estimands and uncertainty

Physical-well root mean square errors were averaged within parent, then across parents. Pooled R² used the total squared error relative to a zero treatment contrast. Paired intervals were calculated from 10,000 resamples of whole chemical parents and their associated wells and vehicle rotations. Primary comparisons across four development axes used one Holm family of directional paired randomization tests retaining the registered A and B slots. The analysis of the small exact HepaRG cohorts used parent-level cosine and observed ranges. [Primary-test registry and adjustment](../data/dev_family/primary.tsv)

## Discussion

The largest observed gain depends on the availability of a matching early measurement. Separating the early input from the evaluation controls and holding out both parent and target plate made it possible to test a temporal model without adapting to the evaluation plate. Sensitivity to replacing the early response with a dose-matched response of the wrong compound supports the value of the measured input. Under the same-information architecture comparison, structural and objective modifications made smaller changes, including cases in which efficiency improved without practically meaningful reductions in error.

The existing cross-platform comparisons are narrow. Exact chemical and concentration matching substantially reduces support. Differences between HEPG2 and HepaRG and between expression assays occur together. The observations cannot isolate a platform effect from a cell-context effect. HEPG2 is hepatoblastoma-derived, so even the same-assay result does not establish drug-response accuracy in adult hepatocellular carcinoma. A separate evaluation needs independent tumour-cell contexts, randomized physical positions, matched time and exposure, independent vehicle roles and the best available comparator under the same inputs. [HEPG2 lineage evidence](https://doi.org/10.1016/j.humpath.2009.07.003)

## Data availability

Source expression data are published under GEO accessions [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742), [GSE284321](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE284321) and [GSE296254](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE296254). The [project evidence index](../RESULTS.md) links to the retained score summaries, support, and checksums. A publication repository and approved release scope require author confirmation.

## Code availability

Implementation, registered comparison settings, and score replay tools are versioned in this worktree. The final article must name a release that readers can access and state the conditions of access. [Scientific Reports editorial policies](https://www.nature.com/srep/journal-policies/editorial-policies)

## Figure 1

**Time transfer and cross-assay direction.** (a) Relative reduction in parent-macro RMSE by the low-rank temporal model against each specified comparator. Lines show 99% paired parent bootstrap intervals. The two lower bars use post hoc registered controls and the indicated support. (b) Frozen predictions across the LINCS and HepaRG score wells sharing canonical parent and measured concentration. Points are descriptive parent-macro cosine, with three or five parents per comparison.
