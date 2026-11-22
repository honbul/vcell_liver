# V0 translation audit on adult TCGA-LIHC

V0 produced a LIMORE 72-hour Activity Area estimate for all **368** eligible primary tumors across **88** registered compounds. The native RNA distance rule flagged **0/368** as out of domain. In a separate fixed PROGENy projection calibrated on cell lines, **368/368** fell outside the reference envelope. The results show that the applicability estimate depends on the representation. The tumor cohort supplies genomic and clinical annotations without panel-matched treatment outcomes.

## Cohort and molecular annotation

The April 2022 public GDC STAR sample sheet contained 371 distinct primary-tumor participants. Deterministic sample-vial selection and an adult age documented in the PanCanAtlas Clinical Data Resource (CDR) retained 368. The RNA axis has 60,660 GENCODE v36 genes. GDC FPKM and TPM were retained in separate matrices. Clinical CDR survival time and event fields were available for all 368. Treatment or survival labels did not enter fit, forecast, selection, or association tests. [GDC](https://portal.gdc.cancer.gov/projects/TCGA-LIHC) · [PanCanAtlas CDR and ABSOLUTE](https://gdc.cancer.gov/about-data/publications/pancanatlas) · [RNA receipt](data/cohort_receipt.json)

| Annotation | Observed support |
|---|---:|
| Hoshida S1 / S2 / S3 / unassigned | 108 / 68 / 143 / 49 |
| Proliferative S1+S2 / non-proliferative S3 | 176 / 143 |
| ABSOLUTE purity, same sample vial | 351 |
| ESTIMATE immune and stromal scores | 368 |
| Open WES mutation profile support | 359 |
| TP53 / CTNNB1 / AXIN1 / ARID1A / TERT coding variants | 103 / 93 / 26 / 26 / 1 |
| GISTIC CNA support | 358 |
| FGF19 / CCND1 / MET / VEGFA high-level gains | 24 / 25 / 9 / 24 |
| CDKN2A deep loss | 20 |

The Hoshida Table 3S sets were obtained from MSigDB. Classification uses gene-wise centering and scaling over the unlabelled tumor cohort, then within-sample rank enrichment. Each signature is compared with 499 size-matched random sets. A best signature must pass a three-set correction and cohort FDR 0.05. Continuous Hallmark E2F and G2M scores describe proliferation separately. Official ESTIMATE stromal/immune signatures and 14 fixed PROGENy footprints provided the independent expression annotations. The TERT entry covers coding calls from masked WES. Promoter mutations require a separate noncoding assay. [Hoshida et al.](https://doi.org/10.1158/0008-5472.CAN-09-1089) · [ESTIMATE](https://doi.org/10.1038/ncomms3612) · [PROGENy](https://doi.org/10.1038/s41467-017-02391-6) · [CNA source](https://github.com/cBioPortal/datahub/blob/86690e1ed9752b1dcd50b5657f5f05eafa4b6b78/public/lihc_tcga_pan_can_atlas_2018/data_cna.txt) · [Annotation receipt](data/annotation_receipt.json)

## Domain shift and cell admixture

The V0 checkpoint was fit on 77 LIMORE samples. GDC RNA covered **958/1,000** V0 genes, and 18,256 genes were common to the complete LIMORE and TCGA protein-coding axes before the checkpoint restriction. Projection and UMAP parameters were fit on cell-line references. The comparator cell-line audit found 17,834 common genes across TCGA, LIMORE and DepMap 25Q3, with 21 adult HCC CCLE profiles and 12 adult HCC GDSC2 profiles. In the separate unwhitened rank-PCA audit, median nearest-LIMORE distances were 17.745 for TCGA, 5.412 for CCLE-HCC and 5.308 for GDSC2-HCC. These distances use different units from the whitened checkpoint diagnostic. [Projection and model freeze](data/freeze.json) · [PCA and UMAP](data/domain_shift.pdf)

| Input or diagnostic | Tumors inside the line reference | Median distance | Threshold |
|---|---:|---:|---:|
| V0 checkpoint, native log₂(FPKM+1) | 368/368 | 3.537 | 5.066 |
| V0 checkpoint, source-quantile ranks | 368/368 | 3.484 | 5.066 |
| Common-gene projection, native FPKM | 368/368 | 3.636 | 5.023 |
| Common-gene projection, TPM sensitivity | 368/368 | 3.418 | 5.023 |
| Common-gene projection, within-sample ranks | 368/368 | 3.589 | 4.987 |
| Fixed 14-pathway PROGENy projection | 0/368 | 10.294 | 4.184 |

| V0 input | Distance vs purity Spearman ρ | Distance vs immune ρ | Distance vs stroma ρ | Per-drug median absolute partial ρ, purity / immune / stroma |
|---|---:|---:|---:|---:|
| Native FPKM | +0.209 | −0.057 | −0.135 | 0.051 / 0.093 / 0.116 |
| Source-quantile ranks | −0.230 | +0.336 | +0.441 | 0.101 / 0.094 / 0.215 |

Partial rank correlations control for the other two admixture measures. The largest absolute adjusted stroma association rose from 0.465 with native FPKM to 0.563 under source-quantile mapping. The changes identify sensitivity to immune and stromal composition in bulk RNA. They cannot identify the unobserved pure-tumor counterfactual. [Purity figure](data/purity_distance.pdf) · [Per-drug adjustment](data/purity_confounding.tsv)

## Biological concordance

Each drug prediction was centered by the same tumor's mean across the registered panel. Copy-number contrasts were assessed unadjusted and in HC3 regression with ABSOLUTE purity and ESTIMATE immune and stroma. The adjusted native FPKM results all had 99% intervals spanning zero.

| Copy-number contrast | Altered / known | Drugs | Adjusted Activity Area difference [99% CI] | FDR |
|---|---:|---:|---:|---:|
| FGF19 gain and FGFR inhibitor panel | 24/358 | 6 | −0.00264 [−0.00567, +0.00039] | 0.096 |
| CCND1 gain and CDK4/6 inhibitor | 25/358 | 1 | −0.00415 [−0.00932, +0.00102] | 0.096 |
| MET gain and MET inhibitors | 9/358 | 4 | +0.00209 [−0.00404, +0.00822] | 0.632 |
| VEGFA gain and VEGFR inhibitors | 24/358 | 8 | +0.00013 [−0.00266, +0.00291] | 0.908 |
| CDKN2A loss and CDK4/6 inhibitor | 20/358 | 1 | +0.00144 [−0.00412, +0.00699] | 0.632 |

The FGF19 point estimate was opposite the expected direction. LIMORE reported FGF19/FGFR associations, and the fisogatinib study used selective FGFR4 inhibition with FGF19 protein expression as the biomarker. The V0 registry has six broad or multitargeted FGFR agents and omits a selective FGFR4 inhibitor. Its GISTIC-amplification contrast supplies a limited test of that mechanism. [LIMORE](https://doi.org/10.1016/j.ccell.2019.07.001) · [Fisogatinib Phase I](https://pubmed.ncbi.nlm.nih.gov/31575541/) · [Biological contrasts](data/biologic_concordance.pdf) · [Copy-number score table](data/panels.tsv)

| Adjusted per-drug contrast | V0 native, FDR <0.05 | V0 rank bridge, FDR <0.05 | V0 additive ridge, native |
|---|---:|---:|---:|
| CTNNB1-only vs TP53-only | 51/88 | 61/88 | 0/88 |
| Hoshida S1/S2 vs S3 | 67/88 | 72/88 | 0/88 |

The differences stratify the model's own scores. Across 88 drugs the CTNNB1-only versus TP53-only effect correlated **−0.381** with the measured LIMORE contrast in the native mode and **−0.436** with rank input. Sign agreement was 25.0% and 28.4%. LIMORE had one CTNNB1-only and 46 TP53-only cultures among 76 linked profiles, so this descriptive contrast relies on a single CTNNB1 culture. Higher discovery counts than additive ridge therefore indicate personalization differences inside the model, with limited external support for their direction. [Subtype scores](data/subtype_strata.tsv) · [Driver comparison](data/limore_driver_concordance.tsv)

The median within-tumor Spearman agreement between the V0 and the additive-ridge rankings was 0.906 across 368 tumors on the same 88 compounds. [Metrics receipt](data/metrics.json)

## Reproduction and scope

The study reads the SHA-256 verified model, cohort, and source expression from read-only paths. The registered CPU forecast was frozen before the separate annotation score. An end-to-end CLI replay accepts the checkpoint file or enclosing release directory, and writes outputs under the patient audit worktree namespace. The top-level V0+ adapter recognizes the documented schema-1 release, uses its own TPM-based preprocessing, and compares shared-drug ranks with a supplied V0 baseline. The completed V0+ package was unavailable for this audit.

```bash
python -m hcc_patient_eval.run --model /path/to/v0/checkpoint.joblib --baseline /path/to/v0_baseline --out /path/to/pat-eval-new
```

The current implementation requires the research runtime and input data enumerated in [the protocol](PROTOCOL.md). This analysis does not establish therapeutic response or patient-level clinical utility. [Run and score evidence](data/score_receipt.json) · [External-cohort access audit](EXTERNAL_COHORTS.md)
