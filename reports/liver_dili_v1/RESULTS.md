# Liver pathway and clinical DILI reference-label evaluation

## Data and identity audit

FDA DILIrank 2.0 contains **1,336** entries, including 217 Most, 351 Less, 414 No and 354 Ambiguous. The supplied workbook also includes the historical v1 sheet and marks 300 new entries. Exact-name PubChem lookup and RDKit parent standardization resolved **1,217** entries to a unique stereochemical parent. The binary evaluation includes **887** eligible resolved FDA entries (535 positive, 352 negative). Six exact stereochemical duplicates remain grouped. Stereo observations retain their original distinct IDs and labels. The historical split scored 155 resolved and determinate new entries (47 positive, 108 negative) using only historical v1 concern labels for fitting. [FDA workbook](https://www.fda.gov/media/113052/download) · [Split and denominator receipt](data/split_receipt.json)

| Public modality | Matched assay rows | Exact stereochemical structures in assay | FDA 2.0 exact structure overlap | FDA 2.0 connectivity overlap |
|---|---:|---:|---:|---:|
| Open TG-GATEs PHH, 24 h | 357 | 144 | 101 | 104 |
| GSE284321 HepaRG, 24 h | 29,251 | 1,128 | 66 | 105 |
| GSE296254 HepaRG, 24 h | 7,681 | 291 | 65 | 121 |
| cpg0004 A549 Cell Painting, 48 h and dose code 6 | 1,900 publisher profiles with stereo mapping | 1,653 | 430 | 559 |

The overlaps use the 1,217 structure-resolved FDA entries, including ambiguous labels for coverage auditing. For scoring, the split receipt excludes all such labels. The two HepaRG assays share 66 exact structures. PHH shares 31 with GSE284321 and 16 with GSE296254 before dose matching. The common expression axis contains 13,663 gene symbols. The HepaRG delta arrays are the existing plate-control R14 reductions, with prior development exposure in both datasets. Affymetrix PHH preparation produced 383 independent compound/experiment/dose contrasts on 153 named assay chemicals. Of these, 286 have DNA and 383 have LDH viability observations. The public attribute archive contains 2,606 human cell-viability rows across 158 named chemicals. RMA is local to each chemical experiment and no classifier uses viability or clinical exposure fields. The RMA workers used a track-local nonthreaded preprocessCore build after the default library's pthread failures. [PHH data and terms](https://dbarchive.biosciencedbc.jp/data/open-tggates/LATEST/README_e.html) · [GSE284321](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE284321) · [GSE296254](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE296254) · [Cell Painting release](https://github.com/broadinstitute/lincs-cell-painting/tree/061870127481dcd73c29df85ebcfddeac2ed0828) · [Assay receipt](data/assay_receipt.json)

Independent fixed definitions use WikiPathways, Hallmark and PROGENy. TG-GATEs PHH is the source of 398 TXG-MAPr modules, so those modules would have prior PHH assay exposure. Those scores were not used in the independent fixed comparisons. Learned modules were clustered inside each training fold. [TXG-MAPr source](https://doi.org/10.1007/s00204-021-03141-w)

DILImap's public WikiPathways matrices were accessible as 1,041 training observations × 472 pathways on 249 compounds and 210 validation observations × 469 pathways on 51 compounds. Their metadata contain FDA identifiers overlapping 241 training and 40 validation compounds and no release-increment IDs. The project publishes custom noncommercial terms, and full training gene-level matrices are access-restricted through the package API. Only metadata and access were audited here. [DILImap public data](https://github.com/Cellarity/DILImap/blob/00d7e4ea232d18109c27cbbec131f09f4ed6245f/docs/datasets.rst) · [DILImap terms](https://github.com/Cellarity/DILImap/blob/00d7e4ea232d18109c27cbbec131f09f4ed6245f/LICENSE) · [GEO GSE308567](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308567)

The EPA Tox21/ToxCast invitrodb v4.3 public assay metadata include three BSEP/ABCB11 endpoint names, 58 CYP endpoint names and 17 mitochondrial endpoint names by keyword in the endpoint worksheet. Endpoint names are an access and coverage check. A compound-level screen feature would require DSSTox identity, quality and potency reconciliation. ChEMBL resolves BSEP to CHEMBL6020 and human CYP3A4 to CHEMBL340. Neither source entered the classifiers. [EPA assay resources](https://www.epa.gov/comptox-tools/exploring-toxcast-data) · [ChEMBL BSEP](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL6020) · [ChEMBL CYP3A4](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL340) · [Public source audit](data/source_audit.json)

The PHH registry contains 766 unique treated CEL barcodes and 306 unique control barcodes. None are shared across named chemicals. The HepaRG source-control audit detected shared R14 plates between the scaffold-only fit and query rosters. Primary hypothesis A fits therefore removed all training conditions on any held HepaRG plate. The purge retained 12–13 fit compounds per GSE284321 fold and 5–7 per GSE296254 fold. [Physical support and viability](data/source_audit.json)

## Hypothesis A. Bidirectional pathway transfer

At 24 hours, exact stereochemical identity and an in vitro concentration ratio within threefold over the 1–1,000 µM range gave 23 GSE284321–PHH pairs and 13 GSE296254–PHH pairs. Exactly matched concentrations account for 9 and 2 pairs. The three fixed pathway spaces use independently published definitions. Fit-only gene modules use only the outer fit pairs. Each reported prediction comes from a held-scaffold and held-HepaRG-plate fit. The ridge penalty was tuned inside the fit scaffolds. Controls include the fit target mean, the unmodified measured source vector, and same-fit label permutation. [Freeze](data/transfer_plate_freeze.json) · [Score receipt](data/transfer_plate_receipt.json)

| Cohort | Direction | Pairs | Genes | WikiPathways | Hallmark | PROGENy | Fit-only modules |
|---|---|---:|---:|---:|---:|---:|---:|
| GSE284321 | HepaRG → PHH | 23 | 0.070 / 52.1% | 0.177 / 56.5% | 0.112 / 53.8% | 0.129 / 53.4% | 0.115 / 53.3% |
| GSE284321 | PHH → HepaRG | 23 | 0.032 / 55.9% | 0.083 / 51.2% | 0.221 / 54.9% | 0.140 / 54.3% | 0.068 / 64.5% |
| GSE296254 | HepaRG → PHH | 13 | 0.096 / 53.1% | 0.434 / 68.4% | 0.013 / 49.0% | 0.143 / 58.2% | 0.219 / 58.0% |
| GSE296254 | PHH → HepaRG | 13 | 0.093 / 56.8% | -0.123 / 63.2% | -0.263 / 40.4% | -0.061 / 58.2% | 0.215 / 58.3% |

Each model cell shows mean parent cosine / mean sign agreement. The directional signal varied by cohort and direction. In GSE296254 HepaRG → PHH, the WikiPathways score had cosine 0.434 and sign agreement 68.4%. The cosine change versus the training target mean was −0.034 with paired 99% interval [−0.106, +0.027]. None of the 20 ridge-versus-fit-mean cosine comparisons had a 99% lower bound above zero. The observed pathway agreement therefore did not establish a learned, compound-specific cross-platform improvement. The scaffold-only companion is included as a sensitivity analysis. [All arms and scores](data/transfer_plate_metrics.tsv) · [10,000 paired parent draws](data/transfer_plate_paired.tsv) · [Scaffold-only comparison](data/transfer_exact_metrics.tsv)

## Hypothesis B. FDA reference-label prediction

The two scaffold schemes use five outer, four inner grouped folds. Butina uses a 0.60 cluster-center similarity cutoff and keeps connectivity groups indivisible. All schemes have zero train–test connectivity overlap. The release-increment fit has 702 historical entries and 444 positives. The held increment has 155 entries and 47 positives. This is an FDA label-release shift with contemporary assay and molecular-embedding features. [Per-fold support and similarities](data/split_support.tsv)

| Split | Test entries | Nearest-training Tanimoto min | 5th percentile | Median | 95th percentile | Max |
|---|---:|---:|---:|---:|---:|---:|
| Bemis–Murcko | 887 | 0.013 | 0.195 | 0.323 | 0.664 | 1.000 |
| Butina | 887 | 0.018 | 0.196 | 0.340 | 0.576 | 0.755 |
| FDA release increment | 155 | 0.033 | 0.205 | 0.269 | 0.667 | 0.867 |

Four Murcko query entries reached Tanimoto 1.000 because the ECFP bit vectors matched despite distinct connectivity IDs. None reached 1.000 in Butina or the release increment. Each forecast and its fit and query connectivity rosters were checked across all 11 folds.

Model selection, scaling, pathway cross-fitting, phenotype PCA and Platt calibration use fit data. Measured query RNA appears only in the measured-RNA analysis. For predicted pathway inputs, auxiliary fit rows that share an evaluation connectivity or scaffold are removed in every outer and nested fold. ChemBERTa is frozen. The multitask model uses a shared molecular representation and masked auxiliary pathway loss. The final classifier replay used LIBLINEAR after LBFGS convergence warnings, with the same fits and fold registry.

| Split | Features | Model | N | AUROC | AUPRC | Balanced accuracy | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Murcko | ECFP + properties | Logistic | 887 | 0.698 | 0.751 | 0.661 | 0.215 | 0.059 |
| Murcko | ECFP + properties | LightGBM | 887 | 0.694 | 0.754 | 0.632 | 0.211 | 0.028 |
| Murcko | Structure + ChemBERTa | Logistic | 887 | 0.703 | 0.751 | 0.661 | 0.211 | 0.029 |
| Murcko | Structure + ChemBERTa | LightGBM | 887 | 0.726 | 0.768 | 0.675 | 0.202 | 0.038 |
| Murcko | Structure + ChemBERTa | Multitask MLP | 887 | 0.684 | 0.735 | 0.639 | 0.214 | 0.053 |
| Murcko | Structure + predicted pathways | Logistic | 887 | 0.704 | 0.750 | 0.657 | 0.211 | 0.033 |
| Murcko | Structure + predicted pathways | LightGBM | 887 | 0.719 | 0.760 | 0.661 | 0.205 | 0.055 |
| Murcko | Structure + Cell Painting | Logistic | 887 | 0.634 | 0.692 | 0.655 | 0.233 | 0.102 |
| Murcko | Structure + Cell Painting | LightGBM | 887 | 0.739 | 0.777 | 0.666 | 0.199 | 0.042 |
| Murcko | Measured RNA reference | Logistic | 140 | 0.516 | 0.864 | 0.652 | 0.141 | 0.059 |
| Murcko | Measured RNA reference | LightGBM | 140 | 0.374 | 0.792 | 0.486 | 0.147 | 0.090 |
| Butina | ECFP + properties | Logistic | 887 | 0.676 | 0.719 | 0.654 | 0.220 | 0.058 |
| Butina | ECFP + properties | LightGBM | 887 | 0.697 | 0.765 | 0.616 | 0.210 | 0.059 |
| Butina | Structure + ChemBERTa | Logistic | 887 | 0.692 | 0.734 | 0.650 | 0.215 | 0.042 |
| Butina | Structure + ChemBERTa | LightGBM | 887 | 0.727 | 0.792 | 0.671 | 0.204 | 0.042 |
| Butina | Structure + ChemBERTa | Multitask MLP | 887 | 0.683 | 0.736 | 0.641 | 0.217 | 0.044 |
| Butina | Structure + predicted pathways | Logistic | 887 | 0.689 | 0.731 | 0.647 | 0.215 | 0.036 |
| Butina | Structure + predicted pathways | LightGBM | 887 | 0.713 | 0.773 | 0.665 | 0.211 | 0.042 |
| Butina | Structure + Cell Painting | Logistic | 887 | 0.626 | 0.683 | 0.650 | 0.236 | 0.107 |
| Butina | Structure + Cell Painting | LightGBM | 887 | 0.742 | 0.800 | 0.661 | 0.201 | 0.056 |
| Butina | Measured RNA reference | Logistic | 140 | 0.641 | 0.914 | 0.642 | 0.137 | 0.048 |
| Butina | Measured RNA reference | LightGBM | 140 | 0.438 | 0.831 | 0.563 | 0.143 | 0.051 |
| Release increment | ECFP + properties | Logistic | 155 | 0.598 | 0.352 | 0.577 | 0.339 | 0.332 |
| Release increment | ECFP + properties | LightGBM | 155 | 0.627 | 0.476 | 0.555 | 0.328 | 0.335 |
| Release increment | Structure + ChemBERTa | Logistic | 155 | 0.612 | 0.357 | 0.644 | 0.327 | 0.317 |
| Release increment | Structure + ChemBERTa | LightGBM | 155 | 0.677 | 0.491 | 0.611 | 0.326 | 0.361 |
| Release increment | Structure + ChemBERTa | Multitask MLP | 155 | 0.571 | 0.334 | 0.586 | 0.325 | 0.315 |
| Release increment | Structure + predicted pathways | Logistic | 155 | 0.614 | 0.358 | 0.645 | 0.327 | 0.319 |
| Release increment | Structure + predicted pathways | LightGBM | 155 | 0.655 | 0.456 | 0.588 | 0.326 | 0.342 |
| Release increment | Structure + Cell Painting | Logistic | 155 | 0.503 | 0.282 | 0.645 | 0.315 | 0.247 |
| Release increment | Structure + Cell Painting | LightGBM | 155 | 0.683 | 0.478 | 0.573 | 0.331 | 0.372 |
| Release increment | Measured RNA reference | Logistic | 5 | 0.167 | 0.533 | 0.500 | 0.323 | 0.419 |
| Release increment | Measured RNA reference | LightGBM | 5 | 0.000 | 0.478 | 0.167 | 0.353 | 0.448 |

Cell Painting uses the exact structure-only output for all missing profiles. The audit matched the raw score, probability, decision and interval for all 2,428 fallbacks across both estimators and the 11 folds. The measured-RNA reference covers 140 entries in the scaffold splits and five in the release increment. Its positives are 117/140 and 3/5 respectively. Comparisons on the same measured support are provided for both arms. [Full classification and overlap table](data/metrics.tsv) · [Freeze inventory](data/classification_receipt.json)

### Paired modality ablation

| Split | LightGBM contrast | N | AUPRC change [99% CI] |
|---|---|---:|---:|
| Murcko | Predicted pathways vs structure | 887 | -0.009 [-0.025, +0.007] |
| Murcko | Predicted pathways vs shuffled pathways | 887 | -0.005 [-0.025, +0.016] |
| Murcko | Cell Painting vs structure, with fallback | 887 | +0.009 [-0.017, +0.032] |
| Murcko | Cell Painting vs availability mask, matched | 333 | +0.002 [-0.038, +0.036] |
| Murcko | Cell Painting vs shuffled profiles, matched | 333 | -0.009 [-0.049, +0.037] |
| Butina | Predicted pathways vs structure | 887 | -0.020 [-0.040, +0.002] |
| Butina | Predicted pathways vs shuffled pathways | 887 | -0.007 [-0.030, +0.017] |
| Butina | Cell Painting vs structure, with fallback | 887 | +0.008 [-0.015, +0.030] |
| Butina | Cell Painting vs availability mask, matched | 333 | +0.002 [-0.037, +0.035] |
| Butina | Cell Painting vs shuffled profiles, matched | 333 | +0.003 [-0.027, +0.035] |
| Release increment | Predicted pathways vs structure | 155 | -0.035 [-0.100, +0.045] |
| Release increment | Predicted pathways vs shuffled pathways | 155 | -0.024 [-0.118, +0.048] |
| Release increment | Cell Painting vs structure, with fallback | 155 | -0.013 [-0.074, +0.058] |
| Release increment | Cell Painting vs availability mask, matched | 49 | -0.024 [-0.097, +0.042] |
| Release increment | Cell Painting vs shuffled profiles, matched | 49 | -0.024 [-0.096, +0.068] |

The structure-plus-ChemBERTa LightGBM had higher AUPRC than its predicted-pathway counterpart in all three splits. Adding Cell Painting gave small whole-cohort point gains in the scaffold splits. The phenotype-value contrasts against matches with a mask or shuffled profiles included zero. Neither pathway input nor Cell Painting showed a reliable incremental gain across the strict holdouts. Release ECE was 0.361 for structure LightGBM and 0.342 with predicted pathways, consistent with substantial calibration error under the prevalence shift. [Connectivity-paired effects, 2,000 draws](data/modality_paired.tsv)

## Offline package verification

The checkpoint and local encoder were verified by SHA-256. The locked CLI produced 62 pathway scores, a calibrated probability and interval, and separate applicability flags. The non-fit example returned exactly the same JSON with the default release and explicit model path. The low-neighbor example triggered the OOD branch at Tanimoto 0.100 against a threshold of 0.195. An invalid input returned a nonzero exit. [CLI checksums](data/cli.sha256) · [Default CLI check](data/cli_default.sha256) · [Model card](model_card.md)

All recorded model, assay and scoring jobs used CPU in the dedicated G1 namespace. The data files, checkpoint and detailed forecasts reside on G1. [Slurm ledger](data/jobs.tsv)
