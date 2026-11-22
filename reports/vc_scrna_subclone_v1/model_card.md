# Donor-held-out HCC single-cell resistance

The model estimates relative in vitro resistance for sorafenib, lenvatinib, regorafenib, cabozantinib and trametinib. Fit-only empirical distributions align GDSC2, CTRPv2, PRISM and LIMORE endpoints by source and drug. GSE182593 contributes three pairs of training organoids per fit. The fourth donor's phenotype is read during separate scoring. [GSE182593](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182593)

- LIMORE scores use source-held-out parent lineages and 1,000- or 4,000-read downsampling to produce pseudobulk and pseudo-single-cell queries. Tahoe SNU-423 vehicle cells provide measured single-cell expression for one exact LIMORE lineage.
- GSE149614 uses RNA-CNV deviation against independent normal hepatocyte donors and an epithelial marker check. GSE151530 and GSE125449 use the authors' malignant annotations. The 19 shared NCI donors use the same identifiers for all fold assignments. The companion cohort does not contribute to adaptation.
- Patient scores test observational concordance across paired primary, PVTT, lymph-node and NCI timepoint samples. The NCI interval represents checkpoint inhibitor therapy. These patients have no measured single-cell sorafenib response. [GSE149614](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149614) · [NCI atlas](https://pmc.ncbi.nlm.nih.gov/articles/PMC8604764/) · [NCI biopsy-ID map](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE229nnn/GSE229772/suppl/GSE229772_scRNA_id_name_match.txt.gz)

Bulk-only and organoid-augmented Ridge, ElasticNet and LightGBM are compared with ABCB1, stemness, EMT and hypoxia markers. The SCAD-family model uses adversarial alignment with gradient reversal. The scDEAL-family model uses separate denoising autoencoders, multiple RBF-kernel MMD alignment, and latent state consistency. Both implementations use the same continuous outcome. scGPT uses the frozen Track 6 encoder on a common expression axis, comparing supervised fitting with both adaptation losses. Exact pretraining membership of either public atlas is unverified. [SCAD](https://pmc.ncbi.nlm.nih.gov/articles/PMC10104628/) · [scDEAL](https://www.nature.com/articles/s41467-022-34277-7)

High resistance uses source-only score quantiles. The scorer tests rank partial correlations within donor, adjusting for biopsy, UMI count, detection, mitochondrial fraction and S/G2M cell cycle measures. UMAP and grouped transcriptional states are descriptive and rendered after scoring.

Run `python -m vc_scrna_subclone.predict --help` and `python -m vc_scrna_subclone.score --help` offline. The G1 batch scripts list the raw counts, fixed checkpoints and registered artifacts. Scoring reads clinical metadata and outcomes after prediction hashes have been written.

The CPU figure stage uses UMAP-learn 0.5.3 and PyNNDescent 0.5.13 from the shared study runtime. The UMAP panels use separate within-donor color scales because they show forecasts from different held-donor folds.

## Observed transfer

Across 74 held cell lineages and five drugs, the scGPT adversarial model reached within-drug Spearman 0.0836 for 4,000-read pseudobulk. Bulk-trained LightGBM reached 0.1572. The same scGPT model predicted higher resistance in three of four held organoid pairs, with mean difference +0.0281, 95% donor bootstrap interval [−0.0144, 0.0815], and exact sign-flip p of 0.50. In GSE151530, its donor-mean partial correlations were −0.1105 for UMI count, +0.2193 for detected genes, and +0.1000 for the G2M score. The primary-to-PVTT and primary-to-lymph comparisons each retained one paired donor, and the NCI temporal comparison retained three. The observed data support treating the expression-state figures as descriptive. [Complete held-donor scorecard](RESULTS.md)
