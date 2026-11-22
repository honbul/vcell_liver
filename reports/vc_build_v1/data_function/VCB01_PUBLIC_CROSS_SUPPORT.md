# VCB-01 public cross-source support

**50 independent adult HCC families and 68 exact compound parents are supported.** Freeze/score uses the registered query IDs and fits on public GDSC2, CTRPv2 and PRISM20Q2 response measurements. LIMORE has previous project exposure. The test families are new to the three training resources under the recorded identity audit.

| Scope | Observed rows | Cell models | Adults with HCC | Compounds |
|---|---:|---:|---:|---:|
| GDSC2 release 8.5 | 24,839 | 714 | 12 | 42 |
| CTRPv2 post-QC | 31,097 | 731 | 19 | 49 |
| PRISM Repurposing 20Q2 secondary | 9,749 | 717 | 18 | 18 |
| Three sources combined | 65,685 | 1,014 | — | 68 |
| LIMORE held-out adult HCC | 3,400 | 50 | 50 | 68 |

Only five compounds have fit observations in all three sources. Source-specific support therefore stays visible in the evaluation. The pooled fit catalogue counts 991 recorded cell/patient groups. Native AUC values within a source have lower-is-more-sensitive direction. Test LIMORE Activity Area has higher-is-more-sensitive direction. Absolute units and assay/dose windows differ. The RNA axis contains 17,871 shared unique coding gene symbols, with DepMap default bulk log2(TPM+1) fit RNA and GEO LIMORE deposited bulk log2(FPKM+1) query RNA. The rank-transfer analysis declares its preprocessing separately. [Sanger GDSC protocol](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) · [NCI CTRPv2](https://ccgdata.cancer.gov/ctd2-data-project/broad-institute-screening-dependencies-cancer-cell-lines) · [PRISM 20Q2](https://doi.org/10.6084/m9.figshare.20564034.v1) · [DepMap 25Q3](https://depmap.org/portal/data_page/)

## Lineage lock

All source cell rosters were audited, including GDSC1 for exposure despite its exclusion from fitting. Checks used Cellosaurus v57.0 names, synonyms, secondary accessions, hierarchies, same-individual relationships and cross-references plus DepMap ModelID, RRID, PatientID, COSMIC and Sanger IDs. Fit entries require an unambiguous DepMap mapping and a default RNA profile. Among 77 age-confirmed LIMORE cell models, 55 models across 54 families had no related entry in those source rosters. For the 50-family cohort the first model ID per family was ranked lexicographically and the first 50 families retained. This gave 49 CLC cultures and MHCC97H. CLC19 represents the donor shared by CLC19 and CLC20, and CLC20 stays out of the scored set. Four other clean family groups remained unused: SNU354, SNU368, SNU739 and Tong. The lineage list is outcome-free. An independent G1 replay reproduced the same manifest byte-for-byte. [Cellosaurus](https://www.cellosaurus.org/) · [Locked roster](../../../data_manifests/vc_build_v1/function/cross_holdout_roster.tsv)

## Chemical and endpoint matching

CTRP and PRISM use published source structures. GDSC source catalogue names were resolved to PubChem structures and matched by full standardized isomeric parent key. A documented selection picks a single measured source identity and dose window per parent using only coverage metadata and the provider's MTS010 preference. CTRP experiments require the released `SNP-matched-reference` tag. Duplicate experiment/cell identifiers in the metadata collapse before joining to preserve the original number of observations. PRISM rows require a passing STR indicator and carry the published fitted endpoint, including the provider's upstream processing. Matched query targets are stored apart from input rows. [GDSC chemistry audit](../../../data_manifests/vc_build_v1/function/gdsc_pug_match.tsv) · [Selected source assay windows](../../../data_manifests/vc_build_v1/function/cross_source_protocols.tsv) · [Actual data receipt](cross_receipt.json)

## Alternative public sources

| Dataset | Access and matching assessment |
|---|---|
| HRA004348 | The organoid paper links RNA sequencing to GSA-human and separates the 301-compound screen from the detailed 64-compound/21-organoid curve experiments. The accession endpoint was unavailable during the access audit. No matched public RNA-to-response bundle was verified. |
| LICOB OEP003191 | The paper points to NODE for WES/RNA/protein matrices. The public project URL returned a JavaScript shell. An anonymously downloadable matched RNA-to-response bundle was not verified. The paper separately requires an MTA for physical organoids. |
| Caruso LCCL | Public basal RNA and native AUC matrices are available. The identity/age audit supports 28 adult HCC cell models in 26 families, 24 shared with LIMORE and 2 absent from LIMORE. Fifteen drugs share the original manufacturer catalogue code. Tanespimycin and decitabine add two CAS-verified isomeric products. The dataset is a candidate for assay shift with explicit shared-family reporting and low novel-family support. The source AUC should be evaluated in its own units. |

[HRA004348 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897507/) · [GSA accession](https://ngdc.cncb.ac.cn/gsa-human/browse/HRA004348) · [LICOB paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949980/) · [NODE OEP003191](https://www.biosino.org/node/project/detail/OEP003191) · [LCCL public catalogue and matrices](https://lccl.zucmanlab.com/download.html) · [Tanespimycin identity](https://www.sigmaaldrich.com/US/en/product/sigma/a8476) · [Decitabine identity](https://www.apexbio.cn/decitabine-nsc127716-5aza-cdr.html)

## Rebuild

After producing the LIMORE bundle, run the following under the same Python stack from the worktree root. The G1 `cross-v1` receipt records source hashes, protocol support and expected outputs. Paths already used by a signed run are immutable.

```sh
PYTHONPATH=src python -m vc_build.prepare --config configs/vc_build_v1/contract.yaml \
  --bundle cross --output /path/to/new/cross-output
```

The G1 bundle has `train/`, `test_inputs/`, `test_labels/`, `audit/` and `receipt.json`. The prepared test inputs contain no response column. A separate metadata-only check verified all artifact hashes, 3,400 distinct one-to-one scoring IDs, 50 unique query groups, disjoint fit/query IDs and the same RNA column schema. No held-out expression from R18 was used.
