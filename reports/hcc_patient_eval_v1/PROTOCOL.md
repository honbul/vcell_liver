# TCGA-LIHC patient audit inputs and execution

All patient and training matrices are read from their registered source paths. The analysis writes to a separate worktree and G1 namespace. Slurm names begin with `pat-eval-` and request CPU only. The results folder has aggregate statistics and figures.

| Input | Registered resource |
|---|---|
| GDC LIHC STAR FPKM and TPM | `/mnt/S1/sdata/pubdata/TCGA/TCGA_LIHC` and `gdc_sample_sheet.2022-04-18.tsv` |
| Adult age, clinical stage and outcome metadata | `/mnt/S1/sdata/pubdata/TCGA/PanCanAtlas_CDR/TCGA-CDR-SupplementalTableS1.xlsx` |
| Public masked somatic variant calls | `/mnt/S1/sdata/pubdata/TCGA/TCGA_LIHC_masked_maf_GDC/maf`, GDC May 2022 call file metadata |
| GISTIC discrete CNA | cBioPortal LIHC PanCancer Atlas 2018, `data_cna.txt`, repository file revision `86690e1ed9752b1dcd50b5657f5f05eafa4b6b78` |
| ABSOLUTE purity | GDC public file `4f277128-f793-4354-a13d-30cc7fe9f6b5` |
| PROGENy | `saezlab/progeny:data/model_human_full.rda` revision `c10e270121fb01b5cec6c078013f8991d66efba5`, exported to TSV |
| Fixed signatures | MSigDB Hoshida S1/S2/S3 and Hallmark E2F and G2M, plus the original ESTIMATE R package immune and stromal scores |
| Adult-HCC reference RNA, compounds and fitted weights | The released V0 and additive ridge in G1 `vc_build_v1/function/function-registered.ekrmqfpZ/full`, with adult LIMORE RNA and registry in `vc_build_v1/function/limore-v1` |
| Additional cell-line reference RNA | The read-only `vc_build_v1/function/cross-v1/train` DepMap-derived matrix, with DepMap 25Q3 `Model.csv` adult HCC roster and GDSC2 source records |
| LIMORE genotype comparison | `/mnt/S1/sdata/pubdata/LIMORE/Cancer_functional_genes_of_81_cell_lines.txt` plus published Activity Area rows |

## Execution

1. `python -m hcc_patient_eval.cohort` prepares deterministic adult primary-tumor vials and a cohort receipt with source hashes.
2. `python -m hcc_patient_eval.annotations` creates the canonical patient annotation table and ESTIMATE input. `Rscript experiments/hcc_patient_eval/estimate.R` renders the ESTIMATE score file. The input, filtered file and score hashes are recorded in `estimate.sha256`.
3. `python -m hcc_patient_eval.run --stage freeze --model <package> --baseline <ridge-package> --out <new-directory>` runs prediction and source-fit domain projection. The score phase checks `--digest` before reading annotations.
4. `python -m hcc_patient_eval.run --stage score --model <package> --out <same-directory> --digest <freeze-sha256>` generates separate score tables, figures and an automatic Markdown report. The default `--stage all` performs the same stages sequentially.

The G1 V0 implementation uses Python 3.10 and the release's scikit-learn 1.6.1 and RDKit 2025.03.5. Analysis additionally uses statsmodels 0.14.5, UMAP 0.5.9.post2, matplotlib 3.10.8, Pandas, PyArrow, NumPy, SciPy and openpyxl. Set NumPy and BLAS thread counts according to the CPU allocation. When relocating inputs, pass `--cohort`, `--raw`, `--annotations`, `--limore`, `--limore-maf`, `--cross`, `--depmap` and `--upstream` as needed.

The V0+ adapter reads a schema-1 `release.json`, verifies the advertised checkpoint and compound registry, uses the V0+ fit bundle at `--cross`, and derives its source-distance threshold from the read-only fit expression and lineages. Pass the V0+ runtime via `--upstream` and an optional V0 release via `--baseline`. If two releases use different endpoint scales, the cross-release comparison is a native within-sample rank agreement on shared compounds. Additional releases can use `--adapter module:factory` to provide the expression contract, compound and gene axes, fit references and prediction interface.

Drug target tests quantify concordance of the model's own predictions. The choice of prognosis, molecular class or clinical stage never determines forecast fitting or release selection. Independent biology comparisons and expected target relationships, including the fit-exposed LIMORE reference, are interpreted at their measured supports.
