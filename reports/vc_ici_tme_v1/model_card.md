# Track 7 immune checkpoint blockade response model

The benchmark estimates objective response from pretreatment tumor RNA. Its primary external endpoint is RECIST CR/PR versus SD/PD in 17 anti-PD-1–treated GSE202069 HCC participants. Each person contributes one baseline profile. The held-out GSE140901 assay has 24 treated HCC participants and evaluates a separately trained NanoString-panel projection. GSE141198 contributes 148 exploratory HCC expression profiles without a registered objective response endpoint. All models are for retrospective research.

| Fit cohort | Baseline patients | Responders | Endpoint |
| --- | ---: | ---: | --- |
| IMvigor210 | 298 | 68 | Best confirmed overall response |
| Gide | 73 | 40 | Baseline anti-PD-1 or anti-PD-1 plus anti-CTLA-4 |
| Hugo | 26 | 14 | Baseline anti-PD-1 melanoma |
| Riaz | 49 | 10 | Baseline anti-PD-1 melanoma |
| Total | 446 | 132 | Cohort-local response coding |

The fitting stage reads non-HCC outcome labels and label-free HCC assay keys and expression. Cohort omission determines inner validation. Hyperparameters maximize unweighted mean source cohort AUROC. The final fit weights patients by inverse square root cohort size. Out-of-cohort source predictions determine monotone probability calibration and the balanced-accuracy threshold. HCC outcomes enter the scorer after the prediction bundle has a SHA-256 receipt. Metrics include stratified patient-bootstrap intervals and paired differences on the identical external patients.

The public HCC single-cell prior uses annotated primary tumor cells from GSE149614 and HCC tumors from GSE151530. Each donor contributes an equally weighted expression profile within a population. The seven node types are malignant, CD8 T, regulatory T, TAM, CAF, endothelial and other. Deconvolution uses weighted NNLS with reference-only landmark genes and a fixed reference abundance distribution. Fractions are RNA mixture estimates. Per-type Hallmark activities allocate bulk deviations with reference profiles, so individual cell states are estimated. LIANA consensus supplies directed human ligand–receptor complexes. At a complex, the least expressed subunit determines its abundance. Directional edge features combine inferred sender and receiver abundance. The NanoString panel fixes the available-gene axis and rebuilds landmarks from the HCC reference on that axis. Source measurements and labels remain disjoint from external outcomes.

Graph attention has two directed message-passing layers, typed node functions, and relation channels. Ablations remove inter-node messages, infer under a uniform mixture, or remove Hallmark pathway scores. Ridge, ElasticNet and LightGBM use within-profile ranks of genes selected by fit-only variance. A pathway and deconvolution Ridge shares the exact fitted graph input. Biomarkers include CD274 RNA, IFNG RNA, the IFN-γ six-gene score, the published weighted 18-gene GEP, and NNLS purity. The author release of TIDEpy uses fit-only per-gene centering and cancer class Other. Its public model and gene-map hashes are recorded in the frozen prediction bundle. The NanoString endpoint uses only biomarker signatures for which every required gene is measured. Gide lacks CCL5, so its patients do not calibrate the published 18-gene score. RNA-seq, rlog and NanoString units vary across cohorts. The published GEP normalization has been adapted using assayed housekeeping genes and a fixed log-base conversion. Original assay cutoffs do not transfer.

The graph simulation attenuates VEGFA–KDR, TGFB1, SPP1–CD44, WNT, TIGIT, HAVCR2/TIM3 and LAG3 edge channels by 50% and 90% where the registered gene axis supports them. WNT also attenuates the malignant-node WNT pathway. The scorer summarizes the frozen perturbation outputs in all HCC participants, those with low predicted response and measured nonresponders. Changes are model sensitivity estimates under a hypothetical feature intervention. The small primary HCC cohort limits precision. Bulk projection and ligand–receptor priors leave spatial contact, cell state and protein abundance unmeasured.

## Offline execution on G1

Use the pinned runtime described in the preflight ledger. `python -m vc_ici_tme.reference` builds the reference. `python -m vc_ici_tme.predict --help` describes the fitting and freezing interface. `python -m vc_ici_tme.score --help` describes independent scoring from the frozen directory. `experiments/vc_ici_tme_v1/predict_ici.slurm` runs the three registered targets with per-target receipts. All data and weight paths are local to G1.

## Sources

- GSE202069, GSE140901, GSE141198, GSE149614 and GSE151530 GEO records. [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/)
- Ayers et al., fixed 6-gene and 18-gene schedules and their disclosed coefficients. [JCI article](https://doi.org/10.1172/JCI91190), [public author patent Table 3B](https://patents.google.com/patent/AU2020204557A1/en)
- Jiang et al., Tumor Immune Dysfunction and Exclusion. [Nature Medicine](https://doi.org/10.1038/s41591-018-0136-1), [TIDEpy author release](https://github.com/jingxinfu/TIDEpy)
- Dimitrov et al., human ligand–receptor consensus. [LIANA author repository](https://github.com/saezlab/liana)
- Finn et al., [IMbrave150](https://doi.org/10.1056/NEJMoa1915745) assesses the combination of atezolizumab and bevacizumab in unresectable HCC. Its treatment evidence is contextual and outside this model benchmark.

Two TIGIT edges supplement LIANA consensus with structural evidence for the PVR and NECTIN2 ligands. The TIDE adapter aligns the author model and expression gene indexes explicitly to support the pinned pandas release. Genes outside the expression assay remain missing in the author correlation. [PVR–TIGIT cocrystal](https://doi.org/10.1073/pnas.1120606109), [NECTIN2–TIGIT cocrystal](https://pmc.ncbi.nlm.nih.gov/articles/PMC5500806/)
