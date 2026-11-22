# HepG2, Chronos and the LIMORE genetic bridge

The benchmark measures genomic and pharmacological transfer in public human cell cultures. HepG2 is hepatoblastoma-derived. The adult HCC Chronos evaluation uses DepMap histology, age and donor identity. SK-HEP-1 is excluded based on endothelial provenance. [SK-HEP-1 provenance](https://pubmed.ncbi.nlm.nih.gov/1371504/)

| Evaluation | Fitting labels | Held-out labels and grouping |
| --- | --- | --- |
| HepG2 transcriptional change | K562 essential and genome-wide, RPE1 and Jurkat effects after GEM-specific non-targeting subtraction | External HepG2. Gene-cold excludes all test Ensembl targets from source fitting and selection. Same-gene evaluation requires a source fit intervention. Every query uses the same negative-control-only HepG2 reference. |
| Chronos | DepMap 25Q3 effect with default basal RNA, CNV and mutation from non-liver lines | Independent adult HCC donors. Gene and donor identity control training, selection and evaluation. |
| LIMORE drug response | Non-test LIMORE Activity Area | Generic Murcko scaffolds and parent connectivity. The target-cold experiment removes any training or validation compound sharing a registered human protein target with a test compound. |

HepG2 delta is mean single-cell log1p(CP10k) minus the matched control mean. The drug bridge joins ChEMBL 37 direct human mechanism claims and LIMORE target annotations by registered compound identity. Unseen-target status applies to registered targets and cannot exclude undocumented activity. Chronos features use exact DepMap model IDs. Unavailable values use the non-liver assay median, with matched-only metrics reported separately. [GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) · [ChEMBL API](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)

The GEARS-style model projects public STRING neighbors alongside ESM-2 and negative-control context. It has its own decoder and graph topology. Baselines use no change, mean, same-gene copy, target expression attenuation, Ridge, ElasticNet and LightGBM. Separately centered residuals, raw error and DEG recovery are computed after the forecast freeze. [GEARS](https://doi.org/10.1038/s41587-023-01905-6) · [Systema](https://pmc.ncbi.nlm.nih.gov/articles/PMC13271886/)

Chronos ablations omit cell context or gene embeddings. Drug-bridge models share heads and query rows across chemical, genetic and combined features and measure the gain from each assay. Predicted HepG2 signatures use a model chosen solely by non-hepatic validation. The scorer first averages any sibling cultures within their registered lineage group.

Run `python -m vc_crispr_bridge.predict {hepg2,chronos,bridge} --help` and the corresponding `python -m vc_crispr_bridge.score` command offline. The Slurm scripts give all required input paths. Output receipts record file SHA-256 hashes and registration links.

The final target registry joins independent GDSC2, CTRPv2 and PRISM screen target annotations through verified native compound IDs. The union covers 87 LIMORE compounds and 231 registered genes. Exact structure matches supplied 32, 48 and 17 LIMORE compounds, respectively. The strict native-target-extended split removes every compound with a known target shared with the other roles. Across the five held folds each test set contains 17–19 compounds and has zero overlap in recorded targets with fit and validation. SHA-256 receipts preserve the independently frozen ChEMBL, LIMORE and source-screen records.

## External performance

The conditional MLP had centered RMSE 0.11167 and top-20 DEG recall 8.03% on 38 held HepG2 genes. LightGBM reached 0.11121 and 11.84%, and the no-change model reached centered RMSE 0.11115. On held adult HCC Chronos cultures, conditional MLP and LightGBM reached mean within-gene Spearman 0.00913 and 0.00938, respectively. The validation-selected joint bridge added 0.00085 within-drug Spearman on scaffold-held drugs with paired 95% interval [−0.01183, 0.01303]. On registered-target holdout it added 0.00857 [0.00311, 0.01475]. Some bridge ElasticNet candidates emitted solver convergence warnings. The observed genetic gain is small and concentrated in the registered-target test. [Complete external scorecard](RESULTS.md)
