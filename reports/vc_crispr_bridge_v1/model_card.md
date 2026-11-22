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
