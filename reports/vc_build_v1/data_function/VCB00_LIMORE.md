# VCB-00 LIMORE data connection

The 2019 Cancer Cell publisher's Table S4 `Activity Area` sheet has 7,290 observed cell–compound endpoints (81 models by 90 compounds). The shared LIMORE drug file agrees for every row to within 5 × 10⁻¹⁰ AA units. We use the publisher's numeric values and join the public GEO basal RNA by original cell identity. The exact supplement and derived bundle checksums are in [the receipt](limore_receipt.json). [Qiu et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7505724/) · [Table S4](https://ars.els-cdn.com/content/image/1-s2.0-S1535610819303289-mmc5.xlsx) · [GEO GSE97098](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE97098)

| Stage | Models | Independent cell families | Compounds | Rows |
|---|---:|---:|---:|---:|
| Publisher Table S4 | 81 | — | 90 | 7,290 |
| Adult HCC, supported chemistry and deposited basal RNA | 77 | 74 | 88 | 6,776 |

Table S1 and the prior public Cellosaurus/DepMap lineage audit identify Hep-G2 and Huh-6 as hepatoblastoma, Hep3B as pediatric HCC (age 8), and Mahlavu as age unconfirmed. CLC19/CLC20 are same-donor cultures. HLE/HLF and PLC8024/PLC/PRF/5 share family IDs. The RNA feature axis contains 19,739 protein-coding genes with unique symbol and Ensembl identifiers. The bulk RNA is a baseline proxy measured outside a matched screening well. [Table S1](https://ars.els-cdn.com/content/image/1-s2.0-S1535610819303289-mmc2.xlsx)

Each Table S4 product code was verified against the manufacturer's catalogue and mapped by product CAS to one PubChem record. The audit retains the catalogue code, formulation SMILES, vendor SDF key where available, PubChem stereo structure and evidence hashes. The exact isomeric parent key identifies model rows. Connectivity and scaffold identifiers are used for split grouping. Cisplatin and oxaliplatin retain their original published responses in Table S4 but are ineligible for the fragment-parent fingerprint path because the PubChem representation contains disconnected platinum ligand components. [Structure identity audit](../../../data_manifests/vc_build_v1/function/limore_chemistry.tsv)

AA is the author-aggregated 72-hour CellTiter-Glo dose-curve summary. Larger AA indicates greater response. Drug-dependent 7/10-point ranges and original-batch visibility are included in the source record. Published AA values are retained, including values above 1.0. Curve summary requests carry the assay protocol, measured dose range and time. This profile has no per-dose functional response output. [Source assay methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7505724/#S21)

Rebuild on G1 from the repository root with Python 3.10+ and pandas, NumPy, RDKit, PyArrow, PyYAML and openpyxl available.

```sh
PYTHONPATH=src python -m vc_build.prepare --config configs/vc_build_v1/function_sources.yaml --output /path/to/new/function-bundle
```

`limore-v1` is saved under `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/function/`. The public dataset was previously used in the Pathway A1 and softprior work. VCB internal group-held-out estimates are a new registered split on this development resource.
