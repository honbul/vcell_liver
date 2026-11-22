# Additional HCC cohort access audit

| Cohort | Evidence and public access | Use in this audit |
|---|---|---|
| GSE109211, BIOSTORM | GEO matrix inspected and frozen on G1. It has 140 FFPE tumors, 67 assigned sorafenib and 73 placebo. | Exploratory only. The randomized trial evaluated recurrence prevention after resection or ablation. The companion signature was derived in the BIOSTORM subset and is not an independently adjudicated drug-sensitivity label for each tumor. The trial's parent cohort did not show a sorafenib recurrence-free survival benefit. |
| GSE14520 | Both GEO matrices were checked. GPL3921 has 445 profiles and GPL571 has 43 profiles, including tumor, adjacent tissue and controls. | Candidate external molecular-cohort check. Sample count includes tissues and needs HCC-only filtering and explicit probe harmonization. |
| ICGC LIRI-JP | HCCDB originally included 212 HCC and 177 adjacent profiles from LIRI-JP. The source ICGC Data Portal now redirects to its retirement notice. Authorized ICGC 25K access is described separately. | Cohort enumerated from the published HCCDB table. No raw clinical outcomes were joined to the TCGA audit. |
| HCCDB | The original resource documented 15 cohorts, 3,917 profiles and 2,396 tumor profiles. The linked live expression server was unreachable during this audit. HCCDB6 contains 225 tumor profiles from GSE14520 and HCCDB15 contains 356 from TCGA-LIHC. | Sources overlap the public cohorts already inventoried. Count overlap explicitly before using HCCDB for an external replication. |

Sources are [BIOSTORM GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109211), [BIOSTORM trial companion paper](https://doi.org/10.1136/gutjnl-2018-316408), [GSE14520 GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14520), [ICGC retirement and access](https://dcc.icgc.org/index.html), and the [HCCDB source paper and cohort counts](https://pmc.ncbi.nlm.nih.gov/articles/PMC6205074/).

[SHA-256 of downloaded GEO matrices](data/external_geo.sha256)
