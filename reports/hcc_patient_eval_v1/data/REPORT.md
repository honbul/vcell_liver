# V0 TCGA-LIHC translation audit

The frozen release was evaluated on 368 adult primary tumors and 88 registered compounds. Gene mapping covered 958/1000 checkpoint genes.

| Input | RNA OOD | Abstentions | Median nearest fit distance |
|---|---:|---:|---:|
| native_fpkm | 0/368 | 0/368 | 3.537 |
| source_quantile_rank | 0/368 | 0/368 | 3.484 |

| Fixed-label Hoshida subtype | Tumors |
|---|---:|
| S1 | 108 |
| S2 | 68 |
| S3 | 143 |
| unassigned | 49 |

| Adjusted copy-number contrast | N | Effect | 99% CI | FDR |
|---|---:|---:|---:|---:|
| FGF19:FGFR_panel | 351 | -0.0026 | [-0.0057, +0.0004] | 0.096 |
| CCND1:CDK4_6_panel | 351 | -0.0042 | [-0.0093, +0.0010] | 0.096 |
| MET:MET_panel | 351 | +0.0021 | [-0.0040, +0.0082] | 0.632 |
| VEGFA:VEGFR_panel | 351 | +0.0001 | [-0.0027, +0.0029] | 0.908 |
| CDKN2A:CDK4_6_panel | 351 | +0.0014 | [-0.0041, +0.0070] | 0.632 |

See the accompanying metrics and figures for purity, comparator and gene-program sensitivity. Reported scores represent the released model's screening endpoint. TCGA-LIHC contributes tumor annotations and lacks matched response labels for this panel.
