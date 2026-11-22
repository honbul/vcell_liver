# Same-support molecular context comparison

The publisher's per-cell genomic table covered all 77 age-confirmed adult HCC models and all 74 lineage groups. The identical 6,776 held lineage rows were available to each arm. Publisher target/family labels mapped to 101 candidate genes across 79 of 88 compounds, with at least one RNA gene measured for every mapped compound. The Hallmark signatures retained 42 to 197 genes each. Copy-number and sequence flags are the categorical calls in LIMORE Table S3. They include FGF19 amplification in 10 and CDKN2A homozygous deletion in seven of the analyzed cell lines.

| Feature block added to the same direct ridge | Cell-macro MAE | Cell-macro NDCG@5 | Relative paired lineage MAE gain, 95% CI |
|---|---:|---:|---:|
| Direct ridge reference | 0.108870 | 0.869238 | — |
| Basal target gene expression | 0.108796 | 0.869239 | +0.078% [−0.093%, +0.250%] |
| Basal Hallmark signatures × chemistry | 0.108870 | 0.871790 | +0.001% [−0.607%, +0.610%] |
| HCC driver sequence/CNA flags × chemistry | 0.109313 | 0.869382 | −0.465% [−1.323%, +0.333%] |
| Combined | 0.109309 | 0.873649 | −0.438% [−1.384%, +0.528%] |

The combined arm's secondary C NDCG change was +0.00439 [0.00014, 0.00945] with an unadjusted paired lineage interval. The single arm's size, the comparisons across several blocks and the MAE results leave practical added value unresolved. The prototype retains the original input contract and reports molecular features as a separate secondary test. External reuse of the drug-target map still requires target availability in the exact overlap with the training-source panel and matched expression coverage.

Slurm 32312 wrote predictions from the registered comparison, and 32313 separately scored the sealed table. The verification matched the forecast, separate outcome and annotation hashes. Freeze SHA-256 is `bd1c4b16dcf88b0013ab44e9ffc86d2c72fc50c006876ab0b0b29894ad03dc23`. The data and administrative evidence are on G1 under `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/function/molecular-context.PY2Y8mtE`. [Aggregate evidence](molecular_evidence.json) [Registered protocol](../MOLECULAR_CONTEXT_PROTOCOL.md) [Publisher source](https://pmc.ncbi.nlm.nih.gov/articles/PMC7505724/)
