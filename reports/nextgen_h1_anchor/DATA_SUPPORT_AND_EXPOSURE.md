# H1 data support and exposure

The [identity and SHA-256 receipt](data_support_receipt.json) was generated on G1 before any H1 response value was decoded. The audited source bundles were read-only. Exposure checks used the [VC Build work ledger](../vc_build_v1/v0plus/RUN_LEDGER.tsv), [source correction](../vc_build_v1/v0plus/SOURCE_PANEL_EXTENSION.md), [GDSC expansion](../vc_build_v1/v0plus/IMPROVE_PDO_AND_GDSC_SUPPORT.md), [previous 5-sentinel results](../vc_build_v1/PUBLIC_EXTERNAL_RESULTS.md) and [latest local learning curve results](../vc_build_v1/v0plus/HCC_LOCAL_LEARNING_CURVE_RESULTS.md). The workspace preflight dated 2026-11-22 also records the original source access and the LICOB RNA workbook integrity issue. H1 uses the separately registered 76-drug RNA and response roster.

| Tier | Dataset | Groups | Panel | Metadata missing | Exposure |
|---|---|---:|---:|---:|---|
| 1 | GDSC2 | 714 | 68 | 4.89% | Prior model development and assay-held evaluation |
| 1 | CTRPv2 | 720 | 68 | 5.47% | Prior model development and assay-held evaluation |
| 1 | PRISM20Q2 secondary | 704 | 68 | 9.88% | Prior model development and assay-held evaluation |
| 2 | LIMORE | 50 | 68 | 0% | Prior sentinel, external and local evaluations |
| 3 | LICOB | 38 | 76 | 0% | Prior external and local evaluations |
| 4 | LIMORE → LICOB | 38 target donors | 53 | 0% | Both cohorts previously exposed. Separate target-only anchor protocol |
| Auxiliary | Wong | 10 | 62 registered | 0% roster | Prior external evaluation. Real assay coverage assessed in the scorer |
| Excluded | Yang | — | 5 | — | Prior exposure. Fewer than 15 query drugs |

The three-screen exact chemical intersection is 24 even with no density threshold and four at 80% coverage. A single dense 40-drug three-screen panel is infeasible using these exact-parent registrations. H1 uses 68-drug panels fixed independently within each screen. The same exact structure key remains mandatory for pooling source covariance or evaluating transfer. The 2D→LICOB support at ≥80% within-source coverage is 20 GDSC2, 36 CTRPv2, nine PRISM20Q2 and 53 LIMORE.
