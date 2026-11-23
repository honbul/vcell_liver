# H1-v2 support and exposure

H1-v2 reuses the signed v1 split, group identities, panel and held query support. The v1 project and earlier VC work evaluated these source cohorts. H1-v2 provides a new locked technical comparison on exposed cohorts. [v1 input and exposure audit](../nextgen_h1_anchor/DATA_SUPPORT_AND_EXPOSURE.md) · [v1 group integrity](../nextgen_h1_anchor/fold_integrity_receipt.json) · [v1 scoring receipt](../nextgen_h1_anchor/score_receipt.json)

| Tier | Study | Independent groups on primary support | Registered drugs | Q8 |
|---|---|---:|---:|---:|
| 1 | GDSC2 | 680 | 68 | 60 |
| 1 | CTRPv2 | 622 | 68 | 60 |
| 1 | PRISM20Q2 | 430 | 68 | 60 |
| 2 | LIMORE | 50 | 68 | 60 |
| 3 | LICOB | 38 | 76 | 68 |
| 4 | LIMORE → LICOB | 38 target, 50 source | 53 | 45 |

Main results use the exact v1 parent panels and registered high-density within-screen support. Cabozantinib is unregistered in the LICOB panel. Its subgroup row will report lack of support. The Tier 4 target has no RNA inputs or local donor fits. Wong is outside the H1-v2 Tier 1–4 scope.

The preflight verified the immutable v1 input and score receipts and the 16 included task denominators. G1 provided LightGBM 4.6.0 and PyTorch 2.6.0+cu118. A typing-extensions version mismatch on three CPU nodes was repaired with a read-only module copy under the H1-v2 dependency directory. [Data and runtime receipt](preflight_receipt.json) · [Isolated module receipt](dependency_overlay_receipt.json)
