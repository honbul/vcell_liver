# Post-Hoc Zero-Retraining Paired 95% CI Audit: `E1 Cchem-int` & `F01-only` vs `E4` vs `E5`

**Date**: 2026-11-25  
**Scope**: Paper B (`H2-v2` + `E1` + `E3` + `E4` + `E5`) zero-retraining cross-experiment reconciliation  
**Script**: [run_e1_e4_e5_paired_audit.py](file:///home/gsun/work/vc_liver/paper_b_h2_multiview/analysis/run_e1_e4_e5_paired_audit.py)  
**Outputs**:
- [E1_E4_E5_METRICS.tsv](file:///home/gsun/work/vc_liver/paper_b_h2_multiview/tables/E1_E4_E5_METRICS.tsv)
- [E1_E4_E5_PAIRED_95CI.tsv](file:///home/gsun/work/vc_liver/paper_b_h2_multiview/tables/E1_E4_E5_PAIRED_95CI.tsv)
- [E1_E4_E5_DRUG_POTENCY_DIAGNOSTIC.tsv](file:///home/gsun/work/vc_liver/paper_b_h2_multiview/tables/E1_E4_E5_DRUG_POTENCY_DIAGNOSTIC.tsv)
- [e1_e4_e5_paired_audit_receipt.json](file:///home/gsun/work/vc_liver/paper_b_h2_multiview/tables/e1_e4_e5_paired_audit_receipt.json)

---

## 1. Executive Summary

This audit closes the last cross-experiment statistical gap identified during Paper B pre-manuscript review: direct 2,000-bootstrap paired 95% confidence intervals comparing **`E1 Cchem-int`** (the budget-matched chemistry+target potency intercept + `C1` kernel baseline) and **`E1 F01-only`** (pure LINCS drug-potency intercept with zero cell personalization) directly against **`E4` (`H2-LOSO-Full`)** and **`E5` (`H2-v3-Combined`, `P3B-Shared`, `P3B-InScreen`, `P3C-EqualScreenWeight`, `P1-Minimax`, `P2-Decoupled`, `P3A-ScreenDebiased`)** on identical native rows across all five shared cohorts (`DRP`, `GDSC1`, `LIMORE`, `LICOB`, `LICOB18`).

**Core findings (0 retrained models; exact machine replay verified across 2,502 archived metrics & CIs):**

1. **`E4` (`H2-LOSO-Full`) did NOT surpass `E1 Cchem-int` on `GDSC1`:** despite improving `GDSC1 NDCG@5` by `+0.0971` over `H2-v2`, `H2-LOSO-Full` remained significantly **below** `Cchem-int` on `GDSC1 all NDCG@5` (`-0.0221 [-0.0342, -0.0100]`; `0.5343` vs `0.5564`) and `Precision@5` (`-0.0391 [-0.0524, -0.0263]`), and its `LINCS-observed within-drug ρ` gain crossed zero (`+0.0129 [-0.0026, +0.0278]`).
2. **`E5` (`H2-v3-Combined`, `P3B-Shared`, and `P1-Minimax`) DECISIVELY surpass `E1 Cchem-int` on ALL FOUR metrics simultaneously on BOTH `DRP` and `GDSC1`:**
   - **`GDSC1` (`66` novel-scaffold external drugs, `39,545` rows, `713` cells):**
     - `H2-v3-Combined − Cchem-int`: `all ΔNDCG@5 = +0.0617 [+0.0504, +0.0728]` (`0.6181` vs `0.5564`), `all ΔP@5 = +0.0405 [+0.0279, +0.0535]` (`0.4160` vs `0.3755`), `obs Δρ = +0.0319 [+0.0149, +0.0482]` (`0.2081` vs `0.1761`), `obs Δinteraction ρ = +0.0966 [+0.0546, +0.1389]` (`0.1875` vs `0.0909`).
     - `P3B-Shared − Cchem-int`: `all ΔNDCG@5 = +0.0310 [+0.0184, +0.0440]`, `all ΔP@5 = +0.0248 [+0.0096, +0.0391]`, `obs Δρ = +0.0319 [+0.0149, +0.0482]`, `obs Δinteraction ρ = +0.0966 [+0.0546, +0.1389]`.
     - `P1-Minimax − Cchem-int`: `all ΔNDCG@5 = +0.0828 [+0.0729, +0.0926]` (`0.6391` vs `0.5564`), `all ΔP@5 = +0.0205 [+0.0092, +0.0313]`, `obs Δρ = +0.0277 [+0.0118, +0.0435]`, `obs Δinteraction ρ = +0.0761 [+0.0368, +0.1142]`.
   - **`DRP` (`1,186` drugs, `670,830` rows, `1,014` cells):**
     - `H2-v3-Combined` (InScreen) `− Cchem-int`: `all ΔNDCG@5 = +0.0646 [+0.0612, +0.0678]`, `all ΔP@5 = +0.0750 [+0.0704, +0.0793]`, `obs Δρ = +0.1280 [+0.1194, +0.1366]`, `obs Δinteraction ρ = +0.1664 [+0.1452, +0.1898]`.
     - `P3B-Shared` (pure $f_{\text{shared}}$, no screen ID) `− Cchem-int`: `all ΔNDCG@5 = +0.0627 [+0.0593, +0.0661]`, `all ΔP@5 = +0.0573 [+0.0524, +0.0620]`, `obs Δρ = +0.0312 [+0.0241, +0.0385]`, `obs Δinteraction ρ = +0.1233 [+0.1045, +0.1441]`.
3. **Mechanistic regime split (`LINCS-observed` vs `LINCS-unobserved`):**
   - On the **`48` LINCS-observed novel-scaffold `GDSC1` drugs**, `H2-v3-Combined` achieves `NDCG@5 = 0.6943` (`+0.0944 [+0.0831, +0.1061]` over `Cchem-int 0.5999`) and `P1-Minimax` reaches `0.7199` (`+0.1200 [+0.1102, +0.1298]`), backed by a large jump in novel-drug mean potency Spearman correlation (`+0.6358`–`+0.6571` vs `Cchem-int +0.2634`).
   - Conversely, on the **`18` LINCS-unobserved novel-scaffold `GDSC1` drugs**, **`Cchem-int` (`0.5039`) still decisively outperforms every `H2` variant** (`H2-v3-Combined 0.3609`, `-0.1430 [-0.1531, -0.1328]`; `P3B-Shared 0.3320`, `-0.1719`; `H2-v2`/`P1-Minimax` `0.2994`, `-0.2045`), because `H2` falls back to weaker chemistry-target intercepts when perturbation signatures are absent. On `DRP`'s `462` LINCS-unobserved drugs, `H2-v3-Combined` (`0.6400`) slightly beats `Cchem-int` (`0.6282`, `+0.0118 [+0.0091, +0.0146]`).
4. **HCC external boundary (`LIMORE`, `LICOB`, `LICOB18`):**
   - On `LIMORE` (2D HCC, `50` lines), `H2-v3-Combined` / `P3B-Shared` eliminate `H2-v2`'s `obs ρ` deficit vs `Cchem-int` (`-0.0373 → -0.0152 [-0.0406, +0.0118]`) while significantly improving `obs interaction ρ` (`+0.0342 [+0.0054, +0.0628]`) at parity on `all NDCG@5` (`+0.0036 [-0.0119, +0.0199]`).
   - On 3D `LICOB` (`38` donors), absolute `within-drug ρ` stays near zero (`0.0010`–`0.0361`) for all models, confirming the 2D→3D personalization boundary; on `LICOB18` (`18` novel drugs), `H2-v3-Combined` trails `Cchem-int` on `all NDCG@5` (`-0.0474 [-0.0759, -0.0235]`) due to its `9`-drug unobserved fallback, whereas `P3B-Shared` (`+0.0063 [-0.0185, +0.0310]`) matches and `P1-Minimax` (`+0.0790 [+0.0507, +0.1088]`) beats `Cchem-int`.

---

## 2. Provenance & Exact Machine Replay Audit

All frozen predictions were verified by SHA-256 against their archived `E1`, `E4`, and `E5` receipts before scoring with the identical `followup_h2.inference` engine (`DRAWS = 2,000`, `seed = 495123`; Bayesian cluster bootstrap over transitive `scaffold_block` for rank metrics and over `lineage_group` for retrieval metrics; `unique_top5_sets` reported strictly as a descriptive statistic without bootstrap CIs).

| Audit Check | Items Verified | Max Abs Diff | Worst-Case Entry |
|---|---:|---:|---|
| Row-wise forecast parity (`E1 F11` vs `E5 H2-v2`, `E1 F00` vs `E5 H2-v1`, `E4` vs `E5 H2-LOSO-Full`, parts `0–3`) | 4 roles (`717,043` rows) | **`0.0`** | Exact bit-level match |
| Replay vs archived `E5_METRICS.tsv` (macro across 5 cohorts × 3 supports) | `870` | **`2.84e-14`** | `DRP / lincs_observed / P3B-InScreen / unique_top5_sets` |
| Replay vs archived `E5_PAIRED_95CI.tsv` (`delta`, `lo95`, `hi95`) | `1,152` | **`9.97e-17`** | `LICOB / lincs_observed / P2-Decoupled - H2-v2 / interaction_spearman` |
| Replay vs archived `E1_METRICS.tsv` (`Cchem-int`, `F01-only`, `F00`, `F11`) | `300` | **`3.24e-10`** | `DRP / lincs_unobserved / H2-v2 / interaction_spearman` (`0.0` on GDSC1/LIMORE/LICOB/LICOB18) |
| Replay vs archived `E1_PAIRED_95CI.tsv` (`delta`, `lo95`, `hi95`) | `180` | **`1.29e-09`** | `DRP / lincs_unobserved / H2-v2 - F01-only / interaction_spearman` |

---

## 3. Direct Paired 95% CIs vs `E1 Cchem-int`

### 3.1 `GDSC1` (`66` novel-scaffold drugs · `48` LINCS-observed · `18` LINCS-unobserved · `713` cells · `39,545` rows)

*(Previously exposed external audit; all predictions use pure shared cross-platform inference $f_{\text{shared}}$.)*

| Arm | `all NDCG@5` | `all P@5` | `obs ρ` | `obs inter ρ` | `uniq Top-5` *(desc.)* | `all ΔNDCG@5 vs Cchem-int` | `all ΔP@5 vs Cchem-int` | `obs Δρ vs Cchem-int` | `obs Δinter ρ vs Cchem-int` |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| **`Cchem-int` (`E1`)** | 0.5564 | 0.3755 | 0.1761 | 0.0909 | 90 | — | — | — | — |
| `C1` | 0.4221 | 0.2565 | 0.1761 | 0.0909 | 92 | `-0.1343 [-0.1423, -0.1258]` | `-0.1190 [-0.1286, -0.1081]` | `0.0000` | `0.0000` |
| `H2-v2` (`F11`) | 0.4371 | 0.2357 | 0.2065 | 0.1604 | 93 | `-0.1192 [-0.1330, -0.1045]` | `-0.1398 [-0.1558, -0.1236]` | `+0.0304 [+0.0124, +0.0473]` | `+0.0695 [+0.0352, +0.1028]` |
| `H2-LOSO-Full` (`E4`) | 0.5343 | 0.3364 | 0.1890 | 0.1485 | 260 | **`-0.0221 [-0.0342, -0.0100]`** | **`-0.0391 [-0.0524, -0.0263]`** | `+0.0129 [-0.0026, +0.0278]` | `+0.0576 [+0.0133, +0.1003]` |
| **`H2-v3-Combined` (`E5`)** | **0.6181** | **0.4160** | **0.2081** | **0.1875** | 211 | **`+0.0617 [+0.0504, +0.0728]`** | **`+0.0405 [+0.0279, +0.0535]`** | **`+0.0319 [+0.0149, +0.0482]`** | **`+0.0966 [+0.0546, +0.1389]`** |
| **`P3B-Shared` (`E5`)** | **0.5873** | **0.4003** | **0.2081** | **0.1875** | 69 | **`+0.0310 [+0.0184, +0.0440]`** | **`+0.0248 [+0.0096, +0.0391]`** | **`+0.0319 [+0.0149, +0.0482]`** | **`+0.0966 [+0.0546, +0.1389]`** |
| **`P1-Minimax` (`E5`)** | **0.6391** | **0.3961** | **0.2038** | **0.1670** | 160 | **`+0.0828 [+0.0729, +0.0926]`** | **`+0.0205 [+0.0092, +0.0313]`** | **`+0.0277 [+0.0118, +0.0435]`** | **`+0.0761 [+0.0368, +0.1142]`** |
| `P3C-EqualScreenWeight` | 0.5384 | 0.3339 | **0.2100** | 0.1680 | 131 | `-0.0180 [-0.0307, -0.0049]` | `-0.0416 [-0.0570, -0.0257]` | `+0.0339 [+0.0150, +0.0522]` | `+0.0771 [+0.0355, +0.1169]` |
| `P2-Decoupled` | 0.5253 | 0.3342 | 0.2038 | 0.1670 | 377 | `-0.0311 [-0.0435, -0.0183]` | `-0.0414 [-0.0562, -0.0278]` | `+0.0277 [+0.0118, +0.0435]` | `+0.0761 [+0.0368, +0.1142]` |
| `P3A-ScreenDebiased` | 0.6396 | 0.3977 | 0.0938 | 0.1774 | 184 | `+0.0832 [+0.0739, +0.0923]` | `+0.0222 [+0.0111, +0.0327]` | `-0.0823 [-0.1528, -0.0165]` | `+0.0865 [+0.0450, +0.1278]` |

#### Mechanistic Support Decomposition on `GDSC1` (`48` LINCS-observed vs `18` LINCS-unobserved)

| Arm | `obs NDCG@5` (`48` drugs) | `obs ΔNDCG vs Cchem-int` | `unobs NDCG@5` (`18` drugs) | `unobs ΔNDCG vs Cchem-int` | `all Δρ vs Cchem-int` (`66` drugs) | Drug-Mean Potency $\rho$ (`all` / `obs` / `unobs`) |
|---|---:|---|---:|---|---|---|
| `Cchem-int` (`E1`) | 0.5999 | — | **0.5039** | — | — | `+0.2191` / `+0.2634` / **`+0.1950`** |
| `H2-v2` | 0.4951 | `-0.1048 [-0.1207, -0.0875]` | 0.2994 | `-0.2045 [-0.2156, -0.1932]` | `+0.0221 [+0.0081, +0.0357]` | `+0.3708` / `+0.5715` / `-0.1950` |
| `H2-LOSO-Full` (`E4`) | 0.5934 | `-0.0065 [-0.0196, +0.0069]` | 0.3949 | `-0.1090 [-0.1190, -0.0993]` | `+0.0094 [-0.0023, +0.0204]` | `+0.4750` / `+0.6093` / `+0.1538` |
| **`H2-v3-Combined`** | **0.6943** | **`+0.0944 [+0.0831, +0.1061]`** | 0.3609 | `-0.1430 [-0.1531, -0.1328]` | **`+0.0232 [+0.0104, +0.0366]`** | **`+0.4901`** / `+0.6358` / `+0.1517` |
| **`P3B-Shared`** | **0.6569** | **`+0.0570 [+0.0430, +0.0719]`** | 0.3320 | `-0.1719 [-0.1831, -0.1601]` | **`+0.0232 [+0.0104, +0.0366]`** | `+0.4596` / **`+0.6571`** / `-0.1765` |
| **`P1-Minimax`** | **0.7199** | **`+0.1200 [+0.1102, +0.1298]`** | 0.2994 | `-0.2045 [-0.2156, -0.1932]` | **`+0.0201 [+0.0080, +0.0332]`** | `+0.4307` / `+0.6494` / `-0.1950` |
| `P3C-EqualScreenWeight` | 0.6016 | `+0.0017 [-0.0124, +0.0170]` | 0.3320 | `-0.1719 [-0.1831, -0.1601]` | `+0.0247 [+0.0101, +0.0390]` | `+0.4586` / `+0.6449` / `-0.1765` |

---

### 3.2 `DRP` (`1,186` drugs · `724` LINCS-observed · `462` LINCS-unobserved · `1,014` cells · `670,830` rows)

*(Held-out 3-fold outer scaffold CV across `CTRPv2`, `GDSC2`, `PRISM20Q2`; both `P3B-Shared` [no screen ID] and `P3B-InScreen` / `H2-v3-Combined` [uses screen component $\tau f_s$] are reported side-by-side.)*

| Arm | `all NDCG@5` | `all P@5` | `obs ρ` | `obs inter ρ` | `all ΔNDCG@5 vs Cchem-int` | `obs Δρ vs Cchem-int` | `obs Δinter ρ vs Cchem-int` | `unobs ΔNDCG vs Cchem-int` |
|---|---:|---:|---:|---:|---|---|---|---|
| **`Cchem-int` (`E1`)** | 0.6014 | 0.2607 | 0.3109 | 0.1732 | — | — | — | — (`0.6282`) |
| `H2-v2` (`F11`) | 0.6721 | 0.3353 | 0.3480 | 0.2814 | `+0.0707 [+0.0669, +0.0744]` | `+0.0371 [+0.0303, +0.0438]` | `+0.1082 [+0.0914, +0.1262]` | `-0.0562 [-0.0589, -0.0536]` |
| `H2-LOSO-Full` (`E4`) | 0.6640 | 0.3273 | 0.3210 | 0.2623 | `+0.0626 [+0.0589, +0.0663]` | `+0.0100 [+0.0041, +0.0160]` | `+0.0891 [+0.0711, +0.1096]` | `+0.0035 [+0.0007, +0.0061]` |
| **`H2-v3-Combined` (InScreen)** | 0.6660 | **0.3356** | **0.4390** | **0.3396** | **`+0.0646 [+0.0612, +0.0678]`** | **`+0.1280 [+0.1194, +0.1366]`** | **`+0.1664 [+0.1452, +0.1898]`** | **`+0.0118 [+0.0091, +0.0146]`** |
| **`P3B-Shared` (no screen ID)** | 0.6641 | 0.3179 | 0.3422 | 0.2965 | **`+0.0627 [+0.0593, +0.0661]`** | **`+0.0312 [+0.0241, +0.0385]`** | **`+0.1233 [+0.1045, +0.1441]`** | `-0.0562 [-0.0589, -0.0536]` |
| **`P3C-EqualScreenWeight`** | 0.6710 | 0.3281 | 0.3563 | 0.2954 | **`+0.0696 [+0.0660, +0.0731]`** | **`+0.0454 [+0.0381, +0.0526]`** | **`+0.1222 [+0.1040, +0.1426]`** | `-0.0562 [-0.0589, -0.0536]` |
| **`P1-Minimax`** | **0.6753** | 0.3316 | 0.3477 | 0.2864 | **`+0.0739 [+0.0704, +0.0776]`** | **`+0.0367 [+0.0302, +0.0434]`** | **`+0.1131 [+0.0953, +0.1330]`** | `-0.0562 [-0.0589, -0.0536]` |
| `P2-Decoupled` | 0.6705 | 0.3338 | 0.3477 | 0.2864 | `+0.0691 [+0.0654, +0.0727]` | `+0.0367 [+0.0302, +0.0434]` | `+0.1131 [+0.0953, +0.1330]` | `+0.0093 [+0.0065, +0.0120]` |

---

### 3.3 HCC External Cohorts (`LIMORE`, `LICOB`, `LICOB18`) vs `E1 Cchem-int`

| Cohort | Arm | `all NDCG@5` | `obs ρ` | `obs inter ρ` | `all ΔNDCG@5 vs Cchem-int` | `obs Δρ vs Cchem-int` | `obs Δinter ρ vs Cchem-int` |
|---|---|---:|---:|---:|---|---|---|
| **`LIMORE`** (`68` drugs, `59` obs, `50` 2D HCC) | `Cchem-int` (`=C1`) | 0.5387 | 0.3005 | 0.2465 | — | — | — |
| | `H2-v2` | 0.5251 | 0.2633 | 0.2714 | `-0.0136 [-0.0316, +0.0044]` | `-0.0373 [-0.0645, -0.0086]` | `+0.0249 [-0.0024, +0.0523]` |
| | `H2-LOSO-Full` (`E4`) | 0.5478 | 0.2994 | 0.2746 | `+0.0091 [-0.0043, +0.0218]` | `-0.0012 [-0.0247, +0.0205]` | `+0.0281 [-0.0053, +0.0645]` |
| | **`H2-v3-Combined` / `P3B-Shared`** | 0.5423 | 0.2854 | 0.2807 | `+0.0036 [-0.0119, +0.0199]` | `-0.0152 [-0.0406, +0.0118]` | **`+0.0342 [+0.0054, +0.0628]`** |
| | `P3C-EqualScreenWeight` | 0.5216 | **0.3024** | **0.3045** | `-0.0170 [-0.0335, -0.0015]` | `+0.0019 [-0.0228, +0.0249]` | **`+0.0580 [+0.0338, +0.0829]`** |
| | `P1-Minimax` | 0.5344 | 0.2870 | 0.2865 | `-0.0043 [-0.0158, +0.0068]` | `-0.0136 [-0.0305, +0.0042]` | **`+0.0400 [+0.0209, +0.0592]`** |
| **`LICOB`** (`76` drugs, `55` obs, `38` 3D donors) | `Cchem-int` | 0.6454 | 0.0010 | 0.1048 | — | — | — |
| | `H2-v2` | 0.6622 | 0.0260 | 0.1184 | `+0.0168 [+0.0007, +0.0334]` | `+0.0250 [-0.0012, +0.0502]` | `+0.0136 [-0.0104, +0.0372]` |
| | `H2-LOSO-Full` (`E4`) | 0.6378 | 0.0022 | 0.1164 | `-0.0077 [-0.0194, +0.0025]` | `+0.0012 [-0.0245, +0.0287]` | `+0.0116 [-0.0194, +0.0423]` |
| | `H2-v3-Combined` / `P3B-Shared` | 0.6360 | 0.0361 | 0.1230 | `-0.0094 [-0.0265, +0.0058]` | `+0.0352 [+0.0093, +0.0620]` | `+0.0182 [-0.0127, +0.0494]` |
| | `P3C-EqualScreenWeight` | 0.6533 | 0.0303 | 0.1210 | `+0.0079 [-0.0062, +0.0218]` | `+0.0293 [+0.0077, +0.0493]` | `+0.0162 [-0.0090, +0.0418]` |
| **`LICOB18`** (`18` novel drugs, `9` obs, `38` 3D donors) | `Cchem-int` | 0.6819 | -0.0919 | 0.0865 | — | — | — |
| | `H2-v2` | 0.6182 | -0.0746 | 0.0590 | `-0.0637 [-0.0892, -0.0372]` | `+0.0172 [-0.0505, +0.0865]` | `-0.0275 [-0.1271, +0.1111]` |
| | `H2-LOSO-Full` (`E4`) | 0.6598 | -0.0811 | 0.0777 | `-0.0221 [-0.0475, +0.0021]` | `+0.0108 [-0.0669, +0.1182]` | `-0.0088 [-0.1505, +0.1805]` |
| | `H2-v3-Combined` | 0.6344 | -0.0297 | 0.0807 | **`-0.0474 [-0.0759, -0.0235]`** | `+0.0622 [-0.0410, +0.1710]` | `-0.0058 [-0.1152, +0.1493]` |
| | `P3B-Shared` | 0.6882 | -0.0297 | 0.0807 | `+0.0063 [-0.0185, +0.0310]` | `+0.0622 [-0.0410, +0.1710]` | `-0.0058 [-0.1152, +0.1493]` |
| | `P3C-EqualScreenWeight` | 0.6986 | -0.0941 | 0.0548 | `+0.0167 [-0.0062, +0.0399]` | `-0.0022 [-0.0574, +0.0477]` | `-0.0317 [-0.1566, +0.1407]` |
| | `P1-Minimax` | **0.7609** | -0.1031 | 0.0647 | **`+0.0790 [+0.0507, +0.1088]`** | `-0.0113 [-0.0624, +0.0299]` | `-0.0218 [-0.1434, +0.1427]` |

---

## 4. Direct Paired 95% CIs vs `E1 F01-only` (Personalization Gain Over Pure Potency Intercept)

| Cohort | `H2-v2 − F01-only` (`all ΔNDCG@5`) | `H2-LOSO-Full − F01-only` (`all ΔNDCG@5`) | `H2-v3-Combined − F01-only` (`all ΔNDCG@5`) | `P3B-Shared − F01-only` (`all ΔNDCG@5`) | `P1-Minimax − F01-only` (`all ΔNDCG@5`) |
|---|---|---|---|---|---|
| **`DRP`** | `+0.0282 [+0.0260, +0.0302]` | `+0.0202 [+0.0178, +0.0226]` | **`+0.0221 [+0.0192, +0.0249]`** | `+0.0202 [+0.0183, +0.0221]` | **`+0.0315 [+0.0292, +0.0337]`** |
| **`GDSC1`** | `+0.0066 [-0.0013, +0.0141]` | `+0.1037 [+0.0885, +0.1182]` | **`+0.1875 [+0.1746, +0.2000]`** | `+0.1568 [+0.1485, +0.1654]` | **`+0.2086 [+0.1944, +0.2221]`** |
| **`LIMORE`** | `+0.0150 [-0.0150, +0.0467]` | `+0.0377 [+0.0222, +0.0532]` | **`+0.0322 [+0.0203, +0.0446]`** | `+0.0322 [+0.0203, +0.0446]` | `+0.0243 [+0.0015, +0.0464]` |
| **`LICOB`** | `+0.0463 [+0.0100, +0.0793]` | `+0.0218 [-0.0010, +0.0413]` | `+0.0200 [+0.0019, +0.0350]` | `+0.0200 [+0.0019, +0.0350]` | `+0.0320 [+0.0046, +0.0565]` |
| **`LICOB18`** | `+0.0129 [+0.0024, +0.0246]` | `+0.0545 [+0.0317, +0.0799]` | `+0.0292 [+0.0061, +0.0514]` | `+0.0829 [+0.0705, +0.0955]` | `+0.1556 [+0.1265, +0.1855]` |

---

## 5. Direct Implications for Paper B Claim Governance

1. **Upgraded Cross-Platform Claim (now backed by direct paired 95% CIs):**
   - Paper B can now state with strict paired-CI proof that while pure `LOSO-CV` (`E4`) alone closed most of the `H2-v2` cross-platform retrieval deficit without beating `E1 Cchem-int` on `GDSC1` (`-0.0221 [-0.0342, -0.0100]`), adding the hierarchical multi-task screen kernel (`P3B-HierarchicalMTK` / `H2-v3-Combined`) or minimax Pooled–LOSO selection (`P1-Minimax`) **decisively outperforms `Cchem-int` on ALL FOUR metrics simultaneously** on both `DRP` and `GDSC1` (e.g., `H2-v3-Combined` on `GDSC1`: `+0.0617` `NDCG@5`, `+0.0405` `P@5`, `+0.0319` `obs ρ`, `+0.0966` `obs interaction ρ`).
2. **Sharp Mechanistic Boundary (zero overclaiming):**
   - The `GDSC1` Top-5 advantage over `Cchem-int` comes **exclusively from LINCS-observed novel scaffolds (`48` drugs: `+0.0944` to `+0.1200` `NDCG@5`, drug-mean potency $\rho = 0.64$–$0.66$ vs `0.26`)**, whereas on **`18` LINCS-unobserved novel scaffolds**, `Cchem-int`'s dedicated 10-neighbor chemistry+target anchor (`0.5039`) still beats all `H2` fallbacks (`0.2994`–`0.3949`).
   - This directly reinforces Paper B's core thesis: **perturbation signatures (`LINCS 24h`) provide genuine orthogonal gains in both potency transfer and cell-line personalization when measured, whereas unobserved drugs remain governed by chemistry/target fallback quality.**
3. **Transparent Dual Reporting Retained:**
   - On `DRP`, report both `P3B-Shared` (`ρ = 0.3422`, `+0.0312` vs `Cchem-int`, no screen ID) and `H2-v3-Combined` / `P3B-InScreen` (`ρ = 0.4390`, `+0.1280` vs `Cchem-int`, uses screen-specific $\tau f_s$).
   - Keep `unique_top5_sets` strictly descriptive (no cluster-bootstrap CI claims), and retain the `LICOB` 3D personalization negative boundary (`ρ ≤ 0.036`) and `LICOB18` `H2-v3-Combined` unobserved-drug fallback deficit (`-0.0474` vs `Cchem-int`).
