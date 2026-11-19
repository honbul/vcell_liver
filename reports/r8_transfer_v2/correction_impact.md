# R8-P0 정정 영향 (correction_impact)

**작성일:** 2026-11-19 · 브랜치 `research/r8-observation-transfer-v2` · 사양 동결 F1 `1b59b76`(YAML 문법 수정 `6a66d0b`) · 근거: 비공개 기획서(2026-11-19) §5.1.
모든 계산은 CPU 닫힌식 재계산이다(훈련 없음). 확인 9 family·HCC는 **이미 노출된 패널**이며 여기서는 정정 영향만 보고한다(새 후보 선택 없음).

## 1. 정정 영향표 (old original / old replay / corrected)

| 항목 | old original (v1 저장값) | old replay (v2 코드, v1 경로) | corrected | 영향 범위 |
|---|---|---|---|---|
| P0-A 문구 3개 + 재현행 표기 | v1 FINAL_RESULTS §0·§2.1, BRANCH_DECISION, NOT_RUN | — | `ERRATA_v1.md` | 문서만. 수치 변화 없음 |
| P0-A `anchor_matched` 행 | 없음 | C0_iid / C1_iid / shrunk_mean 재계산이 v1 저장값과 일치: 일반·확인 최대 5.0e-8, matched 최대 5.0e-7, HCC 단위 최대 5.0e-6(v1 표 6자리 반올림) | `anchor_matched_rows.tsv`, `anchor_decomposition.tsv` | 새 행 추가. C0 − anchor_matched의 centred 예측 차이 최대 9.7e-7(float32 정밀도) |
| P0-B final_inner K = 8 κ_f 정합성 | V3·kernel scale을 κ_f = 8(K16 값)로 적합, K8 anchor는 κ_f = 4 | 확인 1,377행 최대 5.0e-8, matched 11,628행 최대 5.0e-7, HCC 76,194 단위 최대 5.0e-6 | V3·scale을 K8의 κ_f = 4로 적합(`k8_nuisance_keys.tsv`); source 재선택 후 바뀐 상수는 `kn_chem+f0` K8 λ 0.3 → 0.1 하나 | K8의 C0_V3, C1_V3, P_*_V3(s), kn_*, J2, J3(+0-overlap 행)만. K16·K32는 수정 전후 **정확히 0 차이** |
| P0-B inner0/1/2 | 모든 K가 같은 κ_f | 상수 재현 차이 0 | 재선택 상수 차이 0 | source OOF·E1/E2(v1) 영향 없음 → 재계산 불필요 |
| P0-C tie 규칙 | anchor κ_b 오름차순, J1 a/c 사전식 | 반올림 전 inner 후보 점수 376개 선택(4 inner × v1/v2 run × 모든 grid·K, 예산 grid 포함) | v2 규칙(큰 κ_b, 작은 a+c 우선) | **exact tie 0건**, 선택 변경 0건, 최소 1·2위 차이 1.6e-7. 출력 재계산 없음 |

## 2. P0-B 영향의 크기 (K = 8, ΔRMSE, corrected − replay)

| 패널 (이미 노출) | J3 | J3 0-overlap | kn_f0 | kn_all | kn_chem+f0 (재선택) | C0_V3, C1_V3, P_* |
|---|---:|---:|---:|---:|---:|---:|
| 확인 9 family | +0.00017 | +0.00021 | −0.00002 | +0.00005 | +0.00003 | < 1e-6 |
| 확인 matched E1 | +0.00023 | +0.00043 | −0.00016 | −0.00004 | +0.00002 | < 1e-6 |
| 확인 matched E2 | +0.00014 | +0.00014 | −0.00017 | −0.00005 | +0.00004 | < 1e-6 |
| HCC 동등 평균 | +0.00018 | +0.00024 | −0.00004 | +0.00003 | +0.00004 | < 1e-6 |

episode 단위 최대 변화는 0.0020이다. 패널 평균의 변화는 최대 4.3×10⁻⁴(확인 matched E1의 J3 0-overlap 행)이다. K8은 v1에서도 보조 결과였고, v1 주 판단(K16)에 쓰인 행은 하나도 바뀌지 않았다. v1 K8 대조표의 개별 판정을 이 수정값으로 다시 쓰지는 않았다(값은 `k8_panel_impact.tsv`).

## 3. 바뀌지 않은 경로 (확인함)

C1-iid 원 kernel, 모든 iid C0/C1, fx_*(원 상수, V3_fx = κ_f 8 일치), K16 주 결과와 K32, inner0/1/2의 모든 상수, §10 예산 상수(재현 차이 0).

## 4. 앞으로의 tie 구현

v2 선택 코드(`engine2.RULES`)는 문서의 순서를 따른다: 큰 κ_b, 작은 η, 큰 λ, 작은 a+c. 문서가 정하지 않은 κ_f와 J2/J3 scalar 순서는 v1 동작을 유지한다. 반올림된 표시값의 동점은 tie로 취급하지 않는다.
