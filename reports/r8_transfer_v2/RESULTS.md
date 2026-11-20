# R8 v2 — 정정(P0)과 관측 환경 전이 후속(E1 / E2-A / E2-B) 결과

**작성일:** 2026-11-19 · 브랜치 `research/r8-observation-transfer-v2` (기준 R8 후속 v1 tip `2f4329e`) · 사양 동결 **F1 `1b59b76`** (YAML 문법만 고친 `6a66d0b`) · source 결정 동결 **F2** `0be3577` (코드 `84dbfa3`) · E2-B 사양 **F3 `84dbfa3`** · 근거: 비공개 기획서(2026-11-19) §1, §2, §5, §8, §9.

모든 수치는 CMap LINCS 2020 Level 3, 978 landmark, (세포주, 약물 그룹) unit 단위 ΔRMSE 또는 centred RMSE다. 차이는 "앞 − 뒤"이고 음수면 앞 방법의 오차가 작다. source는 family cluster bootstrap 95% CI(2,000회), HCC는 약물 ID 동기화 bootstrap. `[k/3]`은 앞 방법이 좋았던 outer fold 수.

**노출 지위.** 이 트랙의 모든 자료는 R8 또는 R8 후속 v1에서 이미 분석되었다. P0는 **기존 결과 정정**, E1/E2-A의 source 수치는 **이미 노출된 자료의 후속 분석**(새 matched episode 집합이지만 같은 family·plate), HCC 표는 **동결된 후속 평가**다. 새 독립 자료나 blind test는 없다.

## 0. 요약

- **P0 정정 완료.** v1 문구 3개와 재현행 표기를 정정했고(`ERRATA_v1.md`), C0/C1과 같은 κ_f의 `anchor_matched` 행을 모든 패널에 추가했다. 공통 보정의 순수 효과(같은 anchor)는 source OOF −0.0094, HCC −0.0211. K8 κ_f 불일치 수정은 K16/K32를 정확히 보존했고 K8 보조 행을 ≤ 4×10⁻⁴ 바꿨다. inner 선택에 exact tie는 0건이었다.
- **E1.** 환경 shift(support가 다른 X 반복 plate)에서 공통 보정의 이득은 −0.0285 → −0.0023으로 거의 사라지지만 약물별 보정(C1 − C0)은 −0.0030 → −0.0019로 유지된다(H1 지지). J3는 C1보다는 낮았지만 같은 metadata의 단순 공통항 교체 C1J1과의 차이가 확립되지 않아(−0.00072 [−0.00170, +0.00013]) **채택하지 않는다**(H2). 기본 방법 C1 유지, 별도 plate를 아는 경우 C1J1 같은 단순 공통항 조정을 우선.
- **E2-A.** shift 조건의 약물별 정보는 같은 약물의 source 반응(f0) 공간에서 온다(f0 추가 −0.0025 centred, f0 대응 순열 +0.0037); 화학 구조의 추가 기여는 shift 조건에서 미확립.
- **E2-B (조건 충족으로 실행).** 같은 약물 source 반응을 완전히 막으면 화학 경로(ridge anchor + chem kernel)가 fallback보다 −0.0024(centred) 작게 개선하지만, 같은 약물 반응을 쓰는 계약과의 차이(−0.028)가 훨씬 크다.
- **HCC(동결 후속, 노출된 패널).** J3 − C1 −0.0026 [−0.0040, −0.0011], centred 미확립; C1J1 ≡ C1. 후보 변경 없음.
- **미실행:** R8-E3(다음 측정 결정 없음) 등 — `NOT_RUN.md`.

## 1. R8-P0 — 정정 (CPU 닫힌식 재계산, 훈련 없음)

**목적.** v1 보고서의 잘못된 설명 세 가지를 고치고, C0/C1과 같은 anchor를 쓰는 기준선을 추가하고, K = 8의 κ_f 불일치와 tie 규칙의 실제 영향을 측정한다.

**수정 내용.** (A) `ERRATA_v1.md`: "C0 = 축소 평균" → 같은 κ_f의 `anchor_matched`일 때만 동일; "예측 차이 ~1e-6" → family 평균 ΔRMSE 차이; "공분산 효과는 전부 축소량" → E1/E2의 C0 공통 보정에서만 정확. v1 `FINAL_RESULTS.md`에는 정정 안내만 추가하고 원문은 보존했다. 재현표 105행을 원 상수 C0/C1 재현 60행, 상수 없는 기준선 30행, **재현이 아닌** 독립 재선택 shrunk_mean 15행으로 구분했다(`reproduction_row_kinds.tsv`). (B) covariance를 (partition, reference, κ_f), kernel scale을 (partition, reference, K, κ_f)로 식별하도록 바꿨다(`engine2.Ctx2`). (C) inner 후보 점수 grid를 반올림 없이 다시 만들고(`grids.tsv.gz`, 서버) 문서화된 tie 순서를 구현했다.

**실험 입력/정답.** v1과 같은 동결 입력·상수·episode(source OOF 91 family, source E1/E2 41 family, 확인 9 family와 그 matched 5 family, HCC 3계열 × 3 split). 정답은 점수 계산에만 쓰였다.

**결과.**

| 정정 | 결과 |
|---|---|
| 재현 (old replay) | C0_iid / C1_iid / shrunk_mean 재계산이 v1 저장값과 일치: episode 최대 5.0e-8(일반)·5.0e-7(matched), HCC 단위 5.0e-6(v1 표가 6자리 반올림). 4 inner의 v1 상수·J 상수·§10 예산 상수 재현 차이 0 |
| anchor_matched (K16, ΔRMSE) | source OOF 0.54613 (shrunk_mean 0.54649), C0 0.53676, C1 0.53254; HCC 동등 평균 0.61613 (shrunk_mean 0.61445), C0 0.59503, C1 0.59047 |
| 분해 anchor_matched → C0 → C1 (K16 ΔRMSE) | source OOF −0.00937 [−0.01196, −0.00695] → −0.00421 [−0.00558, −0.00289]; source E1 −0.02662 → −0.00281; source E2 +0.00150 [−0.00151, +0.00441] → −0.00232 [−0.00379, −0.00098]; 확인 −0.00879 → −0.00316; HCC −0.02110 [−0.02540, −0.01630] → −0.00456 [−0.00657, −0.00275] |
| centred 동일성 | C0와 anchor_matched의 centred 예측 차이 최대 9.7e-7(float32), centred RMSE 차이 0. 반면 shrunk_mean − anchor_matched centred는 source OOF +0.00046 [+0.00007, +0.00092]로 **0이 아니다** (v1 문구 1의 오류 확인) |
| P0-B 키 | 불일치는 final_inner(및 이를 쓰는 global 평가)의 K = 8 하나뿐(예측 κ_f 4, V3·scale κ_f 8). inner0/1/2는 모든 K가 같은 κ_f |
| P0-B 영향 | K16·K32 행은 수정 전후 **정확히 0 차이**. source 재선택 후 바뀐 상수는 `kn_chem+f0` K8 λ 0.3 → 0.1 하나. 이미 노출된 패널의 K8 변화: J3 +0.00017(확인) / +0.00018(HCC 동등 평균), J3 0-overlap 최대 +0.00043, kn_* ≤ 2×10⁻⁴, C0_V3·C1_V3·P_* < 10⁻⁶ (`k8_panel_impact.tsv`, `correction_impact.md`) |
| P0-C tie | 반올림 전 inner 선택 376개(v1·v2 run, 모든 grid·K, 예산 grid 포함)에서 **최상 점수의 exact tie 0건**, 선택 변경 0건, 1·2위 최소 차이 1.6e-7 → 출력 재계산 불필요 (`tie_rule_impact.tsv`) |

**의미.** v1의 수치 결론은 바뀌지 않는다. 바뀌는 것은 설명이다: 공통 보정의 순수 효과는 같은 anchor 기준으로 source OOF −0.0094, HCC −0.0211이며, v1의 `shrunk_mean → C0` 수치(HCC −0.0194)에는 anchor 선택 차이(−0.0017)가 섞여 있었다. K8 정합성 수정의 영향은 보조 결과에서 ≤ 4×10⁻⁴이다.

**다음 행동.** 없음(정정 완료). 향후 선택 코드는 v2 tie 규칙과 K별 nuisance 키를 쓴다.

## 2. R8-E1 — 관측 환경 전이와 joint 구조의 추가 가치

**목적.** 공통 보정이 support–query 환경 공유에 민감하고 C1의 약물별 residual이 더 잘 전달되는지(H1), J3가 C1과 같은 metadata의 단순 기준선 C1J1을 함께 넘는지(H2) 확인한다.

**수정 내용 / 입력 계약.** v1 `*_unknown_env`를 `*_zero_overlap_fallback`으로 이름을 바로잡고(query DMSO x0_q는 계속 사용), planned / known_separate / missing(재선택·fallback) 계약을 분리했다(`INPUT_CONTRACTS.md`). 새 arm C1J1 = f* + b_q(J1) + r(C1)은 C1의 λ,η를 고정하고 a,c를 **복합 예측 자체의 inner 점수로** 골랐다(J1 단독 최적값 복사 아님). 민감도 C1J1_full은 a,c,λ,η를 함께 재선택(660 후보). 모든 상수는 계약별 source inner에서 골랐다(gen_planned / gen_missing / M1 / M2).

**실험 입력/정답.** outer0–2의 outer-test source-train family. 일반 episode G(91 family, 279 episode, seed 1–3)와 환경 보류 matched 설계: 같은 query·같은 support 약물·같은 정답에서 support를 query plate(M1, o_qS = 1) 또는 다른 X 반복 plate(M2, o_qS = 0)에서 가져온 paired episode(41 family, 43 세포주, 360 ordered pair × 3 seed = 1,080쌍; v1은 2 pair·seed 1). 적격 pair는 모두 한 compound plate의 X 반복(44 compound plate, 20개가 여러 family에 공유) → 다른 compound plate/날짜 stratum 없음. full-row 순열 불변성 최대 8.3e-7.

**비교 조건.** K = 16 주, ΔRMSE와 centred RMSE. Holm(주 검정 2개). 채택 규칙: 평균 < 0, 95% CI < 0, outer fold ≥ 2/3 같은 방향. 실용 기준 상대 0.5%.

**결과 (주).**

| 검정 | 수치 | 판정 |
|---|---|---|
| H2 (M2 known_separate) J3 − C1 | −0.00169 [−0.00270, −0.00089] [3/3] (−0.24%) | — |
| H2 J3 − C1J1 | −0.00072 [−0.00170, +0.00013] [2/3] (−0.10%) | **차이 미확립 → J3 채택 안 함** (p_IUT 0.101, Holm 불통과) |
| H2 centred J3 − C1(= C1J1) | −0.00046 [−0.00086, −0.00012] | (채택 규칙이 깨져 라벨에 쓰이지 않음) |
| H1 (Δc − Δr), M2 − M1 | +0.0250 [+0.0192, +0.0325], compound-plate cluster [+0.0198, +0.0286] | **지지** (p < 0.001, Holm 통과) |
| 공통 보정 C0 − anchor_matched | M1 −0.0285 [−0.0355, −0.0226]; M2 −0.0023 [−0.0041, −0.0006] (cluster [−0.0026, +0.0010]) | 환경 shift에서 이득의 ~92% 소실 |
| 약물별 보정 C1 − C0 | M1 −0.0030 [−0.0048, −0.0015]; M2 −0.0019 [−0.0031, −0.0009] [3/3] | shift 후에도 유지 (Δr +0.0011) |

**결과 (보조, K16 ΔRMSE).**
- K = 8 / 32 견고성(M2): J3 − C1J1 −0.00096 [−0.00153, −0.00040] / −0.00069 [−0.00176, +0.00035]; H1 (Δc − Δr) +0.0184 / +0.0290, C1 − C0 (M2) −0.0006 / −0.0030 — H1은 세 K 모두 같은 방향, J3 우위는 K에 따라 확립 여부가 바뀐다.
- G planned: J3 − C1 −0.00023 [−0.00080, +0.00037] (미확립), J3 − C1J1 −0.00130, C1J1 − C1 +0.00107 [+0.00060, +0.00161] (일반 episode에서는 C1J1이 C1보다 나쁨), J2 − J1 −0.00166 [3/3], J2 − C1 +0.00243.
- M1 planned: J3가 C1(−0.00081)과 C1J1(−0.00113)을 모두 넘고 centred도 CI < 0 — 같은 plate에서만, 0.5% 미만.
- M2 known_separate: C1J1 − C1 −0.00097 [−0.00145, −0.00051] [3/3], J1 − C0 −0.00105 — 별도 plate를 **아는** 것은 공통항의 크기를 줄이는 데 쓰인다(M2 κ_b = 64 등 재선택).
- missing 계약: G에서 J3 − C1 +0.00144 [+0.00057, +0.00231](J3가 나쁨); M1 missing에서 J3 − C1 +0.0031(재선택) / +0.0102(0-overlap fallback). source에서 missing 계약으로 재선택하면 fallback보다 J3가 M1에서 −0.0071 좋아지지만 C1을 넘지 못한다.
- 고정 상수 민감도: 일반 episode 상수를 M1/M2에 복사하면 C1이 M2에서 +0.0038, C1J1이 +0.0088 나빠진다 — 계약별 재선택 없이 정보가 다른 계약을 비교하면 결론이 바뀐다.

**의미.** (관찰) 공통 보정의 이득은 support가 query와 같은 plate일 때 대부분 생기고 다른 X 반복 plate에서는 거의 사라지는 반면, 약물별 residual 항은 두 경우 모두 유지된다. (관찰) joint 모형 J3는 환경 shift 조건에서 C1보다는 낮았지만 같은 metadata의 단순 공통항 교체(C1J1)와의 차이는 확립되지 않았다. (해석·가설) "query가 별도 plate임을 아는 것"의 가치는 대부분 공통항 축소로 얻을 수 있다. **검증하지 않은 범위**: 독립 배양·날짜·compound plate, 외부 패널.

**다음 행동(사전 규칙 적용).** J3 채택 안 함 → joint 모형 확대 보류, 단순 환경 보정(C1J1) 우선; 같은 plate 관측값 예측에서만 J3가 앞섬; unknown·별도 환경의 기본은 C1. 독립 외부 패널 검증은 진입 조건(M2에서 의미 있는 ΔRMSE·centred 개선)이 충족되지 않아 해당 없음.

## 3. R8-E2-A — 같은 anchor에서 kernel의 추가 정보 가치 (E1 episode 재사용)

**목적.** 같은 f*·κ_f·support/query에서 kernel block(chem, f0, ctrl)의 정보 가치가 환경 shift(M2)에서도 유지되는지 확인한다.

**비교 조건.** C0(η = 0), kn_ctrl, kn_chem+ctrl, kn_f0+ctrl, kn_all; block 정규화 scale(K별 κ_f, source-train), 각 arm의 λ,η는 같은 후보 예산(η = 0 포함)으로 계약별 inner에서 선택. 대조: residual-only shuffle 3개, support f0 표현의 n0 층화 순열(anchor 고정).

**결과 (K16, centred RMSE; 괄호는 ΔRMSE).**

| 대비 | M2 known_separate (주) | M1 planned | G planned |
|---|---|---|---|
| f0 추가 (kn_all − kn_chem+ctrl) | −0.00250 [−0.00380, −0.00153] (−0.00182) | −0.00299 | −0.00333 |
| chem 추가 (kn_all − kn_f0+ctrl) | −0.00018 [−0.00040, +0.00002] (−0.00006) | −0.00001 [−0.00018, +0.00015] | **−0.00041 [−0.00068, −0.00014]** |
| ctrl 단독 − C0 | 0 (한 plate 안에서 상수) | 0 | −0.00108 |
| f0 층화 순열 − kn_all | +0.00366 [+0.00229, +0.00553] | +0.00484 | +0.00504 |
| residual shuffle − kn_all | +0.0042 ~ +0.0045 | +0.0055 ~ +0.0058 | +0.0092 ~ +0.0097 |

**의미.** 사전 라벨 **"same-drug source-response-assisted transfer"** 충족(M2): 환경이 바뀐 조건에서 약물별 보정의 정보는 다른 세포주에서 측정한 같은 약물 반응(f0) 공간에서 오며, f0의 약물 대응을 끊으면(anchor는 그대로) 보정이 오히려 해가 된다. 화학 구조의 추가 기여는 M2·M1에서 확립되지 않았고, 일반 episode에서만 작게(−0.0004) 확인된다. 다른 구조 표현 전체로 일반화하지 않는다.

**다음 행동.** E2-B 진입 조건 (b) 충족 → §4.

## 4. R8-E2-B — 동일 약물 source 반응이 없을 때 (조건부, 진입)

**진입 근거.** 사전 조건 (a)는 거짓(HCC 9개 target×split 모두 동일 약물 source 반응이 없는 query 0개). 조건 (b)는 참(§3: M2에서 f0 의존성과 f0 대응 사용 확인). 사양 `e2b_spec.yaml`을 E2-B 숫자 이전에 커밋했다(F3 = `84dbfa3`; 코드만 고친 amendment `6dfac2b`: 첫 sru job 2개가 inner 선택 중 속성 오류로 멈춤, 점수 없음).

### 4.1 입력 채널 민감도 (E1 episode, 미관측 약물 일반화가 **아님**)

query 반응이 PCA·nuisance 적합에 이미 들어 있으므로 입력 의존성 설명으로만 쓴다. mask별로 κ_b, λ, η를 같은 예산으로 재선택(K16, centred RMSE; mask − full).

| mask | G | M1 | M2 |
|---|---|---|---|
| f0 kernel 제거 (anchor 유지) | +0.0033 [+0.0024, +0.0043] | +0.0030 | +0.0025 [+0.0015, +0.0038] |
| anchor 제거 (f0 kernel 유지) | +0.0236 [+0.0194, +0.0284] | +0.0113 | +0.0113 [+0.0078, +0.0149] |
| 둘 다 제거 | +0.0323 [+0.0261, +0.0392] | +0.0190 | +0.0184 [+0.0122, +0.0256] |
| 둘 다 제거한 kernel(chem+ctrl) − anchor 없는 C0 | −0.0019 [−0.0026, −0.0011] | −0.0003 [−0.0009, +0.0003] | −0.0004 [−0.0010, +0.0002] |

같은 약물 source 반응은 주로 **anchor**를 통해 쓰이고(제거 시 +0.011 ~ +0.024), kernel의 f0 채널은 그 다음이다. 두 채널을 모두 끄면 화학+ctrl kernel의 남은 기여는 일반 episode에서만 확립되고(−0.0019) 환경 shift(M2)에서는 미확립이다.

### 4.2 source-reference-unavailable 검증 (일반 episode)

**입력/정답.** canonical 약물 group을 3개 drug fold로 고정 분할(crc32). fold j의 query 약물은 source f0 표·f0 PCA basis·kernel scale·ridge 사상·모든 선택(inner partition 포함)에서 제외했다. support 약물의 f0는 유지(다른 팔과 동일), query control 관측. 적격 규칙(query ≥ 10, support ≥ 16, outer fold×drug fold마다 family ≥ 10)은 9개 조합 모두 충족(16–22 family, `e2b_sru_eligibility.tsv`); 점수는 71 family. covariance는 어떤 E2-B 팔도 쓰지 않는다.

| 방법 (K16) | ΔRMSE | centred RMSE |
|---|---:|---:|
| fb_C0 (f0_q = 0) | 0.56769 | 0.54352 |
| fb_C1chem | 0.56623 | 0.54220 |
| ridge_C0 (ECFP ridge anchor) | 0.56269 | 0.54244 |
| ridge_C1chem | 0.56126 | 0.54113 |
| ref_C0 (full f0; 정보가 더 많은 참조) | 0.53589 | 0.51669 |
| ref_C1 (full f0) | 0.53272 | 0.51341 |

| 대비 (K16, centred RMSE) | 차이 [95% CI] |
|---|---|
| fb_C1chem − fb_C0 | −0.00132 [−0.00241, −0.00054] [3/3] |
| ridge_C0 − fb_C0 | −0.00108 [−0.00154, −0.00063] [3/3] (ΔRMSE −0.00500) |
| ridge_C1chem − fb_C0 | −0.00240 [−0.00353, −0.00149] [3/3] (ΔRMSE −0.00643, 상대 −1.1%) |
| ref_C1 − ridge_C1chem (입력 계약 차이) | −0.02772 [−0.03513, −0.02064] |

**의미.** 사전 라벨 **"chemistry_transfer_without_same_drug_response" 충족**: 같은 약물의 source 반응이 전혀 없을 때도 화학 구조(ECFP ridge anchor + chem/ctrl kernel)가 fallback보다 작게 개선한다. 그러나 같은 약물 source 반응을 쓰는 계약(ref)과의 차이(−0.028)가 화학 경로의 이득(−0.0024)보다 10배 이상 크다. 두 계약은 입력이 다르므로 이 차이를 아키텍처 우열로 부르지 않는다. 다른 구조 표현이나 "화학은 일반적으로 안 된다"는 주장으로 확장하지 않는다. matched plate에서는 drug hold-out 뒤 표본이 희소해 실행하지 않았다(사양대로 일반 episode만).

**다음 행동.** 실제 목표에 같은 약물 source 반응이 없는 query가 포함될 때의 기준선은 ridge_C1chem으로 둘 수 있다(이번 결과는 source 내부, 이미 노출된 자료). 외부 확인은 그런 query가 있는 새 자료가 생길 때.

## 5. HCC 동결 후속 표 (이미 노출된 벤치마크, 한 번 계산)

source 결정과 final_inner 상수를 F2 `0be3577` (코드 `84dbfa3`)로 동결한 뒤 한 번 계산했다(job 27230, 79 s). HCC 값으로 어떤 후보도 바꾸지 않았다. 세 계열 동등 평균, test query, K = 16, 약물 ID 동기화 bootstrap.

| 방법 (planned) | ΔRMSE | | 대비 | 차이 [95% CI] |
|---|---:|---|---|---|
| anchor_matched | 0.61613 | | J3 − C1 | −0.00260 [−0.00404, −0.00105] (계열 −0.0069 / −0.0006 / −0.0003) |
| C0 | 0.59503 | | J3 − C1J1 | −0.00260 [−0.00404, −0.00105] |
| C1 = C1J1 = C1J1_full | 0.59047 | | J3 − C1 centred | −0.00022 [−0.00066, +0.00024] |
| J2 | 0.59083 | | C1 − C0 | −0.00456 [−0.00657, −0.00275] |
| J3 | 0.58786 | | J2 − C1 | +0.00036 [−0.00145, +0.00233] |

- final_inner에서 C1J1의 선택값(a = 0.5, c = 0)은 C1의 공통항(κ_b = 16, K16 → α = 0.5)과 같아서 HCC에서 C1J1 ≡ C1이다(J1 ≡ C0도 같은 이유).
- missing 계약: J3 재선택 − C1 −0.00082 [−0.00158, −0.00006]; 0-overlap fallback − C1 +0.00202 [+0.00098, +0.00301]. K8/K32의 J3 − C1: −0.00289 / −0.00206.
- **해석의 범위.** HCC의 J3 우위(−0.0026, centred 미확립)는 source 주 검정(M2에서 C1J1 대비 미확립)과 같은 질문이 아니며 채택을 바꾸지 않는다. 이 패널은 R8·v1에서 여러 번 채점되었다.

## 6. 미실행 기록

`NOT_RUN.md`: R8-E3(실제 다음 측정 결정 없음; unlock = 획득 계획과 비용 단위·상한 확정), query control 미관측 계약, label-free overlap prior, 다른 compound plate/날짜 stratum(자료 없음), 독립 외부 패널(자료 없음).

## 7. 자원 기록·편차

| 항목 | 값 |
|---|---|
| Slurm job | 36개 (`manifests/r8_transfer_v2/slurm_jobs.tsv`): 완료 28, 실패 6(e1 select 4: 사양 YAML 문법; sru 2: episode 속성 오류 — 모두 점수 계산 전), report dry-run 2 (1개는 sh 문법으로 즉시 실패) |
| CPU | 예약 core-시간 약 51 (8코어 × 경과), 실제 user+sys CPU 약 49시간. 최대 메모리 8.4 GB (sru) |
| GPU | 0 |
| 디스크 | 서버 `assets/` 28 MB (상한 5 GB). 공유 파일시스템 여유 166 GB 유지 |
| 동시 job | 최대 6 (track 상한 6) — 단, 11:46에 `r8v2_` job 7개가 잠시 대기열에 있었다(곧 4개가 YAML 오류로 3초 만에 종료) |

**편차·사건 (모두 기록, 결과 해석에 영향 없음).**
1. 사양 F1(`1b59b76`)의 YAML 문법 오류 2곳 → 따옴표만 추가(`6a66d0b`, amendment 기록). 실패한 run 디렉터리 `e1sel_*_v1`은 비어 있으며 새 run ID `_v2` 사용.
2. E2-B sru 첫 실행(`e2bsru_outer{0,1}_v1`, job 27234/27235)이 inner 선택 중 `o_qs` 속성 오류로 중단 → 코드만 수정(`6dfac2b`, amendment), 새 run ID `_v2`.
3. `p0k8_global_v1`(job 26844)은 `1b59b76`에서 시작했고 실행 중 서버 HEAD가 `6a66d0b`(YAML만 변경)로 바뀌어 result.json의 commit 필드가 시작 commit과 다르다. 로그 start 줄의 commit이 실제 코드다.
4. F2 manifest(`e1_source_freeze.json`)는 로그인 노드에서 `freeze.py`로 썼다(JSON 한 개, 계산 없음).
5. E1 report 코드를 전체 실행 전에 outer1/2 두 fold로 dry-run했다(scratch 디렉터리, 삭제). 주 대비·임계값은 F1에서 고정된 그대로이며 이 dry-run 이후 바뀐 것은 없다.
6. job 27259/27260 로그 start 줄의 `dirty=2`는 다른 노드의 NFS 상태로 보이며, 같은 job의 result.json `dirty_tree`는 false다.

**미실행.** R8-E3(진입 조건 없음), query control 미관측 계약, label-free overlap prior, 다른 날짜/배양 stratum, 독립 외부 패널 — `NOT_RUN.md`.

