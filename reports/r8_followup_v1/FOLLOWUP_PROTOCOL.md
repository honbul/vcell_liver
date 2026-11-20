# R8 후속 분석 v1 — 실행 프로토콜 (FOLLOWUP_PROTOCOL)

**작성일:** 2026-11-18 · 기준 커밋 `5530acb` · 설정 파일 `configs/r8_followup_v1/followup.yaml`
비공개 기획서(817줄, SHA-256 `b89ea16e…361b56d`)를 따르며, 기획서 본문은 Git에 넣지 않는다. 버전·노출 이력은 `VERSION_AND_EXPOSURE.md`에 있다.

## 1. 질문

새 세포주에서 K개 약물(support)의 반응을 측정했을 때 다른 약물(query)의 978 landmark ΔRNA를 얼마나 예측할 수 있는가. 그중 (1) 모든 query에 같은 **공통 보정**, (2) 약물마다 다른 **약물별 잔차 전달**, (3) **같은 측정 환경(plate)을 공유해서 얻는 이득**을 구분한다.

예측 = f*(축소 동일 약물 평균) + b(공통 보정, C0) + r(약물별 보정, C1). 

## 2. 필수 단계 (모두 수행)

| 단계 | 내용 | 산출물 |
|---|---|---|
| P0 | 기준 버전·자료 hash·노출 기록 | `VERSION_AND_EXPOSURE.md`, 이 문서 |
| P1 | 공분산 V0 → V1 → V2 → V3 (기획서 §4.3 그대로). V0는 원 수치를 반올림 범위에서 재현해야 한다. floor 사용·음수 잔여값 비율·fold별 ≥30 쌍/≥10 family 규칙 기록 | `covariance_repair_versions.tsv` |
| P1 | C0 네 경로(iid / weights-only / shrink-only / both), C1-iid 대 C1-V3; fixed-hyper와 source 재선택을 분리 | `covariance_effect_paths.tsv`, `fixed_hyper_vs_retuned.tsv` |
| P1 | 같은 plate(E1) 대 다른 반복 plate(E2), 같은 query 정답·support 약물 목록 | `matched_plate_repaired.tsv` |
| P2 | f*, b, r의 centred 분해; C1 직접 잔차 shuffle(사전 고정 순열 3개) + 전체 행 순서 불변 검사; 영향계수·LOSO; n0별 오차; reference 반분 민감도; 반복 양성 대조 | `residual_components.tsv`, `matched_plate_repaired.tsv` |
| P2 | C1 kernel 구성 6종, 원 scale(민감도)과 block 정규화 scale(정보 가치, 같은 λ·η 예산) 분리 | `kernel_components.tsv` |
| P2 | 기존 B01/B11 checkpoint의 FiLM 출력 경로 민감도 (재학습 없음) | `state_path_usage.tsv` |
| P2 | label 없는 source–HCC 입력 범위 비교 | `source_target_input_coverage.tsv` |

## 3. 입력 계약

- 모든 방법이 받는 것: query 약물의 source family 평균 반응 f0와 그 family 수 n0, support K개의 측정 ΔRNA, 약물 구조(ECFP4), 대응 DMSO 대조군 발현, support의 plate overlap·row weight(메타데이터).
- **query의 처리 후 RNA는 입력·정규화·가중·상수 선택 어디에도 쓰지 않는다.** 점수 계산(`fmethods.score`)에서만 읽는다.
- 새 분기 §8(J1/J2/J3)에서만 추가로 쓰는 것: **query–support plate overlap** `o_qS = W_q W_S'` (사전에 계획된 detection plate 배치와 고정 집계 가중치; 처리 후 QC로 정한 가중치가 아님). 별도 입력 계약 문서(`query_overlap_contract.md`)는 그 분기를 실행할 때 작성한다. query plate를 모르는 경우의 **알 수 없는 새 환경 모드**(o_qS = 0)를 따로 보고한다.
- 같은 metadata를 받는 단순 기준선(J1)을 반드시 함께 계산해 정보 추가의 이득과 구조의 이득을 구분한다. joint 모형은 기존 b_S를 **대체**하며 그 위에 더하지 않는다.

## 4. 분할과 선택

`VERSION_AND_EXPOSURE.md` §4 참조. 요약: family 단위 3-fold outer, outer-train 안의 고정 inner split(crc32 규칙)으로 상수를 고르고, outer-train으로 nuisance를 다시 적합해 outer-test family를 평가한다. 원 상수를 그대로 쓴 행은 "fixed-hyper, source 개발 분석"이다.

평가 episode: outer-test 세포주마다 support seed 1/2/3, query 상한 없음(원 확인 episode 규칙). 선택 episode: inner-validation 세포주마다 seed 1, query 상한 200(원 dev 규칙). 같은/다른 plate: outer-test family의 적격 세포주마다 plate 쌍 2개, A↔B 두 방향, seed 1(원 train 규칙).

## 5. 지표와 통계

- **주 지표:** K=16, 978 유전자 unit ΔRMSE, 원 집계 순서(query 평균 → support seed → family 안 세포주 → family 동등 평균). E1/E2는 별도 표.
- 보조: centred RMSE / centred Pearson / ΔPearson(정의되지 않으면 NA와 개수), 크기 비율, retrieval. retrieval은 **동률 보정 기대 reciprocal rank**(정답보다 가까운 후보 a개, 정답 포함 동률 t개일 때 (1/t)Σ1/(a+j))와 Top-k = min(max(k−a,0),t)/t를 쓰고, 원 mid-rank 규칙 값은 `legacy`로 따로 둔다. 무작위 기대값 H_m/m를 함께 적는다.
- **불확실성:** source·matched plate는 family cluster bootstrap(2000회, seed 20261117; family 안의 세포주·seed·plate 쌍·A↔B를 함께 둠). HCC는 세 계열·분할의 query 약물 ID 합집합에 대해 **약물 ID별 가중치를 동기화한** paired bootstrap. 계열별 조건부 bootstrap을 따로 낼 때는 `SeedSequence` 독립 stream을 쓴다.
- **운영 기준:** 상대 ΔRMSE 0.5%. 채택 후보 = source OOF 평균 개선 + 95% CI가 0 아래 + 3개 outer fold 중 2개 이상 같은 방향. 0.5% 기준 충족 여부는 따로 표시한다. CI가 0을 포함하면 "개선 미확인". **동등성**은 90% CI가 ±0.5% 안에 들 때만 말한다.
- 같은 표의 여러 대조는 탐색 분석이다. p-value를 독립 확증으로 읽지 않는다.

## 6. 분기 규칙 (기획서 §7)

P1·P2의 **source OOF 결과만**으로 §7 표의 코드(D1–D9)를 판정하고, 실행 전에 `BRANCH_DECISION.md`(관측 → 해석 → 선택 → 미선택 이유, 종료 조건)를 커밋한다. 새 분기는 최대 두 개: §8 joint 모형(J0/J1/J2, J3는 근거가 있을 때만) + §9.1 / §9.2 / §10 중 최대 하나. 근거가 약하면 닫힌식 분석으로 마무리한다. 실행하지 않은 분기는 `NOT_RUN.md`에 이유와 함께 남기고 숫자를 지어내지 않는다.

## 7. 동결과 기존 패널 평가

source에서 고른 후보 1개를 상수·코드와 함께 동결 커밋한 뒤, 확인 9 family(같은/다른 plate 5 family 포함)와 HCC 세 계열에 동결 후보와 기준선을 **한 번** 적용한다. HCC에서 ρ, scale, gate, 학습률을 고르지 않는다. 버그 수정으로 재평가하면 사유·파일·전후 값을 모두 남긴다.

## 8. 계산 규칙

Slurm만 사용(partition `cpu`, `gpu`), job 이름 접두사 `r8f_`, 불변 run ID, 다른 사용자·다른 연구(r6f_/r7f_/r7_) job은 건드리지 않는다. 닫힌식 분석은 CPU. GPU는 동시에 1개 이하이며, 선택된 분기가 신경망 재학습을 요구할 때만 쓴다.

## 9. 변경 기록

| 날짜 | 변경 | 사유 |
|---|---|---|
| 2026-11-18 | 작성 | — |
| 2026-11-18 | J 모형을 §8 분기 전용으로 분리 (`4c2da64`) | 분기 결정 전 계산 방지 (개발 실행에서 J1/J2 값을 본 사실은 `BRANCH_DECISION.md` §7에 공개) |
| 2026-11-18 | 분기 결정: §8(J0–J3) + §10 (`e606c21`) | D2/D3/D4/D7 |
| 2026-11-18 | branch 평가 재실행 `_v2` (`5ecbcb8`) | 키 중복 버그로 `_v1` 실패 |
| 2026-11-18 | 동결 후보 J2 (`73be6d3`) | 사전 규칙: 채택된 새 모형 중 OOF 최저 |
