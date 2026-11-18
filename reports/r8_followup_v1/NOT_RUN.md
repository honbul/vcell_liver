# R8 후속 분석 v1 — 실행하지 않은 것 (NOT_RUN)

**작성일:** 2026-11-18 · 아래 항목에는 결과 숫자가 **없다.** 빈 칸을 성공·실패로 읽지 않는다.

| 항목 (기획서 위치) | 실행 여부 | 이유 |
|---|---|---|
| §9.1 고정/학습 kernel 혼합 (ρ ∈ {0,.25,.5,.75,1}) | 미실행 | 진입 조건 D5를 source OOF로 판정할 수 없었다(원 교차검증이 fold별 B00 checkpoint를 남기지 않음). 엄격한 OOF에는 fold별·inner별 B00 재학습(GPU)이 필요하고, §9/§10 중 하나만 허용되는데 D7(§10)은 직접 측정된 근거가 있었다 (`BRANCH_DECISION.md` §5) |
| §9.2 episode 구성 변경 (matched cross-plate sampler) | 미실행 | D6 근거 없음 (OOF learned kernel 결과가 없고, 같은 plate 편향을 보일 자료가 없음) |
| B10 / B11 재학습 (수정 covariance로 학습) | 미실행 | D1 불충족. 닫힌식에서 V3가 iid보다 좋지 않았고(D4) 학습 상호작용 가설이 없다. 기존 checkpoint에 Ω만 바꾼 민감도도 하지 않았다(V3 ≈ V0로 Ω 차이가 1e-6 수준) |
| state-free / state-aware 재학습 (FiLM 정보 가치) | 미실행 | 기획서 §6.3의 기본 범위는 기존 checkpoint 민감도까지이며, 재학습은 선택 분기 예산에 속한다. 분기로 선택하지 않았다 |
| B00 / 학습 모형의 OOF E1/E2, OOF 잔차 shuffle | 미실행 | fold별 checkpoint가 없다. 기존 B00 shuffle 결과(+0.0119, R8)는 C1의 직접 증거로 쓰지 않았고, C1 shuffle은 직접 수행했다 |
| GPU 사용 | 0 GPU-시간 | 선택된 분기(§8, §10)가 모두 닫힌식이라 GPU가 필요 없었다 |
| §10 보조 설계 (A/B 1:1 평균 query) | 미실행 | 사전 규칙상 주 설계(세 번째 plate C를 query로 고정)에 OOF 적격 family가 10개 이상(32 family)이어서 필요 없었다 |
| §10 coverage 기반 support 선택 후보 | 미실행 | 무작위 배분의 결과가 먼저였고, D16·REP8 차이가 상수 선택에 따라 부호가 바뀌어 추가 후보를 둘 근거가 없었다 |
| §11.4 "공통 query core 고정, support만 변경" 보조 비교 | 미실행 | 조건부 항목("필요하면 1회")이다. 주 결론이 support 선택 변동에 좌우되지 않아 실행하지 않았다. support seed는 1/2/3을 평균하고 family bootstrap으로 보고했다 |
| §5.3 특정 화학군 집중 | 대체 | 화학군 메타데이터가 없어 "query별 Tanimoto 상위 3 support의 영향 비중"으로 대체했다 (`residual_components.tsv`의 `top3_chem_neighbour_share`) |
| J3의 유전자별(978개) 독립 상수 | 미실행 | 기획서 §8.4대로 공유 scalar로 시작했고, J3가 C1-iid를 넘지 못해 확대하지 않았다 |
| HCC·확인 family에서의 상수·ρ·gate·학습률 선택 | 금지 | 규칙상 하지 않는다. 동결 후 한 번만 적용했다 |
| 독립 배양·다른 날짜 반복 plate | 자료 없음 | 적격 plate 쌍은 모두 한 compound plate의 X 반복이다. 독립 배양 일반화는 검증하지 않았다 |
| 잠재 생물학적 평균 L (기획서 §3.2) | 식별 불가 | 현재 자료로 직접 관측되지 않는다. E2를 L로 부르지 않는다 |

## 실패·취소한 작업

| job | 이름 | 상태 | 원인과 조치 |
|---|---|---|---|
| 26129, 26130, 26131 | `r8f_beval` (`beval_outer*_v1`) | 실패 | matched-plate 행에서 `n_query` 키가 메타데이터와 점수 dict에 중복(`TypeError`). 일반 episode 부분은 끝났으나 결과 파일을 쓰기 전에 중단. 커밋 `5ecbcb8`에서 수정 후 새 run ID `beval_outer*_v2`로 다시 실행 |
| 26132 | `r8f_brep` (`brep_v1`) | 취소 (pending, 의존성 미충족) | 위 실패로 시작되지 않음. `brep_v2`(job 26188)로 대체 |
| 26201, 26211 | `r8f_finrep` (동결 후 요약) | 실패 | 요약 코드의 대조 label 누락, 이어서 all-NaN 열 제거 문제. 요약 코드만 고치고(커밋 `7075549`, `9157d68`) job 26212로 다시 요약. 예측(`final_v1`, job 26200)은 한 번만 계산했고 다시 계산하지 않았다 |

개발 점검 실행(`*_dev1`, job 26022–26063)은 코드 점검용이며 공식 결과로 쓰지 않았다. 공식 P1/P2 결과는 그와 같은 숫자를 다시 냈다(같은 결정론적 계산).
