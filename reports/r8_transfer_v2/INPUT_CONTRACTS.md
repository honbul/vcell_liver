# R8 v2 — 입력 계약 (E1/E2-A)

**작성일:** 2026-11-19 · 근거: 비공개 기획서(2026-11-19) §2.1, §5.2 · v1 계약 `reports/r8_followup_v1/query_overlap_contract.md`를 대체하지 않고 이름과 의미를 바로잡는다.

## 1. v1 `*_unknown_env` 이름 정정

v1의 `J1_unknown_env`, `J2_unknown_env`, `J3_unknown_env`는 **o_qS = 0을 넣었지만 query의 DMSO 대조군 x0_q는 계속 사용했다.** 따라서 "query 환경 정보가 전혀 없음"이 아니다. v2에서는 이 행을 `*_zero_overlap_fallback`이라고 부른다: plate metadata가 없을 때 "query가 어느 support plate와도 겹치지 않는다"고 **가정한** fallback이며, 상수는 o_qS를 아는 계약(planned)에서 고른 값을 그대로 쓴다. v1 표의 숫자 자체는 바뀌지 않는다.

## 2. 계약

| 계약 | query–support plate overlap o_qS | query DMSO 대조군 x0_q | 상수 선택 (source inner) |
|---|---|---|---|
| planned (계획된 환경) | 실제 값 (계획된 plate 배치와 고정 집계 가중치) | 관측 | 일반 inner episode, 실제 o_qS (`gen_planned`); matched 같은 plate 평가에는 inner matched 같은 plate episode (`M1`) |
| known_separate (알려진 별도 plate) | 0 — **사실로서** 알려짐 | 관측 | inner matched 다른 X 반복 plate episode, o_qS = 0 (`M2`) |
| missing — 재선택 (`missing_reselected`) | 전달하지 않음 (방법에는 0으로 입력) | 관측 | 일반 inner episode에서 **선택할 때도** o_qS = 0 (`gen_missing`) |
| missing — fallback (`missing_zero_overlap_fallback`) | 전달하지 않음 (0으로 대체) | 관측 | planned 상수 그대로 (공유되지 않는 환경을 가정한 fallback) |
| query control 미관측 | 전달하지 않음 | 전달하지 않음 | **미실행**: 이 트랙의 사용 시나리오에 필요하지 않다 (`NOT_RUN.md`) |

- C0, C1, anchor는 o_qS를 쓰지 않으므로 planned와 missing에서 같다. 같은 평가 episode에서는 모든 방법이 같은 query·support·정답을 쓴다.
- missing 계약에서 방법은 query가 같은 plate인지 다른 plate인지 모른다. 그래서 M1/M2 평가에도 같은 `gen_missing` 상수를 쓴다.
- 고정 상수 민감도: `fixed_gen_constants` = 일반 episode에서 고른 상수를 matched M1(o_qS = 1)/M2(o_qS = 0)에 그대로 적용한 행. 정보가 다른 계약의 상수를 복사해 성능 저하를 선언하지 않기 위해 주 비교는 계약별 재선택 행이다.
- source의 label-free plate 설계로 정한 overlap prior 후보는 만들지 않았다 (`NOT_RUN.md`).

## 3. 허용·불허 정보 (v1과 같음)

허용: support의 측정 반응 y_S(K개), 같은 약물의 source 반응 f0(다른 세포주, family 단위 평균), 화학 구조(ECFP), support·query의 DMSO 대조군 profile, 계약에 명시한 plate 배치 metadata. 불허: query 처리 후 발현, query 정답의 평균·크기, 처리 후 QC로 정한 가중치, HCC·확인 family의 값으로 고른 상수.
