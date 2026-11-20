# R10 v3 — 독립 cross 확인의 자료 차단 사유와 확보 요건 (P1-R10 미실행)

상태: **자료 대기.** 근거: private plan 2026-11-20 §3.2·§3.4, 판정 규칙 freeze `2b91274`([`RUN_SPEC.md`](RUN_SPEC.md)). 후보 3개의 조건별 판정은 [`data_eligibility.tsv`](data_eligibility.tsv). 발현 행렬은 내려받지 않았고 어떤 새 자료도 채점하지 않았다. 기존 Sci-Plex3 test(노출 자료)와 R9 환자 자료는 대체로 쓰지 않았다.

## 1. 확인한 후보와 막힌 조건

| 후보 | 위치 | 막힌 필수 조건 | 요점([관측]) |
|---|---|---|---|
| K1 Sci-Plex2 (GSM4150377) | 공개 GEO | C2 fail, C4·C6 partial | A549 1개 세포주, 약물 4개(R10 split에 없는 약물), 0.1–100 µM·24 h. 약물×용량당 6 well이 hash-plate 접두 A/B에 3/3. 배양 날짜·생물학적 replicate 표시 없음, library 1개. 단일 hash라 v1의 W/P 두-hash QC 규칙을 그대로 쓸 수 없고, QC 통과 세포 median UMI 4,438(source A549 1,298). E1000 용량 72 well 중 4개가 단일-hash QC 유사 규칙에서 20세포 미만. |
| K2 Sci-Plex4 (GSM4150379) | 공개 GEO | C2 fail, C4 fail | A549·MCF7, HDACi 2개(abexinostat, pracinostat; 둘 다 R10 **train** 약물). 보충제 없는 용량 계열은 세포주별로 plate 하나(A549 plate10, MCF7 plate5)에만 있어 cross query가 없음. 다른 plate의 1 µM은 inhibitor arm(dose 0)과 함께 처리된 pracinostat뿐. |
| K3 Tahoe-100M | 서버 보유 | C3 fail, C2 fail, C6 fail/미검증 | 용량 0.05·0.5·5 µM뿐(100–1,000–10,000 nM·10–100–1,000 nM 없음). 용량이 plate와 거의 완전히 결합(14 plate 중 13개가 96 sample 중 ≥ 90개를 한 용량으로 채움). drug×dose 1,137 조건 중 975개는 sample 1개. 플랫폼·유전자 공간 다름. K562 없음. |

[추론] 세 후보 모두 기존 자산과 같은 계열(같은 GEO series 또는 서버 보유 대형 screen)이지만, "새 독립 배양에서 endpoint와 다른 replicate·plate의 중간 용량 query"라는 계약을 문서화된 형태로 제공하지 않는다.

## 2. 필요한 자료(최소 계약)

1. **세포주:** A549·K562·MCF7 중 하나 이상(기존 line-specific atlas를 재적합 없이 적용). 새 HCC 세포주는 별도 연구.
2. **조건:** 24 h, 각 약물에 100 nM·1 µM·10 µM(주 episode E1000)과 vehicle; 10 nM은 있으면 보조(E100).
3. **독립성 기록:** 배양 날짜(또는 passage/thaw 단위)·생물학적 replicate·culture plate·sequencing run이 세포 또는 well 수준에서 식별될 것. plate만 다른 반복은 생물학적 반복으로 세지 않는다.
4. **배치 설계:** 같은 batch 안에서 endpoint(저·고 용량)와 중간 용량 query가 서로 다른 replicate·plate에 놓이는 균형 배치(용량이 plate와 완전히 결합되지 않을 것). endpoint replicate의 vehicle well 포함.
5. **원자료:** UMI count 행렬(QC 전) + well/hash 배정 원표 + 세포 metadata. Sci-Plex3와 같은 hash QC를 적용하려면 두-hash(well×plate) 설계 또는 동등한 품질 지표가 필요하다(아니면 QC 규칙을 사전 지정해 별도 계약으로 둔다).
6. **약물:** R10 split의 train·development 약물과 겹치지 않는 약물을 주 평가에 쓰는 것이 원래 test 과제(관측 endpoint 기반 보간, drug zero-shot 아님)와 맞다. train 약물은 보조 층으로만.
7. **규모(정밀도 검토, [`precision_review.tsv`](precision_review.tsv)):** source P6의 약물 단위 변동으로 근사(between-batch 변동 미포함 → **낙관적 하한**).
   - 세 세포주 동일 가중, 관측 효과(−0.0102, test·노출 자료) 기준 80% 검정력: **약 19 약물**; 실용 기준 0.005를 효과로 두면 약 79 약물.
   - A549 단독: test 기반 약 25 약물, development 기반 약 123 약물(관측 효과); 0.005 기준 약 317–482 약물.
   - 참고: K1(A549, 4 약물)이라면 95% 구간 반폭 ≈ 0.031–0.038(정규 근사; 4 단위 bootstrap/t 분포면 더 넓음), 검정력 ≈ 0.07–0.20.
   - batch 일반화를 주장하려면 위 약물 수를 **여러 독립 batch(목표 3개 이상; 검정력 보장 숫자 아님)**에 걸쳐 반복해야 한다. 새 batch 하나는 그 batch에서의 재현만 확인한다.
   - well당 QC 세포 ≥ 20(v1 적격 규칙), vehicle ≥ 50.

## 3. 확보 경로와 예상 비용

| 경로 | 내용 | 비용·조건 | 결정 주체 |
|---|---|---|---|
| R1 신규 실험(권장 경로) | 위 §2 계약으로 A549/K562/MCF7 24 h 용량 계열을 독립 배양 batch ≥ 3에서 생성(sci-Plex류 hash 다중화 또는 동등 fixed-cell 다중화). 예: 3 세포주 × 20 약물 × 3 용량 × 2 plate 위치 + vehicle ≈ batch당 ~400 well, 3 batch ≈ ~1,200 well, well당 100–150 세포 ≈ 12–18만 세포 | 금액 견적은 받지 않았다(외부 접촉·유료 서비스 금지). 비용 결정 요인: well·세포 수, library 방식, 세포당 sequencing read 수, 약물 구입. [미검증] | 사용자(예산·wet-lab 협력) |
| R2 원저자 metadata 문의 | Sci-Plex2/3의 배양 날짜·hash-plate=culture-plate 여부 확인 | 외부 연락 필요(이번 사이클 금지). 확인되더라도 K1은 1 batch·4 약물·A549라 **외부 예비 검증** 수준을 넘지 못함(§2.7) | 사용자 |
| R3 조건 완화 후 K1 예비 실행 | C2를 사용자가 면제하고 K1을 "외부 예비 검증(단일 batch, plate-level cross, 4 약물)"으로 명명해 freeze 후 1회 평가 | 공개·무료·로그인 불필요, UMI 행렬 182 MB(≤ 30 GB). 계산 비용은 v2 평가 pilot(16 triplet/세포주당 head 묶음 ~75 CPU-초)로 보아 수 CPU-시간 이하로 예상 [미검증, pilot 전]. 사전 지정 필요: 단일-hash QC 규칙, A/B plate를 cross 단위로 쓰는 정의, 약물 4개 단위 resampling. 정밀도상 gate 2(95% 상한 < 0)는 효과가 약 0.03 이상이어야 충족 가능 | 사용자(기준 완화는 결과 전 결정) |
| R4 후보 상한 확대 | 이번 상한(3개) 밖의 공개 다용량 24 h 단일세포 약물 screen을 추가 탐색 | metadata 탐색 비용만. 이번에는 탐색하지 않음 | 사용자 |

## 4. 자료 확보 후 바로 고정할 것(P1-R10 `frozen_protocol.yaml` 초안 항목)

- 입력 계약: 저·고 용량 endpoint well + endpoint replicate vehicle; 중간 용량 RNA·구성비·세포 수·depth 금지; query batch vehicle은 쓰지 않음(별도 계약 없이는).
- 모델: v1 `r10_b3_v1`의 D11[cross]·W[cross](그리고 보조 D10[cross]·t*[cross]·D00) seed 0·1·2, hash는 v2 `artifact_registry.tsv`와 대조; seed는 기술 반복으로 먼저 평균; 외부 최고 seed 선택 금지.
- 표현: v1 train-only HVG 2,000/PCA 20/GMM K=4 per line, 재적합 금지; 주 지표 = 고정 atlas 재판독 actual 구성비 TV.
- 주 대비: ΔTV = TV(D11[cross]) − TV(W[cross]); 기준 평균 ΔTV ≤ −0.005 **그리고** 95% CI 상한 < 0; within_mean_rmse·ED는 W 대비 상대 악화 95% 상한 ≤ +5%(W 값이 0 근처면 source에서 정한 절대 허용치).
- 통계 단위: seed → triplet → replicate 방향 → 세포주 → 약물; batch가 여럿이면 batch×약물 교차 의존을 반영한 paired 재표집, batch가 적으면 batch별 결과 + batch 조건부 약물 CI.
- 표본·비용 상한과 QC 규칙을 결과 전에 고정; 결과를 보며 표본 추가 금지.
