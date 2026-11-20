# R6-R 실행 명세 (동결 전 작성) — 반복 가능한 약물 차이 진단

근거: private plan 2026-11-19 §3.2. 기계가 읽는 명세: `configs/r6_specificity_v2/r6r_spec.json`. 코드: `src/r6_specificity_v2/repeat_eligibility.py`(metadata만), `src/r6_specificity_v2/repeat_audit.py`.
노출 지위: **이미 노출된 자료의 후속 진단**(source TRAIN 약물 well; 21개 cross-plate 약물은 v1 `positive_control_replicate_wells.tsv`에서 이미 요약됨). blind test가 아니다.

## 가설
한 plate에서 잰 같은 pool 안의 약물 차이 `Δ_d − Δ_e`가 다른 plate의 같은 세포주·같은 약물쌍에서 (잘못 짝지은 약물쌍보다) 재현된다. v1의 cross-plate pool 중심화 cosine 약 0.010은 낮은 반복성뿐 아니라 plate마다 다른 약물 구성으로 만든 pool 평균의 차이도 포함했을 수 있다.

## 적격 집합 (응답값을 읽기 전에 metadata로 고정, `manifests/r6_specificity_v2/`)
- 사용 가능한 well: v1 well 표의 valid, R6 계약에 pool 존재, 세포 ≥20, barcode 반쪽마다 ≥10. 같은 (세포주, 약물, plate)의 well이 둘 이상이면 세포 수 가중 평균.
- TRAIN 약물 209개 중 두 plate 이상에 있는 약물 21개(세포주×약물 964개). plate는 5개(plate11/13/2/5/8).
- pair link = (세포주, plate 쌍, 두 plate에 공통인 약물쌍): 2,773개, 고유 약물쌍 59개, 고유 약물 20개(123은 공통 짝이 없음), plate 쌍 6개, 세포주 47개.
- 주 집합 = 같은 stratum에 겹치지 않는 wrong-match 쌍이 있는 link(공통 약물 ≥4): 2,362개, 약물 17개, plate 쌍 3개(0-4: 5약물, 1-4: 9약물, 3-4: 4약물). link 비중으로 본 유효 plate 쌍 수 1.91.
- 식별 가능성 기준(사전): 주 집합 약물 ≥8, plate 쌍 ≥2 → 충족(17, 3).
- 같은 plate의 다른 well: 8개 약물(각각 한 plate에만), (세포주, 약물, plate) 376개.
- metadata 한계: Tahoe sample metadata에 날짜·계대·배양 batch 열이 없다 → **독립 배양 여부 식별 불가**. 다른 plate = 다른 well·plate일 뿐이다. 또 같은 구조 그룹(R6의 약물 단위)이 plate마다 다른 catalog 화합물(염 형태; 일부는 입체이성질체·관련 화합물: 64 crizotinib/(S)-crizotinib, 68 glycyrrhizinate/glycyrrhizic acid, 205 doxorubicin/epirubicin, 320 salicylate)인 경우가 많다(`repeat_compound_names.tsv`).

## 비교 조건
- matched: `c_P(d,e)`와 `c_Q(d,e)` (같은 세포주, 다른 plate). 한 pool 안의 차이이므로 대조군 기준 평균은 정확히 상쇄되어 쓰지 않는다.
- wrong-match: 같은 stratum의 겹치지 않는 쌍 `(d',e')`의 `c_Q(d',e')` (양방향 평균; cosine은 정렬 d'<e', RMSE는 방향 평균).
- zero: 상수(0) 대비 예측.
- 기술적 분할: 같은 well의 barcode 반쪽 A 대 B(측정 잡음만의 상한), 그리고 잡음 수준을 맞춘 cross-plate 반쪽(P의 A 대 Q의 B).

## 주 대비 (두 개, Holm)
- P1 = 약물 macro 평균 [matched cos − wrong-match cos], 유리 방향 > 0.
- P2 = 약물 macro 평균 [matched RMSE − wrong-match RMSE], 유리 방향 < 0, wrong-match 수준 대비 상대값도 보고. 실용 기준: 상대 |차이| ≥ 0.5%.
- 불확실성: 약물 구조 그룹 블록 bootstrap 2,000회(seed 0), pair link 가중 = 두 약물 multiplicity의 곱. 양측 bootstrap p, Holm α=0.05.
- 판정: `repeatable` = P1·P2 모두 Holm 유의·유리 방향 + P2 실용 기준 + plate leave-one-out 모든 행에서 두 점추정 유리 방향. `weak_or_uncertain` = 하나 이상 유의·유리하지만 위 조건 미충족. `not_established` = 둘 다 아님. `unidentifiable` = 식별 기준 미충족.
- R6-E 진입: 판정이 `repeatable` 또는 `weak_or_uncertain`이고, 서버에 provenance가 분명한 label-free 고정 분자 표현이 대용량 다운로드 없이 있을 때.

## 감도 분석 (모두 공개, 판정에 쓰지 않음)
S1 공통 anchor 중심화(같은 stratum의 다른 공통 약물 평균, query 제외), S2 v1식 pool 중심화와 pool leave-query-out 중심화, S3 plate leave-one-out, S4 기술적 분할, S5 동일·염 변형 화합물만, S6 같은 plate 다른 well(공유 중심화 편향을 막기 위해 다른 well들의 서로소 절반으로 각각 중심화), S7 약물 leave-one-out 범위, plate 쌍별·약물별 표.

## 미리 적는 한계
plate 5개·주 집합 plate 쌍 3개라 plate 모집단 CI는 주지 않는다(leave-one-out은 기술적 감도일 뿐). 평가 subset을 전체 자료 대표로 확대하지 않는다. 음성 또는 불안정한 결과는 약물의 생물학적 효과가 없다는 증거가 아니다.

## 배선 확인
`--wiring-test`(실제 응답 대신 난수, 출력은 서버 scratch)로 두 번 실행해 P1, P2, S6가 0 근처임을 확인했다(job 26736, 그 뒤 S6 중심화 수정 후 재확인). 실제 응답은 동결 커밋 뒤 한 번만 읽는다.
