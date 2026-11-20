# R10 v3 판정 (P0-R10 / P1-R10)

판정 규칙 freeze: `2b91274`([`RUN_SPEC.md`](RUN_SPEC.md)). 근거: private plan 2026-11-20 §3.

## P0-R10 — 독립 자료 적격성

- **결과 상태: 완료(적격성 확인) → 판정 `자료 대기`.**
- 후보 3개(Sci-Plex2 GSM4150377, Sci-Plex4 GSM4150379, 서버 보유 Tahoe-100M) 모두 필수 조건 중 하나 이상이 `fail`이다([`data_eligibility.tsv`](data_eligibility.tsv)).
  - Sci-Plex2: C2 fail(배양 날짜·생물학적 replicate 미식별, library 1개), C4·C6 partial.
  - Sci-Plex4: C2 fail, C4 fail(용량 계열이 plate 하나에만 있음), 약물 2개는 R10 train 약물.
  - Tahoe-100M: C3 fail(0.05/0.5/5 µM만), C2 fail, C6 fail/미검증.
- RUN_SPEC 판정 표의 "적격 후보 0" 행 → `자료 대기`, [`blocked_data_requirements.md`](blocked_data_requirements.md).

## P1-R10 — cross 독립 확인

- **결과 상태: 자료 대기(미실행).** 실패한 실험으로 세지 않는다.
- `frozen_protocol.yaml`, `per_triplet_metrics.tsv`, `per_batch_drug_metrics.tsv`, `primary_result.tsv`, `secondary_results.tsv`는 만들지 않았다(채점할 적격 자료 없음).
- 기존 Sci-Plex3 test(노출)와 그 재분할, R9 환자 자료로 대체하지 않았다.
- v2의 cross 결과(P6 −0.0102 [−0.0151, −0.0051])는 계속 **노출 자료의 후속 비교**로만 표시하며, 독립 재현은 미확인이다.

## 다음 행동

- 기본: 자료 대기 유지. 새 모델·새 학습·HCC 적응을 시작하지 않는다.
- 여는 조건: §2 계약을 만족하는 자료 확보 → `frozen_protocol.yaml` freeze → 1회 평가.

## 사용자 결정 필요(하나 선택)

1. 신규 독립 batch 실험(경로 R1)의 예산·협력 여부, 또는
2. Sci-Plex2를 C2 면제 하의 **외부 예비 검증**(단일 batch, A549, 4 약물, plate-level cross)으로 실행할지(경로 R3). 정밀도 검토상 95% 구간 반폭 ≈ 0.03 이상이라 채택 기준 충족 가능성은 낮다는 점을 결정 전에 고려.
