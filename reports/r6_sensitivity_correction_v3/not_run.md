# P0-R6 v3 미실행 기록

| 항목 | 상태 | 이유 | 여는 조건 |
|---|---|---|---|
| S6 같은 plate·다른 well 행 재계산 | 범위 제외 | wrong-match 비교 기준이 없다(pool 중심화만 함). §6.2의 comparator 정정 대상이 아니다. v2 값(`reports/r6_specificity_v2/repeat_contrast_audit.tsv`)이 그대로 유효하다 | 없음 |
| S7·S3 행의 comparator 재구성 bootstrap | 범위 제외 | 사전 명세가 재구성 bootstrap을 primary와 S5 v3의 P1/P2로 한정했다. S7 요약은 v2와 같은 점추정 범위 형식이다 | 다음 사이클에 별도 명세 |
| 재구성 bootstrap 분포 진단(replicate 저장, 빈 comparator 처리 방식 변경) | 미실행 | 결과를 본 뒤 방식을 바꾸면 사후 재설계가 된다. 관측한 사실(제외 link 수, p가 말해 주는 분포 중심)만 보고한다 | 사용자가 다음 사이클에 별도 계산으로 승인할 때. 처리 방식은 결과를 보기 전에 명세로 고정한다 |
| S5-S1 anchor 필터의 S7 버전 | 범위 제외 | v2 S7은 P1/P2에만 정의되어 있었다 | 없음 |
| 화합물 lot 동일성 필터 | **자료 대기** | Tahoe sample metadata에 lot·catalog batch 열이 없다. 이름 계열(identical/salt_variant)은 lot 동일성을 뜻하지 않는다 | lot 기록이 있는 반복 자료 |
| 독립 배양(날짜·계대) 반복성 | **자료 대기** | metadata에 배양 날짜·계대·batch 열이 없다(v2 확인) | `r6_pause_and_resume.md`의 자료 |
| plate 모집단 CI | **자료 대기** | 0.5 µM plate 5개, 주 집합 plate 쌍 3개(v3 S5는 2개), 모두 plate8 포함 | plate가 더 많은 반복 설계 |
| per-well 배열이 없어 정정하지 못한 S5/S7 항목 | 해당 없음 | 필요한 배열(v1 `halves_v1/treated_halves.npy`)이 모두 있어 전부 다시 계산했다. 추정해서 채운 값은 없다 | — |
| Tahoe 5 µM / 0.05 µM cross-plate well 추출, Sci-Plex3 교차 연구 비교 | 범위 제외 | §6.3은 후보가 있는지만 정리하라고 했다(`candidate_repeat_data_inventory.tsv`). 추출·비교는 새 분석이다 | 사용자 승인과 사전 명세 |
| R6-E 구조 표현 probe, 새 학습, 새 분자 표현, set encoder | 범위 제외 | plan §6.1·§6.2. 이번 결과는 자동 재개 조건이 아니다 | 독립 반복 자료로 반복성이 확인되고 사용자가 결정할 때 |
| 검증·시험 약물 평가 | 범위 제외 | 정정 대상이 source TRAIN 진단뿐이다 | — |
