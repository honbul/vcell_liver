# V3 진행 수정사항

V2 계획을 유지한다. 검토 기준은589159e460b4d0e78f3e9fe01fc699bcbc829602이며, 이후 이미 해결된 부분은 재학습하지 않는다. 기존 환경·모델·데이터·예측·노트북을 보존한다.

| 요구 | 적용 경로 |
|---|---|
| 실행 인수 및 중복 방지 | Slurm 실제 상태, 완료 run.json, prediction hash를 확인. 기존48조건과 native response 결과를 재사용 |
| 조직 위치 | 원본 site→canonical tissue_site를 정확히 연결하고 donor×tissue_site×lineage별 OOF 오류·support 집계 |
| 외부 release 의존성 | 필수48조건·최종 bundle·입력 인터페이스 검증은 유지. 선택에서 제외된 KG 탐색은 optional provenance로만 기록 |
| 외부 평가 범위 | HCC 환자의 알려진 다섯 비악성 세포 계열 전이. Fixed-six/fixed-five·known coverage·Unknown을 함께 보고 |
| Control 감도 | 원본 예측을 유지한191target 집계와20개 control-resampling seed; saved native 모델 두 개의 사전 고정 첫 target 입력 감도 |
| 공식 재현 차이 | 저장된 입력·source·설정·예측·metric만 제한 점검. 추가 학습이나 임의 허용오차 없이 수치 불일치 보존 |
| 연구 결과 전달 | 단일 [결과 인덱스](../reports/research_phase2/RESULTS_INDEX.md), 연구 요약, 기존06–10 실행 예제 갱신 |

새 benchmark/model, 전체 재학습, 반복 보안 감사, 전면 리팩터링, notebook 개수 증가는 하지 않는다. 이미 생성된 external freeze·prediction·score를 지우거나 새 blind test로 발행하지 않는다. 서버 작업이 진행 중이면 해당 상태를 명시하고 전체 연구 완료라고 쓰지 않는다.
