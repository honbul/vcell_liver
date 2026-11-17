# predictions/anchored_distribution_r3

- `final_condition_metrics.tsv.gz`: 동결 후 held-out 약물 평가의 조건(세포주×약물)·방법·복원 변형별 지표(seed 평균, 세포 단위 값 없음).
- `server_files.json`: 서버에만 있는 생성 세포(`assets/gen/final/*.npz`, 표본추출 seed 0, 복원 v2)와 평균 예측 배열의 경로·크기·SHA-256.

세포 단위 발현 배열(실제·생성)과 가중치는 Git에 넣지 않는다. 절대 DDPM, 잔차 DDPM, CellFlow의 예측은 없다(학습하지 않음; `reports/anchored_distribution_r3/RESULTS.md` 참고).
