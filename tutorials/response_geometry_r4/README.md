# R4 튜토리얼

| 노트북 | 필요한 것 | 내용 |
|---|---|---|
| `01_read_results.ipynb` | 저장소의 작은 표만 | 시험 결과·짝 비교·반응 크기별 오차 읽기 |
| `02_predict_unseen_drug.ipynb` | 서버 unit 저장소 + 동결 checkpoint 1개 | validation 약물 하나를 처리 전 RNA와 구조만으로 예측해 실측과 비교 (test 정답은 열지 않음) |

두 노트북은 서버에서 `experiments/response_geometry_r4/notebooks.sbatch`(job 25382)로 실행되어 출력이 저장돼 있다. 아무것도 학습하지 않는다.
