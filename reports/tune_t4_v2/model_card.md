# T4 v2 모델 카드

공개 L1000의 978개 랜드마크에 대해 같은 물질·같은 농도의 6시간 DMSO 차감 벡터를 24시간 DMSO 차감 벡터로 예측한다. 검증 맥락은 HEPG2와 PHH다. HEPG2 132군과 PHH 67군을 원래 plate·fold·세 DMSO 역할을 유지해 채점했다. 추론 시점에는 24시간 입력이나 외부 서비스가 필요하지 않다.

## 오프라인 실행

G1에 동결된 모델은 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/top3_tuning_v2/c/runs`에 있다. 저장소 루트에서 Python, numpy, pandas, SciPy, scikit-learn, PyTorch, RDKit, h5py, joblib을 갖춘 환경으로 실행한다.

```bash
artifact=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/top3_tuning_v2/c/runs
python -m tune_t4_v2.predict \
  --cell HEPG2 --input expression_6h.tsv \
  --models "$artifact/release-02/forecasts" \
  --lock "$artifact/release-02/freeze.json" \
  --pan "$artifact/pretrain-01" --out prediction_24h.tsv
```

입력 TSV는 고유 `sample_id`, 화합물의 표준 parent SMILES `parent`, 양수 `dose_uM`, 그리고 숫자로 표기한 978개 Entrez 유전자 열을 사용한다. 각 6시간 발현은 입력 plate의 DMSO 기준으로 먼저 차감한다. CLI는 해당 scaffold를 학습에서 제외한 동결 모델만 사용하고, 모델 수와 각 랜드마크의 평균 예측을 출력한다. 논문의 최종 성적은 웰별 등록 fold·plate·DMSO 역할에 맞는 단일 조각 예측 기준이며 CLI는 사용 가능한 조각의 평균을 반환한다.

HEPG2에서 동등한 6시간 입력을 사용하는 rank-16과 State 적응보다 높은 예측 정확도를 확인했다. PHH의 세 few-shot 예산에서는 공통 배율을 앞서지 못했다. 모델은 정해진 실험 처리 기하와 세포 맥락에서 24시간 발현 대비를 예측한다. [성적표](RESULTS.md) · [오프라인 재생 영수증](evidence/offline_receipt.json) · [동결 모델 목록](evidence/release_freeze.json)
