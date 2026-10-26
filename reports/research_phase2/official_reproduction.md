# 공식 예제 실제 재현

scGPT source는 cebd6fae655b9c585a4807daa3ac31bb764f06b4의 Tutorial_Annotation.ipynb로 고정했다. 공식 MS reference7844/query13468cells, seed0,10epochs,batch32,length3001을 유지한 R00_ms_full_seed0_v2를 A30에서 실행했다. Official key mapper 적용 뒤 pretrained attention projection24개 tensor의 실제 일치를 확인했다.

| 지표 | 원 notebook 저장 출력 | 실제 실행 |
|---|---:|---:|
| Accuracy | .875 | .863528 |
| Macro precision | .775 | .715310 |
| Macro recall | .737 | .694437 |
| Macro F1 | .732 | .675581 |

이는 실제 실행한 modified_reproduction이다. 원 notebook과 정량 일치했다고 주장하지 않는다. Non-flash attention, key mapping, runtime·stochastic kernel 차이가 있으며, F1 차이 .0564의 원인을 분리하는 실험은 하지 않았다. Published variance 범위가 확인되지 않아 재현 허용오차를 임의로 만들지 않았다. Training1476초, inference101초, process GPU peak9.80GiB를 관측했다.64+64-cell smoke score는 정량 benchmark로 집계하지 않는다. 데이터 hash는 manifests/research_phase2/official_scgpt_ms.json, 원 예측·weights는 서버 데이터 영역에 있다.

V3에서는 추가 학습 없이 제한 점검했다. 원 notebook·실행 script hash, Preprocessor/train_test_split/set_seed 호출, 원 concatenation의 query ID suffix `-1`과 author label 순서,13468개 저장 예측의 accuracy/precision/recall/F1 재계산을 확인했다. 기본 config 차이는 fast_transformer True→False 하나였다. 저장 prediction에서 F1 .675580933이 재현되므로 metric 계산 착오로 차이를 설명할 근거는 없었다. Attention24tensor load 확인은 backend 수치 동일성을 보장하지 않는다. 남은 backend/loader/runtime 차이의 기여는 분리하지 않았으며, 원 notebook 저장 점수의 반복 분산도 미확인이다. 실제 근거는 `official_scgpt_gap_audit/audit.json`이다. Pre-execution execution_contract의 running 표기는 역사적 snapshot이며 완료 판정은 final run.json과 실제 예측을 따른다.

| 점검 항목 | 원 notebook와 실제 실행의 관계 |
|---|---|
| Data/label | 같은7844/13468×3000 공식 입력; 원 author label과 query `-1` ID 변환 후 순서 일치 |
| Split | 원 reference train_test_split 호출과 seed0 유지; HCC donor split로 바꾸지 않음 |
| Preprocessing/binning | 원 Preprocessor 호출·기본 설정 유지 |
| Pretrained key | Non-flash key 이름에 맞추는 공식 helper 적용; 실제 QKV24tensor 일치 확인. 원 partial-name-match loader와 구현 차이가 있음 |
| Attention/runtime | fast_transformer True→False, torchtext 호환 wrapper와 runtime 차이. 영향 크기를 분리한 실험 없음 |
| Checkpoint 선택 | 원 epoch loop의 낮은 validation val_loss로 best_model 선택; test F1로 재선택하지 않음 |
| 평가 | 같은 accuracy/macro precision/recall/F1 정의. 저장 prediction 재계산 일치. Worker cap4 및 시각화·원격 logging 생략은 별도 호환 변경 |

최적화·checkpoint 선택·metric call의 제한적 source 비교는 `official_scgpt_gap_audit/source_comparison.json`에 별도 기록한다. Backend 차이가 F1 차이의 원인임을 증명한 결과로 해석하지 않는다.

GEARS R01_norman_simulation_seed1_v2는 공식 simulation seed1, hidden64,batch32/test128,20epochs를 실제 완료했다. 제공 processed matrix91205cells 중 원 코드의 GO eligibility filter를 통과한89357cells를 사용했다. Test는107interventions/28754cells이며 K562 CRISPRa single/dual-target activation이다. 간이나 HCC 결과가 아니다.

Native MSE .00528548, DE MSE .18092044, Pearson .985051, DE Pearson .866546을 기록했다. 원 DE dictionary를 그대로 쓰는 공식 재현과, heldout DE를 학습에 넘기지 않는 strict HepG2 적용은 별도 task다. Native train()의 validation/test/deeper analysis 비용이 training 시간에 포함되므로 순수 optimizer 시간으로 해석하지 않는다. 원 논문과의 수치 일치 범위를 확립하지 않았으며 native 실행 완료로 보고한다.

동일한 조건별 평균으로 다시 집계한 비교에서는 GO/PCA/ridge DeltaRMSE .05004, GEARS .06921이었다. 이는 원 cell-level/DE metric과 다른 평가 단위다. 실제 동일 test population 검증과 bootstrap은 response_evaluation_norman에, 실행 시간의 범위는 resource_scope.md에 있다.

조건부 scVIDR는 원 VIDR 모듈을 사용해 mouse TCDD portal-treated holdout1074cells를 예측했다. Max100epochs 중 early stopping으로35epochs를 실행했고, native arithmetic와 regression을 같은 학습 VAE·동일 sampled control IDs로 비교했다. scvi0.13 API child 환경과 누락 LinearRegression import 수정, train-only5000HVGs 및 한 context 선택을 명시한다. 원 논문의 모든 celltype/dose 수치 재현이나 biological replicate 재현을 주장하지 않는다. 실패한v1/v2와 v3 저장 모델 복원을 함께 기록했다.
