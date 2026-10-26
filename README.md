# HCC Virtual Cell 튜토리얼

00–05와 아래10k 표는 V1 보존 기록이다. 현재 연구의 진입점은 [단일 결과 인덱스](reports/research_phase2/RESULTS_INDEX.md)와 [연구 요약](reports/research_phase2/final_summary.md)다. [V2 진행 상태](reports/research_phase2/status.md), [V3 보완](docs/RESEARCH_AMENDMENT_V3.md), [데이터 카드](reports/research_phase2/data_card.md)를 함께 읽는다. V1의 두 test 환자는 새 blind test로 사용하지 않는다. 외부 최종 추론은 진행 중이며 전체 연구 완료를 선언하지 않는다.

공개 GSE149614 raw count에서 PCA, scGPT whole-human, Geneformer-V2-104M의 환자 분리 세포 유형 분류와 Reactome 경로 특징을 비교한다. Geneformer 가상 삭제는 **실측 섭동 검증 전 embedding 변화 탐색**이다.

G1에 두 격리 환경과 공개 HCC 입력을 준비하고, 2,000세포 PCA/scGPT/Geneformer 비교 및 KG 실험을 실제 실행했다. 한국어 노트북 6개는 새 kernel에서 Run All을 통과했다. 10,000세포 비교, 100세포 가상 삭제 및 110개 정합성 점검을 완료했다. 실제 결과는 [요약](reports/final_summary.md), 사용법은 [서버 튜토리얼](docs/SERVER_TUTORIAL.md)을 참고한다.

## 서버 경로

- 접속: `ssh gssong@g1.sysmed.kr`
- 코드/환경/노트북: `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver`
- 원본 count/metadata: `/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE149614`
- 변환 데이터/체크포인트/출력: `/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver`
- `data`, `checkpoints`는 위 데이터 영역을 가리키는 로컬 symlink이며 Git에서 제외한다.

## 시작

```bash
ssh gssong@g1.sysmed.kr
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
source envs/activate.sh scgpt
python scripts/00_inspect_environment.py --out reports/environment.json
python scripts/02_baseline_annotation.py --config configs/pca_smoke.yaml
python scripts/03_scgpt_annotation.py --config configs/scgpt_smoke.yaml
source envs/activate.sh geneformer
python scripts/04_geneformer_annotation.py --config configs/geneformer_smoke.yaml
```

사전 준비된 64세포 파일은 `$VCELL_DATA_ROOT/subsets/smoke.h5ad`. 원본 다운로드/전체 sparse 변환은 `scripts/download_hcc.sh`와 `scripts/01_prepare_hcc.py --config configs/hcc_gse149614.yaml`; 작은 실습에서 다시 실행할 필요 없다. `configs/paths.yaml`에 절대 경로를 명시했다.

전체 계획과 생물학적 한계는 [실행 요청서](docs/AGENT_EXPERIMENT_PLAN.md), [모델 비교](docs/MODEL_COMPARISON.md), [자원 예산](reports/run_budget.md)를 참고한다. 모델 입력 차이 때문에 실용 pipeline 비교이며 아키텍처만의 비교가 아니다. frozen embedding + supervised logistic head를 사용하며 시스템 전체를 zero-shot classifier라 부르지 않는다.

raw count·대형 표현·가중치·비밀값은 Git에 올리지 않는다. 각 자산의 URL, revision, SHA256, size와 실제 실행 환경/자원을 manifests, envs, reports, results에 기록한다. test donor는 downstream holdout이며 사전학습 중복은 확인 불가 시 unknown이다.

## 튜토리얼과 검증

`notebooks/00_*.ipynb`부터 `05_*.ipynb`까지 순서대로 진행한다. 기본 입력은 64세포이며 자동 전체 다운로드를 하지 않는다. `source envs/activate.sh scgpt` 후 `python scripts/08_execute_notebooks.py`로 화면 없이 Run All을 실행할 수 있다. 실제 출력/실행 기록은 [notebook_execution.json](reports/notebook_execution.json)에 있다.

2,000세포 test macro-F1: PCA **0.9359**, scGPT **0.9327**, Geneformer **0.9210**. scGPT+Reactome **0.9288**로 추가 이득이 없었다. peak CUDA allocated는 scGPT 약0.37GiB, Geneformer 약1.65GiB(각 batch8/1200, batch4/4096; frozen inference). 전체 CLI 시간은 stages, 모델 자원 측정 구간은 run.json에 기록했다. [환경 lock 안내](envs/README.md), [모델 비교](docs/MODEL_COMPARISON.md), [실제 결과](reports/final_summary.md)를 참고한다.


## 10,000세포 실제 비교

같은 두 test 환자의 1,240세포에서 측정했다.

| 입력 표현 | macro-F1 | balanced accuracy | peak process GPU GiB |
|---|---:|---:|---:|
| PCA | 0.9566 | 0.9498 | CPU |
| scGPT whole-human | 0.9527 | 0.9496 | 0.72 |
| Geneformer-V2-104M | 0.9456 | 0.9463 | 2.50 |

이 설정에서는 foundation embedding이 PCA를 넘지 못했다. scGPT+Reactome의 macro-F1은 0.9436이며 추가 이득이 없었다. 두 모델은 frozen embedding + supervised head로 비교했다. 가상 삭제는 train hepatocyte100세포에서 ALB/HNF4A를 독립 실행했고 no-op/발현량 유사 무작위 대조를 포함한다. **실측 섭동 검증 전 embedding 변화 탐색**이며 발현량·약효 예측이 아니다.

[최종 정합성 점검](reports/completion_audit.json), [자산 체크섬 검증](reports/asset_verification.json), [큰 로컬 산출물 위치/해시](manifests/local_artifacts.json). 이는 V1 종료 시점 기록이다. 당시 필수 단계의 미해결 차단은 없었고, 미세조정/scFoundation/GEARS는 실행하지 않았다. 사전학습 중복은 unknown이다. 다음 실험은 고정 protocol의 leave-one-patient-out 평가다.


## V2 연구 확장 (진행 중)

[연구 계획 V2](docs/RESEARCH_EXPERIMENT_PLAN_V2.md)에 따라 기존 환경과 V1 결과를 보존하며 확장한다. [실행된 연구 튜토리얼 06](notebooks/06_research_development_cv.ipynb)은 분류와 공식 예제, [튜토리얼 07](notebooks/07_measured_perturbation_response.ipynb)은 실측 반응 기준선을 읽고 선택적으로 재실행하는 인터페이스다.

[08 개선 실험](notebooks/08_data_model_improvements.ipynb), [09 라벨 없는 새 HCC 추론](notebooks/09_new_hcc_query_annotation.ipynb), [10 mouse liver TCDD](notebooks/10_conditional_mouse_liver_chemical_response.ipynb)도 실제 실행했다. 기본 동작은 저장된 실제 결과 읽기이며, 새 학습·추론은 명시적으로 옵션을 켰을 때만 실행한다. [실행 hash](reports/research_phase2/tutorial_execution.json)와 [자원 측정 범위](reports/research_phase2/resource_scope.md)를 함께 제공한다.

HCC02/HCC09를 제외한 8명, 63,036개 전체 개발 세포의 4-fold donor OOF 결과다. 모든 평가 세포를 유지하고, 학습은 fold당 12,000개 자연 표본으로 제한했다. 이 결과는 반복 확인한 개발 결과이며 새 blind test가 아니다.

| 동결 표현 | pooled macro-F1 |
|---|---:|
| PCA | 0.9348 |
| scGPT | 0.9377 |
| Geneformer V2 | 0.9471 |
| Geneformer CLcancer | 0.9458 |
| scVI | 0.9379 |

[분류 결과와 provenance](reports/research_phase2/annotation_full_summary/pooled_metrics.tsv), [scGPT 공식 예제의 실제 재현 차이](reports/research_phase2/official_reproduction.md), [모델별 실행 상태](reports/research_phase2/model_registry.tsv)를 참조한다. LiverCT도 4개 fold를 완료했다. 비라벨 query adaptation의 native macro-F1은 0.9017, prediction coverage는 0.8871이며 별도 평가다. 같은 적응 모델의 SCANVI head 대조는 0.7061이다. 외부 reference와 label 양이 다르므로 구조만의 비교로 해석하지 않는다. GEARS의 실측 반응 예측은 분류 및 Geneformer embedding shift와 다른 task다. scFoundation 공식 가중치 다운로드는 HTTP403으로 차단되어 근거를 남겼다. 외부 GSE151530은 모든 선택을 고정한 원 batch의 추론을 진행 중이다. 알려진 다섯 비악성 lineage 전이만 평가하며, 최종 점수는 아직 생성되지 않았다.

실측 HepG2 CRISPRi의214개 target/2000gene 평가에서는 GO/PCA/ridge RMSE .10584, GEARS .13091이었다. 별도 공통191target/1977-primary-gene 조건에서 scGPT 사전학습 .12394, 무작위 초기화 .12268, GO/PCA/ridge .11073이었다. 사전학습 가중치의 추가 이득은 확인되지 않았다. Simple/native 비교에는 control 평균화 차이도 포함되며, V3 [control 감도](reports/research_phase2/control_sensitivity.md)에서 GO/ridge의 sampled-control 입력 RMSE .13050과 한 target의 saved-model 입력 감도를 별도 확인했다. 서로 다른 gene/target panel 점수를 합치지 않는다. [실측 genetic 결과](reports/research_phase2/leaderboard_genetic_response.tsv), [KG 압축 대조](reports/research_phase2/kg_compression_diagnostic/pooled_metrics.tsv), [가설 registry](reports/research_phase2/hypothesis_registry.tsv)에서 음성 결과까지 확인할 수 있다.

조건부 mouse liver TCDD portal-treated holdout1074cells에서는 같은 VAE의 native arithmetic RMSE .13854, scVIDR regression .07629였다. 동물별 assignment가 없어 biological replicate CI를 주장하지 않으며, 사람 성인 HCC 약효 예측으로 해석하지 않는다. [Chemical 결과](reports/research_phase2/leaderboard_chemical_response.tsv)는 별도 표다.
