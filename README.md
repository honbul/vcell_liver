# HCC Virtual Cell 튜토리얼

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

[최종 정합성 점검](reports/completion_audit.json), [자산 체크섬 검증](reports/asset_verification.json), [큰 로컬 산출물 위치/해시](manifests/local_artifacts.json). 필수 단계의 미해결 차단은 없으며, 미세조정/scFoundation/GEARS는 선택적 후속으로 실행하지 않았다. 사전학습 중복은 unknown이다. 다음 실험은 고정 protocol의 leave-one-patient-out 평가다.
