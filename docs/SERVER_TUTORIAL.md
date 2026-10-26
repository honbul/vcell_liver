# G1에서 직접 실행하는 HCC Virtual Cell 튜토리얼

## 1. 접속과 경로

```bash
ssh gssong@g1.sysmed.kr
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
source envs/activate.sh scgpt
```

작업 루트는 지정한 `main` 아래의 `vcell_liver`다. raw 파일은 `/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE149614`, 모델/가공 데이터/큰 출력은 `/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver`다. 현재 경로 설정은 `configs/paths.yaml`. 다른 위치에서 임의 상대 경로로 다운로드하지 않는다.

## 2. 작은 실습 (기본 64세포)

준비된 파일 `$VCELL_DATA_ROOT/subsets/smoke.h5ad`를 사용한다. 전체 다운로드를 다시 시작할 필요가 없다. 원본 count와 metadata를 확인해 만든 실제 HCC subset이며 synthetic/mock 데이터가 아니다.

VS Code의 Remote SSH로 서버의 이 저장소를 열고 `notebooks/`의 파일을 순서대로 연다. Python kernel은 `.venvs/vcell-scgpt/bin/python`을 선택한다. Geneformer 단계는 notebook 안에서 전용 환경의 Python을 호출한다. 노트북 6개는 각각 새로운 kernel에서 Run All을 실제 실행했고, 기록은 `reports/notebook_execution.json`에 있다.

1. `00_environment_and_model_contracts.ipynb`: GPU·경로·checkpoint 계약.
2. `01_hcc_data_and_patient_split.ipynb`: sparse raw count·QC·원 annotation·환자 분할.
3. `02_baseline_and_scgpt.ipynb`: train-only PCA와 공식 scGPT 추론·분류 비교.
4. `03_geneformer_comparison.ipynb`: V2 공식 tokenizer와 104M frozen embedding.
5. `04_knowledge_graph_features.ipynb`: 경로 특징과 크기/유전자 참여 횟수를 보존한 random 대조.
6. `05_virtual_perturbation_interpretation.ipynb`: train hepatocyte의 작은 독립 유전자 삭제·no-op·발현량 유사 대조.

화면 없이 전체를 재실행하는 명령:

```bash
python scripts/08_execute_notebooks.py
```

노트북 실습의 run ID는 `tutorial_*`로 분리해 최초 2,000/10,000세포 비교 결과를 덮어쓰지 않는다. 64세포 결과는 동작 확인이며 본 성능 비교의 표본 크기가 아니다.

## 3. 같은 실험을 CLI로 실행

```bash
source envs/activate.sh scgpt
python scripts/02_baseline_annotation.py --config configs/pca_smoke.yaml
python scripts/03_scgpt_annotation.py --config configs/scgpt_smoke.yaml
source envs/activate.sh geneformer
python scripts/04_geneformer_annotation.py --config configs/geneformer_smoke.yaml
source envs/activate.sh scgpt
python scripts/05_kg_features_ablation.py --config configs/kg_smoke.yaml
source envs/activate.sh geneformer
python scripts/06_geneformer_virtual_perturb.py --config configs/isp_smoke.yaml
```

2,000세포는 config의 `smoke`를 `pilot`으로, 10,000세포 분류/KG 비교는 `comparison`으로 바꾼다. 가상 삭제는 `isp_pilot.yaml`에서 최대100세포만 사용한다. 분류 모델의 `C`와 class weight는 고정 grid를 validation으로만 고른다. test는 HCC02/HCC09이며 성능을 보고 split/config를 바꾸지 않는다.

## 4. 출력 찾기

- `results/<run_id>/run.json`: status, model/checkpoint/hash, input/split hash, actual dtype/batch/sequence와 실측 자원.
- `results/<run_id>/metrics.json`, `per_class.tsv`, `by_patient.tsv`, `confusion_matrix.tsv`: 실제 평가.
- `$VCELL_DATA_ROOT/outputs/<run_id>/predictions.tsv`: 원본 cell ID 순서의 prediction/confidence.
- `$VCELL_DATA_ROOT/features/<run_id>.npz`: embedding/PCA와 cell ID; Git 제외.
- `$VCELL_DATA_ROOT/tokenized/<run_id>/cells.dataset`: 공식 Geneformer V2 tokenized dataset; Git 제외.
- `reports/model_comparison_measured.tsv`, `final_summary.md`: 모든 완료된 비교 요약.
- `reports/kg_ablation_pilot.tsv`, `kg_ablation_comparison.tsv`: KG 성능 변화와 실제 prediction 변경 수.
- `reports/virtual_perturbation_embedding_shift*.tsv`: token 존재 여부와 1−cosine; 실측 섭동 검증 전 탐색.
- `reports/figures/*_umap.png`: train-fitted UMAP을 label/patient로 각각 표시. 성능 증거 대신 쓰지 않는다.
- `manifests/local_artifacts.json`: 큰 산출물의 위치·크기·hash·재생성 명령.

## 5. 실제 결과 예시와 자원

2,000세포 / test253세포에서 PCA macro-F1=0.9359, scGPT=0.9327, Geneformer=0.9210이었다. 이 실험에서는 foundation model이 PCA를 이기지 못했다. scGPT+Reactome은 0.9288로 scGPT보다 낮았고 test 예측 4개가 변경됐다. 더 큰 비교와 정확한 전체 지표는 `reports/final_summary.md` 참조.

A30에서 scGPT batch8/length1200의 peak CUDA allocated 약0.37GiB, process GPU 약0.70GiB였다. Geneformer batch4/length4096/float32의 2,000세포 값은 약1.65GiB / 2.47GiB였다. 이 수치는 해당 frozen inference 조건의 관측값이며 fine-tuning 최소 사양이 아니다. run.json의 wall_seconds는 모델 로딩·입력 처리·embedding 중심의 측정 구간이며, 전체 CLI 시간은 `reports/stages/<run_id>.json`에 별도로 있다. CPU RSS는 해당 프로세스의 누적 최대값이다. 다른 gmx 작업과 공유한 서버의 관측 시간이다.

## 6. 실패 해결

GPU OOM이면 config의 batch_size를 1로 줄인다. sequence/gene 수를 줄이면 다른 실험 ID로 기록한다. GPU VRAM 네 장을 합산하지 않는다. 다른 사용자 작업을 종료하지 않는다. scGPT는 flash-attention을 설치하지 않은 공식 non-flash 경로로 실행된다. CPU fallback은 느리며 CUDA 비교와 섞어 보고하지 않는다.

파일이 없다면 `bash scripts/download_hcc.sh` 후 `python scripts/01_prepare_hcc.py --config configs/hcc_gse149614.yaml`을 실행한다. 이는 전체 71,915세포 sparse 변환이며 작은 실습의 기본 동작이 아니다. 모델/환경을 재구성할 때는 `envs/README.md`의 lock 안내와 `scripts/bootstrap_server.sh`를 참고한다. scGPT pip check의 누락 5개는 사용하지 않는 학습/integration extras이며 해당 원문과 inference 환경의 범위를 기록했다.

## 7. 해석의 경계

분류는 관측 세포 유형을 예측하는 supervised head다. Geneformer의 가상 삭제는 rank token을 지운 뒤 embedding이 얼마나 달라지는지 보는 탐색이다. 실제 KO/약물 처리 뒤 발현량·logFC·생존율·약효를 예측하지 않는다. 정상/질환 목표 상태를 검증하지 않아 정상화 방향도 주장하지 않는다.

GEARS의 실측 반응 예측은 대조군/처리군, 정확한 intervention annotation과 별도 perturbation split이 필요하다. tumor/adjacent label을 유전자 개입 정답으로 바꿀 수 없다. HepG2 실험도 성인 환자 HCC 모델과 구분한다. 사전학습 corpus overlap은 unknown이며 이 donor split을 pretraining-unseen으로 해석하지 않는다.
