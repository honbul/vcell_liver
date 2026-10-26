# /goal — 간암 Virtual Cell 공개 모델 재현·비교·사용자 튜토리얼

작성 기준: 2026-10-26  
작업 저장소: https://github.com/honbul/vcell_liver  
기본 후보: **scGPT + Geneformer-V2-104M**  
필수 데이터: **GSE149614**  
후속 조건부 후보: GEARS + 실제 유전자 섭동 데이터

> 이 파일은 AI agent에 전달할 실행 요청서다. 현재 문서 작성 단계에서 모델 실행이나 GPU 측정을 완료했다는 뜻이 아니다. 아래 스크립트·CLI·노트북 이름은 **구현할 목표 인터페이스**이며 이미 존재하는 코드라고 가정하지 않는다. 문헌 재현(공식 예제)과 새로운 HCC 적용 실험을 구분한다.

## 0. 수행 목표와 범위

공개 간암 scRNA-seq 데이터로 모델 설치부터 입력 변환, 작은 추론, 환자 단위 평가, KG 연계, 사용자가 직접 따라 하는 한국어 튜토리얼까지 연결하라. 계획 작성이나 설치 확인만으로 종료하지 말고, 가능한 범위에서 실제 실행 결과를 남겨라.

최종적으로 다음을 답할 수 있어야 한다.

1. scGPT, Geneformer, scFoundation, GEARS의 입력·task·메모리·공개 코드/가중치/라이선스가 어떻게 다른가?
2. 주어진 장비에서 scGPT와 Geneformer를 실제로 실행할 수 있는가? 어떤 설정에서 얼마의 메모리를 쓰는가?
3. 환자 HCC의 세포 유형 분류에서 동결 모델 표현이 PCA 기준선보다 유용한가?
4. 경로 지식 특징을 더했을 때 실제 예측 성능이 달라지는가?
5. Geneformer의 가상 유전자 삭제 결과는 무엇을 의미하고 무엇을 의미하지 않는가?

**필수 실행:** 환경 조사, GSE149614 준비, PCA 기준선, scGPT smoke+환자 분리 파일럿, Geneformer 비교, 작은 KG 추가 실험, 실제 실행된 사용자 튜토리얼, Git 반영.

**범위 제한:** 새 foundation model 사전학습, 대규모 전체 KG 재구축, 전체 Tahoe-100M 다운로드, 복잡한 multi-agent/RAG 서비스, 웹 UI는 이번 필수 범위가 아니다. 기존 작업을 갈아엎거나 보안 감사·전면 리팩터링·반복적인 리뷰를 선행조건으로 만들지 않는다. 아래 데이터 정합·누수 방지·성능 확인은 연구 결과의 유효성을 위한 필수 확인이다.

## 1. 작업 원칙과 금지 사항

- scGPT를 먼저 끝까지 연결하고 Geneformer를 동일 protocol로 추가한다. 네 모델을 모두 설치하느라 첫 결과가 지연되지 않게 한다.
- 스모크 성공, 공식 예제 재현, HCC 신규 적용, 생물학적 검증은 서로 다른 상태로 기록한다.
- 사용자 장비의 GPU/RAM/OS는 미확정이다. 다른 프로젝트의 24 GB GPU 정보를 이 프로젝트의 사실로 사용하지 않는다.
- 관측 HCC의 tumor/adjacent 구분을 유전자 KO/약물 처리의 실험 정답으로 바꾸지 않는다.
- Geneformer의 embedding shift를 post-perturbation expression, logFC, 효능 또는 생존율이라고 보고하지 않는다.
- RNA 발현량을 0으로 만드는 조작, LLM 생성값, KG 링크를 실측 섭동 정답으로 사용하지 않는다.
- 가짜 지표·임의로 채운 실행 로그·실행되지 않은 notebook output을 만들지 않는다. 실패는 원인과 재현 명령을 남긴다.
- 모델이 baseline을 못 이기더라도 그대로 보고한다. 성공 기준은 특정 F1 숫자가 아니라 재현 가능성과 비교의 타당성이다.
- 자동으로 유료 API/클라우드에 가입하거나 대규모 학습을 시작하지 않는다. 모델 가중치·데이터 사용 조건과 다운로드 접근 제한을 지킨다.
- API key·토큰·환경변수 비밀값·환자별 비공개 자료·공유받은 미출판 리뷰/랩미팅 원본을 public Git에 올리지 않는다.

## 2. Phase 0 — 저장소·장비 확인과 실행 계획 확정

### 2.1 저장소

`git@github.com:honbul/vcell_liver.git` 또는 현재 인증된 접근 경로를 사용하라. 기존 파일·branch·미커밋 변경을 읽고 보존하라. 이 요청서를 제공한 뒤 다른 작업자가 변경했을 수도 있으므로 빈 저장소라고 고정하지 않는다. force push를 사용하지 않는다.

루트 README에 목표, 현재 실행 상태, 빠른 시작, 튜토리얼 순서, 결과 링크를 만든다. 원격 쓰기 권한이 없으면 로컬 commit과 정확한 오류를 남기고 모델 작업은 계속한다. 실제 push 성공 전에는 원격 반영 완료라고 쓰지 않는다.

### 2.2 장비·환경 인벤토리

OS, Python, CPU, RAM, disk free, GPU별 model/VRAM/사용량, CUDA driver, PyTorch/CUDA 호환성을 기록한다. `nvidia-smi` 실패도 기록하되 credentials나 전체 환경변수는 수집하지 않는다. 다른 실행 중 작업을 종료하거나 GPU를 강제로 독점하지 않는다.

모델별 격리 환경을 사용한다. 예: `vcell-scgpt`, `vcell-geneformer`, 선택적 `vcell-gears`. 공식 버전 제약을 먼저 읽고 설치하며, 임의로 모든 패키지를 최신 버전으로 통일하지 않는다. 성공한 후 실제 lock/pip freeze를 저장한다. 기본 환경의 torch를 제거하지 않는다.

### 2.3 예산과 축소 규칙

아래는 첫 파일럿의 제안값이며 사용자 장비의 보장 사양이 아니다.

| 항목 | 초기 제한 |
|---|---|
| 모델 수 | scGPT와 Geneformer 2개. 하나가 차단되면 가능한 모델로 먼저 완결 |
| 모델 smoke | 64세포, batch 1부터 시작 |
| end-to-end pilot | 최대 2,000세포, 가능한 모든 환자를 포함 |
| 비교 확대 | 최대 10,000세포, 이후 전체 데이터는 추가 필요성과 자원 보고 후 결정 |
| 미세조정 | 동결 비교 후 필요할 때만 작은 epoch budget, validation early stopping |
| 가상 섭동 | 최대 100세포 × 소수의 사전 지정 유전자. 전 유전체 스캔 금지 |
| 재시도 | 동일 실패를 무한 반복하지 말고 원인에 따라 최대 2개 실질적 대안 시험 |

GPU가 없거나 작은 경우 먼저 CPU 데이터·PCA 작업과 작은 공식 추론을 수행한다. GPU 기능이 필수인 경로는 blocked로 남긴다. OOM은 batch 축소 → 불필요한 gene-level 출력 제거 → 공식 부분 동결/지원되는 정밀도 순서로 대응한다. sequence/HVG 수 변경은 별도 실험으로 기록하며, 다른 GPU의 메모리가 자동 합산된다고 가정하지 않는다.

산출물: `reports/environment.json`, `reports/run_budget.md`, `envs/*`, `manifests/upstream_versions.tsv`.

## 3. Phase 1 — 공개 자산과 checkpoint 고정

### 3.1 scGPT

공식 코드: https://github.com/bowang-lab/scGPT  
공식 annotation: https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html  
공식 reference mapping: https://scgpt.readthedocs.io/en/stable/tutorial_reference_mapping.html  
간단한 공개 quickstart: https://virtualcellmodels.cziscience.com/quickstart/scgpt-quickstart

기본은 공식 whole-human checkpoint의 동결 cell embedding 경로다. weights, config, vocab을 한 세트로 받는다. 외부 제3자 checkpoint를 검증 없이 대체하지 않는다. dependency 충돌이 있으면 upstream commit과 README에 맞는 환경을 고정하고 flash-attention 없는 지원 경로부터 시험한다.

### 3.2 Geneformer

공식: https://huggingface.co/ctheodoris/Geneformer  
시작: https://geneformer.readthedocs.io/en/latest/getstarted.html

기본 checkpoint는 **Geneformer-V2-104M**이다. root의 기본 가중치가 316M일 수 있으므로 subdirectory·config·실제 parameter count를 확인한다. 암 특화 `Geneformer-V2-104M_CLcancer`는 추가 비교 후보이지 기본 모델과 같은 이름으로 섞지 않는다.

HF revision, weight hash, tokenizer dictionary, gene median dictionary를 함께 고정하라. git-lfs clone이 전체 가중치를 모두 받지 않도록 선택 다운로드를 사용한다. 모델 카드의 자동 생성된 일반 텍스트 AutoTokenizer 예제를 그대로 복사하지 말고 공식 `TranscriptomeTokenizer` 경로를 따른다.

V1-10M으로 축소해야 한다면 V1 tokenizer/2048 설정과 함께 별도 실험명으로 수행한다. 이를 V2 결과라고 보고하지 않는다.

### 3.3 비교 대상 문서화

scFoundation: https://github.com/biomap-research/scFoundation  
GEARS: https://github.com/snap-stanford/GEARS

두 모델은 공개 자산·task·입력·라이선스를 비교 표에 포함하되 필수 설치하지 않는다. scFoundation의 code Apache-2.0와 별도 MODEL_LICENSE를 구분한다. GEARS는 범용 pretrained cell classifier가 아니라 실측 genetic perturbation predictor임을 명시한다.

모든 다운로드에는 원 URL, upstream revision, filename, SHA256, size, 확인 시점, license/usage note를 저장한다. bytes를 받지 못한 파일은 download_verified=false로 둔다. 모델 파일을 Git에 직접 넣지 않는다.

## 4. Phase 2 — GSE149614를 표준 AnnData로 구성

공식 등록: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149614

우선 다운로드 파일:

- `GSE149614_HCC.metadata.updated.txt.gz`
- `GSE149614_HCC.scRNAseq.S71915.count.txt.gz`

공식 HTTP 후보 경로:

```text
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149614/suppl/GSE149614_HCC.metadata.updated.txt.gz
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149614/suppl/GSE149614_HCC.scRNAseq.S71915.count.txt.gz
```

경로 변경 시 GEO supplementary 링크에서 다시 해소하라. 원시 FASTQ/EGA 접근이나 1.2 GB normalized 파일을 첫 단계의 필수 조건으로 삼지 않는다. 공개 count matrix와 metadata로 진행하라.

### 4.1 원본 확인 — 실제 헤더를 먼저 읽을 것

파일 방향(gene×cell/cell×gene), gene identifier 종류, 값이 raw count인지, cell barcode, patient/sample/tissue/cell annotation 열을 확인한다. 아래 스키마는 **목표 스키마**이지 원본 열이름을 단정한 것이 아니다. 변환 mapping을 코드와 문서에 남긴다.

```text
AnnData.obs:
  cell_id, patient_id, sample_id, tissue_site,
  cell_type_original, cell_type_coarse, split,
  n_counts, n_genes, pct_counts_mt
AnnData.var:
  gene_id_original, gene_symbol, ensembl_id, id_mapping_status
AnnData.layers['counts']:
  변형되지 않은 raw count, sparse
AnnData.uns:
  dataset_accession, source_manifest, schema_mapping, qc_config
```

메타데이터-행렬 cell ID의 교집합/누락/중복을 모두 기록한다. 가능한 한 original cell ID를 유지한다. 모호한 ID는 임의의 유사문자열로 채우지 않는다. symbol→Ensembl 매핑은 고정된 human annotation을 이용하고, one-to-many/중복 gene 처리와 폐기량을 기록한다. Ensembl version suffix 제거 전후 매칭도 기록한다.

최초 로딩에서 전체 행렬을 dense로 만들지 말고 chunk/sparse 변환을 사용한다. `counts`는 log 변환하지 않은 상태로 보존한다. library size 계산 시점과 gene filtering 이전/이후 합계의 차이를 기록한다.

### 4.2 QC와 라벨

원 연구의 processed 데이터라는 점을 고려해 무조건적인 과도한 재필터링을 피한다. 기본값을 설정하되 세포당 count/gene/mitochondrial 분포와 조직별 특성을 먼저 확인하고 변경 근거를 남긴다. 재필터링 전후 환자·클래스 수를 보고한다.

첫 task는 원 annotation의 넓은 세포 계열 분류다. 예: T/NK, B, myeloid, endothelial, fibroblast, hepatocyte 등은 **원본에 실제 존재하는 범주와 매핑 가능한 경우에만** 사용한다. T/NK가 합쳐져 있으면 임의로 두 정답 클래스로 나누지 않는다. 모든 tumor-origin cell을 malignant로 분류하거나 hepatocyte label만으로 악성 여부를 결정하지 않는다.

metadata에 전문가 annotation이 없으면 marker-based labeling을 원 논문의 정답이라고 부르지 않는다. 그 경우 공식 예제 재현을 먼저 마치고 annotation 부재를 blocked 항목으로 남겨 대체 공개 cohort를 검토한다.

### 4.3 split과 subset

환자 단위 6/2/2를 기본안으로 하되 label coverage를 확인하고 사전 고정한다. 동일 환자의 tumor/adjacent/PVTT/MLN은 같은 split이다. rare label이 train에 없으면 unknown 또는 미평가로 표시한다. test 성능을 본 뒤 유리한 split으로 바꾸지 않는다.

가능한 모든 환자를 포함하는 64세포 smoke, 2,000세포 pilot, 10,000세포 비교 subset을 deterministic하게 만든다. subset은 환자·라벨 구성을 보존하도록 sampling하고 실제 수가 부족하면 전체를 사용한다. 파일마다 split manifest와 세포 목록을 보존한다. 원본 test-label 접근은 분할/평가 준비 목적에 한정하고 특징 선택·하이퍼파라미터 최적화에 사용하지 않는다.

산출물: `data/processed/*.h5ad`(로컬), `reports/data_card.md`, `reports/qc_summary.tsv`, `manifests/label_mapping.tsv`, `manifests/gene_mapping_summary.tsv`, `manifests/splits/*.tsv`.

## 5. Phase 3 — PCA 기준선과 scGPT test run

### 5.1 기준선

train 환자로 HVG 선택·normalization 설정·scaling·PCA·분류기 학습을 수행한다. 각 세포 내부의 library normalization은 test에도 동일한 고정 규칙으로 적용할 수 있지만, 데이터 전체로 추정하는 scaling/HVG/PCA는 train만 사용한다.

기본 분류기는 multinomial logistic regression이다. class weighting과 C는 작은 사전 정의 grid를 validation에서 고르고 저장한다. 동일 모델 비교에는 동일한 클래스 정의와 평가 세포 목록을 사용한다. 필요한 경우 nearest-centroid를 간단한 보조 기준선으로 추가한다.

### 5.2 scGPT smoke

1. 모델·vocab·config 로딩 및 GPU device 선택.
2. 64세포를 checkpoint 기대 입력으로 변환. 모델 경로가 내부 normalization/binning을 수행하는지 확인해 중복 변환하지 않는다.
3. vocabulary 매칭률, 세포당 유효 토큰 수, 출력 shape, NaN/Inf를 확인.
4. 원본 cell ID와 embedding 행 순서가 같음을 확인.
5. peak memory와 wall time을 실제 측정.

### 5.3 HCC 파일럿

2,000세포에서 전체 절차를 실행하고 10,000세포까지 확대한다. 동결 scGPT embedding에 기준선과 같은 종류의 분류기를 학습한다. 이는 **zero-shot embedding + supervised head**로 표기한다. 전체 시스템을 zero-shot classifier로 표시하지 않는다.

필수 출력: macro-F1, balanced accuracy, per-class precision/recall/F1, pooled confusion matrix, patient별 결과, cell별 prediction/confidence, train/val/test 셀 수, runtime/VRAM. UMAP은 label과 patient로 각각 색칠하며 시각화만으로 우수성을 결론 내리지 않는다.

## 6. Phase 4 — Geneformer 비교와 선택적 미세조정

공식 tokenizer: https://geneformer.readthedocs.io/en/latest/geneformer.tokenizer.html

동일 원본 count에서 **별도 Geneformer 입력**을 만든다. scGPT용 HVG-only 또는 binned matrix를 재사용하지 않는다. 공식 요구대로 raw counts, Ensembl, n_counts, V2 dictionary, special token, maximum length를 맞춘다. cell-level count 정규화와 rank encoding은 공식 tokenizer를 사용한다.

64세포 smoke 후 같은 2,000/10,000세포 split에 대해 frozen embedding을 만든다. 같은 종류의 분류기를 학습하고 PCA/scGPT와 비교한다. 입력 유전자 범위가 다르므로 결과를 실용 pipeline 비교로 해석한다. 아키텍처만의 차이라고 단정하지 않는다.

### 6.1 선택적 개선

동결 결과를 먼저 저장한 뒤, 목표·GPU 예산이 허용하면 한 모델의 분류용 fine-tuning만 작은 범위로 진행한다. 기본/CLcancer 비교나 부분 동결도 별도 run ID를 부여한다. 고정 learning rate grid와 early stopping을 사용하며 무제한 HPO를 하지 않는다. full fine-tune가 실패하면 사실을 보고하고 기존 frozen 결과를 보존한다.

사전학습 corpus와 GSE149614/GSE151530의 중복은 확인 가능한 공개 목록에서 조사한다. 확인 불가이면 `pretraining_overlap=unknown`으로 기록한다. downstream donor split만으로 pretraining-unseen 성능이라고 보고하지 않는다.

## 7. Phase 5 — KG 최소 연계와 실제 효과 측정

### 7.1 데이터 경계

기존 KG export가 제공되면 human gene ID, relation, pathway/target, evidence/source version을 읽는다. 원본을 수정하거나 전체 KG를 다시 만들지 않는다. 제공되지 않으면 공개 Reactome human gene-pathway mapping으로 동일 인터페이스를 구현한다.

공식 소스: https://reactome.org/download-data

초기 관계 스키마:

```text
head_id, head_type, relation, tail_id, tail_type,
source_name, source_version, evidence_id, species
```

세포별 gene expression은 환자 데이터이고 gene-pathway membership은 외부 prior라는 구분을 유지한다. 단순 경로 membership을 해당 세포에서의 활성화/인과관계라고 부르지 않는다.

### 7.2 필수 최소 실험

train 통계로 gene scaling을 고정한 뒤 고정 gene set의 평균 z-score 등 재현 가능한 pathway proxy를 계산한다. 다음을 **동일한 split과 classifier family**로 비교하라.

- PCA + classifier
- 선택한 foundation embedding + classifier
- pathway features + classifier
- foundation embedding + pathway features + classifier
- foundation embedding + size-matched random gene-set features + classifier

필수 모델링은 단순 feature concatenation이다. GNN·cross-attention·RAG를 추가해야만 완료되는 것으로 만들지 않는다. 경로 집합과 규제 강도는 validation으로 고르고 test를 변경에 사용하지 않는다. random gene-set seed와 size/membership 분포를 기록한다.

DoRothEA/CollecTRI는 signed TF 정보가 필요한 후속 옵션이다. 공식 사이트의 최신 권고·license를 확인한다: https://saezlab.github.io/dorothea/

### 7.3 설명용 결과

선택된 유전자/경로의 출처를 별도 표로 연결하라. 설명만 붙인 결과와 예측 입력으로 사용한 결과를 구분한다. 근거가 없으면 없다고 표시하고 LLM으로 존재하지 않는 pathway edge를 생성하지 않는다.

산출물: `reports/kg_ablation.tsv`, `reports/kg_integration.md`, `manifests/kg_sources.tsv`.

## 8. Phase 6 — Geneformer 가상 섭동 사용자 실습

공식: https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html  
프로토콜: https://doi.org/10.1038/s41596-026-01364-8

이 단계의 목적은 **표현 변화 분석의 사용법과 한계를 배우는 것**이다. first-pass는 작은 세포 하위집합에서 pretrained 또는 검증된 task-finetuned 모델의 유전자별 삭제 전후 cosine shift를 구한다.

- 세포 계열을 먼저 고정하고, 해당 계열에서 실제 발현·vocabulary가 확인되는 소수의 gene을 사전 지정한다.
- 후보 선택 근거를 기록하되 test 결과를 보고 gene을 고르지 않는다. 특정 gene 이름을 ‘간암 치료 표적 정답’으로 강제하지 않는다.
- **여러 gene ID를 한 리스트에 주면 함께 삭제하는 공식 동작에 주의한다.** 독립 비교는 한 gene씩 실행한다.
- 동일 세포의 no-op 및 발현 수준이 비슷한 무작위 gene 대조를 포함한다. 각 조작에서 실제 token이 존재했는지 확인한다.
- 최대 100세포를 시작점으로 하고, 큰 gene/세포 전수 스캔은 하지 않는다.
- 분류용 head가 세포 유형만 학습했다면 그 confidence 변화를 ‘정상화’라고 해석하지 않는다.
- 목표 상태로의 이동을 분석하려면 동일 계열 안의 신뢰할 상태 라벨과 train/reference에서 정의한 목표 embedding이 있어야 한다. 없으면 방향성 없이 변화 크기만 보고한다.
- 임베딩 이동이 크다고 생물학적 영향이나 약효가 큰 것이 아니다. 결과 파일과 그림에 ‘실측 섭동 검증 전 탐색’이라고 명시한다.

산출물: `reports/virtual_perturbation_embedding_shift.tsv`, `reports/virtual_perturbation_limitations.md`, 실행 notebook.

## 9. GEARS는 조건부 후속: 실행하지 않아도 필수 단계 미완료가 아님

실제 정량적 perturbation task를 우선하라는 별도 합의나 자원 여유가 있을 때 아래 경로로 확장한다.

1. 공식 Norman 등 데이터의 tutorial을 작은 규모로 재현한다. 이 결과에 HCC라는 이름을 붙이지 않는다.
2. GSE264667의 **HepG2 subset**에서 guide→target, control, modality(CRISPRi/KO 등), day, batch, count matrix를 확인한다. GEO 설명만으로 상세 intervention type을 추정해 고정하지 않는다.
3. 같은 세포주 내에서 유전자/perturbation 단위 split을 만들고 train 대조군·처리군으로 학습한다. test perturbation의 결과는 학습·retrieval·coexpression graph에 들어가지 않게 한다.
4. no-change/control mean, train perturbation mean, 단순 선형/가능한 additive 기준선을 비교한다. delta expression, perturbation별 성능과 DE recovery를 함께 보고한다.
5. 측정 시점·control 정의가 맞지 않는 데이터나 너무 적은 perturbation을 연결해 겉보기 모델만 만들지 않는다.

공식 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667  
관련 논문: https://doi.org/10.1038/s41588-025-02169-3

**생물학적 범위:** HepG2는 hepatoblastoma 유래로 분류된 간 유래 세포주이다(https://www.cellosaurus.org/CVCL_0027). 성인 HCC 환자 모델과 구분한다. GEARS 공식 구현은 세포 유형 간 전이용이 아니며, 약물 이름을 유전자 개입으로 바꿔 넣는 사용법도 아니다.

약물 반응으로 확장할 때 Tahoe-100M의 필요한 간 유래 subset만 별도 검토하고, 처음부터 100M 전체를 받지 않는다. native GEARS의 대체 데이터로 자동 취급하지 않는다.

## 10. 공통 실험 기록과 채점 규칙

각 run에는 다음 항목을 기계가 읽을 수 있게 저장한다.

```json
{
  "run_id": "unique_id",
  "status": "planned|downloaded|executed|blocked|failed",
  "task": "annotation|embedding_shift|measured_perturbation",
  "dataset_accession": "GSE149614",
  "data_hash": "measured_after_download",
  "split_manifest_hash": "measured",
  "model_name": "explicit_checkpoint_name",
  "upstream_commit": "resolved_commit",
  "checkpoint_revision": "resolved_revision",
  "checkpoint_sha256": "measured",
  "seed": 42,
  "n_cells": 0,
  "gene_mapping_coverage": null,
  "batch_size": 1,
  "max_sequence_length": null,
  "dtype": "record_actual",
  "gpu_name": "record_actual",
  "peak_cuda_allocated_gib": null,
  "peak_cuda_reserved_gib": null,
  "peak_process_gpu_gib": null,
  "peak_cpu_rss_gib": null,
  "wall_seconds": null,
  "metrics": {},
  "limitations": []
}
```

위의 0/null/문자열은 스키마 예시이며 실제 run 결과에서 미실행 값으로 남기거나 실측값으로 바꾼다. 영어/한국어 문서에서는 같은 기준을 사용한다.

분류의 주 지표는 macro-F1·balanced accuracy이다. 환자별 결과, 클래스별 recall, 평가 제외/unknown 수, confidence 사용 시 coverage도 포함한다. bootstrap을 적용한다면 독립 단위를 세포가 아니라 환자로 고려하고, test 환자가 적어 CI가 불안정하다는 점을 보고한다.

필수 누수 방지는 환자 중복, label feature 포함, train/test 전처리 혼합, test 기반 하이퍼파라미터 변경이다. 표현/label을 셔플하는 단순 sanity check로 우연히 label이 입력된 경우도 확인한다. 복잡한 보안 스캔이나 대규모 test framework를 이번 목표보다 우선하지 않는다.

## 11. 사용자 튜토리얼과 목표 CLI

다음 명령은 **구현할 인터페이스 예시**다. 실제 코드와 옵션을 일치시키고 실행 확인 후 README에 확정한다.

```bash
python scripts/00_inspect_environment.py --out reports/environment.json
python scripts/01_prepare_hcc.py --config configs/hcc_gse149614.yaml
python scripts/02_baseline_annotation.py --config configs/pilot.yaml
python scripts/03_scgpt_annotation.py --config configs/scgpt_smoke.yaml
python scripts/03_scgpt_annotation.py --config configs/scgpt_pilot.yaml
python scripts/04_geneformer_annotation.py --config configs/geneformer_pilot.yaml
python scripts/05_kg_features_ablation.py --config configs/kg_pilot.yaml
python scripts/06_geneformer_virtual_perturb.py --config configs/isp_pilot.yaml
python scripts/07_collect_results.py --runs results/ --out reports/final_summary.md
```

노트북 구성:

```text
notebooks/00_environment_and_model_contracts.ipynb
notebooks/01_hcc_data_and_patient_split.ipynb
notebooks/02_baseline_and_scgpt.ipynb
notebooks/03_geneformer_comparison.ipynb
notebooks/04_knowledge_graph_features.ipynb
notebooks/05_virtual_perturbation_interpretation.ipynb
```

각 notebook은 ‘무엇을 하는가 → 입력 확인 → 실행 → 실제 출력 → 생물학적 해석 → 실패 해결’ 순서로 작성한다. 새 kernel에서 Run All을 실제 실행해 숨은 변수가 없는지 확인한다. 큰 데이터는 자동 전체 다운로드하지 않고 작은 로컬 subset으로 기본 실행되게 한다.

필수 사용자 산출물은 ① 작은 데이터로 따라 할 수 있는 명령, ② 예상 파일 경로, ③ **실제 측정된** 결과 예시, ④ GPU/메모리 부족 대안, ⑤ 분류·embedding shift·실측 반응 예측의 차이 설명이다.

## 12. 폴더와 Git 반영 규칙

```text
vcell_liver/
  README.md
  AGENTS.md
  docs/AGENT_EXPERIMENT_PLAN.md
  docs/MODEL_COMPARISON.md
  envs/                    # 실행 성공 환경 잠금
  configs/                 # 원본 열 mapping, split, 모델, budget
  scripts/                 # 재사용 실행 경로
  notebooks/               # 실제 실행된 튜토리얼
  manifests/               # source/version/hash와 소규모 익명 식별자
  reports/                 # 요약, 지표, figure, resource 측정
  results/                 # 작은 prediction/metric, 큰 파일은 외부 경로 명세
  data/                    # gitignore, raw/processed/subsets
  checkpoints/             # gitignore
  third_party/             # 원본 복제는 gitignore, commit만 기록
```

의존 source code를 통째로 복사해서 기존 저작권/라이선스를 지우지 않는다. 필요한 upstream patch는 patch file과 원 commit으로 남긴다. public Git에는 원시 환자 데이터, 전체 cell-level count, 가중치, 비밀값, 비공개 내부 자료를 올리지 않는다. 공공 데이터도 재배포 조건을 확인하고 기본은 다운로드 manifest만 제공한다.

단계별 작은 commit과 push를 수행한다. 권장 순서:

1. 장비 인벤토리·환경·데이터 스키마와 split
2. PCA + scGPT smoke/pilot과 결과
3. Geneformer 비교와 resource 측정
4. KG ablation 및 선택적 embedding perturbation
5. 사용자 튜토리얼·최종 비교표·한계 정리

각 commit은 해당 코드·config·요약 지표·실행 로그를 연결한다. 성공한 결과만 골라 업로드하지 말고 실패/blocked 기록도 남긴다. 대용량 raw output은 위치·크기·해시와 재생성 명령을 manifest에 기록한다. 모든 raw를 억지로 Git에 넣지 않는다.

## 13. 완료 판정과 최종 보고 형식

필수 완료는 다음과 같다.

- 현재 장비에서 적어도 한 모델이 공개 HCC count를 받아 실제 embedding/prediction을 생성한다.
- 두 번째 모델은 동일 protocol로 비교하거나, 차단 원인과 실제 시도·대체 실행을 명시한다.
- 환자 단위 split과 PCA 기준선을 포함한 실제 지표가 있다.
- KG 추가 실험은 예측 변경 여부와 추가 이득을 검증한다. 내부 KG가 없으면 Reactome으로 수행한다.
- 가상 섭동을 실행했다면 실제 발현 예측과 다른 출력임을 모든 보고서에서 일관되게 표시한다.
- 작은 subset에서 사용자용 notebook이 처음부터 끝까지 실행된다.
- 코드/환경/checkpoint/data/split/version이 기록되고 저장소에 push된 commit이 확인된다.

최종 보고에는 **수행한 것 / 차단된 것 / 실제 성능 / 실제 VRAM / 재실행 명령 / 다음 한 가지 실험**을 포함한다. 모델의 생물학적 가설은 결과와 분리한다. 하나의 pipeline이 완성되면 결과를 먼저 남기고, 추가 작업을 이유로 전체 보고를 미루지 않는다.

## 14. 핵심 참고문헌

1. scGPT: https://doi.org/10.1038/s41592-024-02201-0
2. Geneformer 원 연구: https://doi.org/10.1038/s41586-023-06139-9
3. Geneformer 프로토콜: https://doi.org/10.1038/s41596-026-01364-8
4. Geneformer V2/양자화: https://doi.org/10.1038/s43588-026-00972-4
5. scFoundation: https://doi.org/10.1038/s41592-024-02305-7
6. GEARS: https://doi.org/10.1038/s41587-023-01905-6
7. GSE149614 원 연구: https://doi.org/10.1038/s41467-022-32283-3
8. Perturb-seq/TRADE: https://doi.org/10.1038/s41588-025-02169-3
9. Zero-shot 한계 평가: https://doi.org/10.1186/s13059-025-03574-x
10. Perturbation 선형 기준선: https://doi.org/10.1038/s41592-025-02772-6

공식 code/docs와 본 파일의 예산 제안은 원 논문 성과와 구분한다. 상세 모델 비교·출처·라이선스·데이터 범위는 동봉한 Word 비교 보고서 및 `docs/MODEL_COMPARISON.md`를 참고한다.
