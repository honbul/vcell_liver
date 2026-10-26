# 간암 Virtual Cell 공개 모델 비교와 첫 재현 실험

확인 기준일: 2026-10-26. 연구 리더에게 제공한 18쪽 Word 보고서의 저장소용 요약본이다. 문헌·공식 코드·공개 데이터 등록정보를 검토한 결과이며, 이 저장소에서 모델 학습·추론 또는 GPU 측정을 완료했다는 뜻이 아니다. 상세 실행 지시는 [AGENT_EXPERIMENT_PLAN.md](AGENT_EXPERIMENT_PLAN.md)를 따른다.

## 1. 권고

**우선 후보는 scGPT와 Geneformer-V2-104M이다.** GSE149614에서 환자 단위 세포 유형 분류를 먼저 재현하고, 같은 split·label·classifier family로 비교한다. 기존 KG 또는 Reactome 기반 경로 특징을 추가해 실제 이득을 측정한다.

Geneformer 가상 삭제는 임베딩 변화 분석으로 제한한다. GEARS는 실제 genetic perturbation 데이터가 준비된 후의 조건부 경로다. 네 모델을 모두 설치하거나 새 foundation model을 학습하는 것은 초기 필수 작업이 아니다.

## 2. 입력·가능 task·공개 자산

| 모델 | 입력 | 가능한 task / 추가 조건 | 공개 자산·선정 |
|---|---|---|---|
| scGPT | gene vocabulary에 맞춘 발현값·토큰, AnnData. 사용 checkpoint 경로에 맞는 normalization/binning | 동결 embedding, reference mapping, 분류 미세조정. integration/perturbation은 해당 task의 데이터와 절차 필요 | 공식 코드·weights·tutorial 공개. 코드 MIT. **첫 후보** |
| Geneformer | raw counts, Ensembl ID, n_counts, rank-value encoding. 버전 일치 dictionary | embedding, 세포/유전자 분류 미세조정, 가상 삭제/과발현의 embedding shift. 절대 발현량 복원기는 아님 | 공식 HF 코드·weights·docs 공개. Apache-2.0. **두 번째 후보, V2-104M 지정** |
| scFoundation | 공식 19,264 gene 순서의 expression matrix, read-depth·전처리 상태 | cell/gene embedding, enhancement. 분류/약물/섭동은 하위 모델을 결합·학습 | 코드 Apache-2.0. weights 별도 비상업 연구 조건. 입력/환경 부담 때문에 후순위 |
| GEARS | control와 genetic perturbation scRNA-seq, condition·cell_type·gene_name, gene graph | 미관측 단일/조합 유전자 섭동의 전사체 반응. 범용 cell classifier 아님 | 공식 코드·data loader·Colab 공개, MIT. 실제 섭동 데이터 확보 후 |

주요 근거: [scGPT 원 논문](https://doi.org/10.1038/s41592-024-02201-0), [Geneformer 원 논문](https://doi.org/10.1038/s41586-023-06139-9), [scFoundation 원 논문](https://doi.org/10.1038/s41592-024-02305-7), [GEARS 원 논문](https://doi.org/10.1038/s41587-023-01905-6).

사전학습 embedding에 supervised classifier를 학습한 결과는 **zero-shot embedding + supervised head**다. 시스템 전체를 zero-shot 분류라고 부르지 않는다.

## 3. GPU 메모리 — 공식 관측과 예산은 다르다

아래 숫자는 **미측정 파일럿 장비 배정 제안**이며 공식 최소 VRAM이나 실행 보장값이 아니다. 파라미터 수 또는 weight file 크기만으로 필요한 VRAM을 결정하지 않는다.

| 모델 | 확인된 공식 근거 | 시작 예산 제안 |
|---|---|---|
| scGPT | CZI quickstart는 T4 및 M3 Mac 32 GiB에서 embedding 예제를 시험. 전체 HCC fine-tuning 사양은 아님 | frozen inference 16 GB급에서 batch 1부터; 작은 fine-tune 24 GB급 검토 |
| Geneformer V2-104M | docs는 GPU 권고. 일관된 최소 VRAM 수치 미확인. 양자화 논문은 상대 절감 보고 | frozen inference 24 GB급, sequence 4096·batch 1 시작. fine-tune는 부분 동결/지원 양자화/더 큰 VRAM 검토 |
| scFoundation | README는 memory 부족 시 부분 동결, gene_batch ≤5 안내. VRAM 최소 숫자 없음 | cell embedding 24 GB급에서 작은 시험. full fine-tune/큰 gene output은 48 GB 이상 장비 검토 |
| GEARS | pretrained Norman 예제 Colab Pro 안내, GPU 종류·최소 VRAM 미고정 | 작은 graph·hidden64·작은 batch를 16–24 GB급에서 시험 |

근거: [CZI scGPT Quickstart](https://virtualcellmodels.cziscience.com/quickstart/scgpt-quickstart), [Geneformer get started](https://geneformer.readthedocs.io/en/latest/getstarted.html), [scFoundation model README](https://github.com/biomap-research/scFoundation/blob/main/model/README.md), [GEARS](https://github.com/snap-stanford/GEARS), [Geneformer scaling/quantization](https://doi.org/10.1038/s43588-026-00972-4).

실제 run마다 model/checkpoint, batch, sequence, dtype, GPU 이름, baseline 사용량, peak allocated/reserved, process GPU memory, CPU RSS, wall time을 저장한다. OOM은 batch부터 축소한다. 입력 gene/sequence 수를 줄였다면 별도 조건으로 기록한다. GPU 여러 장의 VRAM이 자동으로 합쳐진다고 가정하지 않는다.

## 4. 왜 scGPT부터인가

공식 annotation/reference mapping tutorial과 CZI quickstart가 있어 작은 end-to-end 재현을 만들기 쉽다. 기본은 whole-human checkpoint의 frozen cell embedding이며, train 환자의 label로 단순 선형 분류기를 학습한다.

`GSE149614 count → QC/ID mapping → scGPT 전용 전처리 → frozen embedding → 선형 classifier → held-out 환자 label`

64세포 smoke → 모든 환자를 포함하는 2,000세포 pilot → 최대 10,000세포 비교로 확대한다. PCA+logistic regression을 반드시 비교한다. model이 이기지 못해도 실제 결과와 원인을 남긴다.

checkpoint config·vocab·weights를 함께 고정한다. flash-attention 없는 지원 경로부터 시도하고, CZI 예제의 base 환경 torch 제거 명령을 작업 서버에서 그대로 실행하지 않는다. HVG/PCA/scaling은 train에서만 학습한다.

공식: https://github.com/bowang-lab/scGPT  
분류: https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html  
참조 매핑: https://scgpt.readthedocs.io/en/stable/tutorial_reference_mapping.html

## 5. Geneformer 버전과 가상 섭동의 의미

V1 원 논문과 현재 V2를 구분한다. 기본은 **Geneformer-V2-104M**이고 root default를 무심코 받아 316M으로 바꾸지 않는다. V2의 비암 transcriptome pretraining과 CLcancer의 암 도메인 continual learning을 구분하고, 암 특화 버전이 항상 더 낫다고 가정하지 않는다. corpus overlap은 확인 불가이면 unknown으로 남긴다.

Geneformer에는 scGPT용 HVG/binned matrix를 재사용하지 않는다. raw counts·Ensembl·n_counts와 V2 token/median dictionary를 맞춘다. 별도 classifier를 같은 patient split에서 학습한다.

in-silico perturber의 `delete`는 gene token을 제거하고 `overexpress`는 rank 앞쪽으로 옮긴다. 출력은 cell/gene embedding shift다. 시간별 expression, logFC, viability 또는 drug efficacy와 다르다. `genes_to_perturb`에 여러 genes를 주면 공동 섭동하는 동작이므로 독립 비교는 하나씩 호출한다.

2026년 Nature Protocols는 raw-count tokenization → pretrained embedding 평가 → task fine-tuning → embedding-shift 기반 후보 순위의 절차를 소개한다. 공개 초록과 공식 docs를 튜토리얼 틀로 사용하되, HCC 검증 결과로 전용하지 않는다.

공식 카드: https://huggingface.co/ctheodoris/Geneformer  
Tokenizer: https://geneformer.readthedocs.io/en/latest/geneformer.tokenizer.html  
Perturber: https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html  
Protocol: https://doi.org/10.1038/s41596-026-01364-8

## 6. scFoundation과 GEARS를 후순위로 둔 이유

scFoundation은 고정 19,264 gene 순서, read-depth 및 pre_normalized 설정을 지켜야 한다. cell embedding과 `N×19264×hidden` gene output의 메모리 규모가 다르다. downstream task마다 적합한 head와 데이터가 필요하다. 첫 비교가 끝난 뒤 full-gene 표현 또는 GEARS 결합 가설이 명확해질 때 추가한다. 코드 공개와 모델 가중치의 사용 조건은 다르다.

- https://github.com/biomap-research/scFoundation
- https://github.com/biomap-research/scFoundation/blob/main/MODEL_LICENSE

GEARS는 관측 HCC scRNA-seq만으로 특정 gene intervention의 정답을 만들 수 없다는 제약이 있다. 공식 저장소는 cross-cell-type transfer를 지원하지 않는다고 명시하고, single perturbation만 학습해 조합을 잘 예측할 것이라고 가정하지 말라고 설명한다. 약물 이름을 gene perturbation처럼 넣는 것도 native 사용법이 아니다.

후속은 공식 Norman 데이터의 방법론 재현 → 실제 liver-line Perturb-seq → 독립 HCC intervention validation 순서다. Norman 결과를 HCC 결과로 표시하지 않는다.

## 7. 데이터 선정

### 우선: GSE149614

10명 HCC 환자의 원발 종양, PVTT, 전이 림프절, 비종양 간을 다룬 atlas. GEO에는 21샘플과 약 7만세포 processed 자료가 있다. 처음에는 다음 두 파일만 받는다.

- `GSE149614_HCC.metadata.updated.txt.gz` — GEO 등록 약 478.4 KB
- `GSE149614_HCC.scRNAseq.S71915.count.txt.gz` — GEO 등록 약 157.7 MB

별도 normalized 파일은 초기 필수가 아니다. raw sequencing은 EGA 접근이지만 공개 processed count로 시작할 수 있다. 본 문서 작성 시 파일 내부 header/값은 검사하지 않았으므로 agent가 먼저 조사해야 한다.

원 연구: https://doi.org/10.1038/s41467-022-32283-3  
등록: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149614

환자 단위 6/2/2를 기본안으로 두되 label coverage를 확인하고 사전 고정한다. 한 환자의 여러 조직이 train/test에 걸치지 않게 한다. tumor tissue cell 전체를 malignant로 라벨링하지 않는다. adjacent liver를 건강한 간과 동일시하지 않는다.

### 외부 후보: GSE151530

HCC와 iCCA가 혼합된 liver cancer 자료다. histology/환자 metadata로 HCC subset을 확인해야 한다. cohort의 label이 다르면 mapping과 unknown 범위를 별도 기록한다.

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151530

### 실제 유전적 섭동: GSE264667

Jurkat과 HepG2 pooled CRISPR scRNA screen이다. dual-sgRNA library와 day-7 readout이 등록되어 있다. intervention type, guide-target, control 정의는 실제 자료로 확인한다. dual sgRNA가 자동으로 두 유전자 조합을 뜻하지는 않는다.

GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667  
TRADE 관련 논문: https://doi.org/10.1038/s41588-025-02169-3

**HepG2는 hepatoblastoma 유래로 정정된 세포주**다. 간 유래 pilot에는 쓸 수 있으나 성인 환자 HCC 근거와 구분한다: https://www.cellosaurus.org/CVCL_0027

### 약물 후속: Tahoe-100M

약물·용량 조건의 단일세포 반응 자원이다. drug-dose condition 수와 distinct compound 수를 혼동하지 않는다. 필요한 간 유래 subset만 별도 검토하며 전체 100M을 첫 단계에 받지 않는다. genetic perturbation GEARS와 별도의 task 설계가 필요하다.

2026 Cell 논문: https://www.sciencedirect.com/science/article/pii/S0092867426010081

## 8. 기존 공유 KG 논문과 연결

| 자료 | 검증한 연구 대상 | HCC 모델링에 가져올 부분 |
|---|---|---|
| Hetionet/Rephetio | 이종 biomedical graph와 경로 특징의 drug repurposing | ID 통합, 경로 근거, 단순 graph-feature baseline |
| PrimeKG | 질환·약물·gene·pathway 지식 자원 | 외부 prior. 세포별 관측 상태와 구분 |
| BioMedKG/PrimeKG++ | multimodal node representation과 link prediction | 후속 gene/drug embedding. cell-response 성능과는 다름 |
| MEDAKA | 약물 설명서에서 LLM으로 KG 추출 | 추출 품질·provenance. patient cell simulator는 아님 |
| VCWorld | retrieved evidence+LLM의 gene-level DE/DIR | 기존 지식 활용과 비교. 전체 transcriptome dynamics와 구분 |

공개 출처:

- https://doi.org/10.7554/eLife.26726
- https://doi.org/10.1038/s41597-023-01960-3
- https://doi.org/10.3389/fsysb.2025.1651930
- https://arxiv.org/abs/2509.26128 (preprint)
- https://github.com/GENTEL-lab/VCWorld

미출판 내부 자료나 팀의 원시 실행 결과는 public repo에 재배포하지 않는다. 팀 로컬 코드와 공개 VCWorld 코드가 동일하다고 가정하지 않는다.

## 9. KG 연계 제안

먼저 existing KG export 또는 Reactome의 human gene-pathway mapping을 사용한다. 세포별 상태는 count에서 얻고, 일반적 pathway membership은 KG에서 얻는다. membership이 실제 활성도나 인과성의 증거를 대신하지 않는다.

**설명용:** 분류/섭동 결과에 pathway·출처를 연결하되 prediction은 바꾸지 않는다.

**예측용:** train gene-scaling으로 계산한 pathway mean-z-score 등의 proxy를 frozen embedding과 concatenate하고 같은 classifier를 학습한다.

필수 ablation:

1. PCA + classifier
2. frozen model embedding + classifier
3. pathway-only + classifier
4. embedding + pathway + classifier
5. embedding + size-matched random gene sets + classifier

이는 knowledge-informed feature modeling이다. retrieval+LLM generation이 없다면 RAG라고 부를 필요가 없다. train/validation만으로 pathway 선택·규제 강도를 정한다.

Reactome: https://reactome.org/download-data  
TF-target 후속: https://saezlab.github.io/dorothea/ — 공식 사이트는 CollecTRI를 권고하므로 DoRothEA를 유지하면 이유와 사용 조건을 기록한다.

## 10. 평가와 완료 기준

동일 donor split·cell IDs·label schema로 비교한다. 모델별 공식 전처리는 유지하되, 서로 다른 gene input 범위를 사용하면 아키텍처만의 효과라고 해석하지 않는다.

분류 지표는 macro-F1, balanced accuracy, class recall, patient별 결과와 pooled confusion matrix다. UMAP은 표현 탐색이지 정량적 성능의 대체 지표가 아니다. train에서 HVG/PCA/scaling/classifier를 학습하며 test를 섞은 batch correction은 기본 분석에서 제외한다.

Geneformer embedding shift와 GEARS expression MSE를 한 순위표에 합치지 않는다. 실제 perturbation task에는 no-change/control mean, train perturbation mean, 단순 linear/additive baseline을 포함한다.

독립 평가가 보여준 baseline의 중요성:

- https://doi.org/10.1186/s13059-025-03574-x — 기존 scGPT/Geneformer 버전의 zero-shot 평가. 모든 최신 버전의 순위가 아님.
- https://doi.org/10.1038/s41592-025-02772-6 — 유전적 섭동 예측과 단순 선형 비교. 본 HCC cohort에서의 결과는 아님.

완료 기준은 실제 HCC pipeline, 공정한 비교, 실측 메모리, 재실행 가능한 한국어 튜토리얼과 Git 이력이다. ‘F1 90%’ 같은 근거 없는 통과 기준 또는 ‘KG가 반드시 성능을 올려야 함’을 설정하지 않는다.

## 11. 튜토리얼 순서

환경·checkpoint → HCC count/metadata·patient split → PCA 기준선 → scGPT frozen embedding → Geneformer 공정 비교 → KG 추가 실험 → 소규모 가상 삭제 해석 → 실제 결과·실패·다음 실험 보고.

새 kernel에서 작은 subset의 notebook을 Run All로 검증한다. 이미 존재하지 않는 스크립트를 실행했다고 쓰거나 mock output을 실제 결과로 표시하지 않는다. 세부 구현 목표와 Git 단계별 반영 규칙은 실행 요청서를 따른다.

## 확인 범위

일부 출판사 논문은 공개 초록·서지·그림 설명 범위만 확인했고, 실행 지침은 공식 코드·docs로 보완했다. 체크포인트 실제 다운로드, GPU 요구량, metadata header, pretraining corpus overlap은 실행 단계에서 확인해야 한다. 공식 사실과 본 문서의 실행 예산·설계 제안을 구분한다.

## G1 실제 실행 결과로 연결

위 문헌 표와 초기 예산은 실제 측정 결과와 구분한다. 2026-10-26 G1 A30에서 두 모델을 실행했고, 한국어 작은 subset 노트북 6개를 새 kernel에서 검증했다. 2,000세포의 환자 분리 test macro-F1은 PCA 0.9359 / scGPT 0.9327 / Geneformer-V2-104M 0.9210이었다. 두 모델 모두 이 PCA 기준선을 넘지 못했다. 해당 frozen inference의 peak CUDA allocated는 scGPT(batch8,length1200) 약0.37GiB, Geneformer(batch4,length4096,float32) 약1.65GiB였다. fine-tuning 사양이나 논문 재현 결과가 아니다.

정확한 10,000세포 비교, 자원 측정 범위, KG ablation 및 한계는 [실제 최종 요약](../reports/final_summary.md), 재실행은 [서버 튜토리얼](SERVER_TUTORIAL.md)을 따른다.
