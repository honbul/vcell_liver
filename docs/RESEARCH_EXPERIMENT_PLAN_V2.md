# 간암 Virtual Cell 연구 확장 실험 기획서
## 기본 모델·간 특화 모델 비교 → 데이터 분석 → Data-centric / Model-centric 개선

- **작업 저장소:** https://github.com/honbul/vcell_liver
- **검토한 기준 커밋:** `243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79`
- **작성일:** 2026-10-26
- **실행 대상:** 기존 G1 서버 환경과 저장소를 인수하는 AI 연구·개발 agent
- **권장 저장 위치:** `docs/RESEARCH_EXPERIMENT_PLAN_V2.md`
- **문서의 성격:** 저장소의 코드·기록된 결과, 첨부 리뷰 논문, 추가로 확인한 공개 원전을 바탕으로 작성한 **후속 실행 요청서**다. 이 문서 작성 과정에서 원격 GPU 실험을 다시 수행한 것은 아니다.

> **이번 목표는 튜토리얼 제작 자체가 아니다.** 기존 튜토리얼과 실행 환경을 재사용하여 연구용 비교 실험을 수행하고, 그 결과로 데이터와 모델을 개선하라. 튜토리얼은 새 모델·데이터·분석을 사용자가 쉽게 실행하는 창구로 계속 갱신한다.
>
> 아래에서 **[저장소 확인]**, **[첨부 논문]**, **[외부 조사]**, **[실험 제안]**을 구분한다. 구체적인 실험 예산과 설정값은 별도 표시가 없는 한 본 요청서의 시작안이지 논문에서 검증된 최적값이 아니다.

---

## 0. Agent에게 전달할 핵심 지시

다음 목표를 실제 코드·데이터·결과로 완성하라. 설치, 문헌 조사, 실행 화면 또는 노트북 개수만 늘리고 종료하지 않는다.

1. 기존 PCA·scGPT·Geneformer 결과와 환경을 보존하고, **공식 예제 재현 / 신규 HCC 적용 / 실측 섭동 예측**의 완료 상태를 분리한다.
2. 공식 튜토리얼 또는 벤치마크를 최소 하나 재현한다. 기본 경로는 scGPT 공식 annotation 예제이며, GEARS의 공식 genetic perturbation 예제를 다음 경로로 연결한다.
3. 사람 HCC 세포 유형 분류에서 **PCA, scGPT, Geneformer, scFoundation**을 동일한 평가 체계로 비교한다. scVI를 단순하지 않지만 실용적인 비-foundation 대조군으로 추가한다.
4. 간 특화 **LiverCT**와 암 도메인 **Geneformer CLcancer 또는 scGPT pan-cancer**를 평가한다. LiverCT의 query adaptation은 기존 동결 모델과 입력 정보 접근 범위가 다르므로 별도 평가 조건으로 표시한다.
5. **GEARS와 단순 수치 예측 기준선**으로 실제 유전적 섭동 반응 예측을 수행한다. 간 유래 세포주 결과와 성인 HCC 환자 결과를 구분한다. Geneformer 토큰 삭제 결과로 이 단계를 대체하지 않는다.
6. 위 기준선·특화 모델 결과와 데이터 분석에서 가설을 뽑아 **data-centric 개선**, **model-centric 개선**, **두 개선의 결합**을 비교한다. LoRA는 실제로 지원되는 모듈에 적용하고, 동결 모델 및 부분 미세조정과 대조한다.
7. 모든 연구 단계에서 실행 코드·설정·지표·실패 기록·데이터 및 모델 버전을 Git에 남기고, 동일 코드를 호출하는 사용자 튜토리얼을 갱신한다.

**기본 진행 순서**

`기존 자산 인수 → 공식 재현 → 기본 모델 비교 → 특화 모델 비교 → 오류·데이터 분석 → 데이터 개선 / 모델 개선 → 결합 실험 → 외부 최종 평가`

실측 섭동 경로는 분류 경로와 데이터·지표가 다르므로 병렬 진행할 수 있다. 특정 모델의 다운로드·환경 문제가 다른 경로 전체를 막게 하지 않는다.

**이전 요청서와의 관계:** 기존 `AGENTS.md`와 `docs/AGENT_EXPERIMENT_PLAN.md`는 초기 튜토리얼 단계의 이력이다. 이번 요청서는 이후 연구 범위를 확장한다. 기존 파일을 지우지 말고 새 요청서로의 연결을 추가한다. 이전의 “scFoundation/GEARS는 선택적 후속”, “최대 10,000세포 파일럿”을 이번 연구의 영구 제한으로 사용하지 않는다. 다만 데이터 접근·자원 제약은 실제 확인 후 기록한다.

---

## 1. 현재까지 확인된 상태와 출발점

### 1.1 재사용할 실제 자산

**[저장소 확인]** 기준 커밋의 README, `reports/final_summary.md`, `reports/data_card.md`, 환경 기록 및 주요 실행 코드를 확인했다. 기록된 상태는 다음과 같다. [G1–G7]

| 항목 | 현재 기록 | 이번 단계에서의 처리 |
|---|---|---|
| 서버 환경 | G1, NVIDIA A30 24 GiB × 4, scGPT/Geneformer 격리 환경 | 기존 환경 재사용. 현재 GPU 사용자는 다시 확인하고 다른 작업을 중단하지 않는다 |
| HCC 데이터 | GSE149614, 71,915세포 × 25,712유전자, 원 count·metadata·변환 파일 | 전체 재다운로드하지 않는다. 기존 manifest와 경로 활용 |
| 분류 파이프라인 | PCA, 동결 scGPT, 동결 Geneformer + supervised logistic head | 연구 기준선으로 보존·확장 |
| 비교 규모 | 64세포 smoke, 2,000세포 pilot, 10,000세포 비교 | smoke는 기능 확인에만 사용. 연구 분석은 환자 단위·전체 적격 데이터로 확장 |
| KG 연계 | Reactome 특징 결합, 크기/유전자 참여 분포 대조 | 기존 음성 결과 유지. 오류 분석 없이 동일 결합을 반복하지 않는다 |
| 가상 삭제 | train hepatocyte 100세포의 ALB/HNF4A 독립 토큰 삭제 | 표현 민감도 탐색으로 유지. 실측 반응 예측과 분리 |
| 튜토리얼 | 6개 노트북의 새 kernel Run All 완료 기록 | 전면 재작성하지 않고 연구 코드의 사용 예제로 갱신 |
| 미완료 | 공식 원 논문/예제의 정량 재현, scFoundation·GEARS 실행, fine-tuning, 외부 HCC 평가, 실측 섭동 예측 | 이번 요청의 주요 보완 대상 |

서버 경로는 현재 설정을 기준으로 사용한다.

```text
코드: /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
원본: /mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE149614
자산: /mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver
환경: source envs/activate.sh scgpt
      source envs/activate.sh geneformer
```

환경 기록은 특정 시점의 관측이다. A30 네 장이 지금 비어 있거나 한 작업이 96 GiB를 자동으로 사용할 수 있다는 뜻이 아니다.

### 1.2 현재 결과를 올바르게 읽기

**[저장소 확인]** 10,000세포 subset 중 동일한 두 test 환자의 1,240세포를 평가한 기록이다. [G2]

| 입력 표현 / 방법 | Macro-F1 | Balanced accuracy | 관측된 peak process GPU |
|---|---:|---:|---:|
| PCA + logistic regression | 0.9566 | 0.9498 | CPU |
| scGPT whole-human 동결 + logistic regression | 0.9527 | 0.9496 | 약 0.72 GiB |
| Geneformer-V2-104M 동결 + logistic regression | 0.9456 | 0.9463 | 약 2.50 GiB |
| scGPT + Reactome | 0.9436 | 0.9428 | 특징 결합/분류 단계 별도 |
| scGPT + matched random gene sets | 0.9531 | 0.9495 | 특징 결합/분류 단계 별도 |

위 결과는 **6개 broad lineage 분류, 한 환자 분할, 동결 표현 + 학습된 head**에서 얻은 것이다. 다음과 같이 해석한다.

- 이 조건에서는 foundation embedding과 단순 Reactome 결합의 이득이 확인되지 않았다.
- 이를 “PCA가 모든 간암 task에서 우수하다”, “foundation model 또는 KG는 불필요하다”로 확대하지 않는다.
- 낮은 동결 추론 VRAM은 전체 미세조정이나 긴 sequence의 메모리 요구량을 보장하지 않는다.
- 현재 실행 시간은 모델별 측정 범위가 다르므로 그대로 효율 순위를 만들지 않는다. 이후에는 데이터 처리·모델 추론·head 학습·전체 시간을 각각 기록한다.
- ALB는 100세포 중 65세포, HNF4A는 46세포에서 삭제 대상 토큰이 있었다. 토큰이 없는 세포의 거의 0인 변화는 생물학적 무효과의 증거가 아니다.

**출발 가설:** broad lineage 분류가 이미 상대적으로 쉬운 문제일 수 있고, 데이터·토큰 표현·환자 차이·head 학습이 관측 차이에 영향을 주었을 수 있다. 이는 아직 원인이 확정된 설명이 아니라 후속 분석할 가설이다.

### 1.3 기존 코드에서 최소한으로 확장할 부분

현재 `scripts/common.py`는 데이터셋, 일부 seed, 저장 위치를 특정 실험에 맞춰 사용한다. 새 cohort와 교차검증에 필요한 `dataset_id`, `split_id`, `seed`, `evaluation_regime`, `run_id`만 설정으로 분리하라. 기존 CLI는 유지한다. [G4]

현재 환자별 macro-F1은 전체 train class 목록을 사용하여 그 환자에 없는 class도 0으로 포함한다. 기존 수치는 수정 없이 보존하고, 새 보고서에는 **공통 ontology의 pooled macro-F1**과 **support를 명시한 환자별 점수**를 구분한다. 환자마다 존재하지 않는 세포 계열 때문에 낮아진 값을 모델 실패로만 해석하지 않는다.

`03_scgpt_annotation.py`는 길이 1,200의 공식 random token sampling 경로, `geneformer_utils.py`는 V2 tokenizer 및 second-to-last-layer CLS를 사용한다. 이를 현재 기준선으로 고정하고, 길이·pooling 변경은 별도 모델 입력/표현 실험으로 기록한다. 정상 동작을 확인한 코드를 근거 없이 결함으로 취급하지 않는다. [G5–G6]

---

## 2. 첨부 논문을 연구 설계에 반영하는 방식

**[첨부 논문]** *Toward AI Virtual Cells for Hepatology: Representation, Generation, Dynamics, and Intervention in Single-Cell Models*는 세 가지 경로를 구분한다. [R1]

| 논문의 구분 | 의미 | 이번 연구에서의 역할 |
|---|---|---|
| Generative representation | 관측 세포 상태의 표현·잡음 처리·분포 모델링 | 분류/표현 비교, scVI와 foundation embedding 기준선 |
| Dynamics and transport | 시간·조건 사이 상태 전이 추정 | 이번 핵심 범위 밖. 시간·계보 정답 없이 pseudotime을 실제 치료 반응으로 바꾸지 않는다 |
| Pretrained / foundation transfer | donor·질병·플랫폼 등 새로운 조건으로의 전이 | 환자 단위 개발 평가와 외부 HCC cohort 평가 |
| Cross-cutting perturbation test | 미관측 개입에 대한 실제 반응 예측 | GEARS 등 실측 perturbation 학습·평가 경로 |

논문은 세포 분류, 관측 상태 재구성, plausible profile 생성만으로 독립적인 간 개입 반응 예측을 입증할 수 없다고 구분한다. 특히 22–25쪽과 Figure 5는 **개입·생물학적 맥락·용량을 무엇까지 제외했는지**, **반응 방향·차등 발현 회복·불확실성을 무엇으로 평가했는지**를 강조한다. Figure 4는 간 특화 근거와 일반 single-cell 근거를 구분하고, Figure 6은 표준 평가 이후 예측–실험–갱신으로 발전하는 순서를 제안한다.

이번 연구에서는 이 구분을 그대로 유지한다. “간 데이터에서 돌아감”, “HCC 세포를 분류함”, “간 유래 세포주의 미관측 섭동을 예측함”, “환자 치료 반응을 검증함”을 같은 성과로 표기하지 않는다.

**확인 범위:** 첨부된 본문·그림·참고문헌을 검토했다. 논문에서 언급하는 Supplementary Table 1A/1B/2 원본은 별도로 제공되지 않았으므로 전체 목록을 확인했다고 주장하지 않는다. 아래 LiverCT와 암 특화 체크포인트는 본문 요약을 넘어 **외부 원전·공식 코드에서 추가 확인한 후보**다.

---

## 3. 비교 문제를 세 가지로 분리한다

### A. 사람 HCC 세포 유형 분류와 표현 전이 — 주 경로

```text
입력: raw scRNA-seq + 허용된 기술 metadata
출력: 사전에 합의한 세포 유형, 확률/기권 여부
평가: donor 분리 개발 평가 + 별도 HCC cohort
```

PCA, scGPT, Geneformer, scFoundation, scVI와 적합한 간/암 특화 방법을 비교한다. 초기 label은 기존 6개 broad lineage를 유지한다. finer subtype이나 malignant state는 독립적으로 검증 가능한 annotation이 있을 때 별도 task로 연다.

### B. 실측 genetic perturbation 반응 예측 — 별도 주 경로

```text
입력: 대조군 상태 + 유전적 개입의 표적/종류 + 이용 가능한 맥락
출력: 처리 후 발현 또는 대조군 대비 발현 변화
평가: 학습에 없는 perturbation / 조합, 실제 측정 response
```

GEARS, no-change/mean/선형 기준선을 우선 비교한다. scGPT/scFoundation의 task-specific perturbation 모델은 공식 구현과 입력 정합성이 확인될 때 비교한다. Geneformer embedding + 새 response decoder는 **새로 구성한 확장 모델**이며 원래 Geneformer의 기본 기능으로 표시하지 않는다.

### C. Chemical perturbation / dose response — 조건부 확장

실제 약물·용량·시간별 전사체가 확보되면 PRnet, scVIDR 등 task에 맞는 방법을 검토한다. 유전자 KO와 약물 처리는 동일한 개입이 아니므로 GEARS에 약물명을 gene ID처럼 넣지 않는다. 이 경로의 착수 조건은 8절에 명시한다.

> **하나의 종합 점수로 모델을 줄 세우지 않는다.** 분류 F1, embedding shift, 발현 delta error, 약물 독성 분류는 서로 다른 정답과 출력이다. task와 평가 조건별 leaderboard를 만든다.

---

## 4. 기본 모델과 특화 모델의 실행 우선순위

### 4.1 기본 비교군

| 모델 | 분류 경로 A | 실측 섭동 경로 B/C | 이번 실행 요구 |
|---|---|---|---|
| PCA | train-only 변환 + 공통 선형 head | PCA만으로는 predictor가 아님. 개입 특징을 받는 ridge/저차원 response 모델을 별도로 구성 | 모든 대응 task의 단순 기준선 유지 |
| scGPT whole-human | 기존 동결 표현 재사용, 이후 부분/LoRA 미세조정 | 공식 perturbation task가 지원하는 설정에 한해 별도 학습 | 필수 |
| Geneformer-V2-104M | 기존 V2 동결 표현, 이후 task 미세조정 | 토큰 삭제는 표현 탐색. 수치 발현 예측에는 별도 학습 구조 필요 | 분류 필수, 토큰 삭제는 부가 분석 |
| scFoundation | 공식 gene 순서·read-depth 규칙으로 cell embedding + 공통 head | 공식 하위 모델/GEARS 결합은 별도 확장 | 새 분류 기준선으로 실행 시도 및 결과 확보 |
| GEARS | cell classifier로 강제하지 않음 | 실제 유전적 perturbation 데이터와 gene graph로 학습 | 공식 재현 + 간 유래 실측 데이터 적용 |
| scVI / 필요 시 scANVI | counts 기반 비-foundation 비교군 | 모델 자체를 미관측 개입 predictor로 부르지 않음 | scVI 추가. LiverCT adaptation 비교에는 scANVI/scArches 활용 검토 |

모델별 공식 입력과 task 범위를 지킨다. Geneformer에 scGPT의 HVG-only/binned matrix를 넣거나, scFoundation의 gene order를 임의 변경하지 않는다. 모델마다 입력 유전자 범위가 다르면 **실용 파이프라인 비교**라고 해석한다. 같은 feature 조건을 사용한 보조 실험은 입력 계약을 훼손하지 않는 범위에서만 수행한다. [E1–E5]

### 4.2 간/암 특화 후보 — 실제 실행 후보와 제외 후보를 구분

| 후보 | 특화의 정확한 범위 | 적용·비교 계획 | 주의점 |
|---|---|---|---|
| **LiverCT / uniLIVER** | 정상 사람 간 reference를 이용한 세포 유형·비전형 상태·hepatocyte zonation mapping | **최우선 간 특화 분류 비교군**. 공식 demo 후 HCC 데이터에 적용 | HCC 전용 foundation simulator가 아니다. 기본 query adaptation은 별도 평가 조건 |
| **Geneformer-V2-104M_CLcancer** | 암 transcriptome에 대한 continual learning | 기본 V2-104M과 같은 분할·head로 비교하는 우선 암 도메인 후보 | 암 특화이지 간암 전용은 아님. 사전학습 중복 별도 기록 |
| **scGPT pan-cancer** | 다양한 암 데이터의 pretrained checkpoint | 전체 모델 inventory에 포함. 추가 비교 여력이 있거나 CLcancer 접근이 막히면 실행 | 전체 human 모델과 입력·모델 차이를 기록. HCC 전용이라고 쓰지 않음 |
| **scVIDR** | mouse liver TCDD dose response 등의 직접 간 적용 근거 | chemical 경로의 작고 명확한 방법론 재현 후보 | mouse·독성·동일 약물 용량 전이를 인간 HCC 약효 예측으로 확대하지 않음 |
| **PRnet** | 화학 섭동 반응; PHH·간 유래 세포주를 포함한 평가 | chemical 데이터가 확보된 경우 우선 검토 | liver-included이지 HCC 전용 모델은 아님 |
| **DeepLiver** | mouse liver의 enhancer sequence/chromatin와 zonation 조절 | 관련 데이터가 생기면 별도 조절/멀티오믹스 task | 현재 RNA-only 분류표에 억지로 넣지 않음 |
| **LiverDCP** | 간 질환·세포·단백질 맥락의 관계/단백질 표현, preprint | 후속 KG 특징의 가능성만 inventory에 기록 | turnkey 세포 분류/처리 후 발현 predictor로 취급하지 않음 |

**[외부 조사]** LiverCT 공식 코드와 가중치 다운로드 경로, HCC 예제 `Lu2022_subsampled.h5ad`가 공개되어 있다. 이 Lu2022는 현재 사용하는 GSE149614 원 연구이므로 **독립적인 외부 검증 데이터가 아니다**. 공식 `cell_states_annot()`는 query 데이터를 `SCANVI.load_query_data()`로 연결하고 기본 20 epoch 학습 후 분류한다. 평가 시 query의 비라벨 발현값을 사용하는 방법임을 명시하라. [E6]

**실행 최소 범위:** A 경로의 기본 모델 비교를 완료한 뒤 **LiverCT + 암 특화 체크포인트 1개**를 평가한다. 나머지 후보를 모두 설치하거나 미리 대규모 KG를 재구축할 필요는 없다. 실행할 수 없는 후보는 다운로드·라이선스·입력 부적합 등 이유를 기록한다. “간 특화”라는 이름만으로 후보를 추가하지 않는다.

---

## 5. 데이터와 분할 계획

### 5.1 사용할 데이터

| 데이터 | 역할 | 접근·정합 확인 | 해석 범위 |
|---|---|---|---|
| **GSE149614** | 기존 주 개발 cohort, 분류·오류 분석 | 기존 raw/processed/metadata 재사용 | 환자 HCC 및 주변 조직. observational 데이터 |
| **GSE151530의 HCC subset** | 우선 외부 최종 평가 cohort | HCC/iCCA 혼재. histology, patient, 반복 biopsy, label·counts 확인 | 후속 치료/시점 차이는 metadata로 다루되 자동 counterfactual 정답으로 사용하지 않음 |
| **GSE189903의 HCC subset** | 외부 cohort 대체 또는 추가 독립 검증 후보 | multiregion·iCCA 혼재·환자 중복 확인 | 모든 부위는 동일 donor에 묶음 |
| **공식 scGPT annotation 데이터** | 공식 방법론 재현 | 공식 tutorial 제공 파일·분할·체크포인트 고정 | HCC 성능이 아니라 공식 예제 실행/정량 재현 |
| **GEARS 공식 Norman 등** | genetic perturbation 기준 방법 재현 | 공식 데이터·simulation split·개입 종류 확인 | 비간 데이터의 방법론 재현. Norman의 CRISPRa를 KO로 바꾸어 표기하지 않음 |
| **GSE264667의 HepG2 subset** | 간 유래 실측 genetic perturbation | guide→target, 개입 방식, control, day, batch, count 확인 | HepG2는 hepatoblastoma 유래. 성인 HCC 환자 결과와 분리 |
| scVIDR 공식 mouse liver / PRnet 대응 데이터 | conditional chemical 경로 | dose·time·source control·replicate·접근 조건 확인 | liver evidence의 종류와 종을 명시 |

공식 GEO 등록정보만 확인된 신규 데이터의 파일 내부, label 품질, 구체적인 guide 처리 방식까지 확인했다고 주장하지 않는다. 실행 agent가 실제 파일을 읽어 `data_card`를 작성한다. [D1–D5]

**HCC 실측 섭동 자료 탐색:** HCC cell line, organoid, 환자 유래 시스템에서 대조군과 개입 후 scRNA-seq가 공개된 후보를 추가로 찾되, accession·counts·조건 metadata·접근성을 확인한 후보만 넣는다. “논문에 등장함”과 “지금 실행 가능함”을 구분한다. 첨부 리뷰의 sorafenib-resistant Huh7 사례는 원 논문의 데이터 제공 방식과 실험 설계부터 확인하며, 접근 가능한 accession이 없는데 임의로 만들어 넣지 않는다. [E12]

### 5.2 이미 본 test와 새로운 최종 test의 분리

**[저장소 확인]** 기존 test 환자는 `HCC02`, `HCC09`다. 이 성능이 이미 비교·공유되었으므로 이후 개선을 설계하면서 두 환자를 새 blind test처럼 취급하지 않는다. [G3]

**[실험 제안]** 다음 역할로 고정한다.

| 평가 구간 | 데이터 | 목적 |
|---|---|---|
| Legacy regression | 기존 HCC02/HCC09 및 기존 10k 설정 | 과거 결과 재현·호환성. 신선한 최종 검증 아님 |
| Development grouped CV | 기존 train+validation 8명 | 기본/특화 모델 비교, 오류 분석, 데이터·모델 선택 |
| External final | GSE151530 HCC 또는 사전에 정한 대체 cohort | 선택한 파이프라인의 최종 외부 평가 |

개발 CV는 8명에 대한 **donor-grouped 4 folds**를 기본 시작안으로 한다. 각 outer train의 head·하이퍼파라미터 선택은 다시 donor가 겹치지 않는 내부 validation으로 수행한다. fold는 성능 확인 전 저장하고 모든 모델에 동일하게 적용한다. class support가 부족하면 모델 출력이 아니라 metadata 기준으로 split 정책을 먼저 조정·고정한다.

개발 OOF 결과는 개선 아이디어를 얻는 용도로 사용하므로, 반복 실험 이후 이를 완전히 독립적인 최종 성능이라고 주장하지 않는다. 전체 10명 LOPO는 추가 진단용으로 가능하지만 외부 cohort 검증의 대체가 아니다.

### 5.3 외부 cohort를 소비하지 않는 원칙

외부 cohort에서는 accession, histology, 환자 중복, count 형식, label ontology, 사용 권한을 확인할 수 있다. 그러나 모델별 성능을 보고 유리한 cohort·세포·라벨을 고르지 않는다. 외부 label mapping은 원 annotation과 ontology로 사전 고정하고, 후보 모델의 예측을 정답으로 사용하지 않는다.

모델·데이터 개선안·임계값·평가 지표가 정해진 후 최종 후보들을 한 번의 고정 evaluation batch로 비교한다. 결과가 나쁘면 그대로 보고한다. 이후 그 cohort를 보고 추가 개선을 한다면 **development로 전환되었음을 표시**하고 새로운 최종 holdout을 확보하거나 독립성 제한을 명시한다.

새 training cohort를 data-centric 개선에 추가할 경우, 이미 final로 지정한 cohort는 사용하지 않는다. 공개 pretrained model의 corpus와 중복 여부는 확인 가능한 범위에서 별도 조사하고, 불명확하면 `pretraining_overlap=unknown`으로 유지한다.

### 5.4 평가 데이터와 label의 현실성

처음에는 기존 10k로 연구 파이프라인을 연결하되, 그 수가 최종 데이터 규모가 되어서는 안 된다. 비교가 안정되면 **GSE149614의 전체 적격 세포 또는 사전에 정한 donor-balanced training 규모**로 확대한다. main test는 가능한 원래 세포 분포를 유지하며, 과거의 환자×라벨 층화 subset 결과와 구분한다.

metadata의 `stage`, `virus`, `site`, `patient`, QC 지표를 분석하되, 표본이 적어 원인과 환자가 완전히 겹치면 etiology·stage 일반화를 입증했다고 쓰지 않는다. 학습 task에서 허용하지 않은 true subtype·악성 여부를 모델 특징으로 넣지 않는다.

새로운 세포 subtype 정답이 없다면 unsupervised cluster 또는 후보 모델의 label을 ground truth로 만들지 않는다. Tumor tissue의 모든 세포를 malignant로 취급하지 않고, CNV 추정/marker 기반 악성 label은 독립 원 annotation과 구별된 weak-label 탐색으로 남긴다.

---

## 6. Phase R — 공식 튜토리얼/벤치마크 재현

### 6.1 기본: scGPT 공식 annotation 예제

공식 tutorial의 Multiple Sclerosis 예제 데이터를 사용하여 **공식 학습·평가 흐름**을 재현한다. 이 결과를 HCC 성과 표에 합치지 않는다. [E1]

실행 순서:

1. 공식 코드 revision, tutorial 버전, 데이터 파일, pretrained weights/config/vocab, 원 평가 split을 저장한다.
2. 기존 서버 환경에서 작은 smoke로 입출력만 확인한다.
3. 원 tutorial의 학습·평가를 실행하고 최종 score·예측·실행 자원을 남긴다.
4. 원문/공식 예제에 비교 가능한 숫자가 있으면 같은 지표로 차이를 보고한다. 원 숫자 또는 동일한 split이 없으면 “공식 파이프라인 실행 재현”으로 표시한다.
5. 설치를 위해 변경한 dependency, batch 축소, attention backend, preprocessing 변경은 patch와 영향 범위를 기록한다. 수정한 방법을 원 방법과 같다고 하지 않는다.

**재현 상태 표준:**

- `smoke_passed`: 작은 입력이 끝까지 동작.
- `official_pipeline_reproduced`: 공식 데이터·절차를 실제 실행.
- `quantitative_reproduction`: 원 설정·지표·비교 대상이 확인되고 차이를 보고함.
- `modified_reproduction`: 기능·자원상 수정이 있어 원 설정과 차이를 명시.

설치 성공을 quantitative reproduction으로 올리지 않는다. 비교 허용 오차가 필요하면 run 전에 원 결과의 변동·지표 정의를 보고 정한다. 근거 없는 “90% 이상이면 재현” 같은 통과선을 만들지 않는다.

### 6.2 GEARS 공식 예제는 실측 섭동 경로의 입구

`norman` 또는 접근 가능한 공식 데이터에서 `PertData`, `simulation` split, `GEARS` 학습·평가 흐름을 재현한다. 공식 예제의 device 번호를 그대로 복사하지 말고 할당된 GPU를 사용한다. [E4]

이 단계는 분류 경로가 완료될 때까지 무조건 기다리지 않아도 된다. 다만 공식 데이터·split·평가가 정리되기 전에 HepG2 전용 수정부터 시작하지 않는다.

**산출물:** `reports/research_phase2/official_reproduction.md`, upstream/data/checkpoint manifests, 실제 학습 log·prediction·metric·patch.

---

## 7. Phase A/S — 기본 모델과 특화 모델 비교

### 7.1 기본 모델 비교 A

동일한 donor fold와 평가 세포에서 다음을 수행한다.

| 실험 ID | 모델 | 학습 대상 | 비교 목적 |
|---|---|---|---|
| A00 | PCA + logistic regression | train HVG/scale/PCA/head | 기존 단순 기준선 |
| A01 | scGPT whole-human + logistic head | head만 | 기존 foundation 기준선 |
| A02 | Geneformer-V2-104M + logistic head | head만 | 다른 입력 표현/사전학습 비교 |
| A03 | scFoundation cell embedding + logistic head | head만 | full-gene/read-depth 표현 비교 |
| A04 | scVI latent + 동일 계열 head | train counts로 scVI, head | foundation이 아닌 count model과 비교 |

scVI에 batch 변수를 줄 때 inductive inference에서 처음 보는 donor batch를 어떻게 처리하는지 먼저 명시한다. test donor를 포함하여 모델을 다시 학습하는 방법은 A의 strict inductive 점수와 섞지 않고 아래 adaptation 표로 이동한다. PCA의 gene 선택·scaler·PCA는 fold별 train에서만 fit한다.

현재 단순 분산 상위 gene 선택을 사용한 A00을 보존하고, 다른 HVG 방식은 별도 sensitivity 또는 data-centric 변경으로 비교한다. 비교 도중 조용히 PCA만 약하게 만들거나 foundation 모델에 더 많은 label을 제공하지 않는다.

### 7.2 특화 모델 비교 S

**S01 — 암 도메인 같은 계열 비교:** 우선 Geneformer-V2-104M_CLcancer를 기본 V2-104M과 같은 tokenizer 버전·representation extraction·head·fold로 비교한다. tokenizer/config가 실제로 다르면 그 차이를 기록한다. scGPT pan-cancer도 같은 원칙으로 추가 가능하다. 이는 “암 도메인 특화” 효과의 실용 비교이지 사전학습 데이터 크기·중복·학습 비용을 통제한 순수 인과 실험은 아니다. [E2–E3]

**S02 — LiverCT native mapping:** 공식 demo를 실행한 뒤 같은 HCC 데이터의 예측을 얻는다. 정답을 보지 않고 `pred_lv1`/`voting_lv2_1` 등 원 출력과 6개 broad lineage의 mapping을 정한다. 평가 불가능한 항목·Unknown·새로운 세포 유형은 숫자와 coverage를 남긴다. [E6]

LiverCT 실행에는 다음 조건을 적용한다.

- 공식 구현이 query의 비라벨 데이터를 이용해 학습하는 **query-adaptive** 설정임을 결과 표에 표시한다.
- 공유된 원 모델 폴더를 수정하지 않는다. 코드의 `my_surgery_model` 저장 경로는 run/fold별 복사본 또는 별도 경로로 분리한다. 한 fold의 adaptation 결과를 다음 fold의 시작 가중치로 사용하지 않는다.
- `finetune_epoch=0`으로 바꾸는 것만으로 엄밀한 inductive 방법이 된다고 가정하지 않는다. zero-adaptation 경로의 지원·batch 처리·출력 의미가 확인되면 별도 실험명으로 평가하고, 아니면 strict 표에서는 `N/A`로 둔다.
- 비전형/중간 상태·zonation 출력은 해당 gold label이 없으면 탐색적 분석이다. high score를 악성 세포나 치료 반응 정답으로 쓰지 않는다.
- 최종 외부 cohort에서 native adaptation을 허용한다면, **비라벨 target 발현만 사용**, adaptation epoch/조건은 개발 단계에서 고정하고 그 비용을 기록한다. 외부 label로 adaptation 중단·선택을 하지 않는다.

### 7.3 서로 다른 정보 접근 범위를 분리하는 두 표

**표 A-I: Inductive**

Train label·train statistics로 학습하고, test는 추론 시 자신의 관측 발현만 제공한다. target cohort 전체를 이용한 추가 학습·batch fitting은 하지 않는다.

**표 A-Q: Query-adaptive / reference mapping**

Target의 비라벨 발현을 이용한 adaptation이 허용된다. LiverCT native, 동일한 정보 조건을 허용한 scANVI/scArches 등 적절한 비교군을 둔다. 각 방법의 reference 데이터와 학습 label 양도 표시한다. 가능한 방법만 비교하며 지원되지 않는 모델을 억지로 변환하지 않는다.

참고로 두 표의 숫자를 병기할 수는 있으나, A-Q의 더 높은 점수를 “구조가 우수함”의 근거로 단정하지 않는다. 이용한 정보량과 native head가 다르다는 제한을 명시한다.

### 7.4 분류 평가 지표

**주 지표:** 사전에 고정한 공통 label 범위에서 pooled macro-F1.

**함께 보고:** balanced accuracy, per-class precision/recall/F1/support, donor별 성능, confusion matrix, evaluation coverage/unknown, rare class recall, 자원·시간.

환자별 F1은 해당 환자에서 관측되거나 예측된 class의 support를 함께 제시하는 점수와 공통 label 점수를 구분한다. 존재하지 않는 class의 zero 처리 규칙을 명시한다. 다른 방법이 어려운 세포를 Unknown으로 제외한 경우 answered-only 점수와 전체 coverage를 함께 보고한다.

확률적 confidence를 주장할 때는 validation으로 temperature scaling 등의 보정 여부를 결정하고 ECE/Brier 또는 risk–coverage를 기록한다. embedding distance 자체를 calibrated uncertainty라고 부르지 않는다. 배치 효과 제거 후 UMAP이 보기 좋아지는 것을 주 성능 지표로 사용하지 않는다.

---

## 8. Phase B/C — 실제 섭동 반응 예측

### 8.1 공식 GEARS benchmark 재현

공식 split을 유지하여 원 실행을 먼저 확보한다. 기존 논문에 등장하는 train/test DE gene 처리 등이 연구용 평가 정책과 다르면 원 재현 결과와 strict 추가 결과를 별도 표로 둔다. 원 방법을 조용히 수정한 뒤 원 수치를 재현했다고 주장하지 않는다.

### 8.2 GSE264667 HepG2에 적용

**[외부 조사]** GSE264667은 Jurkat/HepG2 pooled CRISPR single-cell screen이며, 공개 등록에는 dual sgRNA 설계와 day-7 readout이 기재되어 있다. 각 guide가 어떤 gene을 어떤 방식으로 조작했는지는 실제 metadata·원문 방법으로 확인해야 한다. dual sgRNA를 자동으로 두 gene 조합이라 해석하지 않는다. HepG2의 기원은 성인 HCC와 구별한다. [D4–D5]

필요한 최소 data contract:

```text
cell_id, dataset_id, cell_line, sample_id, replicate_id,
intervention_type, target_gene_ids, guide_ids,
control_type, time, batch, library_size
+ raw_counts / 공식 전처리된 expression
```

각 필드가 없으면 `unknown`으로 기록한다. 다른 sample의 대조군을 무조건 합치지 않는다. non-targeting control, guide assignment, guide efficiency/복수 감염, low-count/QC의 처리 규칙을 데이터 카드에 명시한다.

**실험 분할**

- 같은 target의 다른 guide는 하나의 intervention group에 묶는다.
- train/validation/test를 perturbation 단위로 분리한다. 처리 세포를 무작위로 나누는 split만으로 unseen perturbation을 주장하지 않는다.
- 대조군 입력이 허용되는 benchmark인지, 평가 replicate의 control을 제공하는지 명시한다. test 처리군 response는 어떤 학습·feature selection·graph 계산·retrieval에도 들어가지 않는다.
- GEARS의 coexpression graph, HVG, scaling, early stopping은 train 범위에서 산출한다. 외부 GO/PPI prior는 허용하되 버전과 범위를 저장한다.
- 조합 데이터가 충분하지 않으면 단일 유전자 holdout task로 한정한다. single-only 학습으로 조합 성능을 보장하지 않는다.
- 기본 GEARS는 cross-cell-type transfer를 위한 모델이 아니므로 HepG2와 Jurkat을 섞어 학습하고 transfer 성능이라고 발표하지 않는다. [E4]

### 8.3 B 경로의 기준선과 모델

| ID | 방법 | 핵심 조건 |
|---|---|---|
| B00 | No-change / matched control mean | 변화가 없다는 예측. 반드시 포함 |
| B01 | Train-perturbation mean response | train intervention의 평균 효과. test 효과를 이용하지 않음 |
| B02 | PCA/저차원 + ridge response predictor | unseen target에도 정의되는 외부 gene/pathway 특징을 사용. PCA만으로 predictor라고 하지 않음 |
| B03 | GEARS | 공식 genetic response 학습, 실제 control/treated 데이터 |
| B04 | scGPT의 지원되는 perturbation 경로 | 입력·task·gene 매칭을 확인한 별도 task fine-tuning |
| B05 | scFoundation + 공식 response/GEARS 결합 | B03과 데이터·분할을 맞추고 pretrained 표현의 추가 이득 평가 |
| 선택 | Geneformer embedding + response decoder | 새 결합 모델임을 명시. baseline 완료 후 필요 시 수행 |

B04/B05를 실행할 수 없는 gene 패널·환경일 경우 이유와 범위를 기록한다. 임의의 0 채우기 또는 알려지지 않은 target one-hot으로 형식만 맞춰 결과를 만들지 않는다. B00–B03을 우선 완결하고, 적합한 foundation 확장 1개를 추가한다.

B02는 예를 들어 고정된 intervention feature에서 train response의 PCA coefficient를 예측한 뒤 gene space로 복원하는 ridge 모델로 구성할 수 있다. **test target의 실제 response에서 intervention embedding을 만드는 것은 금지**한다. one-hot만 쓰면 완전히 미관측 target의 효과를 학습할 수 없다는 제한을 명시한다. 조합 additive baseline은 각 구성 단일 개입이 train에 실제 존재하는 하위집합에서만 사용한다. [E13]

### 8.4 B 경로의 평가

동일한 세포를 개입 전후 모두 측정한 paired data가 아니라면 서로 다른 세포를 임의로 짝지어 paired error를 계산하지 않는다. 기본은 조건별 pseudobulk 또는 집단 통계다.

```text
Δμ(condition) = μ(treated, condition) − μ(matched control)
```

**주 평가:** 사전 정의 gene universe의 perturbation별 Δ-expression RMSE/MAE와 조건별 평균. 큰 세포 수의 조건이 자동으로 전체 점수를 지배하지 않도록 condition-macro 집계도 제공한다.

**보조 평가:** Δ-expression 상관, response sign, DE gene retrieval AUPRC/precision@k, 효과 크기별·guide coverage별 결과, 직접 표적 gene 제외 민감도 분석. 상관이 정의되지 않는 거의 0인 response는 임의로 0/1 처리하지 말고 수와 정책을 기록한다.

실제 test response로 계산한 DE genes는 **평가자가 정의하는 정답**으로 사용할 수 있다. 단, 해당 목록·logFC·순위를 입력 feature, gene selection, loss 가중치 선택 또는 model selection에 넘기지 않는다. all-gene 평가를 기본으로 하고 test DE-subset 평가는 evaluator-only임을 명시한다.

독립 replicate가 있으면 replicate/pseudobulk 단위로 평가·불확실성을 추정한다. 세포 수를 생물학적 반복 수로 간주하지 않는다. replicate가 없으면 biological reproducibility와 통계 검정 범위의 제한을 명시한다. 분포 생성 모델을 비교할 때만 적합한 distribution metric을 추가하고, 평균 predictor와 그 출력 능력을 혼동하지 않는다.

### 8.5 C 경로의 착수 조건과 제한

다음이 확보될 때만 chemical response 경로를 연다: raw/processed expression, control, 약물 정체성, dose, time, cell context, 이용 가능한 replicate, 원 split 또는 재현 가능한 split.

- **scVIDR 재현:** mouse liver의 TCDD dose/cell-type response를 재현하는 작은 경로. 특정 cell type의 treated response holdout과 control 제공 여부를 구분한다. 새로운 약물 일반화나 HCC 효과로 해석하지 않는다. [E7]
- **PRnet 적용:** 해당 모델의 공개 데이터·입력 모달리티·chemical descriptor에 맞춰 약물 holdout 또는 context holdout을 평가한다. 데이터/가중치 링크가 막히면 공식 대체 저장소를 확인하고 접근 실패를 기록한다. [E8]
- 사람 HCC/간 유래 chemical screen을 택하면 drug×dose×time×context 단위로 분할한다. 구조 유사도 scaffold holdout은 구조와 충분한 compound 수가 있을 때 별도 난이도로 추가한다.
- bulk PHH, mouse liver, HepG2/C3A, 성인 HCC 환자 데이터를 단일 같은 task로 합치지 않는다.

**범위 관리:** A/S/B의 실행을 지연시키며 C 모델을 전부 설치하지 않는다. 첫 연구 확장에서는 **실측 genetic 경로 1개를 완결**하고, chemical 경로는 데이터 적합성에 따라 PRnet 또는 scVIDR 하나부터 진행한다. HCC 실측 response 자료가 부족하다는 결론도 근거와 함께 남기되, 이미 가능한 분류·간 유래 실측 실험은 계속한다.

---

## 9. Phase I — 개선 전에 반드시 얻을 데이터·모델 통찰

기준선과 특화 모델의 **개발 OOF 결과**를 이용해 다음 분석을 수행한다. 외부 최종 score는 보지 않는다.

| 분석 축 | 확인할 내용 | 가능한 후속 가설 — 아직 결론 아님 |
|---|---|---|
| 환자·조직·label | 환자별 세포 수, rare class, tissue/virus/stage의 분포와 혼동 | 특정 donor의 과대표집 또는 label 범위 차이 |
| 기술/QC | UMI, 검출 gene, mitochondrial 비율, doublet 후보, platform | 낮은 정보량 또는 기술적 domain shift |
| gene 정합 | symbol/Ensembl 매핑, 중복, 버전, count coverage | 특정 lineage의 중요한 gene이 더 많이 누락됨 |
| 토큰 표현 | 길이, truncation, random sampling, 중요 gene 포함률 | scGPT 1200 제한/Geneformer rank 입력이 subgroup에 미치는 영향 |
| 오류 구조 | 전체/환자/class별 confusion, confidence, 모델 간 공통·고유 오류 | broad task의 포화, subtype 경계 또는 부정확한 confidence |
| KG 특징 | pathway 크기·중복·희소성·정규화, 실제 prediction 변경 | 단순 concat의 차원 증가/중복이 이득을 상쇄 |
| 섭동 신호 | target/guide, response magnitude, viability·cell composition, control batch | 약한 perturbation, control mismatch, 일반 stress가 개입 신호를 가림 |
| 학습량 | donor 수 증가 vs 동일 donor 내 세포 수 증가 | 세포 수보다 새로운 환자·맥락 추가가 더 유용할 가능성 |

생물학적 해석을 위해 알려진 hepatocyte zonation·면역·stromal 경로를 참고할 수 있다. 그러나 marker와 일치함만으로 모델이 그 기전을 학습했거나 causal effect를 재현했다고 결론 내리지 않는다. [R1]

**실험 전에 작성할 hypothesis registry**

```text
hypothesis_id
observed_pattern / evidence_run_ids
candidate_explanation
planned_change
fixed_components
expected_affected_subgroup
primary_metric_and_direction
negative_control
success_or_rejection_rule
status / conclusion
```

최초 개선 라운드는 **우선순위가 높은 data 가설 최대 3개, model 변경 계열 최대 3개**로 제한한다. 근거 없이 전처리·모델·loss·데이터를 동시에 바꾸지 않는다. 하나의 결과가 부정적이면 이를 삭제하거나 다른 지표로 성공처럼 바꾸지 않는다.

---

## 10. Phase D — Data-centric 개선

### 10.1 고정 조건

선택한 baseline backbone, head family, 모델 최적화 설정을 고정한다. 데이터 변경으로 epoch당 sample 수가 바뀌면 총 update 수·본 example 수를 함께 기록한다. train 데이터가 많아졌다는 효과와 동일 데이터의 품질 개선 효과를 분리한다.

평가 데이터는 원칙적으로 고정한다. QC 정책이 test에도 적용되는 deployable pipeline이면 전체 모집단 점수/coverage와 공통 적격 세포 점수를 둘 다 보고한다. 어려운 test 세포를 제거한 뒤 남은 셀의 F1만 높아진 것을 모델 개선으로 표기하지 않는다.

### 10.2 우선 실험 후보

| ID | 실험 | 구체적 조작 | 대조·주의점 |
|---|---|---|---|
| D01 | 정합·품질 개선 | 개발 데이터에서 확인된 ID 누락·중복·QC 문제를 원자료 기준으로 수정 | label을 후보 예측에 맞춰 고치지 않는다. mapping 수정과 QC 변경을 한꺼번에 하지 않음 |
| D02 | 학습 표본 구성 개선 | donor-balanced sampling 또는 donor×class sampling, rare class 학습 전략 | test 분포 유지. 동일 셀 수의 무작위 sampling 대조. class_weight 변경은 model/loss 요인으로 별도 기록 |
| D03 | 학습 데이터 다양성 | 외부 최종 cohort와 분리된 reference/development cohort 추가 | total cell 수를 맞춘 within-cohort 확대 대조와 donor 수 효과 비교 |
| D04 | 지식 특징 개선 | 오류 유형에 맞는 pathway/TF prior 선택·압축·정규화 또는 신뢰도 조정 | 같은 차원·규모의 random prior, pathway-only, embedding-only 대조 |
| 선택 | 현실적인 증강 | train-only count thinning 등 측정 과정에 근거한 작은 강건성 실험 | 생물학적 perturbation label을 합성하지 않음. 원 count·원 label 유지 |

실제 관측 문제와 연결되는 항목부터 **최소 2개를 실행**한다. 예를 들어 ID 매핑에 문제가 없으면 억지로 D01을 만들지 말고 D02/D03/D04에서 근거가 있는 가설을 고른다.

**KG의 위치:** 기존 KG export가 있으면 정체성·species·source·관계 타입을 보존해 사용한다. 없으면 이미 구현된 Reactome을 재사용한다. 새 대형 KG 구축이나 RAG 서비스를 이번 모델링의 선행조건으로 만들지 않는다. Reactome membership, TF regulation, 예측된 KG link는 측정된 세포 상태·섭동 정답과 구분한다.

LiverDCP와 같은 context-specific 단백질 prior를 활용한다면, **test의 true cell type을 알아야 선택 가능한 embedding을 몰래 입력하지 않는다.** 허용된 context 또는 train에서 구축한 예측 context만 사용한다. true context를 쓴 값은 oracle 진단으로 따로 표시한다. [E10]

### 10.3 증강과 label 원칙

LLM이 생성한 gene response, 모델이 예측한 expression, 토큰 삭제 결과를 실측 ground truth에 섞지 않는다. pseudo-label/생성 데이터 연구는 필요 시 별도 augmentation 실험으로 열고 teacher, train 범위, uncertainty, 원 실측 test에서의 이득을 명시한다. 첫 개선 라운드의 필수 항목은 아니다.

---

## 11. Phase M — Model-centric 개선과 LoRA

### 11.1 모델 선택과 공통 대조

A/S 개발 평가에서 성능·donor 변동·자원·입출력 적합성을 함께 보고 **미세조정할 foundation backbone 1개, 필요 시 2개**를 선정한다. 단순히 기존 test F1가 가장 큰 모델을 고르지 않는다. LiverCT는 native reference 방법의 비교군으로 유지하되 LoRA 적용 대상으로 강제하지 않는다.

| ID | 모델 변경 | 비교 목적 |
|---|---|---|
| M00 | 동결 backbone + 기존 linear head | 현재 기준점 |
| M01 | 동결 backbone + 작은 동일 학습 head | head 용량/최적화만으로 해결되는지 확인 |
| M02 | 마지막 1–2개 block 또는 공식 지원 부분 미세조정 | pretrained 표현의 task 적응 효과 |
| M03 | LoRA + 동일 head | 적은 학습 parameter로 비슷한 이득을 얻는지 확인 |
| M04 | Full fine-tuning | 자원이 허용될 때 부분/LoRA의 비교 상한 후보. 무조건 우월하다고 가정하지 않음 |
| 선택 | 길이·pooling·loss 조정 | 오류 분석에서 근거가 있는 한 요인씩 별도 비교 |

M02/M03/M04는 **같은 head 구조, train label, 데이터 버전, 선택 기준**으로 비교한다. M00의 sklearn logistic head와 학습형 neural head의 차이도 영향을 주므로, backbone 변화의 공정한 대조는 M01을 사용한다.

“모델 파라미터 수정”은 우선 학습 대상·정규화·learning rate·dropout·sequence·pooling 조정이다. pretrained hidden size/층 수를 임의로 바꾸고 대다수 가중치가 로드되지 않은 상태를 같은 pretrained 모델의 튜닝이라고 하지 않는다. 구조 변경은 후속 독립 가설과 baseline이 있을 때만 수행한다.

### 11.2 LoRA 구현 요구사항

**[외부 조사]** 공식 PEFT 문서는 custom model에서 실제 module 이름을 확인해 adaptation 대상을 지정하고 학습 parameter를 확인하도록 안내한다. 범용 챗봇 예제의 `q_proj/v_proj`를 모든 세포 모델에 그대로 적용하지 않는다. [E11]

- Geneformer는 사용 checkpoint의 BERT 계층과 `query`/`value` 등 실제 Linear module을 조회하여 target을 정한다.
- scGPT는 attention backend와 fused projection 여부에 따라 target이 달라진다. fused QKV가 별도 Linear module이 아닌 경우, 지원 경로를 확인하거나 명시적인 adapter/부분 미세조정으로 전환한다. 연결되지 않는 PEFT 설정으로 “LoRA 완료”라고 하지 않는다.
- 기본 탐색 예시는 `rank ∈ {8,16}`, `alpha=2×rank`, dropout 0 또는 0.05 중 제한된 조합이다. 이 값은 시작 제안이지 생물학 모델의 최적값이 아니다.
- r·alpha·target module·trainable parameter 수·base frozen 여부·head parameter 수·learning rate를 run별 저장한다.
- adapter 초기화 전후 eval 출력 일치, 한 minibatch의 nonzero gradient/parameter update, save/reload 결과 일치를 작게 확인한다.
- classifier head가 저장 대상에서 빠지지 않게 한다. adapter만 저장했다면 정확한 base revision·tokenizer·vocab·preprocessing·label mapping을 manifest로 연결한다.
- QLoRA는 필요한 경우만 별도 quantization 요인으로 수행한다. 양자화까지 동시에 바꾼 결과를 순수 LoRA 효과로 해석하지 않는다.

### 11.3 최초 학습 예산

**[실험 제안]** 초기 탐색은 backbone당 한 개발 fold·한 seed로 후보를 줄이고, 사전에 정한 기준으로 선택한 후보를 모든 개발 fold와 3개 seed에서 확인한다. 처음부터 모델×데이터×rank×epoch의 전 조합을 실행하지 않는다.

| 항목 | 시작안 |
|---|---|
| GPU | 할당된 A30 24 GiB 1장부터. 독립 실험 병렬은 할당 확인 후 |
| microbatch | 1부터 메모리 측정 후 확대; gradient accumulation으로 effective batch 명시 |
| 학습 epoch | 초기 상한 10, validation early stopping patience 2. 원 공식 재현은 원 설정 별도 |
| lr 후보 | head/LoRA와 부분/full FT에 서로 적합한 소수 후보를 사전 지정; 예: LoRA 1e-4/3e-4, 부분 FT 1e-5/3e-5 |
| precision | 실제 지원 여부를 확인한 BF16/FP16; NaN·loss scaling 기록 |
| 최종 반복 | 선택된 설정 seed 3개. 동일 seed의 대응 실험을 연결 |
| OOM 대응 | microbatch 축소 → 지원 checkpointing/동결 → 입력 길이 변경은 별도 조건으로 표시 |

자원 제약으로 full FT가 안 되면 그 사실을 남긴다. 기존 동결 추론의 0.72/2.50 GiB를 근거로 학습도 가능하다고 가정하지 않는다. 실패한 full FT를 성공한 LoRA와 같은 범주의 완료 결과로 집계하지 않는다.

### 11.4 간 도메인 추가 적응은 별도 확장

간/HCC train reference로 continual pretraining을 수행할 수는 있으나 첫 개선의 기본 전제는 아니다. label task fine-tuning 및 data-centric 가설을 확인한 후 수행한다. 필요하다면 **같은 양의 일반 세포 추가 학습 대조**를 두어 간 특화 효과와 단순 추가 학습량을 분리한다. 외부 final cohort를 비라벨이라고 사전학습에 넣는 경우 strict inductive 결과와 별도 regime으로 표시한다.

---

## 12. Phase DM — 데이터 개선과 모델 개선의 기여 분리

선택한 동일 backbone에서 다음 2×2 비교를 수행한다. baseline·특화 모델 비교와 개발 분석이 끝나기 전부터 여러 변경을 합치지 않는다.

| 실험 | 데이터 | 모델 | 답하는 질문 |
|---|---|---|---|
| DM00 | D0: 원 기준 데이터 | M0: 고정 대조 모델 | 출발 성능 |
| DM10 | D1: 선택한 데이터 개선 | M0 | 데이터만 바꿔도 개선되는가? |
| DM01 | D0 | M1: 선택한 모델 개선 | 모델만 바꾸면 개선되는가? |
| DM11 | D1 | M1 | 두 개선이 보완적인가? |

분류처럼 높을수록 좋은 score에서는 다음을 보고한다.

```text
Data gain        = score(D1,M0) − score(D0,M0)
Model gain       = score(D0,M1) − score(D0,M0)
Combined gain    = score(D1,M1) − score(D0,M0)
Interaction      = score(D1,M1) − score(D1,M0) − score(D0,M1) + score(D0,M0)
```

오차처럼 낮을수록 좋은 지표는 개선 방향을 반대로 정의하거나 `score=-error`로 맞춘다. 동일 split·동일 seed·동일 평가 셀/조건을 사용한다. 추가 학습량 또는 데이터 규모가 다르면 표에 명시한다.

최종 report는 단일 best run이 아니라 평균·변동·환자별/perturbation별 쌍대 차이를 보여준다. 신뢰구간은 독립 단위인 donor 또는 intervention/replicate group을 고려한다. cell bootstrap만으로 매우 좁은 biological CI를 만들지 않는다. fold가 겹친 학습 데이터를 쓴다는 점과 donor 수가 적다는 제한도 적는다.

개선 판단 기준은 외부 점수를 본 뒤 바꾸지 않는다. 개발 지표의 변화, 대부분 donor/조건에서의 방향, rare class/효과 회복의 악화 여부, 계산비용을 함께 보고한다. 점수가 오르지 않으면 “이번 조건에서 추가 이득 없음”으로 남긴다. 성능 향상이 없었다고 전체 연구를 미완료로 숨기지 않는다.

---

## 13. 실행 순서, 병렬 작업, 단계별 종료 조건

| 순서 | 작업 | 필수 종료 산출물 | 병렬 가능 항목 |
|---|---|---|---|
| P0 | 현재 HEAD·자산 인수, protocol·외부 holdout 지정 | 변경 대상 목록, 고정 split/평가 계약 | 신규 데이터·특화 후보 접근성 확인 |
| P1 | 공식 scGPT 재현 | 원 설정/변경/실측 score 표 | GEARS 공식 환경·data smoke |
| P2 | A00–A04 기본 분류 비교 | 공통 fold 결과와 자원, 실패 범위 | B 공식 benchmark |
| P3 | LiverCT + 암 도메인 후보 | regime별 특화 비교 결과 | HepG2 데이터 처리 |
| P4 | 오류·데이터 분석 | 근거 있는 가설 우선순위 및 반증 조건 | B00–B03 간 유래 실측 평가 |
| P5 | data-centric 실험 | 데이터만 변경한 결과 최소 2개 | 같은 D0 위 model-centric 탐색 |
| P6 | 부분 FT/LoRA 등 model-centric | 동결/동일 head/부분/LoRA 비교 | 가능한 foundation-response 확장 |
| P7 | DM 결합·외부 final | 고정 후보 최종 평가와 한계 | chemical 후속은 별도 scope |
| 상시 | 튜토리얼·Git 갱신 | 해당 결과를 재실행하는 사용자 경로 | 연구 코드 완료 시에만 파생 갱신 |

한 모델의 설치가 막히면 독립 경로는 계속 진행한다. 같은 문제를 설정 변화 없이 반복하지 않는다. scFoundation/LiverCT가 실제로 차단되면 오류·원본 version·시도한 대안·대체 결과를 기록하고, 모든 후보를 돌렸다고 쓰지 않는다. 데이터가 없어서 미실행인 것을 구현 완료로 처리하지 않는다.

### 첫 작업 세션의 구체적 행동

1. `README.md`, `AGENTS.md`, `reports/final_summary.md`, `reports/data_card.md`, `configs/paths.yaml`, `scripts/common.py`를 읽고 현재 HEAD를 기록한다.
2. 기존 데이터·checkpoint 경로와 GPU 할당을 확인한다. 이미 받은 71,915세포 count를 다시 만들거나 모든 기존 notebook을 무조건 다시 돌리지 않는다.
3. 새 research branch와 `reports/research_phase2/`를 만든다. old run ID를 재사용하지 않는다.
4. 외부 cohort·donor fold·label mapping·metric 규칙을 고정하고 A00/A01의 한 개발 fold를 실행한다.
5. 공식 scGPT 재현과 scFoundation/LiverCT 의존성 확인을 시작한다. GEARS는 별도 환경에서 병렬 준비한다.
6. 첫 결과가 나오면 코드·config·결과를 묶어 commit한다. 보고서 양식이 완벽해질 때까지 실제 실행을 미루지 않는다.

---

## 14. 구현 구조와 기록 규칙

### 14.1 최소 확장 원칙

기존 `scripts/00`–`11`, `envs/activate.sh`, config, feature cache를 활용한다. 범용 플랫폼·웹 UI·복잡한 workflow engine을 새로 만드는 것은 목표가 아니다. 재사용이 필요한 처리만 작은 모듈로 분리한다.

아래 이름은 **새로 구현할 인터페이스 예시**이며 현재 파일이 존재한다는 뜻이 아니다.

```text
configs/research/
  protocol_annotation.yaml
  protocol_perturbation.yaml
  external_holdout.yaml
  models/*.yaml
  data_variants/*.yaml
  finetuning/*.yaml

scripts/research/
  reproduce_official.py
  run_annotation_cv.py
  run_liverct_mapping.py
  run_measured_perturbation.py
  diagnose_errors.py
  finetune_model.py
  collect_research_results.py

reports/research_phase2/
  status.md
  official_reproduction.md
  model_registry.tsv
  dataset_registry.tsv
  leaderboard_annotation_inductive.tsv
  leaderboard_annotation_query_adaptive.tsv
  leaderboard_genetic_perturbation.tsv
  hypothesis_registry.tsv
  improvement_comparison.tsv
  external_final.md
```

사용자용 설명과 튜토리얼은 한국어로 작성한다. 코드·논문의 고유 용어는 필요한 범위에서 원문을 병기한다.

### 14.2 run에 필요한 필드

```yaml
run_id: unique_immutable_id
status: planned | running | executed | failed | blocked | not_applicable
task: annotation | genetic_response | chemical_response | embedding_shift
regime: inductive | query_adaptive | official_reproduction | exploratory
model_id: explicit_checkpoint_and_method
base_checkpoint_revision: pinned
upstream_commit: pinned
project_commit: pinned
dataset_version: pinned
split_id: pinned
data_variant: D0_or_named_variant
model_variant: M0_or_named_variant
seed: recorded
training_donors: manifest_reference
validation_donors: manifest_reference
evaluation_donors_or_perturbations: manifest_reference
label_mapping: versioned
input_contract: versioned
pretraining_overlap: known_overlap | not_found_in_checked_manifest | unknown
allowed_target_information: explicit
trainable_parameters: measured
gpu_name: measured
peak_cuda_allocated_gib: measured
peak_cuda_reserved_gib: measured
peak_process_gpu_gib: measured
wall_preprocessing_seconds: measured
wall_training_seconds: measured
wall_inference_seconds: measured
wall_total_seconds: measured
metric_protocol: versioned
prediction_path: actual_path
failure_or_limitations: explicit
```

위 내용은 스키마이지 실측값이 아니다. 실제 실행하지 않은 항목은 `executed`로 만들지 않는다. 모델 간 서로 다른 feature 차원, test 제외 수, native reference 학습 label, query adaptation 여부도 비교표에 포함한다.

큰 count·가중치·embedding·예측 원본은 서버 자산 영역에 두고 경로·크기·SHA256·생성 명령을 manifest에 기록한다. 가능한 작은 지표·분할·예측 요약·failure log는 Git에 반영한다. 공개 저장소에는 비밀값·접근 제한 데이터·재배포 권한 없는 checkpoint·미출판 내부 문서 원본을 올리지 않는다.

### 14.3 필요한 검증만 수행

필수는 cell/feature 순서, gene ID, donor/perturbation split, train-only fit, label 누수, 실제 adapter update, checkpoint reload, metric 정합성이다. 새로운 보안 감사, 110개보다 많은 테스트 수 달성, 반복적인 전면 코드 리뷰를 연구 진행의 선행조건으로 삼지 않는다. 이미 통과한 부분은 변경 영향이 있을 때만 재확인한다.

---

## 15. 튜토리얼은 연구 결과를 실행하는 인터페이스로 유지한다

기존 6개 노트북을 제거하지 않는다. 연구용 함수·CLI·config를 그대로 호출하도록 갱신하고 학습 로직을 notebook에 중복 구현하지 않는다.

| 사용자 동작 | 튜토리얼에서 제공할 것 |
|---|---|
| 공개 모델을 비교해 본다 | 저장된 실제 비교표, 모델별 작은 inference, 학습 데이터/정보 조건 설명 |
| 새 HCC 데이터를 넣는다 | raw-count/metadata schema, gene 정합, label 없는 inference 경로 |
| 일반 모델과 간/암 특화를 비교한다 | LiverCT와 cancer checkpoint 실행, adaptation 여부 표시 |
| 실제 섭동 예측을 본다 | 작은 real intervention subset, control/treated 정답과 예측 비교 |
| 개선 전후를 확인한다 | D0M0/D1M0/D0M1/D1M1 동일 샘플 비교와 configuration |
| 결과를 해석한다 | 성공·실패 사례, coverage/uncertainty, 표현 변화와 실측 효과의 차이 |

각 경로는 **기존 결과 열람 모드**와 **작은 재실행 모드**를 구분한다. 큰 데이터 전체 다운로드·10개 fold 재학습을 노트북 기본 동작으로 만들지 않는다. 노트북의 출력에는 실제 `run_id`를 표시하고, 재실행하지 않은 저장 결과를 방금 실행한 결과처럼 쓰지 않는다.

연구 단계가 완료될 때 그 기능의 작은 notebook 또는 기존 notebook section만 갱신하고 새 kernel에서 확인한다. 튜토리얼 미관·문서량 때문에 모델 비교와 학습을 뒤로 미루지 않는다.

---

## 16. 완료 기준과 최종 보고

이번 목표의 완료 여부는 다음 표로 판정한다. 성능이 반드시 높아야 한다는 조건은 두지 않지만, 실행하지 않고 검토만 한 항목을 완료로 표시하지 않는다.

| 요구 | 완료 근거 |
|---|---|
| 리더의 공식 재현 요구 | 최소 1개 공식 tutorial/benchmark의 실제 데이터·설정·score·변경점 |
| 기본 분류 비교 | PCA/scGPT/Geneformer/scFoundation의 공통 protocol 결과 또는 구체적 실행 차단 근거. scVI 대조 포함 |
| 간/암 특화 비교 | LiverCT 및 암 특화 checkpoint 1개의 task·정보 조건을 맞춘 평가 |
| HCC 일반화 | 환자 분리 개발 평가와 사전 지정 외부 HCC cohort의 최종 결과 |
| 실제 perturbation | 실측 intervention response에서 GEARS와 단순 기준선 비교. 간 유래와 HCC 범위 구분 |
| Data-centric 개선 | 개발 관측에 연결된 가설, 데이터만 변경한 대조 실험 |
| Model-centric 개선 | 동결 대조, 같은 head, 부분 FT와 실제 LoRA 또는 지원 제약에 따른 명시적 대체 |
| 결합 효과 | 동일 backbone의 2×2 data/model 요인 분리 |
| 지속 사용 | 실제 실행 코드 기반 튜토리얼, checkpoint/data/config로 재현 가능한 경로 |

최종 보고서는 다음 질문에 답해야 한다.

- 어떤 모델이 어떤 task·평가 조건에서 실용적인가? 비용까지 고려한 다음 단계 후보 1–2개는 무엇인가?
- PCA 대비 이득이 있는가? 없으면 어떤 subgroup/조건에서 한계가 관찰되는가?
- 간/암 특화 모델의 이득은 외부 데이터나 query adaptation 사용 차이로 설명될 수 있는가?
- 데이터 개선과 모델 개선 중 무엇이 기여했으며 둘을 합치면 어떤가?
- 실제 생물학적 intervention 성능을 어디까지 검증했고, 무엇은 표현 탐색에 그쳤는가?
- 다음 실험 하나를 고른다면 어떤 근거로 무엇을 하는가?

**Git 운영:** 현재 변경을 보존하고 작업 branch에서 단계별 commit/push한다. force push 금지. 데이터 수집, 기본 비교, 특화 비교, 가설/개선, 최종 평가, 튜토리얼 업데이트를 단계별로 확인 가능하게 남긴다. 원격 반영 성공을 실제 확인하고 완료 보고에 commit을 적는다. 이 기획서 전달 자체와 후속 서버 실험 실행을 구분한다.

---

## 17. 근거·출처와 확인 범위

### 첨부 문헌

**[R1]** Choi Y, Koh IG, Choi M, An JY, VITAL Consortium. *Toward AI Virtual Cells for Hepatology: Representation, Generation, Dynamics, and Intervention in Single-Cell Models*. DOI: https://doi.org/10.3350/cmh.2026.0820  
제공 파일: `cmh-2026-0820 (1).pdf`, 44쪽. 주요 근거: pp.3–6(정의와 평가 범위), pp.15–20(전이·prior·간 적용 근거), pp.21–25(실측 개입·분할·평가), pp.38·41·42·44(Figures 1·4·5·6). 별도 supplement 원본은 미제공.

### 현재 저장소 — 커밋 고정 근거

**[G1]** README: https://github.com/honbul/vcell_liver/blob/243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79/README.md  
**[G2]** 실제 실행 결과 보고: https://github.com/honbul/vcell_liver/blob/243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79/reports/final_summary.md  
**[G3]** 데이터 카드: https://github.com/honbul/vcell_liver/blob/243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79/reports/data_card.md  
**[G4]** 평가·자원 공통 코드: https://github.com/honbul/vcell_liver/blob/243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79/scripts/common.py  
**[G5]** scGPT 실행: https://github.com/honbul/vcell_liver/blob/243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79/scripts/03_scgpt_annotation.py  
**[G6]** Geneformer token/embedding: https://github.com/honbul/vcell_liver/blob/243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79/scripts/geneformer_utils.py  
**[G7]** 서버 관측: https://github.com/honbul/vcell_liver/blob/243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79/reports/environment.json

저장소에 기록된 결과와 관련 코드를 검토한 것이며, 본 기획서 작성자가 서버에서 학습·추론을 재실행해 독립 재현한 것은 아니다.

### 공식 모델·추가 연구

**[E1] scGPT**  
논문: https://doi.org/10.1038/s41592-024-02201-0  
공식 코드/whole-human 및 pan-cancer model zoo: https://github.com/bowang-lab/scGPT  
공식 annotation tutorial: https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html

**[E2] Geneformer**  
공식 모델 카드·버전·가중치: https://huggingface.co/ctheodoris/Geneformer  
문서: https://geneformer.readthedocs.io/en/latest/  
프로토콜: https://doi.org/10.1038/s41596-026-01364-8

**[E3] scGPT의 암 특화 체크포인트**  
공식 README Model Zoo의 pan-cancer 항목: https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo  
학습 corpus가 간암 전용이라는 근거가 아니므로 pan-cancer로 표기한다.

**[E4] GEARS**  
논문: https://doi.org/10.1038/s41587-023-01905-6  
공식 코드·지원 범위·튜토리얼: https://github.com/snap-stanford/GEARS

**[E5] scFoundation / scVI**  
scFoundation 논문: https://doi.org/10.1038/s41592-024-02305-7  
공식 입력·사용법: https://github.com/biomap-research/scFoundation/tree/main/model  
모델 사용 조건: https://github.com/biomap-research/scFoundation/blob/main/MODEL_LICENSE  
scVI는 첨부 리뷰 Section 1.1에서 정리한 count-aware baseline이며 실행 시 공식 scvi-tools 문서를 확인한다: https://docs.scvi-tools.org/

**[E6] uniLIVER / LiverCT — 이번 외부 조사로 추가한 간 특화 비교군**  
논문: https://doi.org/10.1016/j.jgg.2025.01.017  
출판사 공개 설명: https://www.jgenetgenomics.cn/en/article/doi/10.1016/j.jgg.2025.01.017  
공식 README/모델/demo: https://github.com/fyh18/LiverCT  
query adaptation 실제 코드: https://github.com/fyh18/LiverCT/blob/main/liverct_annot.py  
공식 README가 제공하는 pretrained models: https://zenodo.org/record/8239570/files/models.zip  
Lu2022 demo: https://zenodo.org/record/13380493/files/Lu2022_subsampled.h5ad  
문헌·공식 코드에서 자산 경로를 확인했으며, 이 문서 작성 중 가중치를 설치·실행한 것은 아니다. 실행 시 revision·접근성·사용 조건을 다시 고정한다.

**[E7] scVIDR**  
원 연구: https://doi.org/10.1016/j.patter.2023.100817  
공식 코드: https://github.com/BhattacharyaLab/scVIDR  
공개 protocol: https://pmc.ncbi.nlm.nih.gov/articles/PMC12274909/

**[E8] PRnet**  
원 연구: https://doi.org/10.1038/s41467-024-53457-1  
공식 코드: https://github.com/Perturbation-Response-Prediction/PRnet

**[E9] DeepLiver**  
원 연구: https://doi.org/10.1038/s41556-023-01316-4  
연구 코드: https://github.com/aertslab/Bravo_et_al_Liver

**[E10] LiverDCP — 선택적 prior 후보, preprint**  
연구 프로젝트: https://zhaolabutmb.github.io/papers/LiverDCP/  
코드: https://github.com/zhaolabutmb/LiverDCP  
preprint DOI: https://doi.org/10.64898/2026.08.07.743628  
기존 분류/섭동 baseline을 대체하는 turnkey 모델로 권고한 것이 아니다.

**[E11] LoRA의 custom-model 적용**  
공식 PEFT 문서: https://huggingface.co/docs/peft/en/developer_guides/custom_models

**[E12] HCC sorafenib resistance 사례 — 접근성과 endpoint 확인 대상**  
Zhou et al. *Single cell RNA-seq analysis identifies a noncoding RNA mediating resistance to sorafenib treatment in HCC*. https://doi.org/10.1186/s12943-021-01473-w  
장기 resistant state 비교와 급성 약물 response 예측을 구분한다. 공개 데이터 접근성을 확인하기 전 필수 benchmark로 지정하지 않는다.

**[E13] 단순 perturbation 기준선의 필요성**  
Ahlmann-Eltze et al. *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines*. https://doi.org/10.1038/s41592-025-02772-6  
해당 연구가 평가한 설정의 결과이며 모든 최신 모델·간 데이터의 순위를 확정하는 근거는 아니다. Systema와 broader holdout 논점은 첨부 리뷰 [R1]의 References 25–26 및 Section 4.2도 참조한다.

### 공개 데이터와 생물학적 범위

**[D1]** GSE149614: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149614  
원 연구: https://doi.org/10.1038/s41467-022-32283-3

**[D2]** GSE151530: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151530  
NCI scAtlasLC: https://scatlaslc.ccr.cancer.gov/

**[D3]** GSE189903: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189903

**[D4]** GSE264667: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667  
관련 원 연구: https://doi.org/10.1038/s41588-025-02169-3

**[D5]** HepG2 기원/표준 식별: https://www.cellosaurus.org/CVCL_0027

**최종 범위:** 위 출처의 지원 범위를 넘는 모델 구조·하이퍼파라미터·실험 순서·성과 기준은 본 요청서의 연구 제안이다. 모델 성능 개선, 간암 치료 효과 또는 전향적 liver simulator의 완성을 미리 보장하지 않는다.
