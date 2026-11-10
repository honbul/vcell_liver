# Virtual Cell 후속 피드백 5개 항목별 조사·실험 수행계획

**작성 기준:** 2026-11-10 · `honbul/vcell_liver`의 재현 중심 연구 결과(확인된 연구 브랜치 `research/reproduction-first-20261105`, 커밋 `540ec932ae7876b7af316bdbe9fc1998d87c4440`)와 공개 선행연구.  
**문서 성격:** 후속 작업 제안. 아래 신규 실험은 수행 결과가 아니라 **계획**이다. 기존 실험의 수치와 새로 제안하는 방법을 구분한다.

## 총괄

| 피드백 | 목표 | 우선 작업 | 필수 산출물 | 단계 |
|---|---|---|---|---|
| 1. 기존 결과 분석 | Full FT 우위·샘플링 실패의 이유 검토 | 기완료 모델의 로그·개입별 오차 진단 → 최소 ablation | 가설–관측–대안설명–검증 결과표 | P0/P1 |
| 2. 최신 연구 검토 | 방법론을 넓게 파악 | 최신 모델·벤치마크·코드·자료 비교 | 모델/데이터/과제/장단점 매트릭스 | P0 |
| 3. 미관측 세포 일반화 | 새 유전자 및 새 세포 맥락 전이 | CRISPRi 4개 세포주 공통 패널·LOCO 실험 | 2×2 OOD 성능표 및 subgroup/오차 분석 | P1 |
| 4. 간암 적용 | 실제 HCC 관련 perturbation 검증 | HepG2 유전적 자료·SNU-423 약물 자료 적격성 확인 | 접근성/대조군/개입/측정/분할 가능성 데이터 카드 | P0/P1 |
| 5. 연구 방향 | 가장 적합한 1개 실험 경로 선정 | 2~3개 후보의 실행가능성·신규성·위험 평가 | 비교 기획서 + 선정 근거 + 1차 실행 계획 | P0/P2 |

## 1. 현재 실험 결과에 대한 분석

### 1-1. 무엇이 이미 관측됐는가

- **모델 변경:** 동일 Norman 학습 split, 학습률 1×10⁻⁴, 최대 15 epochs, 최초 screening seed 0의 validation ΔRMSE는 Full FT(M0) **0.03429**, Decoder/perturbation-only(M1) **0.05760**, 마지막 2 Transformer block(M2) **0.04003**, QKV-LoRA rank 8(M3) **0.04384**, random initialization + Full FT **0.03685**. 낮을수록 좋다. 따라서 현재 확인되는 것은 **특정 하이퍼파라미터 조건에서의 Full FT 우위**이며 원인이 규명된 결과는 아니다.
- **데이터 변경:** 고정 대조군(D0) **0.03429**, 매 epoch 대조군 재샘플링(D1) **0.03494**, 개입별 균등 노출(D2) **0.03576**(validation). D1과 D0의 3-seed held-out 평균은 각각 **0.04209 / 0.04149**, paired gain(D0−D1) **−0.00060**, 95% 구간 **[−0.00185, +0.00063]**. D2는 반복 홀드아웃에 진출하지 않았다.
- **해석 제약:** M1–M3는 동일 학습률 한 가지를 사용했고 모드별 최적 학습률/LoRA rank/학습 층 수가 탐색되지 않았다. 따라서 “LoRA가 본질적으로 열등하다”는 결론은 불가능하다. D1/D2 효과의 원인도 분리되지 않았다.

### 1-2. 먼저 GPU를 쓰지 않고 가능한 분석

| 분석 ID | 질문·가설 | 기존 데이터로 확인할 것 | 판정 기준 |
|---|---|---|---|
| A1 | Full FT는 더 많은 표현을 조정해 강한 perturbation에 유리한가? | 각 모드의 학습·검증 곡선, 최적 epoch, 조기 종료 여부, 초기/최종 loss, 개입 크기별 RMSE, DE recovery, 표적 종류·단일/이중 개입별 오류 | 이득이 특정 subgroup에 집중되는지; 일반적 개선인지 |
| A2 | PEFT는 단순히 최적화가 부족했는가? | gradient norm, 실제 adapter/decoder 업데이트, 모델별 gradient clipping, validation plateau, 파라미터 변화량, 학습·추론 비용 | 학습이 진행되었지만 capacity 문제인지, optimizer/스케줄 문제인지 구분할 단서 |
| A3 | D1은 대조군 변경이 유익한 다양성보다 잡음을 늘렸는가? | 같은 처리 세포에 연결된 control 간 발현 거리·분산, 변경률, 개입별 대조군 반응 크기, 조건별 오차 변화 | 변경된 control과 residual 오차의 관련성 |
| A4 | D2는 소수 개입 보정 대신 반복 노출로 과적합했는가? | 조건별 고유 세포 수와 반복 횟수, 큰/작은 개입의 validation 오차, 단일·이중 개입 비중, DE 신호 크기 | 개선/악화가 세포 수나 반복 정도와 연결되는지 |
| A5 | 전체 ΔRMSE가 ‘약한 개입’을 과대평가하는가? | 강한/약한 반응 분리, 효과 크기별 ΔPearson, top-DE recall, 퍼튜베이션 구별(retrieval), 예측 효과 축소 편향 | 단순 control-like prediction을 높은 성능으로 오해하지 않는지 |

**중요:** 위는 설명 가능한 *가설*이지 데이터로 확인된 인과 설명이 아니다. 기존 테스트를 다시 조정 목표로 사용하지 말고, 진단 결과에서 나온 설정은 새 검증 단계로 분리한다.

### 1-3. 실제 추가 실험(작고 통제된 ablation)

| 실험 ID | 변경 요소만 고정 비교 | 왜 필요한가? | 예상 자원·중단 기준 |
|---|---|---|---|
| A6 | M0/M2/M3에 대해 **모드별 학습률** 2~3값, 동일 업데이트 예산·동일 검증 분할 | 동일 LR이 PEFT에 불리했는지 확인 | 먼저 1 fold/1 seed; 유의미한 개선 시 3 seed |
| A7 | LoRA rank **8 vs 16(필요 시 32)**, QKV만 vs QKV+FFN; partial 마지막 **2 vs 4** blocks | 구조 표현력/학습 가능 위치의 문제 분리 | 한 번에 하나씩, 무제한 조합 탐색 금지 |
| A8 | 고정 control vs epoch 재샘플링 vs **control 평균/상태·배치 매칭** | 짝짓기 잡음 vs 다양성 효과 분리 | 배치 정보 없으면 그 한계를 명시, 임의 ‘실제 paired cell’ 주장 금지 |
| A9 | 자연 분포 vs **중복 없이 균형 다운샘플링** vs 실제 D2 oversampling | 균형화 효과와 중복 노출 효과 분리 | 동일 update 수 및 희귀 개입별 오차 함께 보고 |

**산출물:** `기존 관측 → 원인 가설 → 진단 근거 → 대안 설명 → 결정적 ablation → 결론/미결정` 형식의 1~2쪽 표. 원인 단정은 ablation으로 분리된 변수에 한해서만 한다.

## 2. 다양한 모델 및 최신 연구 검토

조사는 단순 모델명 수집이 아니라 **무슨 일반화 문제를 해결하는지**, **입력 정보 접근 범위**, **정답·평가지표**, **공식 코드·가중치·자원**, **제약**을 동일한 양식으로 비교한다.

| 유형 | 대표 연구 | 해결하려는 문제 | 조사할 핵심 비교·주의점 |
|---|---|---|---|
| 단순 기준선 | no-change, 평균 반응, GO/Ridge, 최근 선형 baseline 연구(2025) | 복잡한 모델이 최소한 넘어야 하는 수준 | 데이터 접근/평가 정의·반응 효과 크기 일치 |
| 유전자 그래프 | **GEARS**, **TxPert** (2026) | 미관측 유전자·조합·다른 세포주로의 전이 | GO/STRING/Reactome 그래프 활용, batch 적절성, multi-KG 기여도 |
| 사전학습 Transformer | **scGPT**, Geneformer, scFoundation, PertAdapt | 제한된 라벨과 새로운 개입에 전이되는 표현 | 임베딩만인지 실제 response decoder인지, frozen/FT 범위 |
| 세포 집단 상태 전이 | **State** (Cell, 2026), Stack (2026) | 개별 세포의 노이즈를 넘어 집단 분포 변화 및 세포 맥락 전이 | 집단(set) 입력·대조군 정보, 사전학습 비용, pretrained 이용 가능성 |
| 지식기반 zero-shot | **PIE** (2026 preprint) | 새 세포주·표적·데이터셋에서의 DE/변화량 예측 | NCBI/GO/Cellosaurus/DepMap 등 prior, 소스 사용량 ablation, 제공된 checkpoint |
| 화학적 반응 | CPA/chemCPA, State, PIE 등 | 새로운 약물·용량·세포 상태 반응 | 유전자 CRISPRa/i와 약물을 같은 입력 과제로 혼합하지 않기 |

**중요한 2025–2026 벤치마크:**
- 2025년 *Nature Methods* 논문은 복잡한 perturbation predictor가 선형 기준선을 일관되게 능가하지 못함을 보고한다.
- 2026년 *Nature Methods* 대규모 일반화 벤치마크는 방법 27개·데이터 29개를 비교하며 OOD context를 강조한다.
- 2026년 *Science Advances* 비교는 25개 데이터셋에 걸쳐 미관측 개입, 조합, 세포 간 전이 과제를 평가하고, 세포 간 전이에서 일관되게 우수한 방법이 없음을 보고한다.
- **Systema**는 처리군·대조군 간의 시스템적 변동만 맞히는 방법을 생물학적 개입 효과를 맞힌 방법으로 오인할 위험을 지적한다.

**산출물:** (i) 모델 비교표 8~12개, (ii) 사용 가능한 checkpoint/코드 상태, (iii) 모델별 재현 데이터·훈련 비용, (iv) 각 모델이 검증한 OOD 종류, (v) 실제 실행 후보 2~3개 + 선정/제외 이유. 전부 실행하지 않는다.

## 3. 미관측 유전자와 미관측 세포 일반화

### 용어·검증 문제 분리

| 조건 | 표적 perturbation | 세포 맥락 | 질문 |
|---|---|---|---|
| In-domain | 학습에서 본 표적 | 학습 세포주 | 기준 구현 검증 |
| Unseen perturbation | **미관측 표적** | 학습 세포주 | 새로운 유전자를 조작했을 때 반응 예측 |
| Unseen cell context | **학습에서 관측한 표적** | **학습에 없는 세포주** | 같은 조작을 처음 보는 세포에서 예측 |
| Double OOD | **미관측 표적** | **학습에 없는 세포주** | 표적과 세포 맥락 모두 처음 보는 조합 예측 |

여기서 '학습하지 않은 세포'는 단순히 다른 세포 barcode가 아니라 **perturbation 처리 데이터를 학습하지 않은 세포주/세포 유형/환자**를 의미하도록 사전 정의한다. 이번 제안의 첫 검증 단위는 **세포주**다.

### 데이터 및 실험 설계

- **1순위 공개 유전적 개입 패널:** Replogle K562·RPE1과 Nadig HepG2·Jurkat의 **CRISPRi** Perturb-seq. 공식 GEO `GSE264667`, PIE의 4-line 재가공 자료, scPertEval의 공개 정리 자료가 있다. 공통 표적/유전자/guide/시간·처리량/대조군 metadata를 먼저 확인한다. **Norman K562 CRISPRa와 무통제 병합하지 않는다.**
- **데이터 계약:** cell-line ID, perturbation target, guide, control, batch/sample, day, library-size normalization, gene panel, cell count·effect strength, 분할 단위·독립 반복 확인.
- **LOCO:** K562·RPE1·Jurkat에서 학습, HepG2의 **개입 처리 세포는 전부 미학습**, HepG2의 *비개입 대조군만* 허용하여 예측. 이를 4개의 세포주에 대해 돌아가며 수행. 다른 세포주에서 같은 표적을 본 경우/못 본 경우를 다시 나눈다.
- **모델 후보:** 최소 no-change/훈련 평균/nearest-cell-line/GO-Ridge, 여력이 있으면 TxPert·PIE·State 중 1~2개. GEARS는 원 구조상 멀티세포주 학습 및 unseen context 전이의 직접적인 지원 범위를 검증해야 하며 자동 비교군으로 강제하지 않는다.
- **지표:** 조건별 ΔRMSE, ΔPearson, 발현 변화 방향, 통계적으로 정의한 DEG 복원 AUPRC/F1, perturbation discrimination/retrieval, 개입 효과 크기별 오류, 가능하면 distributional distance. Control-like 공통 효과를 맞힌 것과 특정 perturbation 신호를 구분한다.
- **검증 원칙:** 선택은 source 세포주들의 validation에서만, target 세포주의 처리 후 정답은 최종 평가에서만. ‘새 세포주에서 control을 관측할 수 있음’은 별도 regime로 표시. 같은 public dataset에 반복 접근하면 최초 blind test라고 표기하지 않는다.

**산출물:** OOD 4조건 및 4개 LOCO fold 성능표, target/context coverage, 세포주별 실패 패턴, 모델·기준선과의 짝지은 효과·불확실성. 자료 확보 전에는 대규모 학습 착수 금지.

## 4. 간암 관련 자료 확대 조사와 적용

| 자료 | 실제로 제공하는 정답 | 연구에서의 역할 | 1차 판정·주의점 |
|---|---|---|---|
| **GSE149614** HCC 환자 조직 | 비개입 관측 scRNA-seq, 기존 세포 라벨 | 세포 상태·악성 관련 표현 분석 | **개입 후 RNA 정답 없음**; perturbation 예측의 직접 평가자료 불가 |
| **GSE264667** HepG2/Jurkat | CRISPRi 이후 단일세포 RNA + 대조군 | 유전적 섭동·cell-line transfer 검증 | **HepG2는 간모세포종 유래**, 성인 환자 HCC 직접 검증 아님 |
| **Tahoe-100M SNU-423** | 약물·용량 처리 후 단일세포 RNA + DMSO | 성인 HCC 유래 세포주의 **화학적** perturbation 예측 | **후보 확인**; cell-line/plate/sample/drug/dose 정합 및 충분한 반복 실측 필요 |
| **Replogle K562/RPE1, Nadig Jurkat** | CRISPRi 후 RNA | 타 세포주의 유전적 효과를 간 유래 맥락으로 전이 | modality·target overlap·batch 통제 필수 |
| 기타 환자 HCC·organoid | 자료별로 다름 | 장기 목표: 환자·조직 맥락 | 개입 전후 단일세포 RNA가 존재하는지 개별 검증 필요 |

**데이터 조사 순서:** accession 확인 → 실제 raw/normalized expression 파일 확인 → control과 treated 연결 정보 → target/drug/dose/time/replicate → 사용 가능한 조작 종류·규모 → train/val/test 기준 → 다운로드·저장·GPU 예산 → 적격/조건부/부적격 판정. GEO 초록에 ‘CRISPR screen’이 있다고 해서 post-perturbation RNA 정답이 있다는 뜻은 아니다.

**제안 가능한 두 개의 분리된 간 관련 실험:**
1. **Genetic:** K562/RPE1/Jurkat CRISPRi로 학습하여 HepG2로의 transfer를 시험한다(새 개입 HepG2 처리군은 평가용으로 보류). 기존 연구에서 이미 본 GSE264667은 *새로운 blind cohort*로 칭하지 않는다.
2. **Chemical/HCC:** Tahoe-100M의 SNU-423에서 약물·용량·plate-matched DMSO가 충분한지 확인한다. 다른 49개 세포주에서 학습한 반응으로 SNU-423 약물 반응을 예측하는 leave-one-cell-line-out 연구를 설계한다. 모델은 chemCPA·State·PIE 등 **화학적 개입 지원 모델**에서 선택한다. 사람 환자의 약효/예후 예측으로 과장하지 않는다.

## 5. 후속 연구 방향 3개

| 후보 | 연구 질문 | 관련 모델·데이터 | 핵심 평가 | 장점 | 주요 위험 |
|---|---|---|---|---|---|
| **R1. 세포주 간 유전자 섭동 일반화(우선)** | ‘다른 세포주에서 본 유전자 억제 반응을 미학습 세포주, 특히 HepG2에서 예측할 수 있는가?’ | TxPert/PIE/State + 단순 baseline, 4-line Replogle/Nadig CRISPRi | LOCO 4 fold, seen/unseen target 분리, Δ 및 DEG 지표 | 피드백 2~4번을 한 실험에 연결, 공개 정규화 자료·코드 존재 | cell-line confounding, batch/guide, K562·RPE1과 HepG2의 차이, 성인 HCC 아님 |
| **R2. Full FT 우위 원인 규명** | ‘사전학습 표현을 수정할 필요가 있는가, 아니면 PEFT 최적화·control 구성 문제인가?’ | 기존 Norman 데이터, scGPT M0/M2/M3·random, 제한된 LR/rank/레이어·control ablation | 동일 update budget, seed 반복, 강한 반응/DE/subgroup, GPU 비용 | 기존 자산 재사용·원인 분석에 직접 답변 | 참신성은 상대적으로 작고 hyperparameter tuning으로 끝날 위험 |
| **R3. HCC 약물 반응의 세포주 간 전이** | ‘다른 암세포주의 약물·용량 반응으로 HCC 유래 SNU-423의 전사체 반응을 예측하는가?’ | Tahoe-100M SNU-423+기타 세포주, chemCPA/State/PIE | 미관측 세포주, 미관측 약물/용량, plate 대조군, DEG/response retrieval | **실제 성인 HCC 유래 세포주**와 연결, 임상 관련 가설 발굴 | 대용량·공변량·배치·약물 활성 효과, 환자 HCC 아님 |

**실행 권고:** R1의 공개 데이터·코드 적격성 확인을 1차 Gate로 삼고, 동시에 R2의 기존 로그 분석 및 R3의 SNU-423 소규모 추출을 수행한다. R1의 데이터/모델 정합성이 확보되면 R1의 축소 LOCO proof-of-concept 1개를 실행한다. 이후 R1과 R3 중 해석 가능한 1개를 주제로 선정한다. 세 주제를 동시에 전면 학습하지 않는다.

## 권장 수행 순서와 중단 기준

| 순서 | 작업 | 계산자원 | 다음 단계 Gate |
|---|---|---|---|
| G0 | 기존 실험 결과·코드·학습/검증 조건 감사 및 오류 subgroup 표 | CPU/기존 파일 | 성능 우위가 어떤 개입/유전자에서 나타나는지 설명 가능 |
| G1 | 모델 8~12개, 최신 일반화 벤치마크, 데이터 후보 수집·비교 | 조사/소량 CPU | 코드·가중치·입력·공통 표적 존재 여부 확인 |
| G2 | 4-cell-line 데이터 계약, HepG2 및 SNU-423 소량 추출·대조군 검증 | 소량 CPU/스토리지 | 실측 처리 후 RNA, control, 적합한 split 확인 |
| G3 | R1 LOCO 한 세포주로 단순 기준선 + 최신 후보 1개 검증 | 제한 GPU | 반응 관련 지표에서 baseline 대비 이득 또는 해석 가능한 실패 증거 |
| G4 | 살아남은 후보 2개 이하, 4 LOCO 및 seed 반복, 예산·한계 분석 | GPU | 외부 맥락 전이 결과의 반복성·불확실성 확인 |
| G5 | 3개 방향의 근거를 비교하고 주 연구 방향 1개 선정 | 분석 | 신규 가설·데이터·검증 가능성·실제 HCC 연계 명시 |

### 보고서 체크리스트

- [ ] 기존 Full FT와 PEFT 비교의 **한 가지 학습률·한 분할·초기 단일 시드** 제약 명시
- [ ] D1/D2가 각각 *대조군 연결*과 *개입별 노출 빈도*를 변경했다는 점 명시
- [ ] 실험마다 **가설 / 고정 조건 / 변경 변수 / 데이터 분할 / 결과 / 대안 설명 / 다음 검증**을 표로 작성
- [ ] **원 논문 수치 재현**과 **자체 공통 벤치마크**를 구분
- [ ] **Unseen perturbation**과 **Unseen cell context** 및 **double OOD**를 구분
- [ ] CRISPRa vs CRISPRi vs 화학적 개입을 혼합하여 동일 성능 지표로 비교하지 않음
- [ ] HepG2 vs SNU-423 vs 환자 HCC의 생물학적 증거 수준을 구분
- [ ] 테스트셋 반복 사용·사전학습 데이터 노출·배치/반복 정보의 미확인 사항 표시
- [ ] 완료/미완료/접근 차단 및 GPU 측정 범위·실제 자원 기록

## 선행연구·공개 데이터 출처(선별)

1. Ahlmann-Eltze et al., *Nature Methods* (2025), 단순 선형 기준선 비교: https://www.nature.com/articles/s41592-025-02772-6
2. Wei et al., *Nature Methods* (2026), 일반화 벤치마크(27개 방법, 29개 데이터): https://www.nature.com/articles/s41592-025-02980-0
3. Tian et al., *Science Advances* (2026), 25개 데이터·3개 일반화 과제 비교: https://pmc.ncbi.nlm.nih.gov/articles/PMC13557097/
4. *Systema*, *Nature Biotechnology* (2026), 체계적 변동을 넘어선 평가: https://www.nature.com/articles/s41587-025-02777-8
5. Wenkel et al., *TxPert*, *Nature Biotechnology* (2026), 다중 KG와 cross-cell 예측: https://www.nature.com/articles/s41587-026-03113-4
6. Adduri et al., *State*, Cell (2026), 집단 상태 전이: https://arcinstitute.org/tools/state
7. Verma et al., *PIE* (2026 **preprint**), 미관측 맥락·개입·데이터셋: https://arcinstitute.org/news/pie · https://github.com/ArcInstitute/pie
8. Replogle/Nadig 4개 세포주 CRISPRi 전처리: https://huggingface.co/datasets/arcinstitute/PIE_replogle_nadig_essential · https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/docs/user-guide/datasets.md
9. HepG2/Jurkat 원본 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667
10. Tahoe-100M 공식 자료·스키마: https://huggingface.co/datasets/tahoebio/Tahoe-100M · https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/tahoe-100M/README.md
11. Cellosaurus SNU-423: https://www.cellosaurus.org/CVCL_0366
12. 2026 Virtual Cell Challenge 공식 OOD 맥락 설명: https://arcinstitute.org/news/virtual-cell-challenge-2026
13. 현재 연구 결과: https://github.com/honbul/vcell_liver/blob/540ec932ae7876b7af316bdbe9fc1998d87c4440/reports/FINAL_REPORT.md
