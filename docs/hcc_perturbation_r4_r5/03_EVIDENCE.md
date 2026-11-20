# 근거와 선행연구: R4·R5

**조사 기준일: 2026-11-17.** 논문·저자 공개자료와 제안자의 새로운 가설을 분리한다. 논문의 성능 개선은 그 논문의 데이터와 지표에서 보고한 결과이며 HCC의 개선을 보장하지 않는다. 공개 API·다른 모델의 예측을 실측 정답으로 사용하지 않는다.

## 1. 이번에 확인한 주요 연구

| ID | 원문·발표 상태 | 확인한 내용 / 이 기획에서 가져올 것 | 그대로 주장하지 않을 것 |
|---|---|---|---|
| S01 | [MolPhenix: How Molecules Impact Cells](https://arxiv.org/abs/2409.08302), 2024-09-10 공개 논문 | 분자와 현미경 표현의 정렬에서 비활성 조건·농도·샘플 사이 유사성을 반영한 손실을 연구했다. R4의 ‘다른 약물은 무조건 negative가 아니다’라는 출발점. | 이미지 retrieval의 개선율을 RNA 예측 개선율로 옮기지 않는다. |
| S02 | [MICON: Integrating chemical structures as treatments improves representations of microscopy images for morphological profiling](https://doi.org/10.1371/journal.pcbi.1014483), PLOS Computational Biology, 2026-07-29 | 약물을 이미지와 동일한 객체로 정렬하기보다 상태를 바꾸는 treatment로 모델링했다. R4는 구조와 세포 상태의 상호작용을 남긴다. | 세포 영상에서의 효과가 전사체에서도 입증됐다는 뜻은 아니다. |
| S03 | [WAVE: Learning chemical-induced gene expression perturbations with WAVE](https://doi.org/10.1038/s41467-026-77645-3), Nature Communications, 2026 | 대조군 발현과 분자 fingerprint를 결합한 약물 RNA 예측. 공개 README는 GeneVAE·DrugNN·GeneDrugFusion을 설명한다. R4의 기반 및 최신 비교군. | 원형 출력과 우리가 변경한 직접-Δ 출력의 점수를 동일 모델 원형 결과로 표기하지 않는다. |
| S04 | [DECANT](https://doi.org/10.1093/bioinformatics/btag662), Bioinformatics, 2026 | 처리군–대조군 cell set에서 맥락과 약물 관련 표현을 구분한다. 처리 후 자료를 받아 표현을 분석하는 문제와 사전 예측을 구분해야 함을 보여준다. | 처리 후 RNA를 입력받는 분석기를 처리 전 예측기와 그대로 겨루지 않는다. |
| S05 | [MAP: A knowledge-driven framework for predicting single-cell responses for unprofiled drugs](https://doi.org/10.1038/s42256-026-01286-w), Nature Machine Intelligence, 2026 | 지식·다중모달 약물/유전자 표현을 활용한 미관측 약물 예측. 단순 ECFP 교체 이상의 기전 정렬 연구가 이미 있다는 직접 선행이다. | R4가 최초의 기전-aware 모델이라고 주장하지 않는다. 이번에는 대규모 KG 구축을 하지 않는다. |
| S06 | [CytoWave](https://arxiv.org/abs/2611.02311), 2026-11-02 공개 프리프린트 | 관측 세포 표현만 사전학습하는 대신 실제 유전자 개입 반응을 사용한 두 단계 사전학습·contrastive 정렬을 제안한다. | 유전적 섭동·교차 종 결과를 약물/HCC에서 검증된 결과로 옮기지 않는다. |
| S07 | [A Bayesian active learning platform for scalable combination drug screens](https://doi.org/10.1038/s41467-024-55287-7), Nature Communications, 2025-01-02, BATCHIE | 확률적 모델과 순차적 배치 선택으로 실험 공간을 효율적으로 학습한다. R5의 전역 예측 불확실성 감소·batch 중복 제거의 근거. | 조합약물 생존율 결과를 단일약물 978차원 RNA 예측과 직접 비교하지 않는다. |
| S08 | [MORPH](https://www.biorxiv.org/content/10.1101/2025.06.27.661992v2), 2026-07-01 개정 프리프린트; [초기 공개 전문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12236822/) | 초기 공개 전문의 active-learning 절은 불확실성·다양성 선택이 random과 비슷했던 조건 및 learned-loss 선택의 이득을 보고한다. R5는 random을 강한 비교군으로 유지한다. | 최신 개정의 모든 세부 결과를 초기 전문과 동일하다고 간주하지 않는다. 유전자 선택 알고리즘의 약물 적용은 adapted baseline이다. |
| S09 | [Many Needles in a Haystack: Active Hit Discovery for Perturbation Experiments](https://proceedings.mlr.press/v306/rubbi26a.html), ICML 2026 Proceedings | ‘최대값 하나’, ‘hit 여러 개’, ‘전체 예측 오차 감소’가 다른 목적임을 구분하고 Probability-of-Hit를 제안한다. | R5의 RNA 정확도 목적에 hit 획득 점수를 그대로 쓰거나 큰 RNA 변화를 약효로 해석하지 않는다. |
| S10 | [Systema](https://doi.org/10.1038/s41587-025-02777-8), Nature Biotechnology, 2025 온라인 / 2026 권호 | 공통 변화와 섭동 특이적 변화의 구분을 강조한다. | 평균 차감만으로 batch가 제거되거나 인과성이 입증됐다고 하지 않는다. |
| S11 | [Deep learning perturbation models can outperform baselines on calibrated metrics](https://doi.org/10.1038/s41587-026-03307-w), Nature Biotechnology, 2026-10 | 양성·음성 대조로 평가 지표가 반응 신호에 민감한지 확인할 필요를 제시한다. | 유리한 지표만 사후 선택하거나 모든 음성 결과를 지표 탓으로 돌리지 않는다. |
| S12 | [TxPert](https://doi.org/10.1038/s41587-026-03113-4), Nature Biotechnology, 2026-05-01 | batch에 맞는 대조군과 retrieval, split-half 진단을 함께 다룬다. | 유전적 섭동 그래프/비공개 표현의 결과를 그대로 복제할 수 있다고 가정하지 않는다. |
| S13 | [A systematic comparison of single-cell perturbation response prediction models](https://pubmed.ncbi.nlm.nih.gov/42715312/), Science Advances, 2026-09-11 | 데이터·과제·지표와 개입 효과 크기에 따라 성능이 달라지는 비교를 제공한다. | ‘복잡한 모델은 늘 실패한다’ 또는 ‘항상 성공한다’는 결론으로 읽지 않는다. |

## 2. 코드 접근을 직접 확인한 범위

### WAVE — R4의 필수 외부 비교군
- [공식 저장소](https://github.com/ZJUFanLab/WAVE)
- 확인한 HEAD: `e27e453beb48d0ff912585060b5a09cb04335262`.
- README: baseline expression, compound fingerprint, treated expression을 사용하는 모델과 학습 인터페이스가 있음.
- 이번 작성에서는 설치·학습·가중치 노출 검사까지 실행하지 않았다. 에이전트는 해당 commit의 실제 모델/학습 코드를 확인하고 입출력 fixture를 만든다.
- 기본 README의 random split과 test logging을 그대로 사용하지 않는다. 본 연구의 전역 약물 split과 별도 평가기로 변경하며 변경점을 기록한다.

### BATCHIE — R5의 acquisition 기준선
- [공식 저장소](https://github.com/tansey-lab/batchie)
- 공개 README에서 Bayesian model 기반 batch selection과 retrospective mode, holdout 분리를 확인했다.
- 기본 입력은 scalar viability 중심이다. R5의 다변량 RNA용 posterior wrapper가 필요하다. 결과는 `PDBAL-RNA adaptation`으로 표시하며 원 논문의 직접 재현이라 부르지 않는다.
- 버전과 최종 commit은 최초 설치 때 고정한다. 대형 workflow를 그대로 띄우는 것이 필수는 아니다.

### DECANT — 비교 적격성 주의
- [공식 저장소](https://github.com/bliulab/DECANT)의 README는 ‘reproduction skeleton’이며 대형 가중치·가공 cell-bag는 포함되지 않았다고 설명한다.
- 이번 데이터는 L1000 bulk다. 따라서 DECANT를 임의 변형해 필수 전사체 baseline으로 넣지 않는다. 관측 phenotype 분석과 chemical-only 미래 예측의 차이를 설명하는 가까운 연구로 사용한다.

### MAP / MORPH
- [MAP 공식 코드](https://github.com/MAGIC-AI4Med/MAP)는 검색 결과에서 확인했지만 이번 기획에서 설치·모든 입력 자산의 실행 가능성까지 검증하지 않았다.
- MORPH의 방법과 acquisition 비교는 공개 논문에 근거한다. 유전자 모델을 약물 모델로 옮긴 비교는 명확히 adapted라고 표시한다.

## 3. 재사용할 프로젝트 근거

| ID | 정확한 소스 | 확인 범위 |
|---|---|---|
| P01 | [연구 B 데이터 카드](https://github.com/honbul/vcell_liver/blob/d67bf68495be49a96598473c22fb5e2409d24429/reports/transigen_adaptation_b1/DATA_CARD.md) | CMap2020 Level3 실파일·동일 978 genes·nominal 10µM/24h·source161·HCC3·plate별 대조군 연결. 이번에 원문 재확인. |
| P02 | [연구 B 결과](https://github.com/honbul/vcell_liver/blob/d67bf68495be49a96598473c22fb5e2409d24429/reports/transigen_adaptation_b1/RESULTS.md) | 이전 mean-offset 비교, HCC 검증 라벨 추가 비용. 기존 결과는 가설의 출발점이지 새 연구의 성능 결과가 아니다. |
| P03 | `HCC_3TRACK_METHOD_RESEARCH_2026-11-17/02_R2_EPISODIC_RESIDUAL_ADAPTATION.md` | 주어진 support를 사용하는 학습 규칙이 중심이며 active support selection은 제외했다. 첨부 원문을 다시 읽고 R5와의 경계를 정했다. |
| P04 | `HCC_3TRACK_METHOD_RESEARCH_2026-11-17/03_R3_ANCHORED_RESIDUAL_DIFFUSION.md` | 평균/분포 분리의 생성 연구. 이번 R4/R5는 분포 생성기를 만들지 않는다. |
| P05 | `HCC_3TRACK_METHOD_RESEARCH_2026-11-17/01_R1_SOFT_PRIOR_MULTITASK.md` | sensitivity AA 과제이며 RNA 반응 예측과 구분된다. |

## 4. 이번에 일부러 채택하지 않은 경로

- 최신이라는 이유만으로 diffusion·MoE·LLM 에이전트를 또 추가하지 않는다.
- CRISPRi escape/effectiveness 연구는 흥미롭지만, 이번 약물 자료에 sgRNA/독립 효율 정답이 없으므로 두 계획에 억지로 넣지 않는다.
- 공개 Atlas의 모델 예측을 wet-lab 정답으로 바꾸지 않는다.
- 다중 HCC 계열이 부족했던 genotype-LOCO 가설을 다시 동일한 자료 조건으로 시작하지 않는다.
- 새 약물과 새 세포주, 새 dose, 새 실험실을 한 번에 보류하지 않는다.

## 5. 신규성·근거 수준

R4의 soft similarity와 R5의 Bayesian design 원리는 이미 있다. 이번 제안의 독창성 후보는 **실측 반복 신뢰도·context matching·chemical-only 추론** 또는 **다변량 반응·숨겨진 라벨 비용·실험 잡음·batch 의존**을 하나의 검증 가능한 파이프라인으로 결합하는 부분이다. 가까운 연구와 실제로 중복된다면 제목·주장을 줄이고, 최초 방법이라고 부르지 않는다.

논문·공식 코드 목록 확인과 데이터 카드 재확인까지 수행했다. 새 원자료 전체를 다시 집계하거나 새 모델을 실행한 것은 아니다. 논문 전문을 읽을 수 없었던 자료는 초록·공식 설명의 범위만 요약했고, 구체 수치나 미확인 학습 설정을 추가하지 않았다.
