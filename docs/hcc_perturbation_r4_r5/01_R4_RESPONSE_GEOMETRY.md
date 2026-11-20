# 연구 R4: 실측 반응의 유사성과 신뢰도를 반영한 약물 표현 학습

**영문 가제:** *Reliability-aware response geometry for unseen-compound transcriptional prediction*  
**버전:** R4-v1.0 / 2026-11-17  
**전용 브랜치:** `research/response-geometry-r4`  
**산출 범위:** 약물 개입 후 **평균 RNA 변화 978개**. 환자 약효·단일세포 분포·유전자 knockout을 이번 출력에 포함하지 않는다.

> **주가설:** 약물의 화학 구조 표현을 실측 처리 전후 반응으로 학습할 때, ‘서로 다른 약물은 모두 다른 반응’이라고 강제하기보다 **같은 세포 맥락에서 반응이 비슷한 약물은 가깝게, 불확실한 반응 관계는 약하게 학습**하면, 학습에 전혀 쓰지 않은 약물의 RNA 반응 예측이 개선될 것이다.

## 0. 쉬운 예와 기존 R1–R3와의 차이

구조가 다른 약물 A와 B가 같은 경로를 건드려 비슷한 RNA 변화를 만들 수도 있다. 반대로 구조가 비슷한 A와 C가 이 세포에서 다른 반응을 낼 수도 있다. 화학 구조의 거리만으로 약물 반응의 거리가 정해지지 않을 가능성에 착안한다.

이번에는 **학습 약물의 실제 RNA 반응으로 ‘어떤 약물 특징을 중요하게 읽어야 하는지’를 훈련**한다. 새로운 약물의 처리 후 RNA는 입력으로 받지 않는다.

```text
학습 단계
약물 구조 + 처리 전 RNA ── 학생 예측기 ── 예측 ΔRNA
                                  ↑
학습용 실측 반응들의 관계 ── 신뢰도 있는 유사성 지도

사용 단계
처음 보는 약물 구조 + 처리 전 RNA ── 고정된 예측기 ── 예측 ΔRNA
                                      (실제 처리 후 RNA는 없음)
```

- R1은 민감도 숫자의 입력 prior를 바꾼다. R4는 978차원 RNA 반응 예측이다.
- R2는 target에서 몇 개 반응을 이미 본 상태의 적응이다. **R4는 시험 약물의 반응을 모든 세포주에서 제외한다.**
- R3는 세포 분포 생성이다. R4는 mean response와 입력 표현을 학습하며 diffusion을 사용하지 않는다.
- 기존 B의 약물은 source에서 이미 관측됐었다. 새 split이므로 과거 점수를 직접 비교하지 않는다.

## 1. 문헌 → 미해결 부분 → 검증할 가설

| 근거 | 이미 한 것 | 남는 문제 / 이번 설계 |
|---|---|---|
| MolPhenix [S01] | 이미지–분자의 contrastive 정렬에서 약한 반응과 false negative 문제를 다룸 | RNA에서는 처리 전 상태·대조군·실험 반복을 반영해 관계를 구성해야 함 |
| MICON [S02] | 화합물을 phenotype과 동일한 객체가 아니라 상태를 바꾸는 treatment로 모델링 | 약물만으로 불변 반응 embedding을 만들지 않고 **세포 상태 조건부 반응**을 예측 |
| WAVE [S03] | 대조군·화학 구조를 결합한 전사체 예측 | 직접적인 공개 출발점. 표준 예측 loss에 추가하는 반응 관계의 효과를 분리 |
| DECANT·MAP [S04–05] | 맥락/기전 표현 또는 외부 기전 지식의 중요성을 연구 | 관측 반응 분석과 미래 반응 예측을 구분하고, 대규모 KG 없이 실측 관계를 활용 |
| CytoWave [S06] | 개입 자료 중심 사전학습·정렬을 유전적 섭동에서 제안 | RNA 중심 pretraining의 필요성을 참고하되 HCC 약물 효과의 근거로 확대하지 않음 |

**단순한 ‘contrastive loss 추가’보다 깊은 점:** 같은 맥락끼리 학습쌍을 만들고, 반복 신뢰도로 관계의 확실성을 구분하며, 약한 반응을 강제로 멀리 보내지 않고, 예측기는 원래 수치 정답도 보존한다. 각 설계는 아래 단계에서 별도로 검증한다.

**독창성 한계:** soft-label contrastive, teacher–student, context-conditioned prediction은 기존 기술이다. 최초 주장이 아니라 도메인에 맞춘 결합의 추가 가치를 검증한다.

## 2. 고정할 데이터 계약

| 항목 | 주 분석 |
|---|---|
| 플랫폼 | 이미 확보한 CMap LINCS 2020 **Level 3** 한 release [P01] |
| 개입 | 단일 저분자 약물, 공식 `nearest_dose=10 µM`, 24h |
| 세포 | 기존 non-target source와 HUH7/JHH5/JHH7 중 실제 적격한 계열. HCC의 **학습 약물 반응은 허용** |
| 입력 | 같은 조건에 대응하는 대조군 X0, ECFP4 2,048 bit |
| 출력 | 978개 직접 측정 landmark의 평균 `Δ=X1−X0` |
| 정답 단위 | cell family × canonical compound group × nominal dose/time; plate별 Δ를 계산한 후 plate 동등 평균 |
| 일반화 | **관측 세포 맥락에서 전역 미관측 약물** |
| HCC의 역할 | 사전 지정한 별도 적용 표. HCC 세 계열로 환자 일반화를 주장하지 않음 |

기존 데이터 카드에는 8,288개 약물 그룹과 source161+target3 계열의 자산이 기록돼 있다. 이는 출발점 규모이며 새 분할·family/QC 후 최종 수를 manifest에서 다시 기록한다. 파일을 다시 전부 다운로드하지 않는다.

### 2.1 전처리에서 지켜야 할 것
- Level3에 scRNA용 `normalize_total/log1p`를 적용하지 않는다.
- 기존 gene 순서·약물 별칭·퇴화 plate 제외 규칙을 재사용하되 hash/버전을 남긴다.
- scaler·PCA·반응 평균·유사성 지도는 **training drug의 처리 반응만** 사용한다.
- 학습 sample을 세포나 well 수만큼 중복 가중하지 않는다. cell→drug→plate 순으로 균형화한다.
- 기존 B의 target holdout을 재사용하는 것이 아니다. 본 연구에 맞는 drug split을 별도 생성한다.

### 2.2 분할
1. 같은 이름/parent structure/canonical SMILES/지문으로 묶인 group은 한 role만 가진다.
2. metadata만으로 drug group을 70/15/15 train/validation/test로 나눈다. 모든 세포주에서 같은 group은 같은 role이다.
3. validation은 개발에 사용, test는 동결 이후 한 번만 평가한다.
4. HCC test 수가 적으면 HCC는 탐색적 보조 결과로 보고한다. 더 많은 test를 얻으려고 점수를 보고 split seed를 바꾸지 않는다.
5. 일반 drug-holdout은 scaffold-holdout과 다르다. 성능이 있으면 하나의 사전 고정 scaffold split을 독립 stress test로 추가할 수 있으나 주 과제와 섞지 않는다.
6. 이미 결과를 접한 공개 데이터다. ‘새 seed를 만들었으니 완전 blind’라고 부르지 않는다.

## 3. 첫 작은 분석: 입력 표현을 바꿀 만한 신호가 있는가?

### D1. 화학적 유사성과 반응 유사성의 관계
training의 같은 cell·dose/time에서 약물 쌍을 만든다. ECFP Tanimoto, Δ의 correlation/distance, 공통 반응을 뺀 거리의 관계를 본다.
- 구조가 가까운데 실측 반응이 다르거나, 구조가 먼데 반응이 비슷한 쌍이 있는지 기록한다.
- 다른 plate만 비교해서 생긴 관계인지 확인한다. 약물군·배치의 불균형을 표시한다.
- 단순 상관이 작다고 화학 구조가 쓸모없다는 결론을 내리지 않는다.

### D2. 그 반응 이웃이 반복해서 관찰되는가?
독립 plate가 있는 training 조건에서, plate들을 두 묶음으로 나누어 Δ와 반응 이웃을 각각 계산한다. 반복이 없는 조건은 `reliability_unknown`이다.
- 별도 plate 반복과 같은 plate의 well 반복을 구분한다.
- 같은 값에 임의 noise를 붙여 독립 반복을 만들지 않는다.
- 공통 DMSO 오차가 약물 쌍의 유사성을 부풀릴 수 있으므로 반복이 있는 곳에서 control pool을 분리하는 민감도 진단을 한다.

### D3. 약한 반응에서는 왜 강한 negative가 위험한가?
DMSO–DMSO 차이 또는 training 반복 오차와 비슷한 반응을 찾는다. 이 조건들 사이를 무조건 멀리 보내는 것이 실제 반응 관계와 맞는지 본다.
- 약한 반응을 전체 학습에서 삭제하지 않는다.
- **오차 회귀에는 그대로 포함**, 관계 학습만 불확실성에 맞춰 약화한다.

**분기:** 반복 이웃 자체가 불안정하면 교사 지도를 크게 만들지 않는다. train-only shrinkage/저차원 안정화 한 번을 시험하고 그래도 재현되지 않으면 이 가설의 전제를 지지하지 못한 것으로 종료한다. 모든 대조를 통과할 완벽한 데이터만 요구해 무조건 정지하지 않는다.

## 4. 구현할 모델: 공개 WAVE 기반의 통제된 변형

### 4.1 기반과 원형을 구분
공식 WAVE는 GeneVAE, DrugNN, GeneDrugFusion으로 구성되며, 원형은 복원한 기저 발현에 약물 delta를 더한다 [S03]. 확인한 commit은 `e27e453beb48d0ff912585060b5a09cb04335262`다.

- **W0 native:** 원형 학습/출력. 우리 drug split과 공통 평가를 적용한다.
- **W1 Δ-anchor:** 같은 encoder·DrugNN·fusion의 표현을 사용하되, 최종 `X0+Δhat`에서 관측 X0를 anchor로 하고 직접 Δ 회귀를 학습한다. 변경된 모델임을 이름에 표시한다.
- **W2 proposed:** W1과 같은 파라미터·회귀 손실·계산 예산에서 아래 반응 관계 목적을 추가한다.

이렇게 해야 W2의 이득이 원형의 기저 재구성 오차를 제거했기 때문인지, 새 표현 학습 때문인지 구분할 수 있다. W0→W1은 출력/학습 계약 변경, W1→W2는 반응 관계 학습의 효과다.

### 4.2 student: 구조 + 기저 상태 → 반응
- 기저 RNA encoder와 약물 ECFP encoder는 WAVE 구현을 출발점으로 한다.
- 융합 표현은 cell/drug의 상호작용을 유지한다. 모든 cell에서 같은 drug vector와 실측 반응을 강제로 일치시키지 않는다.
- 첫 비교에서 거대 화학 foundation model·새 KG·MoE를 넣지 않는다.
- BatchNorm을 변경해야 하면 원형과 변경 모델에 대한 영향 구분을 source smoke에서 기록하고 비교 팔 모두 동일하게 적용한다.

### 4.3 teacher: training 실측의 ‘관계’
training 조건 i의 Δ를 y_i라 한다. 같은 cell의 training drug 평균 m_c를 빼고 training-only scale D로 정규화한다.

`u_i = Uᵀ D⁻¹ (y_i − m_c)`

- U는 training 반응에서만 구한 PCA 64축을 기본으로 한다. 최종 output은 그대로 978차원이다.
- 낮은 분산의 잡음을 확대하지 않게 scale에 training 기반 floor를 둔다.
- teacher는 외부 생성 모델이 아니라 **실측값에서 만든 고정 지도**다.
- m_c는 약물마다 다른 생물학적 반응을 남기기 위한 정의일 뿐, batch를 완전히 제거한 인과 반응이 아니다.
- teacher 조건을 두 반복 view로 만들 수 있으면 서로 다른 view에서 positive와 이웃 관계를 구성한다. 부족하면 단일 consensus를 쓰되 그 부분에 높은 신뢰도를 부여하지 않는다.

### 4.4 soft target와 신뢰도
같은 cell·dose/time의 batch 안에서 teacher cosine similarity를 사용해 q_ij를 만든다.

`q_i = (1−ρ) · one_hot(i) + ρ · softmax(sim(u_i,u_j)/τ_T)`

- 기본 ρ=0.5. ρ=0은 일반 paired alignment 대조다.
- 서로 다른 약물이 비슷한 반응이면 반드시 0인 negative로 두지 않는다.
- 독립 반복에서 이웃 순서가 불일치하는 pair는 q의 비대각 기여를 줄인다.
- 매우 약한/불확실한 조건의 row weight r_i를 0~1로 수축한다. 반복이 없는 것은 별도 unknown prior로 두며 WT/무반응처럼 확정하지 않는다.
- 신뢰도 산식·threshold는 training 반복에서 정하고 validation으로 제한 선택한다. 시험 반응을 써서 ‘고품질 subset’을 학습에 제공하지 않는다.

학생의 예상 반응 표현 z_i와 teacher key u_j의 cross-modal logits로

`L_rel = − mean_i r_i Σ_j q_ij log softmax_j(sim(z_i,u_j)/τ_S)`

를 계산한다. 모든 teacher·q·r는 stop-gradient다. `L_pred = mean_condition mean_gene(Δhat−Δ)²`를 함께 유지한다.

`L_total = L_pred + λ_rel L_rel`

λ_rel=0이면 W1으로 환원된다. 서로 다른 약물/평균을 무조건 밀어내는 hard-negative loss를 추가하지 않는다. z projection을 쓰면 동일 크기의 projection을 대조에도 붙이거나 무조건 사용하지 않는 capacity 대조를 둔다.

### 4.5 학습과 사용의 차이
학습에는 training의 X1이 손실·teacher 계산에 필요하다. **validation/test의 X1은 예측 input으로 전달하지 않는다.** 최종 함수는 `predict(x0, drug_features)`이며 teacher 자료를 조회할 수 없다.

단계 학습을 쓰면 W1도 같은 update 수와 같은 training label을 사용하게 한다. 더 오래 학습한 효과를 표현 학습 효과로 비교하지 않는다.

## 5. 비교군과 작은 실험의 순서

| 실험 | 변경 | 답할 질문 |
|---|---|---|
| R4-E0 | mean/Ridge + 공개 WAVE native | 정답·입출력·기준 성능 점검 |
| R4-E1 | WAVE-derived Δ-anchor W1 | 기저 재구성보다 직접 변화량 학습이 유리한가? |
| R4-E2 | W2 통합 방법 vs W1 | 새 표현 학습의 실제 이득이 있는가? |
| R4-E3 | one-positive vs soft similarity | hard negative가 병목이었는가? |
| R4-E4 | 동일 soft target, reliability 사용/제거 | 신뢰도를 다룬 이득인가? |
| R4-E5 | cell-matched pairing vs 통합 pairing | cell 맥락을 무시한 정렬의 문제인가? |

E3–E5를 처음부터 전 조합 grid로 실행하지 않는다. E2에서 성능 또는 진단상의 유망한 신호가 있으면 하나씩 추가한다. 설명용 retrieval만 좋아지고 ΔRMSE가 나쁘면 예측 성공으로 표기하지 않는다.

### 강한 baseline
- 무변화·training cell별 평균 반응.
- ECFP+X0 Ridge와 작은 Δ-MLP.
- 같은 teacher를 바로 회귀하는 latent regression baseline: contrastive 관계가 반드시 필요한지.
- 공개 WAVE native와 동일 출력계약 W1.
- **same-drug mean은 여기서 불가능**하다. test drug의 어떤 cell 반응도 학습에 없기 때문이다. 이를 쓰면 누수다.
- MAP은 가까운 문헌 비교다. 추가 KG·별도 입력을 가진 MAP 전체를 필수 실행에 넣어 범위를 확대하지 않는다.

## 6. 두 번의 근거 기반 개선 cycle

| 관측 | 허용 수정 | 검증해야 할 것 |
|---|---|---|
| 학습 alignment는 좋고 test 회귀는 나쁨 | teacher 차원/λ를 낮추거나 초기 정렬 후 회귀 중심으로 전환 | 정렬 과제만 풀고 실제 반응량을 잃었는가? |
| 같은 반응 약물을 강제로 구분하며 불안정 | hard→soft pair target; source의 반복 이웃만 사용 | false-negative 완화의 효과 |
| 기저 상태/plate로 이웃이 결정 | 같은 context로 pair를 제한하고 독립 plate view 사용 | 기술적 환경 shortcut 감소 |
| 약한 반응이 거의 사라짐 | 관계 loss에서만 weight를 낮추고 회귀 표본수는 동일 유지 | 효과가 큰 조건만 골라 생긴 이득인지 |
| 반복 신뢰도와 학습 이득이 무관 | reliability를 끄고 더 단순한 정렬만 보고 | 임의 가중치가 필요하지 않았다는 결과 |

최대 두 cycle, cycle마다 핵심 변경 하나. LR·모델 크기·gene panel·정답을 동시에 바꾸지 않는다. 성능이 없을 때 새 데이터 플랫폼을 추가하지 않는다.

## 7. 평가 및 해석

**주지표:** 조건별 978-gene ΔRMSE → drug 평균 → cell-family 동등 평균. 같은 test 조건을 모든 모델에 사용한다.

보조지표:
1. per-condition ΔPearson.
2. 예측과 실측에서 **각자의 cell별 평균**을 뺀 Pearson. inference에는 시험 평균을 사용하지 않는다.
3. source와 독립적으로 고정한 후보 풀에서 retrieval(MRR, normalized rank). 후보 수를 보고하고 비슷한 약물이 여러 정답인 상황을 별도로 표시한다.
4. 실측 |Δ| top-20 방향/회복률. 통계적 DEG로 부르지 않는다.
5. 예측 RMS·실측 RMS, 조용한 조건/강한 조건의 성능을 함께 보고한다. 강도별 층화는 채점용이며 설정을 고르는 데 사용하지 않는다.

CI는 drug group 전체를 재표집해 모든 cell을 함께 이동시킨다. HCC 3개 평균은 그 3개에 대한 요약이지 HCC 모집단 통계가 아니다.

학습·선택 중에는 test를 보지 않는다. test의 효과 크기로 정한 Top-20은 채점에 사용할 수 있지만, 해당 gene set을 모델 입력·loss로 돌려보내지 않는다.

## 8. 진행·종료 판단

- 진단에서 관계가 재현되지 않아 λ_rel=0이 계속 선택되면 ‘실측 관계를 auxiliary로 쓰는 이득 미확인’으로 종료 가능.
- W1이 W0를 이기고 W2가 W1을 못 이기면, direct-Δ output의 이득이지 contrastive 성공이 아니다.
- W2가 같은 정보의 W1·latent regression·Ridge를 반복해서 이겨야 추가 가치가 있다.
- source panel에서는 좋고 HCC에서는 불명확하면 HCC 개선을 주장하지 않는다.
- 신규성은 S01–S06과 비교해 좁혀 쓴다. 기전 latent, pathway causality, 처음 보는 환자 예측을 주장하지 않는다.

## 9. 운영·산출물

개발 subset: training drug의 metadata coverage로 선택한 최대 1,000개 약물 × 최대 12개 cell. 본 실험은 조건이 허용되는 전체 panel로 확대한다. 숫자는 과학적 필수조건이 아니라 운영 상한이다.

초기 LR 두 개, λ_rel 두 개 정도의 제한 탐색 후 최종 후보 두 개 이하를 3 seed로 반복한다. 실제 GPU 메모리와 처리량은 smoke에서 측정한다. GPU가 많다는 이유로 조합 전체를 무제한 탐색하지 않는다.

```text
configs/response_geometry_r4/
manifest/response_geometry_r4/{data_contract,drug_split,pair_views,teacher_provenance}.json
src/response_geometry_r4/
reports/response_geometry_r4/{RESULTS,BEGINNER_SUMMARY,DECISIONS,EXPOSURE}.md
tutorials/response_geometry_r4/
```

`DECISIONS.md`는 관측→가설→수정 하나→예상 차이→실제 결과→다음 결정을 남긴다. 원자료는 read-only. R1–R3의 신규 모델/결과에 의존하지 않는다.

## 10. /goal

```text
/goal
honbul/vcell_liver의 research/response-geometry-r4에서 R4 계획을 수행하라.
기존 CMap2020 Level3 10µM/24h·978genes를 재사용하되, 약물 group을 모든
cell에 걸쳐 train/validation/test로 분리한다. HCC 학습 약물은 허용하고,
시험 약물의 실측 반응은 어느 cell에서도 모델·teacher·prior 학습에 사용하지 않는다.
먼저 같은 context의 구조–반응 관계와 독립 plate 반복의 안정성을 분석한다.
공개 WAVE native, 공통 Δ-anchor 변형, 신뢰도·유사성 기반 정렬을 순서대로 비교한다.
teacher는 training wet-lab 반응만으로 만들고 추론 함수에는 X0와 화학 정보만 제공한다.
weak response를 전체 회귀 학습에서 삭제하거나 다른 약물을 모두 hard negative로 강제하지 않는다.
초기 실패는 관측 근거에 맞춘 수정 최대 두 번으로 검증하며, 개선이 있을 때 ablation한다.
Ridge와 같은정보·같은예산 baseline을 포함하고 고정 평가와 HCC 하위결과를 구분한다.
가중치/데이터 노출, negative 결과, 작은 튜토리얼과 쉬운 해설을 전용 경로에 기록하라.
기존 branch를 덮어쓰거나 R1–R3의 결과를 새 정답으로 사용하지 않는다.
```

**근거:** S01–S06, S10–S13, P01–P05. 전체 제목·원문·확인 범위는 `03_EVIDENCE.md` 참조.
