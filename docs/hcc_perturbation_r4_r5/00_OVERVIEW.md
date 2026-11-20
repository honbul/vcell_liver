# 추가 섭동 연구 2개 — R4·R5 실행 개요

**작성·조사 기준일: 2026-11-17**  
**상태:** 기획서 작성. 새 실험·Git 수정·서버 학습 미실행.

## 결론: 무엇을 새로 연구하는가?

| 연구 | 큰 가설 | 예측 과제 | 새로 바꾸는 부분 |
|---|---|---|---|
| **R4 반응 관계 기반 약물 표현** | 실측 반응의 유사성과 신뢰도를 약물 표현 학습에 반영하면, 전역 미관측 약물의 RNA 예측이 좋아질 것이다. | 관측 세포에서 **학습하지 않은 약물** → 평균 ΔRNA 978개 | 학습 pair 구성, chemical/context 표현의 감독 신호, soft alignment, 공통 Δ output |
| **R5 noise-aware 실험 선택** | 불확실성 크기만 보지 않고, 측정 변동과 전역 예측 이득·batch 중복을 함께 고려하면 적은 실측값으로 더 정확하게 예측할 것이다. | 새 HCC에서 **source에 알려진 약물** 중 일부를 관측 → 나머지 ΔRNA | oracle/라벨 비용, posterior·noise 표현, outcome-adaptive acquisition, 고정 예측기 평가 |

**둘 다 RNA 반응 예측이다.** R1의 viability/sensitivity 연구와 달리 약물 처리 전후의 발현 변화가 실측 정답이다. R5는 예측기 자체보다 데이터를 고르는 알고리즘이 방법론 기여다. 다른 task의 성공률을 같은 수치로 비교하지 않는다.

## 1. 왜 이 두 개를 골랐는가?

최근 연구의 흐름은 단순한 backbone 교체만이 아니다.
- MolPhenix·MICON은 분자–phenotype 대응에서 반응이 비슷하거나 없는 조건을 어떻게 학습시킬지, 약물이 상태를 변화시킨다는 구조를 어떻게 남길지를 다룬다 [S01–02].
- WAVE·MAP·DECANT는 약물 구조와 상태/기전 표현을 전사체 예측 또는 분석에 연결한다 [S03–05].
- BATCHIE·MORPH·2026 ICML의 hit-discovery 연구는 ‘무엇을 추가로 측정할 것인가’와 목표 함수의 중요성을 다룬다 [S07–09].

R4는 **무엇을 배우도록 입력 표현을 감독할지**, R5는 **무엇을 측정해 학습할지**에 초점을 둔다. 모델 크기·diffusion·LLM이라는 명칭을 성공 근거로 삼지 않는다.

## 2. 기존 연구와의 위치

| 연구 | 범위 | 이번 두 연구와 중복하지 않는 점 |
|---|---|---|
| R1 | HCC RNA→민감도, soft pathway prior | 출력이 다름 |
| R2 | 주어진 소량 support를 사용하는 episodic adaptation | R5는 support 선택, readout 고정. R2의 완료를 기다리지 않음 |
| R3 | Tahoe 평균·분포 분리 생성 | 이번에는 분포 생성 없음 |
| R4 | 전역 미관측 약물의 RNA 예측 | 시험 약물 반응을 source에서도 사용하지 않음 |
| R5 | known drug/new target의 라벨 효율 | source 동일 약물 반응 사용 가능. 이 차이를 섞지 않음 |

## 3. 확보한 자산을 우선 사용

이번에 다시 읽은 프로젝트 데이터 카드 [P01]는 CMap2020 Level3 파일과 978 landmark 입력을 실제로 확보했다고 기록한다. source 161개 풀과 HUH7/JHH5/JHH7 조건이 출발점이다. 이전 HCC-only genotype 연구의 ‘8개 HCC 계열’ 게이트를 다시 요구하지 않는다.

- R4: 새 **전역 약물 split**이 필요하다. HCC training drugs는 허용하므로 HCC zero-shot을 주장하지 않는다.
- R5: source/meta-validation과 HCC oracle을 분리한다. 실제 관측 비용을 K로 세며 추가 target-validation은 없다.
- Level5, Tahoe, RNA sensitivity 데이터와 섞지 않는다.
- source의 다른 암종 정보를 이용하는 연구이며 HCC3는 제한된 적용 맥락이다. 환자 결과와 같지 않다.

## 4. 개발을 허용하되 끝없는 탐색은 하지 않기

각 연구는 네 단계를 따른다.

1. **전제 진단:** 개선할 신호가 실제 데이터에 있는지 작은 분석으로 확인.
2. **통합 구현:** 데이터 표현과 모델·학습 또는 acquisition을 함께 설계.
3. **근거 기반 수정 최대 두 차례:** 실패가 생기면 진단표에서 해당 요소 하나를 바꾸고 전후를 기록.
4. **기여 분리와 동결 평가:** 통합 방법의 이득을 확인한 뒤 ablation. 주지표를 보고 유리하게 변경하지 않음.

‘초기 구현이 음성이면 무조건 종료’도, ‘될 때까지 새 기법을 더하기’도 아니다. 조건이 충족되지 않는 가설은 줄일 수 있어야 한다.

## 5. 병렬 운영

| 항목 | R4 | R5 |
|---|---|---|
| branch | `research/response-geometry-r4` | `research/noise-aware-perturbation-design-r5` |
| 주 자원 | GPU 1개부터, pairing/teacher는 CPU cache | CPU 병렬 posterior/replay, 필요 시 작은 surrogate GPU |
| 첫 출력 | 구조–반응 관계·반복 신뢰도 보고 | random 학습곡선·uncertainty 진단 |
| 공개 비교 | WAVE native + 직접 Δ 변형 | PDBAL-RNA + learned-loss acquisition |
| 쉬운 튜토리얼 | 약물 구조만으로 새 약물 RNA 예측 | 다음 실험 선택→실측 공개→query 예측 |

공유 원자료는 read-only, cache key에 split/role/hash 포함. 다른 branch의 target 예측·test 정답을 이번 알고리즘의 입력으로 읽지 않는다. 최종 후보마다 3 seed, 선택 정책은 paired initial seed 반복을 사용한다. 실제 장비 점유는 scheduler에서 확인하며 성능·메모리·wall time을 미리 단정하지 않는다.

## 6. 지금 바로 전달할 문서

- `01_R4_RESPONSE_GEOMETRY.md`: R4 데이터·수식·통제 비교·진단·/goal.
- `02_R5_NOISE_AWARE_DESIGN.md`: R5 oracle·예산·Bayesian 선택·통제 비교·/goal.
- `03_EVIDENCE.md`: 최신 문헌, 실제 확인 범위, 코드 공개 상태, 신규성 경계.

## 7. 기대하는 결론의 수준

좋은 결과가 나오면 R4는 **반응에 맞춘 표현 학습의 미관측 약물 일반화**, R5는 **같은 실측 비용에서 더 유용한 데이터를 획득하는 규칙**을 주장할 수 있다. 어떤 경우에도 다음은 자동으로 따라오지 않는다.

- 약물의 인과 기전을 복원했다.
- HCC 환자 치료 반응을 예측했다.
- retrospective replay가 새 wet-lab에서 검증됐다.
- 모든 RNA 예측기보다 좋다.

논문으로서의 가치는 최신 이름을 사용했는지가 아니라, **왜 이 데이터에 그 수정이 필요한지와 강한 단순 방법을 넘는지**로 판단한다.
