# 연구 R4 — 결정 기록 (DECISIONS)

각 항목은 **관측 → 가설 → 바꾼 것 하나 → 예상 차이 → 실제 결과 → 다음 결정** 순서다. 항목은 실행 전에 쓰고, 결과는 실행 후 같은 항목에 덧붙였다. 규칙 파일은 `configs/response_geometry_r4/protocol.yaml`이다.

## R4-0. 데이터·분할 (job 25295 실패, 25296 완료)

- 첫 unit 생성 job 25295는 B 데이터 경로 인자 오류로 배열을 읽기 전에 멈췄다. 경로만 고쳐 25296으로 다시 실행했다.
- 그 사이 규칙 변경 1건(amendment 1): update 예산 6,000 → 20,000, 검증 간격 250 → 500. **발현 값을 하나도 읽기 전**이며 모든 팔에 같게 적용한다.
- 약물 그룹 8,288개를 5,802 / 1,243 / 1,243(train / validation / test)으로 나눴다(seed 20261117, metadata만 사용). unit 수 56,431 / 11,731 / 11,519. HCC 시험 약물 그룹은 HUH7 10, JHH5 98, JHH7 10개다.
- 누출 fixture 13개 모두 통과(`leak_fixtures.json`).

## R4-1. 진단 D1–D3 (job 25298, training 약물만)

**관측**
- D1: 15개 세포주(큰 12개 + HCC 3)에서 ECFP Tanimoto와 세포별 평균을 뺀 Δ cosine의 Spearman 중앙값 0.021. Tanimoto ≥ 0.6 쌍 1,543개 중 478개(31%)는 반응 cosine ≤ 0. cosine ≥ 0.5 쌍 4,037개 중 3,759개는 Tanimoto ≤ 0.2.
- D1: 같은 plate를 공유하는 쌍의 평균 cosine(세포주별 0.05–0.10)이 공유하지 않는 쌍(약 0)보다 컸다. **plate 공유가 반응 유사성에 섞여 있다.**
- D2: 반복 plate가 있는 training unit 37,030 / 56,431개(66%). split-half Pearson(세포 평균 제거) 중앙값 0.118, 사분위 0.024–0.283. teacher 공간 top-10 이웃 겹침 중앙값 4.42개(무작위 기대 1.18개, 80개 세포주).
- D2: 같은 half(같은 plate 묶음) 쌍 |cos| 중앙값 0.322, 다른 half 쌍 0.300.
- D3: 69%의 unit이 크기/잡음 비 < 1.5. 반응 크기 3분위별 split-half Pearson 중앙값 0.074 / 0.121 / 0.423.

**판정(규칙 그대로):** 전제 실패 조건은 "split-half Pearson 중앙값 ≤ 0.1 그리고 겹침 ≤ 무작위의 2배"다. 0.118 > 0.1이고 4.42 > 2.35이므로 **전제 실패 아님 → 모델 단계로 진행.** 안정화 시도는 하지 않는다.

**가설(검증 안 함):** 약한 반응의 이웃 관계는 대부분 잡음일 수 있어 hard negative가 해로울 수 있다(D3의 3분위 차이와 일치하지만 학습 효과는 아직 측정하지 않았다). plate 공유 효과 때문에 cell-matched batch 안의 이웃이 plate shortcut일 수 있다.

## R4-2. 닫힌 해법 기준선 (job 25299, validation)

- source panel validation ΔRMSE: 무변화 0.5782, training 세포 평균 0.5538, ECFP+X0 Ridge 0.5469(α=10,000), 세포 평균 + ECFP Ridge 0.5530(α=10,000).
- **두 Ridge 모두 α 격자의 가장 큰 값에서 선택됐다.** 격자는 규칙대로 두고 바꾸지 않는다(기준선이 더 강해질 여지가 있다는 뜻이며, 제안법이 이 기준선을 넘더라도 "격자 끝의 Ridge"라고 함께 적는다).

## R4-3. 스크리닝 계획 (실행 전 기록)

- 개발 subset(12개 세포주, training 그룹 1,000개, training unit 9,985개, 해당 세포주 validation unit 6,557개)에서 seed 42 한 번씩:
  W1 LR {3e-4, 1e-3}; W2 LR × λ {0.05, 0.5}; latent regression LR × λ; Δ-MLP LR. 총 12 run.
- 팔마다 개발 validation ΔRMSE가 가장 낮은 설정을 고르고, 전체 panel에서 seed 42/43/44로 다시 학습한다. W2 최종 후보는 최대 2개.
- W0(원형)는 원형 학습률·batch 32로 격자 없이 전체 panel에서 40 epoch × 3 seed.
- smoke(job 25300, 25301): W0 3,000 update 약 50초(A30), W2 1,000 update 약 15초, GPU 메모리 기록은 run record에 있다. smoke 점수는 선택에 쓰지 않는다.

## R4-3 결과. 개발 subset 스크리닝 (job 25310, 25313, 25314, 25316, 25323–25330; seed 42)

첫 제출 중 10개 job은 GPU 이름을 출력하는 줄(`nvidia-smi | head -1`)이 SIGPIPE(종료 코드 141)로 실패해 Python이 시작되기 전에 끝났다. run 디렉터리가 만들어지지 않았으므로 같은 run id로 다시 제출했다(25323–25332).

| 팔 | 설정 | 개발 validation ΔRMSE | 최적 update |
|---|---|---:|---:|
| W1 | lr 3e-4 | 0.5025 | 1,500 |
| W1 | lr 1e-3 | 0.5027 | 3,000 |
| W2 | lr 1e-3, λ 0.5 | **0.5014** | 13,000 |
| W2 | lr 1e-3, λ 0.05 | 0.5015 | 1,000 |
| W2 | lr 3e-4, λ 0.5 | 0.5018 | 2,000 |
| W2 | lr 3e-4, λ 0.05 | 0.5020 | 1,000 |
| latent regression | lr 1e-3, λ 0.05 | 0.5035 | 3,000 |
| latent regression | lr 1e-3, λ 0.5 | 0.5039 | 3,000 |
| latent regression | lr 3e-4, λ 0.05 | 0.5048 | 3,000 |
| latent regression | lr 3e-4, λ 0.5 | 0.5073 | 1,500 |
| Δ-MLP | lr 3e-4 | 0.5016 | 3,000 |
| Δ-MLP | lr 1e-3 | 0.5025 | 5,000 |

- **관측:** 팔 사이 차이는 0.001–0.006으로 작다. 대부분 1,000–3,000 update에서 최저점이 나오고 그 뒤 validation 오차가 올라간다. 예측 크기(RMS 0.06–0.11)가 실제(약 0.58)보다 훨씬 작다.
- **선택(규칙 그대로, 개발 validation 최저):** W1 lr 3e-4; W2 후보 2개 = (lr 1e-3, λ 0.5), (lr 1e-3, λ 0.05); latent regression (lr 1e-3, λ 0.05); Δ-MLP lr 3e-4.
- **평가 코드 결함 발견·수정:** 보조지표 retrieval MRR에서 동점 처리가 잘못되어, 세포 안에서 예측이 상수에 가까우면(모든 후보가 동점) 순위 1로 계산됐다(무변화·세포 평균 기준선의 MRR 1.0, Δ-MLP lr 1e-3의 0.92가 이 결함 때문). 동점은 중간 순위로 바꿨다. **MRR은 어떤 선택에도 쓰지 않았다.** 기준선 validation 표는 수정한 코드로 다시 계산한다. 개발 run의 MRR 값은 결함이 있는 값이므로 인용하지 않는다.
- 다음: 전체 panel에서 선택 설정 × seed 42/43/44 (W1 3, W2 6, latent regression 3, Δ-MLP 3).
