# R12 다축 비교 실행 명세

실행 요청은 `VCELL_R12_MULTI_AXIS_EXPERIMENT_PLAN_2026-11-21.md`이며 SHA-256은 `682cc79ac270368a95f88bc051bb3cb5d7d42b97025a5493f93f3b7f2cf432c9`다. 출발 커밋은 R11 병합 `1e1c6cd`다.

## 데이터와 평가 경로

- Tahoe-100M revision `2dc57900b7981cfcf5e211527169a0b006546a95`의 0.5 µM·24 h, plate2/5/8/11을 사용한다. Plate13은 기존 격리를 유지한다.
- 행은 물리 well×cell line×entity×dose×time이다. 20세포 이상이고 양 물리 DMSO 역할의 적격 control이 존재해야 한다. 대비는 pool당 최소 3개 화합물군을 요구한다.
- 모든 mapped-gene count 합으로 CP10k 정규화한 well pseudobulk에 log1p를 적용한다. 각 fit의 처리 웰에서 정한 유전자 2,000개를 예측한다. 예측 입력은 ECFP4, 자기 역할 DMSO와 사전 고정 주석이다.
- 과거 validation은 chemCPA에서 checkpoint, 학습률, 모델 선택에 사용했다. 과거 final은 이미 채점됐다. Route N의 outer 3-fold와 outer-train별 inner 2-fold를 metadata로 고정했다. [노출 기록](exposure_ledger.tsv) · [분할](split_manifest.tsv)
- Outer 평가는 프로토콜을 동결한 Tahoe 내부 후속이다. R11 historical final 재평가는 project-seen replay로 표시한다.
- Inference는 outer score 응답 파일을 읽지 않는다. R-A의 feature, train delta, 선택, calibration은 A를 사용한다. scorer는 동결 이후 A+예측을 절대 발현과 비교하고, 예측 delta를 처리 발현−B와 비교한다. R-B는 역할을 바꾼다.

## 모델과 개발 선택

| R12 | R11 대응 및 재적합 |
|---|---|
| B0 | B0 no-change |
| B1 | B1_line, group-balanced line 평균 |
| B2 | B1_pool, shrinkage 0/2/8/32 |
| B3 | B2_linear, full ECFP4+raw DMSO Ridge |
| B4 | B2_mlp, full ECFP4+raw DMSO MLP |
| B5 | E, ECFP4 PCA32×DMSO PCA16 상호작용 Ridge |

Offset은 group-balanced pool→line→global, anchor shrinkage 2다. 학습 행은 해당 compound identity group 전체를 제외한다. Query offset은 fit 통계만 사용한다. 절대와 대비 selector 및 epoch는 각 role×outer 내부에서 독립 선택한다. 기준선 tie 허용차는 점수 0.000001이다. Exact L2는 완전한 pool을 pack에 넣고 전체 train weight로 정규화한다. NN은 hidden 128, AdamW weight decay 0.001, dropout 0.1을 사용한다. 최초 lr 0.0003과 seed 42, 개발 확대 lr 0.001, 최대 100 epochs와 두 점수의 공통 patience를 사용한다. 90번째 epoch 이후 정점이 있는 계열과 matched 대조는 horizon을 한 번 늘린다.

S1–S3는 2단계 global→line 규제 회귀다. Shift attribution의 원시 base를 사전에 고정했다. B2는 shrinkage 2, B3/B5는 α=1,000, B4/A4는 lr 0.0003의 20 epochs다. 보정은 parent fit 안의 2-fold group-cross-fit에서만 학습한다. 두 하위 모델에 parent-train의 출력 좌표를 공유하며 각자의 PCA, scaler, offset을 fit한다.

C4 및 VICReg는 같은 물리 control의 metadata-seeded cell sample을 쓴다. VICReg 두 view는 서로 겹치지 않고 정답, offset, 패널을 공유한다. C3은 own-control의 line 평균을 지운 plate 연관 편차에서 rank 0–2를 평가한다. C5는 같은 plate의 고정 순열 다섯 개, C6은 가능한 동일 line의 다른 train plate를 교체한다. A1–A6는 raw와 PCA16 각각에서 concat, FiLM, rank-16 bilinear를 비교한다. J1에는 A4/A5의 MSE, A4의 L2, A5의 L2가 있다.

J2는 한 role×outer의 context, head, loss winner를 조합한다. 보정은 그 조합의 cross-fit 예측으로 계수를 다시 적합한다. Selector는 최선 single, J2 두 개 및 B_contrast를 inner 대비 오차에서 비교한다. 축 winner는 각 신규 family에서 선택한다. 기본 확인 seed는 42, 43, 44이다.

## 공식 효과와 불확실성

| 공식 대비 | 주 지표 |
|---|---|
| Shift−B_abs | 절대 발현 RMSE |
| Context−B_contrast | 약물 대비 RMSE |
| Structure−B_contrast | 약물 대비 RMSE |
| Loss−B_contrast | 약물 대비 RMSE |
| Integrated−B_contrast | 약물 대비 RMSE |

오차는 gene RMSE를 group→entity→line→well 균등으로 집계한다. 대비는 outer×plate×line×dose×time별 정답과 예측을 각각 중심화한 뒤 계산한다. Seed 및 역할을 평균하며 compound group에 같은 양수 Exp(1) 가중을 적용한다. 10,000회마다 pool을 다시 중심화한다. 공식 구간은 99%, 탐색 구간은 95%다. 구간 상한이 0 아래면 관측 조건의 개선 근거, 하한이 0 위면 열세다. 양측을 포함하면 차이가 미확립이다.

## 자원 및 산출물

G1/Slurm에서 동시 GPU 2개와 CPU 4개(작업당 8 threads 이하)를 사용한다. 상한은 96 GPU-hour, 512 CPU core-hour, 노드 가용 RAM의 75%, 신규 다운로드 20 GiB, scratch 100 GiB다. 서버에 예측·정답·모델을 보관하고 해시와 위치를 기록한다.
