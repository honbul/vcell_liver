# S·L 구조와 손실의 DEV A 비교

GSE92742 HEPG2 6 h의 보호 입력 제외 A 개발 명부를 사용한다. 1,862 parent와 6,773개 물리 웰을 기존 outer 3-fold와 inner 분할로 유지한다. 서로소 물리 DMSO fit·select·score를 순환한다. 978개 유전자에서 정답은 score DMSO를 뺀 처리 대비다. 약물 donor를 입력으로 주지 않는다. 적합·채점에서 같은 약물 대비 pool의 물리 RNA plate·세포주·시간·용량을 맞춘다. 기존 캐시와 실제 용량이 다른 웰 284개 때문에 pool 구성만 실측 µM로 정정한다. 모든 후보의 훈련 batch 구성에 같은 실측 기준을 적용한다. [실측 지원 원표](data/x_l_support/l1/support.tsv)

모든 후보는 같은 위치 anchor와 구조 커널 규칙을 사용한다. 각 arm의 학습률, epoch, 실험 변형 값과 커널·신경망 gain을 독립 select 정답의 R17 공동 손실로 선택한다. 기존 학습률 `.0003, .001`, epoch `1, 5, 10, 20, 40`, gain `0, .125, .25, .5, .75, 1`과 seed 42를 쓴다. R17 DEV의 checkpoint 문자열·gain 동률 규칙을 적용한다. 외분할에는 고른 설정만 outer-fit에서 다시 적합한다. S0와 L0는 같은 위치·커널·MSE FiLM 결합 `B_R17`을 가리킨다.

| Arm | 사전 고정 변경 |
|---|---|
| S1 | 동일 폭 concat. 추가된 구조 효과는 실제 parameter 수 및 MAC와 함께 보고 |
| S2 | 기존 FiLM 벡터의 방향과 별도의 비음수 magnitude. MSE 유지 |
| S3 | Fit-only parent 가중 PCA 출력, rank `32, 64, 128` |
| L1 | 동일 pool의 서로 다른 parent별 평균 잔차에 대한 짝 제곱차 + MSE. 가중 `.01, .1, 1`. 두 parent 미만 pool의 추가항은 0 |
| L2 | Fit 잔차 norm 하위 10분위수로 가중·안정화한 방향항 + MSE. 가중 `.01, .1, 1` |
| L3 | Fit 잔차 scale에 `.5, 1, 2`배 전환점을 둔 Huber. 영점 주변 MSE 곡률 유지 |

기준선 `B0`, `B_pos`, `B_R15`, `B_mean`, `B_resmean`, `B_context`도 같은 축에 둔다. `B_context`는 고정 상수 drug branch와 위치 anchor로 구조 커널 없이 학습한다. S0의 화학 parent shuffle, 맥락 제거, 맥락 shuffle을 재학습한다. Shuffle은 적합과 조회에서 독립이며, 화학은 parent, 맥락은 같은 세포의 물리 plate 단위다. 이 세 조합은 공통 커널을 공유하고 신경망 가지의 기여를 진단한다. S/L의 화학 커널을 뺀 예측도 기술 비교로 저장한다.

주 비교는 S2−S0와 L1−L0의 전체 결합이다. 두 지표는 parent-macro RMSE와 pooled R²이며 짝 parent bootstrap 10,000회, 양측 99% 구간을 낸다. 주 비교에는 parent 교환 10,000회의 방향별 단측 p와 두 값의 최대를 기록한다. T·X와 함께 등록된 DEV family의 Holm 보정은 전체 주 비교를 확보한 뒤 적용한다. 여기의 p값은 보정 전 수치다. Fold별 점추정, 주 비교의 plate 하나씩 제외한 민감도, 절대 대비 norm과 cosine, 별도 pool-centered RMSE·cosine·Top-1·Top-5 및 tie 기대값을 보고한다. 운영 기준은 공동 상대 RMSE 0.5% 및 R² 0.5%p 개선, 3 fold 중 최소 2개에서 같은 방향, 약물 특이 대조의 지원이다.

예측 단계는 기존 predictor-only 묶음과 발현값 없는 실측 dose metadata만 읽는다. 전체 9개 조각을 공통 해시와 함께 동결한 뒤 별도 프로세스가 score 처리 대비를 읽는다. 이 A 실행은 구조·손실의 초기 비교이며 동일 약물 donor가 있는 B의 별도 입출력 계약을 평가하지 않는다.
