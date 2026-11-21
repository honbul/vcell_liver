# D1·D2 개발 비교 명세

GSE92742 HEPG2 6 h의 승인된 A 개발 집단 1,862 parent와 6,773 처리 웰에 기존 outer 3-fold와 nested inner 분할을 적용한다. R18 보호 측정·별칭·donor를 제외한 원 개발 처리 명부를 그대로 쓴다. 각 41개 RNA plate에서 세 DMSO 역할의 최소 표본수에 맞춰 원 ID 순서로 선정했다. 각 역할에 162개씩 총 486개 물리 웰이다. 역할은 A→B→C로 순환한다. 입력 적합과 선택에 각 회전의 fit/select 대조군만 제공한다. [대조군 및 분할](data/preflight/d1/access/support.tsv) · [접근 영수증](data/preflight/d1/access/receipt.json)

D1은 같은 fit 대조군과 batch, seed, 후보 checkpoint, 학습 정답을 쓴다. 위치 anchor의 gain은 양쪽에 공통으로 독립 select 대비에서 골라 잠근다. 이 공통 anchor를 뺀 잔차로 학습하고, 선택 단계에서만 같은 처리 웰에서 빼는 DMSO를 변경한다. `D1_shared`는 fit DMSO, `D1_independent`는 select DMSO를 뺀 정답에서 고른다. Gain·checkpoint, 동일한 내부 독립 정답에서의 선택 후 손실, 잔차와 `(처리−fit DMSO)−(처리−select DMSO)` 사이 cosine을 기록한다.

D2는 독립 select 정답을 공통으로 쓴다. `D2_legacy`는 gain 1에서 최적 checkpoint를 먼저 고르고, 그 checkpoint에서 gain을 정한다. `D2_joint`는 checkpoint와 gain을 동시에 고른다. `D1_independent`와 `D2_joint`는 동일한 입력과 정책으로 같은 배열을 반환해야 한다. R17의 개별 기존 비교군 7종에 동일한 두 정책을 적용한다. FiLM/concat은 기존 학습률 두 개와 다섯 epoch를, XPert·TranSiGen·chemCPA는 기존 모델별 epoch 목록을 쓴다. ChemBERTa·PRnet은 고정 checkpoint 하나이므로 D2의 두 정책이 동일해야 한다. 출력 gain은 기존 `0, .125, .25, .5, .75, 1`에 고정하고 원 R17 공동 손실을 쓴다. 동률은 순차 정책에서는 원 checkpoint 순서와 낮은 gain으로, DEV 정책에서는 낮은 gain과 config ID 사전순으로 해소한다. 최종 적합은 선택된 checkpoint당 같은 seed로 한 번만 수행한다.

실제 예측이 시작되면 7개 모델과 9개 역할 조각을 모아 예측과 선택 기록을 동결한다. 별도 프로세스가 SHA-256, 행·유전자·모델 배열을 대조한 뒤 정확한 score 역할의 DMSO와 그 역할의 held 처리 결과를 읽는다. 주 비교는 두 정책의 parent-macro RMSE와 pooled R²의 paired 차이, parent 10,000회 재표집 99% 구간이다. 각 정책의 공통 위치 anchor 대비 점수와 모델 간 순서도 함께 기록한다. 공유 정책이 큰 gain을 고르고 독립 채점에서도 악화할 때 D1 가설을 지지한다. D2의 내부 선택 차이와 외부 성능 차이는 각각 보고한다.
