# S·L 구조와 손실의 DEV B 비교

GSE92742의 검증된 X 물리 명부에서 HEPG2 6 h·실측 10 µM 493군과 978개 landmark를 사용한다. 기존 R15 canonical parent 3-fold와 5-way 내부 선택, 물리 DMSO의 fit·select·score 세 역할을 유지한다. HEPG2의 outer held 처리값과 score DMSO는 별도 채점에서만 읽는다. R18 보호 처리·별칭·정규화·donor를 제외한 기존 DEV 범위다.

같은 화합물의 A375·A549·HT29·MCF7 6 h·10 µM 측정이 B 입력이다. 각 세포주에서 물리 웰 반복의 fit DMSO 대비를 평균하고 네 세포주를 균등 가중한다. donor-only plate에서는 그 plate의 독립 fit DMSO를 사용한다. HEPG2 적합 내부와 held에서 donor의 입력 규칙은 같다. 대조군 `B_cell`의 gain과 `B_pos`의 gain을 독립 select에서 R17 규칙으로 정한다. R17과 같은 순서로 gain 1의 donor와 교차 적합 위치를 차감한 처리 효과에 커널·신경망 가지를 적합한다. 전체 결합의 공동 anchor는 `B_pos + gain × B_cell_raw`이며 `B_blend_anchor` 자체를 함께 채점한다.

S0–S3, L0–L3, fit-only 평균·잔차 평균·맥락 대조, S0 화학/맥락 대조의 구성과 후보 설정은 [S/L A 명세](SL_PROTOCOL.md)를 따른다. S0와 L0는 같은 refit `B_R17`이다. `B_R15`는 gain 1 donor를 둔 R17 방식의 원 커널 예측이다. B의 커널에는 R15의 고정 Tanimoto·sibling·mixed donor·MLP 격자를 그대로 쓰고, 설정 선택은 해당 fold의 독립 select로 제한한다. 신경망은 같은 R12 구성·seed 42·공동 선택 격자를 공유한다. `B_cell_parent_shuffle`은 동일한 held donor 벡터를 fold 내 parent 단위로 순열한 기술 대조이며 donor의 gain은 원 비교와 공유한다.

주 비교는 S2 및 L1의 전체 결합 대 `B_R17`이다. 동일 10 µM, 동일 6 h, 같은 물리 RNA plate·부모 fold·역할을 pool 단위로 쓴다. A와 같은 지표·99% 짝 parent 구간·parent swap p·plate 민감도·norm·검색 기술값 및 0.5% RMSE·0.5%p R²의 공동 운영 기준을 적용한다. 화학 특이성은 공통 anchor와 커널을 주고 다시 적합한 S0 화학 순열과 함께 본다. 이 B 코호트는 이미 X와 T 개발에 쓰였으므로 별도 확인 코호트로 해석하지 않는다.

예측 조각 9개를 등록된 X metadata, 표준 설정, 코드 해시와 함께 동결한다. 채점 프로세스는 동결 해시와 모든 배열의 row·gene·arm 순서를 검증한 뒤 등록된 HEPG2 score 명부의 실제 대비를 읽는다.
