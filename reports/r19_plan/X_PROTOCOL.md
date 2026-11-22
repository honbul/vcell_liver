# X0–X2. 세포주와 화합물을 함께 제외한 DEV A 비교

GSE92742의 실측 10 µM·6 h에서 HEPG2·A375·A549·HT29·MCF7에 공통인 493 parent를 사용한다. R18 보호 처리·별칭·score 대조군과 B donor를 기존 등록 명부대로 제외한다. 한 RNA plate에 여러 세포주가 기록된 경우 해당 plate를 제외한다. 모든 처리·대조군은 물리 RNA 좌표와 Level 3의 표본 ID를 대조한다. [등록된 지원](data/x_l_support/x-basal/receipt.json)

외분할에서 한 세포주와 기존 CRC32 parent fold 하나를 동시에 제외한다. HEPG2를 포함한 5개 세포주를 각각 한 번씩 제외한다. 4개 외부 적합 세포주 중 고정 목록에서 평가 세포주 다음에 오는 하나를 내부 선택에 쓴다. 기존 nested CRC32 규칙의 select parent도 모든 내부 적합 세포주에서 제외한다. 독립적인 물리 DMSO fit·select·score를 RNA plate별 CRC32로 배정하고 세 번 회전한다. X0의 평가 세포주에는 허용된 fit DMSO 입력이 있다. 모든 모델의 정답은 score DMSO를 뺀 처리 결과다.

세 모델은 같은 parent 가중 MSE FiLM, 같은 구조 표현과 폭 16의 맥락 입력을 사용한다. 새로운 세포주에는 같은 출발점으로 fit-only 공통 잔차 평균을 사용한다. 조회 화합물의 donor 처리 효과는 주입하지 않는다. X0는 fit DMSO로만 학습한 PCA, X1은 fit 세포주 ID와 새 세포주에 동일 가중 평균, X2는 DepMap 25Q3의 단일 non-stranded 기저 RNA profile과 fit 세포의 표준화·PCA를 쓴다. 조회 세포의 기저 RNA는 학습된 변환으로만 매핑한다. 공통 학습률·epoch·gain은 R17/DEV의 두 학습률, 다섯 checkpoint와 여섯 gain으로 독립 select 손실에서 공동 선택한다. 음성 대조는 X2의 화학 입력 제거, 맥락 제거, fit parent 화학 순열, fit 세포주 기저 profile 순열이다. 순열은 seed 1501로 고정한다.

주 비교는 X2−X1의 전체 지원 parent-macro RMSE와 pooled R²다. HEPG2와 세포주별 결과, 같은 parent·역할의 세포주 공통 평균을 뺀 상호작용 오차도 보고한다. Parent 단위 짝 bootstrap 10,000회, 양측 99%를 쓴다. 주 비교의 부모별 양측 후보-기준 쌍 교환에서 두 방향별 단측 p의 최대를 남기고 DEV family 보정은 등록된 축의 수치가 모이면 적용한다. 사전 운영 문턱은 상대 RMSE +0.5%, R² +0.5%p, 세 fold 중 두 개 같은 방향 및 음성 대조의 지원이다. 이 실험은 새 맥락 일반화의 A 비교다. X3의 B donor 조합과 CNV의 추가 기여는 별도 명세를 사용한다.
