# DILI 확증 명세

- Curation unit은 FDA 기록 ID다. 구조가 해소되고 이진 concern이 확정된 행을 등록한다. 양성은 Most 또는 Less, 음성은 No다. Connectivity가 같은 항목은 학습과 채점 사이에 나누지 않는다. Cmax와 1일 투여량은 입력에서 제외한다.
- 역사 모델 선택은 v1 release 학습 행 702개로 제한한다. Murcko scaffold 층화 그룹 분할 안에서 바깥 5-fold와 안쪽 4-fold를 사용한다. 세 CLAMP 계열의 순위용과 소수 선별용 후보는 각각 바깥 OOF AUROC와 AP로 정한다. 배포 후보는 0.5×(AUROC + AP) − 0.10×Brier − 0.05×ECE로 정한다. 보정기는 같은 바깥 OOF에서 Brier + 0.2×ECE를 최소화한다.
- Prior-EM 후보는 역사 보정 참조 분포에서 오프셋을 한 번 적합한다. 질의 코호트에 맞추어 prior를 재추정하면 단일·배치 일치 요건을 충족할 수 없으므로 추론에는 동일 오프셋을 적용한다.
- 스캐폴드 확인 비교에서는 DILI v2 동결 모델과 같은 입력·학습 행을 모든 비교군에 적용한다. 등록 Murcko 및 Butina의 각 학습 fold에서 holdout 증분 155개를 전역 제외한다. CLAMP 설정과 보정기 계열은 역사 동결값을 쓰고, fold별 계수는 해당 학습 fold 안에서 적합한다. FDA 증분 점수는 역사 모델에서 그대로 산출한다.
- DILI v1 구조 + ChemBERTa LightGBM은 원 구현을 동일한 학습 행으로 재적합한다. DILI v2 동결 예측과 동결 checkpoint의 Chemprop 계열 D-MPNN은 기록 해시와 fit/query 범위를 검증한다. D-MPNN은 멤버 logit 평균에 sigmoid를 적용하고 0.5로 분류한다.
- 모든 예측 영수증과 행을 대조한 뒤 라벨을 연다. 각 모델의 pooled OOF와 비가중 fold 평균을 모두 낸다. 그룹은 InChIKey connectivity로 고정하고 10,000개 짝지은 군집 bootstrap에서 99% 구간을 산출한다. ECE는 10개 균등폭 구간을 사용한다.
- 연구 차원에서 이전 성적에 이미 노출된 FDA 버전 증분이다. 이 실행의 모델·보정기 선택은 역사 학습 자료로 제한했다. 이 실험에서 통과한 코드와 모델 카드만 연구용 패키지로 묶는다.
