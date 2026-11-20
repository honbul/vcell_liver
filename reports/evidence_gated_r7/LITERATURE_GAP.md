# R7 선행연구와 이번 질문의 차이 (LITERATURE_GAP)

근거 문헌의 성능 수치는 이 자료에서 기대할 개선 폭으로 쓰지 않는다. 출처는 비공개 기획서의 근거 목록(S1–S15, G1–G9)이며, 여기에는 공개 서지 정보만 옮긴다.

| 선행 연구 | 이미 한 것 | R7이 새로 묻는 것 | 신규성이 성립하지 않는 경우 |
|---|---|---|---|
| XPert (Guo 등, Nature Machine Intelligence 8:96–112, 2026) | 약물 구조·약물–표적 등 이질 그래프 embedding·기저 발현을 dual-branch transformer로 결합해 약물 반응 예측 | 주석의 **종류·결측·작용 방향 미상**을 입력에 표시하고, 결측을 학습 중 흉내 내며, 근거와 세포 상태에 따라 그래프 전달량을 조절하면 나아지는가 | 같은 split에서 XPert나 같은 정보를 단순 연결한 모델을 넘지 못하면 "활용 방식의 가치"는 주장하지 않음 |
| TxPert (Wenkel 등, Nature Biotechnology 2026) | 복수 KG와 graph attention으로 **유전적** 개입 반응 예측 | 화학적 개입, 불완전한 약물 주석 | 점수를 직접 비교하지 않음(과제가 다름) |
| Ahlmann-Eltze 등 (Nature Methods 22:1657–1661, 2025) | 단순 선형·평균 기준선이 강함 | 최소 기준선(train_cell_mean, ECFP+X0 Ridge)을 같은 표에 둔다 | 기준선을 못 넘으면 그대로 보고 |
| Systema (Nature Biotechnology 44:1050–1059, 2026) | 공통 이동과 개입 특이 신호를 나눠 평가 | ΔRMSE와 함께 centred Pearson, retrieval을 보고 | — |
| OmniPath (NAR 54:D652, 2026), ChEMBL | 관계 유형·출처·assay 메타데이터 | 출처 수를 독립 실험 수로 쓰지 않고, confidence를 작용 확률로 해석하지 않음 | — |

**이번 연구가 하지 않는 것**
- 입력 KG를 다시 GRN 정답으로 채점하지 않는다.
- attention이나 gate 값을 인과 edge로 해석하지 않는다.
- 데이터베이스 수를 늘리지 않는다(초기 두 자원으로 제한).
- 결과가 나빠도 새 손실(InfoNCE 등)을 붙이지 않는다.
