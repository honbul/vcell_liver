# 연구 R8 — 가장 가까운 선행연구와의 경계 (LITERATURE_GAP)

**작성일:** 2026-11-17 · 체계적 문헌고찰이 아니다. 이번 작업 중에 논문 본문을 새로 내려받아 읽지 않았고 외부 검색도 하지 않았다. 아래는 기획 근거 문서에 정리된 서지와 요약, 그리고 앞선 연구(R2)의 기록에 근거한다. 선행연구가 보고한 이득은 그 논문 조건의 결과이며 이 연구의 결과를 뒷받침하지 않는다.

## 1. 한눈에

| 선행연구 | 그 연구가 한 것 | 이 연구가 가져온 것 | 이 연구가 다른 점 | 주장하지 않는 것 |
|---|---|---|---|---|
| Bertinetto et al., *Meta-learning with differentiable closed-form solvers*, ICLR 2019 (arXiv:1805.08136) | Ridge 같은 닫힌 해법을 적응 단계로 두고 그 해법을 통과해 표현을 학습 | 학습된 embedding 위의 K×K 회귀를 Cholesky로 푸는 구조(C2/B 계열) | support 관측이 서로 독립이 아니라고 보고 `K + λΩ`로 푼다(Ω는 복제 수·plate 공유 메타데이터에서만 만든다). 공통 보정과 잔차 회귀를 분리해 같은 앵커 위에서 비교 | 새로운 Ridge 해법의 최초 제안 |
| Garnelo et al., *Conditional Neural Processes*, ICML 2018 | support 집합을 조건으로 새 입력의 출력을 예측 | 비교군 `generic CNP` | 앵커·분해가 없는 CNP를 같은 입력·같은 선택 규칙으로 다시 학습해 비교 | CNP 도입이 새 기여라는 주장(R2가 이미 사용) |
| Lehtinen et al., *Noise2Noise*, ICML 2018 | 독립·영평균 잡음 가정에서 잡음 섞인 정답만으로 복원을 학습 | "반복 측정을 서로의 검증으로 쓴다"는 생각(D1, D2) | 같은 plate·같은 대조군을 쓰는 측정은 독립이 아니므로, 서로 다른 RNA plate의 쌍만 쓰고 기준 평균도 교차 적합으로 만든다 | 독립 view 가정이 이 자료에서 성립한다는 주장 |
| Neufeld et al., *Inference after latent variable estimation…*, Biostatistics 2024 | 추정과 추론에 같은 자료를 두 번 쓰는 문제를 count 분할로 다룸 | 같은 well·같은 plate를 나눈 것을 새 실험으로 세지 않는다는 원칙(`support_cost.tsv`가 약물 그룹·well·plate를 따로 셈) | count 분할 자체는 쓰지 않는다(Level 3 자료) | count splitting을 구현했다는 주장 |
| Ahlmann-Eltze et al., Nat. Methods 2025 | 유전자 개입 예측에서 단순 기준선이 딥러닝과 같거나 나음을 보고 | 무변화·같은 약물 평균·축소 평균 보정을 먼저 세우고 그 위의 이득만 본다는 원칙 | 화합물 개입, 세포주 간 소량 적응 | "딥러닝은 원래 안 된다"는 일반 명제 |
| Systema (Viñas Torné et al., Nat. Biotechnol.) | 공통 이동과 개입 특이 신호를 나눠 평가 | 양쪽 평균을 각각 뺀 centred 지표를 주지표 옆에 둠 | 지표 정의는 이 연구의 것 | Systema 지표를 재현했다는 주장 |
| Zheng et al., *TranSiGen*, Nat. Commun. 2024 | 기저·처리 후 발현을 VAE로 학습해 약물 유도 전사체를 예측 | 공개 구조를 XC pool 없이 다시 학습해 as-is / offset / full FT 비교군으로 사용 | 같은 라벨 예산(K개, target 검증 0개) | 공개 가중치 성능의 재현 |
| FiLM류 조건화(feature-wise modulation) | 조건 벡터로 중간 표현을 차원별로 곱·더함 | 전역 대조군 상태를 약물 표현에 곱하는 경로(B01, B11) | support 중심화·LayerNorm을 지나도 지워지지 않는다는 것을 합성 자료로 먼저 확인(`fixtures.json`) | 새로운 조건화 연산의 제안 |

## 2. 구별점 후보와 한계

1. **상관된 support의 관측 모형**을 잔차 회귀 해법 안에 넣고, 그 효과를 "공통 보정만 한 것(C0)"과 "학습 없는 kernel(C1)"에 대해 같은 앵커에서 분리해 본다.
2. **평균 차감으로 지워지는 전역 상태**를 되살리는 경로를 별도 조건으로 두고, 같은 크기의 무의미한 벡터(G_dummy)와 비교한다.
3. **같은 plate / 다른 plate support** 비교로, 보정이 세포주를 배운 것인지 plate 환경을 배운 것인지 나눠 본다.

각 요소는 이미 있는 기법이다. 조합의 신규성은 확정하지 않는다. 그리고 효과가 있었는지는 `RESULTS.md`의 수치로만 판단한다.

## 3. 이번에 하지 않은 것

- 논문 본문·보충자료에서 수치를 새로 추출하지 않았다. 외부 웹 요청을 하지 않았다.
- 다른 공개 반응 모델(XPert, State 등)은 이 연구의 비교군이 아니다(다른 연구가 맡는다).
