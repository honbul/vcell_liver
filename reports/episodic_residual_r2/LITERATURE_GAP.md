# 연구 R2 — 가장 가까운 선행연구와의 경계 (LITERATURE_GAP)

**작성일:** 2026-11-17 · 이 문서는 체계적 문헌고찰이 아니다. 방법 설계와 가장 가까운 공개 연구 몇 편과 겹치는 부분·다른 부분을 적는다.
이번 작업 중에 논문 본문을 새로 내려받아 읽지는 않았다. "확인 범위" 열은 무엇에 근거해 적었는지를 구분한다. 선행연구가 보고한 이득은 그 논문 조건의 결과이며 이 연구의 결과를 뒷받침하지 않는다.

## 1. 한눈에

| 선행연구 | 그 연구가 한 것 | 이 연구가 가져온 것 | 이 연구가 다른 점 | 주장하지 않는 것 | 확인 범위 |
|---|---|---|---|---|---|
| Garnelo et al., *Conditional Neural Processes*, ICML 2018 (https://proceedings.mlr.press/v80/garnelo18a.html) | 관측 예시 집합을 평균 집계해 조건으로 넣고 새 입력의 출력을 예측하도록 여러 과제에서 학습 | "support 집합을 입력으로 받는 모델을 source 과제들에서 학습한다"는 틀. 비교군 `generic_cnp`가 이 구조다 | 출력이 978차원 RNA 변화이고, 과제 단위가 세포주(family)이며, f0·공통 offset·약물별 잔차를 구조적으로 나눈 변형과 나누지 않은 CNP를 같은 입력으로 비교 | CNP가 이 자료에 적합하다는 일반 주장 | 널리 알려진 공개 논문의 방법 개요. 본문 수치는 인용하지 않음 |
| Bertinetto et al., *Meta-learning with differentiable closed-form solvers*, ICLR 2019 (https://arxiv.org/abs/1805.08136) | Ridge 같은 닫힌 해법을 적응 단계로 두고, 그 해법을 통과해 표현을 학습 | 제안법의 핵심 계산(embedding 위의 K×K Ridge, λ 학습, Cholesky solve) | 분류가 아니라 다출력 회귀, 정답에서 같은 약물 평균과 support 평균 잔차를 먼저 빼고 중심화한 잔차만 Ridge에 넣음, target 검증 라벨 0개 | 새로운 Ridge 알고리즘이나 미분 가능한 해법의 최초 제안 | 방법 개요. 본문 수치는 인용하지 않음 |
| metaDRP (2025; https://pubmed.ncbi.nlm.nih.gov/41245657/) | 약물–조직 과제로 few-shot 약물 **민감도** 예측 | 약물 반응 분야에서 few-shot/meta-learning 자체는 이미 있다는 사실 | 출력이 민감도 scalar가 아니라 978개 유전자의 RNA 변화. 점수를 직접 비교하지 않음 | few-shot 약물 반응 예측의 최초 시도 | 기획서의 근거 문서(S08)에 적힌 요약만. **본문을 직접 읽지 않았다** |
| Zheng et al., *TranSiGen*, Nat. Commun. 2024 (https://www.nature.com/articles/s41467-024-49620-3) | 기저·처리 후 발현과 그 관계를 VAE로 학습해 약물 유도 전사체를 예측 | 공개 구조를 그대로 가져와 비교군(그대로 / +offset / full FT)으로 사용. 가중치는 연구 B가 HCC를 뺀 source로 다시 학습한 것 | 같은 라벨 예산(K개, target 검증 0개)에서 재실행. 설정(학습률·BN·update 수)은 source 가상 target으로 선택 | 공개 가중치 성능의 재현, 저자 논문 수치와의 비교 | 연구 B의 코드·기록과 공개 저장소 구조 |
| Ahlmann-Eltze et al., Nat. Methods 2025 (https://www.nature.com/articles/s41592-025-02772-6) | 유전자 개입 예측에서 단순 선형 기준선이 딥러닝 모델과 같거나 나음을 보고 | 무변화·같은 약물 평균·mean-offset·Ridge를 먼저 세우고 그 위의 이득만 본다는 원칙 | 화합물 개입, 세포주 간 소량 적응 과제 | "딥러닝은 원래 안 된다"는 일반 명제 | 기획서 근거 문서(S15) 요약과 공개 초록 수준 |
| Systema, Nat. Biotechnol. 2025 (https://www.nature.com/articles/s41587-025-02777-8) | 체계적 변이(공통 이동)와 개입 특이 신호를 구분해 평가 | 공통 offset에 영향받지 않는 지표(양쪽 평균을 각각 뺀 centred RMSE/Pearson, centred retrieval)를 주지표 옆에 둠 | 지표 정의는 이 연구의 것이며 Systema의 지표를 재현한 것이 아님 | Systema 지표로 평가했다는 주장 | 기획서 근거 문서(S16) 요약 |
| *Deep learning perturbation models can outperform baselines on calibrated metrics*, Nat. Biotechnol. 2026 (https://www.nature.com/articles/s41587-026-03307-w) | 지표의 민감도를 점검하면 일부 모델의 개입별 이득이 드러난다고 보고 | 지표를 결과에 맞춰 바꾸지 않고, source에서 양성·음성 대조(반복 측정 상관, 잔차 섞기)로 지표가 신호를 볼 수 있는지 확인 | — | 해당 논문의 보정 지표를 구현했다는 주장 | 기획서 근거 문서(S17) 요약만. **본문을 직접 읽지 않았다** |

## 2. 이 연구의 구별점 후보와 그 한계

이 연구가 시험한 조합은 다음 네 가지를 동시에 고정한 것이다.

1. 기본 예측을 신경망이 아니라 **source의 같은 약물 평균**(가상 target의 family를 뺀)으로 둔다.
2. support 잔차를 **공통 offset과 약물별 중심화 잔차로 나누고**, 학습된 embedding 위의 Ridge는 중심화 잔차에만 쓴다.
3. target에서 쓰는 처리 후 라벨은 **K개가 전부**이고, λ·학습률·중단·모델 선택은 모두 source episode로 정한다.
4. mean-offset, 약물 구조 잔차 Ridge/KRR, 같은 입력의 일반 CNP, 공개 구조의 세 가지 적응을 **같은 support·같은 query**에서 비교한다.

각 요소는 위 선행연구에 이미 있다. 조합의 신규성은 확정하지 않는다. 그리고 이 조합이 단순 기준선을 넘었는지는 `RESULTS.md`의 수치로만 판단한다 — 넘지 못한 부분은 그대로 적었다.

## 3. 이번에 하지 않은 것

- 위 논문들의 본문·보충자료에서 수치를 새로 추출하지 않았다.
- metaDRP, GEARS류, CellFlow 등 다른 공개 모델을 재현하지 않았다(이 과제의 필수 비교군이 아니다).
- 화합물 표현을 ECFP4에서 사전학습 화학 모델로 바꾸는 비교는 하지 않았다(기획서가 첫 실험에서 제외).
