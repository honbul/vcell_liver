# 연구 R2 — 단계별 결정 기록 (DECISIONS)

각 단계를 **실행하기 전에** 다섯 줄(관측·가설·예측·실험·결정 규칙)을 적고, 실행 뒤에 `결과`와 `결정`을 덧붙인다. 앞에 적은 줄은 고치지 않는다.
수치의 원자료는 `manifests/episodic_residual_r2/`에 있다. "가설"은 검증 전 설명이며 확정 원인이 아니다.

---

## 단계 R2-0 — source 전용 기준선·episode 구성·누출 fixture (2026-11-17, 실행 전 기록)

- **관측:** 연구 B(`d67bf68`)에서 세 HCC 계열 모두 mean-offset 보정이 full fine-tuning보다 시험 ΔRMSE가 낮았다. B의 적응은 k=16개 외에 target 검증 약물(14/130/22 그룹)의 라벨로 학습률·중단 시점을 골랐다. B의 source 분할(145/16)은 세포주 이름 단위였다.
- **가설:** (검증 전) 공통 offset을 뺀 뒤에도 약물별 잔차에 support로부터 예측 가능한 성분이 남아 있을 수 있다.
- **예측:** 가설이 맞다면 source 가상 target에서 ECFP 잔차 Ridge/KRR이 mean-offset보다 query ΔRMSE와 centred 지표에서 낮아야 한다. 맞지 않으면 선택되는 λ가 격자의 가장 큰 쪽으로 가고 점수가 mean-offset과 같아진다.
- **실험:** (1) 161개 source 세포주의 family를 metadata로 정하고 옛 145/16 분할의 family 중복을 확인·최소 수정한다. (2) B의 plate 행을 (세포주, 약물 그룹) 단위로 접는다(source만; HCC 값은 읽지 않는다). (3) episode 생성기와 f0(가상 target family 제외)의 누출 fixture를 합성 자료와 실제 store에서 통과시킨다. (4) 고정 episode에서 무변화·f0·mean-offset·ECFP Ridge·Tanimoto KRR·동일 입력 비학습 KRR을 채점한다. λ는 meta-train dev episode로만 고른다.
- **결정 규칙:** fixture가 하나라도 실패하면 학습으로 넘어가지 않는다. family 중복이 있으면 중복 family 전체를 meta-train으로 옮기는 최소 수정만 한다.
