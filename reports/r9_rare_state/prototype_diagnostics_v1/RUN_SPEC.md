# R9-E1/E2 prototype_diagnostics_v1 — 실행 명세 (실행 전 고정)

근거: private plan 2026-11-19 §6.3. 기계 판독 명세: `configs/r9/prototype_diagnostics_v1.json` (이 문서와 같은 commit이 freeze).

- **지위**: 이미 반복 개발한 source(GSE149614 바깥 LOPO, 모든 donor 노출)의 **후속 원인 진단**. 독립 검증 아님. A4/GSE151530 결과는 읽지 않는다.
- **과제**: sparse B/Endothelial/Fibroblast N64/D4 × 시험 donor 10 × seed 0/1/2 = 90 task. v1 target 제외 fold 표현(`<test>__excl_<target>`)과 `select(obs, cond, test, seed)`의 학습 세포를 그대로 사용(v1 run JSON의 design·n_train과 대조).
- **E1 arm**(모두 L_donor,class, Adam 300 step, lr 0.02): P1_current(=v1 P1_donor 재적합), P3_current(=v1 A10 재적합), P1_fixedvar, P3_fixedvar, P3_pooledvar(각 성분이 P1_fixedvar의 class별 고정 분산을 공유). 같은 K의 arm은 초기 membership·평균·mixing weight 공유. frozen arm은 log-variance를 optimizer에서 제외.
- **E2 arm**(현재 분산 규칙): A11_refit(K=3C, v1 fallback), supported_stop(첫 admissible split 부재 시점의 K*, source fold 안에서만 결정), matched_kstar(supported_stop과 같은 class별 성분 수의 class별 k-means). 후보별 support·held-out gain·frac_pos·rejection_reason 기록. K*=K인 경우도 적합하고 `kstar_eq_K`로 표시(중복 fit 생략 대신 동일성 확인).
- **기준선**: 같은 90 key의 v1 저장 예측 LR_class, LR_donor, MLP_donor(강한 기준선)와 P1_donor, A10, A11. 
- **지표**: 주 = 시험 donor의 target AUPRC(양성 <10이면 NA). 보조 = target recall/false positive, NLL, 분산 floor 비율, log-variance 변화, 성분 ESS, 최대 donor 점유율, 학습 CE vs held-out CE, K*, fallback 시점·거절 사유.
- **집계**: seed를 (target, donor) 안에서 평균 → donor별 평가 가능한 target을 동일 가중 평균한 donor 벡터 → donor paired bootstrap 10,000회(seed 0). target별 대비는 보조.
- **주 대비**: E1 (a) P3_pooledvar−P1_fixedvar(K 효과, class 분산 고정), (b) P3_current−P3_fixedvar(분산 최적화 효과), (c) P3_fixedvar−P3_pooledvar(성분별 초기 분산 효과). E2 (d) supported_stop−A11_refit, (e) matched_kstar−A11_refit, (f) supported_stop−matched_kstar. P3_fixedvar−P1_fixedvar는 순수 K 효과가 아니므로 보조.
- **다중성**: family(3개)별 Bonferroni 동시 98.33% CI로 '확립' 판정, 95% CI 병기.
- **실용 크기**: AUPRC 0.01(plan §2.3 제안값, 결과 보기 전 고정). 범주: 동시 CI가 0 제외 & |평균|≥0.01 → established_practical; 동시 CI가 0 제외 & |평균|<0.01 → established_below_practical; 90% CI가 [−0.01, +0.01] 안 → equivalent_within_margin(사전 margin); 그 외 → not_established.
- **조건부 E2 분산 재적합**: (b)의 동시 CI 상한 <0 그리고 평균 ≤ −0.01이면 E2 세 arm을 frozen 성분 분산으로, (c)가 같은 조건이면 pooled class 분산으로 저장된 membership에서 재적합. 아니면 미실행 기록.
- **해석 규칙**: plan §6.3.2/6.3.3 그대로(설정 파일 `decision_rules_from_plan`). P1/A10/A11보다 좋아지는 것은 원인 분석 결과일 뿐 LR/MLP를 이겼다는 뜻이 아니며, 새 arm 채택은 여기서 결정하지 않는다.
- **비용 규칙**: 1 (donor, seed) × 3 target pilot으로 CPU 시간·메모리 측정 → 90 task 추정이 cap(동시 14 job, CPU) 아래 ~24 h 안이면 전체 실행.
