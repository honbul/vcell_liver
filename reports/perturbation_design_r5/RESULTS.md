# 연구 R5 — 결과 (RESULTS)

**작성일:** 2026-11-17 · 브랜치 `research/noise-aware-perturbation-design-r5` · 동결 commit `a3c8146` · HCC 평가 job 25387 (`hcc_eval_v1`, 한 번)

모든 수치는 CMap LINCS 2020 Level 3, 10 µM·24 h, 978개 landmark 유전자, (세포주, 약물 그룹) 단위다. **이미 공개된 측정을 봉인했다가 순차 공개한 retrospective 실험 설계 평가**이며 새 wet-lab 결과가 아니다. 같은 약물의 source 반응을 쓰므로 약물 zero-shot도 아니다. HCC 자료는 연구 B·R2에서 이미 채점된 공개 벤치마크다(`EXPOSURE.md`).

## 1. 결론 먼저

- **질문:** 새 세포주에서 처리 well K개만 측정할 수 있을 때, 어떤 약물을 고르면 **고정된 같은 예측기**가 나머지(봉인 query) 약물의 평균 ΔRNA를 더 잘 맞히는가?
- **판정(프로토콜 `decision.adopt_proposed`, 문자 그대로):** 제안 정책(noise-aware outcome-adaptive mixture, `proposed_nam`)은 meta-validation에서 random 대비 −0.0022 [−0.0052, +0.0002], fixed_vr 대비 −0.0006 [−0.0039, +0.0020]이라 **채택 조건(세 비교 모두 상한 < 0)을 충족하지 못했다. 채택하지 않는다.** 수정 1회(ν 폭 보정)는 dev에서 개선이 없어 미채택, 수정 2는 dev 진단에 근거가 없어 실행하지 않았다.
- **mixture·noise-aware 부분의 추가 기여는 보이지 않았다:** proposed − fixed_vr는 meta-val −0.0006 [−0.0039, +0.0020], HCC3 −0.0001 [−0.0011, +0.0008]. 프로토콜상 “더 단순한 고정 설계가 채택 변형”에 해당한다. noise-aware 주장은 하지 않는다(축소 실험은 실행 조건 불충족으로 미실행).
- **관측된 방향:** 고정 전역 분산 감소(`fixed_vr`)는 random보다 source(−0.0017 [−0.0026, −0.0008], 7/8 계열)와 HCC3(−0.0033 [−0.0051, −0.0017])에서 모두 낮았다. 다만 diversity 대비 source 상한이 +0.0001이라 이것도 프로토콜의 세 조건을 모두 넘지는 못한다. 크기는 random AUC의 약 0.3–0.5%다.
- **분명한 음성 결과:** 불확실성이 큰 약물을 고르는 정책(latent_variance +0.0098, total_variance +0.0222)과 MORPH식 오차 예측 선택(learned_loss +0.0286)은 source·HCC 모두에서 random보다 **나빴다**(구간이 0을 포함하지 않음).

## 2. 실행한 것

| 단계 | 내용 | job |
|---|---|---|
| R5-0 | 프로토콜 commit(값을 읽기 전) → store 생성(대표 well, plate DMSO 분산, 기저) | 25334·25335(경로/구문 오류로 값 읽기 전 실패), 25336(store 생성 후 manifest 기록 실패), 25356 |
| 고정 점검 | oracle 한 행 공개·재공개 금지·query 공개 금지, GP 수치, 합성 잡음 fixture | smoke(srun) |
| 개발 | D3 잡음 모형·게이트, readout 선택+D1, GP 초모수, 온도, D2, learned-loss 학습 | 25367 |
| E0–E2, E4 | dev 5정책 → 추가 3정책 | 25368, 25370 |
| meta-val | 8정책 × 8계열 × seed 10, 반복-분리 stress | 25371, 25372 |
| 수정 1 | 보정 진단 → ν 폭 γ 적합 → learned-loss 재학습 → dev/meta-val 재평가 | 25376, 25378, 25380(25379 취소), 25381, 25385 |
| 수정 2 검토 | batch 중복·plate 집중 진단(근거 없음 → 미실행) | 25384 |
| 동결·HCC | freeze → mock 배선 점검(source 3계열) → HCC 한 번 | 25386, 25387 |
| 튜토리얼 | 노트북 실행 | 25389 |

전체 목록은 `manifests/perturbation_design_r5/run_ledger.tsv`. 모든 작업은 CPU였고 GPU는 쓰지 않았다.

## 3. 라벨 비용 (cost ledger)

- 한 번의 공개 = 미리 정한 **처리 well 1개**(대표 행; HCC 행은 모두 단일 well). 같은 약물의 다른 반복·alias는 돌려주지 않는다. 처리 well 수 K=4/8/16/32이며 **target 검증 라벨은 0개**다. readout·GP·온도·learned-loss 설정은 모두 source dev 계열에서 정했다.
- 대조군(DMSO) well은 plate마다 한 번 세는 고정 overhead로 기록했다. HCC K=32에서 HUH7 3 plate·39 well, JHH5 정책에 따라 5.0–6.0 plate·64–78 well, JHH7 1 plate·13 well (`runs/hcc_eval_v1/cost.tsv`).
- source 대표 행 410개는 2개 이상 well의 평균이라 source episode의 처리 well 수 평균이 32.2–32.8이다.
- 선택 시간(episode당, K=32까지): proposed 13.5초(meta-val)·HCC JHH5 16.3초, pdbal 2.2초, 나머지 0.1초 미만.

## 4. 개발 단계의 진단 (source dev 16계열)

| 진단 | 관측 |
|---|---|
| D1 random 학습곡선 | main readout ΔRMSE K=4/8/16/32: dev 0.607/0.566/0.544/0.530, meta-val 0.602/0.557/0.530/0.516. 라벨이 늘면 오차가 준다. |
| D3 측정 잡음 예측 | 반복쌍 47,193(학습)/3,771(검증). meta-val 6계열 Spearman 0.243 [0.085, 0.410] → 게이트 통과. 가장 큰 계수는 plate DMSO 분산(log ctl_sd, 1.71). |
| D2 후보 하나 추가의 이득 (K=4) | 고정 전역 감소 점수와 +0.23, 잠재 분산 −0.22, 예측 총분산 −0.37, 예측 ν −0.41(16계열 평균 Spearman). K=16에서 모두 약해짐. |
| 보정 | 다음 공개 well의 표준화 오차비 전체 0.87–1.04, ν 5분위별 0.65→1.52. query 잠재평균 비율 1.15→1.36. mixture weight 최댓값 0.20→0.33(6개 가설). |
| 사전 격자 경계 | sb2=0.03, 온도 T=1/32가 각 격자의 최솟값으로 선택됨. 격자는 넓히지 않았다. |

## 5. meta-validation (8계열 × seed 10; 동결 설정)

main readout(f0 + offset + 잔차 KRR, α=0.5, λ=3), 계열 동등 평균. 대조는 계열 bootstrap 95% (`runs/mval_v1/`).

| 정책 | AUC | K=16 | random 대비 AUC [95%] | 낮은 계열 |
|---|---:|---:|---|---:|
| proposed_nam | 0.5349 | 0.5292 | −0.0022 [−0.0052, +0.0002] | 6/8 |
| fixed_vr | 0.5354 | 0.5288 | −0.0017 [−0.0026, −0.0008] | 7/8 |
| random | 0.5371 | 0.5303 | — | — |
| pdbal_rna | 0.5375 | 0.5319 | +0.0004 [−0.0039, +0.0041] | 2/8 |
| diversity | 0.5377 | 0.5319 | +0.0006 [−0.0015, +0.0028] | 4/8 |
| latent_variance | 0.5469 | 0.5415 | +0.0098 [+0.0069, +0.0127] | 0/8 |
| total_variance | 0.5593 | 0.5572 | +0.0222 [+0.0128, +0.0334] | 0/8 |
| learned_loss | 0.5657 | 0.5622 | +0.0286 [+0.0207, +0.0357] | 0/8 |

- 다른 기준: proposed − diversity −0.0028 [−0.0059, −0.0002]; proposed − fixed_vr −0.0006 [−0.0039, +0.0020]; fixed_vr − diversity −0.0023 [−0.0052, +0.0001].
- **두 번째 readout(mean-offset만)으로 반복:** proposed − random −0.0010 [−0.0040, +0.0018], fixed_vr − random −0.0012 [−0.0023, −0.0002]. 순서는 main readout과 같다.
- **반복-분리 stress(7계열; 공개는 첫 반복 plate, query 정답은 다른 반복 plate):** proposed − random −0.0031 [−0.0063, −0.0004], fixed_vr − random −0.0002 [−0.0034, +0.0022]. 주 지표가 아니며 판정에 쓰지 않았다.
- 수정 1(ν 폭 γ=2)을 적용한 meta-val(정보용): proposed − random −0.0015 [−0.0046, +0.0014].

## 6. HCC 동결 후 평가 (한 번, `runs/hcc_eval_v1/`)

seed 10개(분할·초기 4개), query 수(seed당) HUH7 21 · JHH5 196 · JHH7 33. 구간은 **계열마다 약물 그룹 재표집**(seed는 고정)이며 HCC 모집단에 대한 구간이 아니다.

| 정책 | HUH7 | JHH5 | JHH7 | HCC3 평균 AUC | HCC3 − random [95%] |
|---|---:|---:|---:|---:|---|
| proposed_nam | 0.7833 | 0.5157 | 0.6012 | 0.6334 | −0.0035 [−0.0054, −0.0018] |
| fixed_vr | 0.7842 | 0.5177 | 0.5987 | 0.6335 | −0.0033 [−0.0051, −0.0017] |
| pdbal_rna | 0.7869 | 0.5193 | 0.6025 | 0.6362 | −0.0007 [−0.0030, +0.0016] |
| random | 0.7919 | 0.5183 | 0.6005 | 0.6369 | — |
| diversity | 0.7920 | 0.5184 | 0.6012 | 0.6372 | +0.0003 [−0.0010, +0.0016] |
| latent_variance | 0.7921 | 0.5343 | 0.6179 | 0.6481 | +0.0112 [+0.0079, +0.0145] |
| total_variance | 0.7965 | 0.5403 | 0.6211 | 0.6527 | +0.0158 [+0.0127, +0.0187] |
| learned_loss | 0.8076 | 0.5465 | 0.6251 | 0.6597 | +0.0228 [+0.0195, +0.0262] |

- proposed − fixed_vr: HUH7 −0.0010 [−0.0034, +0.0013], JHH5 −0.0020 [−0.0026, −0.0013], JHH7 **+0.0025** [+0.0008, +0.0040], HCC3 −0.0001 [−0.0011, +0.0008].
- 계열별 방향이 다르다: proposed − random은 HUH7 −0.0086, JHH5 −0.0025, JHH7 +0.0007 [−0.0022, +0.0035]. JHH7은 plate가 한 장이라 ctl_sd가 모든 후보에서 같아 noise-aware 부분이 작동할 여지가 없다.
- mean-offset readout HCC3 AUC: proposed 0.6407, fixed_vr 0.6412, random 0.6429, diversity 0.6426.
- HCC 결과는 source 판정을 바꾸지 않는다(프로토콜). source와 같은 방향이지만 제안 정책은 이미 채택되지 않았다.

## 7. 관측 / 가설 / 미확인

| 구분 | 내용 |
|---|---|
| 관측 | 잠재·총 분산 최대화와 learned-loss 선택은 source·HCC 모두 random보다 나쁘다. |
| 관측 | 고정 전역 분산 감소는 random보다 작게 낫고, proposed와 구분되지 않는다. |
| 관측 | plate DMSO 분산으로 단일 well 반복 차이를 일부 예측할 수 있다(meta-val Spearman 0.24). proposed는 예측 ν가 작은 well을 더 골랐다(dev 0.367 vs pool 0.410). |
| 가설 | 분산 최대화가 나쁜 이유는 잡음이 큰 well이나 f0가 약한 약물에 분산이 몰리기 때문일 수 있다(D2와 같은 방향이지만 원인 분리는 안 함). |
| 가설 | outcome 적응이 거의 작동하지 않은 것은 온도 T=1/32에서 weight가 균등에 가깝기 때문일 수 있다. |
| 미확인 | 새 wet-lab 순차 실험에서의 효과, 다른 dose·time·플랫폼, 다른 readout(TranSiGen 등) 조합, 반복 측정 선택. |

## 8. 주장하지 않는 것

새 실험에서 라벨을 절약한다는 주장, “random의 K32를 K16으로 달성” 같은 외삽(곡선상 근거 없음), noise-aware·outcome-adaptive 설계의 추가 기여, HCC 환자 반응, 다른 RNA 예측기에서의 일반화.
