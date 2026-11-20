# R8-E1 / E2-A source 결정 (HCC를 읽기 전에 작성)

**작성일:** 2026-11-19 · 사양 F1 `1b59b76`(+YAML 문법 `6a66d0b`) · 평가 run `e1ev_outer{0,1,2}_v1` (jobs 26881–26883), 요약 job 27229 · 표: `manifests/r8_transfer_v2/e1_source_decisions.json`, `e1_contrasts.tsv`, `environment_transfer.tsv`, `information_ablation.tsv`.
source-train 91 family(일반 episode)·41 family(matched E1/E2, 360 ordered pair × 3 seed)의 **이미 노출된 자료 후속 분석**이다. K = 16, family cluster bootstrap 95% CI, `[k/3]` = 앞 방법이 좋았던 outer fold 수.

## 주 판정 (Holm, 두 검정)

| 검정 | 결과 | 판정 |
|---|---|---|
| H2: M2(다른 X 반복 plate support, known_separate)에서 J3가 C1과 C1J1을 **모두** 넘는가 (IUT) | J3 − C1 −0.00169 [−0.00270, −0.00089] [3/3]; J3 − C1J1 −0.00072 [−0.00170, +0.00013] [2/3]; p_IUT = 0.101 | **채택 안 함** (C1J1 대비 CI가 0 포함). 더 강한 기준선은 C1J1(0.69192 vs C1 0.69289); 상대 차이 −0.10%, 실용 기준 0.5% 미달 |
| H1: 공통 보정의 이득이 환경 shift에서 약물별 보정보다 크게 줄어드는가 | (Δc − Δr) +0.0250 [+0.0192, +0.0325] [3/3에서 양수], compound-plate cluster CI [+0.0198, +0.0286]; C1 − C0 (M2) −0.00188 [−0.00308, −0.00089] | **지지됨** (bootstrap p < 0.001, Holm 통과) |

세부: C0 − anchor_matched는 M1 −0.0285 [−0.0355, −0.0226], M2 −0.0023 [−0.0041, −0.0006] (compound-plate cluster CI [−0.0026, +0.0010]); C1 − C0는 M1 −0.0030, M2 −0.0019.

## 기획서 §5.2 결정 규칙의 적용 (문자 그대로)

- J3 신규 채택 근거 없음 (주 전이 조건 M2에서 C1J1 대비 미확립).
- C1J1과 J3의 차이가 M2에서 확립되지 않았다 → 단순 환경 보정(C1J1)을 우선하고 joint 모델 확대는 보류.
- 계약별로 재선택한 기준선 대비 J3가 두 기준선을 모두 넘은 곳은 같은 plate(M1 planned: −0.00081 / −0.00113, centred도 CI < 0)뿐이고 크기는 0.5% 미만 → 같은 환경 관측값 예측용으로만 한정. unknown/별도 환경에서는 C1 유지. (M2의 `fixed_gen_constants`·`missing_reselected` 행에서도 J3가 앞섰지만(−0.0049 / −0.0035 대 C1), 그 C1/C1J1은 M2용으로 재선택되지 않은 일반 episode 상수를 쓴 행이라 채택 근거로 쓰지 않는다.)
- 일반 episode(G planned)에서 J3 − C1 −0.00023 [−0.00080, +0.00037] → 미확립.

## E2-A (M2 known_separate, centred RMSE)

f0 추가(kn_all − kn_chem+ctrl) −0.00250 [−0.00380, −0.00153] [3/3]; chem 추가(kn_all − kn_f0+ctrl) −0.00018 [−0.00040, +0.00002] (미확립); f0 층화 순열 − kn_all +0.00366 [+0.00229, +0.00553] → 사전 라벨 **same-drug source-response-assisted transfer** 충족, f0 대응 사용 확인. (G planned에서는 chem 추가 −0.00041 [−0.00068, −0.00014]로 화학 추가 기여가 확립되었다 — 보조 view.)

## 다음 단계 (사전 규칙)

- E2-B 진입 조건 (b) 충족 → `configs/r8_transfer_v2/e2b_spec.yaml` (F3) 커밋 후 채널 민감도와 source-reference-unavailable 검증 실행.
- HCC는 이 결정과 final_inner 상수를 동결(F2, `e1_source_freeze.json`)한 뒤 한 번만 계산한다. HCC 값으로 어떤 후보도 바꾸지 않는다.
