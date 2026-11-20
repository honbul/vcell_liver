# 원 R7 보고서(`b962a84`)의 XPert 행에 대한 정정 메모

이 메모는 원 R7 보고서(`reports/evidence_gated_r7/RESULTS.md`)를 고치지 않고 덧붙이는 기록이다. 원문은 그대로 둔다.

## 무엇이 문제였나

원 R7의 XPert 비교군(공식 입력 `xpr_e100_s{42,43,44}`, 판정 뒤 추가한 A-2 XPert-ECFP `xpe_e100_s{42,43,44}`)은 upstream XPert 학습 스크립트의 자동 checkpoint 선택(EarlyStopping, validation `loss4 = 1 − unit 평균 ΔPearson`)을 썼다. 그 validation 행은 R4 validation 전체 11,731 unit이었고, **HCC validation 129 unit(HUH7 13, JHH5 99, JHH7 17)이 포함**됐다(`manifests/r7_xpert_followup_v2/selection_population_audit_part1.json`). 원 R7 계약에서 HCC validation으로 설정을 고르는 것은 허용되지 않았다. 원 R7의 자체 모델(chem_only, B11 등)은 이 경로를 쓰지 않았으므로 해당하지 않는다.

- test 정답 학습 누출이 아니다. HCC **train** 약물 학습은 원 계약에서 허용됐다.
- 원 checkpoint 폴더에는 선택된 checkpoint 하나만 있어 epoch별 재선정이 불가능했다 → 재학습으로 정정.

## 정정 결과

- 재학습한 6 trajectory(공식 3, ECFP 3)에서 source-only 선택과 원 규칙(HCC 포함) 선택은 6/6 같은 epoch를 골랐다. 선택 mask의 영향은 0이다.
- 정정값(R4 test, source family-equal ΔRMSE, 3 seed 평균): 공식 XPert **0.53776**(원 보고 0.53604), XPert-ECFP **0.54845**(원 보고 0.54385).
  - 원 보고값과의 차이(+0.0017 [−0.0031, +0.0075], +0.0046 [+0.0013, +0.0079])는 mask가 아닌 재학습 변동이다.
- 패턴 지표(공식): centred Pearson 0.183 → 0.191, MRR 0.352 → 0.361.
- 원 R7 결론(B11 미채택, XPert ΔRMSE 같은 범위, 패턴 지표 우위)의 방향은 바뀌지 않는다.
- 상세: `reports/r7_xpert_followup_v2/RESULTS.md` §1, `manifests/r7_xpert_followup_v2/selection_population_audit.json`, `manifests/r7_xpert_followup_v2/final/correction_*`.
