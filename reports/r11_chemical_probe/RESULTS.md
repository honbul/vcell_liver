# R11 Tahoe-100M 화학 표현 비교

## 이전 진행과 이번에 재사용한 자산

| 조사한 진행 | 확인한 결과 | R11에서 반영한 내용 |
|---|---|---|
| R6 `1b47c0c` | 기존 전체 오차 개선의 상당 부분을 같은 plate·세포주의 pool 평균이 설명했다. 반복 비교의 S5·S7 계산을 바로잡은 뒤에도 독립 반복성은 미확립이다. | 기존 화합물 분할과 선형·평균 상태 기준선을 이어받았다. 주 지표는 같은 pool 안의 약물별 차이다. |
| R8 `e7bcff3` | 미관측 약물 선택의 overlap 검사 351개가 통과했다. 화학 경로의 중심화 RMSE 차이는 −0.00266 (95% CI [−0.00394, −0.00142], −0.489%)로 작았고 기존 채택은 보류됐다. | R8의 관측된 동약물 반응을 R11 입력에서 배제했다. 기존 수치와 R11 점수를 직접 차감하지 않는다. |
| R9 `60a75b6` | GSE151530 25명 결과에서 높은 조건부 정확도와 제한된 출력 범위가 함께 나타났다. 희소 fibroblast는 75 donor×seed 조합 중 49에서 출력이 없었다. | 예측 오차·적격 범위·실용 가치를 별도로 보고한다. |
| R10 `2cc491e` | 기존 cross의 TV 개선 −0.0102는 보존됐다. 독립 확인 후보 0/3 적격으로 독립 Cross P1은 자료 대기 중이다. | R11 Tahoe 결과는 별도 과제의 평가다. |

[이전 자산 재사용 표](manifest/reuse.tsv) · [R6 결과](https://github.com/honbul/vcell_liver/blob/1b47c0ce85c82d5d8227432b9a234a2814d22fe2/reports/r6_sensitivity_correction_v3/RESULTS.md) · [R8 결과](https://github.com/honbul/vcell_liver/blob/e7bcff32c0b09206471eb0c3828e7f42fc8a4699/reports/r8_unseen_drug_selection_v3/RESULTS.md) · [R9 결과](https://github.com/honbul/vcell_liver/blob/60a75b6b261f5d0f7b93357b3ae1724b7bbbb549/reports/r9_output_utility_v3/usability_decision.md) · [R10 결과](https://github.com/honbul/vcell_liver/blob/2cc491e118bf55bb2dfd361904a76b1e8e82e4b5/reports/r10_independent_cross_v3/RESULTS.md)

## E0–E1 계약 점검

Metadata 검증에서 네 plate의 source 223 처리 웰·203 화합물군, final 후보 69 처리 웰·68 화합물군을 확인했다. 0.5 µM 표본 4개는 과거 화합물 분할 범위 바깥에 있었고, `plate13`의 적용 대상 source 19개·final 5개와 DMSO 두 웰은 격리했다. R6에서 사용하지 않았던 historical validation의 미사용 정책을 유지했다. 각 주 plate에 물리적 DMSO 웰 A/B가 존재한다. [표본 manifest](manifest/samples.tsv) · [자료 영수증](manifest/receipt.json)

동결된 ChemBERTa backbone의 CLS 출력은 384차원, 확인된 backbone 파라미터는 3,279,600개다. ECFP4와 동일 canonical 입력의 289 entity를 캐시했다. 누락과 truncation은 0, 짧은 CLS 재추출의 최대 절대차는 2.98×10⁻⁷이었다. [표현 영수증](manifest/feature_receipt.json)

실측 파일럿은 8개 화합물군과 DMSO 두 웰, 총 769,030세포 범위에서 끝났다. E와 C의 role A 주 오차는 각각 0.078088과 0.078209, role B는 0.078088과 0.078211이었다. 전체 ECFP4 선형 기준선은 두 role 모두 0.077983이었다. 이 수치는 실행 가능성 점검용이다. 별도 동일 설정 재실행에서 E와 C 및 2-epoch MLP의 예측 최대 절대차는 0이었다. [파일럿 기록](manifest/pilot) · [재실행](manifest/pilot/replay.json)
