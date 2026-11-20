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

## E2 개발 비교 및 E3 결정

Source 17,949,646세포에서 처리군 11,150개 well×line 행 중 10,809개가 QC를 통과했다. HepG2/C3A와 SNU-423에는 각각 적격 221개 웰과 201개 화합물군이 있었다. DMSO는 400개 well×line 중 393개가 QC를 통과했다. [개발 자료 coverage](manifest/source/coverage.tsv)

모델별 선택 설정의 개발 평균이다. 그룹 수로 fold를 가중했으며, 두 역할의 모델 선택은 각각 수행했다.

| 모델 | R-A 주 대비 RMSE | R-B 주 대비 RMSE | R-A 전체 ΔRMSE | R-B 전체 ΔRMSE |
|---|---:|---:|---:|---:|
| B0 no-change | **0.092946** | **0.093455** | 0.132567 | 0.128065 |
| B1 세포주 평균 | 0.092946 | 0.093455 | 0.111404 | 0.109875 |
| B1 축소 pool 평균 | 0.092946 | 0.093455 | **0.096054** | **0.096631** |
| B2 전체 ECFP4 선형 | 0.096854 | 0.097367 | 0.100355 | 0.100920 |
| B2 mean-state MLP | 0.093112 | 0.093624 | 0.100997 | 0.101517 |
| E. ECFP4 + 동일 Ridge | 0.096001 | 0.096170 | 0.099193 | 0.099442 |
| C. 동결 ChemBERTa + 동일 Ridge | 0.100266 | 0.100015 | 0.103790 | 0.103637 |
| 개발 진단. 대응을 치환한 C | 0.102135 | 0.101924 | 0.105511 | 0.105407 |

축소 pool 평균은 no-change 대비 전체 ΔRMSE를 낮췄다. 같은 pool에서 약물 간 예측 차이는 두 기준선 모두 0이다. 약물–표현 대응이 있는 C가 치환 C보다 낮은 오차를 냈어도, 두 역할의 사전 지정 진입 규칙에서 C가 E·B*·치환 C를 모두 이긴 fold는 없었다. 별도 OOF 잔차 규칙도 충족하지 않았다. **E3 FiLM은 미실행**이다. [R-A 개발 표](manifest/development/A/grid.tsv) · [R-B 개발 표](manifest/development/B/grid.tsv)

개발 종료 후 B*를 두 역할 모두 B0으로, E/C의 Ridge α를 1,000으로 동결했다. B2 MLP의 두 역할 선택은 학습률 0.0003과 final 1 epoch였다. 공식 검정 대비는 C−B* 1개, 귀속 대비는 C−E다. [동결 기록](manifest/freeze.json)

### Source 반복 진단

실물 반복 웰 중 정확히 같은 entity의 cross-plate 반복 그룹은 두 그룹이었다. 두 역할의 97–98개 cell-line 반복쌍에서 pool 중심화 cosine 중앙값은 0.421과 0.404였다. 같은 plate 반복은 한 그룹뿐이었다. 소수 화합물의 기술·배양 조건 의존성을 포함한 수치다. 기존 R6의 같은 웰 기술 half-split은 과거 gene panel과 정규화 계약에서 10,756행·202그룹, 중심화 cosine 중앙값 0.811이었다. 이 수치는 별도 기술 진단이다. [물리 웰 반복표](manifest/repeats/summary.tsv) · [기술 분할 영수증](manifest/repeats/receipt.json)

## E4 최종 동결 평가

0.5 µM, 24시간, plate2·5·8·11의 처리 웰 69개와 화합물군 68개를 한 차례 평가했다. 50개 세포주의 3,450개 처리 웰×세포주 후보 행에서 유효 세포 수 기준으로 88행을 제외했다. 물리적 DMSO 양 역할의 공통 평가 행은 3,347개였다. 196개 적격 pool에 모두 포함됐으며, pool당 행은 5–18개였다. 적합과 query에 사용된 공통 offset의 fallback은 양 역할 모두 0행이었다. [최종 자료 영수증](manifest/final/receipt.json) · [채점 영수증](metrics/receipt.json)

| 동결 arm | 약물 대비 RMSE ↓ | 전체 ΔRMSE ↓ | 절대 발현 RMSE ↓ | pool 내부 PDS ↑ | 기대 top-1 ↑ |
|---|---:|---:|---:|---:|---:|
| B0 no-change, B* | **0.093259** | 0.130030 | 0.130030 | 0.5000 | 0.0586 |
| B1 세포주 평균 | **0.093259** | 0.124795 | 0.109176 | 0.5000 | 0.0586 |
| B1 축소 pool 평균 | **0.093259** | **0.123667** | **0.096420** | 0.5000 | 0.0586 |
| B2 전체 ECFP4 선형 | 0.097353 | 0.127586 | 0.101064 | 0.4925 | 0.1134 |
| B2 mean-state MLP | 0.093411 | 0.124922 | 0.098098 | 0.5388 | 0.0873 |
| E. ECFP4 + 동일 Ridge | 0.096430 | 0.126222 | 0.099690 | 0.5268 | 0.0883 |
| C. 동결 ChemBERTa + 동일 Ridge | 0.100156 | 0.129195 | 0.103450 | 0.5226 | 0.0948 |

공식 대비 **C−B*는 +0.006898 (95% CI [+0.004277, +0.009129], 상대 +7.40%)**였다. 같은 head에서 표현만 바꾼 보조 대비 C−E도 +0.003727 (탐색적 95% CI [+0.001610, +0.005777], 상대 +3.86%)이었다. 동결된 ChemBERTa가 이 계약에서 약물별 대비 오차를 키웠다. 비교군 B1은 세포주·pool 공통 이동을 반영해 전체 오차가 줄었고, 약물 대비 점수는 B0와 같다. C의 PDS 0.5226과 top-1 0.0948은 다른 질문에 대한 기술 지표이며, 공식 판정은 주 RMSE를 따른다. [전체·role·plate·세포주 지표](metrics/arm_metrics.tsv) · [동결 대비와 구간](metrics/paired.tsv)

### 간계열 세포주

| 세포주 | 행 / 그룹 | 관측 대비 RMS | C 예측 대비 RMS | B* / E / C 주 RMSE | C−B* (기술적 95% CI) | B1 pool / C 전체 ΔRMSE |
|---|---:|---:|---:|---|---|---|
| HepG2/C3A | 69 / 68 | 0.090142 | 0.032637 | 0.090142 / 0.093796 / 0.096876 | +0.006734 [+0.004308, +0.008814] | 0.119226 / 0.124560 |
| SNU-423 | 69 / 68 | 0.091842 | 0.030776 | 0.091842 / 0.094113 / 0.097346 | +0.005505 [+0.002663, +0.007943] | 0.116903 / 0.121141 |

두 세포주에서 C가 예측한 약물 대비 벡터의 크기는 실측 대비보다 작았으며, 약물 대비 오차는 각 행의 기준선보다 높았다. HepG2/C3A는 hepatoblastoma 유래이고 SNU-423는 성인 HCC 유래다. 이 비교의 해석 단위는 두 세포주와 관측 조건이다. [HepG2/C3A](https://www.cellosaurus.org/CVCL_1098) · [SNU-423](https://www.cellosaurus.org/CVCL_0366)

### 민감도와 범위

- C−B* 점추정은 네 plate 각각 +0.00595~+0.00938이었다. 한 plate씩 제외한 재집계도 +0.00599~+0.00723이었다. 모든 재집계의 구간 하한이 0보다 높았다.
- 69개 test entity의 최대 train Tanimoto 중앙값은 0.253, 범위는 0.142–0.873이었다. 비어 있지 않은 scaffold가 source와 겹친 test entity는 9개였다. 사전에 지정한 unseen-scaffold subset은 57그룹·2,816행이었다. 양 역할에서 C 주 오차 0.10226 / 0.10178, E 0.09854 / 0.09835, B0 0.09512 / 0.09512였다. 고정 모델 한 번으로 채점한 보조 결과다. [구조 유사도](manifest/test_similarity.tsv) · [scaffold 보조표](metrics/unseen_scaffold_stress.tsv)
- Bootstrap은 68개 화합물군에 2,000회 양수 Exp(1) 가중을 적용하며 관측 plate, control, 모델 적합에 조건부다. 두 역할과 여러 seed의 점수는 같은 화합물군 안에 묶었다. 이전 프로젝트에서 사용한 Tahoe 자료 안에서 규약을 동결한 화합물 holdout이다.
- R11은 CPU로 실행했다. 원자료 처리 시간은 개발 약 11분, final 약 5분이었다. 체크포인트를 고정해 한 번 특징을 캐시했고 새 GPU 사용은 0시간이고, G1 전용 자산 디렉터리의 종료 시점 사용량은 6.9 GiB였다. [실행 기록](run_ledger.tsv)

## 재현과 검증

G1의 두 역할별 예측 벡터, 모델 파일, 정답 벡터, 평가 행, 2,000회 bootstrap 가중치·통계는 SHA-256과 서버 경로를 모두 기록했다. 로컬 보고서에는 작은 집계 파일과 평가 행의 압축본을 함께 둔다. [산출물 인덱스](manifest/artifacts.tsv)

채점 이외의 검증 과정에서 동결 파일·코드와 원자료 집계·예측 파일의 해시를 확인했다. source/final 간 화합물군과 물리적 처리 웰의 교집합은 각각 0이었다. 두 DMSO 역할의 물리적 웰도 겹치지 않았다. 독립 재집계가 최종 예측 벡터에서 전체·두 간계열의 두 대비 점추정과 가중 반복 표본을 다시 계산한 24건의 최대 절대차는 8.46×10⁻¹⁷이었다. 저장된 22개 percentile 구간도 일치했다. [완료 검증](manifest/completion_audit.json)
