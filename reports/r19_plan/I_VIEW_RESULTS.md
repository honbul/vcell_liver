# DEV-ARCH I 뷰 확보 및 지원 결과

R18 예약 측정, 그 서명과 별칭 및 B donor 예외를 제외하는 검사를 R17 개발 처리 웰 6,773개에 적용했다. 겹치는 물리 측정·서명은 0개이며 개발 모집단은 1,862 parent다. 세 뷰 각각 전체 100 parent 및 각 outer fold 20 parent의 기준을 통과했다. [출처·명세](I_VIEW_PROTOCOL.md) · [전체 및 교집합 원표](data/i_views/modality_coverage.tsv)

| 뷰 | 매칭 | Outer fold 0 / 1 / 2 | 범위 |
|---|---:|---:|---|
| I1 ChEMBL 37 | 1,467 / 1,862 (78.8%) | 491 / 495 / 481 | R7 원본을 읽기 전용으로 사용해 정량 activity 146,492행을 확보했다. 활동별 action type이 채워진 행은 9,478개다. 별도 표적 기전 주석을 활동값이 있는 것처럼 채우지 않았다. |
| I2 Cell Painting | 361 / 1,862 (19.4%) | 117 / 132 / 112 | A549 48 h, dose code 6의 물리 plate별 DMSO 정규화 consensus. 공통 morphology feature는 1,781개다. |
| I4 PRISM | 722 / 1,862 (38.8%) | 253 / 241 / 228 | 공개 19Q4에서 간 유래·간 유래로도 주석된 26개 및 tissue를 확인 못한 10개를 제거해 542개 비간계열 세포주로 지문을 만들었다. |
| 세 뷰 공통 | 344 / 1,862 (18.5%) | 112 / 126 / 106 | 두 뷰 이상을 보유한 군은 725개다. |

R18 명부에 같은 parent가 있는 경우까지 제외하는 민감도에서 세 뷰 공통은 248 / 1,564군이고 fold별로 85 / 91 / 72군이다. PRISM 배포본의 treatment-wise ComBat은 출판자 전처리다. 최종 지문에 직접 포함되는 세포주 축이 전부 비간계열임을 별도로 확인했다. [PRISM 세포주 감사](data/i_views/PRISM_cell_line_audit.tsv)

I2·I3·I4를 `PLANNED`로 전환했다. I3의 두 조합에서 내·외 fold를 포함한 24개 PSD 검증의 최소 고윳값은 0.00347, 구조 fallback의 최대 오차는 0이었다. 적합 fold에서만 정한 표준화와 PCA를 적용한 뒤 세 뷰 공통 실사용 지원은 outer fold별 109 / 124 / 105군이며, 동일 R18 parent까지 제외하면 82 / 89 / 70군이다. 이 단계의 증거는 입력 지원과 커널 수치 성질까지다. [I3 수치 검증](data/i_views/kernel_support.tsv) · [fold 전처리 지원](data/i_views/transform_support.tsv)

새 공개 파일은 171,300,016바이트이며, G1의 전체 새 작업 디렉터리는 감사 시점에 185,288,346바이트였다. 모든 공개 원본은 SHA-256과 연결된 라이선스를 기록했다. ChEMBL의 기존 30 GB SQLite도 별도 SHA-256을 기록했다. 원본과 feature는 `g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_dev_i/coverage.bk00jUHu`에 보존했다. [공개 원본 명부](data/i_views/public_source_manifest.json) · [ChEMBL 영수증](data/i_views/I1_receipt.json) · [작업 원장](data/i_views/jobs.tsv)

## 단일 뷰·결합 대조군 지원

CPU 작업 `30714`는 등록된 25개 커널 arm의 내·외부 적합과 query 축, 대칭성, I0 fallback 행을 두 모집단에서 확인했다. 전체 300조합이다. 등록 개발 모집단의 outer 평가에서 I3의 두 뷰 보유군은 113 / 128 / 110 parent이고, 구조 I0로 넘기는 군은 173 / 167 / 186 parent다. PRISM을 추가한 I3_all의 세 뷰 보유군은 109 / 124 / 105 parent이고, I0 fallback은 165 / 162 / 180 parent다. 이 점검은 공개 뷰 값과 개발 처리 웰의 잠금 메타데이터만 사용했다. 약물 반응 RMSE·R²는 실제 DEV 예측·동결·채점 단계에서 측정한다. [arm별 공통·부분·결측 지원](data/i_views/control_kernels/arm_support.tsv) · [검증 영수증](data/i_views/control_kernels/receipt.json)

## I 트랙 반응 실험 준비

CPU 작업 `30729`는 HEPG2 6 h 개발 웰 6,773개를 원 R15 처리 캐시 인덱스에 일대일로 맞추고, 9개 fold·역할 조합의 예측용과 채점용 명부 54개를 해시로 잠갔다. R18 예약 처리·별칭, score 역할의 물리 웰, B donor와 donor 정규화 대조군과의 겹침은 0개다. 과거 C3의 fit/select 맥락과 공유하는 역사적 대조군 128개는 별도 기록했다. 적합 집합에 같은 Murcko 골격이 없었던 평가 parent는 fold별 419 / 427 / 435개였다. 골격이 없는 비고리 parent 7 / 2 / 10개는 별도 층이다. [잠근 지원표](data/preflight/i_track/support.tsv) · [입력 영수증](data/preflight/i_track/receipt.json)

R17 계열 FiLM과 25개 커널의 합성 시험을 G1 CPU 작업 `30731`에서 실제 코드로 실행했다. 합성 처리 30웰로 적합해 합성 평가 16웰 × 12유전자의 예측과 대조군을 생성하고 해시를 검증했다. I 트랙의 실제 약물 반응 효과와 반복 구간은 후속 연구 데이터 채점에서 결정한다. [작업 원장](data/i_views/jobs.tsv)

## 실제 HEPG2 6 h 반응 평가 · 2026-11-22

**I1–I4 모두 `NO_USEFUL_GAIN`이다.** 네 주 후보의 기준선 대비 상대 RMSE 개선 99% 구간 상한이 모두 사전 문턱 0.5%를 밑돌았다. 분석은 HEPG2 6 h의 기존 6,773개 물리 웰, 1,862개 화합물에서 수행했다. 웰이나 화합물 수에는 세 역할 회전을 중복 계상하지 않았다. HEPG2 6 h 개발자료의 짝 비교로 해석한다. [운영 판정 명세](I_OPERATIONAL_PROTOCOL.md) · [판정 원표](data/i_views/efficacy/decision_summary.tsv)

| 후보 | 모든 해당 뷰가 있는 평가 parent | 상대 RMSE 개선 % [99%] | ΔR² %p [99%] | 두 지표 개선 fold | 운영 판정 |
|---|---:|---:|---:|---:|---|
| I1 ChEMBL | 1,326 | −0.101 [−0.163, −0.037] | −0.250 [−0.421, −0.095] | 0/3 | NO_USEFUL_GAIN |
| I2 Cell Painting | 361 | +0.062 [−0.107, +0.244] | +0.211 [−0.260, +0.733] | 2/3 | NO_USEFUL_GAIN |
| I3 ChEMBL + Cell Painting | 351 | +0.134 [−0.011, +0.284] | −0.076 [−0.523, +0.401] | 2/3 | NO_USEFUL_GAIN |
| I4 비간계열 PRISM | 722 | +0.121 [+0.042, +0.199] | +0.062 [−0.183, +0.289] | 2/3 | NO_USEFUL_GAIN |
| I3_all 세 뷰 | 338 | +0.129 [−0.040, +0.301] | +0.022 [−0.417, +0.497] | 2/3 | 보조 분석 |

각 행은 동일한 화합물에서 전향적으로 재적합한 R17 계열 결합 `B_R17`과 비교한다. 양수는 후보에 유리하다. 각 후보 비교는 전체 100개와 fold별 20개를 모두 넘었으며, 가용성에서 나온 표본 편향과 실제 값의 효과는 같은 지원의 mask·shuffle 대조로 따로 평가했다. [동일 지원 채점표](data/i_views/efficacy/score/same_support.tsv) · [역할을 묶은 상대 효과와 fold별 점추정](data/i_views/efficacy/operational/operational.tsv)

- I4는 PRISM을 보유한 722군에서 RMSE 개선 구간 전체가 0 위에 있었다. 99% 상한은 +0.199%였고 R² 구간은 0을 포함했다. 전체 1,862군의 구조 fallback을 포함했을 때는 RMSE +0.025%, ΔR² −0.023%p였다.
- I3는 351군에서 구조 기준선 I0 대비 RMSE가 +0.133%로 개선됐고, 더 강한 B_R17 대비 R² 점추정은 −0.076%p였다. 전체에 적용했을 때 B_R17 대비 RMSE −0.086%, ΔR² −0.226%p였다.
- I2·I3·I4와 I3_all은 동일 지원의 mask-only 및 parent-shuffle 대비 두 점추정이 모두 유리했다. 네 주 후보 중 셔플 대비 두 지표 개선에 이르지 못한 것은 I1이었다. 구조 기준선 I0에서 B_R17로 갈 때 전체 규모의 차이는 RMSE 0.024%, R² 0.086%p였다. [전체 fallback](data/i_views/efficacy/score/full_fallback.tsv) · [음성 대조](data/i_views/efficacy/score/controls.tsv)
- 내부 선택은 등록한 규제값과 출력 gain의 격자에서 수행했다. I1은 9조합 중 7곳에서 gain 0.25, I4는 8곳에서 0.5를 선택했다. 가용성·골격별 채점, FiLM을 뺀 커널 진단, plate 하나씩 제외한 민감도도 보존했다. [선택 영수증](data/i_views/efficacy/forecast_metadata/) · [가용성·골격별](data/i_views/efficacy/score/strata.tsv) · [커널 단독](data/i_views/efficacy/score/kernel_only.tsv) · [plate 민감도](data/i_views/efficacy/score/plate_sensitivity.tsv)

입력 추출 `30732` → 예측 `30733` → 전체 동결 `30734` → 별도 채점 `30735` → 운영 분석 `30753`이 모두 종료 코드 0으로 끝났다. 동결 SHA-256은 `605f86ec99f862987f72db05fe92455809d0d02be2c32ed80bee2062deb94bf7`이다. 9개 예측 영수증과 선택표, 채점 원표와 보조 분석의 모든 해시를 대조했다. CPU 할당 합계는 0.98 CPU시간, GPU 사용량은 0이었다. G1의 I 트랙 전체 신규 작업 공간은 약 2.71 GB였다. [동결 기록](data/i_views/efficacy/freeze.json) · [채점 영수증](data/i_views/efficacy/score/receipt.json) · [실행·자원 원장](data/i_views/efficacy/resource_usage.tsv)
