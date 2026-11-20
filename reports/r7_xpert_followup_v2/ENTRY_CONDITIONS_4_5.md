# §4.5 새 HCC·graph 후속 — 진입 조건 기록 (미실행)

근거: private plan 2026-11-19 §4.5. 이 트랙에서는 새 HCC 자료나 graph 실험을 실행하지 않았다. 아래는 다음 분기를 열기 위한 조건만 적는다.

## 1. 새 HCC 세포주·환자로의 전이 (cell-OOD)

- 현재 R7-C의 HCC 결과는 **같은 HCC 세포주(HUH7, JHH5, JHH7)의 새 약물**로의 일반화다. HCC train 약물의 반응으로 HCC별 a_c, b_c를 적합하는 것이 원 계약에서 허용됐기 때문이다. 새 세포주 전이가 아니다.
- 새 세포주·환자에는 train 약물 반응이 없으므로 cell별 평균이나 a_c, b_c를 직접 적합할 수 없다.
- 진입 조건: (a) 실제 개입·용량·시간·batch·vehicle metadata와 보류 정답이 있는 새 HCC 반응 자료, (b) 반응 label 없이 쓸 수 있는 fallback을 사전에 지정한 cell-OOD 계약(source global a₀, b₀ 또는 관측 control 기반 보정), (c) 현재 drug-OOD(HCC train 사용) 결과와 분리한 보고.
- 이번 트랙의 산출물 중 그 계약에 바로 쓸 수 있는 것: Cg(global) 팔의 계수와 "HCC global-only" 진단 결과(`manifests/r7_xpert_followup_v2/oof/per_seed_summary.tsv`의 `|hcc_global_only` 행).

## 2. Graph 확대

- 진입 조건: 더 넓은 출력(978 landmark 밖), 새 주석, 독립 반응 자료가 있고, 실제 target seed가 출력에 전달되는 경로가 명확할 때 새 질문으로 연다.
- 그때에도 metadata M, 빈 표적 G0, 재배정 GS 또는 seed-difference 대조를 유지한다.
- 원 R7과 R7 후속 v1의 음성 결과만 근거로 3-hop이나 gate 크기를 자동으로 늘리지 않는다. 이번 트랙의 R7-C/R7-L 결과는 graph에 대한 근거가 아니다.

## 3. 현재 상태

두 분기 모두 **미실행**. 위 자료·계약이 없기 때문이며, 이 트랙의 결과와 무관하게 열 수 없다.
