# R15 학습 전 LINCS 감사

두 GEO 원본의 메타데이터와 이전 실험의 입력 원장을 먼저 확인했다. 새 R15 모델과 Tahoe 외부 비교의 채점은 이 감사의 범위 밖이다. HepG2는 간모세포종 유래로 표시하며, HUH7과 Tahoe SNU-423은 별도 간세포암 평가 범위다. [HepG2 기원 연구](https://pubmed.ncbi.nlm.nih.gov/19751877/) · [L1000 원 논문](https://doi.org/10.1016/j.cell.2017.10.049)

## 노출

| 라운드 | 사용 범위와 역할 | 과거 HEPG2 측정 범위 | 과거 비간계열 측정 범위 |
|---|---|---:|---:|
| R6 | LINCS를 독립 반복 자료 후보로 검토했다. 적합·설정 선택에 들어가지 않았다. | 0 | 0 |
| R8 | CMap 2020 Level 3, 명목 10 µM·24 h. source 반응과 설정 선택에 쓴 family bank, 별도 HUH7·JHH HCC 평가의 합집합이다. HCC는 과거 적응·채점이 가능한 등록 범위를 보수적으로 포함했다. | 4,407웰·1,120 canonical parent | 226,627웰·130계열 |
| B1 | CMap 2020 Level 3, 명목 10 µM·24 h. HEPG2·PHH 포함 source 161계열, 별도 HUH7·JHH5·JHH7 대상. | 4,407웰·1,120 canonical parent | 241,644웰·159계열 |
| R13 | B1 source에서 Tahoe catalog 대응군을 제외해 PCA·방향 prior와 Ridge 전이에 사용했다. 공개 훈련은 24 h 자료다. | 4,008웰·1,059 canonical parent | 219,938웰·145계열 |
| P2 | 두 GEO release의 24 h Level 5. Tahoe 매칭군 서명 bank, 구조·서명 혼합 및 직접 사상에 사용했다. | 143서명·23 pert_id | 8,799서명·36계열 |
| P3 | 같은 24 h Level 5 bank에서 HepaRG 매칭군을 가져와 고농도 구조 결합에 사용했다. | 384서명·52 pert_id | 18,476서명·37계열 |

GSE92742 HEPG2 6 h·5–20 µM 3,345군에서는 **동일 서명 또는 그 구성 물리 웰의 확인된 선행 노출이 다섯 경로 모두 0**이었다. 3,258군은 B1의 24 h 비간계열 측정에 이미 같은 canonical parent가 있었다. B1의 24 h HEPG2와 같은 물리 웰이 들어간 서명은 5–20 µM에서 GSE92742 182개(138군), GSE70138 229개(205군)였다. P2·P3의 동일 Level 5 서명 겹침도 두 release의 24 h에서 확인했다. 6 h의 동일 측정 노출과 다른 시간·세포주의 화학 동일성 노출을 각각 기록했다. 이전 checkpoint와 response bank는 R15 적합·초기화에 사용하지 않는다. [노출 범위 원표](audit/exposure-verified/scope.tsv) · [시간별 선행 측정·화합물 겹침](audit/exposure-verified/overlap.tsv) · [원장 입력 해시](audit/exposure-verified/receipt.json)

## 위치와 배치

GSE92742의 `rna_plate`·`rna_well`과 GSE70138이 제공하는 `det_plate`·`det_well`을 사용했다. 이 두 좌표계의 의미는 release별로 유지했다. Level 3과 4의 인스턴스에서 동일 canonical parent·시간·정량 용량의 다른 plate 쌍을 만들었다. Level 5 MODZ의 좌표는 `distil_id`를 인스턴스에 조인해 단일 좌표로 확인된 서명에만 부여했다. 아래 위치 예측자는 같은 좌표의 다른 plate에서 온 fit parent 반응을 평균했다. 비교 donor의 pool 중심도 fit parent만 사용했다. 대조로 같은 plate에서 좌표를 순열했고, 이미 합쳐진 MODZ에서 물리 plate가 하나라도 겹치는 donor는 제외했다.

| 자료와 범위 | 배치 반복과 vehicle | 같은 위치·다른 약물 예측자 0.25배의 B0 대비 RMSE 감소율 |
|---|---|---|
| GSE92742 HEPG2 6 h, 5–20 µM | 같은 약물의 cross-plate 쌍 29,054개 중 같은 좌표 14,275개(49.1%). DMSO A−B 대비의 plate 간 평균 코사인 Level 3 0.00323, 순열 0.00001. Level 4 0.00045, 순열 −0.00087. | Level 3 **+0.013% [−0.014, +0.039]**, Level 4 **−0.004% [−0.026, +0.019]**. Level 5의 단일 좌표 서명에서는 **+0.172% [+0.099, +0.246]**. 99% 화합물군 구간. |
| GSE92742 HEPG2 24 h, 5–20 µM | 2,060개 중 같은 좌표 978개(47.5%). DMSO 대비 코사인 Level 3 0.01924, 순열 −0.00510. | Level 3 −0.505% [−0.752, −0.270], Level 4 −0.583% [−0.952, −0.287]. Level 5의 적격 pool은 한 배치로 다른 배치 donor가 없었다. |
| GSE70138 HEPG2 24 h, 5–20 µM | 1,551개 중 같은 검출 좌표 1,113개(71.8%). DMSO 대비 코사인 Level 3 0.01604, 순열 0.00028. | Level 3 −0.061% [−0.481, +0.349], Level 4 −0.329% [−0.653, −0.020], Level 5 −0.396% [−1.107, +0.289]. |

GSE92742 HEPG2 6시간의 5–20 µM MODZ 서명 4,394개 모두에서 인스턴스 연결을 확인했다. 단일 좌표에 해당한 서명은 4,356개였다. 위치 배치가 반복되며 MODZ에 작은 예측 이득이 남는다. Level 3/4에서 같은 규모의 RMSE 이득은 확인되지 않았다. 6·24시간 HEPG2 분석에 들어간 vehicle은 DMSO였고, 원본의 PBS는 4시간 조건에 있었다.

[GSE92742 위치 원표](audit/GSE92742/position.tsv) · [GSE70138 위치 원표](audit/GSE70138/position.tsv) · [Slurm 감사 기록](audit-jobs.tsv)

## 자료 지원

GSE92742의 HEPG2 `trt_cp` 5,774개 서명, **3,927 pert_id**를 재현했다. 6시간은 3,880 ID와 **3,433 canonical parent**였다. 5–20 µM에서 구조가 확인된 parent는 3,313개였고, `distil_cc_q75 ≥ 0.20` 및 최소 두 distil 측정 조건을 충족한 서명이 있는 parent는 1,888개였다. 전체 HEPG2 중 3,890 ID는 다른 비간계열 세포주의 측정이 있었다. 6시간·5–20 µM 비간계열 측정이 있는 6시간 HEPG2 ID는 3,793개였다. 값은 assay 전체의 원본 메타데이터 집계이며 실제 평가는 같은 DMSO와 구조·중복 제거 지원에 따라 줄 수 있다.
