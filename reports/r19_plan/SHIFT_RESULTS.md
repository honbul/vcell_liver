# LINCS–Tahoe 전이 병목과 T 트랙 지원

## T0. 동결된 Tahoe 외부 평가의 재검증

기존 R15 외부 예측의 약물 대비를 전역 배율 하나로 얼마나 복구할 수 있는지 확인했다. G1의 원 동결·채점·방향 진단에서 공개된 표 여섯 파일의 SHA-256을 저장소 사본과 대조했다. 5 µM, 세포 수 기준 20 또는 200, 기존 `all-A` 및 `LINCS-supported`, 등록 세포주 층에서 총 224개의 집계행이 일대일로 대응했다. pooled cosine과 RMS 비로 계산한 `2ρc − c²`는 원 R²와 최대 절대오차 `3.6 × 10⁻¹⁵`로 일치했다. [원 외부 채점](../r15_lincs/data/external/scored/points.tsv) · [검증 영수증](data/t0/score-only/receipt.json)

| 같은 간 유래 지원 집단 | 원 pooled R² | 예측 RMS를 실측과 같게 조정한 R² | 방향 고정·전역 비음수 배율의 최대 R² |
|---|---:|---:|---:|
| R15-A · 203군 전체 | −45.61% | −94.55% | +0.074% |
| R15-A · B와 공통 108군 | −47.93% | −90.07% | +0.247% |
| R15-B · 공통 108군 | −342.48% | −69.27% | +2.360% |
| 같은 화합물의 비간계열 donor `B_cell` · 공통 108군 | −346.05% | −70.46% | +2.182% |

108군 공통 표는 HepG2/C3A와 SNU-423, 공통 161 유전자, 최소 20세포의 집계다. 같은 위치를 이용한 기준선 `B_pos-all`의 실제 R²는 +3.265%로, R15-B의 배율 상한보다 0.905%p 높다. 두 세포주에서 따로 계산해도 R15-B 상한은 +2.311%와 +2.424%이고 위치 기준선 R²는 +2.921%와 +3.527%였다. 10개 T2 비교군 중 같은 108군에서 원 R²가 가장 높았던 FiLM은 +1.716%였다. [동일 지원 기준선 표](data/t0/score-only/baseline_gaps.tsv)

R15-B의 예측 대비 RMS는 실측의 2.011배, 에너지 합산 cosine은 0.154였다. 배율을 줄이면 큰 과대 예측은 완화할 수 있다. 현재 벡터의 방향에서 더 얻을 수 있는 pooled R²에는 위 수치의 제한이 있다. 공통 양의 배율은 기존 cosine과 검색 순위를 그대로 유지한다. 수식에서 얻은 최고 배율 0.07641은 이미 채점된 표의 진단값이며, T1과 R18의 선택·예측에는 사용하지 않는다. 이 상한은 미래 모델이나 별도 방향 변환, parent-macro RMSE의 상한으로 해석할 수 없다. [방향·크기·상한 원표](data/t0/score-only/direction_decomposition.tsv)

DEV의 전향적인 B_R17 및 새 T1–T4 비교는 이 예전 표에서 회수할 수 없다. 여기서는 예전 점수·집계만 사용했으며 실제 모델 재적합이나 처리 배열 읽기는 수행하지 않았다.

최소 200세포의 별도 107군에서는 R15-B의 방향 고정 상한이 +2.469%, 같은 지원의 위치 기준선이 +2.099%였다. 이 층에서는 이상적인 배율이 위치 기준선보다 높은 점추정을 얻을 가능성이 남아 있다. 관측된 R15-B 원점은 R² −365.24%였고, 새 배율을 외부 정답 없이 선택해 검증하는 것은 별도 실험이다. [세포 수 감도 원표](data/t0/score-only/baseline_gaps.tsv)

## T1–T4의 계산 계약

T1은 source OOF 전체 효과의 에너지 내적으로 비음수 scalar를 적합한다. 상한은 2이고 예측 에너지가 0이면 gain은 0이다. 부모와 domain을 함께 제외한 source cross-fit은 허용된 적합 행의 ID만 처리 정답 공급 함수에 넘긴다. 이 목적함수는 큰 오차를 포함한 pooled R² 좌표의 보정이며 parent-macro RMSE는 별도 평가한다.

T2는 같은 assay·집계·유전자 좌표로 등록한 물리 fit 대조군에서만 분산을 추정한다. target fit 대조군은 최소 12웰과 3plate가 필요하다. Source 분할에서 정할 shrink·floor·clip을 적용해 양의 diagonal을 구한다. T3는 행과 유전자 축이 확인된 source OOF 예측에 해당 diagonal을 적용한 뒤 scalar를 다시 적합한다. 각 보정 함수는 target 처리 정답을 받지 않는다.

T4 시차 연산자는 정확한 release·세포주·parent·실측 용량에서 맞춘 6 h와 24 h source pair를 이용한다. Canonical parent마다 총 가중치는 1이다. 적합 pair에서만 평균과 입력 SVD를 계산하고 rank `4, 16, 32`, ridge `0.1, 1`을 후보로 둔다. 독립 parent 수와 수치 rank가 모자란 후보는 지원 부족으로 남긴다. 시차 연산자는 identity, source scalar, 적격한 직접 24 h donor와 비교한다. 위 네 계산의 검증에는 합성 배열을 사용했다.

## T2·T3. 5 µM Tahoe의 물리 fit 대조군과 LINCS 지원

Tahoe 5 µM 평가 plate에 대응하는 fit DMSO는 HepG2/C3A와 SNU-423 각각 **5개 물리 웰, 5개 plate**였다. 세 fold, 두 역할 회전, 세포 수 기준 20/200의 24개 조합에서 같았다. 회전이나 같은 웰에 든 여러 세포주를 독립 실험으로 추가 계수하지 않았다. 최소 12개 웰을 충족하지 못해 이 외부 비교의 T2·T3는 `BLOCKED_CONTROL_SUPPORT`다. Tahoe의 물리 웰×세포주 CP10k 로그 pseudobulk와 LINCS Level 3의 물리 웰 log2 L1000도 집계·측정 단위가 다르다. ln(2) 변환은 로그 밑을 맞출 뿐이며 이 두 분산을 같은 좌표에서 측정했다는 근거가 없다. [Tahoe 역할별 지원](data/t2_control_support/tahoe_fit_controls.tsv)

GSE92742의 실측 10 µM에서는 R18 보호 역할과 허용 예외의 측정, 별칭, donor 정규화 대조군을 배제한 뒤 아래의 same-assay 후보가 남았다.

| GSE92742 세포·시간 | parent | 물리 RNA plate | fit DMSO A / B / C | R18의 모든 등록 대조군 역할도 제외한 경우 |
|---|---:|---:|---:|---|
| HEPG2 6 h | 1,793 | 41 | 189 / 178 / 162 | 1,755 parent, 35 plate, 145 / 135 / 121웰 |
| HEPG2 24 h | 18 | 6 | 43 / 41 / 38 | 독립 세 역할을 유지한 plate 없음 |
| PHH 6 h | 42 | 3 | 33 / 32 / 32 | 42 parent, 3 plate, 33 / 32 / 32웰 |
| PHH 24 h | 528 | 18 | 81 / 79 / 68 | 528 parent, 18 plate, 81 / 79 / 68웰 |

PHH의 시간대별 donor 독립성은 별도 검증이 필요하다. GSE70138의 집계는 detection 좌표 기준이라 물리 RNA 검증을 끝낸 domain 수에는 넣지 않았다. 이 표는 대조군 수와 공통 assay 지원이다. 보정 효과와 leave-domain-out의 parent 분리 가능성은 별도 판정한다. CPU 작업 `30757`은 원 메타데이터와 해시로 잠근 행 명부만 사용했다. [LINCS 지원표](data/t2_control_support/lincs_domains.tsv) · [감사 영수증](data/t2_control_support/receipt.json) · [G1 작업](data/t2_control_support/job.tsv)

같은 10 µM 명부의 HEPG2 6 h를 통째로 hold-out하고 그 domain의 1,793 parent를 모든 source에서 빼면, HEPG2 24 h 1군·3 plate, PHH 6 h 0군, PHH 24 h 219군·15 plate가 남았다. 고정 parent fold 하나씩을 모든 source에서 제외하는 설계에서는 PHH 6 h의 적합 지원이 31·29·24군, PHH 24 h는 354·340·362군이었다. R18의 모든 대조군 역할까지 빼면 HEPG2 24 h는 지원에서 빠지며 HEPG2 6 h held fold는 583·589·583군이다. 전체 domain 제외와 fold별 parent 제외의 적합 구성은 다르므로 두 설계를 같은 대조로 해석할 수 없다. [교차 지원 원표](data/t2_parent_support/leave_domain_out.tsv) · [감사 영수증](data/t2_parent_support/receipt.json) · [G1 작업](data/t2_parent_support/job.tsv)

## T4. 동일 세포주·화합물·실측 용량의 6↔24 h 지원

GSE92742 원 QC 메타데이터에서 R18 보호 측정과 별칭·donor를 제외하고, 같은 세포주와 정확한 실측 용량으로 6 h와 24 h를 연결했다. 각 plate에 세 역할을 배정할 수 있는 독립 DMSO가 최소 3웰 있어야 한다. 처리 식별자가 충돌한 RNA 좌표 54개는 그에 연결된 108개 레코드를 모두 제외했다. 첫 실행은 충돌을 검출하고 중단했고 같은 작업 ID의 재실행에서 제외 규칙을 적용했다.

| GSE92742 | parent | 세포주·parent·용량 쌍 | 양쪽 시간에 각각 2개 이상 RNA plate가 있는 쌍 |
|---|---:|---:|---:|
| 전체 | 4,458 | 13,117 | 12,924 |
| HEPG2 | 132 | 247 | 247 |
| PHH | 67 | 104 | 104 |
| HUH7 | 0 | 0 | 0 |

HEPG2 parent는 hold-out fold별 36·42·54군이며 실측 10 µM 83군, 2.5 µM 50군이다. 두 시간에 사용된 서로소 RNA plate는 각각 41개와 12개다. PHH의 10 µM은 36군이며 시간별 donor 독립성은 미검증이다. T2의 10 µM 수는 기존 R15 캐시에서 센 것이므로 원 QC 릴리스로 계산한 T4와 모수가 다르다. GSE70138은 DET 좌표 기준만 확인되어 RNA 기반 지원에 포함하지 않았다. 이 감사는 실제 발현을 읽지 않고 지원만 검증했다. [시간 쌍 원표](data/t4_support/support.tsv) · [실측 용량별 표](data/t4_support/dose_support.tsv) · [감사 영수증](data/t4_support/receipt.json) · [G1 작업](data/t4_support/jobs.tsv)

이 지원을 고정해 별도로 실시한 HEPG2 T4에서 저랭크 시간 연산자는 단순 배율 대비 RMSE를 7.33% [99% 구간 5.06, 9.58] 줄였다. 독립 24 h 반복과의 차이는 두 지표 모두 0을 포함했다. [T4 동결 실험 결과](T4_RESULTS.md)
