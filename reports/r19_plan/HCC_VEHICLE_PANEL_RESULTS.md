# Tahoe SNU-423의 물리 DMSO와 평가 대상 민감도

**Tahoe 5 µM 평가 plate 다섯 개에는 물리 DMSO가 각각 두 웰이었다.** 기존 적합·선택·채점의 세 역할을 같은 plate 안에서 모두 분리할 수 없다. 원 Tahoe sample metadata와 보유 A/B 명부를 대조했다. Plate 9에서 DMSO라고도 표기된 별도 웰은 실측 명부상 5 µM trametinib solvate 처리였고 vehicle은 0 µM `DMSO_TF`로 확인했다. [사전 명세](HCC_VEHICLE_PANEL_PROTOCOL.md) · [물리 대조군 지원](data/hcc_vehicle_panel/lock/controls.tsv)

기존 SNU-423의 203개 화합물·275웰·978개 유전자에 동일 처리의 A·B 대비를 적용했다. 아래 비율은 A−B의 parent 균등 RMS를 A·B 평균 반응의 RMS로 나눈 기술 지표다. 99% 구간은 같은 평가 pool에서 parent를 다시 뽑아 계산했다. [역할 차이 원표](data/hcc_vehicle_panel/diagnostic/role_gap.tsv)

| 평가 대상 | A−B RMS | 반응 RMS 대비 차이 |
|---|---:|---:|
| SNU-423의 웰 대비, plate pool 중심화 전 | 0.05705 | 54.08% [99% 46.84, 62.65] |
| SNU-423의 웰 대비, plate pool 중심화 후 | 1.47×10⁻¹⁷ | 수치 정밀도 수준 |
| 웰마다 가용 세포주를 평균한 기존 P4, plate pool 중심화 후 | 0.001511 | 2.85% [99% 2.23, 3.53] |
| plate 공통 세포주 패널로 평균, plate pool 중심화 후 | 2.39×10⁻¹⁷ | 수치 정밀도 수준 |
| 적합에서 정한 10개 세포주 패널로 평균, plate pool 중심화 후 | 1.59×10⁻¹⁷ | 수치 정밀도 수준 |

같은 plate의 SNU-423에 동일 DMSO를 사용하면 A↔B 교체는 모든 처리 웰에 같은 유전자 벡터를 더한다. Plate pool 중심화로 그 항이 상쇄됐다. 기존 P4는 웰마다 가용 세포주가 달라 평균 DMSO 차이도 웰마다 바뀔 수 있었고, A−B의 일부가 남았다. 패널을 고정하면 이 평균 항도 plate 내부 상수가 된다. 두 고정 방식은 각각 203군과 275웰 전부를 유지했다. Plate별 공통 패널은 45–48개 세포주였고, 다섯 plate와 세 fold 전체에서 공통인 세포주는 후속 메타데이터 확인에서 44개였다. 최대 절대 A−B 차이는 두 고정 방식 모두 1.2×10⁻¹⁵ 이하였다. [pool별 지원](data/hcc_vehicle_panel/lock/pool_support.tsv)

이 진단은 **채점 대상의 차량 교체 민감도**를 분리했다. 모델 입력의 DMSO 변화나 내부 설정 선택에서 생기는 편향은 별도 재적합을 요하며, 기존 P4의 성능 판정은 기존 대상에 해당한다. 공통 패널로 새 모델을 적합하거나 평가하지 않았다.

명세 `b15430d`를 고정했다. 초기 잠금 `31544`에서 DMSO 문자열 검색이 약물 solvate를 포함해 명부 일치 검사가 멈췄다. 정확한 vehicle 이름과 0 µM를 대조하도록 수정한 뒤 잠금 `31546`과 별도 진단 `31547`이 모두 종료 코드 0이었다. 기존 개발 정답 캐시만 읽었고 예약 발현 접근과 신규 수집은 없었다. [작업 원장](data/hcc_vehicle_panel/jobs.tsv) · [잠금 영수증](data/hcc_vehicle_panel/lock/receipt.json) · [진단 영수증](data/hcc_vehicle_panel/diagnostic/receipt.json)
