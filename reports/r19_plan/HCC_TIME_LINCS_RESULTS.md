# 성인 간암 LINCS 시간 짝 지원 감사

**보유 LINCS 세 release에서 R18 예약을 지키며 HEPG2의 6→24시간 모델을 성인 간암에 평가할 적격 짝은 0군이다.** HUH7, JHH-5, JHH-7, SNU-423, SNU-387, PLC/PRF/5의 기존 단일 약물 메타데이터를 대조했다. 두 시간의 동일한 실제 농도, 서로 다른 RNA plate, 시간별 최소 두 plate 및 각 plate의 독립 DMSO 3웰을 요구했다. [명세](HCC_TIME_LINCS_PROTOCOL.md) · [JHH-5 등록](https://www.cellosaurus.org/CVCL_0364) · [JHH-7 등록](https://www.cellosaurus.org/CVCL_2805)

| Release와 세포주 | 6 h / 24 h 원 서명 | 예약 제외 전 적격 반복 parent | 예약과 해당 release 별칭 제외 후 |
|---|---:|---:|---:|
| GSE92742 HUH7 | 267 / 260 | 94군·174 약물–농도 조건 | 0군 |
| GSE70138, 감사한 여섯 세포주 | 0 / 0 | 0군 | 0군 |
| CMap 2020 HUH7 | 267 / 260 | 2군·2 약물–농도 조건 | 0군 |
| CMap 2020 JHH-5 | 0 / 2,152 | 0군 | 0군 |
| CMap 2020 JHH-7 | 0 / 363 | 0군 | 0군 |

GSE92742 HUH7에서는 예약 제외 전 두 시간에 등장한 화합물이 113군이고 서명 QC 뒤 95군이었다. 정확한 실측 농도와 RNA plate 반복을 적용해 94군이 남았다. 적격 처리 웰은 시간별 602·605개였고 fold별 반복 지원은 26·29·39군이었다. 그 1,207웰 중 457웰은 R18 물리 웰 또는 서명과 직접 일치했다. 원래 여섯 RNA plate의 적격 DMSO 184웰은 예약 도메인을 제외한 뒤 plate마다 0개였다. 따라서 직접 처리 웰이 남는 경우에도 이 plate에서 서로소 대조 역할을 만들 수 없었다. [서명 집계](data/hcc_time_lincs/audit/signature_support.tsv) · [물리 지원](data/hcc_time_lincs/audit/physical_support.tsv) · [plate별 vehicle 지원](data/hcc_time_lincs/audit/plate_retention.tsv)

CMap HUH7은 R18 제한을 적용하기 전에도 GEO와 공유하는 서명·웰·대조군만 제외하면 두 시간 짝이 0군이었다. JHH-5와 JHH-7은 모두 주어진 감사 범위에서 6시간 서명이 없었다. G1에는 기존 Level 3 파일이 있어 파일 존재를 기록했다. 감사는 파일을 열기 전에 종료했으며, 규약의 100군 및 fold별 20군을 충족하는 cohort는 없었다. 94군은 예약 이전의 메타데이터 수치이며, HCC에서의 예측 효과는 미실행이다. [영수증](data/hcc_time_lincs/audit/receipt.json)

G1 CPU 작업 `31517`에서 단계별 지원을 센 뒤 `31518`에서 GEO 별칭 단독 제외와 plate별 감소를 분리했다. 두 작업이 종료 코드 0이었고 공유 단계의 원표는 일치했다. GPU 할당과 신규 다운로드는 0이며 R18 발현은 미개봉 상태다. [작업 원장](data/hcc_time_lincs/jobs.tsv) · [자원 내역](data/hcc_time_lincs/resource_usage.tsv)
