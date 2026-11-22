# 공개 오가노이드·일차 간세포 자료 접근 감사

2026-11-22 KST. 공식 카탈로그·응답 헤더와 원논문·보충표 설명을 확인했다. 행렬 본문, 원시 서열, 환자 자료는 읽지 않았다.

## OEP003191 / LICOB

**공개 생물학 자료 파일과 약물 반응표의 출처를 확인했다. 모델 ID의 교집합과 파일별 재사용 조건은 확인 대기다.** 원논문은 57명에서 확립한 간암 오가노이드 65개 중 44개를 HCC로 분류한다. 공개 보충표 S1은 모델 설명, S5는 76약물 패널, S6–S8은 오가노이드의 72시간 CellTiter-Glo 3D 약물 반응에서 계산한 AUC·IC50·Emax다. 약물 반응은 오가노이드 배양 시험값이다. 원논문은 시험에서 일부 후보를 PDX에서도 검토했으며 이 자료를 오가노이드 반응과 별도 실험으로 다룬다. 기저 RNA 후보는 NODE 행렬에서 처리 전 오가노이드 RNA로 식별되는 열이다. 원 종양 조직 RNA나 환자 임상 정보를 입력으로 쓸 경우 별도 사용 범위와 사용자 승인이 필요하다. [원논문·보충표](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949980/) · [NODE 공식 프로젝트](https://www.biosino.org/node/project/detail/OEP003191)

| NODE 공식 공개 경로 | 메타데이터 증거 |
|---|---|
| [관련 분석 카탈로그](https://www.biosino.org/node/api/app/project/getRelatedAnalysis/OEP003191) | 분석 OEZ00014368에 파일 OED00796358과 체크섬 OED00796361이 둘 다 `security=Public`으로 등록됐다. 카탈로그의 `accessible` 표시와 별도로 해당 공개 파일 URL에서 실제 응답 헤더를 확인했다. |
| [OED00796358 공개 파일](https://www.biosino.org/node/api/download/node/data/public/OED00796358) | 로그인 없는 HEAD가 HTTP 200, `application/octet-stream`, `Content-Length: 69058905`, `Content-Disposition: attachment;filename=LICOB_omics_matrix_data.xlsx`을 반환했다. |
| [OED00796361 체크섬](https://www.biosino.org/node/api/download/node/data/public/OED00796361) | 로그인 없는 HEAD가 HTTP 200, 32바이트, `LICOB_omics_matrix_data.xlsx.md5` 첨부 헤더를 반환했다. |

공개 다운로드 경로가 확인됐지만 RNA 시트의 오가노이드 열·표본 ID와 S6–S8 행 ID는 아직 대조 전이다. 공개 카탈로그, 파일 헤더와 논문의 데이터 가용성 진술에는 이 행렬에 적용할 구체 재사용 라이선스가 확인되지 않았다. 세포 분양의 MTA는 별도 물질 조건이다. 파일 단위 연구 이용 조건과 환자 유래 자료 범위를 확인한 뒤, 적격한 경우 오가노이드 RNA·반응·환자군 중복에 대해 ID 감사부터 한다. 이 자료는 이번 외부 평가 분모에 포함하지 않았다. [자료·물질 이용 설명](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949980/#S22)

## HRA004348 / 주 등 연구

**공개 원시 서열 메타데이터와 약물 보충표가 있다. 로그인 없는 오가노이드 기저 RNA·반응 결합 자료는 확인되지 않았다.** 상세 반응은 초기 301약물 스크리닝과 별개인 64약물·21개 hepatobiliary 오가노이드의 5점 농도 측정이다. S4에는 약물 패널과 약물별 AUC 표준화 반응 점수가 포함된다. GSA의 PRJCA016100 등록과 RNA run·모델 ID 예를 확인했으나 공식 연구 페이지에 브라우저 확인 응답이 돌아와 이 accession의 원서열 접근 표시를 독립 확인하지 못했다. 원논문은 raw·분석 자료의 요청을 저자에게 보내고 공유 가능한 자료를 MTA로 제공한다고 안내한다. 공개 S4로 기저 RNA를 대체할 수 없다. 접근 승인과 실제 분석용 RNA 및 ID 대응표가 필요하다. [원논문·S4](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897507/) · [공식 연구](https://ngdc.cncb.ac.cn/gsa-human/browse/HRA004348) · [공식 RNA run 카탈로그](https://ngdc.cncb.ac.cn/gsa-human/browse/runDetail/HRR1166768) · [NGDC 공식 접근 지침](https://ngdc.cncb.ac.cn/gsa-human/document/GSA-Human_Request_Guide_for_Users_us.pdf)

## 선택적인 일차 간세포 확장

| 자료 | 현재 이용 가능한 범위와 역할 |
|---|---|
| Open TG-GATEs PHH | G1 공유 경로 `/mnt/S1/sdata/pubdata/TGGATEs`에 human in vitro CEL 원본·속성과 cell viability ZIP이 이미 있다. 기존 메타데이터 감사에서 24시간 처리 157화합물·785배열, 대조 314배열이다. 기존 LINCS PHH와 정확한 parent·농도·시점 쌍으로 연결된 적격 행은 당시 0이었다. 새 PHH 내 별도 assay의 후보이며 HCC 반응이나 독립 cross-assay 근거로 집계하지 않는다. 공식 이용 조건은 CC BY-SA 2.1 Japan 및 아카이브 별도 출처 표기다. [기존 감사](../r19_plan/T4_CROSS_ASSAY_RESULTS.md) · [공식 다운로드](https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html) · [공식 이용 조건](https://dbarchive.biosciencedbc.jp/en/open-tggates/lic.html) |
| DILImap PHH | 저자 문서에 학습용 WikiPathways signature와 검증용 raw·gene·pathway 자료가 공개로 안내된다. 학습용 원시 count 및 gene signature는 저자 신청 대상이다. GEO GSE308567과 패키지 공개 범위·조건·농도 키를 파일별 확인한 후 별도 PHH 프로그램 연구에 쓸 수 있다. 기존 G1에 이 accession의 직접 보유는 확인하지 못했다. 처리 후 서명을 받는 저자 DILI 예측과 투약 전 RNA+구조의 HCC 예측은 입력과 정답이 다르다. [저자 데이터 문서](https://github.com/Cellarity/DILImap/blob/main/docs/datasets.rst) · [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12615732/) |

이 감사에서 학습·채점이나 제한 자료 신청을 수행하지 않았다. 두 오가노이드 자료는 LIMORE 공개 대체 축과 별도의 접근 후보로 기록했다.
