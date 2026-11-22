# 공개 HCC 3D 외부 자료 재감사

## LICOB 배포본

공식 NODE 파일의 손상 구간 `bytes=41943040-50331647`을 표준 HTTP Range 요청으로 다시 받았다. 서버는 HTTP 200과 동일한 전체 파일 69,058,905바이트를 반환했다. SHA-256 `c40ef61dcb7453415fcf1fdbeb557016e1935c9097715dfb22d75225b6fbf69e` 및 8,388,608바이트 길이의 0 구간을 재확인했다. 출판 MD5 `3687652d53aba2b8cdac1c2ef59e59b7`와 수신 MD5 `96e1a019b16f07ff6d6df16eb3aa16c5`가 불일치한다.

원논문이 연결한 LICOB 웹 포털은 로컬과 G1에서 HTTP와 HTTPS 모두 403이었다. 미국 NCI의 ICPC 프로젝트 페이지도 동일한 NODE 프로젝트를 가리킨다. 이 감사에서 NODE 배포본의 검증된 HCC RNA·반응 교집합은 **0개**였다. 응답표의 **43개 HCC × 76약물**과 RNA 헤더 ID는 확보했다. 파일별 재배포 라이선스가 확인된 행렬은 없다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949980/) · [NODE 공개 객체](https://www.biosino.org/node/api/download/node/data/public/OED00796358) · [ICPC 등록](https://icpc.cancer.gov/projects/liver-cancer-organoids-for-precision-oncology/) · [1차 감사](STEP1_SCALE_AND_LICOB.md)

후속으로 저자 iLICOB 패키지의 별도 공개 RNA를 검증해 34모델·29명의 성인 HCC를 공식 S6과 연결하고 53약물을 채점했다. [LICOB 새 검증](LICOB_AUTHOR_RESULTS.md)

## 공개 대체 자료

| 자료 | 실제 다운로드 또는 대조 | RNA·반응 지원과 해석 |
|---|---|---|
| Hu et al. GSE90653 | GEO GPL570 행렬과 주석의 gzip 무결성 확인. 원논문 Figure 4c에 나온 HCC PDX 16개 모두 같은 모델명의 F2 비동결 GSM과 연결 | 16개 독립 PDX × sorafenib 1약물, 소스 RNA와 16,088개 공통 유전자. 그림 순위 기반 약물 내 Spearman 기술 평가. [원논문 PDF](https://lihonglab.github.io/assets/pdf/Intl%20Journal%20of%20Cancer%20-%202019%20-%20Hu%20-%20Establishment%20of%20a%20hepatocellular%20carcinoma%20patient%E2%80%90derived%20xenograft%20platform%20and%20its.pdf) · [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90653) |
| Zhang et al. 3DP-PLC / HRA019374 | 출판사 보충 ZIP 781,364바이트의 CRC 검증. S1의 HCC 분류와 S3·S13의 수치 ID를 대조 | 단일배양 HCC 51개 × 4 TKI의 AUC가 모두 있고 공배양 HCC 33개 × 4 TKI를 확인했다. S3. 표제의 54개는 전체 행 수와 다르다. 발현-반응을 함께 분석한 단일배양 HCC 11개를 본문이 기술한다. 보충 ZIP은 반응표, 유전자별 Cox 요약과 경로 요약을 싣고 있다. HRA019374에서 무인증 표본×유전자 행렬은 확보하지 못했다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13700430/) · [출판사 보충 ZIP](https://media.springernature.com/full/springer-static/esm/art%3A10.1186%2Fs12943-026-02751-1/MediaObjects/12943_2026_2751_MOESM2_ESM.zip) |
| Kim et al. | EMBL-EBI 보충 ZIP의 CRC 검증과 S1–S10 열·시트 점검 | S7에 HCC 오가노이드 18개 × 8약물 AUC·IC50. RNA 관련 표는 유전자 서명, 분석 요약 및 차등발현표다. 연구의 데이터 공개 문구는 교신저자 요청을 안내한다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13639540/) · [공개 보충표](https://www.ebi.ac.uk/biostudies/files/S-EPMC13639540/CTM2-16-e70853-s001.zip) |
| Wong et al. | BioStudies의 일반명 보충 파일에 타 논문 내용이 포함됐다. NCBI PMC 배포본의 XML·보충 파일 4개는 전부 해당 논문의 MD5와 일치했다. PDF와 DOCX 안의 저자 링크에서 23,252,930바이트 공개 XLSX를 로컬·G1에서 독립 수신해 SHA-256·CRC 확인 | S1의 성인 HCC 확인, S6 기저 TPM, S7 반응의 교집합은 10명이다. 전체 100약물 중 구조가 학습 등록부와 일치하는 27약물을 등록했다. 원논문 PMC 라이선스는 CC BY-NC-ND이며 별도 워크북의 재배포 라이선스는 확인되지 않았다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12205652/) · [NCBI 버전·MD5 메타데이터](https://pmc-oa-opendata.s3.amazonaws.com/PMC12205652.1/PMC12205652.1.json) · [공개 워크북](https://www.surgery.cuhk.edu.hk/downloads/JHep_Reports/Wong%20AM%20et%20al_JHep_Reports_Supplementary%20Tables_v1.xlsx) |
| Müller et al. GSE275443 | GEO 발현 표본 10개의 종과 실험 설계 확인 | 형질전환 마우스 HCC 오가노이드다. 사람 HCC 외부 확인에 적격인 표본은 0개다. [원논문](https://www.nature.com/articles/s41586-025-08585-z) · [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275443) |
| Zhu 연구진의 HCC PDX | PMC 공식 XML을 받은 뒤 논문이 안내한 EGA 등록과 DAC 대조 | Sorafenib·regorafenib 실측을 보고했다. PDX의 exome·RNA-seq 묶음은 EGA `EGAD00001005734`에 30시료로 등록되어 있고 `EGAC00001001387`의 접근 허가 대상이다. [원논문](https://doi.org/10.1002/hep.31096) · [EGA DAC](https://ega-archive.org/dacs/EGAC00001001387) |
| 환자 유래 PLC 오가노이드 단일세포 연구 | PMC 공식 XML의 데이터 공개 문구 확인 | HCC 5개를 포함한 11개 PLC 환자 오가노이드를 연구했다. 저자 요청을 통해 기저 발현 데이터를 제공한다고 명시한다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12992744/) |

출판사 ZIP SHA-256은 3DP-PLC `af264009d89c28e481f180f1330a2131c4a57b84989966cd528783881eac4b06`, Kim `c8fabfa454c453506292cf113a8889f1bf033af34c389e9adf3f21ab3f233598`다. Wong 워크북은 `724e0276c474c8afbdacb734c291f53e0399a77a4e473b5ae57751dd637841e7`다. [Wong 사전 명세](WONG_3D_PROTOCOL.md)

## 독립 PDX 기술 채점

| 방법 | Sorafenib 감수성 약물 내 Spearman | 환자 재표집 95% 구간 |
|---|---:|---:|
| 약물 평균 | .000 | [.000, .000] |
| 약물별 Ridge | −.274 | [−.745, .291] |
| V0+ inductive | −.324 | [−.787, .237] |

V0+와 Ridge의 차이는 −.050 [−.845, .779]이다. 공개 그림 순위에서 양의 전이 근거를 확보하지 못했다. LIMORE 2D sorafenib의 V0+ Spearman도 −.076이었다. PDX에서 측정한 것은 치료 4주 후 종양 성장 순위다. 이 채점은 1약물이며, 수치 표에서 계산하는 Pearson과 다약물 약물선별 지표의 지원은 없다. 공개 microarray에는 GEO 제출자의 다표본 quantile 정규화가 포함된다. 파일 단위 재배포 허가는 확인하지 않았다. [기술 평가 등록](PDX_TECHNICAL_PROTOCOL.md) · [등록 영수증](pdx_registration_receipt.json) · [동결 영수증](pdx_forecast_receipt.json) · [수치 원표](pdx_score.json)

새 GEO GSE343846의 공개 FPKM과 표본별 SOFT를 대조했다. RNA 6개가 모두 lenvatinib 처리 후에 채취됐으므로 투여 전 모델의 평가에서 제외했다. 앞서 확보한 Zhang·Kim 공개 반응표로 배양 조건과 기술 반복의 실측 일치도를 별도로 측정했고, 두 자료의 약물 관계를 예측에 추가한 보정은 LICOB·Wong 공동 기준에 미달했다. [후속 감사와 성적](HCC_3D_PUBLIC_RESPONSE_RESULTS.md)

## HCMI와 DepMap NextGen 메타데이터 확인

발행사 Broad NextGen Supplementary Table 1의 ZIP 무결성과 SHA-256 `41cb8e26e8d8109855f7d2a993e5b585ba4575679fb7b61e383c0a9c713e9262`를 확인했다. 공식 Model ID 314개 중 간·담도 관련 라벨은 담도계 3개였다. 조직형은 간내 담관암, 담관 내 유두상 종양, 담낭 내 유두상 종양 각각 하나였으며 HCC는 0개였다. 파일은 G1에 보관했다. [Broad NextGen 원 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC13581591/) · [발행사 메타데이터](https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs41586-026-10843-7/MediaObjects/41586_2026_10843_MOESM3_ESM.xlsx)

NCI GDC의 공개 cases API에서 HCMI-CMDC 등록 805명 중 `Hepatocellular carcinoma, NOS`는 1명이었다. `HCM-CSHL-0617-C22`의 3D 오가노이드와 원발 조직은 각각 공개 유전자 정량 파일 1개에 연결됐다. 같은 증례의 공개·제한 파일 87건에는 해당 모델의 약물반응 측정 표가 없었고, 공개 문헌과 카탈로그에서도 연결을 확인하지 못했다. 발현과 반응을 함께 가진 새 HCC 외부 평가에는 0건을 등록했다. [NCI GDC HCMI-CMDC](https://portal.gdc.cancer.gov/projects/HCMI-CMDC) · [HCMI 해당 모델](https://hcmi-searchable-catalog.nci.nih.gov/model/HCM-CSHL-0617-C22)
