# V0+ 1단계

학습 소스의 계통 그룹 991개를 330/331/330개로 보류한 3-fold 검증이다. CTRPv2 31,097행, GDSC2 24,839행, PRISM20Q2 9,749행에서 각각 자기 소스 원 AUC를 채점했다. 구조·RNA 회귀를 같은 설정으로 실행해 훈련 척도의 영향만 비교했다. LIMORE 50계통과 LICOB 반응은 이 표에 들어가지 않았다.

| 소스 | 약물 내 Spearman 기존 → 보정 | 약물 내 Pearson 기존 → 보정 | 세포 내 NDCG@5 기존 → 보정 | Precision@5 기존 → 보정 | 세포 내 Spearman 기존 → 보정 |
|---|---:|---:|---:|---:|---:|
| CTRPv2 | .232 → .425 | .258 → .469 | .916 → .880 | .566 → .571 | .733 → .724 |
| GDSC2 | .274 → .388 | .288 → .452 | .876 → .877 | .533 → .545 | .725 → .698 |
| PRISM20Q2 | .144 → .237 | .139 → .227 | .899 → .904 | .704 → .722 | .738 → .747 |

보정한 약물 평균의 약물 내 Spearman·Pearson은 모든 소스에서 0이다. 보정 회귀가 약물 평균보다 세포 내 Spearman이 높은 폭은 CTRPv2 +.043, GDSC2 +.062, PRISM +.002다. 기존과 보정의 직접 비교에서는 두 소스의 세포 내 Spearman이 낮아졌다. 환자별 약물 순위가 안정화되는지는 후속 모델의 양쪽 공동 지표로 확인한다. 기존 LIMORE의 Spearman 하락을 이 척도 결함 하나로 귀속할 근거는 아직 없다.

| 보정 앵커 | 약물 수 | fold당 같은 세포·약물 쌍 | 기울기 범위 |
|---|---:|---:|---:|
| CTRPv2 → GDSC2 | 28 | 6,767–6,833 | .04481–.05113 |
| PRISM20Q2 → GDSC2 | 6 | 1,215–1,254 | .84158–.90074 |

등록 커밋 `9a2e9b9`, G1 Slurm `32338`. 모델별 원표와 bootstrap은 [scale_audit.json](scale_audit.json)에 있다.

## LICOB 실제 다운로드와 무결성

| 검사 | 확인 값 |
|---|---|
| 원논문 등록 | 57명·65개 간암 오가노이드 중 HCC 44개, 전 연구 약물 패널 76개 |
| NODE 파일 | `OED00796358`, 공개 HTTP 200, 69,058,905바이트, 독립 요청 2개에서 같은 SHA-256 |
| 배포 측 MD5 | `3687652d53aba2b8cdac1c2ef59e59b7` |
| 실제 MD5 | `96e1a019b16f07ff6d6df16eb3aa16c5` |
| 실제 SHA-256 | `c40ef61dcb7453415fcf1fdbeb557016e1935c9097715dfb22d75225b6fbf69e` |
| ZIP 검사 | 워크북의 `G. RNA-seq (LICOB)`가 참조하는 `xl/worksheets/sheet8.xml`에서 압축 해제 오류. 다음 ZIP 로컬 헤더 위치 49,660,364에서도 오류 |
| S1/S5/S6–S8 | PMC는 HTML을 반환했고 AAAS에서 HTTP 403을 받았다. EMBL-EBI BioStudies의 복사본을 다운로드해 ZIP 검사 통과. SHA-256 `0e158e1129349e3e32dc6840104f92959b95788705edb9e191ffe47ac21fa906`. |
| ID 교집합 | S1의 HCC 44개가 `G. RNA-seq`의 헤더 44개와 일치한다. S6 AUC, S7 IC50, S8 Emax 각각 HCC 43개 × 76개에 3,268개 수치가 있다. S5 이름/alias 대조에서 기존 학습 68개 패널과 53개 후보가 일치했다. 구조 기반 최종 약물 매핑은 별도 검증이 필요하다. |
| 분석용 사용 조건 | NODE의 두 파일이 `Public`이고 논문이 재분석용 matrix를 NODE에 안내한다. 파일별 재배포 라이선스는 확인되지 않았다. 조직·원시 서열 파일 자체는 평가 입력에서 제외한다. |
| 적격 교집합 | RNA까지 검증한 오가노이드 0개. RNA 시트 `A1:BN23123`의 헤더는 65개이며, 원 시트는 저자가 z-score 정규화한 값으로 표시한다. 아카이브는 약 42–50 MB 위치에 0바이트 구간이 있다. paired-tissue RNA 시트도 로컬 ZIP 헤더가 없다. |

워크북과 응답 헤더는 G1의 `vc_build_v1/licob-v0plus/`에 보관했다. 손상 RNA를 제외했다. 전체 환자 패널로 유전자별 z-score를 만들었다면 자료의 원 변환을 확인해 transductive 조건에서 다룬다. 독립 단일 샘플 검증에는 변환 전의 기저 RNA가 필요하다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949980/) · [EMBL-EBI 보충자료](https://www.ebi.ac.uk/biostudies/studies/S-EPMC10949980) · [NODE 파일](https://www.biosino.org/node/api/download/node/data/public/OED00796358) · [공식 MD5](https://www.biosino.org/node/api/download/node/data/public/OED00796361)

### 대체 공개 3D 자료

| 자료 | 실제 공개 범위와 적용 가능성 |
|---|---|
| GSE182593 | G1에서 GEO 공식 FTP 인덱스 및 FPKM/count 파일 링크 확인. 4환자의 sorafenib 민감·획득 내성 8배양 상태와 3회 라이브러리씩 24개 RNA다. 이 설계는 내성 비교에 쓸 수 있고, 독립 환자의 투여 전 다약물 계통 지표에는 부족하다. [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182593) |
| GSE84073 / Broutier et al. | 출판사 S1 RPKM, S5 반응 XLSX를 실제 다운로드하고 ZIP 검사했다. 기저 RPKM과 약물 반응이 연결되는 독립 HCC는 HCC-1/HCC-3 2개, 29약물·배양물당 116회 측정이다. 약물 내 개인화의 최소 5환자 요건에 미달한다. [원논문과 보충 Dataset 1/5](https://www.nature.com/articles/nm.4438?error=cookies_not_supported) · [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84073) |
| HRA004348 | 21개 간담도 오가노이드·64개 약물의 S4 및 run 메타데이터 확인. 무인증으로 연결 가능한 기저 RNA 행렬은 확인되지 않았다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897507/) · [GSA-Human](https://ngdc.cncb.ac.cn/gsa-human/browse/HRA004348) |
| GSE191224 | Lenvatinib의 PDX 내성·민감 비교가 공개됐다. 한 약물의 획득 내성 실험으로 다약물 3D 외부 평가와 구분한다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10838288/) |
| Wong et al. | 100약물 오가노이드 연구. 논문이 보충표를 제공하며 식별자 단위의 투여 전 전사체·연속 반응 결합은 추가 확인 대상이다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12205652/) |
| Nuciforo et al. | 초기 1,642개 화합물은 4개 대표 오가노이드에서 스크린했다. 공개 S2–S5는 처리 약물 정보이며, 논문은 대응 분자 프로파일을 후속 단계로 기술한다. [원논문](https://doi.org/10.1016/j.celrep.2026.117895) |
| HRA006499 / Yang et al. | 399개 원발성 간암 오가노이드·144명·7개 약물 연구. GSA-Human의 accession을 찾았고 G1 요청이 1,014바이트의 브라우저 확인 HTML을 반환했다. 공개 파일에서 동일 단위의 기저 RNA와 반응 매트릭스까지 연결되는 경로는 미확인이다. [원논문](https://doi.org/10.1016/j.ccell.2024.03.004) · [GSA-Human](https://ngdc.cncb.ac.cn/gsa-human/browse/HRA006499) |
