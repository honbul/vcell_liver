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
| S1/S5/S6–S8 | 원논문이 연결한 PMC 다운로드가 XLSX 대신 HTML을 반환. 일반 브라우저에서도 인증 화면 이후 다운로드 이벤트가 없었다. AAAS 요청 HTTP 403 |
| 분석용 사용 조건 | NODE의 두 파일이 `Public`이고 논문이 재분석용 matrix를 NODE에 안내한다. 파일별 재배포 라이선스는 확인되지 않았다. 조직·원시 서열 파일 자체는 평가 입력에서 제외한다. |
| 적격 교집합 | 무결성과 S1·S6 연결을 확인한 오가노이드 0개. RNA 시트 범위 `A1:BN23123`은 헤더 수준에서만 읽혔다. |

두 워크북 원본과 응답 헤더는 G1의 `vc_build_v1/licob-v0plus/`에 보관했다. 손상 파일의 일부 RNA를 완전한 코호트로 채점하면 유전자 누락과 접근 경로를 평가 결과에 섞게 된다. [원논문·보충자료](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949980/) · [NODE 프로젝트](https://www.biosino.org/node/project/detail/OEP003191) · [파일](https://www.biosino.org/node/api/download/node/data/public/OED00796358) · [공식 MD5](https://www.biosino.org/node/api/download/node/data/public/OED00796361)

### 대체 공개 3D 자료

| 자료 | 실제 공개 범위와 적용 가능성 |
|---|---|
| GSE182593 | G1에서 GEO 공식 FTP 인덱스 및 FPKM/count 파일 링크 확인. 4환자의 sorafenib 민감·획득 내성 8배양 상태와 3회 라이브러리씩 24개 RNA다. 이 설계는 내성 비교에 쓸 수 있고, 독립 환자의 투여 전 다약물 계통 지표에는 부족하다. [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182593) |
| HRA004348 | 21개 간담도 오가노이드·64개 약물의 S4 및 run 메타데이터 확인. 무인증으로 연결 가능한 기저 RNA 행렬은 확인되지 않았다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897507/) · [GSA-Human](https://ngdc.cncb.ac.cn/gsa-human/browse/HRA004348) |
| GSE191224 | Lenvatinib의 PDX 내성·민감 비교가 공개됐다. 한 약물의 획득 내성 실험으로 다약물 3D 외부 평가와 구분한다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10838288/) |
| Wong et al. | 100약물 오가노이드 연구. 논문이 보충표를 제공하며 식별자 단위의 투여 전 전사체·연속 반응 결합은 추가 확인 대상이다. [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12205652/) |
| Nuciforo et al. | HCC 환자 유래 3D 약물 스크린. 원논문 기준 초기 1,642개 화합물 스크린은 4개 대표 오가노이드이며, 기저 RNA와의 동일 표본 결합은 추가 확인 대상이다. [원논문](https://doi.org/10.1016/j.celrep.2026.117895) |
