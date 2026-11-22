# VCB-02/03 상태 예측 개발 평가

**성인 HCC 상태 출력과 세포주별 반응은 승격 기준 미통과.** 실제 JHH-5/JHH-7 처리 RNA에서 측정한 24시간 vehicle 대비 변화량으로 상태 프로그램 모델 3개를 학습·동결했고, 별도로 채점했다. 기저 RNA는 별도 DepMap의 log2(TPM+1) profile이다. 실제 처리 실험의 투약 직전 RNA는 확보되지 않았다. 평가 8축은 각 적합 집합에서만 만든 데이터 기반 방향이며, 각 모델의 별도 좌표계다. 예제의 8개 숫자는 분자 경로나 분류 확률에 대응하지 않는 연속 RNA 대비 좌표다. [사전 명세](PROTOCOL.md) · [지원](cmap/support.tsv) · [동결](cmap/freeze.json) · [채점](cmap/score_receipt.json)

| 평가 | 공통 구조 지원 | RNA 상호작용 macro RMSE | 가장 강한 단순 비교군 | 상대 개선, 짝 95% 구간 |
|---|---:|---:|---|---:|
| C, JHH-5 전부 제외 후 적합 | 444개 정확 구조, 447웰, 442 connectivity군 | 0.6512 | 공통 약물 반응 × 0.5, 0.6272 | −3.84% [−8.70%, +0.57%] |
| C, JHH-7 전부 제외 후 적합 | 98개 정확 구조, 98웰 | 0.8495 | 공통 약물 반응, 0.8553 | +0.69% [−0.55%, +1.71%] |
| C-both, 두 세포 제외 후 동일 약물의 세포 차이 | 95개 정확 구조, 192웰 | 0.8144 | 기저 RNA 가산, 0.7973 | −2.14% [−4.66%, −0.15%] |

JHH-5 변화 방향의 cosine은 상호작용 0.242, 공통 반응 × 0.5 0.193이었다. JHH-7은 0.514와 0.522, 95약물의 세포 차이는 0.060과 0.089였다. JHH-7에서는 RNA 배정을 적합 시점에 바꾼 대조도 +1.12%의 점추정을 보였고 구간은 0을 포함했다. JHH-5는 위치와 플레이트가 모두 다른 반복이 3개 정확 구조·6방향 웰이었다. 이 반복의 방향 cosine은 0.101, 변화 예측 landmark RMSE는 0.615로 무변화 0.461보다 컸다. JHH-7은 1개 플레이트로 반복이 없다. 평가 화합물 connectivity를 재표집한 95% 구간은 관측한 두 JHH 세포주에 조건부다. 개별 질의의 예측구간과 용도별 보정은 이 상태 출력에서 검증하지 않았다. [모든 방법의 결과](cmap/metrics.tsv) · [반복 결과](cmap/repeat.tsv)

연결 전 JHH-5의 615웰에서 두 유기 성분으로 등록된 clonixin-lysinate 1웰을 단일 화합물 모형에서 제외했다. 남은 614웰은 611개 정확 구조, 608개 connectivity군이다. JHH-7은 99웰·99구조였다. 두 세포주의 CMap 대상 웰은 R19 감사에서 확인한 B1 노출 측정이다. 제한된 자료는 최종 검증에서 사용하지 않았다. R18 예약 처리·차량·해당 별칭 차단은 고정된 R19 메타데이터 허용 명부에 근거했다. 새 점수는 이 허용 명부와 동일한 Level 3 물리 좌표에만 적용했다. HUH7 LINCS는 예약 차단 후 적격 차량 대조가 결측이다. Tahoe에는 독립 성인 HCC 세포로 인정할 SNU-423만 있고, HepG2/C3A와 SK-HEP-1은 각각 hepatoblastoma 파생과 내피 맥락이다. [대상 원본 지원](../../r19_plan/HCC_LINCS_24H_SUPPORT.md) · [시간 및 R18 제외 감사](../../r19_plan/HCC_TIME_LINCS_RESULTS.md) · [DepMap 25Q3](https://depmap.org/portal/data_page/) · [Cellosaurus SK-HEP-1](https://www.cellosaurus.org/CVCL_0525) · [Tahoe 반복 감사](../../r19_plan/TAHOE_HCC_REPEAT_RESULTS.md)

### HepaRG 별도 개발 arm

GSE284321 10 µM·24시간 TempO-Seq의 log1p(CP10k) vehicle 대비 변화량에서 plate와 connectivity를 함께 격리했다. 적합 798개 군·2,186웰·73플레이트, 평가 77개 군·92웰·14플레이트다. 한 축만 평가 역할인 967웰은 양쪽에서 제외했다. 구조 커널 RMSE 1.7429, 적합 평균 1.7000으로 상대 개선은 **−2.52% [−4.70%, −0.61%]**였다. 프로그램 방향 일치는 각각 49.3%·51.0%였다. 단일 HepaRG 맥락이며 독립 0 h RNA는 없다. R14 어댑터가 공개 원 count로 산출한 유전자 매핑·QC·차량 대비를 재사용하고 새 적합에서 프로그램과 회귀기를 학습했다. [사전 명세](HEPARG_PROTOCOL.md) · [전 결과](heparg/metrics.tsv) · [원 연구의 EPA 자료 등록](https://catalog.data.gov/dataset/integrating-transcriptomic-and-targeted-new-approach-methodologies-into-a-tiered-framework)

### 실행 및 후속 통합 자료

G1의 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/state` 아래에 인덱스·체크포인트·예측·예제 요청과 결과가 있다. CMap 작업은 인덱스 32296, 예측 32298, 동결 32299, 채점 32300, 개발 부가 라벨 내보내기 32301이다. 첫 인덱스 제출 32295는 Python 환경의 h5py 결측으로 종료됐고 32296에서 기존 공용 의존성 경로를 사용해 복구했다. HepaRG 작업은 예측 32307, 동결 32308, 채점 32309다. 본 작업은 CPU만 사용했다. G1 체크포인트와 기저 RNA+구조 예제 요청을 새 Python 프로세스에서 실행해 8개 프로그램 출력을 재현했다. G1에서 상태 전용 회귀 테스트 5개를 통과했다. [실행 원장](jobs.tsv)

`cmap-state-v1/auxiliary/`는 행 순서가 고정된 8,214웰 × 978 Entrez 유전자의 원 관측 대비와 별도 basal 입력을 담는다. 713웰은 JHH 성인 HCC이고 나머지는 비간계열 6개 학습 맥락이다. LIMORE에 정확 연결되는 성인 HCC 맥락은 JHH-5와 JHH-7이며 두 세포 각각에 원 AA 88조건이 있다. 양쪽 자료에서 full InChIKey로 일치하는 구조는 JHH-5 20개, JHH-7 17개, 합집합 21개다. L1000 978개 Entrez 중 976개가 해당 LIMORE protein-coding Ensembl 축과 일대일 매핑된다. 이 교차 자료는 연구소와 endpoint가 다르며 24시간 RNA와 72시간 기능을 같은 배양에서 측정한 대응 자료가 아니다. VCB-04는 적합 fold에 포함된 lineage의 상태 라벨만 선택할 수 있다. [부가 라벨 영수증](cmap/auxiliary_receipt.json) · [기저 샘플 명부](cmap/basal_identity.tsv)

실험용 체크포인트와 예제는 G1 실행 경로의 `cmap-state-v1/C-JHH5`에 있다. 같은 디렉터리의 `checkpoint.joblib`과 `demo_request.json`을 `python -m src.vc_build.state_eval apply --checkpoint <checkpoint> --request <request>`에 전달하면 된다. G1 배치 파일의 동일 Python 및 의존성 경로를 사용한다. 출력은 승격 미통과 상태를 지닌 연구용 개발 출력이다. [재현 배치 명세](../../../experiments/vc_build_v1/state.sbatch)

원자료와 체크포인트는 소스 이용 조건에 따라 G1 내부에 보관했다. CMap Beta의 CLUE 이용 조건은 학술·비영리 연구 사용과 원자료 재배포 제한을 명시한다. DepMap 25Q3는 Portal 이용 조건으로 배포됐다. Git에는 코드, 사전 명세와 집계 근거만 포함한다. [CLUE 이용 조건](https://clue.io/releases/data-dashboard) · [DepMap 공식 라이선스 답변](https://forum.depmap.org/t/ccle-depmap-license-clarification/4686/2)
