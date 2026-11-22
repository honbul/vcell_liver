# 동일 원발 HCC RNA의 카운트·TPM 재현 명세

동결한 `single-agent-structured-68`의 CPU CLI가 연구실 정수 카운트에서 환자별 상대 점수를 재현하는지 평가한다. 동일 aliquot에서 GDC가 공개한 unstranded counts와 unstranded TPM을 짝으로 사용한다. 입력의 기술적 일치가 평가 대상이다. 약물 반응·치료·생존 자료를 연결하지 않는다.

## 입력과 방법

- TCGA-LIHC 공개 STAR Counts 원발 종양 371개 파일 중 파일 UUID 오름차순으로 원발 종양 aliquot이 유일하게 매칭되고 참가자가 중복되지 않는 처음 16개를 선택한다. 입력 매니페스트와 각 파일의 SHA-256을 확인한다. GDC 공식 GENCODE v36의 유전자별 exon 구간 합집합 길이를 사용한다. 측정 TPM에서 길이를 역추정하지 않는다. [GDC STAR Counts](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/) · [GDC 참조 주석](https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files)
- 같은 원본의 Ensembl 유전자 행에서 유일하게 대응하는 gene symbol을 세 입력 모두에 사용한다. 여러 ID가 같은 심볼을 쓰는 경우 해당 심볼을 제외한다. 재구성 TPM의 분모는 길이를 산출할 수 있는 전체 입력 심볼을 사용하므로 GDC의 전체 행 분모와 다를 수 있다.
- 공개 TPM을 기준으로 기존 카운트 CPM과 카운트/염기 길이 기반 TPM을 각각 비교한다. 신규 CLI는 `--scale bulk_counts --gene-lengths`를 사용한다. 길이 옵션은 유효한 양수 길이가 있는 유전자로 계산하고 길이가 없는 유전자를 결측으로 넘긴다. 체크포인트, 화합물, 커버리지 규칙, 적합 중앙값을 공통으로 사용한다. 각 입력의 예측 결과를 저장하고 해시를 기록한다. [배포 체크포인트](STRUCTURED_RELEASE_MODEL_CARD.md)

## 사전 기준

세 입력에서 각각 16명 × 68약물 예측이 모두 있어야 한다. 길이 보정은 공개 TPM과 비교한 모델 유전자 내 순위 Spearman 중앙값이 .99 초과, 종양 내 약물 순위 중앙값이 .99 초과, 약물 내 종양 순위 중앙값이 .95 초과인 경우에 통과한다. 점수 MAE 차이(길이 보정 − CPM)의 환자 짝지은 4,000회 부트스트랩 95% 구간 상한도 0 미만이어야 한다. 평균 상위 5개 일치율과 최대 점수 절대차를 함께 보고한다.

이 평가는 GDC 공개 RNA에서 소프트웨어 입력 형식의 재현성에 한정된다. 다른 카운터·참조 주석·실험 배치에서의 재현성과 환자별 치료 예측은 별도 성능 범위다.
