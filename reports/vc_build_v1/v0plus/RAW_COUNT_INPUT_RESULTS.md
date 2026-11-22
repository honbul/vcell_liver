# V0+ 원발 HCC 원시 카운트 입력 검증

**유전자 길이를 적용한 카운트 입력은 사전 입력 재현 기준을 모두 통과했다.** TCGA-LIHC 원발 종양 16명의 같은 aliquot에서 GDC 공개 TPM을 기준으로 고정된 구조 커널의 68약물 점수를 비교했다. 세 입력에서 각각 1,088쌍이 예측됐다. [사전 명세](RAW_COUNT_INPUT_PROTOCOL.md) · [CLI 실행과 입력 형식](STRUCTURED_RELEASE_MODEL_CARD.md)

| 공개 TPM과의 일치 | 원시 카운트 → 기존 CPM | 원시 카운트 + exon 길이 → TPM |
|---|---:|---:|
| 종양 내 모델 유전자 순위 Spearman 중앙값 | .97093 | .999999998 |
| 종양 내 68약물 순위 Spearman 중앙값 | .97244 | 1.00000 |
| 약물 내 16명 종양 순위 Spearman 중앙값 | .98382 | 1.00000 |
| 상위 5약물 평균 일치 | 91.25% | 100% |
| 점수 평균 절대차 | .0143898 | .00000112 |
| 점수 최대 절대차 | .0690514 | .00000568 |

길이 보정과 CPM의 점수 MAE 차이는 −.014389였다. 16명 짝지은 재표집 95% 구간 [−.014944, −.013832]의 상한은 0 미만이었다. 길이 보정에서는 16명 모두 상위 5약물 구성이 일치했다. 모든 입력에서 전체 모델 유전자 최소 커버리지는 99.84%, 선택 유전자 최소 커버리지는 99.74%였다. [결과 영수증](raw_count_input_receipt.json)

60,660개 Ensembl 행에서 고유한 유전자 심볼 59,317개를 유지했고 모두에 동일 GENCODE v36 주석으로 exon 합집합 길이를 구했다. 파일 UUID와 해시는 G1 원본 매니페스트에 기록했다. GDC 공개 발현으로만 실험했으며, 외부 파일은 환자 반응이나 치료 기록과 연결하지 않았다. 이번 수치는 같은 읽기 데이터에서의 입력 형식 일치다. 다른 파이프라인이나 배치 간 호환성과 환자별 임상 성능은 이 평가 범위 밖이다. [GDC STAR Counts](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/) · [GDC GENCODE 참조 주석](https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files)

명세와 코드는 `4ca2dff`에서 고정됐다. Slurm `33704`는 CPU에서 1분 38초, 종료 코드 0으로 완료됐다. 입력별 예측 파일의 SHA-256을 재검증했다. 결과 영수증의 SHA-256은 `498ccd2d1984681afe033ca4497944237fb7ec6ef4f3c141ab0e07f4e9574a4b`다. [작업 원장](RUN_LEDGER.tsv)
