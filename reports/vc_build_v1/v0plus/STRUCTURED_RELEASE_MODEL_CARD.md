# V0+ 구조 커널 68약물 오프라인 패키지

`single-agent-structured-68`은 단일 약물로 정정한 GDSC2·CTRPv2·PRISM20Q2의 65,678행으로 적합한 구조 커널을 CPU에서 실행한다. 동일 학습표의 약물별 Ridge와 약물 평균도 선택할 수 있다. 기존 `single-agent-corrected-70`과 독립된 패키지다. 정확한 화합물 parent와 투약 전 bulk RNA에서 공개 스크린에 정렬한 감수성 점수를 계산한다. [배포 전 명세](STRUCTURED_RELEASE_PROTOCOL.md) · [70약물 패키지](V0PLUS_CORRECTED_MODEL_CARD.md)

## 실행

G1에서 폴더 전체를 사용한다. 폴더를 옮길 때는 체크포인트와 약물 등록부를 함께 복사한다. 구조 커널은 약 70.3 MB이며 세 체크포인트를 포함한 폴더는 약 71.4 MB다.

```text
/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/v0plus-runs/structured-release-6d2c8c6/V0plus
```

저장소의 `src`와 PyTorch·NumPy·pandas·SciPy·scikit-learn·RDKit·joblib이 있는 환경에서 실행한다. 확인된 환경은 Python 3.10.12, PyTorch 2.8.0, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1, RDKit 2025.03.5, joblib 1.6.0이다.

```bash
PYTHONPATH=src python -m vc_build.v0plus_cli predict \
  --package /path/to/V0plus --rna expression.tsv --scale bulk_tpm \
  --compounds compounds_of_interest.tsv --out predicted.tsv
```

기본값은 `structured`다. `--model drug_ridge`와 `--model drug_mean`으로 대조군을 선택한다. 발현 TSV는 `sample_id`와 HGNC 유전자 열 또는 긴 형식의 `sample_id`, `gene`, `value`를 받는다. 원시 카운트에는 `bulk_counts`, 이미 로그 변환된 값에는 `bulk_log2_tpm_plus_1`을 쓴다. Ensembl ID를 쓸 때 `--gene-map`에 `input_id`, `gene_symbol` 두 열을 전달한다. 약물 파일은 `compound_id`에 정확한 parent InChIKey를 넣거나 `smiles`를 준다. 생략하면 68약물을 요청한다.

전체 17,871개 유전자와 보조 특징용 768개 고변이 유전자의 커버리지가 각각 85% 이상이어야 한다. 결측 허용 범위에서는 소스 적합 표본의 유전자별 중앙값으로 대체한 후 각 샘플의 유전자 순위를 계산한다. 예측에는 약물 평균과 개인화 편차를 함께 반환한다. 미등록 parent, 발현이 모두 0인 샘플, 커버리지 미달은 상태 열에 기록한다.

## 기존 LIMORE 50계통의 성적

| 모델 | 약물 내 Spearman | 약물 내 Pearson | 계통 내 Spearman | NDCG@5 | Precision@5 |
|---|---:|---:|---:|---:|---:|
| 약물 평균 | .000 | .000 | .530 | .710 | .264 |
| 약물별 Ridge | .243 | .272 | .513 | .732 | .276 |
| 구조 커널 | .327 | .380 | .547 | .739 | .280 |
| CellHit 비교군 | .104 | .154 | .553 | .749 | .296 |

동일 68약물·3,400쌍에서 구조 커널의 약물 내 상관이 더 높았고, CellHit의 계통 내 약물 순위와 상위 5개 지표가 더 높았다. 이 표는 기존 동결·별도 채점의 결과다. 배포 검사에서 LIMORE 정답을 다시 열지 않았다. 3D의 개인화 성능과 환자 치료 효과는 별개의 검증 대상이다. [원 채점·모델 비교](LIMORE_CORRECTED_RESULTS.md)

## 실제 CPU 입력 검증

| 확인 범위 | CLI 결과 | 관측 |
|---|---|---|
| LIMORE 50계통·68약물 | 세 모델 각각 3,400개 예측 | 구조 커널의 기존 GPU 동결 예측 대비 최대 절대 차이 `2.48e-6`. Ridge와 약물 평균은 각각 `2.22e-16` |
| 단독 및 배치 예측 | 구조 커널 68개 모두 재현 | 한 샘플 단독 실행과 전체 배치 차이 최대 `8.94e-8` |
| 입력 검증 대조 | 모델별 136개 조합 보류 | 전체 발현이 0인 샘플과 유전자 80% 입력. 별도 미등록 parent 요청도 보류 |
| Wong 원발 HCC 조직 RNA 10개 | 62약물 요청 중 27약물·270개 예측, 미등록 350개 보류 | 전체 유전자 최소 99.27%, 선택 유전자 최소 99.09%. 반응 없이 실제 조직 입력 경로를 확인 |
| LICOB 가공 오가노이드 RNA 43개 | 76약물 요청에서 예측 0개 | 53약물의 정확한 parent가 등록돼 있다. 43개 발현 프로필 모두 음수가 포함돼 입력 검증에서 거절됐다. 별도로 계산한 전체 유전자 커버리지도 80.69%로 기준에 미달 |

LIMORE 구조 커널 재현 명령은 14.75초, Wong 조직 RNA는 14.38초였다. 전체 재현 작업에서 자식 CLI 최대 RSS는 866,800 KiB였다. LICOB의 이전 연구용 동결 평가는 소스와 공통인 14,421개 유전자로 모델을 재적합한 별도 평가다. 이 배포판의 17,871개 유전자 모델에는 해당 공개 가공 파일을 직접 넣을 수 없다. [CLI 재현 영수증](structured_release_replay_receipt.json) · [LICOB 공통 RNA 재적합](LICOB_WIDE_RNA_RESULTS.md)

`7323ac1`에서 명세를 고정했다. GPU 적합·패키지 `33530`과 CPU 재현 `33537`이 종료 코드 0으로 완료됐다. 최초 패키지 `33528`은 정정 학습 영수증에 없는 소스 집계 필드에서 중단되어 실측 학습 행에서 집계를 직접 계산했다. 최초 CPU 재현 `33531`은 LICOB 가공 발현의 음수 값에서 입력 오류로 중단됐다. 후속 검증은 해당 거절과 커버리지를 기록하고 같은 체크포인트를 그대로 사용했다. 배포 명세 SHA-256은 `a9e6582ae6c9a87cb92075d73ae1dc2deb3919a0876a01897237d597c2a7c754`다. [패키지 영수증](structured_release_package_receipt.json) · [작업 원장](RUN_LEDGER.tsv)
