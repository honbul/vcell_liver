# V0+ 정정 70약물 오프라인 패키지

`single-agent-corrected-70`은 단일 약물로 정정한 GDSC2·CTRPv2·PRISM20Q2 66,678개 학습 행과 1,014개 모델을 사용한다. 총 70약물이 등록됐으며 apatinib과 ivosidenib을 포함한다. Yang의 3D 외부 평가에 포함된 것은 70약물 중 5개이고, 사전 비교 우위 기준은 미통과였다. 출력은 공개 약물 스크린에 맞춘 연구용 배양 샘플 감수성 점수다. [Yang 동결 성적](YANG_HCC_EXTERNAL_RESULTS.md) · [재현 명세](V0PLUS_CORRECTED_RELEASE_PROTOCOL.md)

세 체크포인트와 약물 등록부는 G1의 아래 경로에 있다. 폴더 전체를 복사해 상대 파일 경로를 유지한다. 모델을 실행할 환경에는 이 저장소의 `src`와 호환 라이브러리가 필요하다.

```text
/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/v0plus-runs/corrected-v0plus-bc00031/V0plus
```

```bash
PYTHONPATH=src python -m vc_build.v0plus_cli predict \
  --package /path/to/V0plus --rna expression.tsv --scale bulk_tpm \
  --compounds compounds_of_interest.tsv --out predicted.tsv
```

기본 모델은 `v0plus_inductive`다. `--model drug_ridge`와 `--model drug_mean`으로 비교군을 실행할 수 있다. `--compounds`를 생략하면 70약물 전체를 돌린다. 선택 파일은 `compound_id`에 정확한 parent InChIKey를 넣거나 `smiles`를 제공한다. 입력 발현은 `sample_id`와 HGNC 유전자 열의 TSV 또는 긴 형식 `sample_id`, `gene`, `value`를 쓴다. 측정값에 맞춰 `bulk_tpm`, `bulk_counts`, `bulk_log2_tpm_plus_1`을 선택한다. Ensembl ID를 쓰면 `--gene-map`으로 `input_id`, `gene_symbol` 두 열을 전달한다. 전체 축 17,332개 유전자 및 선택 유전자 커버리지의 최소 기준은 각각 85%다. `data_revision`과 점수의 앵커·개인화 편차가 결과에 기록된다.

재현 환경은 Python 3.10.12, PyTorch 2.8.0, scikit-learn 1.6.1, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, RDKit 2025.03.5, joblib 1.6.0이다. `33290`·`33291`에서 재적합한 세 방법 각각의 1,115개 Yang RNA 예측은 최초 동결본과 수치가 정확히 일치했다. `33292`가 파일별 SHA-256과 70약물 등록부를 묶었다. 패키지 생성과 CLI 재현은 당시의 외부 채점표와 예측을 참조하며 원래 AUC를 추가로 읽지 않았다. [배포 영수증](v0plus_corrected_release_receipt.json)

독립 CPU 작업 `33294`가 실제 RNA 세 샘플에서 각 모델의 210개 예측을 생성했다. 모델별 15개 기존 동결 예측과의 최대 절대 차이는 V0+ `2.21e-5`, Ridge `2.24e-8`, 약물 평균 `2.22e-16`이었다. 발현이 모두 0인 입력의 70개 조합과 미등록 약물 네 요청은 보류 상태로 반환했다. 최초 `33293`은 Ridge CPU 배치에 전체 재적합용 `1e-9`를 적용해 중단했다. 작은 배치의 float32 차이를 반영해 CPU Ridge의 허용치만 `1e-6`으로 분리하고 같은 체크포인트로 검증을 완료했다. [CPU 재현 영수증](v0plus_corrected_replay_receipt.json)
