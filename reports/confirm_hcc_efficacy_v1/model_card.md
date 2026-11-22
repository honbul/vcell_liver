# 정정 구조 커널 연구용 모델 카드

`single-agent-structured-68-confirm-20261123`은 고정한 정정 구조 커널의 CPU 패키지다. 연구용 2D 세포주 후보 선별 도구이며, 환자 임상 치료·용량·3D 무보정 효능을 보장하지 않는다. 결과의 사용 범위는 투약 전 bulk RNA와 68개 정확 parent의 연구용 스크린 감수성 점수다. 비교 평가에서 지정한 단일 기준은 정정 구조 커널이며 사용 범위는 약물 내 감수성 비교용 연구 기준이다.

- 식별자와 전처리. 정확한 부모 InChIKey 또는 해당하는 표준화 SMILES를 받는다. 유전자별 TPM, log₂(TPM+1), 또는 카운트를 명시한다. `bulk_counts`에서는 유전자 길이 TSV를 주면 길이 보정 TPM을 쓰고 길이 파일을 생략하면 CPM으로 환산한다. 결측 입력은 학습 중앙값으로 채운다. 전체 17,871개 및 선택 768개 유전자의 관측 범위가 각각 85% 이상이고 양수 발현이 있어야 한다.
- 거절 상태. 음수·무한대는 처리 전에 입력 오류를 반환한다. 전체 및 선택 유전자 부족 또는 모든 발현 0은 `insufficient_rna_coverage`, 미등록 parent는 `compound_outside_training_panel`과 빈 점수를 반환한다.
- 모형. 기저 RNA 순위, 공개 경로·전사인자, 적합 집단에서 추정한 공개 보조 특성, exact parent 화학구조·표적을 정정 소스에서만 적합했다. 출력은 학습 스크린에 정렬한 높은 값 방향의 점수, 약물 평균과 예측 잔차다. 이 잔차는 개별 예측에서 약물 anchor를 뺀 값이며 검증 보고서의 이중 중심화 잔차 정의는 별도로 적용한다.
- 2D 검증. LIMORE 50계통 × 68약물 동결 예측을 단일 모델로 재생했다. 동결 대비 최대오차 2.22e-16, 단일 파일과 배치 간 최대오차 1.86e-07, 전체 50개 직접 커널 비교 최대오차 2.01e-07였다. CPU `Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz` 4스레드에서 CLI 중앙값 13.968초, warm kernel 중앙값 0.0109초였다.
- 무결성. 학습 입력 65,678행의 SHA-256 `1edd32ef38be60a742efe0dcc376e15910623c1da8cc27d923291bee8a4a38cc`, 배포 체크포인트 `2cb27350315805f3f1872b6a8a172bff53da98ec4892b461fa8fbd0beca19136`. [재생 영수증](package_replay_receipt.json) · [배포 매니페스트](package_release.json)

## G1에서 오프라인 실행

RNA 파일은 `sample_id` 열과 유전자별 열을 갖는 TSV거나 `sample_id`, `gene`, `value` 열의 long TSV다. 약물 파일은 `compound_id` 또는 `smiles` 열의 TSV다. `--compounds`를 생략하면 68개를 전부 예측한다.

```bash
study=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
worker="$study/confirm_20261123/worker1_eff"
export PYTHONPATH="$worker/code/src:$study/vc_build_v1/v0plus-model-deps:$study/vc_build_v1/function/prepare-deps:$study/vcell_liver/vcell_r12_runtime_cu126"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -m vc_build.v0plus_cli predict   --package "$worker/task4/package" --rna rna.tsv --scale bulk_log2_tpm_plus_1   --compounds compounds.tsv --out predicted.tsv
```
