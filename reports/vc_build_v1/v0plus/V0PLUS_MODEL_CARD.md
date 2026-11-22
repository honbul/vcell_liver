# V0+ 오프라인 RNA-seq 예측기

V0+는 연구용 HCC 배양 샘플의 투약 전 bulk RNA-seq와 정확히 일치하는 등록 약물 68개에서 소스 약물 스크린에 맞춘 감수성 점수를 돌려준다. 점수가 높을수록 예측 감수성이 크다. 검증 범위는 LIMORE 성인 HCC 50계통과 Wong 환자 유래 HCC 오가노이드 10개의 곡선 요약치다. 공개 평가에서 약물 내 환자 구별과 환자별 약물 순위의 동시 우위는 미확인이다. Wong 평가는 공통 유전자에서 소스만으로 재적합한 레시피의 전이 평가다. 아래 배포 체크포인트의 직접 CPU 재현 검사는 LIMORE RNA를 사용했다. [최종 외부 결과](STEP4_EXTERNAL_RESULTS.md) · [Wong 3D 성적](WONG_3D_EXTERNAL_RESULTS.md)

체크포인트는 G1의 아래 디렉터리에 있다. 디렉터리 전체를 복사하면 모델과 manifest의 상대 경로를 유지할 수 있다.

```text
/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/v0plus-runs/final-8e5d3f4/V0plus
```

의존 버전은 실행 환경의 Python 3.10, PyTorch 2.8, scikit-learn 1.6.1과 NumPy·pandas·RDKit·joblib이다. 입력 TSV는 `sample_id` 열과 HGNC 유전자별 수치 열을 갖거나, 긴 형식의 `sample_id`, `gene`, `value` 열을 갖는다. 입력 스케일에 따라 `bulk_tpm`, `bulk_counts`, `bulk_log2_tpm_plus_1` 중 하나를 지정한다. 카운트는 CPM 후 log 변환한다. Ensembl ID는 명시적인 `input_id`, `gene_symbol` 매핑 TSV를 전달한다. 등록 전체 유전자와 선택 유전자의 커버리지가 각각 85% 이상이면 누락값을 학습 중앙값으로 채운다. 출력에 두 커버리지를 함께 둔다. 약물 파일은 정확한 `compound_id`나 `smiles`를 사용하며 생략하면 68개 등록 패널을 실행한다.

```bash
PYTHONPATH=src python -m vc_build.v0plus_cli predict \
  --package /path/to/V0plus \
  --rna expression.tsv --scale bulk_tpm \
  --compounds candidates.tsv --out prediction.tsv
```

배포 체크포인트는 `v0plus_inductive`, `drug_ridge`, `drug_mean`이다. `--model drug_ridge`로 약물별 비교 모델을 실행할 수 있다. 결과에는 감수성 점수, 약물 평균 anchor, RNA에 따른 편차, 커버리지, 입력 지원 상태와 체크포인트·평가 해시가 들어간다.

Slurm `32469`에서 실제 LIMORE RNA 3개로 CPU 재현 검사를 수행했다. 등록 204개 조합을 반환했고 미등록 요청 3개에는 보류 상태를 반환했다. GPU 동결 예측과의 최대 절대 수치 차이는 0.000043이었다. CLI wall time 18.70초, 최대 RSS 771,188 KiB였다. [패키지 영수증](v0plus_release_receipt.json) · [재현 영수증](offline_replay_receipt.json)
