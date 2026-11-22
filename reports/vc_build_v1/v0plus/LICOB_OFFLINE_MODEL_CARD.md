# V0+ LICOB 오가노이드 76약물 오프라인 패키지

`public-organoid-76`은 14,421개 유전자의 기저 bulk RNA와 등록 화합물을 입력으로 받는다. GDSC2·CTRPv2·PRISM20Q2의 1,186약물·670,830행에서 학습한 모델 여섯 개를 묶었다. 동일 등록키의 소스 관측이 있는 약물은 57개이며 구조 전이만 사용하는 약물은 19개다. 입체 지정이 일부 부족한 키는 출력에 그 위치 개수를 기록한다. 전체 76약물에서 기본 상호작용 모델의 이전 기술 성적은 약물 내 Spearman .094, Pearson .078, 공여자 내 Spearman .354, NDCG@5 .647, Precision@5 .137였다. 완전 입체 지정 미지원 약물 18개의 기존 승격 기준은 미통과였다. [기존 외부 평가와 신뢰구간](LICOB_COLD_RESULTS.md) · [재현 명세](LICOB_OFFLINE_PROTOCOL.md)

## 실행

G1에서 다음 폴더의 `release.json`, `compounds.tsv`, `models.joblib`을 함께 사용한다. 체크포인트는 58.1 MB다. 저장소 `src`와 PyTorch·SciPy·scikit-learn·RDKit·joblib이 있는 승인된 CPU 환경에서 실행한다. 모델 경로는 폴더에 상대적이며 체크포인트와 약물 등록부의 SHA-256을 로드 전에 확인한다.

```text
/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/v0plus-runs/licob-offline-86415cd/V0plus
```

```bash
PYTHONPATH=src python -m vc_build.v0plus_cli predict \
  --package /path/to/V0plus --rna expression.tsv --scale bulk_rank_preserving \
  --out predictions.tsv
```

공개 LICOB처럼 유전자 간 순서가 의미 있는 가공 RNA는 `bulk_rank_preserving`을 쓴다. 입력은 `sample_id`로 식별하고, HGNC 심볼 열 또는 `sample_id, gene, value`의 긴 표를 받는다. TPM은 `bulk_tpm`, log₂(TPM+1)은 `bulk_log2_tpm_plus_1`, 카운트는 `bulk_counts --gene-lengths gene-lengths.tsv`를 쓴다. 길이 표는 카운트와 같은 주석의 유전자 심볼과 양의 exon 길이 `gene`, `length_bp`를 담는다. 선택적인 `--gene-map`으로 Ensembl ID를 매핑할 수 있다. 샘플의 전체·선택 유전자 커버리지가 각각 85% 이상이어야 한다. 가공 순위 입력은 관측 유전자에 변동이 있어야 한다.

`--compounds`를 생략하면 76약물 전부를 계산한다. 요청 파일에는 `compound_id`의 parent InChIKey 또는 `smiles`를 준다. `--model`은 기본 `interaction_transfer` 외에 `grand_mean`, `chemical_mean`, `neighbor_transfer`, `chemical_kernel`, `molecular_kernel`을 선택한다. `training_support`는 `same_parent_key` 또는 `structure_transfer`이며 `nearest_training_tanimoto`와 `remaining_unspecified_stereo`를 함께 출력한다. `sensitivity_score`는 2D 스크린 정렬 점수다. 출력의 평균 기준과 개인화 편차는 연구용 약물 우선순위에 사용한다.

## CPU 재현 결과

| 범위 | 확인 결과 |
|---|---|
| 기존 LICOB 외부 동결 | 43개 오가노이드 × 76약물 × 6모델의 19,608개 예측 모두 재현. 전체 최대 절대 오차 8.94×10⁻⁷, 기본 모델 1.79×10⁻⁷ |
| 기본 모델 단독·일괄 입력 | 최대 절대 차이 1.79×10⁻⁷. 같은 샘플 76약물 전부 재현 |
| LICOB 추가 기저 RNA | HCCO42에서 76약물 예측 및 전체·선택 유전자 커버리지 100%. S6 AUC·S7 IC50·S8 Emax의 등록 ID 열에서 해당 모델의 행이 없었다 |
| 원발 HCC TCGA-LIHC 16명 | 같은 aliquot의 공개 TPM과 카운트+유전자 길이를 사용해 각각 1,216개 예측. 최대 절대 차이 3.28×10⁻⁶, Top-5 집합 16/16 일치 |
| 잘못된 입력 대조 | 80% 커버리지·상수 RNA에서 모델마다 152개 요청을 보류. 미등록 화합물도 보류. 기존 68약물 패키지의 가공 RNA 입력은 형식 검증에서 거절 |

S1과 공개 RNA에 연결된 HCC 모델은 44개다. 기존 반응 성적은 반응과 연결된 43개 오가노이드의 성인 공여자 38명으로 계산했다. HCCO42의 예측은 수치 재현 영수증만 남겼다. 앞서 공개한 LICOB 가공 RNA를 그대로 사용했고 새 입력의 원시값과 원 연구 전처리의 일치도는 이 회차의 평가 대상 밖이다. TCGA의 결과는 발현 입력 형식 재현이다. 기존 LICOB 반응값의 추가 채점은 수행하지 않았다. [기저 RNA 복구](LICOB_WIDE_RNA_RESULTS.md) · [44번째 모델 영수증](licob_offline_rna_only_receipt.json) · [전체 재현 영수증](licob_offline_replay_receipt.json)

`f263815`와 `86415cd`의 G1 GPU 패키지 `33853`, CPU 재현 `33854`를 실행했다. 추가 RNA는 `9c09196`의 CPU `33855`에서 처리했다. 패키지와 6개 모델, 추가 RNA, TCGA 출력의 해시 연결을 검증했다. 기본 모델의 43개 오가노이드 실행은 18.1초, 44번째 모델 단독 실행은 12.8초였고 전체 재현 시 자식 프로세스 최대 RSS는 889,160 KiB였다. 로컬 테스트 29개를 통과했다. [패키지 영수증](licob_offline_package_receipt.json) · [작업 원장](RUN_LEDGER.tsv)
