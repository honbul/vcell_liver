# 02 공통 Norman 데이터에서 기준선과 튜닝 비교

논문별 재현과 별개로, [공통 분할](../docs/COMMON_NORMAN_DATA.md)의 같은 K562 세포·개입·평가용 대조군 32개·출력 유전자 4,547개를 사용한다. 동일한 전체 미세조정 scGPT에서 D0는 고정 대조군 배정, D1은 회차별 재샘플링이다. 이 비교에서 D1은 검증에서 가장 가까웠던 변경이고, 홀드아웃에서 개선되지 않았다.

아래는 **전체 학습을 다시 수행하는 선택 실행**이다. [01](01_paper_reproduction.md)에서 준비한 **별도 G1 체크아웃**에서 두 논문 작업과 handoff가 성공한 뒤 진행한다. 기존 연구 실행 경로와 제출 기록을 재사용하면 새 훈련이 시작되지 않으므로 새 체크아웃을 사용한다.

```bash
dataset="$PWD/assets/common/C0-Norman-v1-s26"
study="$PWD/assets/study/paper"
paper="$study/paper_status.json"
test -s "$paper"
sbatch --parsable experiments/common_prepare.sbatch --split-seed 26 --output "$dataset"
```

분할 준비 작업이 성공하면 C0와 M을 순서대로 실행한다. 다음 블록의 첫 번째 명령은 실행할 작업만 출력한다. `--submit`을 붙이면 연구와 같은 Slurm 진입점으로 제출하며, 체크아웃의 `logs/common_submissions/`에 등록 번호를 남긴다.

```bash
.venvs/gears/bin/python src/common_submit.py C0 --dataset "$dataset" --workspace "$study" --paper-status "$paper"
.venvs/gears/bin/python src/common_submit.py C0 --dataset "$dataset" --workspace "$study" --paper-status "$paper" --submit
# 위에서 나온 C0 작업 3개의 성공을 확인하고 실행:
.venvs/gears/bin/python src/common_submit.py M --dataset "$dataset" --workspace "$study" --paper-status "$paper" --submit
# 위에서 나온 M 작업 4개의 성공을 확인하고 실행:
sbatch --parsable experiments/common_model_handoff.sbatch "$study" "$dataset"
```

C0는 단순 기준선 4개(무변화, 학습 평균, 가산, GO ridge)와 GEARS 및 전체 미세조정 scGPT(M0)다. M은 같은 검증 데이터와 학습 예산에서 디코더·개입 모듈만 학습(M1), 마지막 두 블록도 학습(M2), QKV LoRA(M3), 동일 구조 무작위 초기화를 비교한다. 원시 출력은 처리 후 유전자 발현 벡터다. [검증 결과와 비용](../reports/common/F/study/tables/tuning_results.tsv)과 [모델 선택](../reports/common/F/study/model_selection.json)을 볼 수 있다.

모델 선택 작업이 성공하면 선택 결과에 따른 전체 확인 절차를 제출한다. D에서 D1과 D2를 고정된 모델로 학습한다. 그 뒤 검증으로 확인 조건을 정하고, 훈련 시드를 반복해 체크포인트를 고정한 뒤에만 테스트를 계산한다. 선정과 테스트 사이에 수동으로 시험 지표를 읽거나 후보를 바꾸지 않는다.

```bash
test -s "$study/model_selection.json"
sbatch --parsable experiments/common_pipeline.sbatch data \
  --primary-workspace "$study" --primary-dataset "$dataset" --split 26
```

이 제출은 하위 작업들을 순차로 등록한다. `logs/common_submissions/`와 `logs/common_pipeline_*.log`에서 이어지는 번호를 확인한다. 최종 성공은 `common_evidence.json`, `tables/table_receipt.json`, 각 예측 점수와 집계표까지 확인한다. [실제 실행 기록](../manifests/slurm/study_submission.tsv), [확정한 체크포인트](../reports/common/F/study/frozen_test.json), [D0와 D1 비교](../reports/common/F/study/tables/factorial_effects.tsv)를 참조한다. 추가 분할은 [사전 규칙](../protocols/common/selection-v1.json)에 따라 판단한다. 본 연구에서는 조건을 충족하지 않았다.
