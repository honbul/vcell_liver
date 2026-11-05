# 01 논문 재현

[GEARS 계약](../protocols/reproduction/gears_norman.yaml)과 [scGPT 계약](../protocols/reproduction/scgpt_norman.yaml)의 공개 Norman 데이터, 논문 지표와 실험 설정을 사용한다. GEARS는 논문의 다섯 분할에서 각 20회 학습하고, scGPT는 저자 Norman 설정으로 최대 15회 학습한다. 원 저자의 비공개 자산 일부는 확보하지 못했으므로 수정 재현이다.

## smoke — 기존 G1 연구 체크아웃

연구의 [LoRA 실행 점검](../experiments/common_lora_check.sbatch)은 준비된 공통 데이터의 학습용 대조군–처리 세포 두 쌍과 128개 유전자만 사용한다. 실제 scGPT에서 표적 플래그, gradient와 업데이트, 저장·복원 결과를 확인한다. 이 점검은 논문 수치를 측정하지 않는다.

```bash
mkdir -p logs
check=$(mktemp -d "$PWD/assets/checks/tutorial.XXXXXX")
sbatch experiments/common_lora_check.sbatch \
  --dataset "$PWD/assets/common/C0-Norman-v1-s26" --output "$check/lora"
# 종료 후: "$check/lora/result.json"에서 smoke_verified 및 state_reload_exact 확인.
```

모델별 GPU 점검을 추가하려면 같은 데이터에 `experiments/common_runtime_check.sbatch gears` 또는 `... scgpt`를 사용하고, 위와 같이 `--dataset`과 새 `--output`을 준다. 학습 배치 한 개와 전체 유전자 패널의 출력 형태를 확인하며 검증·시험 점수는 계산하지 않는다. [GEARS 점검 기록](../reports/preflight/gears_common_gpu.json)과 [scGPT 점검 기록](../reports/preflight/scgpt_common_gpu.json)이 있다.

## full_reproduce — 명시적으로 선택할 때만

원 작업을 보존하려면 먼저 G1에서 별도 체크아웃을 만든다. 충분한 저장 공간을 확보한다. `AUTHOR_DOWNLOADS`에는 [파일 목록과 출처](../manifests/public_assets.tsv)에 맞춘 저자 파일만 둔다. G1의 기존 연구에서 검증해 둔 `assets/public` 경로도 이 인자로 사용할 수 있다. 가공한 학습 결과나 기존 모델을 입력으로 삼지 않는다.

```bash
# 기존 G1 연구 체크아웃에서, 아직 없는 경로를 지정한다.
git worktree add --detach /path/on/g1/norman-replay HEAD
cd /path/on/g1/norman-replay
mkdir -p logs
python3 src/prepare_upstream.py --fetch
AUTHOR_DOWNLOADS=/path/to/verified/author/files
python3 src/prepare_assets.py --source-dir "$AUTHOR_DOWNLOADS"
sbatch --parsable experiments/setup_env.sbatch
```

환경 작업 성공을 확인하고 아래 블록을 **한 번** 실행한다. GEARS A40 작업 배열과 scGPT A100 작업을 제출하고, 두 작업이 성공했을 때만 원 지표 대비표를 계산한다. 예약 디렉터리가 남아 있으면 중간에 끊겼는지 Slurm과 로그로 확인한다. 중복 제출하지 않는다.

```bash
(
  set -euo pipefail
  test -x .venvs/gears/bin/python
  test -x .venvs/scgpt/bin/python
  test -f assets/public/receipt.json
  test ! -e assets/runs/R1 && test ! -e assets/runs/R2 || exit 1
  mkdir logs/.full_reproduce
  r1=$(sbatch --parsable experiments/gears_norman.sbatch)
  printf 'R1-seed1\t%s\nR1-seeds2-5\t%s\n' "$r1" "$r1" > logs/paper_submission.tsv
  r2=$(sbatch --parsable experiments/scgpt_norman.sbatch)
  printf 'R2\t%s\n' "$r2" >> logs/paper_submission.tsv
  sbatch --parsable --dependency="afterok:${r1%%;*}:${r2%%;*}" \
    experiments/paper_handoff.sbatch "$PWD/assets/study/paper" "$PWD/logs/paper_submission.tsv"
)
```

완료 후 `assets/study/paper/reproduction_comparison.tsv`와 `paper_status.json`을 확인한다. 학습이 중단됐으면 [체크포인트 재시작 안내](../docs/SLURM_RESTART.md)에 따라 해당 시드만 재개한다. 필요하면 새 CPU handoff를 제출해 결과를 다시 검증한다.

## 실제 전체 실행 근거

연구에서는 같은 [GEARS Slurm 진입점](../experiments/gears_norman.sbatch)을 `--array=1`, `--array=2-5%2`로 나눠 실행했고, [scGPT 진입점](../experiments/scgpt_norman.sbatch)은 그대로 실행했다. [제출 기록](../manifests/slurm/study_submission.tsv)의 17663·17665·17664가 해당 전체 실행이며, 17672가 최종 파일과 로그를 검증한 handoff다.

| 실행 | G1 연구 체크아웃에 남은 파일 | Git에서 확인할 근거 |
|---|---|---|
| GEARS, 5개 시드 × 20회 | `logs/gears_fig2f_{17663,17665}_*.log`, `assets/runs/R1/R1-paper-public-v1-sN/{model/model.pt,prediction.npz,result.json}` | [시드 1 로그 검증](../reports/paper/R1-s1/execution_audit.json), [다섯 시드와 파일 해시](../reports/paper/paper_status.json) |
| scGPT, 15회 | `logs/scgpt_fig3a_17664.log`, `assets/runs/R2/R2-manuscript-public-v1/{best_model.pt,prediction.npz,result.json}` | [scGPT 전체 결과](../reports/paper/R2/result.json), [로그와 파일 해시](../reports/paper/paper_status.json) |

[원 수치 대비표](../reports/paper/reproduction_comparison.tsv)와 [독립 채점 및 불일치 진단](../reports/PAPER_REFERENCE_RESOLUTION.md)을 함께 본다. 노트북에서는 저장된 대비표를 탐색할 수 있다.
