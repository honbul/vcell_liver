# 튜토리얼 검증 기록

2026-11-06 KST, G1 연구 체크아웃에서 검증했다. 아래 상대 경로의 기준은 G1의 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_reproduction_first_20261105`다.

| 확인한 작업 | 검증 결과 | G1 기록 |
|---|---|---|
| 노트북을 새 Python 3 커널에서 `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3`로 실행 | 코드 셀 4개가 순서대로 실행됐다. 인공 입력을 실제 공통 평가기에 넣었을 때 완벽한 예측의 ΔRMSE는 0, 무변화 예측은 2였다. | `assets/checks/tutorial.nJzQgU/tutorials/executed.ipynb` |
| [01의 소규모 학습](../tutorials/01_paper_reproduction.md) | Slurm 17866. 학습 쌍 두 개와 유전자 128개로 표적 포함, LoRA 12곳의 gradient와 업데이트, 동결 파라미터 유지 및 모델 저장·복원 후 예측 일치를 확인했다. | `logs/common_lora_check_17866.log`, `assets/checks/tutorial.nJzQgU/lora/result.json` |
| 논문 학습과 예측의 엄격한 handoff 재실행 | Slurm 17865. 원 전체 실행의 20회×5 GEARS와 15회 scGPT 체크포인트 기록, 최종 모델·예측·분할의 해시를 확인했다. 새 원 지표 대비표와 상태 JSON은 각각 Git의 [대비표](paper/reproduction_comparison.tsv), [상태](paper/paper_status.json)와 바이트 단위로 일치했다. | `logs/paper_handoff_17865.log`, `assets/checks/tutorial.nJzQgU/paper/` |
| [02의 C0·M 제출 명령](../tutorials/02_common_tuning.md) 미리 보기와 최종 집계 | 제출 옵션 없이 C0 세 작업과 M 네 작업의 인자를 확인했다. 기존에 고정한 체크포인트와 예측 해시를 `src/common_effects.py`로 재검증하고, 세 시드의 짝지은 효과를 다시 집계했다. 새 JSON은 [기존 요약](common/F/study/confirmation_summary.json)과 일치했다. | `assets/checks/tutorial.nJzQgU/confirmation_summary.json` |

전체 학습은 [원 실행 장부](../manifests/slurm/study_submission.tsv)와 [논문 재현 튜토리얼의 근거](../tutorials/01_paper_reproduction.md#실제-전체-실행-근거)에 연결했다. 위 검증에서 전체 학습을 중복 제출하지 않았다. HCC 검토는 [자료 적격성 판정](HCC_DATA_FEASIBILITY.md)에서 종료했다.

| 검증 파일 | SHA-256 |
|---|---|
| `tutorials/inspect_results.ipynb` | `ae9909b083ece428b32054fd21558206880bbd2a82b84d4a5bcdae0b0511b7f9` |
| 실행된 `executed.ipynb` | `0a6caeb5e8be253de60acab247317e9b3d241f58dff41da5e17d7ab7ee288e8b` |
| LoRA smoke `result.json` | `41b1d264f53813056afae2e6450fb29a5416c1391c2357a0202935cc63cce33e` |
| 재생성한 `reproduction_comparison.tsv` | `0ebfebaaad850a93892206136590bb18bc352ea20452360ebb364cb4ecc5f72a` |
| 재생성한 `paper_status.json` | `ae551271b3e2bc652e0d1af7080bc6ab0ae91bd983f900212b007c92b6b01c66` |
| 재계산한 `confirmation_summary.json` | `86db02be16d828ee29124f2fe75abc3a656d098b853729b8de86fed1929079fe` |
