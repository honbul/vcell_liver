# 실행 튜토리얼

저장된 수치와 근거를 둘러보려면 [결과 탐색 노트북](inspect_results.ipynb)을 연다. 노트북은 작은 TSV·JSON 입력만 읽으며, GPU·Slurm·원 데이터가 없어도 실행된다.

| 하고 싶은 일 | 시작점 | 실행 범위 |
|---|---|---|
| GEARS Fig. 2e–f와 scGPT Fig. 3a 재현 | [01 논문 재현](01_paper_reproduction.md) | 두 표본 `smoke` 또는 명시적인 `full_reproduce` |
| 공통 조건에서 기본 모델과 튜닝 비교 | [02 공통 데이터 튜닝](02_common_tuning.md) | 분할·기준선·검증 선택·고정 홀드아웃 |
| 실제 간암 개입 데이터를 사용할 수 있는지 확인 | [03 간 관련 적용](03_liver_application.md) | 후보 3건의 적격성 판단; 이번 연구에서는 학습 차단 |

모든 Slurm 예시는 G1의 해당 연구 체크아웃 루트에서 실행한다. 원본 및 실행 산출물은 그 서버에 있다. 비용이 드는 예시는 따로 표시했다. `sbatch`가 반환하는 번호는 제출 확인이다. `squeue`, `scontrol show job <번호>` 및 실제 산출물로 종료와 성공을 확인한다.

[실행 검증 기록](../reports/TUTORIAL_VERIFICATION.md)에서 새 커널 점검, 저장된 전체 실행의 재검증, G1 출력 위치를 확인할 수 있다.
