# 피드백 5개 항목 실행 작업 분담 (2026-11-10)

계획: [FEEDBACK_PLAN_20261110.md](FEEDBACK_PLAN_20261110.md). 기준 커밋 `540ec93` (`research/reproduction-first-20261105`).

## 경로
- 로컬 브랜치: `research/feedback-20261110`
- G1 worktree: `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_feedback_20261110`
  - `assets/` → `/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_feedback_20261110` (새 산출물)
  - `prior_assets/` → 이전 연구 자산 (읽기 전용으로 취급)
  - `.venvs/` → 이전 연구 환경 (gears, scgpt). 패키지를 새로 설치하지 않는다.

## 작업 흐름과 소유 경로
| 흐름 | 계획 항목 | 코드 | 보고서 |
|---|---|---|---|
| F1-diag | 1-2 A1–A5 (CPU) | `src/feedback/f1_diag*`, `experiments/feedback/f1_diag*` | `reports/feedback/F1/diag/`, `reports/feedback/F1_DIAGNOSTICS.md` |
| F1-ablation | 1-3 A6–A9 (GPU, validation 전용) | `src/feedback/f1_ablation*`, `experiments/feedback/f1_ablation*`, 기존 `src/common_*.py`의 하위호환 옵션 | `reports/feedback/F1/ablation/`, `reports/feedback/F1_ABLATION.md` |
| F2 | 2 문헌·모델 검토 | – | `reports/feedback/F2_LITERATURE_REVIEW.md` |
| F3 | 3·4-genetic 4세포주 CRISPRi 계약·LOCO 기준선 | `src/feedback/f3_*`, `experiments/feedback/f3_*` | `reports/feedback/F3/`, `reports/feedback/F3_CROSS_CELL_LINE.md` |
| F4 | 4-chemical SNU-423·간 자료 조사 | `src/feedback/f4_*`, `experiments/feedback/f4_*` | `reports/feedback/F4/`, `reports/feedback/F4_LIVER_DATA_CARD.md` |
| F5 | 5 연구 방향 종합 | – | `reports/feedback/F5_DIRECTION.md` |

## 공통 규칙
- Norman 홀드아웃(테스트) 결과를 설정 선택에 사용하지 않는다. 새 설정 선택은 검증 분할에서만 한다.
- 무거운 계산은 Slurm(`cpu` 또는 `gpu` 파티션)으로 한다. 다른 사용자 작업과 원본 데이터(`/mnt/S1/sdata/pubdata/Virtual_Cell/*`)를 건드리지 않는다.
- 카운트 행렬, 가중치, 전체 예측은 G1 `assets/`에 둔다. Git에는 코드와 작은 요약(TSV/JSON/MD)만 남긴다.
- 실측, 계획, 미실행, 차단을 구분한다. Slurm job ID와 실제 자원을 기록한다.
