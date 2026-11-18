# 원 R7 실행의 상태 기록 (STATUS_CURRENT_RUN)

원 R7 연구(`research/r7-evidence-gated-mechanism`)는 다른 세션이 진행 중이다. 이 문서는 후속 작업이 **관측할 수 있었던 사실만** 적는다. 완료·성공·실패를 추정하지 않는다. 원 실행의 worktree·job·출력·checkpoint·freeze에는 쓰지 않았다(읽기만 함).

## 1. 후속 작업 시작 시점 (2026-11-18 11:30–11:37 KST)

### Git
| 항목 | 관측 |
|---|---|
| origin의 원 브랜치 head | `a3a9eb5f107766cc0444fcb391beae869d1c00e8` (기획서의 기준 버전과 같음) |
| 원 서버 worktree의 HEAD | `d160e44` (origin보다 2 commit 뒤; 수정하지 않음) |
| 동결 manifest | `manifests/evidence_gated_r7/freeze_manifest.json`이 a3a9eb5에 commit되어 있음. `frozen_commit = d160e44`, 모델 13종 × seed 3, 대비 21개, stress 모델 8종이 기록됨. **`xpert` 목록은 비어 있음(`[]`)** — XPert checkpoint가 들어간 동결본은 아직 Git에 없음 |

### Slurm (`squeue`)
| job | 이름 | 상태 | 경과 |
|---|---|---|---|
| 25809 | `r7_xpr_s42` | RUNNING (g3) | 7:00–7:14 |
| 25810 | `r7_xpr_s43` | RUNNING (g1) | 6:59–7:13 |
| 25811 | `r7_xpr_s44` | RUNNING (g2) | 6:59–7:13 |

그 밖의 `r7_` job은 대기열에 없었다.

### 단계별 상태

| 구분 | 상태 | 근거 |
|---|---|---|
| **validation 결과** | 완료·commit됨 | pilot(P-1), 3-seed 판정(V-1: 신호 없음), 수정 1(M-1/V-2: B11 − chem_only = −0.00100 [−0.00162, −0.00035], 경계에서 성립), 귀속 A-1(재배선·근거 제거에서도 이득 유지). `reports/evidence_gated_r7/DECISIONS.md`, `manifests/evidence_gated_r7/val_*` |
| **mock 결과**(validation unit에서 최종 평가 코드의 배선 검사) | 완료·commit됨 | job 25826, `manifests/evidence_gated_r7/mock/`. 원 연구가 스스로 절차 위반(D-5: mock이 HCC validation 수치를 계산)으로 기록함. test 정답은 쓰이지 않음 |
| **actual-test 결과** | **없음(미실행)** | 서버 `assets/runs/test_eval/`에 `mock_1794946858`만 있고 `test_eval_v1`이 없음. `manifests/evidence_gated_r7/`에 `test_*` 표가 없음 |
| XPert 재학습(100 epoch 상한, seed 42/43/44) | **진행 중** | 로그의 epoch 표시가 78–79(11:06–11:22 기록). 이전 시도 `r7_xp_s42/43/44`(25789–25791)는 원 연구 D-4에 따라 취소·실패로 기록됨. checkpoint·예측 파일은 아직 `assets/runs/xpert/`에 없음(`xp_timing`만 있음) |
| 보고서(RESULTS·CLAIMS·튜토리얼) | Git에 없음 | `reports/evidence_gated_r7/`에 PROTOCOL, DECISIONS, EXPOSURE, KG_CARD, LITERATURE_GAP만 있음 |

- 후속 작업은 XPert 로그에서 epoch 표시와 파일 갱신 시각만 읽었고, 손실·점수 값은 설정 선택에 쓰지 않았다.
- 원 실행의 최종 test 점수는 이 시점에 존재하지 않는다. 생긴 뒤에도 후속 설정이 동결되기 전에는 읽지 않는다.

## 2. 후속 작업 종료 시점

(종료 시 갱신)
