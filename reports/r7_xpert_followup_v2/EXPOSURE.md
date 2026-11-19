# R7 XPert v2 — 노출 기록

근거: private plan 2026-11-19 §2.2, §4.1. 이 트랙은 원 R7(`b962a84`)의 결과를 고치거나 덮어쓰지 않는다. 원 R7의 한계 기록(동결 전 mock에서 HCC validation 수치 계산 D-5, 판정 뒤 추가한 XPert A-2, HG embedding 출처 불확실성)은 그대로 유지한다. A-2를 사전 비교나 OOF calibration 결과로 바꾸어 부르지 않는다.

| 자료 | 이전 노출 | 이 트랙의 사용 | 지위 |
|---|---|---|---|
| R4 train 약물 정답 | 모든 이전 모델의 학습 자료 | outer 3 fold의 보정 적합(A_k)·평가(T_k) | 이미 노출된 자료의 후속 개발 |
| R4 validation non-HCC | R4/R7 선택·개발에 사용 | 모든 XPert checkpoint 선택 | 선택 전용 |
| R4 validation HCC (129 unit) | 원 XPert의 자동 checkpoint 선택에 포함됨(이번 감사로 확인), R7 D-5 mock | 정정 run에서 원 규칙 replay 점수 계산에만 사용 | 새 선택·보정·λ에 사용하지 않음 |
| R4 test | R4가 열고 R7이 다시 평가 | 동결 후 정정 영향·§4.2 6단계 후속 평가 1회 | 노출된 benchmark, 맹검 아님 |

시간 순서(모두 2026-11-19 KST):
1. 감사 part 1(job 26735): 파일·로그·h5ad만 읽음. 점수 계산 없음.
2. pilot(2 epoch, fold 0/seed 42): validation non-HCC 선택 점수와 A_0 inner CV 점수만 계산. T_0 정답은 합성 잡음으로 대체. v1 checkpoint 재현 검사에서 원 R7 로그와 같은 HCC 포함 validation loss4(0.7235)를 다시 계산함(새 정보 아님).
3. 동결 commit `16c524a` (기록 `d1f406c`).
4. 본 학습(trajectory 51개) → outer fold 평가 → R4 test 후속 평가.
