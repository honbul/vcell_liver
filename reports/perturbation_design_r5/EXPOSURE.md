# 연구 R5 — 데이터·가중치 노출 기록 (EXPOSURE)

| 자료 | 이 연구 전에 본 적 | 이 연구에서의 사용 |
|---|---|---|
| CMap 2020 Level 3 source 161계열 | 연구 B·R2에서 처리·사용 | dev(meta-train 16계열): 설정 선택·진단·learned-loss 학습. meta-val 8계열: 정책 비교·판정·D3 평가. |
| HCC 3계열(HUH7, JHH5, JHH7) 처리 후 값 | 연구 B에서 시험셋으로 채점, R2에서 동결 후 평가 | 동결(commit `a3c8146`) 후 job 25387에서 한 번만 열었다. 봉인 경로: `oracle.py`가 `R5_FINAL_EVAL=1`과 commit된 freeze manifest를 요구. |
| HCC plate DMSO 대조군 | B에서 X0로 사용 | 처리 전 정보로 정책 입력(ctl_sd)에 사용. build 단계에서 plate별 분산만 계산·기록(`store_manifest.json`). |
| R2 family 표 | R2 산출물 | metadata 표만 복사(`source_families_from_r2.tsv`). R2의 모델·예측·점수는 읽지 않음. |
| 연구 B 처리 배열·환경 | — | read-only 재사용, 수정 없음. |

- **이 연구의 분할은 새 seed지만 공개 자료를 이미 본 상태다.** 독립 blind cohort라고 부르지 않는다. 특히 B·R2의 HCC 결과(mean-offset이 강함)를 알고 설계했다.
- HCC 정답은 어떤 설정 선택·조기종료에도 쓰지 않았다. 동결 전 HCC 관련 실행은 metadata(그룹 수, plate 수, 대조군 분산)뿐이다.
- 수정 1의 meta-val 재평가(job 25385)는 사전 기록대로 실행했지만, 채택 판단은 dev 결과로 먼저 내렸다(`DECISIONS.md` §4).
- learned-loss 모델은 dev seed 7–10 episode로 학습했다. dev 표의 learned_loss 수치는 학습 episode와 겹치므로 낙관적이며, 판단은 meta-val로 했다.
