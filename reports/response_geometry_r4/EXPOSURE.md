# 연구 R4 — 자료·가중치 노출 기록 (EXPOSURE)

## 1. 이 자료를 전에 본 적이 있는가

- 같은 CMap LINCS 2020 Level 3 자료(10 µM·24 h, 978개 landmark)는 연구 B와 R2에서 이미 학습·채점에 쓰였다. 그 연구들은 **"source 세포주에서 이미 측정된 약물을 HCC에서 예측"**하는 과제였고 약물 분할도 달랐다.
- R4의 약물 그룹 분할(seed 20261117)은 이번에 새로 만들었다. 그러나 공개 자료이고 이 프로젝트의 앞선 연구자가 같은 측정값을 본 적이 있으므로 **완전한 blind 평가가 아니다.** 새 seed라는 이유로 blind라고 부르지 않는다.
- 분할 seed는 점수를 보기 전에 정했고 바꾸지 않았다.

## 2. R4 안에서의 노출 순서

| 자료 | 누가 읽었나 | 언제 |
|---|---|---|
| training 약물의 Δ (모든 세포주, HCC 포함) | 진단 D1–D3, teacher(PCA·scale·세포 평균), 신뢰도, 모든 모델 학습, 기준선 적합 | 규칙 commit 이후 |
| validation 약물의 Δ | 학습률·λ·α 선택, checkpoint 선택(최저 validation ΔRMSE), 개발 판단 | 개발 중 |
| test 약물의 Δ | `test_eval.py` 한 번 | freeze manifest commit 이후 |

- unit 생성 job(25296)은 test unit의 Δ를 계산해 별도 파일(`unit_delta_test.npy`)에 봉인했다. 그 파일은 `Store.test_delta`만 열 수 있고, 이 함수는 `R4_FINAL_EVAL=1`과 freeze manifest가 없으면 거부한다(누출 fixture에서 거부 확인).
- teacher·신뢰도·세포 평균은 training unit만으로 적합한다. validation 값을 무작위로 바꿔도 teacher가 그대로임을 fixture로 확인했다.
- 추론 함수는 `predict_delta(x0, fp)` 하나다. 처리 후 RNA, teacher, 신뢰도를 입력으로 받지 않는다.

## 3. 외부 가중치·코드

- WAVE(ZJUFanLab/WAVE, commit `e27e453beb48d0ff912585060b5a09cb04335262`)의 `wave/model.py`를 수정 없이 불러 썼다. 저자 가중치는 없다(모든 가중치는 이번에 처음부터 학습).
- 연구 B의 처리된 배열과 Python 환경을 읽기 전용으로 썼다. B·R1–R3의 checkpoint, 예측, 점수는 입력으로 쓰지 않았다.
- 세포 family 규칙의 파생 관계 목록은 R2의 규칙 코드(metadata)를 옮겨 적었다. R2의 결과는 쓰지 않았다.
