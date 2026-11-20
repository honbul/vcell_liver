# R6-O 실행 명세 (동결 전 작성) — 훈련 조건을 맞춘 관측 예산 비교

근거: private plan 2026-11-19 §3.4. 기계가 읽는 명세: `configs/r6_specificity_v2/r6o_spec.json`. 코드: `src/r6_specificity_v2/train_obs.py`, `src/r6_specificity_v2/obs_eval.py`. run 목록: `experiments/r6_specificity_v2/tasks/obs_runs.txt`(18 run).
노출 지위: 이미 노출된 source 자료의 후속 개발(v1 2·3절과 같은 source out-of-fold). 시험·검증 약물은 예측하지 않는다.

## 질문과 가설
v1 3절의 N<64 칸은 N=64로만 학습한 모델의 평가라 정보량 효과와 학습·추론 불일치가 섞여 있었다. 가설: N∈{16,32,64}로 학습한 O1이 같은 평가 입력의 N=16에서 N64 전용 O0보다 전체 RMSE가 낮다.

## 팔과 맞춘 조건
- 모델은 M만(v1 M 설정, lr 3e-3, 60 epoch, 마지막 epoch). SD는 추가하지 않는다.
- O0: 훈련 세트마다 N=64. O1: 갱신마다 N∈{16,32,64} 균등.
- 두 팔 모두 fresh view. 주 난수열(pool, 세포 64개, 약물, p, 64개 전체의 박화)은 두 팔에서 같고, O1은 같은 64개 박화 세포 중 앞 N개(같은 barcode의 nested subset)를 쓴다. N은 별도 generator에서 두 팔 모두 뽑아 다른 난수를 바꾸지 않는다. 갱신 수·optimizer·schedule 동일.
- source fold 0/1/2 × seed 42/43/44 × 2팔 = 18 run.
- 평가 입력은 모든 모델과 pool 평균 기준선에 같다: 고정 eval_input 64개 중 앞 N개(N=64/32/16, 보조 N=8은 두 팔 모두 훈련 범위 밖), 저장된 박화 p∈{1, 0.5×3, 0.25×3}.

## 주 대비와 판정 (결과 보기 전에 고정)
- 주 family: N=16에서 O1−O0 전체 RMSE, p=1/0.5/0.25 세 대비, 양측 bootstrap p에 Holm α=0.05. 실용 기준 상대 0.5%.
- 보조 family: N=64(O1의 비용), N=32, N=8(보조). 모두 공개.
- 판정: 세 대비 모두 Holm 유의·음수 → 불일치 몫 지지 / 일부 → 부분 / 없음 → 미지지.
- 같은 모델 안의 동일 N×p(N64·p0.25 대 N16·p1, N64·p0.5 대 N32·p1): O1에서도 CI가 0 아래면 "이 자료·모델에서 같은 N×p일 때 얕게 많은 세포가 유리"가 남는다. 이것은 N×p 비교일 뿐 UMI·비용 비교가 아니다.
- pool 내 대비 RMSE를 모든 대비에서 함께 보고, pool 내 개선이 없으면 약물 고유 반응 회복으로 해석하지 않는다.
- 보조: 실제 UMI 분포·usable cell 표, 그리고 pool마다 16세포 p=1과 같은 기대 UMI가 되도록 64세포를 p*(0.25–1로 제한)로 박화한 UMI 일치 비교.

## 비용
pilot(job 26789, O1 fold 0 seed 42, A30): 학습 22초, job 43초. 18 run 전체 GPU 약 10분 이하로 계획 범위 안이다.
