# Official-example reproduction

Pinned scGPT source: cebd6fae655b9c585a4807daa3ac31bb764f06b4, Tutorial_Annotation.ipynb. Original MS reference 7,844 and query 13,468 cells; seed 0, 10 epochs, batch 32, length 3001 retained. Run R00_ms_full_seed0_v2 executed successfully on A30. All 24 pretrained attention projection tensors were verified after the official key mapper.

| Metric | Stored upstream notebook output | This run |
|---|---:|---:|
| Accuracy | 0.875 | 0.863528 |
| Macro precision | 0.775 | 0.715310 |
| Macro recall | 0.737 | 0.694437 |
| Macro F1 | 0.732 | 0.675581 |

This is an executed **modified reproduction**, not quantitative agreement. Non-flash PyTorch attention, weight-key mapping, runtime versions and stochastic kernels differ; source comparison and executed-script hashes plus compatibility patch are in the run directory. No published variance range was established, so no invented reproduction tolerance is used. The 0.0564 F1 gap remains a limitation. Training took 1,476 seconds, inference 101 seconds, observed process GPU peak 9.80 GiB.

Original data files and hashes: manifests/research_phase2/official_scgpt_ms.json. Raw predictions and weights remain in the data workspace, not Git. The 64+64-cell smoke run is separate and its scores are not the quantitative benchmark.

GEARS official Norman simulation training is a separate actual genetic-response task. Its result will be added only after execution.


## GEARS 공식 Norman 실제 실행

R01_norman_simulation_seed1_v2는 공식 simulation seed1, hidden64, batch32/test128, 20epochs를 실제 완료했다. 원 평가의 MSE .00528548, DE MSE .18092044, Pearson .985051, DE Pearson .866546이다. 공식 test의 107개 intervention을 유지했다. 원본 DE dictionary를 사용하는 공식 재현과, heldout DE를 학습에 주지 않는 strict HepG2 적용은 별도 task다. Native train()에 포함된 내부 평가 시간도 wall_training_seconds에 포함되므로 이를 순수 optimizer 시간으로 해석하지 않는다. 통일된 조건별 평균 평가에서는 단순 GO/PCA/ridge가 GEARS보다 낮은 Delta RMSE를 보였다(각 .05004, .06921).
