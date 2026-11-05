# Response baselines on Norman validation

The first common-benchmark run used the frozen split 26 with 155 training interventions and 34 validation interventions, a common 4,547-gene output panel and the same 32 evaluation controls for every method. The [validation receipt](simple_validation.json) belongs to Slurm job `17673` (exit 0) and matches the [dataset receipt](../../../manifests/common_norman_v1_s26.json) by hash. The prediction bundle remains on the compute server.

| Method | Scored validation interventions | Mean intervention ΔRMSE | ΔRMSE on the same 8 additive-eligible interventions |
|---|---:|---:|---:|
| No change | 34 | 0.07707 | 0.07929 |
| Training mean response | 34 | 0.06779 | 0.06282 |
| Additive response | 8 | 0.05040 | 0.05040 |
| Linear GO ridge | 34 | 0.05071 | 0.04834 |
| GEARS | 34 | 0.06363 | 0.06756 |
| scGPT (M0) | 34 | 0.03429 | 0.03264 |

Lower is better. The additive method is defined only where both component singles have training responses. The matched subset column is the fair comparison with additive. The ridge penalty was selected from four fixed values using validation ΔRMSE; the selected value was 1.0.

The [GEARS run](gears/result.json) completed all 15 epochs on an A40 (Slurm `17675`, exit 0). Its epoch 14 checkpoint was selected by validation ΔRMSE and reproduced the saved score after reload. Across all 34 validation interventions, GEARS improved on no-change and training mean but had more error than linear GO ridge. It had more error than additive on the matched eight interventions.

The [scGPT run](scgpt/result.json) completed all 15 epochs on an A100 (Slurm `17674`, exit 0). Its epoch 15 checkpoint reproduced the validation score after reload. The run updated 51.33 million parameters, preserved all 79,348 target flags per epoch, and recorded 74.1 minutes across training and validation epochs. It had the lowest error across all 34 validation interventions and on the matched eight. This is the M0 validation reference for model tuning; held-out test results are pending.

The [training log audit](training_log_audit.json) records both Slurm log hashes and excerpts. All 15 epoch records for each model match its saved result history, and each selected epoch was the minimum validation ΔRMSE in that history.
