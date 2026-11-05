# Simple response baselines on Norman validation

The first common-benchmark run used the frozen split 26 with 155 training interventions and 34 validation interventions, a common 4,547-gene output panel and the same 32 evaluation controls for every method. The [validation receipt](simple_validation.json) belongs to Slurm job `17673` (exit 0) and matches the [dataset receipt](../../../manifests/common_norman_v1_s26.json) by hash. The prediction bundle remains on the compute server.

| Method | Scored validation interventions | Mean intervention ΔRMSE | ΔRMSE on the same 8 additive-eligible interventions |
|---|---:|---:|---:|
| No change | 34 | 0.07707 | 0.07929 |
| Training mean response | 34 | 0.06779 | 0.06282 |
| Additive response | 8 | 0.05040 | 0.05040 |
| Linear GO ridge | 34 | 0.05071 | 0.04834 |

Lower is better. The additive method is defined only where both component singles have training responses. The matched subset column is the fair comparison with additive. The ridge penalty was selected from four fixed values using validation ΔRMSE; the selected value was 1.0. The test split and the GEARS/scGPT common model results are still pending.
