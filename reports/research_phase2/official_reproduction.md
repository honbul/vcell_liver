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
