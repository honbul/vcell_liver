# 실행 시간과 자원 비교의 범위

비용은 각 실제 run의 wall_total_seconds와 peak_process_gpu_gib를 우선 읽는다. Pretrained checkpoint의 원 사전학습 비용과 LiverCT reference 구축 비용은 측정하지 않았다. Shared reference/adaptation을 쓰는 LiverCT 두 head의 비용은 공유되며 합산해 두 번 청구하지 않는다.

| 경로 | 기록된 시간의 범위 |
|---|---|
| Frozen annotation A-series | Run의 학습은 head grid/refit, 추론은 head evaluation. Backbone feature extraction은 연결된 feature_extraction_metadata의 별도 실제 실행이며 여러 fold에서 재사용 |
| scVI | 학습은 inner/outer100epoch와 head 선택/refit. 추론은 inner/outer 모델의 전체 개발 세포 latent 계산과 head evaluation. 순수 heldout-only latency가 아님 |
| LiverCT | Preprocessing에는 counts alignment와 reference copy, 학습에는20epoch query adaptation, 추론에는 latent와 native voting ensemble/SCANVI head. Fold0v4는 저장 모델 복원이며 원v3 adaptation 시간이 따로 고립되지 않았음 |
| Geneformer partial/LoRA/NN | Training은 inner/outer optimizer 구간, inference는 inner validation 및 outer evaluation과 reload 비교. 기존 frozen cache와 online adapted encoder 비용이 다름 |
| 공식 GEARS Norman | Native train()에 포함된 validation/test/deeper analysis가 training 시간 안에 있음. 별도 inference는 기록용 추가 prediction pass. 순수 optimizer 비용으로 해석하지 않음 |
| Strict HepG2 GEARS | 학습 loop에 epoch별 validation이 포함됨. Final recorded inference는 selection 후 heldout prediction |
| Native scGPT response | Training은 optimizer 구간. Inference는 각 epoch의 validation aggregate와 선택 후 final test aggregate의 합. Pure test-only latency가 아님. Pretrained/random은 동일 설정·동일 물리 GPU에서 연속 실행 |
| scVIDR C00v3 | 저장된 C00v2 VAE를 복원해 새 training0. 원v2 실패 파일의88.30초는 preprocessing/training/실패 prediction 합계이며 순수 학습 시간으로 분리할 수 없음. History에는35 actual epochs, max budget100이 기록됨 |
| Final annotation | Fit와 query-feature extraction, head inference가 서로 다른 artifact. Partial query inference는 encoder representation 시간을 별도 wall_representation_seconds로 기록. Head-only 파일만 전체 pipeline latency처럼 비교하지 않음 |

Slurm의 대기 시간은 Python process wall time에 포함되지 않는다. A30과 A40은 별도 hardware로 표시한다. BF16/FP16은 matmul precision이며 weights/storage는 FP32다. Process GPU memory는 측정된 해당 PID의 값이며 다른 사용자의 프로세스를 합산하지 않는다. Numeric scVI parameter count와 LiverCT requires_grad/hook 제한은 model_parameter_audit 보완 파일에 있다.

CPU peak RSS도 Python parent process 범위이며 DataLoader worker 전체 job memory의 합계가 아니다. 공유 서버 부하·cache와 측정 구간 차이가 있으므로 숫자를 독립적인 hardware 속도 순위로 해석하지 않는다.
