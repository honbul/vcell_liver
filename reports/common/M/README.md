# scGPT model tuning on Norman validation

All five scGPT conditions predicted expression after gene intervention on the same Norman split (26), with fixed training data and controls (D0). They used the same learning rate (0.0001), learning seed (0), and 15 epochs of 56,049 training examples and 1,752 updates per epoch. The shared evaluator scored all 34 validation interventions on the same 4,547 output genes and 32 controls.

| Condition | Trainable parameters | Best epoch | Validation ΔRMSE ↓ | Training and validation, min |
|---|---:|---:|---:|---:|
| [M0: pretrained, full fine-tuning](../C0/scgpt/result.json) | 51,331,586 | 15 | 0.03429 | 74.10 |
| [M1: decoder and intervention encoder](M1/result.json) | 1,053,186 | 14 | 0.05760 | 67.05 |
| [M2: M1 plus the last two transformer blocks](M2/result.json) | 4,209,154 | 15 | 0.04003 | 67.07 |
| [M3: M1 plus rank-8 LoRA in the fused QKV projections](M3/result.json) | 1,249,794 | 15 | 0.04384 | 64.99 |
| [M-random: random initialization, all parameters trainable](M-random/result.json) | 51,331,586 | 15 | 0.03685 | 74.81 |

Every run used an NVIDIA A100 80 GB PCIe and retained all 79,348 target flags per epoch. Times sum training and validation epochs. Each score was recomputed after reloading the best validation checkpoint. The [LoRA checkpoint audit](lora_training_audit.json) independently checked optimizer updates in all 12 attention blocks and verified that the pretrained backbone stayed frozen after epoch 1.

The [validation selection receipt](model_selection.json) verified the saved checkpoint hashes and selected **M0** for both data experiments. M2 had the lowest validation error among the three parameter-efficient alternatives but did not improve on M0. LoRA used 2.4% of M0's trainable parameters and took about 12% less time per full run, at a validation error of 0.04384 versus 0.03429. The comparison between full fine-tuning and random initialization favors pretrained initialization on this validation split and learning seed. The random arm is a descriptive control under the prespecified protocol. These are validation findings; held-out evaluation was not used for selection.

Slurm jobs `17806` and `17807` run D1 (control resampling) and D2 (intervention-balanced sampling) with M0 fixed. The D0 reference is the completed M0 run above.
