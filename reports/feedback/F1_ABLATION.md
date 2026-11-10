# F1-ablation: Full FT 우위·샘플링 결과의 통제 ablation (A6–A9)

계획: [FEEDBACK_PLAN_20261110.md §1-3](../../docs/FEEDBACK_PLAN_20261110.md). 작성 2026-11-10. **상태: 학습 진행 중 — 결과 미확정.**

모든 실행은 **검증 분할 전용**이다. 학습 코드는 `outcomes_val.npz`만 읽고 테스트(홀드아웃) 결과 파일은 열지 않는다. 각 비교는 이전 연구의 seed 0 단일 실행을 기준으로 하는 1 split / 1 seed screening이며, 이번 결과만으로 원인을 단정하지 않는다. 계획대로 유의미한 개선이 보이면 3 seed 반복으로 넘어간다.

## 1. 공통 고정 조건

| 항목 | 값 (이전 C0/M/D 실행과 동일) |
|---|---|
| 데이터 | `C0-Norman-v1-s26` (학습 155 / 검증 34 / 테스트 34 개입, 출력 유전자 4,547개). receipt sha256 `5fefe283…`, 이전 자산(`prior_assets/common`)을 읽기 전용으로 사용 |
| 모델 | scGPT 공개 사전학습 가중치(해시 검증), dropout 0, bf16 autocast |
| 학습 | seed 0, 15 epoch, batch 32, **epoch당 표본 56,049개·업데이트 1,752회**, Adam, StepLR(γ 0.9/epoch), gradient norm clip 1, 학습 배치마다 1,536개 유전자 패널(표적 유전자 강제 포함) |
| 검증 | 매 epoch 34개 검증 개입 × 고정 평가 대조군 32개, 개입별 ΔRMSE의 macro 평균. 최적 epoch는 검증 ΔRMSE로만 선택. 최적 체크포인트 재로딩 후 점수 재현 확인 |
| 하드웨어 | A100 80GB PCIe 우선 (`--gres=gpu:a100:1`), 이전 실행과 같은 GPU 종류 |

학습 루프는 `src/common_train.py`의 루프를 그대로 옮긴 [`src/feedback/f1_ablation_train.py`](../../src/feedback/f1_ablation_train.py)를 쓴다. 바뀌는 것은 [실행 등록부](../../experiments/feedback/f1_ablation_runs.json)에 미리 적은 옵션뿐이다(학습률, `common_scgpt.configure`의 용량 옵션, 대조군 입력, 노출 스케줄). 등록되지 않은 run ID는 실행을 거부한다.

## 2. 실험 설계

| 실험 ID | Run ID (`assets/runs/A/…`) | 고정 조건 | 변경 변수 | 가설 | 판정 기준 |
|---|---|---|---|---|---|
| A6-a | `A-Norman-v1-scgpt-M0lr3em5-D0-s26-r0` | M0, D0, 그 외 §1 | 학습률 1e-4 → **3e-5** | M0의 1e-4가 최적이 아닐 수 있다(대조: M0 기준 자체의 LR 민감도) | 기준 M0(0.03429) 대비 ΔRMSE 차이. M0가 낮은 LR에서 크게 좋아지면 PEFT 비교의 LR 공정성 문제가 M0에도 존재 |
| A6-b | `A-Norman-v1-scgpt-M2lr3em4-D0-s26-r0` | M2(마지막 2블록), D0 | 학습률 1e-4 → **3e-4** | 학습 파라미터가 적은 M2는 동일 업데이트 예산에서 더 큰 LR이 필요하다 | M2 기준(0.04003) 대비 개선 여부, 그리고 M0(0.03429)와의 격차가 얼마나 줄었는지 |
| A6-c | `A-Norman-v1-scgpt-M3lr3em4-D0-s26-r0` | M3(QKV-LoRA r8), D0 | 학습률 1e-4 → **3e-4** | LoRA는 통상 FT보다 큰 LR이 필요하며, 1e-4에서 과소학습됐다 | M3 기준(0.04384) 대비 개선, 곡선이 15 epoch에도 계속 하강하는지 |
| A6-d | `A-Norman-v1-scgpt-M3lr1em3-D0-s26-r0` | M3, D0 | 학습률 1e-4 → **1e-3** | 같은 가설의 상단 값 | 3e-4 대비 추가 개선 또는 불안정(발산·plateau) |
| A7-a | `A-Norman-v1-scgpt-M3r16-D0-s26-r0` | M3, lr 1e-4, D0, QKV만 | LoRA rank 8 → **16** (alpha 32, scale 2 유지) | LoRA 열세는 rank(표현력) 부족 때문이다 | M3 기준 대비 개선. A7-b와 학습 파라미터 수가 같아(1,446,402) **위치 vs rank**를 같은 예산에서 비교 |
| A7-b | `A-Norman-v1-scgpt-M3ffn-D0-s26-r0` | M3, lr 1e-4, D0, rank 8 | LoRA 위치 QKV → **QKV + FFN(linear1, linear2)** | 적응해야 할 위치가 attention이 아니라 FFN에 있다 | M3 기준 및 A7-a 대비 개선 |
| A7-c | `A-Norman-v1-scgpt-M2b4-D0-s26-r0` | M2, lr 1e-4, D0 | 학습 블록 마지막 2 → **4** | partial FT 열세는 학습 가능한 층 수 부족 때문이다 | M2 기준(0.04003) 대비 개선, M0와의 격차 |
| A8 | `A-Norman-v1-scgpt-M0-Dmean-s26-r0` | M0, lr 1e-4, D0과 같은 행 순서·패널·표적 | 단일 고정 대조군 세포 입력 → **학습 대조군 풀 7,353개의 평균 발현** (모든 처리 세포에 동일 입력) | D0의 개별 대조군 짝짓기는 정보가 아니라 잡음이다 (D1의 재샘플링이 이득이 없던 이유와 연결) | D0(0.03429) 대비 차이; D1(0.03494)도 함께 표시. 평가 프로토콜은 그대로(평가 대조군 32개 개별 입력의 예측 평균) 두고, 평균 입력으로 평가한 값을 진단용으로 별도 기록 |
| A9 | `A-Norman-v1-scgpt-M0-Dcap-s26-r0` | M0, lr 1e-4, 고정 대조군 짝, epoch당 56,049 표본·1,752 업데이트 | 자연 노출 → **cap 기반 균형 다운샘플링** (§3) | D2의 악화는 '균형화' 자체가 아니라 소수 개입 세포의 과도한 반복 노출 때문이다 | D0(0.03429)·D2(0.03576)와 비교. A9가 D0 이상으로 좋거나 D2보다 확실히 나으면 반복 노출 가설을 지지, D2와 비슷하게 나쁘면 균형화(분포 이동) 자체가 원인일 가능성 |

**배치 정보 (A8 제약):** Norman 공개 가공본 `obs`에는 `condition, cell_type, dose_val, control, condition_name`만 있고 배치·GEM well·replicate 식별자가 없다(`protocols/common/norman-v1.json`의 `batch` 항목, `controls.json`의 `"batch": "not recorded"`). 따라서 **배치 매칭 대조군은 만들 수 없으며**, 이번 A8은 "단일 세포 짝짓기 vs 풀 평균" 비교로 한정한다. 어떤 대조군도 '실제 짝지어진 세포'라고 주장하지 않는다. 세포 상태(cell-cycle 등) 매칭도 이번 범위에서 제외했다.

## 3. A9 업데이트 예산을 같게 유지하는 방법

- 고정 예산에서 **엄밀한 균형 + 무중복**은 수학적으로 불가능하다. 개입별 세포 수 nᵢ는 52–1,171개(중앙값 299)이고, 155개 개입을 완전히 균형 맞추면 epoch당 개입당 약 361개가 필요해 52개짜리 개입은 각 세포를 약 7번 써야 한다(D2가 바로 이 경우). 무중복(각 세포 ≤1회/epoch)으로 56,049개를 채우는 유일한 방법은 D0 자체다. 같은 이유로 "짧은 epoch를 더 많이" 도는 방식도 전체 학습 동안 희소 개입 세포의 총 반복 횟수는 D2와 같아진다.
- 그래서 A9는 **cap 기반 다운샘플링**으로 균형을 부분적으로만 맞추고 세포 반복은 epoch당 최대 2회로 묶었다. cap c는 2·Σᵢ min(nᵢ, c) ≤ 56,049를 만족하는 가장 큰 정수 **c = 194**다. 개입 i의 epoch당 노출은 eᵢ = 2·min(nᵢ, c)이고, 남은 209개는 cap에 걸린 124개 개입에 seed 순서로 1개씩 더한다.
- 개입 i의 행은 epoch마다 새로 섞은 순열을 앞에서부터 순환하며 뽑는다. 따라서 epoch는 **중복 없는 cap 부분집합 두 개를 이어 붙인 것**과 같다(큰 개입은 두 번째 부분집합에서 다른 세포를 쓴다). 전체를 D0/D2와 같은 RNG 계열로 섞고, 고정 대조군 짝을 그대로 쓴다.
- 결과 (epoch 1, CPU 계산·학습 중에도 매 epoch 기록): 표본 56,049개, 업데이트 1,752회, 고유 세포 44,626개, 세포 최대 반복 2회(2회 사용 11,423개, 3회 이상 0개), 개입별 노출 104–390회. 비교: D0 52–1,171회·반복 1회·고유 56,049개; D2 361–362회·최대 반복 7회·고유 43,029개(3회 이상 1,947개).
- 즉 A9는 노출 균형(최대/최소 비 3.75, D0 22.5, D2 1.0)과 반복(최대 2, D0 1, D2 7) 양쪽에서 D0와 D2의 중간이다. 고정 예산에서는 균형과 반복이 같이 움직이므로, A9 한 점으로 두 요인을 완전히 분리할 수는 없다. 이 한계를 결과 해석에서도 명시한다.

## 4. Smoke test 증거

Slurm job **20305** (g3, **A40**, 2026-11-10 14:08–14:30 KST). 같은 구현으로 9개 설정을 차례로 2 epoch × 20 업데이트만 학습하고 매 epoch 전체 검증을 돌렸다. 최적 체크포인트를 다시 불러와 점수 재현도 확인했다. 출력은 `assets/runs/A-smoke/`에 두었으며 결과로 쓰지 않는다. 요약은 [`F1/ablation/smoke_evidence.tsv`](F1/ablation/smoke_evidence.tsv), 전체는 `smoke_evidence.json`에 있다. 첫 시도(job 20281)는 smoke 전용 동결 검사 코드의 공개 체크포인트 경로 오류(`public/best_model.pt` → `public/scgpt/best_model.pt`)로 두 번째 설정에서 중단됐다. 학습 경로와는 무관하며, 고친 뒤 전체를 다시 실행했다.

| Run | 학습 파라미터 (이전 해당 mode) | 옵션이 실제로 바꾼 것 | 동결 backbone 검사 | lr epoch1→2 |
|---|---:|---|---|---|
| M0lr3em5 | 51,331,586 (= M0) | `model_init.json`이 이전 C0 M0와 **완전히 동일**, 입력 = 고정 짝 대조군 세포 | 해당 없음(전체 학습) | 3e-5 → 2.7e-5 |
| M2lr3em4 | 4,209,154 (= M2) | `model_init.json` 이전 M2와 동일, 학습 블록 10–11 | 129개 텐서 불변 | 3e-4 → 2.7e-4 |
| M3lr3em4 | 1,249,794 (= M3) | `model_init.json` 이전 M3와 동일, QKV LoRA 12개 (down 8×512, up 1536×8, scale 2), 업데이트 후 up norm 최소 0.189 | 153개 불변 | 3e-4 → 2.7e-4 |
| M3lr1em3 | 1,249,794 (= M3) | 위와 같은 구조, up norm 최소 0.604 | 153개 불변 | 1e-3 → 9e-4 |
| M3r16 | **1,446,402** (M3 1,249,794) | QKV LoRA 12개, **down 16×512, up 1536×16**, scale 2 | 153개 불변 | 1e-4 → 9e-5 |
| M3ffn | **1,446,402** | LoRA mount **36개** = QKV 12 + FFN 24(linear1·linear2, 512×512, rank 8), up norm 최소 0.046 | 153개 불변 | 1e-4 → 9e-5 |
| M2b4 | **7,365,122** (M2 4,209,154) | 학습 블록 **8–11** | 105개 불변 | 1e-4 → 9e-5 |
| M0-Dmean | 51,331,586 | 첫 배치 32행의 대조군 입력이 **모두 동일하고 풀 평균과 일치**(D0에서는 짝지은 개별 세포와 일치). 행 순서·노출은 D0와 같음 | – | 1e-4 → 9e-5 |
| M0-Dcap | 51,331,586 | epoch 표본 56,049 / 업데이트 1,752, 고유 세포 44,626, 세포 최대 반복 2, 개입 노출 104–390 (D0: 56,049 / 반복 1 / 52–1,171) | – | 1e-4 → 9e-5 |

- 기본 옵션(LR만 바꾼 A6, Dmean·Dcap)의 `model_init.json`은 이전 C0/M 실행 파일과 바이트 수준 내용이 같다. 즉 `common_scgpt.py` hook을 추가해도 기본값은 이전 구성을 그대로 재현한다(학습 파라미터 이름·개수·로드한 backbone key 동일).
- PEFT 설정에서는 공개 체크포인트에서 로드한 backbone 텐서 중 학습 대상이 아닌 것(M2 129개, M2b4 105개, M3 계열 153개 원본 행렬)이 업데이트 뒤에도 공개 파일과 동일했다. LoRA `up` 행렬은 0으로 초기화되므로 norm이 0보다 크면 실제로 업데이트됐다는 뜻이다.
- A7-a(rank 16)와 A7-b(QKV+FFN, rank 8)는 학습 파라미터 수가 정확히 같다(1,446,402). 같은 예산에서 rank와 위치를 비교할 수 있다.
- A8 진단: 학습한 모델에 평가 대조군 32개의 평균을 넣어 얻은 ΔRMSE를 epoch마다 `val_delta_rmse_mean_control_input`으로 기록한다(smoke 20 업데이트 기준 0.0781 / 0.0766, 같은 시점 주지표 0.0781 / 0.0766).
- smoke 검증 ΔRMSE(0.077–0.129)는 업데이트 20회 뒤의 값이라 성능 근거가 아니다. 경로가 끝까지 실행되고 유한한 값을 낸다는 것만 확인한다.
- smoke는 A40에서 실행했다(설정당 약 130초). 본 실행은 A100을 요청했다.

## 5. 제출한 Slurm 작업

제출 시각 2026-11-10 13:34 KST, G1 Slurm `gpu` 파티션, 각 작업 `--gres=gpu:a100:1 --cpus-per-task=8 --mem=60G --time=07:00:00`. 9개 모두 smoke 작업 **20305**에 `--dependency=afterok:20305 --kill-on-invalid-dep=yes`로 걸었다. smoke 9개 설정이 모두 통과해야 시작하며, 하나라도 실패하면 자동 취소된다. 제출 기록은 G1 `logs/f1_ablation_submissions.tsv`, 로그는 `logs/f1_ablation_<job>.log`에 있다.

| 실험 | Run ID | Slurm job | 요청 GPU |
|---|---|---:|---|
| A6-a | `A-Norman-v1-scgpt-M0lr3em5-D0-s26-r0` | 20311 | A100 |
| A6-b | `A-Norman-v1-scgpt-M2lr3em4-D0-s26-r0` | 20312 | A100 |
| A6-c | `A-Norman-v1-scgpt-M3lr3em4-D0-s26-r0` | 20313 | A100 |
| A6-d | `A-Norman-v1-scgpt-M3lr1em3-D0-s26-r0` | 20314 | A100 |
| A7-a | `A-Norman-v1-scgpt-M3r16-D0-s26-r0` | 20315 | A100 |
| A7-b | `A-Norman-v1-scgpt-M3ffn-D0-s26-r0` | 20316 | A100 |
| A7-c | `A-Norman-v1-scgpt-M2b4-D0-s26-r0` | 20317 | A100 |
| A8 | `A-Norman-v1-scgpt-M0-Dmean-s26-r0` | 20318 | A100 |
| A9 | `A-Norman-v1-scgpt-M0-Dcap-s26-r0` | 20319 | A100 |

**상태 (2026-11-10 14:55 KST 확인):** smoke 통과 뒤 20311–20317 7개가 g4 **A100 80GB PCIe**에서 14:45–14:50에 시작했다. 7개 모두 epoch 1을 오류 없이 마쳤다(업데이트 1,752회, 표적 플래그 79,348개로 이전 실행과 같음). epoch 1 소요는 259–300초로, 같은 mode의 이전 실행(M0 297초, M2 268초, M3 260초)과 비슷하다. 20318(A8), 20319(A9)는 A100 자원을 기다리고 있다. 7개는 약 16:00–16:10에 끝날 것으로 보이고, A8·A9는 A100이 비는 대로(늦어도 앞선 7개가 끝나는 시점) 시작해 약 17:30 전후에 끝날 것으로 예상한다.

실행 시간은 이전 실행 기준 run당 약 65–75분(A100)이다. 제출 시점에는 GPU 노드 4대(g1–g4)가 모두 다른 사용자 작업으로 차 있었다. A100 2–3장이 비면 3–4 wave로 끝나며, 대기 시간이 대부분을 차지한다. A100 대기가 지나치게 길면 대기 중인 작업만 A40으로 바꿀 수 있다(`scontrol update JobId=<id> TresPerNode=gres:gpu:a40:1`). 이 경우 GPU 종류가 결과 TSV의 `gpu` 열에 기록되며, 시간 비교에서는 A100 실행과 분리한다.

이전 실행(`prior_assets/runs/…`)은 읽기만 한다. 새 산출물은 G1 `assets/runs/A/<run-id>/`에 저장한다(`settings.json`, `model_init.json`, `history.json`(매 epoch), `best_model.pt`, `best_validation.json`, `resume.pt`, `result.json`). 실행 디렉터리는 `result.json`이 생기면 불변이다. 시간 제한 등으로 중단되면 같은 run ID에 `--resume`을 붙여 다시 제출한다(설정 해시가 같아야 함).

## 6. 집계 방법

모든 run이 끝난 뒤(또는 중간 상태를 볼 때) G1 worktree 루트에서 실행한다:

```bash
ssh gssong@g1.sysmed.kr
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_feedback_20261110
.venvs/scgpt/bin/python src/feedback/f1_ablation_aggregate.py
# 로컬로 가져오기 (로컬 worktree에서)
rsync -av gssong@g1.sysmed.kr:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_feedback_20261110/reports/feedback/F1/ablation/ reports/feedback/F1/ablation/
```

출력:
- `reports/feedback/F1/ablation/ablation_results.tsv`: 이전 기준 arm 7개와 ablation 9개. 열은 run ID, 실험, 변경, mode/lr/adapt/data_mode, 상태, 학습 파라미터, 최적 epoch, 검증 ΔRMSE, 기준 arm과 그 대비 차이, 보조 기준 arm과 그 대비 차이, ΔPearson, top-20 recall, 방향 일치, 업데이트 수, GPU, 학습·검증 시간(분, epoch 합계로 이전 보고서와 같은 정의), Slurm job.
- `ablation_curves.tsv`: epoch별 검증 ΔRMSE, train loss, lr, 노출 통계. A8은 평균 대조군 입력으로 평가한 진단값도 포함한다.
- `ablation_val_curves.png`: 실험(A6–A9)별 검증 곡선과 이전 기준 arm(점선).

끝나지 않은 run은 `history.json`을 기준으로 `running (epoch k)`, 시작 전이면 `pending`으로 표시한다.

## 7. 결과 — 진행 중

학습이 끝나지 않았다. 집계 후 `reports/feedback/F1/ablation/ablation_results.tsv`(run ID, 변경, 학습 파라미터, 최적 epoch, 검증 ΔRMSE, 기준 arm 대비 차이, GPU, 학습·검증 시간)와 `ablation_curves.tsv`·`ablation_val_curves.png`(epoch별 곡선, 이전 기준 arm 포함)로 채운다. 결과를 해석할 때 지킬 것:

- 1 seed 차이는 이전 연구에서 관측한 seed 간 변동(D0 holdout 0.04012–0.04235)과 비교해야 하며, 그보다 작은 차이는 '차이 없음'으로 보고한다.
- 개선이 보여도 이번 결과는 **검증 분할 screening**이다. 테스트 홀드아웃은 다시 쓰지 않으며, 후속 확인은 3 seed 반복으로 한다.
- A6에서 PEFT가 더 큰 LR에서 M0에 근접하면 "같은 LR이 PEFT에 불리했다"를 지지한다. 그래도 M0보다 나쁘면 LR만으로는 설명되지 않으며 A7 결과와 함께 본다.

## 8. 코드 변경과 재현 경로

| 파일 | 내용 |
|---|---|
| `src/common_scgpt.py` (기존, 하위호환) | `configure(model, mode, partial_blocks=2, lora_rank=8, lora_alpha=16, lora_ffn=False)`와 `build(..., adapt=None)` 키워드 hook. 기본값이면 이전과 같은 경로와 같은 `model_init.json`을 만든다(§4). M2 외 mode에 `partial_blocks`, M3 외 mode에 LoRA 옵션을 주면 거부한다. `adapt`가 주어질 때만 `model_init.json`에 `adaptation` 키를 추가한다 |
| `src/feedback/f1_ablation_train.py` | 등록부 기반 학습 진입점. `common_train.py`의 epoch 루프를 그대로 옮기고 `batches`를 import했다. 이전 자산(`prior_assets/public`, 이전 worktree의 `.upstream`)을 해시 검증한 채로 사용하며, `history.json`을 매 epoch 저장한다 |
| `src/feedback/f1_ablation_data.py` | `TrainingData` 하위 클래스. D0/D1/D2는 상속 그대로, `Dmean`(풀 평균 입력)과 `Dcap`(§3)을 추가 |
| `src/feedback/f1_ablation_aggregate.py` | 결과표·곡선 집계(§6) |
| `src/feedback/f1_ablation_smoke_summary.py` | smoke 증거 요약(§4) |
| `experiments/feedback/f1_ablation_runs.json` | 사전 등록한 9개 run과 기준 arm |
| `experiments/feedback/f1_ablation_{train,smoke}.sbatch`, `f1_ablation_submit.sh` | Slurm wrapper와 1회 제출 스크립트(제출 기록 `logs/f1_ablation_submissions.tsv`, 사본 [`F1/ablation/slurm_submissions.tsv`](F1/ablation/slurm_submissions.tsv)) |

축소(trim)한 실험은 없다. 요청된 9개를 모두 제출했다. 계획 표의 "rank 32(필요 시)"와 "LR 2–3값 전부"는 이번 1차 screening에 넣지 않았다. A6/A7 결과에서 추세가 보일 때만 추가한다.
