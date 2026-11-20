# G4 — 살아남은 후보(PIE, State ST-HVG)의 4-fold LOCO 반복성·불확실성·예산

**작성:** 2026-11-10 · 브랜치 `research/feedback-20261110`(커밋하지 않음) · 계획: [FEEDBACK_PLAN_20261110.md](../../docs/FEEDBACK_PLAN_20261110.md) G4 ("살아남은 후보 2개 이하, 4 LOCO 및 seed 반복, 예산·한계 분석") · 선행: [G3_PIE_HEPG2.md](G3_PIE_HEPG2.md), [F3_CROSS_CELL_LINE.md](F3_CROSS_CELL_LINE.md), [F2_LITERATURE_REVIEW.md](F2_LITERATURE_REVIEW.md) §6 · 실행: G1 Slurm(`cpu`, `gpu`).
**성격:** 아래 수치는 모두 실측이다. 계획·추정·미확인은 따로 표시했다. 새 설정 선택은 하지 않았다. 체크포인트 선택은 각 모델의 공식 규칙(소스 세포주 val)만 따랐고, 보류 세포주의 처리 세포 정답은 채점에만 썼다.

---

## 0. 요약

1. **후보 2개:** PIE(G3 조건부 통과)와 State ST-HVG zeroshot(F2 2순위)이다. 둘 다 Arc의 **비상업 라이선스**다.
2. **PIE seed 반복(HepG2 fold, 공식 레시피, seed 0·1·2, 소스 val로만 선택) — 결과가 반복된다.**

   | 조건 | 공식 AUPRC: seed 평균 [범위] | 공식 체크포인트 | 최선 기준선 | seed − 기준선 (3/3 seed) |
   |---|---|---:|---:|---|
   | 미관측 세포주 `unseen_ctx` | **0.443** [0.441–0.445] | 0.442 | 0.391 | +0.050 ~ +0.054, 모두 유의 |
   | 미관측 perturbation `unseen_pert` | 0.316 [0.312–0.321] | 0.321 | 0.414 | −0.094 ~ −0.102, 모두 유의 |
   | 이중 OOD `unseen_ctx_pert` | 0.244 [0.237–0.248] | 0.244 | 0.381 | −0.133 ~ −0.144, 모두 유의 |

   - seed 범위(≤ 0.011)는 PIE–기준선 격차(0.05–0.14)보다 5–13배 작다. G3에서 본 구조가 3/3 seed에서 그대로 나온다: 표적 유전자 `p_de` 백분위 0.65–0.75, Δ RMSE는 no-change보다 나쁨, 표적 제외 지표와 이중 OOD의 표적 특이 신호(pert-c r, retrieval)는 PIE 우세.
   - **seed에 따라 달라지는 주장도 있다:** 이중 OOD 표적 제외 AUPRC 이득(−0.002 / +0.010 / +0.003, 1/3만 유의), 미관측 세포주 표적 제외 방향 이득(2/3 유의), F3 정답 기준 pert-c r 대 같은 정보량 기준선(1/3 유의). 이런 주장은 근거로 쓰지 않는다.
3. **State ST-HVG zeroshot(4 fold, F3 지표, 원자료 정답) — 단순 기준선을 넘지 못한다.**
   - 처리 세포 제외는 **확인**했다: 공개 TOML + cell-load 코드 기준이며, 대조군은 학습에 포함된다.
   - 입력 공간을 저자 자료와 행 단위로 일치시켰다(99.8–99.9% 행 일치).
   - 주 결과 `final.ckpt`는 4/4 fold에서 F3 calibrated same-target보다 **pert-c r(−0.024 ~ −0.079), retrieval(+0.059 ~ +0.086), 방향, DEG AUPRC가 모두 유의하게 나쁘다.** RMSE는 3/4 나쁘고 RPE1은 같다. no-change 대비 RMSE도 K562·Jurkat에서 나쁘다.
   - `best.ckpt`는 보류 세포주 정답으로 선택된 체크포인트(val이 test로 대체됨)인데, 이것도 넘지 못한다.
4. **G4 판정: 통과(반복성·불확실성 확인됨) — 결론은 "제한적 이득".**
   - PIE의 4-fold 공식 결과는 seed를 바꿔도 같은 결론을 낸다.
   - 그러나 **관측 표적을 새 세포주로 옮기는 과제(원자료 정답)**에서는 PIE·State 모두 보정된 같은 표적 평균을 넘지 못한다.
   - 모델 고유의 이득은 미관측 표적에 대한 **표적 특이 순위·판별**(PIE)뿐이다.
5. **자원:** GPU 25.3 GPU-h(A100 12.2, A40 13.1; 이 중 0.57은 NCCL hang으로 낭비), wall 7.1 h. HF 다운로드 14.3 GB. 새 디스크 37 GB(`assets/g4` 20 GB, `assets/envs/state` 17 GB). **4-fold × 3-seed 전체 PIE 재학습은 A100 기준 약 70 GPU-h**(fold당 2×A100 2.9 h), A40 기준 약 157 GPU-h다. 남은 9개 run은 A100 약 52 GPU-h다.
6. **차단 없음.** 환경 편차 1건(transformers 4.57.6 고정), 하드웨어 편차 2건(NCCL P2P 끔, seed별 GPU 종류 다름)을 문서화했다.

---

## 1. 후보와 선정 이유

| 후보 | 상태 | 선정 이유 | 지원 범위 | 라이선스 |
|---|---|---|---|---|
| **PIE** (`replogle_wdataset`, v1.0.1) | G3 조건부 통과 | 4개 세포주 LOCO 공식 split·공식 체크포인트가 있고, 미관측 세포주·미관측 perturbation·이중 OOD를 한 모델로 평가한다. G3에서 4-fold 공식 수치를 근접 재현했다 | 세 OOD 조건 모두 | 코드 CC BY-NC-SA 4.0, 체크포인트 Arc PIE Model **Non-Commercial** License |
| **State ST-HVG** (`ST-HVG-Replogle/zeroshot`) | F2 2순위 | 같은 4개 세포주에 대해 **세포주별 zeroshot 체크포인트**가 공개되어 있다. 출력이 단일세포 발현이라 F3 지표(원자료 정답)로 바로 채점할 수 있다 | **미관측 세포주·관측 표적만.** perturbation이 one-hot이라 미관측 표적·이중 OOD는 원리적으로 예측 불가 | 코드 CC BY-NC-SA 4.0, 모델 Arc **State Model Non-Commercial License**(2025-06-23판) |

- 두 모델 모두 **학술·비상업 연구용**이다. 산업 협업이나 상업적 이용 전에는 법무 확인이 필요하다.
- TxPert(F2 3순위)는 2개 이하 규칙에 따라 이번 G4에서 제외했다.

---

## 2. 설정

### 2-1. State ST-HVG zeroshot (과제 1)

| 항목 | 값 |
|---|---|
| 환경 | `assets/envs/state/venv` 신규. Python 3.12.16, **arc-state 0.11.1**(고정), cell-load 0.10.4, cell-eval 0.8.2, torch 2.10.0+cu128, lightning 2.6.6, anndata 0.13.4, scanpy 1.12.5. 동결 목록 [G4/env_freeze_state.txt](G4/env_freeze_state.txt) |
| **환경 편차** | 의존성 해석이 transformers 5.22.2를 설치했는데, 이 버전의 LlamaConfig 엄격 검증이 체크포인트 설정(hidden 328, head 12, `head_dim` 64 명시)을 거부했다. arc-state가 허용하는 범위(`>=4.52.3`)의 **transformers 4.57.6**(huggingface-hub 0.36.2)으로 고정하자 정상 로드됐다. 가중치·구조는 바꾸지 않았다 |
| 체크포인트 | HF `arcinstitute/ST-HVG-Replogle` @ `bb6a9562cbbf1fd152df14cc53b4cc7517c77175`, `zeroshot/{hepg2,jurkat,k562,rpe1}`. `final.ckpt`(step 100,000)와 `best.ckpt`(step 58,000, HepG2 기준), 설정·one-hot map, 저자 eval 출력(`eval_best.ckpt/adata_{real,pred}.h5ad`). 총 **14.27 GB**, LFS 파일 40/40 sha256 일치([provenance](G4/provenance_state_fetch.json)) |
| 모델 | ST(state transition), hidden 328, Llama 8층(양방향), cell_set_len 64, HVG 2,000 입력·출력, pert one-hot 2,024, **gem_group batch 임베딩 56개**(키가 `'1'…'56'`뿐이라 세포주 간에 공유된다), 49.4 M 파라미터 |
| 추론 입력 | 보류 세포주 L의 **non-targeting 대조군만**(F3와 같은 원파일). 세포별 행 합으로 1e4 정규화 → log1p → 모델의 HVG 2,000개(기호 일치, 누락 0) |
| 입력 검증 | 저자 eval `adata_real`의 대조군 행과 **행 단위 해시(소수 3자리) 일치**: HepG2 4,964/4,976, Jurkat 11,997/12,013, K562 10,674/10,691, RPE1 11,461/11,485. 대조군 평균 상관 1.000000(최대 차 2e-5). `UMI_count` 기준 정규화는 0행 일치 → State 입력 공간을 정확히 재구성했다 |
| 추론 절차 | 표적마다 같은 고정 대조군 풀 512개(seed 42, 64개 세트 8개)를 basal로 쓰고, 각 세포의 자기 gem_group 토큰을 붙였다. 모델 `predict_step`, 출력 [0, 14] clip(`state tx infer`와 동일). 예측 Δ = 예측 평균 − 풀 평균. 풀 재표집(seed 43)으로 basal 표집 잡음도 측정했다. CPU(c3/c5, 32코어)로 세포주당 9–12분 |
| 평가 표적 | F3 조건 A 표적(소스 ≥1개 세포주에서 ≥20 cells) 중 State one-hot map에 있는 것. HepG2 1,850/2,157, Jurkat 1,937/2,264, K562 1,702/1,998, RPE1 1,879/2,200. 나머지(약 300개/세포주)는 State one-hot map(2,024개)에 없는 표적이다. State 자료(`replogle_nogwps_v2`)의 필터 때문으로 보인다 **[추정]** |
| 채점 | F3 코드(`per_target_metrics`, `retrieval_rank`)와 F3 원자료 정답. **T3** = State ∩ PIE `unseen_ctx` ∩ F3 A 표적(867/884/983/978), **T2** = State의 모든 표적. 유전자 = F3 fold 패널 ∩ State HVG ∩ PIE 축(912–999개). 표적 유전자 **제외**(F3 정의)와 **포함** 두 가지. paired bootstrap 2,000회(표적 단위) |

**학습 자료에서 보류 세포주 처리 세포가 빠졌는지 (검증 결과):**

1. 체크포인트 config의 `toml_config_path`는 `.../replogle_nogwps_v2/hepg2_zeroshot.toml`이다. 같은 이름의 TOML 4개가 State GitHub(`state_preprint_toml_files/replogle_tomls/`, 커밋 `9bbfe78a434a55205e4de834e1ea99f85f7a3add`)에 공개되어 있다([사본](G4/state_zeroshot_tomls/)). 내용은 `[training] replogle = "train"`, `[zeroshot] "replogle.<L>" = "test"`이고 `[fewshot]`은 비어 있다.
2. cell-load 0.10.4 코드(`_process_celltype`)에서 zeroshot 세포주는 **처리 세포 전부가 지정 split(test)로만** 가고, **대조군만 train에 추가**된다("adding all observational data to train").
   - 결론: **보류 세포주의 처리 세포는 학습에서 빠졌다(확인).** 대조군은 학습에 들어갔다. 이는 계획서의 "대상 세포주 대조군만 관측" regime과 같다.
   - 단서: 공개 TOML은 경로가 `/data/...`로 바뀐 사본이다. 학습에 실제로 쓰인 파일과 바이트 단위로 같은지는 확인할 수 없다 **[미확인]**. 체크포인트의 cell-type map(4개 세포주)과 batch 차원(56)은 이 TOML과 일치한다.
3. **체크포인트 선택 누수:** 이 TOML에는 val split이 없다. cell-load의 `val_dataloader`는 val이 비면 **test(= 보류 세포주의 처리 세포)를 val로 쓴다.** `best.ckpt`는 `val_loss` 최소(HepG2 step 58,000, 1.0487)로 골라졌으므로 **보류 세포주 정답으로 선택된 체크포인트**다. 그래서 `final.ckpt`(step 100,000, 선택 없음)를 **주 결과**로, `best.ckpt`는 참고로만 보고한다.
4. 사전 노출: SE 임베딩 사전학습은 쓰지 않는다(ST-HVG, `init_from: null`). 다만 HepG2 자료는 우리가 이전에 본 자료이고(§7), State 저자도 이 fold의 test 결과를 공개했다.

### 2-2. PIE seed 반복 (과제 2)

| 항목 | 값 |
|---|---|
| 레시피 | **공식 체크포인트 자신의 `config.yaml`** 그대로다. DDP 2 GPU, GPU당 batch 16, `accumulate_grad_batches` 4(유효 batch 128), `max_steps` 5,000, bf16-mixed, lr 1e-4, warmup 5%, val 100 step마다, 선택 = **소스 세포주 val 298행**의 AUPRC 최대(`best_auprc.ckpt`). 자료·지식원·split revision도 체크포인트에 고정된 것(`pie_replogle_nadig_essential@20c9fae`, `pie_sources@cb1aaa4`, `PIE_splits@396ab95`)을 썼다. 합성 config와 체크포인트 config를 비교한 차이는 seed·logger·run 경로와 동일 파일로 풀리는 legacy 경로뿐이다 |
| 바꾼 것 | 최상위 `seed` = 0, 1, 2(공식 42). evidence seed(42)는 자료 레시피라 유지했다. `logger.enabled=false`(wandb 없음) |
| **하드웨어로 인한 변경(문서화)** | (i) **NCCL P2P 끔**(`NCCL_P2P_DISABLE=1`). 첫 시도(job 20618, g4 A100 GPU 4+7)는 첫 collective에서 멈췄다(16분간 진행 없음, GPU 100% 점유·문맥 메모리만 사용). 전송 경로만 바뀌고 수식은 같다(all-reduce 순서 차이로 마지막 비트 차이는 가능). 2-rank all_reduce 사전 점검을 넣었다. (ii) **GPU 종류가 seed마다 다르다**: seed 0 = 2× A40 48GB(g3), seed 1·2 = 2× A100 80GB(g4). 저자는 H100이다. 장치 수·batch·누적은 바꾸지 않았다. (iii) A30(24 GB) 노드는 GPU당 약 20 GiB가 필요해 제외했다 |
| 평가 | `pie eval ... ckpt=best_auprc` 세 조건, 1 GPU, `save_predictions=true`. 채점은 G3와 같은 `pie.metrics.score` + 표적 제외 지표(`g3_pie_baselines.extra_metrics`), 그리고 F3 지표 |

---

## 3. 4-fold 결과 (조건별)

PIE 공식 4-fold 수치와 기준선은 G3(GPU 재현, `pie.metrics`)에서 가져왔다. seed는 HepG2 fold에서만 돌렸다. State는 미관측 세포주·관측 표적만 예측할 수 있고, F3 지표로만 채점했다. **"최선 기준선"**은 각 조건에서 공식 AUPRC가 가장 높은 PIE-split 기준선이며, 소스 세포주에서만 선택했다(G3 §3).

### 3-1. 미관측 세포주·관측 표적 (`unseen_ctx`, F3 조건 A)

**(a) PIE 지표 (PIE split 867–983행/fold).** 칸 = 공식 AUPRC / 표적 제외 AUPRC_x.

| fold | PIE 공식 | PIE seed 0 / 1 / 2 | 최선 기준선 (`same_target_cal`) | PIE − 기준선 |
|---|---|---|---|---|
| HepG2 | **0.442 / 0.331** | 0.441 / 0.444 / 0.445 · AUPRC_x 0.327 / 0.334 / 0.328 | 0.391 / 0.282 | +0.052 / +0.049 (seed 3/3 동일 부호·유의) |
| Jurkat | **0.376 / 0.285** | – | 0.357 / 0.261 | +0.018 / +0.024 |
| K562 | **0.422 / 0.306** | – | 0.386 / 0.272 | +0.036 / +0.035 |
| RPE1 | **0.421 / 0.371** | – | 0.386 / 0.342 | +0.035 / +0.029 |
| 4-fold 평균 ± SD | 0.415 ± 0.028 / 0.323 ± 0.037 | | 0.380 ± 0.015 / 0.289 ± 0.036 | |

**(b) F3 지표 (원자료 정답, T3 = State ∩ PIE ∩ F3 표적, HVG ∩ 패널 912–999 유전자, 표적 제외).** 칸 = pert-centred r / retrieval rank↓ / RMSE↓ / DEG AUPRC.

| fold | State final | State best (표적 선택, 참고) | PIE 공식 | **F3 calibrated same-target** | no-change RMSE |
|---|---|---|---|---|---:|
| HepG2 (867) | 0.343 / 0.174 / 0.119 / 0.249 | 0.383 / 0.165 / 0.112 / 0.267 | 0.380 / 0.146 / 0.138 / 0.289 | **0.422 / 0.115 / 0.101 / 0.339** | 0.119 |
| K562 (884) | 0.319 / 0.158 / 0.091 / 0.202 | 0.381 / 0.133 / 0.090 / 0.238 | 0.336 / 0.111 / 0.133 / 0.257 | **0.385 / 0.075 / 0.077 / 0.256** | 0.078 |
| RPE1 (983) | 0.366 / 0.223 / 0.139 / 0.392 | 0.361 / 0.229 / 0.151 / 0.370 | 0.314 / 0.197 / 0.157 / 0.390 | **0.395 / 0.137 / 0.140 / 0.433** | 0.158 |
| Jurkat (978) | 0.316 / 0.173 / 0.106 / 0.180 | 0.333 / 0.182 / 0.108 / 0.203 | 0.323 / 0.123 / 0.117 / 0.223 | **0.340 / 0.109 / 0.089 / 0.235** | 0.097 |
| 4-fold 평균 ± SD | 0.336 ± 0.023 / 0.182 ± 0.028 / 0.114 ± 0.021 / 0.256 ± 0.095 | 0.365 ± 0.023 / 0.177 ± 0.040 / 0.115 ± 0.026 / 0.270 ± 0.072 | 0.338 ± 0.030 / 0.144 ± 0.038 / 0.136 ± 0.017 / 0.290 ± 0.072 | **0.386 ± 0.034 / 0.109 ± 0.026 / 0.102 ± 0.027 / 0.316 ± 0.090** | 0.113 |

- PIE의 HepG2 seed를 F3 지표(F3 패널 3,141 유전자 × 867 표적)로 보면 다음과 같다(§5).
  - seed 0/1/2의 pert-c r은 0.361/0.371/0.355(공식 0.352)이고, F3 calibrated 0.389보다 3/3 seed에서 유의하게 낮다.
  - RMSE 0.112–0.123, top-20 방향 0.61–0.65(기준선 0.83), DEG AUPRC 0.235–0.239(기준선 0.277)도 3/3 seed에서 나쁘다.
- **결론:** 관측 표적을 새 세포주로 옮기는 과제에서 PIE는 자체 DE 지표(PIE 라벨)로는 기준선을 넘는다. 그러나 원자료 Δ 정답으로는 PIE·State 모두 보정된 같은 표적 평균을 넘지 못한다. 이 결론은 4/4 fold와 PIE 3/3 seed에서 같다.

### 3-2. 미관측 perturbation (`unseen_pert`, 소스 세포주 안)

칸 = 공식 AUPRC / 표적 제외 AUPRC_x / 표적 제외 방향 / pert-c r(PIE Δ; 기준선 칸의 pert-c r은 ridge 기준, 평균 기준선은 정의상 ≈0). State는 one-hot이라 예측할 수 없다(N/A).

| fold | PIE 공식 | 최선 기준선 | PIE − 기준선 공식 AUPRC |
|---|---|---|---|
| HepG2 (1,238쌍) | 0.321 / **0.324** / **0.725** / **0.269** | `context_mean_kd` **0.414** / 0.295 / 0.665 / 0.111 | −0.093. seed 0/1/2: −0.099 / −0.094 / −0.102. AUPRC_x는 +0.020 / +0.028 / +0.014로 3/3 유의 |
| Jurkat | 0.326 / **0.322** / **0.728** / **0.284** | `ridge_inline_kd` **0.413** / 0.299 / 0.677 / 0.135 | −0.087 |
| K562 | 0.303 / **0.305** / **0.708** / **0.253** | `context_mean_kd` **0.411** / 0.294 / 0.674 / 0.112 | −0.108 |
| RPE1 | 0.269 / **0.271** / **0.704** / **0.255** | `context_mean_kd` **0.399** / 0.249 / 0.636 / 0.142 | −0.130 |
| 4-fold 평균 ± SD | 0.305 ± 0.026 / 0.306 ± 0.025 / 0.716 ± 0.012 / 0.265 ± 0.014 | 0.409 ± 0.007 / 0.284 ± 0.024 / 0.663 ± 0.019 / 0.125 ± 0.016 | |

HepG2 seed 평균 [범위]: 0.316 [0.312–0.321] / 0.316 [0.309–0.323] / 0.721 [0.720–0.722] / 0.272 [0.268–0.277]. 표적 유전자 `p_de` 백분위는 0.68–0.75다(공식 0.69).

### 3-3. 이중 OOD (`unseen_ctx_pert`, 보류 세포주 × 미관측 표적)

칸 = 공식 AUPRC / AUPRC_x / 방향_x / pert-c r(PIE Δ; 기준선 칸은 ridge 기준). State는 N/A다.

| fold | PIE 공식 | 최선 기준선 | 비고 |
|---|---|---|---|
| HepG2 (386) | 0.244 / 0.245 / **0.663** / **0.209** | `ridge_stacked_kd` **0.382** / 0.240 / 0.623 / 0.144 | seed 0/1/2 AUPRC 0.237 / 0.248 / 0.246. AUPRC_x − `global_mean_kd`: −0.002(n.s.) / **+0.010** / +0.003(n.s.) |
| Jurkat | 0.214 / **0.210** / **0.670** / **0.160** | `global_mean_kd` **0.328** / 0.198 / 0.607 / 0.069 | – |
| K562 | 0.198 / **0.200** / **0.631** / **0.160** | `global_mean_kd` **0.358** / 0.178 / 0.545 / 0.085 | – |
| RPE1 | 0.290 / 0.292 / **0.696** / **0.141** | `ridge_stacked_kd` **0.379** / **0.316** / 0.679 / 0.112 | 유일하게 AUPRC_x 열세 |
| 4-fold 평균 ± SD | 0.237 ± 0.040 / 0.237 ± 0.042 / 0.665 ± 0.027 / 0.167 ± 0.029 | 0.362 ± 0.025 / 0.233 ± 0.062 / 0.613 ± 0.055 / 0.102 ± 0.033 | |

- F3 지표(원자료 정답)에서는 4/4 fold 모두 PIE의 pert-c r(0.15–0.23 대 ridge 0.07–0.16)과 retrieval(0.26–0.31 대 0.39–0.43)이 낫다(G3 §9). RMSE·방향은 4/4 나쁘다.
- HepG2 seed 3/3도 같다: pert-c r 대 ridge +0.086 / +0.098 / +0.080, retrieval −0.147 / −0.133 / −0.141로 모두 유의하다.

---

## 4. State 상세 — 미관측 세포주·관측 표적 (T3, 표적 제외, F3 지표)

열: RMSE↓ / Pearson / pert-centred r / top-20 DEG 방향 / DEG AUPRC / retrieval rank↓. 출처: [state_headline.tsv](G4/state_headline.tsv), [state_f3metrics_summary.tsv](G4/state_f3metrics_summary.tsv).

| fold (n 표적, n 유전자) | 예측기 | RMSE | Pearson | pert-c r | 방향 | AUPRC | retr. |
|---|---|---:|---:|---:|---:|---:|---:|
| HepG2 (867, 957) | **State final** | 0.1189 | 0.278 | 0.343 | 0.699 | 0.249 | 0.174 |
| | State best (참고, 표적 선택) | 0.1115 | 0.338 | 0.383 | 0.739 | 0.267 | 0.165 |
| | State 저자 eval 예측 | 0.1158 | 0.288 | 0.329 | 0.719 | 0.253 | 0.178 |
| | PIE 공식(Δ 헤드) | 0.1377 | 0.399 | 0.380 | 0.680 | 0.289 | 0.146 |
| | **F3 calibrated same-target** | **0.1014** | **0.461** | **0.422** | **0.832** | **0.339** | **0.115** |
| | no-change | 0.1187 | – | – | 0 | 0.159 | 0.500 |
| K562 (884, 999) | **State final** | 0.0908 | 0.221 | 0.319 | 0.709 | 0.202 | 0.158 |
| | State best | 0.0903 | 0.267 | 0.381 | 0.728 | 0.238 | 0.133 |
| | PIE 공식 | 0.1326 | 0.328 | 0.336 | 0.716 | 0.257 | 0.111 |
| | **F3 calibrated** | **0.0773** | **0.339** | **0.385** | **0.799** | **0.256** | **0.075** |
| | no-change | 0.0776 | – | – | 0 | 0.126 | 0.500 |
| RPE1 (983, 912) | **State final** | 0.1393 | 0.380 | 0.366 | 0.765 | 0.392 | 0.223 |
| | State best | 0.1510 | 0.290 | 0.361 | 0.709 | 0.370 | 0.229 |
| | PIE 공식 | 0.1574 | 0.393 | 0.314 | 0.715 | 0.390 | 0.197 |
| | **F3 calibrated** | **0.1396** | **0.484** | **0.395** | **0.829** | **0.433** | **0.137** |
| | no-change | 0.1576 | – | – | 0 | 0.250 | 0.500 |
| Jurkat (978, 998) | **State final** | 0.1064 | 0.205 | 0.316 | 0.684 | 0.180 | 0.173 |
| | State best | 0.1079 | 0.266 | 0.333 | 0.718 | 0.203 | 0.182 |
| | PIE 공식 | 0.1169 | 0.315 | 0.323 | 0.650 | 0.223 | 0.123 |
| | **F3 calibrated** | **0.0894** | **0.337** | **0.340** | **0.787** | **0.235** | **0.109** |
| | no-change | 0.0966 | – | – | 0 | 0.092 | 0.500 |

**Paired bootstrap (State final − 비교 대상, 95% CI, [state_headline_boot.tsv](G4/state_headline_boot.tsv)):**

| fold | 대 F3 calibrated: RMSE / pert-c r / 방향 / AUPRC / retrieval | 대 PIE 공식: RMSE / pert-c r / AUPRC / retrieval | 대 no-change RMSE |
|---|---|---|---|
| HepG2 | +0.018 [0.017, 0.019] / −0.079 [−0.090, −0.068] / −0.133 / −0.090 / +0.059 [0.047, 0.070] | −0.019 / −0.037 [−0.048, −0.026] / −0.041 / +0.028 | +0.000 [−0.002, 0.002] |
| K562 | +0.013 / −0.066 [−0.076, −0.056] / −0.090 / −0.054 / +0.083 | −0.042 / −0.017 [−0.027, −0.007] / −0.054 / +0.047 | **+0.013** [0.012, 0.015] 나쁨 |
| RPE1 | −0.000 [−0.002, 0.002] / −0.029 [−0.041, −0.016] / −0.064 / −0.041 / +0.086 | −0.018 / **+0.052** [0.038, 0.067] / +0.002 (n.s.) / +0.026 | **−0.018** [−0.021, −0.015] 좋음 |
| Jurkat | +0.017 / −0.024 [−0.035, −0.013] / −0.103 / −0.054 / +0.064 | −0.010 / −0.007 [−0.016, 0.002] (n.s.) / −0.042 / +0.049 | **+0.010** [0.008, 0.011] 나쁨 |

- **State final은 4/4 fold에서 F3 calibrated same-target보다 Pearson·pert-c r·방향·DEG AUPRC·retrieval이 모두 나쁘다**(모든 CI가 0을 넘지 않음). RMSE는 3/4 나쁘고 RPE1은 같다.
- no-change 대비 RMSE는 RPE1에서만 좋고, K562·Jurkat에서는 나쁘며, HepG2에서는 같다. 즉 State의 Δ 크기 보정은 세포주에 따라 다르다.
- PIE 공식 대비: RMSE는 4/4에서 State가 좋다(PIE Δ 헤드의 크기 미보정, G3 §5). retrieval은 4/4에서 PIE가 좋다. pert-c r은 HepG2·K562에서 PIE, RPE1에서 State가 좋고 Jurkat은 차이 없다.
- **표적 유전자 포함 변형**(T3 표적의 18–21%만 표적 유전자가 유전자 집합 안에 있음): 결론은 같다. F3 calibrated가 표적 유전자의 강한 감소를 담고 있어 포함 시 격차가 조금 더 커진다(예: HepG2 AUPRC 0.356 대 State 0.246).
- **T2(State가 예측 가능한 모든 표적, 1,702–1,937개):** 같은 순서다. HepG2 기준 State final pert-c r 0.293, F3 calibrated 0.386이다.
- **`best.ckpt`(보류 세포주 정답으로 선택됨)도 F3 calibrated를 넘지 못한다.** HepG2에서 best − F3 calibrated는 pert-c r −0.040 [−0.049, −0.030], retrieval +0.050이다. K562에서만 pert-c r 차이가 없다(−0.004 [−0.013, 0.006]). final 대비 best의 이득은 HepG2·K562에서 pert-c r +0.04~+0.06으로, **선택 누수가 성능을 부풀릴 수 있는 크기**다. RPE1에서는 best가 오히려 나쁘다.
- **재현 점검:** 같은 `best.ckpt`로 우리가 만든 표적별 Δ와 저자 eval의 Δ(저자 필터 세포, 표적당 세포 수만큼의 무작위 basal)의 Pearson 중앙값은 0.83(HepG2)·0.85(K562)·0.89(RPE1)·0.87(Jurkat)이다. 노름 비는 0.97–1.03이다. 우리 풀 512개를 다시 뽑았을 때의 Δ 일치도(0.88–0.92)와 비슷한 수준이므로, 차이는 basal 표집 잡음으로 설명된다. 저자 예측을 그대로 채점해도(`State_authors_best`) 같은 결론이다(HepG2 pert-c r 0.329, AUPRC 0.253).
- **정답 일치도:** 저자 `adata_real`로 만든 Δ를 예측으로 넣으면 F3 정답과 Pearson 0.980–0.992다(표적 평균). 따라서 State가 지는 원인은 정답 전처리 차이가 아니다.

---

## 5. Seed 변동 (PIE, HepG2 fold)

**설정 요약:** 공식 체크포인트 config 그대로이고 seed만 0/1/2로 바꿨다(§2-2). 3개 run 모두 5,000 step을 끝까지 돌았다. `best_auprc.ckpt`는 **소스 세포주 val 298행**에서 골라졌다.

| run | GPU | best step | val AUPRC(소스) | 학습 시간 |
|---|---|---:|---:|---:|
| 공식 체크포인트 (seed 42) | 저자 H100×2 | 2,327 | 0.4853 | 저자 보고 약 1.2 h |
| seed 0 (job 20625) | A40×2 (g3) | 2,428 | 0.4817 | 23,351 s (6.49 h) |
| seed 1 (job 20626) | A100×2 (g4) | 2,226 | 0.4856 | 10,371 s (2.88 h) |
| seed 2 (job 20627) | A100×2 (g4) | 2,833 | 0.4810 | 10,354 s (2.88 h) |

**PIE 지표 seed 변동 ([pie_seed_spread_hepg2.tsv](G4/pie_seed_spread_hepg2.tsv); `pie eval` 자체 csv와 우리 재채점이 일치).** 칸 = seed 평균 [최소–최대] / 범위.

| 조건 | 공식 AUPRC | AUPRC_x | 방향_x | Spearman LFC | L1 판별 | 표적 p_de pct | Δ RMSE↓ | pert-c r (Δ) |
|---|---|---|---|---|---|---|---|---|
| unseen_ctx | 0.443 [0.441–0.445] / 0.004 | 0.330 [0.327–0.334] / 0.007 | 0.757 [0.751–0.761] | 0.725 [0.718–0.739] | 0.812 [0.807–0.820] | 0.984 | 0.099 [0.096–0.104] | 0.320 [0.314–0.328] |
| (공식 / 기준선) | 0.442 / 0.391 | 0.331 / 0.282 | 0.752 / 0.741 | 0.711 / 0.694 | 0.813 / 0.558 | 0.982 | 0.102 / no-change 0.087 | 0.311 / 0.305 |
| unseen_pert | 0.316 [0.312–0.321] / 0.008 | 0.316 [0.309–0.323] / 0.014 | 0.721 [0.720–0.722] | 0.557 [0.549–0.567] | 0.716 [0.711–0.722] | **0.717 [0.679–0.749]** | 0.092 [0.088–0.096] | 0.272 [0.268–0.277] |
| (공식 / 기준선) | 0.321 / 0.414 | 0.324 / 0.295 | 0.725 / 0.665 | 0.539 / 0.529 | 0.717 / 0.503 | 0.686 | 0.092 / 0.081 | 0.269 / 0.020 |
| unseen_ctx_pert | 0.244 [0.237–0.248] / 0.011 | 0.242 [0.236–0.249] / 0.013 | 0.657 [0.654–0.661] | 0.536 [0.529–0.548] | 0.688 [0.682–0.694] | **0.691 [0.648–0.727]** | 0.107 [0.103–0.113] | 0.220 [0.212–0.230] |
| (공식 / 기준선) | 0.244 / 0.381 | 0.245 / 0.238 | 0.663 / 0.623 | 0.501 / 0.577 | 0.692 / 0.502 | 0.667 | 0.112 / 0.090 | 0.209 / 0.017 |

**seed 대 공식 체크포인트(paired bootstrap, [pie_seed_paired_bootstrap_hepg2.tsv](G4/pie_seed_paired_bootstrap_hepg2.tsv)):**

- 공식 AUPRC 차이는 −0.009 ~ +0.003이다. 9개 중 3개(seed 0·2의 unseen_pert, seed 0의 이중 OOD)는 CI가 0을 넘지 않는다.
- 즉 seed 차이는 **표적 단위 bootstrap으로 검출 가능한 크기**다. 하지만 기준선 격차의 1/5–1/10이다.
- 공식 체크포인트는 unseen_ctx·이중 OOD에서 seed 범위 안에 있다. unseen_pert에서는 공식(0.3214)이 seed 최대(0.3206)보다 약간 높다.

**결론이 seed에 따라 바뀌는지 (각 seed − 기준선, 95% CI):**

| 주장 | seed 0 | seed 1 | seed 2 | 공식 (G3) | 판정 |
|---|---|---|---|---|---|
| unseen_ctx 공식 AUPRC > `same_target_cal` | +0.050 ✓ | +0.053 ✓ | +0.054 ✓ | +0.052 ✓ | **견고** |
| unseen_ctx AUPRC_x > `same_target_cal` | +0.045 ✓ | +0.052 ✓ | +0.046 ✓ | +0.049 ✓ | **견고** |
| unseen_ctx 방향_x > `same_target_cal` | +0.017 ✓ | +0.020 ✓ | +0.009 (n.s.) | +0.011 ✓ | seed 의존 |
| unseen_pert 공식 AUPRC < `context_mean_kd` | −0.099 ✓ | −0.094 ✓ | −0.102 ✓ | −0.093 ✓ | **견고** (표적 유전자 미예측) |
| unseen_pert AUPRC_x > `context_mean_kd` | +0.020 ✓ | +0.028 ✓ | +0.014 ✓ | +0.029 ✓ | **견고** (크기는 seed에 따라 2배 차이) |
| unseen_pert 방향_x, pert-c r > 기준선 | +0.055 / +0.257 ✓ | +0.057 / +0.248 ✓ | +0.057 / +0.250 ✓ | +0.061 / +0.249 ✓ | **견고** |
| 이중 OOD 공식 AUPRC < `global_mean_kd` | −0.144 ✓ | −0.133 ✓ | −0.135 ✓ | −0.136 ✓ | **견고** |
| 이중 OOD AUPRC_x > `global_mean_kd` | −0.002 (n.s.) | +0.010 ✓ | +0.003 (n.s.) | +0.007 (n.s.) | **seed 의존 → 이득 없음으로 본다** |
| 이중 OOD 방향_x > `global_mean_kd` | +0.031 ✓ | +0.038 ✓ | +0.032 ✓ | +0.039 ✓ | **견고** |
| 이중 OOD pert-c r(Δ) > `ridge_stacked_kd` | +0.076 ✓ | +0.086 ✓ | +0.068 ✓ | +0.065 ✓ | **견고** |
| F3 정답: unseen_ctx pert-c r > PIE-split `same_target_cal` | +0.006 (n.s.) | +0.015 ✓ | −0.001 (n.s.) | −0.004 (n.s.) | **seed 의존 → 차이 없음** |
| F3 정답: unseen_ctx pert-c r < F3 calibrated | −0.028 ✓ | −0.018 ✓ | −0.034 ✓ | −0.037 ✓ | **견고 (PIE 열세)** |
| F3 정답: 이중 OOD pert-c r > ridge, retrieval < ridge | +0.086 / −0.147 ✓ | +0.098 / −0.133 ✓ | +0.080 / −0.141 ✓ | +0.075 / −0.132 ✓ | **견고** |
| Δ RMSE가 no-change보다 나쁨 (3 조건) | 3/3 | 3/3 | 3/3 | 3/3 | **견고** (크기 미보정) |

- seed 간 순위 변동은 약한 효과에서만 결론을 바꾼다. 효과가 0.01 이하인 주장(이중 OOD AUPRC_x, F3 정답 기준 unseen_ctx 동등성)은 **seed 1개로 판단하면 틀릴 수 있다.**
- seed 0만 A40에서 돌았다. seed 0의 값(예: unseen_ctx 0.441, 이중 OOD 0.237)이 A100 seed들과 같은 범위에 있으므로, GPU 종류 영향은 seed 변동보다 크지 않다. 다만 장치와 seed를 분리해서 검증하지는 않았다 **[미분리]**.
- 한계: 1개 fold × 3 seed다. 다른 3개 fold의 seed 분산은 측정하지 않았다(§9).

---

## 6. G4 판정

**Gate 문구:** "외부 맥락 전이 결과의 반복성·불확실성 확인"

**판정: 통과 — 반복성과 불확실성의 크기를 확인했다. 실질적 결론은 "외부 맥락 전이 이득은 제한적"이다.**

| 질문 | 근거 | 답 |
|---|---|---|
| PIE의 G3 결론이 학습 무작위성에 견고한가? | HepG2 fold 3 seed(§5): 공식 AUPRC 범위 ≤ 0.011, 기준선 격차 0.05–0.14. 14개 주장 중 11개가 3/3 seed와 공식에서 같은 결론 | **예.** 단 효과 ≤ 0.01인 주장 3개는 seed에 따라 바뀐다 |
| PIE의 결론이 세포주(fold)에 견고한가? | 공식 4-fold(G3 §9): 조건별 우열 방향이 4/4 fold에서 같다. 예외는 RPE1 이중 OOD AUPRC_x 하나 | **예** (fold 간 SD 0.03–0.04) |
| 두 번째 후보(State)가 PIE나 기준선보다 나은가? | 4 fold × 2 체크포인트(§4): F3 calibrated 대비 pert-c r·retrieval·방향·AUPRC가 4/4 fold에서 유의하게 나쁘다. 표적 선택 체크포인트도 같다 | **아니오.** 미관측 표적은 원리적으로 예측 불가 |
| 미관측 세포주 전이(관측 표적)에서 모델이 단순 전이를 넘는가? | 원자료 정답: PIE·State 모두 F3 calibrated same-target에 4/4 fold에서 열세. PIE 자체 DE 라벨 기준으로만 PIE가 +0.02~+0.05 우세 | **아니오.** 이득은 라벨·지표 정의에 의존한다 |
| 모델만 할 수 있는 일이 있는가? | 미관측 표적(소스·HepG2)의 표적 특이 순위·판별: 표적 제외 AUPRC·방향·pert-c r·retrieval 우세가 4/4 fold, 3/3 seed에서 같다 | **예, PIE만.** 다만 표적 유전자 자신과 Δ 크기는 맞히지 못한다 |

**G5에 넘길 것:**

1. 유전적 LOCO 경로(R1)에서 "새 세포주" 자체는 모델 없이도 같은 표적 평균 + 크기 보정으로 대부분 해결된다. 모델이 추가 가치를 주는 지점은 **새 표적**(특히 이중 OOD의 표적 특이 패턴)이다.
2. HepG2는 특별히 좋거나 나쁜 fold가 아니다(4-fold 범위 안).
3. 결론에 쓸 지표는 **seed와 fold 모두에서 견고한 것**(표 §5의 "견고" 항목)으로 제한한다. 단일 seed·단일 fold의 0.01 이하 차이는 근거로 쓰지 않는다.
4. State는 이 regime에서 탈락이다. State를 다시 고려하려면 미관측 표적을 다룰 수 있는 perturbation 임베딩 변형이나 few-shot regime(대상 세포주 처리 세포 일부 사용)이 필요하다. few-shot은 계획서의 "처리 세포 전부 미학습" 조건을 깬다.

---

## 7. 한계

- **HepG2는 간모세포종 유래**(15세 남성)다. 성인 HCC나 환자 반응의 근거가 아니다. 이번 결과를 HCC 예측 성능으로 일반화하지 않는다.
- **사전 노출.** GSE264667(HepG2·Jurkat)은 이전 V2 프로젝트에서 이미 본 자료다(`development_previously_exposed`). PIE와 State 저자 모두 이 4-fold test 결과를 공개했으므로, 방법 수준의 설계 선택이 test 결과를 보고 이루어졌을 가능성을 배제할 수 없다. **blind test가 아니다.** PIE 지식원(DepMap CRISPRGeneEffect에 HepG2 열 포함, Cellosaurus 텍스트)도 보류 세포주 정보를 일부 담는다(G3 §6).
- **State 체크포인트 선택 누수.** `best.ckpt`는 보류 세포주 처리 세포의 loss로 골라졌다(§2-1). 주 결과는 `final.ckpt`로 보고했다. 그래도 학습 길이(100k step)와 하이퍼파라미터가 저자의 test 결과를 보고 정해졌는지는 알 수 없다 **[미확인]**.
- **State 학습 대조군.** 보류 세포주의 대조군이 학습에 들어갔다(zeroshot 설계). 우리 regime과 같지만, "대조군도 처음 보는 세포주" 상황과는 다르다.
- **CRISPRi만.** CRISPRa(Norman)·약물(Tahoe SNU-423)과 섞지 않았다. State는 one-hot이라 미관측 표적을 예측하지 못하므로 이중 OOD에서는 비교 대상이 아니다.
- **전처리 차이.**
  - State: 저자 필터 자료(`replogle_nogwps_v2`, HepG2 96,616 세포·1,340 표적)에서 학습. 입력은 우리 원파일 대조군(정확히 일치 확인).
  - PIE: knockdown 필터(HepG2 90,826 세포)와 Mann-Whitney DE.
  - F3: 원파일 전 세포(145,473)와 Welch DE.
  - 세 정답의 Δ는 표적별 Pearson 약 0.98 이상으로 일치한다(State 0.980–0.992, PIE 중앙값 0.986). DEG 정의는 서로 다르므로 DEG AUPRC는 정의 의존적이다.
- **유전자 집합.** State 비교는 HVG 2,000 ∩ F3 패널(912–999개)로 제한된다. G3·F3 표(3,121–3,188개)와 절대값을 직접 비교하지 않는다(예: HepG2 F3 calibrated RMSE 0.101 대 F3 표 0.083).
- **통계.** bootstrap은 표적 단위이고 표적은 독립 반복이 아니다. fold는 4개뿐이고 각 fold가 서로 다른 실험실·라이브러리 묶음이다. seed 반복은 HepG2 fold 1개, 3 seed뿐이다.
- **배치 토큰.** State의 gem_group 임베딩은 세포주 간에 번호만 공유한다. 보류 세포주 대조군도 학습에 들어갔으므로 그 번호의 임베딩은 학습되었지만, 생물학적 의미는 없다.

---

## 8. 자원 (실측) — [G4/slurm_jobs.tsv](G4/slurm_jobs.tsv)

| 항목 | 실측 |
|---|---|
| **GPU 시간** | 총 **25.30 GPU-h**. A100 12.21(seed 1·2 각 5.8, hung 시도 0.57 포함), A40 13.09(seed 0). State 추론·모든 채점은 CPU만 사용했다 |
| PIE 학습 1 run (5,000 step, 2 GPU) | 2× A100 80GB: **2.88 h wall**(10,354–10,371 s), GPU당 약 20–22 GB. 2× A40 48GB: **6.49 h**(23,351 s, A100의 2.25배). 저자 보고 2× H100은 약 1.2 h |
| PIE 평가 (3 조건, 1 GPU) | A100 1.6–2 분, A40 3.5 분 |
| State 추론 (4 fold, CPU 32코어) | fold당 9.6–12.5 분(체크포인트 2개 × basal 풀 + 입력 검증 포함). GPU 불필요 |
| 채점 (CPU) | State F3 지표 4 fold 7 분. PIE seed 채점 6 분, F3 지표 1 분 |
| wall | 첫 제출 17:16 → 마지막 결과 00:24, **7.1 h**. 대부분 seed 0(A40) 학습 시간이다. GPU 대기는 첫 제출이 1분, 재제출 후 seed 0이 약 8분, seed 1·2가 약 24분이었다 |
| HF 다운로드 | **14.27 GB**(State zeroshot 체크포인트 8개 + 저자 eval adata 8개 + 설정). 예산 60 GB 안이다. PIE 자산은 G3 것을 재사용했다(추가 다운로드 0) |
| 디스크 (G1 `assets/`) | `g4/` 20 GB(State HF 14 GB, State 예측 0.24 GB, PIE seed 3 run 약 5.7 GB: run당 `best_auprc.ckpt`+`last.ckpt` 1.9 GB) + `envs/state/` 17 GB(venv 8.1 + uv 캐시 8.1). 총 **37 GB** |
| 실패·낭비 | job 20618(NCCL hang, 0.57 GPU-h). 20619/20620은 대기 중 취소(0 GPU-h). transformers 5.x 비호환으로 환경을 한 번 다시 고정(수 분) |

**전체 4-fold × 3-seed PIE 재학습 비용 (실측 기반 추정):**

| 하드웨어 | run당 | 12 run (4 fold × 3 seed) | 남은 9 run (Jurkat·K562·RPE1) | 동시 실행 시 wall |
|---|---|---|---|---|
| 2× A100 80GB | 5.8 GPU-h, 2.9 h | **≈ 70 GPU-h** | ≈ 52 GPU-h | g4 8장이 모두 비면 4 run 동시 → 3회 × 2.9 h ≈ **9 h** |
| 2× A40 48GB | 13.1 GPU-h, 6.5 h | ≈ 157 GPU-h | ≈ 118 GPU-h | g3 8장 → 3회 × 6.5 h ≈ 20 h |
| 2× H100 (저자 보고, **추정**) | 약 2.4 GPU-h | 약 29 GPU-h | – | – |
| A30 24GB | GPU당 약 20 GiB 필요 → **비권장**(시험하지 않음) | | | |

- 저장소: run당 1.9 GB이고, 12 run이면 약 23 GB다(`last.ckpt`를 지우면 약 11 GB).
- 평가·채점은 run당 GPU 2–4 분 + CPU 약 7 분이다.
- 현실 제약: 이번 세션에서 gpu 파티션은 대부분 다른 사용자가 점유했다. 우선순위가 높은 4-GPU 작업(다른 사용자)이 있으면 backfill이 막혀 2-GPU 작업도 1–2 h 기다릴 수 있다.
- **State 4-fold 재학습**(예: source-only val로 선택 누수 제거)은 하지 않았다. 공개 config는 100,000 step, batch 64, 1 GPU다. GPU 시간은 공개되지 않았다 **[미확인]**. F2의 2–4일 추정을 유지한다.

---

## 9. 남은 것과 재현 명령

**남은 것 (미실행):**

1. **다른 3개 fold의 PIE seed 반복**(9 run, A100 약 52 GPU-h). fold 간 seed 분산이 HepG2와 같은지 확인하려면 필요하다. 아래 명령으로 바로 제출할 수 있다.
2. **PIE 사후 크기 보정**(G3 §5 (ii)). Δ 헤드에 a·Δ+b를 소스 val에서 맞추는 것이다. Δ RMSE가 3/3 seed·4/4 fold에서 no-change보다 나쁘므로, F3 정답 비교의 RMSE·방향 열세가 보정만으로 해소되는지가 다음 질문이다. 선택은 소스 val에서만 해야 한다.
3. **State의 선택 누수 없는 재학습**(소스 세포주 val split 추가) 또는 `final` 외 중간 체크포인트 평가. 이번 결과(best도 기준선 미달)로 보아 결론이 바뀔 가능성은 낮다 **[추정]**.
4. TxPert(F2 3순위)는 2개 이하 규칙으로 제외했다.
5. 장치(A40 대 A100) 효과와 seed 효과는 분리하지 않았다.

**재현 명령 (G1 worktree 루트):**

```bash
# State 환경 (한 번) — arc-state 0.11.1, transformers<5
UV_CACHE_DIR=$PWD/assets/envs/state/uv_cache assets/envs/pie/py312/bin/uv venv --python assets/envs/pie/py312/bin/python3.12 assets/envs/state/venv
UV_CACHE_DIR=$PWD/assets/envs/state/uv_cache VIRTUAL_ENV=$PWD/assets/envs/state/venv \
  assets/envs/pie/py312/bin/uv pip install --torch-backend=cu128 arc-state==0.11.1 torch==2.10.0 "transformers<5"
# State 자산 (14.3 GB) → 추론 (CPU) → F3 지표
sbatch experiments/feedback/g4_state_fetch.sbatch
for f in hepg2 jurkat k562 rpe1; do sbatch --cpus-per-task=32 --mem=128G experiments/feedback/g4_state_infer.sbatch $f; done
sbatch experiments/feedback/g4_f3metrics.sbatch state
# PIE seed 재학습 (공식 레시피, 2 GPU, A30 제외) → 채점
for s in 0 1 2; do sbatch --exclude=g1,g2 experiments/feedback/g4_pie_seed_train.sbatch $s hepg2; done
sbatch experiments/feedback/g4_pie_seed_score.sbatch && sbatch experiments/feedback/g4_f3metrics.sbatch pie_seeds
.venvs/scgpt/bin/python src/feedback/g4_report_tables.py
# (미실행) 다른 fold: <fold> ∈ jurkat k562 rpe1
for f in jurkat k562 rpe1; do for s in 0 1 2; do sbatch --exclude=g1,g2 experiments/feedback/g4_pie_seed_train.sbatch $s $f; done; done
sbatch experiments/feedback/g4_pie_seed_score.sbatch --fold <fold> \
  --g3-pairs assets/g3_pie/baselines/pie_split_per_pair_<fold>.tsv.gz \
  --official-ckpt assets/g3_pie/models/replogle_wdataset/<fold>/best_auprc.ckpt
#   (g4_f3metrics.py의 pie_seeds 부분은 현재 HepG2 고정 — 다른 fold는 fold 인자 추가 필요)
```

**상태:**

| 상태 | 항목 |
|---|---|
| 실측 완료 | State zeroshot 4 fold(입력 검증, 추론, F3 지표, 체크포인트 2종, 재현 점검). PIE HepG2 fold 3 seed 재학습·평가·PIE/F3 지표. 예산 |
| 진행 중·대기 중 | 없음(모든 job 종료, 대기 job 없음) |
| 미실행 | 다른 3 fold seed 반복, PIE 크기 보정, State 재학습, TxPert |
| 차단 | 없음 |

## 10. 산출물

- **코드:** `src/feedback/g4_state_fetch.py`, `g4_state_infer.py`, `g4_pie_seed_score.py`, `g4_f3metrics.py`, `g4_report_tables.py`. 배치 스크립트는 `experiments/feedback/g4_{state_fetch,state_infer,pie_seed_train,pie_seed_score,f3metrics}.sbatch`.
- **보고서 파일 (`reports/feedback/G4/`):**
  - State: `provenance_state_fetch.json`, `env_freeze_state.txt`, `state_zeroshot_tomls/`, `state_infer_<fold>.json`, `state_f3metrics_{summary,paired_bootstrap,meta}.*`, `state_headline*.tsv`, `state_4fold_mean.tsv`
  - PIE seed: `pie_seed_{summary,spread,paired_bootstrap}_hepg2.tsv`, `pie_seed_ckpt_hepg2.json`, `pie_seed_f3metrics_{summary,paired_bootstrap,meta}_hepg2.*`
  - `slurm_jobs.tsv`
- **G1 자산 (git 제외):** `assets/g4/state/{hf,pred,tomls}`, `assets/g4/pie_seeds/hepg2_seed{0,1,2}/`(체크포인트·eval·예측), `assets/envs/state/`.
