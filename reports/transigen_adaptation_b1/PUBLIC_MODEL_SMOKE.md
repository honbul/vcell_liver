# 연구 B · 단계 B0 — 공개 TranSiGen 코드와 예제의 입출력 확인

**작성일:** 2026-11-16 · **브랜치:** `research/hcc-transigen-adaptation-b1` · **실행:** Slurm job 23932 (GPU 1개, A30)
**공개 코드:** `myzheng-SIMM/TranSiGen` commit `8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a` (서버 `assets/external/TranSiGen`, 수정 없음)
**원자료:** `manifests/transigen_adaptation_b1/public_smoke.json`, `upstream_smoke_compat.diff`, `env_pip_freeze.txt`

이 문서의 숫자는 모두 공개 저장소의 **100행 예제**에서 나온 것이다. 코드 경로가 실행되는지를 확인한 것이며 연구 결과가 아니다.

## 1. 확인하려던 것

1. 공개 코드가 지정 commit에서 현재 환경(Python 3.10, PyTorch 2.3.1)으로 실행되는가.
2. 예제 자료의 모양과 척도가 무엇인가. 저장소의 전체 `processed_data_id.h5`에 발현 행렬이 들어 있는가.
3. 공개 full checkpoint에 우리 대상 HCC 세포주가 노출됐는가.
4. 추론이 "대조군 발현 X0 + 약물 표현"만으로 가능한가. 처리군을 다시 만드는 재구성 경로와 구분되는가.
5. 이번 연구에서 쓸 구성(ECFP4, random 초기화)이 학습되는가.

## 2. 실제로 실행한 것

- `src/transigen_adaptation_b1/smoke_public.py`: 예제 파일 설명, 공개 checkpoint 로드·forward·loss·추론, adapter를 통한 ECFP4 새 모델 학습(예제 80행 학습, 20행 검증, batch 16, 1,000 update).
- 공개 학습 진입점 `train_TranSiGen_full_data.py`를 **복사본**에서 세 가지 설정으로 실행했다. 복사본에 적용한 변경은 `np.str` → `str` 두 줄뿐이다(NumPy에서 `np.str`가 삭제됐기 때문). 고정한 원본 checkout은 건드리지 않았다.

## 3. 관측된 사실

### 3.1 예제 자료

| 항목 | 값 |
|---|---|
| `data_example/processed_data.h5` | `x1`(대조군)·`x2`(처리군) 각 100 × 978, float32, 값 범위 0.35–15.0, 평균 8.46 |
| 예제 구성 | 43개 세포주, 화합물 14개, 행마다 고유한 `sig` 이름 |
| `data_example/ECFP4_emb2048.pickle` | 화합물 14개, 길이 2048, 값은 0과 1 |
| `data_example/KPGT_emb2304.pickle` | 화합물 14개, 길이 2304 |
| 저장소의 전체 `data/LINCS2020/processed_data_id.h5` | 78,569행 · 164개 세포주 · 화합물 8,316개의 **ID만** 있다. 발현 행렬은 없다 |

- 공개 코드는 대조군을 `x1`, 처리군을 `x2`라고 부른다. 이 연구의 문서는 대조군을 X0, 처리군을 X1이라고 부른다. 코드에서 두 이름이 섞이지 않도록 adapter 함수의 인자 이름을 `x0`, `x1`로 고정했다.
- 값의 범위와 평균은 L1000 Level 3 척도와 같다. 추가 정규화 없이 Level 3 값을 그대로 넣는 것이 원형의 입력 척도다.
- 전체 X0/X1 행렬은 저장소에 없다. 따라서 본 실험의 발현 자료는 공식 CMap 2020 Level 3에서 직접 만든다(단계 B1).

### 3.2 공개 checkpoint의 대상 노출

- 전체 ID 파일의 164개 세포주에 **HUH7, JHH5, JHH7이 모두 들어 있다.** 100행 예제에도 JHH5가 있다.
- 공개 full checkpoint의 폴더 이름은 `trained_models_164_cell_smiles_split`이다. 즉 대상 HCC 세포주의 반응이 학습에 쓰였을 가능성을 배제할 수 없다.
- 판정: 계획서 §2.2에 따라 **공개 checkpoint는 입출력 확인에만 쓰고 성능 비교에는 쓰지 않는다.** B0는 같은 구조를 대상 계열을 뺀 source에서 다시 학습해 만든다. shRNA로 사전학습한 VAE 가중치도 본 실험에서는 쓰지 않는다.

### 3.3 공개 checkpoint의 forward·loss·추론 (예제 100행, KPGT 2304차원)

| 항목 | 값 |
|---|---|
| 모델 | `TranSiGen`, 파라미터 5,589,356개, 잠재 차원 100 |
| 출력 모양 | 100 × 978 |
| 공개 loss (표본당) | 합계 476.7 = mse_x1 25.4 + mse_x2 176.3 + mse_pert 224.7 + 0.1 × (KLD 세 항 228.5, 139.5, 134.5) |
| **예측** — X0와 약물만 입력, 잠재 평균 경로 | ΔRMSE 0.461, 조건별 ΔPearson 0.682 |
| **예측** — 공개 `forward()` 그대로(잠재 표본 1개, seed 1) | ΔRMSE 0.467, 조건별 ΔPearson 0.674 |
| **재구성** — 측정된 X1을 넣어 다시 만든 것. 예측이 아니다 | ΔRMSE 0.401, 조건별 ΔPearson 0.763 |
| 무변화(Δ=0) | ΔRMSE 0.645 |

- 재구성 경로는 정답인 처리군을 입력으로 받으므로 예측 경로보다 점수가 좋게 나온다. 이 둘을 섞으면 성능이 부풀려진다. 이 연구의 추론 함수 `predict_x1_mean_path(model, x0, feat)`에는 처리군 인자가 아예 없다.
- 공개 `forward()`는 `eval()` 상태에서도 두 잠재 변수를 표본추출한다. 그래서 같은 입력의 출력이 seed마다 달라진다(확인함). 잠재 평균 경로는 같은 공개 층을 표본추출 없이 호출하며 두 번 실행한 출력이 완전히 같다(확인함). 이 연구의 모든 채점은 잠재 평균 경로를 쓴다.
- 예제 100행은 이 checkpoint의 학습 자료였을 수 있다. 위 점수는 일반화 성능이 아니다.

### 3.4 공개 학습 진입점 실행 (복사본, 예제 65/18/17행 smiles split)

| 설정 | 결과 |
|---|---|
| README 구성 (KPGT, shRNA 사전학습 VAE), 3 epoch | 끝까지 실행됨. 검증 loss 3403 → 3365 → 3557, best epoch 1, 평가 CSV 생성 |
| ECFP4, random 초기화, batch 64, 3 epoch | epoch마다 update가 1회뿐이다. 출력된 loss가 `nan`, `kld_pert`가 `inf`였고 `best_model.pt`가 저장되지 않아 `FileNotFoundError`로 끝났다 |
| ECFP4, random 초기화, batch 16, 60 epoch | 처음 몇 epoch의 출력 loss는 `nan`/`inf`였으나 끝까지 실행됐고 best epoch는 54였다. 평가 CSV 생성 |

### 3.5 adapter로 학습한 ECFP4 새 모델 (예제 80행 학습 · 20행 검증)

| update 수 | 0 | 5 | 150 | 300 | 500 | 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| 검증 ΔRMSE (잠재 평균 경로) | 8.44 | 7.47 | 4.39 | 1.91 | 2.09 | 1.74 |

- 같은 20행에서 무변화의 ΔRMSE는 0.662다. 80행으로 처음부터 학습한 모델은 1,000 update 뒤에도 무변화보다 2.6배 나쁘다.
- 학습 loss는 계속 줄었고 `nan`은 나오지 않았다.

### 3.6 코드에서 확인한 차이점

| 공개 코드의 동작 | 이 연구의 처리 |
|---|---|
| `dataset.py`가 `--data_path`와 무관하게 `../data/LINCS2020/data_example/*`를 읽는다 | 공개 `dataset.py`를 import하지 않는다. 자체 loader는 경로에 `data_example`이나 `TranSiGen`이 있으면 거부하고, 파일 hash를 commit된 manifest와 대조한다 |
| README는 `cell_split`, 실행 분기는 `cells_split`이다 | 공개 자동 분할을 쓰지 않고, 고정한 split manifest를 주입한다 |
| `--train_flag` 등이 `type=bool`이라 문자열 `False`도 참이 된다 | 본 실험은 공개 진입점을 쓰지 않는다. 자체 실행기는 값 인자만 받는다 |
| early stopping이 없다. 정해진 epoch를 모두 돌고 검증 loss가 가장 낮은 모델을 저장한다 | 계획서의 patience를 자체 실행기에서 적용한다 |
| 검증 기준이 표본추출이 들어간 전체 loss다 | 선택 기준을 잠재 평균 경로의 ΔRMSE로 통일했다(사전 고정). 공개 loss도 함께 기록한다 |
| 모델 객체 전체를 pickle로 저장한다 | `state_dict`와 SHA-256을 저장한다 |
| 한 행이 (세포주, 화합물) 조건 수준의 발현 쌍이다(78,569행) | 한 행이 (세포주, 화합물, plate)의 대조군 평균·처리군 평균 쌍이다. 저자 집계의 수치 재현이 아니다 |

모델 구조(`model.py`의 `TranSiGen` 클래스)와 loss 함수는 수정 없이 그대로 import해 쓴다.

## 4. 가능한 해석 (검증하지 않은 가설)

- ECFP4·random 초기화에서 처음에 출력되는 `nan`/`inf`는, update가 몇 번 안 된 상태에서 BatchNorm의 누적 통계가 실제 입력 척도(평균 약 8.5)를 따라가지 못한 채 `eval()` 모드로 loss를 계산해 분산 항이 넘친 것일 수 있다. batch 16으로 update 수를 늘리면 유한한 값으로 돌아온 것이 이 설명과 맞지만, 원인을 따로 분리해 시험하지는 않았다.
- 80행으로 처음부터 학습한 모델이 무변화보다 나쁜 것은, 이 구조가 변화량이 아니라 발현의 절댓값(약 8.5 척도) 전체를 decoder로 다시 만들어야 하기 때문일 수 있다. 이것은 예제에서의 관측이며, 본 실험의 B1(HCC-only)이 어떻게 될지는 단계 B3에서 실제 자료로 확인한다.

## 5. 검증하지 않은 것

- 논문 표의 수치 재현은 시도하지 않았다(계획서 §4.2에서 요구하지 않음). 저자의 전체 가공 자료와 KPGT 임베딩 전체가 저장소에 없다.
- 공개 checkpoint가 실제로 어떤 행으로 학습됐는지는 ID 파일과 폴더 이름으로만 추정했다.
- KPGT 표현은 예제 14개 화합물에서만 사용했다. 본 실험은 ECFP4 2048-bit로 고정한다.

## 6. 본 실험으로 넘어가는 결정

- 공개 checkpoint와 shRNA 사전학습 가중치는 주 비교에서 제외한다.
- B0는 대상 HCC 계열(HUH7, HUH751, JHH5, JHH7)을 모든 조건에서 뺀 source로 공개 구조를 다시 학습해 만든다.
- 학습 중 loss가 유한하지 않으면 그 실행을 멈추고 `non_finite_loss`로 기록한다. 그 후보는 선택에서 지는 것으로 처리하며 숨기지 않는다.
