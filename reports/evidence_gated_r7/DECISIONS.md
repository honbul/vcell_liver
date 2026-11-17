# R7 결정 기록 (DECISIONS)

각 결정은 실행 **전에** 적는다. 형식: 관측 → 가능한 원인 → 예측 → 변경 하나 → 판정 기준. 실행 후 결과는 같은 항목 아래 "결과"로 덧붙인다(앞부분은 고치지 않는다).

## 0. 시작 전 확인 (2026-11-18)

- 기획서 수신 확인(비공개, Git 미포함): `00_OVERVIEW_AND_GIT_REVIEW.md` 16097 B sha256 `eb565983…be4f9`, `02_R7_EVIDENCE_GATED_MECHANISM.md` 20081 B `51a86118…`; 전체 해시는 `manifests/evidence_gated_r7/plan_receipt.json`.
- R4 브랜치·서버 자산은 수정하지 않는다. R4 unit 파일은 읽기 전용으로 쓴다. R4 test는 이미 한 번 열렸다(job 25374). 따라서 R7 test는 노출된 공개 benchmark의 사전 등록 후속 평가다.
- 계산은 모두 Slurm에서 하고 job 이름은 `r7_`로 시작한다.

## D-0. chemistry backbone 정의 (실행 전, R4 validation 기록 근거)

- **관측(R4 validation 기록, test 아님):** R4 delta-MLP의 validation source ΔRMSE는 0.566(3 seed)으로 train_cell_mean 0.5538, ECFP+X0 Ridge(α=1e4) 0.5469보다 나빴다(`manifests/response_geometry_r4/baselines_validation.tsv`, `runs/mlp_full_*`). best checkpoint는 3000–7000 update에서 나왔다.
- **가능한 원인(가설):** 원시 Level 3 X0 입력의 척도와 세포 평균 이동을 MLP가 직접 맞춰야 하는 출력 계약. 확인하지 않았다.
- **R7에 주는 위험:** 기전 보정 r은 세포 상태를 입력으로 받는다. 따라서 backbone이 세포 평균을 못 맞추면, r이 "지식"과 무관하게 세포 평균 차이를 메워 점수가 오를 수 있다. 그러면 정보 가치가 과대평가된다.
- **변경(모든 arm 공통, 2×2 수정 횟수에 포함하지 않는 설계 정의):** backbone = `m_c + MLP([ECFP, 표준화 X0])`. R4 MLP 폭(512-512)은 그대로 두고, 출력층은 0으로 초기화한다. m_c와 X0 표준화는 training unit만으로 적합한다.
- **판정:** 이것은 성능 주장이 아니다. pilot의 chem_only가 train_cell_mean보다 나쁘면 그대로 보고하고, 이 정의를 바꾸지 않는다.

## D-1. XPert 비교군 실행 방식 (실행 전)

- 저자 가중치는 L1000 전체로 학습되어 R7 test 약물 반응이 들어 있을 수 있다. 그래서 쓰지 않고, R4 split으로 처음부터 재학습한다.
- 공개 Zenodo 자료(CC-BY-4.0)에서 UniMol 원자 feature와 HG drug embedding을 쓴다. 구조(InChIKey)로 XPert 약물 색인에 연결하고, 연결되지 않는 약물은 UniMol을 생성하고 HG는 0으로 둔다(flag).
- HG embedding 사전학습에 반응 유래 관계가 쓰였는지는 저장소에서 확인할 수 없다(기본 edge PPI·DTI·DDS). **미확인**으로 기록한다.
- `flash_attn`은 같은 수식의 torch SDPA shim으로 바꾼다. 원 코드의 flash 경로는 원자 padding mask를 쓰지 않으며, 그 동작은 그대로 둔다.
- 원 학습 스크립트는 마지막에 `test` split을 평가한다. 그래서 학습 h5ad의 `test` 칸에는 validation 복사본만 넣는다. R7 test unit은 학습 job에 들어가지 않는다.

## D-2. Phase 0 보정: InChIKey 없는 pert_id (2026-11-18 03:1x, pilot 결과를 보기 전)

- **관측(metadata):** `mapping_ambiguities.tsv`에 `no_inchikey`가 384개 pert_id(340 그룹) 있었다. 그중 308 그룹은 어떤 pert_id에도 InChIKey가 없어서 ChEMBL에 연결되지 못했다.
- **원인:** 연구 B 약물 표의 `inchi_keys` 열이 비어 있다. 구조(`rdkit_smiles`)는 모든 pert_id에 있다.
- **변경:** 그런 pert_id는 같은 표의 RDKit SMILES로 InChIKey를 계산해 연결하고, 출처(`inchikey_source=computed_from_smiles`)를 남긴다. 이름으로는 여전히 연결하지 않는다.
- **처리:** 첫 pilot 15개 job(25732–25746)은 시작 약 37초 뒤 취소했다. 보정 결정은 metadata 표만 보고 내렸다.
  - **정정(사실 기록):** 취소 전에 빠른 arm(chem, concat)은 로그에 이른 step의 validation 점수를 이미 출력했다(9개 로그, 43줄).
  - 그중 내가 읽은 것은 결정 후 확인 과정에서 본 chem lr1e-4의 step 3500·4000 두 줄(val 0.5487, 0.5484)뿐이다.
  - 이 점수는 어떤 설정 선택에도 쓰지 않는다. 해당 run 디렉터리(`p_*`)는 미완료로 남기고, 새 pilot은 run id `pilot_*`로 처음부터 다시 실행한다.
  - 결과에 근거한 수정이 아니므로 수정 횟수 2회에 포함하지 않는다.

## P-1. pilot 결과와 LR 선택 (사전 규칙 적용, 2026-11-18 03:45)

validation source panel ΔRMSE, seed 42, 20000 update 중 최저 checkpoint:

| arm | lr 1e-4 | lr 3e-4 | lr 1e-3 | 선택 |
|---|---|---|---|---|
| chem_only | **0.5461** | 0.5478 | 0.5502 | 1e-4 |
| B00 concat | **0.5464** | 0.5476 | 0.5471 | 1e-4 |
| B01 concat+aug | **0.5477** | 0.5478 | 0.5508 | 1e-4 |
| B10 gated | **0.5450** | 0.5463 | 0.5500 | 1e-4 |
| B11 gated+aug | **0.5446** | 0.5449 | 0.5507 | 1e-4 |

- 참고 기준선(validation source): ECFP+X0 Ridge α=1e4 0.5469, train_cell_mean 0.5538.
- best step은 대부분 8500–11500이었다(B11 lr1e-3만 19000). 대칭 연장 조건(대부분이 마지막 평가)에 해당하지 않아 연장하지 않는다.
- 사전 규칙대로 선택 설정에 seed 43·44를 추가한다.
- fixed graph는 선택된 gated arm(B11, aug on)의 열과 lr 1e-4로 seed 42–44를 실행한다.
- 아직 판정이 아니다. validation 신호 규칙은 3 seed 평균과 bootstrap CI로 판단한다.

## D-3. XPert epoch 상한 (timing run 근거, validation 점수 미사용)

- **관측:** timing run(job 25788, 2 epoch, A40)에서 데이터 적재 약 6분, epoch당 학습 4.7분, validation 0.3분이 걸렸다.
- **결정:** 원 설정 `num_epochs 2500`(patience 50)은 seed당 수백 시간이 걸려 실행할 수 없다. 상한을 **100 epoch**로 고정한다.
  - 100 epoch는 약 560만 표본으로, R7 arm의 학습량(20000×64 ≈ 128만 표본, 약 23 epoch)의 4배 이상이다.
  - 원 설정의 다른 값(lr 0.004, 40 epoch 후 절반, 70 epoch 전 batch-정규화 손실, patience 50, validation PCC 손실로 checkpoint 선택)은 바꾸지 않는다.
- seed 42·43·44를 각각 학습한다. 상한은 비교 결과를 보고 바꾸지 않는다.
