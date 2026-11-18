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

## V-1. validation 신호 판정 (사전 규칙, 3 seed, job 25793)

- 3-seed 평균 validation source ΔRMSE:

| arm | ΔRMSE |
|---|---|
| chem_only | 0.54595 |
| B00 | 0.54640 |
| B01 | 0.54659 |
| B10 | 0.54536 |
| B11 | 0.54532 |
| fixed_graph | 0.54548 |

- B11 − chem_only = −0.00063 [−0.00196, +0.00070]. 판정 조건 "≤ −0.001"과 "CI < 0"을 둘 다 충족하지 못했다 → **validation 신호 없음**.
- B11 < B01은 성립했다(−0.00127 [−0.00229, −0.00031]). B11 vs B10은 −0.00004, B11 vs fixed는 −0.00016으로, 둘 다 CI가 0을 포함한다.
- 수축 점검: B11 보정 RMS 0.029 / chemistry RMS 0.173 = 0.17. 보정이 0으로 수축하지는 않았다.

## M-1. 수정 1 — 고정 chemistry backbone 위의 기전 보정 (실행 전 기록)

- **관측(validation, 주석 상태별 B11 − chem_only):**
  - complete −0.00112 [−0.00314, +0.00056]
  - partial −0.00010 [−0.00202, +0.00211]
  - missing +0.00298 [−0.00138, +0.00366]
  - missing 약물에서 B11의 예측은 backbone만으로 나온다(m_available=0). 그런데도 함께 학습한 backbone이 단독 chem_only보다 점추정으로 나빴다.
  - seed별 missing ΔRMSE: B11 0.5967/0.5949/0.6019, chem 0.5924/0.5972/0.5950. 편차가 크다.
- **가능한 원인(가설, 미검증):** 공동 학습에서 보정 항이 주석 있는 약물의 신호 일부를 가져가 backbone 학습이 달라졌다.
- **기획서 근거:** 수정표 1행("근거가 많은 약물에서만 개선, 결측에서 악화 → fallback 강도 보완").
- **변경(하나):** 모든 지식 arm(B00, B01, B10, B11, fixed)의 backbone을 같은 seed의 chem_only checkpoint로 초기화하고 **고정**한 뒤, 보정 항만 학습한다(같은 lr 1e-4, 같은 update 예산과 sampler).
  - seed 대응: s42 → `pilot_chem_lr1e-4_s42`, s43 → `sel_chem_lr1e-4_s43`, s44 → `sel_chem_lr1e-4_s44`.
  - 이렇게 하면 missing 약물의 예측은 chem_only와 정확히 같아진다.
- **예측(사전):**
  - missing stratum 차이는 0이 된다.
  - complete/partial의 차이는 지금과 비슷한 크기(−0.001 내외)로 예상한다.
  - 전체 B11 − chem_only는 −0.0004~−0.0008 정도로 예상하므로, ≤ −0.001 조건은 **충족하지 못할 가능성이 높다**고 미리 적는다.
- **판정 기준:** 사전 등록한 validation 신호 규칙을 그대로 적용한다(변경 없음). 보조로 annotated(complete+partial) unit에서의 B11 − chem_only CI를 보고한다.

## D-4. XPert 실행 디렉터리 분리 (운영 오류 수정, 점수 미관측)

- 원 스크립트는 출력 폴더 이름을 시작 시각(초)으로 정한다. seed 43 run(job 25790)이 다른 run과 같은 초에 시작해 폴더 충돌로 실패했다.
- 또 학습 후 "가장 최근 폴더"를 복사하던 방식은 병렬 run 사이에 checkpoint가 섞일 위험이 있었다.
- 그래서 run 25789·25791을 2 epoch 무렵 취소했다. 이후에는 run마다 별도 작업 디렉터리를 쓰고, 폴더가 정확히 1개인지 확인한다. 새 run id는 `xpr_e100_s{42,43,44}`이다. 학습 설정은 바꾸지 않는다.

### M-1 결과 (job 25816, `val_signal_m1*`)

3-seed 평균 validation source ΔRMSE:

| arm | ΔRMSE |
|---|---|
| chem_only | 0.54595 |
| B00 | 0.54544 |
| B01 | 0.54512 |
| B10 | 0.54513 |
| B11 | 0.54495 |
| fixed | 0.54503 |

- **사전 예측과 비교:**
  - missing stratum 차이 0 → 예측대로(구성상 0).
  - complete: B11 − chem = −0.00137 [−0.00240, −0.00030].
  - partial: −0.00013 [−0.00163, +0.00059].
  - 전체 B11 − chem_only = **−0.00100 [−0.00162, −0.00035]**. 사전 예측(−0.0004~−0.0008, 충족하지 못할 가능성이 높음)보다 컸다.
- **V-2 판정(규칙 그대로):**
  - 다섯 조건이 모두 참 → **validation 신호 성립.**
  - 단, ≤ −0.001 조건은 −0.0010041로 경계에서 충족했다(여유 0.000004).
  - B11 < B01 / B10 / fixed는 점추정만 낮다(−0.00018, −0.00018, −0.00008). 모든 CI가 0을 포함한다.
  - 다섯 기전 arm 모두 chem_only보다 0.0005–0.0010 낮고 CI도 0 미만이다.
- **해석 범위:** validation 기준으로 "외부 지식 정보의 추가 가치" 신호는 있다. "증거·세포 조건부 활용 방식"(B11 vs concat/fixed)의 우위는 확인되지 않는다.
- **수정 2는 실행하지 않는다.** 신호가 성립했고, 이후 설정은 동결 대상이다.
- 수정 1 B11의 best step은 seed 42가 20000(마지막 평가)이고 43·44는 16500·12000이다. 대칭 연장 조건(대부분이 마지막)에 해당하지 않는다.

## A-1. 귀속 실험 (validation 신호 성립에 따른 사전 지정 실행)

수정 1 B11 설정(gated, aug on, lr 1e-4, seed 대응 고정 chem backbone)에서 하나씩 바꿔 3 seed씩 재학습한다.

1. **재배선 그래프 재학습** `--graph rewired`: 채널별 도착점 순열로 차수를 보존하고, 원래 edge는 15.6%만 남는다(`rewired_graph.json`).
2. **근거 feature 제거** `--evidence off`: 근거 종류·작용 부호·매핑 flag를 표적 존재 표시 하나로 바꾸고, 약물 status 벡터도 비운다.
3. **결측 학습 제거:** 수정 1의 B10을 사용한다(이미 실행됨).
4. **gate 제거:** 수정 1의 fixed graph를 사용한다(이미 실행됨).

판정은 validation 3-seed 평균과 B11 대비 paired bootstrap으로 하며, test에서도 같은 대비를 보고한다.

### A-1 결과 (validation, job 25824, `val_attribution*`)

| 대비 | 차이 | 95% CI |
|---|---|---|
| B11 − chem_only | −0.00100 | [−0.00162, −0.00035] |
| B11_rewired − chem_only | −0.00097 | [−0.00156, −0.00036] |
| B11_evidence_off − chem_only | −0.00096 | [−0.00150, −0.00037] |
| B11 − B11_rewired | −0.00004 | [−0.00019, +0.00013] |
| B11 − B11_evidence_off | −0.00004 | [−0.00025, +0.00018] |
| B11 − B10(결측 학습 제거) | −0.00018 | [−0.00047, +0.00011] |
| B11 − fixed(gate 제거) | −0.00008 | [−0.00033, +0.00022] |

- **관측:** edge의 85%를 무작위로 바꾼 그래프로 다시 학습해도, 근거 종류·부호·매핑 flag를 지워도 chem_only 대비 이득이 거의 그대로였다. gate 제거와 결측 학습 제거도 차이를 만들지 않았다.
- **확인된 범위:** validation 이득은 OmniPath 관계 구조, 근거 feature, 조건부 gate, 결측 학습 **어느 것에도 귀속되지 않는다**.
- **가설(미검증):** 남은 공통 요소는 두 가지다. 이 실험으로 둘을 나눌 수는 없다.
  - (a) 어떤 표적 노드가 seed되는가(표적 정체성 = 약물 간 표적 공유 정보)
  - (b) 주석 있는 약물에만 붙는 추가 보정 용량(+ 세포 상태·약물 embedding 입력)
- **미실행:** 표적 정체성을 약물 사이에서 섞어 재학습하는 대조. 사전 등록에 없었고, 수정 횟수도 남기지 않았다(수정 2 미사용 결정 유지). 결과 보고에 미실행으로 남긴다.

## D-5. 절차 위반 기록 — mock 평가가 HCC validation 수치를 계산함 (2026-11-18 05:38)

- **무엇이 있었나:** 최종 평가 배선 검사(`final_eval --mock`, job 25826)는 test 대신 validation unit에서 같은 코드를 돌린다. 그런데 이 코드가 HCC panel(HUH7·JHH5·JHH7, hcc_mean)까지 계산해 로그와 `manifests/evidence_gated_r7/mock/`에 썼다. 이는 protocol의 `no_hcc_selection: no HCC number is computed before the freeze`를 어긴 것이다.
- **내가 본 것:** 로그 끝 부분(v0_fixed_graph의 HCC 행 몇 줄)과 mock 대비표의 B11 − chem_only 행 다섯 줄이다. 예: HCC-mean −0.00092 [−0.00265, +0.00089].
- **영향 범위:**
  - 동결 대상 모델, seed, 대비 목록, stress 모델은 mock 실행 **전에** commit `b6c72e0`(`freeze_args.sh`)으로 고정되어 있었다.
  - XPert 설정(D-3)도 그 전에 고정되었다.
  - 이 수치를 본 뒤 어떤 설정도 바꾸지 않는다. mock 결과 파일은 숨기지 않고 `mock/`에 그대로 commit한다.
- **재발 방지(2026-11-18 22:4x 추가):** 처음에 '없음'이라고 적은 것은 잘못된 판단이었다. `final_eval`의 mock 모드가 HCC panel을 **계산하지 못하도록** 고쳤다(`MOCK_PANELS = ["source"]`, 주석 결측 stress의 HCC 행 제외, 결과 표에 HCC가 들어가면 중단하는 assert). 위반 자체는 취소되지 않으며, RESULTS·CLAIMS·EXPOSURE에 그대로 남긴다.

## F-1. 동결과 최종 평가 (2026-11-18 13:2x)

- 동결 commit은 `a7ab705`(`freeze_manifest.json`)이다. 그 뒤 XPert test 예측(job 25973–25975, 처리 후 RNA 칸 차이 0.0)과 최종 평가(job 25976)를 한 번 실행했다.

## F-2. 사전 규칙에 따른 최종 판정 (test, source panel, 3 seed 평균 unit RMSE, drug-group bootstrap)

| 규칙 | 값 | 판정 |
|---|---|---|
| 정보 가치: (B11 − chem_only) CI < 0 이고 점추정 ≤ −0.001 | −0.00076 [−0.00147, +0.00008] | **미충족** |
| 활용 방식의 가치: 추가로 (B11 − B01), (B11 − fixed) CI < 0 | +0.00021 [−0.00070, +0.00121]; +0.00006 [−0.00022, +0.00041] | **미충족** |
| 강한 기준선 대비 | B11 − ECFP+X0 Ridge −0.00152 [−0.00572, +0.00255]; B11 − XPert −0.00005 [−0.00540, +0.00722] | 주장 불가 |
| fallback 피해(missing stratum B11 − chem > +0.002) | 0(구성상 동일) | 없음 |
| 수축 점검(보정 RMS / chemistry RMS < 1%) | 약 0.2 | 수축 아님 |
| HCC-mean (B11 − chem_only) | −0.00020 [−0.00098, +0.00059] | 미확립 |

**결론:** 제안한 증거·세포 상태 조건부 기전 모듈(B11)은 **채택하지 않는다.** validation에서 경계선으로 선 신호(−0.00100)는 test에서 재현되지 않았다(−0.00076, CI가 0 포함). 귀속 실험에서는 validation과 test 모두 B11 ≈ 재배선 ≈ 근거 제거 ≈ fixed였다. 따라서 그래프 구조·근거 종류·조건부 gate·결측 학습의 기여는 확인되지 않았다.

## D-5 보완 — "HCC를 보고 설정을 다시 고르지 않았다"의 검증 (2026-11-18 13:5x)

D-5의 위반(동결 전에 HCC validation 수치를 계산하고 일부를 봄)은 되돌릴 수 없다. 대신 그 수치가 설정 선택에 들어가지 않았음을 기계적으로 확인했다(`manifests/evidence_gated_r7/hcc_deviation_check.json`).

| 확인 항목 | 결과 |
|---|---|
| 동결 목록 commit `b6c72e0`(04:54:27) | mock job 25826 시작(05:20:56)보다 **먼저**다 |
| 그 뒤 `freeze_args.sh` 변경 | 없음 |
| `freeze_manifest.json`(동결 `a7ab705`, 13:23)의 모델·run id·XPert run·대비·stress 목록 | `b6c72e0`의 목록과 **완전히 같다** |
| `b6c72e0` 이후 동결까지 바뀐 코드 | `ledger.py`(기록 문구), `val_signal.py`(XPert validation 보고) 두 파일뿐. 예측 경로는 그대로 |
| Ridge α 선택 | validation source panel(job 25722) |
| XPert epoch 상한 | D-3 |

두 항목 모두 mock 이전이다.

**결론:** 절차 규칙(동결 전 HCC 수치 미계산)은 어겼다. 그러나 HCC 수치를 본 뒤 고른 설정은 없다. 결과 해석에는 이 위반을 그대로 남긴다.

## D-1 보완 — XPert HG drug embedding 출처 점검 (metadata만, RNA 미사용)

`xpert_hg_check`(job 26028, `xpert_hg_provenance_check.json`) 결과:

- **DDS(약물–약물) 관계 287,834개:** 가중치와 ECFP4 Tanimoto의 Pearson 0.995, Spearman 0.987이다. 가중치 하한은 0.50이고, 표본 edge의 Tanimoto 중앙값은 0.586이다(무작위 쌍은 0.103). → DDS는 **화학 구조 유사도 관계**다(RNA 반응 유래가 아님).
- **DTI 12,890개:** 약물–표적 지식이다. PPI는 단백질 상호작용이다.
- 원 사전학습 코드의 기본 관계는 PPI·DTI·DDS다. 반응에서 나왔을 수 있는 PCC·GSEA 관계 파일은 공개 묶음(`HG_data`)에 **포함되어 있지 않다**.
- **남은 미확인:** 공개된 `HG_drug_embeddings.npy`가 실제로 기본 관계만으로 학습되었는지는 저장소로 확인할 수 없다(사전학습 로그 미공개). 따라서 XPert 결과는 "같은 split·실측 재학습, HG 입력은 구조·표적 지식 기반으로 확인된 범위"로 보고한다.

## A-2. 동결 후 추가 비교군 — 입력을 맞춘 XPert 구조 비교 (실행 전 기록, 2026-11-18 14:0x)

- **이유:** 기획서 §6은 "XPert 공식 전체 설정 비교"와 "ECFP 등 입력을 맞춘 구조 비교"를 나눠 보고하라고 했다. 본 실행에서는 앞의 것만 했고, 뒤의 것은 미실행으로 남겼다. 공식 설정은 UniMol 원자 feature와 출처가 일부만 확인된 HG embedding을 쓰므로, 입력 차이와 구조 차이가 섞여 있다.
- **성격:** R7 test를 이미 연(job 25976) 뒤에 추가하는 **비교군**이다. R7 모델·판정·사전 대비는 바꾸지 않는다(기존 결과 덮어쓰기 없음). 결과는 별도 표(`addendum_*`)로 보고한다.
- **설정(실행 전 고정, 튜닝 없음):**
  - XPert 공식 코드·학습 설정은 본 XPert run과 같다: config_l1000, lr 0.004, batch 128, 100 epoch 상한, patience 50, validation PCC 손실로 checkpoint 선택, seed 42·43·44.
  - 약물 입력은 `--drug_feat morgan` 경로를 쓴다. 우리 모델과 같은 ECFP4 2048 bit를 1024 bit로 접어서(앞뒤 반을 OR) 넣는다. 원 코드의 morgan 입력 차원이 1024이기 때문이다.
  - HG drug embedding은 0이다(출처 미확인 입력 제거).
  - PPI gene vector, X0, cell index는 본 run과 같다.
- **평가:** 학습 후 checkpoint sha256을 별도 addendum 동결 파일에 commit한다. 그 뒤 test 예측을 한 번 하고, 이미 저장된 동결 모델 예측(sha256 확인)과 같은 지표·bootstrap으로 비교한다. 사전 지정 대비는 다음과 같다.
  - xpert_ecfp − chem_only
  - xpert_ecfp − B11
  - xpert_ecfp − xpert
  - xpert_ecfp − ECFP+X0 Ridge
- **판정 범위:** 이 비교는 XPert 구조의 경쟁력을 입력을 맞춰 보는 참고 결과다. 어떤 결과가 나와도 R7 제안 모듈의 미채택 판정은 바뀌지 않는다.

### A-2 결과 (addendum 동결 `28cca70`, 평가 job 26569, `addendum_*`)

- validation(보고용): XPert-ECFP 0.55824. 공식 XPert 0.54802, chem_only 0.54595보다 높다.
- test source(3 seed 평균): XPert-ECFP 0.54385 ± 0.0021, centred Pearson 0.136, MRR 0.322.

| 대비 | 차이 | 95% CI |
|---|---|---|
| XPert-ECFP − chem_only | +0.00711 | [+0.00173, +0.01222] |
| XPert-ECFP − B11 | +0.00786 | [+0.00239, +0.01295] |
| XPert-ECFP − 공식 XPert | +0.00781 | [+0.00313, +0.01482] |
| XPert-ECFP − ECFP+X0 Ridge | +0.00634 | [−0.00101, +0.01372] |

- 처리 후 RNA 칸 검사: validation·test 모두 차이 0.0이다.
- 재사용한 동결 예측의 점수는 본 평가(job 25976)와 같다(B11 0.535984, chem_only 0.536741).
- **해석(관측):** 약물 입력을 ECFP로 맞추고 HG를 제거하면, XPert 구조는 R7 chemistry backbone과 B11보다 ΔRMSE가 높다(CI > 0).
  - 공식 XPert가 B11과 같은 범위였던 것은 UniMol·HG 입력과 함께일 때다.
  - UniMol과 HG 중 무엇의 기여인지는 둘을 동시에 바꿨으므로 분리하지 못했다.
- R7 제안 모듈의 미채택 판정은 그대로다.


## D-1 재보완 — XPert HG embedding 이웃 구조 검사 (job 26580, metadata만)

사전학습 로그가 공개되지 않아 "기본 관계(PPI·DTI·DDS)만으로 학습했는가"는 직접 확인할 수 없다. 대신 공개된 embedding 자체가 그 관계들과 정합하는지 검사했다(`xpert_hg_embedding_check.json`). 약물 1,500개를 뽑아 embedding 공간의 최근접 10개를 보았다.

| 측정 | embedding 이웃 | 무작위 기준 |
|---|---|---|
| 같은 DTI 표적을 공유하는 비율 | 0.477 | 0.0077 |
| DDS 관계로 연결된 비율 | 0.231 | 0.0071(전체 edge 밀도) |
| ECFP Tanimoto 중앙값 | 0.132 | 0.103 |

- **관측:** 이웃 구조는 표적 공유(62배 농축)와 DDS 연결(32배)로 설명된다. 화학 구조 유사도 자체의 상승은 작다.
- **해석(증거, 증명 아님):** 공개 embedding은 표적·약물 관계 기반 사전학습과 정합하며, RNA 반응 유래 관계의 징후는 보이지 않는다. 구조와 정렬된 미기재 관계를 배제하지는 못한다.
- 이 불확실성이 결론에 영향을 주지 않도록, HG를 전혀 쓰지 않는 비교군(A-2)을 별도로 실행했다.
