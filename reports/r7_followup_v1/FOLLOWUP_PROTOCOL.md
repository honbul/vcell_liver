# R7 후속 연구 계약 (FOLLOWUP_PROTOCOL) — target attribution v1

이 문서는 후속 버전 `followup-v1`의 실행 계약이다. 원 R7 연구(`reports/evidence_gated_r7/`)의 계약·동결·결과는 바꾸지 않는다. 기계가 읽는 설정은 `configs/r7_followup_v1/followup.json`이다.

- 기획서(비공개, Git 미포함): `VCELL_R6_R7_FOLLOWUP_ANALYSIS_AND_BRANCHING_PLAN_2026-11-18.md`, 76,509 B, sha256 `e7ff35f244160fd57ad1c026aa78f36db19b6dee0925f3bd3ed261949dd10aac`. 적용 범위: §0–3, §8–14, 부록 B–C.
- 기준 버전: 원 R7 브랜치 `research/r7-evidence-gated-mechanism` @ `a3a9eb5`. 후속 브랜치: `research/r7-followup-target-attribution-v1`.
- 코드: `src/r7_followup_v1/`, job: `experiments/r7_followup_v1/`(이름 접두사 `r7f_`), 산출물: `manifests/·reports/·predictions/r7_followup_v1/`. 서버의 모든 cache·run·checkpoint·예측은 후속 worktree의 `assets/`에만 쓴다.
- 원 R7 자산(지식 cache `kg_v1`, checkpoint, Python 환경)과 R4 unit 파일은 **읽기 전용**으로만 쓴다. 지식 자원은 다시 내려받지 않았다.

## 1. 무엇을 넣어 무엇을 예측하는가

| 항목 | 정의 |
|---|---|
| 한 예시(unit) | (세포주, 약물 그룹). CMap LINCS 2020 Level 3, 10 µM·24 h, R4 unit 파일 그대로 |
| 입력 | 그 세포주의 처리 전 RNA(X0, 978 landmark), 약물 구조(ECFP4), 공개 DB의 약물–표적 주석과 지식 그래프 |
| 정답 | 실측 처리군−대조군 평균 ΔRNA(978 landmark). 다른 모델의 출력이나 KG는 정답이 아니다 |
| 숨기는 것 | 평가 약물의 처리 후 RNA(모든 세포주). 평가 약물의 표적 주석은 입력으로 허용한다 |
| 모형 | `Δ̂ = Δ̂_chem + m_available · r`. 주석이 없는 약물(K0)은 모든 arm에서 backbone 예측과 정확히 같다 |

## 2. 질문과 가설

- **H7-Identity(이번의 주 질문):** 실제 약물–표적 대응이 (a) 주석 상태만 아는 추가 용량, (b) 같은 그래프의 seed 없는 경로, (c) 다른 약물의 표적을 배정받은 경우보다 예측에 더 유용한가.
- H7-Delivery, H7-Reliability는 §10 분기표에서 조건이 충족될 때 **하나만** 다룬다.
- `GR − C`만으로는 지식의 가치를 주장하지 않는다. 핵심 대비는 **`GR − M`과 `GR − GS`**다.

## 3. 자료 노출과 결과의 이름

| 자료 | 이번 용도 | 결과의 이름 |
|---|---|---|
| 원 train 약물의 3-fold 재분할 | D3 귀속 대조, 분기 선택, 분기 실험 | source out-of-fold(내부 재분할 평가) |
| 원 validation(이미 원 연구의 선택에 사용됨) | D2 고정 checkpoint 민감도 | 개발 분석 |
| 원 test | 후속 설정 동결 commit 뒤 1회 | 기존 benchmark 후속 평가 |

- 원 R7 최종 평가의 부분 점수는 후속 설정 선택에 쓰지 않는다. 원 실행의 상태는 `STATUS_CURRENT_RUN.md`에 관측 가능한 사실만 적는다.
- test 정답은 `src/r7_followup_v1/final_eval.py` 한 곳에서만 열고, 후속 동결 manifest가 commit되어 있고 worktree가 깨끗할 때만 열린다.
- **Phase E의 순서 규칙(사전 고정):** 후속 재평가는 원 R7의 1회 최종 평가가 끝난 것이 관측된 뒤에만 실행한다. 원 평가가 끝나지 않았으면 재평가를 하지 않고 `NOT_RUN.md`에 남긴다.
- 3-fold OOF 결과로 분기를 고른 뒤 같은 3-fold에서 한 분기 실험은 독립 확인이 아니라 "내부 재분할 안정성"이다.

## 4. source 분할 (label 미사용, `phase_a.py`)

- 원 **train** 약물 그룹 5,802개를 3개 outer fold로 나눈다(seed 20261118). 한 약물 그룹의 모든 세포주 unit은 같은 fold에 있다.
- 층화는 metadata만 쓴다: K군, HCC unit 유무, 세포주 수 구간(1–3 / 4–9 / 10+). 반응 크기는 보지 않았다.
- held fold k의 outer-train 중 15%(`sub ∈ {3k,3k+1,3k+2}`)를 inner-validation(checkpoint 선택용)으로 고정하고, 나머지가 inner-train이다.
- fold마다 inner-train unit만으로 세포주 평균 `m_c`, X0 표준화, **chemistry backbone**, 모든 보정기를 다시 적합한다(§3.3-6). inner-train unit이 없는 세포주는 inner-train 전체 평균을 쓴다(fold 1에서 14개 세포주; 모든 arm에 같다).
- 검증: `fixtures.py` C5(held·inner-val 정답을 잡음으로 바꿔도 적합 통계와 학습 정답 tensor가 변하지 않음), 표 `source_folds.tsv`, `source_fold_audit.tsv`.

## 5. 비교군 (D3)

| ID | 구성 | 역할 |
|---|---|---|
| C | fold별로 재학습한 chemistry backbone만 | 공통 기준 |
| M | X0·ECFP + 주석 metadata 6개(기전 근거 유무, activity 근거 유무, log 표적 수, signed 비율, exact 매핑 비율, 다중 유전자 표적 비율)를 받는 residual MLP | 주석 상태와 추가 용량 대조 |
| G0 | GR과 같은 그래프 보정기. status·m_available은 (공통 augmentation 뒤의) 원래 값, seed 목록만 비움 | 같은 scaffold의 generic 경로 대조 |
| GS | GR과 같은 보정기. 약물–표적 대응만 §6 규칙으로 재배정 | 표적 정체성 대조 |
| GR | 원 연구 수정 1의 B11(gated graph + 결측 학습)을 그대로 재학습 | 평가 대상 |

- 공통: 같은 fold·seed의 **고정** backbone, 같은 batch 열(seed), 같은 augmentation 열(seed+1000; 약물 stub에 대해 뽑으므로 real/null/no-seed에서 동일 — fixture로 확인), MSE, lr 1e-4, batch 64, 20,000 update(= 128만 표본), 500 update마다 inner-validation source ΔRMSE로 checkpoint 선택. 원 연구에서 validation으로 고른 lr을 다시 고르지 않는다.
- **용량 맞춤:** GR 보정기 parameter는 373,359개다. 그중 landmark까지 2-hop 안에 닿지 못하는 1,113개 노드의 identity embedding(35,616개)은 출력에 영향을 줄 수 없으므로, 작동 parameter는 337,743개다. M은 마지막 은닉 폭 125로 337,945개(+0.06%)다. 구조가 다르므로 완전한 용량 동등성은 주장하지 않는다. G0에서는 seed 투영(`ws`, 320개)이 쓰이지 않는다.
- **G0의 범위(결과를 보기 전 결정):** §10의 2·3행과 F2 진입 조건이 G0에 직접 의존하고 추가 계산량이 작으므로, G0도 9개 cell(3 fold × 3 seed) 모두에서 학습한다(§12.2 상한 54 + 9 = 63회 이내).
- 실행량: backbone 9 + M 9 + GR 9 + GS 27 + G0 9 = **63회**. 그 밖의 graph·gate·hop·mask 조합은 돌리지 않는다.

## 6. 표적 재배정 규칙 (GS, `shuffle.py`, label 미사용)

1. split마다 따로(inner-train / inner-validation / held, 그리고 최종 재평가용 train / validation / test) 약물–표적 이분그래프를 만든다.
2. `(A,u),(B,v) → (A,v),(B,u)` 제한적 double-edge swap. 같은 약물 안에 중복 표적을 만들지 않는다.
3. 약물 쪽 stub의 근거·부호·매핑 feature와 tid 묶음은 약물에 남는다 → 약물별 표적 수, status, m_available, augmentation 추첨이 그대로다. split 안 표적 출현 빈도도 보존된다.
4. 같은 층의 표적끼리만 바꾼다: (도달 군[landmark / 2-hop 안 도달 / 도달 불가], out-degree 구간[0 / 1–4 / 5–19 / 20+], inner-train에 seed로 등장했는지).
5. 유효 기준(사전 고정): 모든 split에서 stub의 90% 이상이 다른 표적으로 바뀌고, 원래 표적 집합이 그대로인 약물이 5% 이하. 미달이면 degree 조건 → 도달 조건 순서로 완화한다. **실제로는 가장 엄격한 수준(완화 없음)에서 12개 mapping이 모두 유효했다**(`shuffle_balance.tsv`).
6. mapping은 실행 전에 고정하며 epoch마다 바꾸지 않는다. 같은 약물은 모든 세포주에서 같은 null 표적을 받는다.
7. null mapping seed `{142, 143, 144}`. 각 fold에서 3 mapping × model seed 42/43/44를 모두 돌리고, mapping 평균이 주 GS 값이다. mapping별 값도 함께 낸다. 세 mapping으로 formal permutation p-value를 주장하지 않는다.

## 7. 사전 고정한 하위군 (K0–K3)

도달 여부는 원 KG의 무방향 거리가 아니라 **모델의 실제 계산 경로**로 정했다: seed는 표적 노드의 은닉 상태에 더해지고, 두 번의 message-passing block은 `edge_index[0] → edge_index[1]` 방향으로만 전달하며, 출력은 landmark 노드에서만 읽는다. 따라서 q/status가 같을 때 표적 t의 seed가 landmark l의 출력을 바꿀 수 있는 조건은 `t = l`(0-hop)이거나 길이 1–2의 방향 경로가 있는 것이다(`fixtures.py` C7로 실제 forward와 대조).

| 군 | 정의 |
|---|---|
| K0 | 표적 주석 없음 |
| K1 | 표적 중 landmark가 하나 이상 |
| K2 | 표적이 모두 landmark 밖이고, 한 표적 이상이 2-hop 안에 landmark에 닿음 |
| K3 | 표적이 모두 landmark 밖이고, 어느 표적도 2-hop 안에 landmark에 닿지 못함 |

군은 `target_reachability.tsv`에 결과를 보기 전에 commit했다(commit `a703314`). 그 밖의 subgroup 표는 탐색이다.

## 8. 지표와 통계

- **Primary(원 연구 그대로):** unit별 ΔRMSE(978 유전자) → model seed 평균(GS는 3 mapping × 3 seed 평균) → family 안 unit 평균 → source(비-HCC) family 동일 가중 평균. OOF에서는 세 fold의 held unit을 합쳐(pooled) 계산하고 fold별 값도 낸다.
- 차이는 `candidate − comparator`, 음수가 개선이다.
- 짝 bootstrap: 약물 그룹을 재표집한다(한 약물의 모든 세포주가 함께 움직임), 5,000회, seed 20261118. 모든 대비에 같은 재표집을 쓴다. model seed·세포 수·유전자 수는 독립 반복으로 세지 않는다.
- **핵심 대비 2개(`GR−M`, `GR−GS`)**에는 Bonferroni 동시 구간(각 97.5%)과 Holm 보정 bootstrap p를 쓴다. 나머지는 95% 구간과 함께 "기술(탐색)"로 표시한다.
- 판정 범주(§3.4): 운영 기준 0.001(절대 ΔRMSE)은 원 연구의 값을 그대로 쓰고 사후에 바꾸지 않는다. 범주는 `stats.category`에 구현했다.
- Retrieval(부록 B): 후보는 같은 세포주의 held unit 실측 반응이고 모든 arm에 같다. 동률 보정 Top-1/Top-5/MRR(`*_tie_expected`)과 원 정의(`legacy_midrank_mrr` = 1/중간순위)를 이름을 나눠 함께 낸다. 두 정의의 숫자는 서로 비교하지 않는다.
- HCC(HUH7·JHH5·JHH7, hcc_mean)는 사전 지정 패널로 보고만 한다. 어떤 선택에도 쓰지 않는다.
- 양성 대조(진단): R4의 split-half(검출 plate 교대 배정) 반응이 있는 held source unit에서 half A로 half B를 맞히는 값을 모델과 같은 목표로 비교한다. 독립 배양 반복이 아니며 배포 가능한 예측기가 아니다.

## 9. §10 분기표의 적용 규칙 (D3 점수를 보기 전에 고정)

pooled source OOF의 값으로 판단한다. CI\*는 핵심 대비의 동시 구간, CI는 95% 구간이다.

| 행 | 조건(구현: `score_oof.py`의 `decision_table_rows`) | 분기 |
|---|---|---|
| 1 | CI\*(GR−M) < 0 그리고 CI\*(GR−GS) < 0 | F1: target-set pooling 대 graph |
| 4 | CI(GR−C) > 0 이고, GR 보정 크기 상위 1/3에서 GR−C > 0 | F3: 보정량 수축(α grid) |
| 5 | 1행이 아니고, K1∪K2에서 CI(GR−M) < 0·CI(GR−GS) < 0 인데 K3에서는 아님 | F4: 도달 가능 여부 gate |
| 3 | CI\*(GR−GS) < 0 이고 CI\*(GR−M)이 0을 포함 | M 유지. F2 진입 조건 충족 시에만 F2 |
| 2 | CI(GR−C) < 0 이고 CI\*(GR−M), CI\*(GR−GS), CI(GR−G0)가 모두 0을 포함 | F2 진입 조건 충족 시에만 F2 1회 |
| 6 | 핵심 대비 두 CI\*가 0을 포함하고, 그중 하나라도 −0.001 이하를 포함 | 모델 추가 없이 점검: 학습곡선, 양성 대조, 주석군 효과 범위 |

- 우선순위는 표의 순서(1 → 4 → 5 → 3 → 2 → 6)다. 2행과 6행이 함께 성립하면, F2 진입 조건이 충족될 때만 F2를 하고 아니면 6행의 점검으로 끝낸다.
- **F2 진입 조건(기획서 §11.2, 셋 모두):**
  (a) generic 경로가 보정을 지배한다 — D2에서 주석 unit의 `RMS{r(S)−r(∅)} / RMS(r) < 0.5`(고정 checkpoint 3개 평균), 또는 D3에서 CI(G0−C) < 0 이면서 CI(GR−G0)가 0을 포함;
  (b) 배울 여지가 있다 — split-half 양성 대조가 같은 목표에서 모든 모델보다 낮은 오차와 높은 동률 보정 MRR을 보임;
  (c) 현재 구조가 표적 정체성 이득을 분리하지 못했다(2행 또는 3행).
- F2를 하면 real 표적과 재배정 표적을 **같은 차분 구조**로 각각 재학습하고(TD, TDS), M과 GS 대조를 유지한다. 종료 조건: 같은 대조에서도 real−null 차이가 없으면 구조 확대를 끝내고 M 또는 C를 유지한다.
- 어느 행에도 해당하지 않으면 분기를 실행하지 않고 그 사실을 적는다. 분기는 하나만 실행한다. 실행하지 않은 분기는 `NOT_RUN.md`에 이유와 함께 남기고 숫자를 만들지 않는다.

## 10. 단계

| 단계 | 내용 | 산출물 |
|---|---|---|
| A | D1 label-free 감사, fixture, 원 실행 상태 기록, D2 | `target_reachability.tsv`, `evidence_feature_audit.tsv`, `realized_masking.tsv`, `rewiring_audit.tsv`, `target_shuffle_manifest.json`, `shuffle_balance.tsv`, `fixtures.json`, `checkpoint_target_sensitivity*.tsv` |
| B | plumbing: fold 0 × seed 42, 전 arm(3 mapping 포함) | `plumbing/` |
| C | 3 fold × seed 42/43/44 | `capacity_attribution*.tsv`, `legacy_vs_tie_corrected_retrieval.tsv`, `hcc_panel.tsv`, `positive_control_split_half.tsv` |
| D | §9 규칙으로 고른 분기 1개(결정은 `DECISIONS.md`에 실행 전 기록) | 분기 표 |
| E | 후속 설정 동결 commit → 기존 benchmark 후속 평가 1회(§3의 순서 규칙) | `final/` |

## 11. 최종 재평가(Phase E)에 쓸 모델의 정의 (사전 고정)

- 원 분할(train으로 학습, validation으로 checkpoint 선택)에서 C는 원 연구의 seed별 chem_only checkpoint를 읽기 전용으로 쓴다(같은 계약으로 이미 학습된 backbone).
- 그 고정 backbone 위에 M, G0, GS(142/143/144), GR, 그리고 선택 분기의 후보·직접 대조를 seed 42/43/44로 후속 코드로 학습한다. 원 B11 checkpoint는 덮어쓰지 않으며, 후속 GR과 구분해 적는다.
- 사전 지정 대비: 핵심 `GR−M`, `GR−GS`(Holm); 기술 `GR−C`, `M−C`, `G0−C`, `GS−C`, `GR−G0`, 분기 대비. 패널: source(primary), K1–K3, HCC 세 계열과 hcc_mean.

## 12. 하지 않는 것

- 게이트 값이나 모델 내부 차이를 인과 기전이라고 부르지 않는다.
- HCC 정답으로 어떤 것도 조정하지 않는다. test 점수로 학습 시간·계수를 고르지 않는다.
- 원 연구의 운영 기준 0.001을 사후에 완화하지 않는다.
- 외부 비교 모델(XPert)이 끝나지 않았으면 외부 우월성 주장만 보류한다.
