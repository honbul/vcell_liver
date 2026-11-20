# R10 v2 결과 — 학습·평가 계약 표(E1a)와 선언/실제 조립 진단(E2)

작성 2026-11-19. 근거: private plan 2026-11-19 §7. 브랜치 `research/r10-contract-assembly-v2`(기점 R10 `0c35756`). **freeze commit `2f43dcb`**(실행 명세 [`RUN_SPEC.md`](RUN_SPEC.md), [`configs/r10/contract_matrix_v2.json`](../../../configs/r10/contract_matrix_v2.json)). 서버 run `assets/runs/r10_contract_matrix_v2/`(eval job 26791, 분석 job 26945).

표기: **[관측]** 측정값 · **[추론]** 측정에서 끌어낸 해석 · **[미검증]** 측정하지 않은 것. 구간은 약물 단위 bootstrap 95%(B=2000). 부호는 a − b, 음수 = a의 실제(재판독) 구성비 TV가 낮음(좋음).

**노출 지위.** test 38약물은 v1에서 이미 채점된 자료다. 아래 test 수치는 **이전에 노출된 자료의 동결된 후속 비교**이며 blind·독립 test가 아니다. development 35약물은 모든 learned head의 선택 집합(낙관적)이다.

## 0. 요약

- [관측] 주 대비 6개(test, 1 µM=E1000, actual TV, Holm): **cross 평가 칸**에서는 cross 학습 D11이 same 학습 D11보다(P2 −0.0122 [−0.0169, −0.0075]), cross D10이 t*_cross보다(P4 −0.0119 [−0.0174, −0.0062]), cross D11이 W_cross보다(P6 −0.0102 [−0.0151, −0.0051]) 낮았다 — 세 개 모두 실용 기준 0.005 이상의 차이로 확립.
- [관측] **same 평가 칸**에서는 same 학습 D11 − cross 학습 D11 = −0.0033 [−0.0095, +0.0021](P1, 차이 미확립·동등도 아님), same D10 − t*_same = +0.0002 [−0.0033, +0.0039](P3), same D11 − W_same = +0.0003 [−0.0038, +0.0043](P5). P3·P5는 사전 margin ±0.005의 90% 구간 안에 있어 **margin 내 동등**으로 판정됐다.
- [추론] 같은 replicate 라벨 계약에서는 learned π-head·분해 조립이 같은 계약에 맞춘 단순 기준선(전역 mixing weight t*_same, 전역 affine W_same)을 넘지 못했다. 분해 구조의 우위(D11 > t*, W)는 cross 계약에서만 관측된다. 원인 비율(생물학/환경)은 이 설계로 식별하지 않는다.
- [관측] E2: 선언 π와 재판독 π의 차이(D11 test E1000 cross 평균 0.0159)가 있지만, 실제 TV는 선언 TV와 같거나 약간 낮았다(D10 matched 칸 gap ≈ 0). development matched 칸에서 선언/실제 기반 대비의 부호가 바뀐 경우가 없었다. 사전 규칙상 **조건부 패키지(§7.2.3 목적함수 arm, §7.2.4 hard partition, §7.1.5 source OOF)는 열리지 않았다.**
- [관측] 재현 점검: v2 평가는 v1의 같은 head·triplet·seed 행을 재현했다(최대 절대 차이: test D11[cross] comp_tv 2.6e-8, 나머지 ≤ 1e-14 또는 0).

## 1. 정정·명확화(상태·계약 문서)

- 목적: v1 문서의 해석 범위 정정(plan §7 도입부, §7.1.3, §7.3.2 끝).
- 수정 내용: [`../ERRATA_AND_CLARIFICATIONS_v2.md`](../ERRATA_AND_CLARIFICATIONS_v2.md) 추가, v1 문서 5개 머리에 정정 노트 링크(원문·수치 보존).
  - E-1 test에서 채점된 learned arm은 선택된 D11[cross] 하나 → head별 factorial 효과의 test 확인이 아님.
  - E-2 C3 t*는 train-cross 적합이며 freeze 이후 추가 → same 계약 비교 기준선으로는 계약 불일치.
  - E-3 same은 같은 replicate 라벨일 뿐 plate 효과 제거가 아님. [관측] 적격 same triplet 2,057개 모두 low·high endpoint plate가 다르고 query는 low plate(1,024) 또는 high plate(1,033)와 같다; cross 2,076개는 query가 두 endpoint plate와 모두 다르다(`plate_relation_counts.tsv`는 dev/test 부분집합).
  - E-4 CI가 0을 포함한 D10 − mixture_t는 동등성 증거가 아님; 0.57 비율은 원인 분해가 아님.
  - E-5 declared/reread 불일치는 채점 버그가 아님.
- 의미/다음 행동: v1 결론 중 "same에서 learned π-head = 전역 mixing weight"는 v2에서 같은 계약 t*_same과 사전 margin으로 다시 판정했다(§2 P3).

## 2. R10-E1a — 학습 계약 × 평가 계약 표

### 목적
cross 학습 head의 same 악화가 계약 불일치와 관련되는지, same 학습 head가 같은 계약의 단순 기준선(t*, W)을 넘는지 본다.

### 수정 내용
- `evaluate.py`: 모든 행에 `train_contract`, `assembly_version`, `atlas_version` 기록; head checkpoint의 계약과 인자 불일치 시 중단; E2 진단 열(`diag_*`).
- `aggregate.py`: KEYS에 세 key 추가, v1 파일은 `v1_unlabelled`로 명시, 학습 계약이 다른 같은 method가 한 약물·칸에서 평균되면 중단(`_unique_methods`), `arm_label()`(예: `D11[same]`).
- `train.py`: checkpoint·selection json에 `train_contract` 기록. `ledger.py`: 로그 prefix 인자.
- 신규 `contract_matrix.py`(사전 명세를 읽어 주/보조 대비·Holm·동등성·층·E2를 계산), `replay_v1.py`, `scripts/r10/tasks.sbatch`, `scripts/r10/contract_matrix_v2_tasks.sh`.

### 실험 입력/정답
- 재사용(hash가 v1 artifact index와 모두 일치, `manifests/.../artifact_registry.tsv`): cross π/Q/W seed 0·1·2(`r10_b3_v1`), same π/Q seed 0·1·2(`r10_c2_same_v1`), t*_cross 0.325(`r10_c3_tstar_v1.json`), atlas·triplet·cache 47 파일.
- 신규: **W_same** 3 seed(같은 코드·격자·same development ED 선택; 세 seed 모두 wd 1e-4, epoch 15/20/15), **t*_same**(train 약물 same triplet; E100 0.35, E1000 0.40).
- 정답: query well의 재판독 구성비(입력에는 들어가지 않음). 입력 계약은 v1과 동일.

### 비교 조건
모든 칸에 D00, t*[cross], t*[same], D10/D01/D11/W[해당 학습 계약]. 평가는 dev·test × E100·E1000 × same·cross triplet, 총 30 task. 칸 안 비교는 두 arm이 모두 있는 triplet 교집합, 학습 계약 효과는 같은 query well, 평가 계약 간 비교는 공통 적격 (세포주, 약물, episode) 집합. 집계 seed → triplet → replicate 방향 → 세포주 → 약물.

### 결과

**2×2 표: test, E1000, actual 구성비 TV**(칸의 모든 arm 공통 triplet; 전체는 `matrix_2x2_comp_tv.tsv`)

| arm(학습 계약) | same 평가 | cross 평가 |
|---|---|---|
| D00(계약 무관) | 0.0942 | 0.1099 |
| t*[same] | 0.0898 | 0.1053 |
| t*[cross] | 0.0876 | 0.1028 |
| D10[same] / D11[same] / W[same] | 0.0900 / 0.0898 / 0.0895 | 0.1034 / 0.1020 / 0.1038 |
| D10[cross] / D11[cross] / W[cross] | 0.0925 / 0.0931 / 0.0920 | 0.0908 / 0.0898 / 0.1000 |

**주 대비**(`primary_contrasts.tsv`; test E1000, n = 38약물, 214–215 triplet)

| id | 대비 | 평가 | 추정 [95%] | 90% 구간 | Holm p | 판정 |
|---|---|---|---|---|---|---|
| P1 | D11[same] − D11[cross] | same | −0.0033 [−0.0095, +0.0021] | [−0.0083, +0.0013] | 0.77 | 차이 미확립 |
| P2 | D11[cross] − D11[same] | cross | −0.0122 [−0.0169, −0.0075] | | < 0.001 | 실용적 차이 확립 |
| P3 | D10[same] − t*[same] | same | +0.0002 [−0.0033, +0.0039] | [−0.0028, +0.0033] | 1 | margin(±0.005) 내 동등 |
| P4 | D10[cross] − t*[cross] | cross | −0.0119 [−0.0174, −0.0062] | | < 0.001 | 실용적 차이 확립 |
| P5 | D11[same] − W[same] | same | +0.0003 [−0.0038, +0.0043] | [−0.0033, +0.0037] | 1 | margin 내 동등 |
| P6 | D11[cross] − W[cross] | cross | −0.0102 [−0.0151, −0.0051] | | < 0.001 | 실용적 차이 확립 |

(p < 0.001은 B=2000에서 0 쪽 bootstrap 표본이 없음을 뜻한다.)

**보조(보정 없음, `secondary_contrasts.tsv`, `contract_interaction.tsv`, `strata_primary_contrasts.tsv`)**
- [관측] test E1000 same 평가: 어떤 learned arm도 D00 대비 구성비 TV 차이가 0을 확실히 넘지 않았다(D11[same] −0.0044 [−0.0095, +0.0009]); 단순 기준선 t*[same] −0.0044 [−0.0072, −0.0016], W[same] −0.0047 [−0.0085, −0.0009], t*[cross] −0.0065 [−0.0112, −0.0018].
- [관측] 상태 내부·전체 분포(test E1000): same 평가에서 cross 학습 D11은 D00보다 나쁘고(상태 내부 평균 RMSE +0.073 [+0.059, +0.087], ED +0.071 [+0.037, +0.106]; v1 재현), same 학습 D11은 그 악화가 없다(D11[same] − D11[cross]: 상태 내부 −0.077 [−0.088, −0.067], ED −0.103 [−0.122, −0.085]). same 평가에서 D11[same] − W[same]은 상태 내부 +0.001 [−0.006, +0.011], ED +0.002 [−0.011, +0.016](차이 미확립). cross 평가에서 D11[cross] − W[cross]은 상태 내부 −0.014 [−0.019, −0.007], ED −0.023 [−0.038, −0.009].
- [관측] Q가 π에 더하는 것(D11 − D10): cross 평가·cross 학습에서 상태 내부 −0.076 [−0.097, −0.054], ED −0.103 [−0.139, −0.064], 구성비 TV −0.0010 [−0.0032, +0.0011]; same 평가·same 학습에서 상태 내부 −0.004 [−0.013, +0.006], ED −0.012 [−0.025, +0.002].
- [관측] contract_interaction(test E1000, 공통 111 세포주×약물): 구성비 TV D11 −0.0152 [−0.0220, −0.0087], D10 −0.0145, W −0.0060, t* −0.0003 [−0.0013, +0.0008]; ED D11 −0.210 [−0.239, −0.182], W −0.203. D00 자체의 평가 계약 차이(cross − same): TV +0.0149 [+0.0064, +0.0234], ED +0.178 [+0.142, +0.220].
- [관측] 2차 episode·role(같은 대비, 보조): test E100에서는 D11[same] − D11[cross](same 평가) −0.0085 [−0.0127, −0.0046], D10[same] − t*[same] −0.0064 [−0.0171, +0.0006], D11[same] − W[same] −0.0028 [−0.0064, +0.0012]. development E1000에서는 D11[same] − W[same](same) −0.0114 [−0.0196, −0.0049], D10[same] − t*[same] −0.0063 [−0.0165, +0.0017](development는 head 선택 집합).
- [관측] 층(test E1000): P2/P4/P6의 cross 이득은 A549·K562에서 뚜렷하고 MCF7에서는 0 근처(P4 MCF7 −0.0016 [−0.0066, +0.0035], P6 MCF7 +0.0018 [−0.0044, +0.0075]); 두 replicate 방향 모두 같은 부호. same 평가에서는 replicate 방향과 query plate 관계가 완전히 겹친다(rep1→rep1 = query가 high plate, rep2→rep2 = low plate) — 둘을 분리할 수 없다.
- [관측] t*의 계약 민감도: t*_same(0.35/0.40)과 t*_cross(0.325) 차이는 작고, test same 평가에서 t*[cross]가 t*[same]보다 오히려 낮았다(t*[same] − t*[cross] +0.0021 [+0.0001, +0.0041]).

### 의미
- [추론] "같은 계약에서 학습하면 좋아지는가": cross 평가에서는 예(P2), same 평가 1 µM에서는 미확립(P1; 100 nM 보조 대비에서는 same 학습이 나았다). 학습 계약의 관련성은 상태 내부·ED에서 훨씬 크다(같은 query well 비교).
- [추론] "π-head가 단순 곡선 위치 조정을 넘는가": cross에서만 예(P4). same에서는 t*_same과 margin 내 동등(P3) — v1의 "구분되지 않음"이 이번에는 사전 margin으로 동등 판정됐다(test, 노출 자료의 후속 비교라는 범위 안에서).
- [추론] "분해 구조가 전역 보정보다 좋은가": cross에서 TV와 상태 내부·ED 모두 D11 < W(P6 및 보조). same에서는 TV(P5)가 margin 내 동등이고 ED도 차이 미확립 → same 계약에서 구조 특유 우위는 확립되지 않음.
- [추론] cross 계약 이득은 replicate 간 이동이 있는 평가에서 나타나고, contract_interaction이 단순 t*에는 거의 없고(−0.0003) learned head·W에서 크다. 이는 learned 모델이 학습 계약의 replicate 관계를 학습한다는 해석과 일치하지만, 생물학/기술 원인 분해는 아니다 [미검증].

### 다음 행동
- 같은 replicate 라벨 계약에서 쓸 기준은 단순 t*·W로 충분하다는 결과를 기록하고, 더 큰 구조를 이 계약의 개선 근거로 쓰지 않는다.
- cross 계약의 이득이 "새 replicate/batch로의 전이"인지는 기존 두 replicate만으로 판단할 수 없다 → 독립 batch 자료 요구(`INDEPENDENT_DATA_REQUIREMENTS.md`).

## 3. R10-E2 — 선언값 vs 실제 조립값 진단(새 학습 없음)

### 목적
선언 π(head 출력)와 생성 cloud의 재판독 π 차이가 성능이나 순위를 제한하는지 진단(plan §7.2.1–7.2.2).

### 수정 내용
`methods.assemble`·mixture에 endpoint 진단(membership entropy, 상태별 effective n 최소, 저지지 상태 질량, π 이동 TV, Q 이동 RMS) 추가; `contract_matrix.e2()`에서 세 TV를 학습·평가 계약·arm·episode·세포주·약물별로 연결.

### 실험 입력/정답
E1a와 같은 평가 행(`head_comp_tv` = TV(π_declared, π_query), `comp_tv` = TV(π_actual, π_query; 주 지표), `assembly_tv_head_vs_reread` = TV(π_declared, π_actual)).

### 결과 (`../assembly_audit_v2/`)
- [관측] D00 허용오차 점검 통과: 1,660 triplet에서 max |declared − actual| = 2.6e-9(≤ 1e-6).
- [관측] test E1000 TV(π_declared, π_actual) 평균: D10[cross] 0.012, D11[cross] 0.016(v1 0.016과 일치), D10[same] 0.007, D11[same] 0.011, D01 0.012–0.018. 90% 분위 ≤ 0.035.
- [관측] 실제 − 선언 TV(gap): matched 칸 D10은 0 근처(test cross −0.0007 [−0.0026, +0.0010], same −0.0005 [−0.0016, +0.0006]); D11·D01은 음수(실제 cloud가 선언보다 query에 더 가깝다; 예 D11[cross] cross −0.0017 [−0.0036, +0.0001]).
- [관측] 진입 규칙(development E1000 matched 칸): U1 gap D10[same] +0.0002 [−0.0010, +0.0015], D10[cross] −0.0010 [−0.0028, +0.0008] → 미충족. U2 8개 쌍 중 부호가 바뀐 쌍 0 → 미충족. epoch 순위는 v1 자산으로 평가 불가.
- [관측] test same 평가·cross 학습에서 D10 − D00의 선언 기반 +0.0016 vs 실제 기반 −0.0017로 부호가 다르지만 두 구간 모두 0을 포함(test는 진입 판단에 쓰지 않음).
- [관측] endpoint 층: mismatch와 π 이동 TV·Q 이동 RMS의 triplet 수준 Spearman 0.24–0.48(D11, test E1000), membership entropy 0.14–0.26, effective n·저지지 질량은 |ρ| ≤ 0.12. triplet은 독립 단위가 아니므로 기술 통계로만 본다. Q/π 이동 진단의 삼분위 절단점은 사전 규칙대로 learned arm 전체(이동 0인 D10/D01 포함)에서 계산되어 D11에는 두 층만 생겼다.

### 의미
- [추론] 불일치는 주로 head가 크게 움직일 때 커지지만, 실제 cloud의 TV를 선언값보다 나쁘게 만들지 않았고 arm 순위도 바꾸지 않았다. 0.016 불일치는 성능 제한의 근거가 아니다.

### 다음 행동
조건부 패키지는 사전 규칙상 실행하지 않는다(§5 미실행 기록).

## 4. 자원·운영

| 항목 | 값 |
|---|---|
| CPU | 17.1 CPU-시간(`/usr/bin/time` user+sys, Slurm 로그 50개; srun 단계 수 분 제외). 평가 30 task ≈ 14.8, W_same 3 seed 1.84, pilot 0.35, t*_same·분석 < 0.05 |
| GPU | 0 |
| 디스크 | 신규 13 MB(`assets/runs/r10_contract_matrix_v2`), v1 자산은 symlink로 읽기만 |
| 동시 job | 최대 12(array `%12`) |
| 기록 | [`run_ledger_v2.tsv`](run_ledger_v2.tsv) |

W 비용은 기획서의 v1 기록(cross π/Q/W 41.9 CPU-시간, 대부분 W)보다 훨씬 작았다(same W 3 seed × 2 wd = 1.84 CPU-시간): 선택 epoch가 15–20(wd 1e-4)·35–50(wd 1e-2)으로 이르게 patience에 걸렸고, v1 seed 1처럼 부하가 큰 노드를 만나지 않았다 [관측; 원인 분해는 미검증]. 평가 task 시간은 노드 부하에 따라 1분–18분으로 달랐다.

## 5. 미실행 기록

| 패키지 | 실행하지 않은 이유 | 여는 조건 |
|---|---|---|
| §7.2.3 목적함수 arm(selection-aligned / loss-aligned π-head) | 사전 진입 규칙 U1·U2 모두 미충족(development matched 칸). 실제 TV가 선언 TV보다 나쁘지 않고 순위 변화 없음 | development(또는 새 source OOF)에서 U1(gap ≥ 0.005, 하한 > 0, 두 계약) 또는 U2(부호 변화 + 구간 0 제외) 충족; 또는 사용자가 새 모델/목적함수 선택을 재개하기로 결정 |
| §7.2.4 hard partition 대조 | §7.2.3과 함께만 여는 규칙(필수 수정 아님) | §7.2.3이 열릴 때 |
| §7.1.5 E1b source OOF(train115 안 5-fold) | 새 후보 선택을 계속할 근거(§7.2.3 진입)가 없음 | §7.2.3이 열리거나 새 모델 선택을 재개할 때. 그때 head-only OOF와 end-to-end OOF를 명칭·비용 분리해 견적 |
| §7.3 독립 batch/HCC 확인(E3) | 자료 없음 | [`INDEPENDENT_DATA_REQUIREMENTS.md`](INDEPENDENT_DATA_REQUIREMENTS.md)의 자료 확보 + 명세 freeze |

## 6. 편차·사건

- 기획서 §7.1.6의 제안 경로 중 `folds.json`은 E1b를 실행하지 않아 만들지 않았다.
- `plate_relation_counts.tsv`는 dev/test triplet 기준이고, 정정 노트 E-3의 수치(2,057 / 2,076)는 train 포함 전체 적격 triplet 기준이다.
- report job 로그에는 `/usr/bin/time` 블록이 두 개(분석·replay)라 ledger에는 첫 블록만 기록됐다(합계 차이 < 3 CPU-초).
- 대기 shell은 한 번에 하나의 polling loop만 썼다(로컬에서 중복 대기 명령 두 개를 즉시 중지).
