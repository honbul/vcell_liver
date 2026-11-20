# R10 DECISIONS

> **v2 정정 노트 (2026-11-19).** 이 v1 문서의 test 범위(D11만 평가), C3 t*의 계약(train-cross, freeze 이후), same 계약과 plate 효과의 관계, CI가 0을 포함한 비교의 해석은 [`ERRATA_AND_CLARIFICATIONS_v2.md`](ERRATA_AND_CLARIFICATIONS_v2.md)에서 정정·명확화했다. 아래 원문과 수치는 보존한다.

각 cycle은 실행 **전에** 관측 → 가설 → 변경 → 판별 가능한 예상 결과를 적고, 실행 후 결과와 다음 결정을 덧붙인다. 이전 항목의 이름과 결과는 고치지 않는다(정정이 필요하면 새 항목으로 남긴다).

---

## C0. 데이터·평가 계약 고정 (2026-11-18, B1/B2 실행 전 작성)

**관측(B0 일부).** GSM4150378 pData는 799,317 barcode, hash 배정 762,795, 24 h 680,685. 24 h culture well(hash combo) 4,608 = 3 세포주 × 2 replicate × 768(188 약물 × 4 용량 + vehicle 16). 각 replicate는 서로 다른 culture plate 8장에 놓이고 plate마다 vehicle well 2개가 있다(pData `culture_plate` = plate hash oligo, 전 well 일치). A549 72 h well은 제외한다.

**고정한 선택(결과를 보기 전).**

| 항목 | 고정값 |
|---|---|
| hash QC | `qval_W<0.05`, `qval_P<0.05`, top-to-second ratio W·P ≥ 10, n.umi ≥ 500 (R10 규칙; 원논문의 정확한 threshold는 미검증) |
| 약물 분할 | pathway_level_1 층화, 동일 SMILES 묶음, seed 20261118, train/development/test ≈ 60/20/20 |
| 표현 | train 약물 + vehicle 세포(well당 ≤64)로 log1p CP10k, HVG 2,000, 표준화, PCA 20 |
| 상태 atlas | 세포주별 GMM(full cov), K∈{4,6} 중 replicate 재현성 규칙(config `atlas.K_rule`)으로 선택, K=2·8은 민감도 |
| episode | E100: 10 nM + 1 µM → 100 nM, E1000: 100 nM + 10 µM → 1 µM (주 평가 E1000), t=0.5 |
| 반복 계약 | 주 조건 cross(endpoint는 한 replicate, query는 다른 replicate, 두 방향 평균 = 약물 안 합산), same은 별도 |
| 적격성 | endpoint·query well QC 세포 ≥ 20, endpoint replicate vehicle ≥ 50 |
| 주 지표 | 출력 세포 구름을 고정 atlas로 다시 읽은 구성비 vs query 관측 구성비의 TV (모든 방법 동일 정의). head가 선언한 구성비는 assembly_consistency에 별도 보고 |
| 보조 지표 | 상태 내부 조건부 평균 RMSE·조건부 ED(query n_eff ≥ 5인 상태, query 비중 가중), 전체 ED·MMD(RBF, train 세포 median 거리), 유전자 Δmean RMSE(선형 PCA decoder, 모든 방법 동일, 재구성 오차 별도) |
| ED 추정 | R3 추정량의 일반화: 그룹(=독립 표본) 안은 가중 U-통계량, 그룹 사이는 가중 평균. 균일 단일 그룹이면 R3 식과 같음 |
| 집계 | seed 평균 → 두 cross 방향 평균 → 세포주 동일 가중으로 약물 → 약물 단위 paired bootstrap(B=2000) |
| D00 | log-dose empirical mixture와 동일한 구름(정의상 일치). head는 그 위의 잔차 |
| 조립 | π-head: 세포 가중치 × Σ_k s_k(z)·π̂_k/π^int_k; Q-head: 세포 이동 Σ_k s_k(z)·r_k. s_k는 endpoint 세포의 고정 membership |
| 학습 | train 약물 cross triplet, 약물 균형 가중; head별 동일 grid(weight decay {1e-4,1e-2}), development 목적함수로 early stopping·선택; D11 = 선택된 π-head + Q-head(공유 trunk 없음) |
| 버전 선택 규칙 | development E1000·cross·3 seed 평균에서, D00 대비 paired 차이의 95% CI 상한 < 0인 learned arm 가운데 주 지표가 가장 낮은 arm. 없으면 D00(직접 보간). D11이 선택되었어도 D10 또는 D01의 (arm − D11) CI가 0을 포함하면 더 단순한 쪽(둘 중 TV가 낮은 것) |
| test | 선택 버전을 freeze commit 후 test 약물에 한 번만 평가(비학습 기준선은 함께 계산) |

**가설.** 이 단계는 가설 검정이 아니라 계약 고정이다.

**판별 결과 예상.** 해당 없음.

### C0 부록 — B1 실행 기록 (2026-11-18, B3 전)

- `r10_b1_v1`(job 26275): 합성 구성비-only 대조가 **잘못 설계**되었다. 가장 작은 vehicle 상태로 질량을 옮기면서 절반의 세포를 비복원 추출해 목표 구성비가 실현되지 않았다(실현 TV 이동 0.006–0.027). 결과는 보존하되 사용하지 않는다. 같은 run의 effective n 계산은 membership이 모두 underflow할 때 0/0을 낼 수 있었다(가중 평균에 거의 영향 없음).
- `r10_b1_v2`(job 26304): neff 수정만 반영해 제출했으나 위 설계 오류를 알게 되어 **취소**(산출물 미사용).
- `r10_b1_v3`(job 26305): 정확한 hard-state 개수로 가장 큰 상태 → 두 번째 상태로 질량 이동(400세포), neff 수정 포함. 이후 모든 B1 수치는 v3. K 선택(세 세포주 모두 K=4)은 v1·v3 동일 규칙·동일 입력.

---

## C1. B3 2×2 v1 (2026-11-18, B1·B2 결과를 본 뒤, B3 실행 전 작성)

**관측(B1·B2, train/development 약물만).**
1. 합성 양성 대조(v3): 구성비-only 변화는 구성비 TV로, 상태 내부-only 변화는 상태 내부 평균 RMSE·조건부 ED로 잡히고, 실제 변화를 같은 조립 규칙(가중치 재조정 / 상태별 이동)으로 되돌리면 해당 지표가 null 수준으로 돌아온다. 평가기는 두 성분을 구분한다(합성 결과이며 생물학적 검증 아님).
2. 같은 조건의 replicate well 간 구성비 TV 중앙값은 0.079–0.099(full n)이고 n=32에서 0.108–0.147로 커진다 → 관측 구성비 차이의 상당 부분이 유한 표본 잡음이다. vehicle 대비 처리 조건의 TV 중앙값(0.071–0.081)도 같은 수준이다. 대부분의 약물·용량은 이 해상도에서 vehicle과 구분되지 않는다.
3. MCF7은 vehicle끼리도 cross-replicate ED(0.35)가 같은 replicate 안(0.05–0.09)보다 훨씬 크다(replicate/plate 묶음 차이). K562는 중간, A549는 거의 없다.
4. 평균 변화의 대칭 분해에서 구성비 항의 비중 중앙값은 0.14–0.26, 상태 내부 항이 더 크다(고정 K=4 atlas 기준 기술 통계).
5. 직접 보간(mixture=D00)의 오류는 source triplet 대부분(E1000 train: 261/313 = 83%)에서 replicate-oracle 한계 안이다. 한계를 넘는 경우는 주로 구성비 쪽(E1000 train 313 약물·세포주 조건 중 37이 composition, 6이 within_state, 9가 both).
6. development E1000에서 고정 state_moment 보간이 D00보다 구성비 TV −0.0022 [−0.0035, −0.0008], 상태 내부 평균 RMSE −0.024 [−0.031, −0.018]로 작게 낫다. nearest_low는 전체 ED가 D00보다 낮다(−0.17 [−0.40, −0.02]); 10 µM endpoint가 혼합에 큰 이동을 넣는 약물이 평균을 끌어올린다.

**가설.** 직접 보간이 replicate 한계를 넘는 오류는 주로 구성비 쪽이며, endpoint 요약에서 그 방향(어느 상태가 과대/과소 보간되는지)을 일부 학습할 수 있다. 상태 내부 오류는 대부분 replicate 잡음 수준이라 Q-head가 개선할 여지가 작다.

**변경.** 2×2(D00/D10/D01/D11)와 compact whole-distribution 모델 W를 C0의 학습·선택 규칙대로 구현한다. 먼저 pilot(train 20약물, development 10약물, seed 0)으로 통로를 확인하고, 이어 전체 train/development에서 seed 0·1·2.

**판별 가능한 예상 결과.**
- 가설이 맞으면: D10의 development E1000 구성비 TV 차이(D10−D00)가 음수이고 95% CI 상한 < 0, D01의 상태 내부 지표 차이는 CI가 0을 포함.
- 가설이 틀리면: D10 차이 CI가 0을 포함(학습한 구성비 보정이 보간을 넘지 못함). 이 경우 §10 표의 "모든 learned arm이 직접 보간과 비슷" 행을 적용한다.
- W가 D10/D11보다 좋으면 분해 구조보다 일반 보정의 이득으로 해석한다.

### C1 부록 — 선택 규칙의 해석 (B3 development 평가 결과를 보기 전, 2026-11-18 19:07 KST)

- C0 "버전 선택 규칙"의 learned arm은 2×2의 D10·D01·D11을 뜻한다. W(compact whole-distribution)는 §8의 비교 기준선이며 버전 후보가 아니다. test에는 선택 버전과 비학습 기준선을 평가하고, W·다른 learned arm은 test에서 평가하지 않는다.
- 학습 로그에서 W와 Q-head의 development 목적함수가 초기값(D00)보다 크게 낮아졌다(W ED 0.35 → 0.12, seed 0). 학습 triplet은 모두 cross-replicate이고 특징에 endpoint의 절대 상태 평균이 들어가므로, head가 **replicate 사이의 체계적 이동**(B1에서 MCF7이 큼)을 배울 수 있다. 이를 가르는 진단: 같은 head를 same-replicate triplet(이동 없음)에 적용한 development 결과와 세포주별 결과. cross에서만 이득이고 same에서 이득이 없거나 손해면 "용량 보간 정보"가 아니라 "replicate 이동 보정"으로 해석한다. 이 진단은 선택 규칙을 바꾸지 않으며 해석에만 쓴다.

### C1 결과 (development, 2026-11-18 19:44 KST 무렵 작성; test는 아직 보지 않음)

run: `r10_b3_v1`(job 26354–26356, seed 0·1·2), 평가 `eval_development_2x2_*`(job 26404), 집계 `r10_report_dev2x2_v1`. 선택된 head는 모든 seed에서 weight decay 1e-4.

| E1000, cross, 35 약물 | 구성비 TV 차이 vs D00 [95% CI] |
|---|---|
| D10 | −0.0292 [−0.0440, −0.0163] |
| D01 | −0.0114 [−0.0183, −0.0060] |
| D11 | −0.0312 [−0.0462, −0.0181] |
| 상호작용 D11−D10−D01+D00 | +0.0094 [+0.0036, +0.0169] |
| D10 − D11 | +0.0020 [+0.0001, +0.0040] |
| D01 − D11 | +0.0198 [+0.0110, +0.0298] |

**규칙 적용(문자 그대로).** 세 learned arm 모두 CI 상한 < 0, 주 지표 최저는 D11. D10−D11과 D01−D11의 CI가 모두 0을 포함하지 않으므로 단순화 조항은 적용되지 않는다 → **선택 버전 = D11 v1**. (D10−D11 하한 +0.0001로 경계에 가깝다는 점을 기록한다.)

**사전 고정 진단(same-replicate, 같은 head).** E1000 same에서 구성비 TV 차이는 D10 −0.0088 [−0.0245, +0.0043], D11 −0.0073 [−0.0230, +0.0057]로 CI가 0을 포함한다. Q-head는 same에서 상태 내부 평균 RMSE를 오히려 키운다(E100 D01 +0.068 [+0.058, +0.078]). cross에서의 큰 ED·상태 내부 이득(D01 ED −0.30 [−0.59, −0.13])은 MCF7에서 가장 크다.

**해석(추론, 미검증).** cross-replicate 계약에서 head 이득의 상당 부분은 replicate 사이 체계적 이동을 배운 결과와 맞는다. same에서 이득이 사라진다는 것은 그 해석과 일관되지만, head를 cross로만 학습했으므로 "same 계약에서 학습한 head는 용량 정보로 이득을 낸다/못 낸다"는 아직 측정하지 않았다. 이 구분은 C2에서 한다.

**다음 결정.** (1) D11 v1을 freeze하고 test 약물에 한 번 평가한다(비학습 기준선·oracle 진단 함께, cross·same 모두). (2) C2: same-replicate 계약으로 head를 다시 학습·평가해 replicate 이동이 없는 조건에서도 learned arm이 D00을 넘는지 development에서 가른다(새 버전 v2; test 결과는 이미 본 뒤이므로 v2를 test에서 보더라도 blind test라 부르지 않는다).

---

## C2. same-replicate 계약에서 다시 학습한 2×2 (2026-11-18, 실행 전 작성; D11 v1 test 평가는 제출됨, 결과 미확인)

**관측(C1).** cross 계약에서 학습한 head는 development cross에서 D00보다 구성비 TV를 낮췄지만, 같은 head를 same-replicate triplet에 적용하면 구성비 이득의 CI가 0을 포함하고 Q-head는 상태 내부 오차를 키웠다. MCF7의 cross 이득이 가장 크다.

**가설.** C1의 learned 이득은 주로 replicate 사이 체계적 이동의 보정이다. replicate 이동이 없는 same 계약에서 학습·선택·평가하면, endpoint 요약만으로 얻는 용량 보간 이득은 작다.

**변경(핵심 1개).** train/development 예제를 same-replicate triplet으로 바꾼다(`--direction same`). 나머지(특징, head 구조, grid, 조립, seed 0·1·2)는 C1과 동일. W는 다시 학습하지 않는다.

**판별 가능한 예상 결과.**
- 가설이 맞으면: development same E1000에서 D10·D11 − D00 구성비 TV 차이의 CI가 0을 포함하거나, 음수여도 C1 cross 이득(−0.03)보다 크기가 훨씬 작다(절반 미만); D01의 상태 내부 이득도 작다.
- 가설이 틀리면(용량 정보가 학습됨): same에서도 D10 또는 D11의 CI 상한 < 0이고 크기가 C1 cross 이득과 비슷하다.
- 이 cycle은 해석용 대조이며 v1 선택·test 결과를 바꾸지 않는다. v2를 test에 평가하지 않는다.

### C2 결과 (development, 2026-11-18 19:58 KST 무렵)

run `r10_c2_same_v1`(job 26430–26432 학습, 26433 평가), 집계 `r10_report_c2_v1`. same 계약으로 학습한 head, development same E1000(35 약물):

| arm − D00 | 구성비 TV [95% CI] | 상태 내부 평균 RMSE | 전체 ED |
|---|---|---|---|
| D10 | −0.0160 [−0.0302, −0.0044] | −0.0004 [−0.0012, +0.0005] | +0.062 [−0.087, +0.317] |
| D01 | −0.0066 [−0.0132, −0.0019] | −0.063 [−0.166, −0.005] | −0.189 [−0.473, −0.023] |
| D11 | −0.0176 [−0.0333, −0.0051] | −0.063 [−0.165, −0.006] | −0.198 [−0.467, −0.037] |

E100 same: D11 구성비 TV −0.0069 [−0.0149, −0.0014], D01 상태 내부 평균 RMSE +0.0056 [+0.0010, +0.0107](악화).

**사전 기준 적용.** same에서 D10·D11의 CI 상한 < 0이고 크기는 C1 cross 이득(D11 −0.0312)의 약 0.57배다. "CI가 0을 포함하거나 절반 미만"이라는 가설 지지 조건을 만족하지 않는다 → **"learned 이득은 주로 replicate 이동 보정"이라는 가설은 기각**. replicate 이동이 없는 계약에서도 구성비 이득의 절반 이상이 남는다. 다만 크기가 C1과 "비슷"하지도 않아, cross 이득의 일부(대략 0.4배)는 replicate 관련일 수 있다(추론, 직접 측정 아님).

---

## C3. 구성비 이득이 "단일 mixing weight 이동"으로 설명되는가 (2026-11-18, 실행 전 작성)

**관측.** C1·C2 모두 π-head(D10)가 구성비 이득의 대부분을 낸다(D01 단독은 작고, 상호작용은 양수 = 비가산). B2에서 nearest_low가 E1000 구성비 TV에서 mixture보다 낮았다(−0.006, CI 0 포함) — 1 µM 반응이 log-dose 중점보다 낮은 용량 쪽에 가까운 약물이 있다는 신호와 일관.

**가설.** D10 이득의 대부분은 상태별 보정이 아니라 "t=0.5 대신 낮은 용량 쪽으로 기운 전역 mixing weight" 하나로 재현된다(용량-반응의 비선형성에 대한 전역 보정).

**변경(핵심 1개).** 비교 기준선 `mixture_t`: train 약물 cross triplet에서 episode별 t*를 격자(0–1, 0.025 간격)로 하나 적합하고, 같은 empirical mixture를 그 t*로 만든다. head·atlas·평가는 그대로.

**판별 가능한 예상 결과.**
- 가설이 맞으면: development E1000 cross에서 D10 − mixture_t 구성비 TV 차이의 CI가 0을 포함(또는 mixture_t가 더 낮음).
- 가설이 틀리면: D10 − mixture_t의 CI 상한 < 0(상태별 보정이 전역 weight 이동보다 낫다).
- mixture_t는 v1 freeze 이후 추가된 기준선이므로 test 수치는 blind가 아닌 사후 비교로만 표시한다.

### C3 결과 (2026-11-18 20:05 KST 무렵)

`r10_c3_tstar_v1.json`: train cross에서 t* = 0.325(E100·E1000 모두; t=0.5 대비 train 구성비 TV 0.1276 → 0.1237, E1000). 평가 run `r10_c3_v1`(job 26449 development, 26450 test).

| development, 구성비 TV | E1000 cross | E1000 same |
|---|---|---|
| mixture_t − D00 | −0.0150 [−0.0237, −0.0068] | −0.0141 [−0.0222, −0.0063] |
| D10(C1, cross 학습) − mixture_t | −0.0142 [−0.0246, −0.0054] | +0.0053 [−0.0046, +0.0135] |
| D10(C2, same 학습) − mixture_t | −0.0067 [−0.0183, +0.0016] | −0.0019 [−0.0107, +0.0052] |

**사전 기준 적용.** C3 기준(cross, C1 head): D10 − mixture_t CI 상한 < 0 → "D10 이득은 전역 weight 이동 하나로 설명된다"는 가설은 cross 계약에서 **기각**.

**추가 분석(사전 기준 밖, 해석용).** same 계약(replicate 이동 없음)에서는 same으로 학습한 D10도 mixture_t와 구분되지 않는다(−0.0019 [−0.0107, +0.0052]). 즉 replicate 이동이 없는 조건에서 learned π-head의 구성비 이득은 "낮은 용량 쪽으로 기운 전역 mixing weight" 하나와 같은 수준이고, cross 계약에서 그 이상으로 얻는 몫은 replicate 구조와 연결된 것으로 보인다(추론; replicate 이동을 직접 분리한 측정은 아님).

**§10 행 판정.** 주 지표 기준으로 cross에서는 "D10만 개선"(D11 ≈ D10, D01 단독은 작음, 상호작용 양수)에 해당한다. replicate 이동과 전역 t를 통제하면(same 계약 + mixture_t) "모든 learned arm이 (튜닝된) 직접 보간과 비슷" 행에 해당한다. 후자 행의 후속인 오류 지도(반응 강도별 실패 지도, endpoint 세포 예산 곡선)는 B2에서 이미 산출했다.

**다음 결정.** 새 cycle을 열지 않는다. C1 v1(D11)의 test 결과를 그대로 보고하고, mixture_t의 test 수치는 freeze 이후 추가한 사후 비교로만 표시한다. 남은 후보(구성비 logit 정칙화, endpoint 불확실성 입력, replicate 이동을 명시적으로 쓰는 계약)는 실행하지 않은 열린 결정으로 남긴다.

### C1 부록 — W 결과 (2026-11-18 21:20 KST; W seed 1 학습이 부하가 큰 노드에서 2시간 걸려 늦게 평가됨)

development E1000 cross: W − D00 구성비 TV −0.016 [−0.024, −0.010], ED −0.326 [−0.629, −0.153]; D11 − W 구성비 TV −0.015 [−0.024, −0.007], ED +0.009 [−0.006, +0.029]. C1의 "W가 D10/D11보다 좋으면" 조건은 구성비에서는 성립하지 않고(D11이 더 낮음), 전체 ED에서는 둘이 구분되지 않는다. → 전체 분포 이득은 분해 구조에 특유한 것이 아니다. 선택(D11 v1)과 test 결과에는 영향 없음(W는 버전 후보가 아니었음).
