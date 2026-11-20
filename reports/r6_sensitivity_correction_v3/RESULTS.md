# P0-R6 v3 결과 — S5/S7 민감도의 비교 기준 재구성 (2026-11-20)

```text
연구: R6 / 질문: v2 반복 대비 감도 분석(S5 동일·염 변형, S7 약물 leave-one-out)의 wrong-match 비교 기준을 query와 같은 규칙으로 다시 만들면 값·판정이 바뀌는가?
실행: 학습 없음. v2 treated-well 평균으로 S5·S7을 comparator 재구성 포함해 다시 계산(동결 4b7812e, job 27757 1회); v2 값은 원 배열에서 오차 ≤5e-7로 재현됨. CI는 고정 comparator 조건부로 표시하고, comparator 재구성 bootstrap을 사전 지정 보조로 추가.
결과: S5 P1 +0.144 [−0.062, +0.361] → +0.171 [−0.078, +0.423]; S5 P2 −0.0107 [−0.0321, +0.0123] → −0.0183 [−0.0399, +0.0070] (link 2,037→1,896, 약물 15→13, plate 쌍 3→2). S7 범위 P1 0.094–0.193 → 0.080–0.191, P2 −0.0205~−0.0106 → −0.0195~−0.0054 (두 경우 모두 17/17 유리 방향). 주 집합 R6-R 판정은 조건부·재구성 bootstrap 모두 not_established.
판정: 완료. S5·S7 결론 변화 없음(모든 CI 0 포함 = 차이 미확립), R6-R `not_established` 유지 → 채택 보류(반복성 근거 미확립).
다음 행동: 모델 확장 보류 유지. R6는 독립 반복 자료가 생길 때까지 일시 정지(r6_pause_and_resume.md).
사용자 결정 필요: 없음 (재개하려면 독립 배양 반복 자료 확보 여부를 정해야 하며, 이것은 비용·외부 협력 문제다).
```

브랜치 `research/r6-sensitivity-correction-v3` (기준 `3499df9` = R6 v2 끝). 근거: private plan 2026-11-20 §6. 동결 커밋 **`4b7812e`**. 서버 작업 폴더 `vcell_liver_r6_sensitivity_v3`; v1/v2 자산은 읽기 전용 symlink로만 사용.
노출 지위: v2에서 이미 읽은 source TRAIN 약물 well에 대한 **분석 정정**이다. blind·독립 검증이 아니며 새 생물학적 반복이 아니다.
차이는 항상 `matched − wrong-match`. 구간은 약물 구조 그룹 블록 bootstrap 2,000회(seed 0, v2와 같은 multiplicity 행렬)의 95% 백분위 구간이다.

## 1. 무엇을 고쳤나

| 항목 | v2 (수정 전) | v3 (수정 후) |
|---|---|---|
| S5 wrong-match 후보 | query만 동일·염으로 거름; wrong 후보에 `other` 계열(0-4의 64 crizotinib/(S)-crizotinib, 3-4의 68 glycyrrhizinate/glycyrrhizic acid) 포함 | wrong 후보도 두 약물 모두 동일·염인 쌍만. 3-4의 query 쌍 3개(19·176·188, link 141개)는 그런 쌍이 없어 제외 |
| S7 약물 g 제거 | g가 query인 link만 제거; g는 다른 link의 wrong 후보로 남음(g에 따라 (link, 후보 쌍) 141~7,701개가 g를 포함) | g를 wrong 후보에서도 제거. 3-4 약물(19, 68, 176, 188)을 빼면 3-4 link 141개는 후보가 없어 제외 |
| S5-S1 anchor | anchor = 다른 공통 약물 전부 | anchor도 동일·염만(≥2개). 제외 0 |
| CI 표시 | 표시 없음(사실상 고정 comparator 조건부) | `conditional_fixed_comparator`/`conditional_fixed_reference`로 표시. 사전 지정 보조로 `rebuilt_comparator` 추가 |

membership은 응답을 읽기 전 metadata로 만들어 동결에 포함했다(`comparator_membership.tsv`, 분석×plate 쌍×query 쌍마다 후보 목록·`other` 포함 여부·g 포함 여부·제외 사유).

## 2. 재현 검사 (계획 §7.2 4항) — 통과
원 treated-well 평균에서 다시 만든 v2 값이 커밋된 v2 표와 일치했다(`v2_reproduction_check.tsv`, 32항목, 최대 절대차 5e-7 ≤ 허용 2e-6): primary P1/P2 추정·CI·p, S5 v2 d_cos/d_rmse, S7 v2 범위, S5-S1 v2 cos_anchor, primary 2,362 link의 link별 6개 값, v2 판정. v2가 쓴 인터프리터가 사라져 다른 numpy/pandas 환경을 썼지만(편차 1) 결과는 같다.

## 3. 결과 (`sensitivity_corrected.tsv`)

### 3.1 S5 — 동일·염 변형 화합물만 (조건부 CI, 주 보고 대비)

| 양 | v2 (전) | v3 (후) | 라벨 전 → 후 |
|---|---|---|---|
| P1 matched − wrong cos | +0.144 [−0.062, +0.361], p 0.198 | +0.171 [−0.078, +0.423], p 0.229 | 차이 미확립 → 차이 미확립 |
| P2 matched − wrong RMSE | −0.0107 [−0.0321, +0.0123], p 0.333 (wrong 대비 −7.4% [−22.4%, +8.6%]) | −0.0183 [−0.0399, +0.0070], p 0.164 (−12.9% [−28.1%, +5.0%]) | 차이 미확립 → 차이 미확립 |
| matched cos / wrong cos | 0.090 / −0.053 | 0.115 / −0.056 | — |
| matched RMSE / wrong RMSE | 0.1327 / 0.1434 | 0.1236 / 0.1419 | — |
| 약물 / link / plate 쌍 / 제외 link | 15 / 2,037 / 3 / 0 | 13 / 1,896 / 2 / 141 | — |

- 관측: 점추정은 두 대비 모두 유리한 쪽으로 커졌지만 구간은 여전히 0을 포함한다. v2 Holm 규칙(p ≤ 0.025 또는 0.05)도 충족하지 않는다(0.164, 0.229).
- 관측: v3 S5에는 plate 쌍 0-4(plate11–plate8)와 1-4만 남는다. 0-4는 v2에서 재현 신호를 몰아서 가진 plate 쌍이므로, 점추정 증가는 그 비중이 커진 것과 방향이 같다. 이 귀속은 따로 검정하지 않았다(가설).

### 3.2 S7 — 약물 leave-one-out (점추정 범위; 약물별 조건부 CI도 기록)

| 양 | v2 (전) 중앙값 [최소, 최대] | v3 (후) | 유리 방향 수 전 → 후 |
|---|---|---|---|
| P1 | 0.159 [0.094, 0.193] | 0.168 [0.080, 0.191] | 17/17 → 17/17 |
| P2 | −0.0149 [−0.0205, −0.0106] | −0.0159 [−0.0195, −0.0054] | 17/17 → 17/17 |

- v3 최솟값은 약물 10(tirabrutinib)을 뺄 때 P1 0.080, 약물 16(trametinib)을 뺄 때 P2 −0.0054(가장 0에 가까움). 둘 다 plate 쌍 0-4 약물이다.
- 약물별 조건부 CI(v2 보고서에는 범위만 있었다): v2 방식에서는 17×2행 중 2행이 0을 제외했다(338을 뺀 P1 +0.193 [+0.001, +0.399]; 93을 뺀 P2 −0.0205 [−0.0383, −0.0022]). comparator를 재구성하면 두 행 모두 0을 포함한다(+0.191 [−0.002, +0.398]; −0.0167 [−0.0355, +0.0026]). v3에서 0을 제외하는 행은 없다. 사전 판정표의 "S7 결론 변화" 기준(유리 방향 수)에는 해당하지 않는다.

### 3.3 S5-S1 공통 anchor 중심화 (조건부, anchor 고정)
cos_anchor +0.025 [−0.100, +0.163] → +0.031 [−0.093, +0.168] (약물 18, link 1,141, 제외 0). 차이 미확립 유지.

### 3.4 primary와 R6-R 판정
- primary 집합과 comparator는 바뀌지 않으므로 조건부 값은 v2 그대로다: P1 +0.156 [−0.031, +0.368], P2 −0.0148 [−0.0340, +0.0064].
- v2 동결 규칙 적용: 조건부 bootstrap → `not_established`(v2 재현), 재구성 bootstrap → `not_established`. v3 기록(더 보수적인 쪽) **`not_established`**. S3 plate LOO는 comparator 영향이 없어 v2와 같다(plate0 제외 시 P1 +0.009, P2 +0.002).

### 3.5 comparator 재구성 bootstrap (사전 지정 보조)

| 대비 | 점추정 | 조건부 CI (p) | 재구성 CI (p) | replicate당 제외 link 평균 |
|---|---|---|---|---|
| primary P1 | +0.156 | [−0.031, +0.368] (0.121) | [−0.145, +0.397] (0.643) | 305 / 2,362 |
| primary P2 | −0.0148 | [−0.0340, +0.0064] (0.163) | [−0.0404, +0.0124] (0.739) | 305 / 2,362 |
| S5 v3 P1 | +0.171 | [−0.078, +0.423] (0.229) | [−0.185, +0.350] (0.970) | 167 / 1,896 |
| S5 v3 P2 | −0.0183 | [−0.0399, +0.0070] (0.164) | [−0.0347, +0.0142] (1.000) | 167 / 1,896 |

- 관측: 재구성 bootstrap에서는 구간 하한이 내려가고 p가 커졌다. S5 v3 P2의 p = 1.000은 replicate의 절반가량이 0 이상이라는 뜻이고, 점추정(−0.018)과 bootstrap 분포 중심이 어긋난다.
- 가설(검정하지 않음): replicate에서 wrong 후보가 모두 빠진 link는 제외하도록 사전에 정했다. 그런데 작은 stratum(0-4: 약물 5개, 3-4: 4개)의 link가 이렇게 자주 빠지고, 이 link가 큰 matched 값을 가진다. 그래서 replicate 구성이 1-4 쪽으로 쏠렸을 수 있다. 이 경우 재구성 CI도 잘 보정된 약물 모집단 CI가 아니다.
- 결론에 쓰는 범위: 어느 bootstrap에서도 유의한 대비는 없다. 사후에 다른 재구성 방식을 시도하지 않았다(미실행 목록).

## 4. 의미
- 관측: 비교 기준을 바로잡아도 S5·S7의 값은 조금 움직였을 뿐이고(S5 점추정은 유리한 쪽), 상태 라벨과 R6-R 판정은 바뀌지 않았다. v2의 "같은 약물쌍 차이가 다른 plate에서 wrong-match보다 더 잘 재현된다는 것은 확인되지 않음"은 그대로다.
- 관측: comparator 재구성까지 반영하면 불확실성이 더 커 보인다. v2 S7의 두 "CI 0 제외" 약물별 행은 comparator 오염과 함께 사라졌다.
- 승격하지 않음: 이 결과는 독립 배양 반복성을 보여주지 않는다. 다른 plate는 다른 well·plate일 뿐이고, 배양 날짜·계대·lot는 기록되어 있지 않다. 구조 표현 실험(R6-E)의 재개 조건으로도 쓰지 않는다.
- 불확실: plate 일반화. 주 집합의 plate 쌍은 3개(v3 S5는 2개)이고 모두 plate8을 포함한다.

## 5. 산출물

| 파일 | 내용 |
|---|---|
| `RUN_SPEC.md`, `configs/r6_sensitivity_v3/spec.json`, `manifests/r6_sensitivity_v3/freeze.json` | 동결 명세·판정표·hash (동결 `4b7812e`) |
| `comparator_membership.tsv` | 분석·버전·plate 쌍·query 쌍별 wrong 후보/anchor 목록, `other`·g 포함 여부, 제외 사유, 세포주 수 |
| `sensitivity_corrected.tsv` | 모든 전·후 행(추정, CI, p, 약물·link 수, CI 유형, 상태 라벨, 제외 수) |
| `link_values_primary.tsv`, `link_values_S5_v3.tsv` | link별 matched·wrong 값(원표) |
| `v2_reproduction_check.tsv`, `manifests/r6_sensitivity_v3/score_receipt.json` | 재현 검사, 입력 hash·버전 |
| `decision.md`, `not_run.md`, `run_ledger.tsv`, `errata_v2.md` | 판정, 미실행, 실행 기록, v2 보고서 정정 주석 |
| `r6_pause_and_resume.md`, `candidate_repeat_data_inventory.tsv` | §6.3 재개 조건·자료 요구·접근 가능 후보 목록 |

## 6. 편차·사건
1. v2 인터프리터(`pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2/envs/chemcpa`)가 없다. 이 `envs/`는 2026-11-19 12:44에 바뀌었다(v2 감사 이후). 그래서 첫 membership job 27745가 exit 127로 끝났다(출력 없음). 기존 분석 환경 `vcell_liver_r10_contract_v2/assets/envs/r10`(numpy 1.26.4, pandas 2.2.2)을 읽기 전용으로 썼고(`PYTHONDONTWRITEBYTECODE=1`), 동결 전에 기록했다. 재현 검사는 통과했다.
2. 동결 전 배선 확인 job 27749(spec 키 오류)와 27750(빈 재현 표 처리 오류)은 난수 입력에서 코드 오류로 끝났다. 고친 뒤 27754에서 통과했다. 실제 응답은 동결 뒤 27757에서 한 번만 읽었다.
3. 서버 worktree를 처음에 상대 경로로 만들다가 `vcell_liver/vcell_liver_r6_sensitivity_v3`라는 잘못된 위치에 생겼다. 방금 만든 빈 worktree라 바로 제거하고 올바른 경로에 다시 만들었다. 다른 파일에는 영향이 없다.
4. 서버에 git 사용자 정보가 없어 서버에서는 커밋하지 않았다. 서버 산출물은 로컬로 복사해 sha256이 같은지 확인한 뒤 커밋했다(로컬 → push → 서버 동기화, 한 방향).
