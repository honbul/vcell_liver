# P0-R6 v3 실행 명세 (동결 전 작성) — S5/S7 민감도의 비교 기준 재구성

근거: private plan 2026-11-20 §6.2·§6.3, §7. 기계가 읽는 명세: `configs/r6_sensitivity_v3/spec.json`. 코드: `src/r6_sensitivity_v3/sensitivity.py`(v2 `src/r6_specificity_v2/repeat_audit.py`의 `Boot`·`cos`·`rmse`를 수정 없이 import).
기준: `research/r6-specificity-observation-v2` @ `3499df9` (R6-R 동결 `e20bdde`, 채점 job 26743).

## 노출 지위
이미 노출된 자료(source TRAIN 약물 well, v2 job 26743에서 읽음)에 대한 **분석 정정**이다. blind가 아니고, 독립 자료가 아니며, 새 생물학적 반복이 아니다. 새 학습·새 분자 표현·set encoder 없음. GPU 0.

## 질문
v2 R6-R 감도 분석 S5(동일·염 변형 화합물만)와 S7(약물 leave-one-out)에서 wrong-match 비교 기준을 query와 같은 필터·제외 규칙으로 다시 만들면, 값과 상태 라벨이 달라지는가?

## 무엇이 잘못되었나 (v2 코드에서 확인한 사실)
- S5: query link만 동일·염 변형으로 걸렀고, 각 link의 wrong-match 값은 `other` 계열 약물(0-4의 64 crizotinib/(S)-crizotinib, 3-4의 68 glycyrrhizinate/glycyrrhizic acid)을 포함한 모든 서로소 쌍 평균으로 남아 있었다.
- S7: 약물 g를 뺄 때 g가 query인 link만 제거했고, 남은 link의 wrong-match 후보에는 g가 그대로 있었다.
- 재표집: 약물 블록 bootstrap에서 각 link의 wrong-match 값은 replicate 안에서 고정이다 → 그 CI는 **고정 comparator에 조건부**다.

## 입력 계약
- 적격 집합: `manifests/r6_specificity_v2/*` 그대로(v2 동결, metadata만).
- 응답: v1 `halves_v1/treated_halves.npy`의 full-well 평균(읽기 전용 symlink). (세포주, 약물, plate)별 세포 수 가중 평균, v2와 동일.
- 이름 계열: v2의 (plate 쌍, 약물)별 `identical / salt_variant / other`.
- 사용하지 않음: 검증·시험 약물, 모델 예측, 새 자료.

## 비교 기준 정의 (membership은 응답을 읽기 전 `comparator_membership.tsv`로 고정, job 27748)

| 분석 | v2 (수정 전) | v3 (수정 후) |
|---|---|---|
| primary | 같은 stratum의 모든 서로소 쌍, 양방향 | 변경 없음 |
| S5 | query 동일·염; wrong = 모든 서로소 쌍 | query 동일·염; wrong = **두 약물 모두 동일·염**인 서로소 쌍만. 후보가 없는 link는 제외·계수 |
| S7 (약물 g) | g가 query인 link 제거; wrong 그대로 | g가 query인 link 제거 + **g를 포함한 wrong 후보 제거**. 후보가 없는 link는 제외·계수 |
| S5-S1 (공통 anchor 중심화) | query 동일·염; anchor = 다른 공통 약물 전부 | query 동일·염 + **anchor도 동일·염만**, anchor ≥2 (v2 규칙) |
| S3 plate LOO | — | 다시 계산만 함(plate를 빼면 stratum 전체가 빠지므로 comparator 영향 없음). v2 판정 재유도용 |
| S6 같은 plate | — | 재계산하지 않음(wrong-match 비교 기준 없음) |

metadata 단계 집계(job 27748): S5 v3는 query 쌍 45개 중 42개 유지, 3개(plate 쌍 3-4의 19·176·188 쌍, link 141개)는 동일·염 wrong 후보가 없어 제외. S7 v3는 780개 (g, query 쌍) 중 12개 제외(3-4의 약물 하나를 빼면 3-4 stratum에 서로소 쌍이 남지 않음). S5-S1 v3는 제외 0.

## 재표집과 CI 표시
- 단위: 약물 구조 그룹 블록 bootstrap 2,000회, seed 0, link 가중 m_d·m_e, 약물 macro (v2 `Boot` 그대로). bootstrap 약물 집합도 v2와 같아 multiplicity 행렬이 같다 → 수정 전·후 행은 같은 replicate로 짝지어진다.
- `conditional_fixed_comparator`: v2 방식. **고정 comparator에 조건부인 CI**로 표시한다.
- `conditional_fixed_reference`: S5-S1의 anchor 평균 고정. 조건부.
- `rebuilt_comparator` (사전 지정 보조, primary P1/P2와 S5 v3 P1/P2에만): replicate b에서 각 link의 wrong-match = 후보(쌍×방향) 값의 가중 평균, 가중 = m_b[d']·m_b[e']. 후보 가중합이 0인 link는 그 replicate에서 제외(matched·wrong 모두). 그 다음 같은 약물 macro. 점추정은 조건부와 같다. 여전히 plate·세포주·적격 link 집합에 조건부이며 plate 모집단 CI가 아니다.
- `point_range`: S7 요약 = leave-one-out 점추정의 중앙값 [최소, 최대] (v2 형식). 약물별 행에는 조건부 CI도 붙인다.

## 주 대비와 판정표 (결과를 보기 전 고정)
- 주 보고 대비: **S5 v3(수정 후)의 P1_cos·P2_rmse (조건부 CI, v2와 같은 공동 주 대비 family)를 S5 v2(수정 전)와 비교**, 그리고 S7 v3 범위를 S7 v2 범위와 비교. 이것들은 감도 행이며 R6-R 판정을 바꾸지 않는다.
- 실용 기준: P2 상대 |차이| ≥ wrong-match RMSE의 0.5% (v2 그대로).
- 행 상태 라벨: CI 행 = `유리 방향, CI 0 제외` / `차이 미확립 (CI 0 포함)` / `불리 방향, CI 0 제외` (유리: P1>0, P2<0). S7 범위 행 = `유리 방향 k/n`.

| 항목 | 규칙 |
|---|---|
| R6-R 판정 | v2 동결 규칙을 (변하지 않은) primary 집합에 (a) 조건부 bootstrap(→ v2 `not_established` 재현 필수), (b) comparator 재구성 bootstrap으로 적용. v3 기록 = (a)(b) 중 더 보수적인 것. S5/S7은 판정을 바꾸지 않는다 |
| S5 변화 | 전→후 추정·CI·라벨·link/약물 수·제외 link 수 보고. P1 또는 P2의 라벨이 바뀔 때만 "S5 결론 변화" |
| S7 변화 | 전→후 중앙값/최소/최대와 유리 방향 수. 유리 방향 수가 n/n에서 n/n 미만으로(또는 반대로) 바뀔 때만 "S7 결론 변화" |
| 패키지 | 모든 행이 쓰이고 v2 재현 검사가 통과하면 `완료`. 재현 검사 실패 시 행은 남기되 전·후 서술은 원인 분리 전까지 `채택 보류` |
| 기본 결정 | **모델 확장 보류**. 유리한 변화도 독립 배양 반복성으로 승격하지 않으며 R6-E(구조 표현) 자동 재개 조건으로 쓰지 않는다 |

## 재현 검사 (계획 §7.2 4항)
원 treated-well 평균에서 v2 primary P1/P2(추정·CI·p), S5 v2 d_cos/d_rmse, S7 v2 범위, S5-S1 v2 cos_anchor, primary 2,362 link의 link별 값, v2 판정을 다시 만들어 커밋된 v2 표(소수 6자리)와 비교한다. 허용 오차 2e-6.

## 자원
CPU job 동시 1개(이 트랙 상한), 2 CPU, 8 GB, 제한 1시간, GPU 0. pilot(난수 배선 확인, job 27754): 38초, peak RSS 1.1 GB. 본 채점은 동결 커밋 뒤 1회.

## 환경 편차 (동결 전 기록)
v2가 쓴 인터프리터(`pubdata/.../vcell_liver_chemcpa_moe_v2/envs/chemcpa`)가 더 이상 없다(그 `envs/` 디렉터리 변경 시각 2026-11-19 12:44, v2 감사 뒤). v3는 기존 분석 환경(`vcell_liver_r10_contract_v2/assets/envs/r10`, numpy 1.26.4·pandas 2.2.2)을 읽기 전용으로 쓴다(`PYTHONDONTWRITEBYTECODE=1`). 버전 차이의 영향은 재현 검사로 확인한다.

## 미리 적는 한계
plate 5개·주 집합 plate 쌍 3개(모두 plate8 포함)라 plate 모집단 CI는 주지 않는다. 독립 배양 여부와 화합물 lot은 metadata로 식별할 수 없다. 음성·불안정 결과는 약물의 생물학적 효과가 없다는 증거가 아니다.
