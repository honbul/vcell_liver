# v2 보고서 정정 주석 (v3, 2026-11-20)

대상: `reports/r6_specificity_v2/RESULTS.md` (브랜치 `research/r6-specificity-observation-v2` @ `3499df9`). 원문은 고치지 않고 이 파일에 정정을 적는다. 근거: private plan 2026-11-20 §6.2.

| v2 위치 | v2 문장/값 | 정정 |
|---|---|---|
| 2.4 감도 표 "S5 동일·염 변형 화합물만: P1 / P2" | +0.144 [−0.062, +0.361] / −0.0107 [−0.032, +0.012] | wrong-match 후보에 `other` 계열 약물(64, 68)이 섞여 있었다. 후보에도 같은 필터를 적용한 v3 값은 +0.171 [−0.078, +0.423] / −0.0183 [−0.0399, +0.0070]이다(약물 13, link 1,896, plate 쌍 2; 3-4 link 141개는 동일·염 wrong 후보가 없어 제외). 상태(CI 0 포함)는 같다 |
| 2.4 감도 표 "S7 약물 leave-one-out 범위 P1 / P2" | 0.094–0.193 / −0.0205 – −0.0106 | 뺀 약물이 남은 link의 wrong 후보에 남아 있었다. 정정값은 0.080–0.191 / −0.0195 – −0.0054이다(두 경우 모두 17/17 유리 방향) |
| 2.4 "S1 공통 anchor (S5)" | `repeat_contrast_audit.tsv`의 S5 cos_anchor 0.025 [−0.100, 0.163] | anchor도 동일·염으로 제한하면 0.031 [−0.093, 0.168]이다 |
| 2.3·RUN_SPEC "약물 구조 그룹 블록 bootstrap" 구간 전체 | 표시 없음 | wrong-match 값과 anchor 평균을 replicate 안에서 고정하므로, 이 구간들은 **고정 comparator(또는 reference)에 조건부인 CI**다. comparator를 재구성한 보조 bootstrap에서도 주 대비는 유의하지 않다(v3 RESULTS 3.5) |
| 2.4 판정 `not_established` | — | 변경 없음(조건부·재구성 bootstrap 모두) |

v2 수치는 v3에서 원 배열로 재현했다(최대 차이 5e-7). 위 정정은 S5/S7 감도 행과 CI 표시만 바꾼다. 주 대비 값과 판정은 바뀌지 않는다.
