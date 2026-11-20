# R7 실험 계약 요약 (PROTOCOL)

기계가 읽는 원본은 `configs/evidence_gated_r7/protocol.yaml`이다. 이 문서는 사람이 읽기 위한 요약이며, 둘이 다르면 yaml이 우선한다. 바꿀 때는 yaml의 `amendments`에 날짜와 이유를 적는다.

## 과제

| 항목 | 정의 |
|---|---|
| 한 예시(unit) | (세포주, 약물 그룹). 입력은 그 세포주의 대조군 RNA(X0)와 약물 정보이고, 정답은 같은 plate의 처리군−대조군 차이(Δ, 978 landmark)를 R4 규칙으로 평균한 값 |
| 자료 | CMap LINCS 2020 Level 3, 단일 저분자, 10 µM·24 h. R4 unit 파일을 읽기 전용으로 사용 |
| 분할 | R4 전역 약물 그룹 분할(70/15/15, seed 20261117). validation·test 약물은 어떤 세포주에서도 학습에 쓰지 않음 |
| 새 약물의 뜻 | "그 약물의 RNA 반응을 학습하지 않았다." 표적 주석은 예측 시 쓸 수 있는 외부 정보로 허용 |
| HCC | HUH7·JHH5·JHH7. 동결 전에는 HCC 점수를 계산하지 않음 |

## 외부 지식 (Phase 0, RNA 미사용)

- **ChEMBL 37**: 구조(InChIKey)로만 연결한다. 근거는 기전(작용 부호 포함), binding activity(pchembl≥6), functional activity(pchembl≥6) 세 종류를 따로 둔다. 효능 수치는 v1 입력에 쓰지 않는다.
- **OmniPath**(omnipath + CollecTRI): 방향이 있는 신호 전달 +/−/부호 미상과 전사 조절 +/−/부호 미상, 모두 6채널이다.
- **결측 처리:**
  - 주석 없음 = unknown이다. 음성이나 RNA 0으로 바꾸지 않는다.
  - 978개 밖의 유전자 노드는 미측정 mask를 갖는다.
  - 작용 부호가 없으면 action_unknown을 표시한다.

## 비교군

| 이름 | 무엇이 다른가 |
|---|---|
| no_change / train_cell_mean / ECFP+X0 Ridge / cellmean+ECFP Ridge | 최소 기준선(R4 방식) |
| chem_only | backbone만: `m_c + MLP([ECFP, X0])` |
| B00 concat | 같은 지식을 고정 벡터로 연결(표적 embedding 합 + 고정 그래프 확산 profile) |
| B01 concat + 결측 학습 | B00 + 지식 결측 augmentation |
| B10 gated | 근거·세포 상태 조건부 gate가 있는 그래프 메시지 전달 |
| **B11 gated + 결측 학습 (제안)** | B10 + 지식 결측 augmentation |
| fixed graph | B1x와 같으나 gate가 관계 종류에만 의존(조건부 gate 제거) |
| XPert | 공식 코드를 R4 split으로 재학습 |

모든 기전 arm은 `Δ = Δ_chem + m_available · r`이다. 주석이 없으면 보정은 정확히 0이 된다.

## 순서와 판정

1. D0(주석 coverage), D1(학습 약물에서 화학·표적·반응 유사성 진단)
2. pilot: 5 arm × LR 3개 × seed 42. arm별 LR은 validation source ΔRMSE로 고른다.
3. 선택된 설정에 seed 43·44를 추가하고, fixed graph는 seed 3개로 돌린다.
4. 수정: source 진단에 근거해 최대 2회. 각 수정은 DECISIONS.md에 먼저 적는다.
5. 동결 commit → 최종 평가 1회(XPert test 예측 포함) → 귀속(조건 충족 시) → 보고서·튜토리얼

**validation 신호**는 다음을 모두 만족할 때다.
- B11의 3-seed 평균이 chem_only보다 0.001 이상 낮다.
- B11이 B01·B10·fixed graph보다 낮다.
- (B11−chem_only)의 약물 그룹 bootstrap CI가 0 미만이다.

신호가 없으면 귀속 실험(재배선 그래프 재학습, 근거 feature 제거)을 하지 않는다. 그 결과는 "미실행"으로 기록한다.
