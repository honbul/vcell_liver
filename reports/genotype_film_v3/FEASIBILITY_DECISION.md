# HCC genotype-FiLM v3 — 실행 가능성 판정 (FEASIBILITY_DECISION)

**2026-11-16 · 감사 commit `3b57156` · 상세는 [DATA_AUDIT.md](DATA_AUDIT.md)**

## 판정: `STOP_DATA_INSUFFICIENT`

선택된 플랫폼 없음. Phase 1–4는 실행하지 않는다.

| 항목 | L1000 GSE92742 | L1000 GSE70138 | L1000 LINCS2020 | Tahoe-100M |
|---|---|---|---|---|
| 서버의 자료 수준 | Level 3·5 | Level 3·5 | **Level 5만** | scRNA raw count |
| 독립 성인 HCC 계열 | **1** (Huh-7) | **0** | **3** (Huh-7, JHH-5, JHH-7) | **1** (SNU-423) |
| 제외한 간 유래 모델 | HepG2, PHH | HepG2, PHH | HepG2, PHH (Huh-7.5.1은 Huh-7 계열) | HepG2/C3A |
| 최적 단일 농도·시간 | (계열 1개) | — | 10 µM · 24 h | 5 µM (계열 1개) |
| 그 조건의 약물 수, ≥1 / ≥2 / ≥3 계열 | 1 / 0 / 0 (최대 68 / 0 / 0) | 0 | 747 / 103 / 5 | 379 / 0 / 0 |
| outer fold별 source ≥3계열 약물 (기준 30개 이상) | 0 | 0 | 0, 0, 0 | 0 |
| 같은 plate 대조군 | 있음 | — | signature 수준으로만 확인 | 있음 |
| 독립 반복 | replicate plate 3장, 설계 1개 | — | JHH-5·JHH-7은 대부분 1회 측정 | 조건의 14 %만 well 2개 이상 |
| 엄격 LOCO | 불가능 (source 계열 없음) | 불가능 | inner 3-fold 불가능 (source 2계열) | 적격 조건 0 / 379 (세 농도 모두) |
| DepMap genotype 연결 | 됨 | — | 3계열 모두 됨 | 됨 |
| genotype 상태의 계열 간 대비 | 없음 (1계열) | — | 변이 상태마다 1계열 | 없음 (1계열) |
| 등급 | `DATA_FAIL` | `DATA_FAIL` | `CONTRACT_CHANGE_REQUIRED`, Level 3가 있어도 `PILOT_ONLY` 상한 | `DATA_FAIL` + `STRICT_LOCO_INELIGIBLE` |

## 탈락 이유

1. Level 3가 있는 두 L1000 release에는 약물 처리된 성인 HCC 계열이 1개(Huh-7)뿐이다.
2. 성인 HCC가 3계열인 CMap 2020은 서버에 Level 5만 있어 계획서의 주 과제(Level 3 발현 차이)를 실행할 수 없다. Level 5로 바꾸는 것은 과제 계약의 변경이다.
3. CMap 2020의 Level 3를 구하더라도 3계열이 상한이다. 세 계열에 공통인 약물은 5개, 한 계열을 보류하면 source는 2계열이라 "source 3계열 이상" 약물은 0개이고 inner 검증 fold를 만들 수 없다. 두 계열은 반복 측정이 없다.
4. Tahoe-100M의 성인 HCC는 SNU-423 하나다. 모든 처리 well에 SNU-423이 함께 들어 있어 엄격 LOCO 적격 조건이 0개다.
5. 네 계열의 genotype은 DepMap에 연결되지만, 한 플랫폼 안에서는 변이 상태마다 계열이 하나씩이라 genotype과 계열을 구분할 수 없다.

## 이 판정이 뜻하는 것과 뜻하지 않는 것

- 뜻하는 것: 서버에 있는 L1000 release와 Tahoe-100M으로는 "다중 HCC 계열 LOCO" 실험을 만들 수 없다.
- 뜻하지 않는 것: genotype 조건화나 FiLM이 효과가 없다는 결론이 아니다. 모델은 하나도 학습하지 않았다.

## 하지 않은 것

pan-cancer 학습, 여러 플랫폼·release 통합, 환자 자료 전이, HepG2/C3A 포함, Level 5 signature 과제로의 전환, CMap 2020 Level 3 내려받기, 공개 모델(TranSiGen) 점검. 모두 연구 범위나 계약을 바꾸는 일이라 자동으로 진행하지 않았다.
