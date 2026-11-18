# R10 DECISIONS

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
