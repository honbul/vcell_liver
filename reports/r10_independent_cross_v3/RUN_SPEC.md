# R10 v3 실행 명세 — P0-R10 독립 자료 적격성 (조건부 P1-R10)

작성 2026-11-20. 근거: private plan 2026-11-20 §3(R10), §2, §7. 기점 브랜치 `research/r10-contract-assembly-v2` @ `293009901383220ddbedaa61171993935a5b03bf`.
이 문서를 포함한 commit이 P0-R10의 **판정 규칙 freeze**다.

> 시점 기록: 후보 metadata(공개 hash sample sheet·pData·gene annotation·GEO sample 기록, 서버 보유 Tahoe-100M 소형 metadata)의 열람은 이 commit 이전에 시작했다. 아래 여섯 조건과 판정 규칙은 기획서 §3.2–3.4의 문언을 그대로 옮긴 것이며, metadata를 본 뒤 완화·추가하지 않았다. 발현 행렬은 내려받거나 채점하지 않았다.

## 1. 질문

기존 Sci-Plex3 24 h 자료에서 보인 cross 개선(`D11[cross] − W[cross]` actual TV −0.0102 [−0.0151, −0.0051], test E1000; 노출 자료)이 **새 독립 배양·측정 자료**에서도 유지되는지 검증할 수 있는 자료가 있는가.

## 2. 탐색 범위(비용 상한, 과학적 기준 아님)

- 기존 자산과 직접 관련된 후보 **최대 3개**. 서버 보유 자료(pubdata·workbench) 우선, 이어 공개 metadata.
- metadata 수준 확인만. 발현 행렬 다운로드는 (공개·무료·로그인/신청 불필요) **그리고** 후보가 metadata 적격 심사를 통과 **그리고** 총 ≤ 30 GB일 때만.
- 새 HCC 세포주, R9 환자 주석 자료, 기존 Sci-Plex3 test의 재사용·재분할은 대상이 아니다.

## 3. 필수 조건(모두 충족해야 적격; 기획서 §3.2)

| id | 조건 |
|---|---|
| C1 | raw count 또는 원 전처리 재현 가능 자료, 약물 ID, dose, 처리 시간, vehicle 정보 |
| C2 | 배양 날짜·생물학적 replicate·plate·sequencing batch를 구분할 수 있음 |
| C3 | 24 h 및 100–1,000–10,000 nM(중간 1,000 nM 예측) 계약과 호환. 10–100–1,000 nM은 보조 |
| C4 | cross의 중간 용량 query가 양 endpoint와 다른 replicate·plate 조건에 있음(same/cross 정의를 파일로 명시 가능) |
| C5 | 중간 용량 RNA·상태 비율·회수 세포 수가 입력·선택에 들어가지 않게 할 수 있음 |
| C6 | 유전자·입력 차원·전처리·line-specific atlas를 재적합 없이 적용 가능 |

해석 규칙(기획서 문언): 기존 세포의 무작위 재분할은 독립 batch가 아니다. 독립 배양 정보 없이 plate만 다르면 생물학적 독립 반복이라 부르지 않는다. 새 batch 하나는 그 batch의 재현만 확인한다.

각 조건 판정값: `pass` / `partial` / `fail` / `unverified`. **하나라도 pass가 아니면 해당 후보는 부적격**이다(partial도 미충족으로 센다).

## 4. 판정 표

| 관찰 | P0-R10 결과 | 다음 |
|---|---|---|
| 적격 후보 ≥ 1, 검정력 검토에서 주 대비 정밀도 확보 | 독립 검증 가능 | P1-R10 명세(`frozen_protocol.yaml`) freeze 후 1회 평가 |
| 적격 후보 ≥ 1, 정밀도 부족 | 예비 검증만 가능 | P1-R10을 "외부 예비 검증"으로 명명해 freeze 후 1회 평가 |
| 적격 후보 0 | **자료 대기** | `blocked_data_requirements.md`; P1-R10 미실행 기록 |

부적격 후보로 P1-R10을 돌리는 것(조건 완화)은 사용자 결정 사항이며, 이 패키지에서 자동으로 하지 않는다.

## 5. 정밀도 검토(자료 확보·채점 전, 기획서 §3.4)

기존 source 변동으로 필요한 독립 단위 수를 가늠한다. 입력: v2 `strata_primary_contrasts.tsv`·`primary_contrasts.tsv`의 P6(`D11[cross] − W[cross]`, E1000, actual TV) 약물 bootstrap 95% 구간. 근사: SE ≈ (hi − lo)/3.92, 약물 SD ≈ SE·√n. 새 자료 n 약물의 정규 근사 검정력 = Φ(|δ|/(SD/√n) − 1.96), 80% 검정력 n = ((1.96 + 0.8416)·SD/δ)². δ는 관측 추정치(낙관적)와 실용 기준 0.005. between-batch 변동은 이 source로 추정할 수 없으므로 결과는 **낙관적 하한**이다. development는 head 선택 집합, test는 노출 자료로 표시하며, 여기서는 분산 계획에만 쓴다.

## 6. 자원

GPU 0. 로그인 노드의 소형 파일 확인만 예정(Slurm job 없음). 발현 행렬 처리는 이 패키지에 없다.

## 7. 산출물

`data_eligibility.tsv`, `candidate_*` 설계 요약표, `metadata_checks.json`, `precision_review.tsv`, `run_ledger.tsv`, `decision.md`, `RESULTS.md`(미실행 목록 포함), 자료가 없으면 `blocked_data_requirements.md`.
재생성 스크립트: `scripts/r10/v3_metadata_eligibility.py`, `scripts/r10/v3_precision_review.py`.
