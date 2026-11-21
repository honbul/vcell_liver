# R19 착수와 D0 검증 기록

## 시작 상태

검토 기준 `f1c087a`는 `origin/main`의 R17 병합이고, 실제 작업 기준 `6a34bff`는 `research/r14-cross-organ-data`의 R19 요청·기획서 커밋이다. 두 브랜치는 병합 이력이 다르다. 시작 시 Git 작업 트리는 깨끗했고 G1 Slurm에 기존 R18·DEV 작업은 없었다. 이후 R19에 한해 기존 R14 원표 복사, R17 CPU 재현, R18 메타데이터 지원 감사를 병행했다.

## R14 원표 회수

과거 완료 원장만으로 판단하지 않고 당시 채점 표를 그대로 확인할 수 있는지 확인했다. G1의 `vcell_liver_r14_cross_organ/execution-v1/`에서 compound, scaffold, MIX-seq, GSE306429, Emerald Bay, lung 원본 집계와 영수증 총 **33개**를 회수했고 기록 SHA-256과 전부 일치했다. Compound·scaffold·lung의 점수는 원래 `arms.tsv`, 지원은 원래 `by_fold_role.tsv`·`source_coverage.tsv` 또는 score receipt에 있으며 외부 세 코호트의 `metrics.tsv`에는 행·군 수가 들어 있다. 이어지는 비교에 이 복사본을 원본 증거로 사용한다. [해시 대조](data/r14-original/hash_reconciliation.tsv)

## D0 CPU 재집계

R17의 기존 결과가 물리 DMSO 분할과 동결 선택·예측으로 재현되는지 물었다. CPU 작업 `30656`이 18개 fold·역할 조합의 90개 동결 예측 영수증을 검증하고 `points.tsv`, `paired.tsv`, `selection.tsv`, score receipt를 원본과 바이트 단위로 맞췄다. 978개 랜드마크 축과 41 plate의 DMSO 529물리 웰 역할 배정도 맞았고, 처리 지원은 A 1,862 parent·6,773웰 및 B 1,815 parent·6,588웰이었다. DEV 기준선 4종은 명세만 등록했으며 신규 학습은 별도 결정이다. [대조 영수증](data/r17-reconciliation.json) · [기준선](baseline_contract.yaml)

## D0 수학 및 차단 검사

공통 양의 배율이나 같은 pool의 공통 이동이 어떤 지표를 실제로 바꿀 수 있는지 시험했다. 테스트 6개가 고정 contrast의 이동 불변성, cosine·검색의 양의 배율 불변성, 0배율·동률, RMSE·R² 항등식, 배열·유전자축, DEV의 예약 차단을 통과했다. 비영 norm이 기존 수치 임계값 위인 경우에 순위 불변성을 적용하며, 절대 delta와 RMSE·R²에는 해당 변환이 영향을 줄 수 있다. 불변인 두 지표 전용 sweep은 `NOT_APPLICABLE_INVARIANT`로 등록했다. [테스트 실행](data/test-results.txt) · [실험 표](experiment_matrix.tsv)

## R18 지원 감사

C1–C3에 등록할 처리 웰, DMSO 3역할 및 시나리오 B의 비간계열 입력이 확보되는지 물었다. 각 release의 서명과 개별 물리 웰을 연결하고, 과거 직접 노출·canonical parent 노출·QC·역할별 지원을 대조해 아래 표를 만들었다. 이 수치는 금지된 역할 공유와 release 중복을 제거한 지원이다. 후속 후보 평가는 R18 원 절차에 따라 별도 동결·채점한다.

| 코호트 | A parent / 물리 웰 / plate | B parent / 물리 웰 / plate |
|---|---:|---:|
| C1 CMap HEPG2 24 h 5–20 µM | 502 / 2,142 / 68 | 500 / 2,137 / 67 |
| C2 GSE70138 HEPG2 24 h, RNA 좌표 확인 | 221 / 1,304 / 7 | 221 / 1,304 / 7 |
| C2 CMap HEPG2 24 h | 426 / 3,392 / 55 | 422 / 3,385 / 55 |
| C3 HUH7 서명상 3 µM, 6 h | 68 / 239 / 3 | 12 / 59 / 3 |
| C3 HUH7 서명상 3 µM, 24 h | 67 / 218 / 3 | 14 / 53 / 3 |

- **C1** R16과 동일한 QC 행이다. R16의 종전 규약 채점 이력이 있고, B1 및 R8과 물리 웰이 일치한 것은 477군·2,070웰이다. 두 GEO release와 중복되는 웰은 0이었다.
- **C2** GSE70138의 공개 GEO 원본은 detection plate 좌표를 제공한다. DET 기준 QC·직접 노출 제거 후 221군·2,808웰을 찾았고, 기존 CMap 메타데이터의 검증된 RNA 좌표와 독립 DMSO 세 역할을 요구했을 때 같은 221군의 1,304웰이 남았다. CMap에서는 두 GEO release와 관련된 2,210개 서명을 제외했다. 두 release를 정확한 µM별로 합치면 A 587군·4,696웰, B 583군·4,689웰이며 100군 이상인 정확 농도는 9개다. 가장 작은 농도 차이도 원래 값을 보존했다. 직접 일치가 없는 웰에도 이전 프로젝트에 등장한 동일 parent가 있다.
- **C3** 서명 `pert_idose`의 3 µM와 물리 웰 농도 2.2972–3.825 µM를 따로 보존했다. 6 h의 QC 68군과 24 h의 QC 74군 중 67군이 직접 노출 제외와 RNA 3역할 조건을 통과했다. 착수 보고의 B 내부 선택 0–2군은 HUH7과 공통인 donor 교집합의 지원이었다. HEPG2 적격 적합군 전체에서 donor를 감사하면 명목 3 µM의 내부 선택 지원은 두 시간 모두 fold별 3–4군이며, 24 h에 같은 실측 용량대를 쓰면 9–11군이다. C3는 이 지원에서 기술 확인으로 남겨 둔다.

[물리 지원 원표](data/support.tsv) · [C2 정확 농도별 지원](data/c2_dose_support.tsv) · [C3 HEPG2 적합·선택 fold](data/c3_hepg2_fold_support.tsv) · [RNA plate별 DMSO](data/r18_control_support.tsv)

## 예약 등록과 실행 기록

`g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r19_plan/audit.YBfVmbuo/r18-registered`와 워크스페이스 `r19-audit-evidence-2026-11-21/`에 17개 감사 원표를 보존했다. 처리 웰·파생 서명·release 별칭 19,787행은 `r18_reserved_manifest.tsv`에, B가 입력으로 사용할 수 있는 비간계열 웰 19,454행은 `r18_B_donor_exceptions.tsv`에 등록했다. 이 숫자는 레지스트리 행 수이며 두 연구의 독립 표본 수는 위 지원표에 있다. guard를 실제 예약 레지스트리에 적용해 DEV에서 거부됨을 확인했다. [감사 해시 영수증](data/r18-support-receipt.json)

착수 시 Slurm은 A30 11슬롯, A40 6슬롯, A100 7슬롯이 가용했고, 다른 사용자 대기 한 건은 만족 불가능한 의존성으로 정지돼 있었다. 기존 LINCS 원본, 구조 특징, 비교 모델 코드와 설정은 G1에서 확인했다. 이번 단계의 GPU 사용량은 0시간이다. [자원 내역](data/resource_usage.tsv) · [실험 상태표](experiment_matrix.tsv)

## 다음 입력 잠금

현재 R18 실행 스냅샷은 G1의 `vcell_liver_r19_plan/confirm-ready.MnXUJ14q/repo`에 있다. 커밋 `1fe24ed`에서 실행 스크립트와 고정 명세를 묶고, 108개 조합의 양 역할 명부 해시, 세 원 자료의 축 메타데이터 영수증, 876 parent 구조 특징, 기존 비교표 5개와 R12 공통 적합 설정을 대조했다. 발현값을 읽기 전 단계이며 예약 발현 접근 승인 이후 스냅샷을 사용할 수 있다. [실행 준비 영수증](data/preflight/r18/readiness/receipt.json)

CPU 작업 30670에서 R18의 대상 7,295웰과 대조군 2,311웰을 기존 R17 역할·parent 분할에 대조했다. C3 24 h B의 HUH7 추가 설계는 공동 parent 분할에 따라 HEPG2와 HUH7을 내부 선택 풀에 등록했고, HUH7-only라면 비었던 fold까지 지원을 확인했다. CPU 작업 30671에서는 D1의 41개 plate별 세 역할에 가장 작은 역할의 DMSO 수를 적용해, 기존 529웰 중 486웰을 균등한 세 역할로 등록했다. D1의 공유/독립 선택 비교와 R18 적합·채점은 각각 별도 단계다. [R18 사전 명세](R18_CONFIRMATION_PROTOCOL.md) · [D1 제어군 지원](data/preflight/d1/plate_support.tsv) · [G1 작업 원장](data/preflight/g1_runs.tsv)
