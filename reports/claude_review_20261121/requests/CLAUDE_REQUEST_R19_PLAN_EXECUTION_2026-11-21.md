# 요청 R19: 기획서 실행 — 1단계 착수 보고와 D0

2026-11-21 · 근거 문서: `/home/gsun/work/vc_liver/VCELL_R18_AND_ARCHITECTURE_RESEARCH_PLAN_2026-11-21.md` (98 KB). 이 요청은 그 기획서의 21절 요청문을 실행으로 옮긴 것이다. 세부 설계는 기획서의 해당 절을 직접 읽고 따른다.

## 이번 요청의 범위

기획서 전체를 한 번에 실행하지 않는다. **착수 보고 + D0 + R18 C1–C3 지원 감사**까지만 한다. 그 결과를 보고한 뒤 다음 단계를 받는다.

## 1. 착수 보고 (모델 학습 전)

기획서 21절의 첫 보고 5항목을 낸다.

1. `f1c087a` 이후 저장소·서버 변경, 진행 중이거나 남아 있는 job.
2. R14 기존 동결 원표(`metrics.tsv`, paired, support, freeze/score receipt) 회수 가능 여부와 해시 대조. 재생성으로 대체하지 않는다. 접근 불가면 경로와 함께 blocker로 적는다.
3. R18 C1–C3 각각의 노출·QC·물리 DMSO 3역할 지원. 이미 일부 집계를 시작했으므로 그 결과를 먼저 확정한다.
4. 실제 자원(GPU 종류·가용 슬롯·대기 중인 다른 사용자)과 필수 자산 유무.
5. 실행 가능한 D/T/X/S/L/I ID 목록과 각각의 `PLANNED/BLOCKED/NOT_APPLICABLE` 사유.

보고는 짧게. 원표는 파일로 남기고 본문에는 수치 요약만 쓴다.

## 2. D0 (기획서 9절)

- R17의 점수·선택·역할 배정·gene axis·물리 웰/parent 수를 CPU-only로 재집계해 대조한다. 기존 결과를 덮어쓰지 않는다.
- `B_mean`, `B_resmean`, `B_context`, `B_blend_anchor`의 DEV 명세를 `baseline_contract.yaml`로 등록한다(실행은 아직 아님).
- 기획서 3.2절의 불변성 unit test를 만든다: 공통 양의 scalar에 대한 cosine/검색 불변성, 같은 pool 공통벡터 이동에 대한 contrast 불변성, RMSE/R²의 해석적 변화, shape/NaN/gene-order 검사. 해당하는 sweep이 수학적으로 무효이면 실행하지 말고 `NOT_APPLICABLE_INVARIANT`로 기록한다.
- 산출물: `progress_reconciliation.md`, `baseline_contract.yaml`, `exposure_manifest.tsv`, `support.tsv`, `tests/` 통과 기록.

## 3. R18-reserved manifest (기획서 5절)

C1–C3의 처리 웰·파생 서명·동일 물리 측정의 별칭을 `r18_reserved_manifest.tsv`로 등록하고, 이후 DEV의 적합·선택·특징 사전학습에서 차단되도록 한다. 특히 C2의 미사용 상태를 DEV가 먼저 소진하지 않게 한다. B의 허용 source donor 입력 예외는 별도 열로 둔다.

## 4. 하지 말 것

- 이번 요청에서 R18 채점과 DEV 학습을 시작하지 않는다. 지원 감사와 명세 고정까지다.
- 봉인·격리·예약 자료 개봉, 새 대용량 다운로드, controlled access, 새 비용 없음.
- 외부 처리 정답으로 gain·γ·gene scale·donor·rank를 고르지 않는다. R15-B의 oracle scalar 0.0764를 실사용 설정으로 복사하지 않는다.
- 원 R15–R17 산출물과 잠금 artifact를 수정하지 않는다.
- 정보 계약 A/B/M/G를 섞지 않는다.

## 5. 지키는 것

감사 → 사전 명세 커밋 → (다음 단계) 예측 동결 → 별도 채점. 관찰과 가설을 구분하고 사전 기준을 문자 그대로 적용한다. GPU는 기본 동시 2개로 시작하고 다른 사용자가 대기하면 8개를 넘지 않는다. 결과는 각 항목마다 **무엇을 물었나 / 무엇이 나왔나 / 어디까지 말할 수 있나 / 다음에 무엇을 하거나 안 하나** 네 문장으로 쓴다.
