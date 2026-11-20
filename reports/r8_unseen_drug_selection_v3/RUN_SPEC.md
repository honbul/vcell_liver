# R8 v3 RUN_SPEC (동결)

- 동결 명세: `configs/r8_unseen_drug_v3/run_spec.yaml` — **freeze SHA `f0589c689efc9cf93dcd10b18d2d4cbbd4300891`** (파일 sha256 `f347aa2fae7ee63173f666cb76fc85a0a4befe117e0096d1caf3f0c7618b5f5e`). 외부(outer) 점수는 이 커밋 이후에만 계산되었다(`run_ledger.tsv`).
- 고정 입력: `configs/r8_unseen_drug_v3/alias_exceptions.tsv` (sha256 `c6b21049…`, 6 cluster / 12 group), 원 R8 store `store_v1` (`manifests/replicate_calibrated_r8/store_manifest.json`, 실행 시 hash 검증).
- 근거: 비공개 기획서 2026-11-20 §4.2–4.5, §7.
- 명세 요약: 질문, 분할(outer family 3 × v2 drug fold 3, inner drug fold 3), 적합 범위, 4개 arm(K=16), 선택 순서(κ_f·κ_b → ridge α → arm별 κ_b → λ·η, 고정 상수 없음), 주 대비 `ridge_C1chem − fb_C0` centred RMSE, two-way family×drug paired bootstrap(2,000, seed 20261120), 실용 기준 상대 −0.5%, fold 방향 ≥ 2/3, 판정표, 자원.
- 동결 후 변경: 명세 변경 없음(amendment 0건). 코드 변경은 report 모듈의 변수 이름 버그 1건(`36b5998`, 통계 정의 불변).
- 자원 상한 변경: 계획 당시 v3r8_ 동시 2개 → 2026-11-20 사용자 요청(coordinator 전달)으로 동시 8개. 설계는 바꾸지 않고 block(outer fold × drug fold)별 job으로 분할만 했다.
