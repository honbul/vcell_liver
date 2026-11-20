# P0-R9 output_utility_v3 — 실행 명세 (재집계 전 고정)

근거: private plan 2026-11-20 §5 (P0-R9), §2, §7. 기계 판독 명세: `configs/r9/output_utility_v3.json`. 이 문서와 그 config를 담은 commit이 freeze이며, 아래 재집계 표를 만들기 전에 commit한다.
기준: `research/r9-normalization-prototype-v2` @ `0cfd516`. 브랜치 `research/r9-output-utility-v3`.

- **질문**: 정규화를 바로잡은 현재 모델(LR_donor, Corrected)은 기존 신뢰도 정책(source OOF 온도 + 2% 경험적 source 오류율 임계값) 아래에서 known 외부 세포 중 몇 개에 답을 내고(fine/parent), 답한 것 중 몇 개가 틀리는가?
- **지위**: GSE151530 HCC 25 donor는 phase2-v2 E00, R9 A4 v1, R9 v2 C1에서 반복 사용된 **개발 진단(development diagnostic)** 자료다. blind·독립 외부 검증이 아니며, 이번 재집계로 정책의 독립 검증을 주장하지 않는다.
- **하지 않는 것**: 신규 학습·재적합 없음. 모델·checkpoint·변환·온도·τ_fine/τ_parent·prior·label 대응표를 바꾸지 않는다. 외부 정답·coverage를 보고 임계값을 고르지 않는다. 새 최소 coverage 합격선을 만들지 않는다.
- **입력**: 저장 예측만 사용 — v1 `assets/runs/a4_external_v1/*.npz`(Original), v2 `assets/runs/a4_external_norm_v2/*.npz`(Replay·Corrected), T2 실패 2 task의 v2 source OOF `a4_external_norm_v2_oof/*.npz`. 모두 이전 서버 worktree로의 읽기 전용 symlink. Calibration 계산 코드는 `scripts/r9/v2/c1_a4_eval.py`에서 그대로 복사하고 저장된 `calibration_provenance.tsv`와 일치를 확인한다.
- **버전**: 주 버전 = **Corrected**(현재 모델). Replay·Original은 보존용으로 같은 방식의 수를 함께 만든다(보조).
- **행 필드 (version×condition×seed×model×donor; plan §5.2)**: `n_known, n_fine, n_parent, n_abstain, n_output(=fine+parent), n_wrong_fine, n_wrong_parent, n_wrong_output, fine_coverage, parent_fraction, abstain_fraction, output_coverage`(각 수/`n_known`, `n_known=0`이면 NA), `output_error = n_wrong_output/n_output`(**`n_output=0`이면 NA**), `fine_error`, `parent_error`(분모 0이면 NA), `output_status ∈ {has_output, no_output, no_known_cells}`. Malignant·unclassified는 정확도에 넣지 않고 donor별 fine/parent/abstain 수만 따로 기록.
- **집계**: coverage = donor 안에서 seed 평균 → 고정 25 donor 동일 가중. 조건부 오류율 = donor 안에서 출력이 있는 seed만 평균 → 출력 seed가 1개 이상인 donor 동일 가중, 유효 donor 수(/25)·유효 donor×seed 행 수(/75) 병기. 전체 세포 합산 오류율(Σ오류/Σ출력, donor×seed 전체)은 별도 보조 열(`pooled_output_error_aux`)이며 donor-macro 값과 섞지 않는다. 무출력 donor 포함 전체 25 donor coverage 표(`coverage_per_donor.tsv`)를 항상 낸다.
- **통계 단위·재표집**: donor가 단위. donor bootstrap 10,000회(seed 0) 95% percentile CI. 세포·seed는 독립 반복으로 세지 않는다.
- **주 보고량(주 대비 없음)**: 기술 통계 재집계이므로 arm 간 대비를 두지 않는다. 주 보고량은 LR_donor·Corrected의 4 조건별 donor-macro fine coverage, output coverage, abstain fraction, 조건부 output error [95% CI]와 유효 donor/seed 수. 다른 6 head·Replay·Original은 보조.
- **분모 주의**: `sparse_<X>`는 source 학습 조건 이름이고 coverage는 외부 known 5 class 전체에 대한 값이다. target AUPRC는 target 양성 ≥10 donor에 대한 지표라 coverage와 donor 수가 다르다. fine은 broad 세포 유형 출력이지 희귀 세부 상태 인식이 아니다.
- **plan §5.4 판정 대응 규칙 (LR_donor·Corrected, 조건별, 결과 보기 전 고정)**
  - 규칙 A: donor-macro output coverage 점추정 < 0.50 → §5.4 1행(출력이 매우 적음: known 세포 과반에 답하지 않음 → 전면 자동 분류 채택 보류, 제한적 고신뢰 보조 용도는 별도 검토). 0.50은 plan §0-2의 "대부분의 세포" 표현을 옮긴 것이며 **합격선이 아니다**. 0.50 이상이라도 어떤 용도도 승인하지 않는다.
  - 규칙 B: donor-macro output error 95% CI 하한 > 0.02(정책 자체의 source 명목 오류율) → §5.4 2행(정책 실용성 부족 기록, 외부 결과로 임계값 재선택 금지).
  - 규칙 C: output coverage ≥ 0.50 그리고 output error 95% CI 상한 ≤ 0.02 → §5.4 3행(그 조건만 기술, 다른 희소 유형으로 일반화 금지, 개발 진단).
  - B·C 모두 아니면 "0.02 대비 미확립". A와 B는 동시에 적용될 수 있다. 무출력 행이 있으면 판정 옆에 명시한다.
- **검사 (통과해야 표를 사용)**: C1 fine+parent+abstain=n_known(전 행), 비율 합=1(known>0 행); C2 수 항등식; C3 (version,condition,seed,model)별 Σn_known=27,212·malignant 15,723·unclassified 7,088, version별 2,100행·중복 0; C4 분모 0 ⇔ NA; **C5 재계산 AUPRC(class별·allmapped·target·primary)가 저장 v2 표와 같음(NA 패턴 동일, |차이| ≤ 1e-5) 그리고 모델 수준 primary도 같음 — 다르면 집계 정정 범위를 벗어난 것이므로 중단하고 원인 분리**; C6 저장된 fine/parent/abstain 비율·fine_error·출력 있는 행의 output_error 일치, 무출력 행의 저장 output_error가 0으로 기록된 수를 센다; C7 T·τ가 calibration_provenance와 일치; C8 요약표는 원표(`output_counts_per_donor_seed*.tsv`)만 읽는 별도 단계에서 재생성.
- **비결정성**: 예측이 모두 저장되어 있으므로 replay하지 않는다. 검사 실패로 재구성이 필요할 때만 plan §5.5에 따라 알려진 1개 사례를 고정 조건으로 replay한다.
- **자원**: CPU job 1개(≤4 CPU, `v3_sbatch.sh`, 이름 `v3r9_*`), GPU 0. 1-task pilot으로 시간·메모리 측정 후 전체 실행.
- **산출물**: `output_counts_per_donor_seed.tsv`(Corrected) + `_original_replay.tsv`, `coverage_per_donor.tsv`, `coverage_error_summary.tsv`, `no_output_summary.tsv`, `per_class_output_pooled.tsv`(보조), `decision_mapping.tsv`, 검사표, `errata.md`, `usability_decision.md`, `decision.md`, `run_ledger.tsv`, `RESULTS.md`, 미실행 목록.
