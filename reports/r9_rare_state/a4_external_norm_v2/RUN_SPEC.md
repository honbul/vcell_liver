# R9-C1 a4_external_norm_v2 — 실행 명세 (외부 점수 산출 전 고정)

근거: private plan 2026-11-19 §6.1. 기계 판독 명세: `configs/r9/a4_external_norm_v2.json` (이 문서와 같은 commit이 freeze).

- **지위**: 이미 한 번 채점된 외부 평가(a4_external_v1, GSE151530)의 **구현 정정**. blind/독립 test가 아니다.
- **고칠 것**: v1 `a1_repr.py`는 외부 count를 source HVG 2,000개로 먼저 자른 뒤 CP10k+log1p 했다(분모 = HVG count 합). source는 공통 unique gene 전체(v1 meta 18,618개; 중복 포함 교집합 18,624개) 합으로 정규화한다. v2 외부 변환 = 공통 unique gene 전체(같은 목록·순서) 정렬 → 그 합으로 CP10k+log1p → source HVG → source scaler → PCA50 → PC scaler. 외부에서 적합하는 것은 없다.
- **고정 대상**: 7 head(LR_donor, P1_donor, GMM_K, A00, A10, A01, A11) × 4 조건 × seed 0/1/2, 선정 모델 LR_donor 유지. 외부 label로 HVG·PCA·threshold·모델·seed를 고르지 않는다.
- **표현 재구성 허용오차 T1** (v1은 scaler/PCA 객체를 저장하지 않음 → 같은 코드·random_state 42로 재적합 후 검증): shared 목록 길이, HVG 목록·순서, fitmask, cell ID는 정확히 같아야 함; source Z 최대 절대오차 ≤ 1e-3, PC별 Pearson ≥ 0.99999(부호 포함); 구 방식으로 재생성한 외부 Z 최대 절대오차 ≤ 1e-3. 통과 → 저장된 v1 source Z로 head 적합, Replay 입력 = 저장된 v1 Z_ext. 실패 → 재구성 Z로 head 적합, Replay/Corrected 모두 재구성 변환으로 투영, calibration은 새 source OOF(필수).
- **Head**: v1 checkpoint 없음 → 84 head를 같은 표본 ID·행 순서·seed·가중치·설정으로 재적합(외부 정답에 맞추는 조정 없음). 한 번 적합한 head로 Replay와 Corrected 입력을 함께 예측.
- **예측 replay 허용오차 T2** (task×model, 외부 50,023세포, temperature 전 확률): argmax 일치 ≥ 0.999 그리고 세포별 최대 |Δp|의 99백분위 ≤ 1e-3.
- **Calibration 정책**: task(조건×seed)의 7 model이 모두 T2 통과(그리고 T1 통과) → v1 source OOF logp에서 v1과 같은 방식으로 T·τ_fine·τ_parent 재사용. 아니면 그 task는 같은 3 donor-group source OOF를 새로 적합(run `a4_external_norm_v2_oof`)하고 새 T·τ를 Replay와 Corrected에 동일 적용. v1 calibration은 Original 열에만.
- **평가 단위**: version별 2,100행(7×4×3×25), primary 유효 1,428개(head당 nat 75, B 45, Endo 48, Fib 36), seed 평균 donor 700행, 모델 요약 28행, unmapped seed 168행/평균 56행. known 분모 27,212세포, malignant 15,723·unclassified 7,088은 출력/보류 분포로만. class 평가 조건(donor 양성 ≥10) 유지. 결측은 NA+사유로 남김.
- **주 대비**: 선정 모델 LR_donor의 4 조건 각각 **Corrected − Replay** (primary 지표: nat = mapped donor-macro AUPRC, sparse = target AUPRC). seed를 donor 안에서 평균 → donor paired bootstrap 10,000회(seed 0) 95% CI; 4개 family의 Bonferroni 동시 98.75% CI 병기. 나머지 6 head·모든 지표와 `Corrected−Original = (Corrected−Replay)+(Replay−Original)` 분해는 보조/서술.
- **실용 크기**: AUPRC 0.01(plan §2.3의 R9 제안값; v1 기준 아님). 크기 서술에만 쓰고 아무것도 선택하지 않는다.
- **판정**: 정규화 계약 일치, model/calibration provenance, 2,100 key 완전성, drift와 정규화 효과 분리가 되면 정정 완료. 점수 방향과 무관. A11 외부 점수가 좋아져도 v1 source 선정을 소급 변경하지 않는다.
