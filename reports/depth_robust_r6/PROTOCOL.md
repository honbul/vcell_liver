# R6 프로토콜 — 관측량이 줄어도 약물 반응 정보를 보존하는 상태 표현

작성 2026-11-17 (pilot 실행 전). 브랜치 `research/r6-depth-robust-response`, 시작점 R3 커밋 `90abe72`.
이 문서는 **실행 규칙의 요약**이다. 비공개 기획 문서 원문은 저장소에 올리지 않고, 읽은 파일의 크기와 해시만 남긴다.

## 0. 읽은 기획 문서 (내용이 아니라 수신 확인용 해시)

| 파일 | 바이트 | sha256 |
|---|---|---|
| `00_OVERVIEW_AND_GIT_REVIEW.md` | 16,097 | `eb5659832a23a4fa5e6d87dbf5fbc4c5d4ceb0553def6fca8d869fdac20be4f9` |
| `01_R6_DEPTH_ROBUST_RESPONSE.md` | 17,683 | `5a139adf0392c1752e7ddb27e73b52e20b2306e3fe1995cdfdcd47df5f1151fe` |
| `04_AGENT_GOALS.md` | 8,266 | `17928d34375b4ea61064fa5ba1d7c000d2e2b8bf476cf5e22e1da3f98336f8ea` |
| `05_EVIDENCE_AND_REFERENCES.md` | 18,312 | `a157014dfb39d50ef91f53547fe44beae2e749bf553fd90156e4338bc1fb3fb9` |

네 파일 모두 마지막 절까지 읽었다. R7·R8 명세(02, 03)는 이 연구의 범위가 아니어서 열지 않았다.

## 1. 질문과 과제

**질문.** 대조군 세포의 RNA 분자를 적게 읽었을 때에도, 학습에 없던 약물의 실제 평균 반응(ΔRNA)을 예측하는 데 필요한 세포 상태 정보를 보존할 수 있는가.

**한 행.** (세포주, 약물 구조 그룹) 조건 하나.
- 입력: 그 조건의 plate에 있는 같은 세포주 DMSO 대조군 세포 64개의 **정수 UMI count**(유지율 p로 박화한 것)와 약물의 ECFP4 2,048비트.
- 정답: `mean(log1p CP10k, 처리군 전체 well) − mean(log1p CP10k, 기준 대조군)` — 2,000개 유전자. **p와 무관하게 고정**이고 박화하지 않는다.
- 독립 표본 단위: 약물 구조 그룹(한 그룹의 모든 세포주를 함께 묶는다). 세포, 박화 seed, 학습 seed는 독립 표본이 아니다.

분류, denoising 점수, 세포 분포 생성은 이 연구의 목표가 아니다.

## 2. 순서 (공통 규칙 그대로)

`diagnostic → joint_pilot → source_locked_selection → limited_revision ≤2 → freeze → final_eval → attribution/report`

1. **R6-0 진단**: 박화가 관측 상태(분자 수, 검출 유전자, 집합 평균, 토큰)와 단순 예측기의 Δ 오차에 주는 변화. p를 0.25 아래로 내리지 않는다.
2. **R6-1 공동 pilot**: B00/B10/B01/B11 × 학습률 3개, seed 42, 개발 fold. 통제군(DEN, NOSTATE)과 teacher도 같은 격자.
3. **선택**: 개발 fold의 주지표만 사용. 시험 약물 70개는 동결 전까지 열지 않는다.
4. **수정 최대 2회**: 실행 전에 `DECISIONS.md`에 관측/가능한 원인/예측/변경 하나/판정 기준을 적는다. 모든 비교군의 대칭 학습 연장은 수정으로 세지 않고 성과로도 쓰지 않는다.
5. **동결**: 설정과 freeze manifest(코드 SHA 포함)를 커밋한 **뒤에** 최종 모델을 학습하고 시험 약물을 한 번 채점한다.
6. **귀속/보고**: 실제로 차이를 만든 모듈만 제거 대조한다.

## 3. 고정 계약 (자세한 수치는 `manifests/depth_robust_r6/DATA_CONTRACT.json`)

- 원자료: Tahoe-100M 공개 parquet mirror(읽기 전용). DMSO 세포의 **정수 count**를 다시 읽어 캐시했다. 이전 연구의 log 정규화 배열을 UMI로 되돌려 쓰지 않는다.
- 0.5 µM 한 용량, 2,000-gene panel, 약물 구조 그룹 209/70/70 분할은 이전 연구 그대로.
- 대조군 역할(바코드 해시 순, 서로 겹치지 않음): `eval_input` 64개(평가 입력), `fit_input` 최대 256개(학습 입력), `reference` 나머지(정답의 대조군 평균).
- 관측량: p = 1.0 / 0.5 / 0.25, N = 64. 박화는 `Binomial(c, p)`를 panel 유전자와 panel 밖 합계에 각각 적용하고 library size는 박화된 합으로 다시 계산한다. 정규화 식은 모든 p에서 같다.
- 개발: train 약물 안의 3-fold(R3와 같은 fold). fold 0·1로 학습, fold 2로 평가. 최종: train 209개로 학습, test 70개로 평가. validation 70개는 쓰지 않는다.
- 추론 입력에 p 값 자체는 넣지 않는다(관측 library size만).

## 4. 비교 모델

| ID | 입력 | 학습 |
|---|---|---|
| nochange / linemean | 없음 / 세포주 식별 | 0 / 세포주별 평균 반응 (참고선) |
| Z0 | 박화 대조군 집합 평균 + ECFP4 | 2-블록 Ridge, 같은 박화 증강 |
| B00 | 기본 입력(유전자 ID, log 정규화 값, 세포 내 순위) | supervised + 박화 증강 |
| B10 | 기본 + 관측 정보(log1p count, c/(c+τ), library size, 검출 수, 검출 비율) | B00와 같음 |
| B01 | 기본 | supervised + 고정 teacher의 반응·잠재 일관성 |
| B11 | 관측 정보 포함 | B01과 같은 목적 |
| DEN (Denoise-control) | B11과 같음 | supervised + 전체 관측 발현 재구성 |
| NOSTATE | RNA 상태 없음(학습 상수) + ECFP4 | supervised |
| Public-reference | 박화 대조군 세포 + 약물 | 공개 모델 하나 (pilot 전 고정, `DECISIONS.md`) |

신경망은 모두 같은 작은 encoder(2 block, hidden 128, 256 토큰, 평균+attention 집계, 전체 panel 요약 MLP skip), 같은 Δ head, 같은 step 수를 쓴다. teacher는 같은 학습 약물만으로 전체 관측에서 학습한 고정 모델이고, 그 학습 비용을 계산량에 포함한다.

## 5. 지표와 판정

- **주지표**: p = 0.25, N = 64에서 조건별 ΔRMSE(2,000 유전자) → 약물 그룹 안 평균 → 약물 그룹 간 동일 가중 평균. 박화 seed 3개와 학습 seed 3개는 조건별로 평균한다.
- **보호 지표**: p = 1.0 오차. 같은 예산의 B00 대비 상대 악화 1%(운영값)를 넘으면 "저관측 이득"과 "원래 성능 손실"을 함께 적고 채택하지 않는다.
- **불확실성**: 약물 그룹 단위 paired bootstrap(5,000회) 95% 구간.
- **"더 좋다"의 운영 정의(사전 지정)**: 주지표의 paired 차이 구간이 0을 포함하지 않고 **그리고** 상대 차이가 1% 이상. 구간이 0을 제외하지만 1% 미만이면 "검출되지만 실용 크기 미달"로 따로 적고 채택 근거로 쓰지 않는다.
- 보조: p별 오차 곡선, 각자 평균을 뺀 ΔPearson, 고정 후보 retrieval, Top-20 큰 변화 방향, 예측 크기, 실제 wall-time·VRAM·관측 분자 수, 세포주별 표, SNU-423 표.
- 기획서 §8의 결론 행을 그대로 적용한다(B11 우위 / B10만 우위 / B01≈B11 / 모두 같음 / 재구성만 개선).

## 6. 운영

Slurm만 사용(job 이름 `r6_`), run ID는 한 번만 쓴다. 생성물은 서버 `assets/`(git 제외). Git에는 코드·설정·manifest·해시·작은 표·보고서만 올린다. count 행렬, 발현 배열, checkpoint, 기획 문서는 올리지 않는다.
이것은 **이미 결과를 본 공개 benchmark에서 사전에 고정한 후속 평가**이며 blind test가 아니다(`EXPOSURE.md`).

## 7. 동결 전 변경 기록 (2026-11-17)

- **수정 1:** 개발 fold pilot에서 모든 신경망 arm이 세포주 평균보다 나빠, 반응 head를 `Δ = g(상태) + h(상태, 약물)`로 나누고 h에 크기 벌점(λ=10)을 넣었다. 모든 arm에 똑같이 적용했다. 실행 전 기록과 결과는 [DECISIONS.md](DECISIONS.md) D7–D9.
- **수정 2:** 쓰지 않았다.
- 4절의 비교 모델, 5절의 지표와 판정 규칙은 바꾸지 않았다. 동결 내용은 `manifests/depth_robust_r6/freeze_manifest.json`.
