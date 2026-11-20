# R10 v3 결과 — cross 개선의 독립 확인 (P0-R10 적격성 / P1-R10 조건부)

작성 2026-11-20. 근거: private plan 2026-11-20 §3. 브랜치 `research/r10-independent-cross-v3`(기점 `2930099`). 판정 규칙 freeze **`2b91274`**([`RUN_SPEC.md`](RUN_SPEC.md)). 표기: [관측] 측정·확인한 값, [추론] 해석, [미검증] 확인하지 않은 것.

## 요약 (패키지별 6줄)

**P0-R10**
```text
연구: R10 P0 / 질문: 기존 cross 개선(D11[cross] < W[cross])을 새 독립 배양·측정에서 확인할 수 있는 자료가 있는가?
실행: 기존 자산과 직결된 후보 3개(Sci-Plex2, Sci-Plex4 공개 metadata; 서버 보유 Tahoe-100M)를 기획서 §3.2 필수 조건 6개로 metadata 수준 심사(발현 행렬 미다운로드).
결과: 적격 0/3. Sci-Plex2 = C2 fail(배양 날짜·생물학적 replicate 미식별)·C4/C6 partial; Sci-Plex4 = C2·C4 fail; Tahoe-100M = C3 fail(0.05/0.5/5 µM만)·C2 fail·C6 fail/미검증.
판정: 자료 대기.
다음 행동: 자료 확보 요청(blocked_data_requirements.md). 새 학습·모델 확장 없음.
사용자 결정 필요: 자료 — 신규 독립 batch 실험 예산/협력(R1) 또는 Sci-Plex2를 C2 면제 외부 예비 검증으로 돌릴지(R3) 중 하나.
```

**P1-R10**
```text
연구: R10 P1 / 질문: 같은 endpoint 관측에서 D11[cross]가 W[cross]보다 중간 용량(1 µM) actual 구성비 TV가 작은가(새 독립 자료)?
실행: 미실행 — 적격 자료 없음(P0). frozen_protocol 및 채점 산출물 없음.
결과: 없음. 정밀도 검토(source 변동, 낙관적): 세 세포주 기준 관측 효과 80% 검정력에 약 19 약물(0.005 기준 약 79), A549 단독 25–123 약물; Sci-Plex2(4 약물)면 95% 구간 반폭 ≈ 0.031–0.038.
판정: 자료 대기(실패 아님). v2 P6 −0.0102 [−0.0151, −0.0051]은 여전히 노출 자료 후속 비교.
다음 행동: 적격 자료 확보 후 frozen_protocol.yaml freeze → 1회 평가.
사용자 결정 필요: P0와 같은 자료 결정 하나.
```

## 1. P0-R10 상세

### 목적
기획서 §3.2의 필수 조건 C1–C6을 모두 충족하는 독립 자료가 있는지 확인(탐색 상한 3개, metadata 수준).

### 수정 내용
- 신규 `scripts/r10/v3_metadata_eligibility.py`(소형 metadata와 v1 atlas의 HVG id만 읽어 설계 요약표 생성), `scripts/r10/v3_precision_review.py`(v2 P6 구간으로 정밀도 검토).
- 기존 코드·v1/v2 산출물은 수정하지 않았다.

### 입력
- 공개 GEO(HTTPS, 일반 User-Agent, 인증 없음): GSM4150377(Sci-Plex2)·GSM4150379(Sci-Plex4)의 hashSampleSheet·pData·gene annotation, GSM4150377/8/9 sample 기록. hash와 크기는 [`metadata_receipt.tsv`](metadata_receipt.tsv). 서버 `assets/meta_dl/GSE139944_sciplex2_4/`(Git 제외).
- 서버 보유: Tahoe-100M `metadata/sample_metadata.parquet`·`cell_line_metadata.parquet`(읽기 전용).
- v1 atlas(읽기 전용): `vcell_liver_r10_dose_composition/assets/cache/r10_v1/atlas_{A549,MCF7,K562}/decoder.npz`의 `hvg_id`, `genes_human.tsv`, `obs_{A549,MCF7}.parquet`의 `n.umi`.

### 결과 ([`data_eligibility.tsv`](data_eligibility.tsv), [`metadata_checks.json`](metadata_checks.json))

| 조건 | K1 Sci-Plex2 | K2 Sci-Plex4 | K3 Tahoe-100M |
|---|---|---|---|
| C1 raw·ID·dose·time·vehicle | pass | pass | partial(처리 시간 로컬 README에서 미확인) |
| C2 배양 날짜·bio-rep·plate·seq batch | **fail** | **fail** | **fail** |
| C3 24 h, 100–1,000–10,000 nM | pass(E100 불가) | pass | **fail** |
| C4 cross query 다른 rep·plate | partial(hash-plate A/B만) | **fail** | partial(용량=plate 결합) |
| C5 중간 용량 정보 차단 | pass | pass | pass |
| C6 유전자·atlas 무재적합 | partial(QC·depth) | pass(기계적) | fail/미검증 |
| 적격 | 아니오 | 아니오 | 아니오 |

- [관측] Sci-Plex2: A549, BMS-345541·dexamethasone·nutlin-3A·SAHA(넷 다 R10 188 화합물 split에 없음), 0/0.1/0.5/1/5/10/50/100 µM, 24 h(GEO 추출 프로토콜). 약물×용량당 6 well, hash-plate 접두 A/B에 3/3(SAHA 100 µM 한 well만 E 표기). library 1개(sciChem18), NextSeq 550(Sci-Plex3는 NovaSeq 6000). 유전자 id 58,347개가 source와 동일, A549 HVG 2,000개 모두 존재. 단일-hash QC 유사 규칙(q < 0.05, ratio ≥ 10, UMI ≥ 500; 기술 요약용) 통과 14,793/24,262 세포, median UMI 4,438(source A549 QC 세포 1,298). E1000 용량 72 well 중 4개 < 20 세포([`candidate_sciplex2_wells.tsv`](candidate_sciplex2_wells.tsv)).
- [관측] Sci-Plex4: A549·MCF7, abexinostat·pracinostat(둘 다 R10 train 약물), HDACi 0.01–10 µM, 보충제 축 별도. 보충제 없는 용량 계열은 A549 plate10, MCF7 plate5에만 있음([`candidate_sciplex4_layout.tsv`](candidate_sciplex4_layout.tsv)). pData에 hash QC 열 없음.
- [관측] Tahoe-100M: 14 plate, 379 약물, 0.05/0.5/5 µM; 13 plate가 한 용량으로 ≥ 90/96 sample; drug×dose 1,137 조건 중 975개가 sample 1개([`candidate_tahoe_plate_dose.tsv`](candidate_tahoe_plate_dose.tsv)). cell_line_metadata에 A549·MCF-7 이름은 있으나 실제 세포 수는 미확인(2.2 GB obs_metadata 미조회), K562 없음.

### 정밀도 검토 ([`precision_review.tsv`](precision_review.tsv))
v2 P6 약물 bootstrap 구간에서 SE ≈ 구간폭/3.92, 약물 SD ≈ SE·√n(정규 근사, between-batch 변동 미포함 → 낙관적).

| source 범위(노출 지위) | 추정 [95%] | 약물 SD | 80% 검정력 n(관측 효과 / 0.005) | n=4 검정력 · 반폭 |
|---|---|---|---|---|
| 전체 세포주, test(노출) | −0.0102 [−0.0151, −0.0051] | 0.016 | 19 / 79 | 0.25 · 0.015 |
| A549, test(노출) | −0.0178 [−0.0284, −0.0079] | 0.032 | 25 / 317 | 0.20 · 0.031 |
| A549, development(선택 집합) | −0.0099 [−0.0242, +0.0018] | 0.039 | 123 / 482 | 0.07 · 0.038 |

[추론] Sci-Plex2(A549, 4 약물, 1 batch)는 조건을 면제해도 사전 기준 2(95% 상한 < 0)를 판정할 정밀도가 없다. 이는 예비 검증 가치에 대한 계획 판단이며 결과 예측이 아니다.

### 의미
- [관측] 기존 자산과 직결된 공개·보유 자료 중 독립 batch 계약을 문서화해 제공하는 자료는 확인되지 않았다(탐색 상한 3개 안에서).
- [미검증] 상한 밖의 다른 공개 자료 존재 여부.

## 2. P1-R10
미실행(자료 대기). 고정할 명세 항목은 [`blocked_data_requirements.md`](blocked_data_requirements.md) §4에 초안으로만 남겼다(채점 전 freeze 대상; 이번에 freeze하지 않음).

## 3. 미실행 기록

| 항목 | 이유 | 여는 조건 |
|---|---|---|
| P1-R10 본 평가(`frozen_protocol.yaml`, `per_triplet_metrics.tsv`, `per_batch_drug_metrics.tsv`, `primary_result.tsv`, `secondary_results.tsv`) | 적격 자료 0 | §2 계약을 만족하는 자료 확보 후 freeze |
| K1·K2 발현 행렬 다운로드 | metadata 심사 불통과(다운로드 조건 미충족) | 사용자가 R3(C2 면제 예비 검증) 선택 |
| Tahoe-100M 세포주별 세포 수(obs_metadata 2.2 GB) | C3 fail로 결론에 영향 없음 | Tahoe를 다른 계약으로 쓰기로 할 때 |
| Tahoe 처리 시간 확인 | 로컬 README에 기재 없음, C3 fail로 결론 불변 | 동상 |
| 추가 후보 탐색(웹 검색) | 후보 상한 3개 도달 | 사용자가 상한 확대(R4) |
| E100 보조 평가 | P1 미실행 | P1과 함께 |

## 4. 편차·사건

- 후보 metadata 열람은 RUN_SPEC freeze(`2b91274`) 이전에 시작했다. 조건·판정 규칙은 기획서 문언 그대로이며 열람 후 바꾸지 않았다.
- 첫 적격성 스크립트 실행(`v3r10_p0_elig_01`, `da9ca1e`)이 JSON 직렬화 오류(tuple key)로 실패 → 수정(`b9e49eb`) 후 새 run ID로 재실행. 결과 파일은 서버·로컬 sha256 일치.
- GEO sample 기록 첫 조회(~00:45Z)는 로컬 임시 파일로 받았다(공개 소형 텍스트). 출처 보존을 위해 서버에 다시 받았고 로컬 사본은 삭제했다.
- `candidate_sciplex2_wells.tsv`의 단일-hash QC는 기술 요약용이며 동결된 전처리 규칙이 아니다.

## 5. 자원

| 항목 | 값 |
|---|---|
| Slurm job | 0개(v3r10_ 제출 없음) |
| CPU | 로그인 노드 소형 파일 처리만: 스크립트 2회 user+sys ≈ 13.5 CPU-초 + one-liner 점검 수십 초 |
| GPU | 0 |
| 다운로드 | 공개 metadata 9 파일, 3.6 MB(서버 `assets/meta_dl/`) |
| 기록 | [`run_ledger.tsv`](run_ledger.tsv) |
