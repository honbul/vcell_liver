# HCC genotype-FiLM v3 — Phase 0 데이터 감사 (DATA_AUDIT)

**작성일:** 2026-11-16 · **브랜치:** `research/hcc-genotype-film-v3` · **감사 실행 commit:** `3b57156` (Slurm job 23893)
**기준 문서:** `HCC_GENOTYPE_FILM_EXPERIMENT_PLAN_v3_2026-11-16.md` (비공개 문서라 저장소에 넣지 않았다)
**판정:** `STOP_DATA_INSUFFICIENT` — Level 3 경로와 Tahoe 경로 모두 `DATA_FAIL`. Phase 1 이후는 실행하지 않았다.

이 문서는 metadata만 읽은 감사 결과다. 발현 행렬의 값, 처리 반응의 크기, 모델 점수는 하나도 사용하지 않았다.

---

## 1. 가설 (확인하려던 것)

계획서 §3의 전제는 "한 플랫폼 안에 독립된 성인 HCC 세포주 계열이 여러 개 있고, 같은 약물·농도·시간으로 처리됐으며, 같은 plate의 대조군과 DepMap genotype이 연결된다"는 것이다. 계획서는 이것을 확인되지 않은 가정으로 표시했고, Phase 0에서 실제 수치로 판정하도록 했다.

진행 기준(계획서 §3.5)은 다음과 같다.

| 등급 | 기준 |
|---|---|
| `DATA_FAIL` | 독립 성인 HCC 계열 3개 미만, 또는 대조군·주석·동일 조건을 연결할 수 없음, 또는 독립 LOCO 불가능 |
| `PILOT_ONLY` | 독립 계열 3–7개, 또는 주요 genotype 상태가 사실상 한 계열에만 있음 |
| `ELIGIBLE_FOR_BENCHMARK` | 독립 계열 8개 이상, 각 outer 평가에서 약물 30개 이상, 각 약물이 source 3개 계열 이상에서 관측 |

## 2. 실제 실행

1. **규칙을 먼저 고정했다.** `manifests/genotype_film_v3/audit_policy.yaml`을 어떤 집계도 하기 전에 commit했다(`473ad36`). 성인 HCC 판정에 "두 출처가 다르면 보류"를 넣은 v2 수정(`ba4a05f`)은 L1000의 간 세포주 이름과 DepMap 간 계열 목록만 본 상태에서, Tahoe 세포주 목록과 모든 조건·반복 수를 보기 전에 했다. 수정 시점과 이유는 정책 파일의 `amendments`에 적었다.
2. **조사한 자료** (모두 서버에 이미 있던 파일이며 새로 받은 것은 Cellosaurus 조회표뿐이다)
   - L1000 `GSE92742`(Phase I)와 `GSE70138`(Phase II): instance·cell·gene metadata 전체. Level 3 행렬은 열·행 ID만 읽었다.
   - L1000 `LINCS2020`(CMap 2020 beta): cell·signature metadata. 서버에는 **Level 5 행렬만 있고 Level 3와 instance metadata는 없다.** 행렬은 열지 않았다.
   - Tahoe-100M: 세포 단위 `obs_metadata.parquet` 100,648,790행을 (plate, well, 약물·농도, 세포주)로 집계.
   - DepMap 25Q3: `Model.csv`, `OmicsProfiles.csv`, `OmicsSomaticMutations.csv`.
   - Cellosaurus 57.0 (2026-10-22): 간 계열 후보 30개 accession의 질환·나이·파생 관계. 조회에는 accession만 보냈다.
3. **세포주 판정.** 플랫폼 주석이 간이거나 DepMap 계열이 Liver인 세포주를 후보로 삼았다. 이름은 Cellosaurus accession 또는 DepMap `StrippedCellLineName`과 정확히 일치할 때만 연결했다(유사 일치 없음). 성인 HCC는 DepMap(Hepatocellular Carcinoma, Adult)과 Cellosaurus(Adult hepatocellular carcinoma)가 모두 그렇게 말할 때만 인정했다. 계열(`family_id`)은 Cellosaurus 조상·동일인 관계와 DepMap PatientID를 묶어 만들었다.
4. **조건·대조군·반복.** L1000은 detection plate 하나를 실험 block 하나로 보고, 같은 plate에 vehicle well이 있을 때만 대조군이 연결된 것으로 셌다. 농도는 metadata에 적힌 값 그대로 사용했다(가까운 농도로 묶지 않음). Tahoe는 처리 well 하나가 block이고 같은 plate의 `DMSO_TF`가 대조군이다.
5. **Tahoe 엄격 LOCO.** 대상 세포주가 들어 있는 처리 well을 모든 source에서 빼고도 같은 약물·같은 농도가 다른 well에 남는지 셌다.
6. **실행 기록.** job 23886은 GSE92742의 plate ID를 `rna_plate`로 읽어 LINCS2020과의 겹침이 0으로 잘못 나왔다. plate ID를 `inst_id` 앞부분으로 바꿔 고쳤다. job 23891은 그 검사를 release 전체에 걸어 다른 계열의 혼합 plate 때문에 멈췄다(간 세포주에는 해당 없음). 최종 수치는 job 23893의 것이다. 세포주·조건 수는 세 실행에서 같았고 달라진 것은 겹침 수와 추가한 표뿐이다.

## 3. 결과

### 3.1 세포주: 플랫폼마다 성인 HCC 계열이 몇 개인가

| 플랫폼 (release) | 간 계열 후보 | 성인 HCC로 인정 | 제외·보류 |
|---|---|---|---|
| L1000 GSE92742 | HEPG2, HUH7, PHH | **Huh-7 (1계열)** | HEPG2: 간모세포종(DepMap·Cellosaurus 모두). PHH: 일차 간세포, 세포주 ID 없음 |
| L1000 GSE70138 | HEPG2, HUH7, PHH (cell_info는 Phase I과 같은 파일) | **0계열** — HUH7에는 약물 처리 instance가 없다 | 위와 같음. 약물 자료는 HEPG2뿐 |
| L1000 LINCS2020 | HEPG2, HUH7, HUH751, JHH5, JHH7, PHH | **Huh-7, JHH-5, JHH-7 (3계열)** | HUH751(Huh-7.5.1)은 Huh-7의 파생 clone이라 Huh-7 계열로만 셈. HEPG2 제외 |
| Tahoe-100M (50개 세포주) | SNU-423, HepG2/C3A | **SNU-423 (1계열)** | HepG2/C3A: 간모세포종 유래 HepG2의 clone |

- 두 출처가 엇갈려 보류된 세포주는 이 네 자료에 없었다. 따라서 보류 세포주를 포함한 민감도 수치도 위와 같다.
- Tahoe의 50개 세포주 중 49개가 DepMap에 연결됐고 Liver 계열은 2개다(`tahoe_all_lines_lineage.tsv`). 연결되지 않은 1개는 hTERT-HPNE(췌장 유래 불멸화 세포)다.
- HepG2/C3A를 넣어도 계열 수는 GSE92742 2, LINCS2020 4, Tahoe 2다. 계획서 규칙상 넣지 않았고, 넣더라도 어떤 release도 8계열에 이르지 못한다.

### 3.2 조건: 가장 유리한 단일 농도·시간에서의 교집합

대조군이 연결된 단일 저분자(trt_cp) 조건만 셌다. "선택 조건"은 정책의 순서(계열 수 → 3계열 이상 약물 수 → block 수, 동률이면 짧은 시간·낮은 농도)로 고른 것이다.

| release | 선택된 농도·시간 | 성인 HCC 계열 | 약물 수 (≥1 / ≥2 / ≥3 계열) | block 수 | 비고 |
|---|---|---:|---:|---:|---|
| GSE92742 | 0.1563 µM · 6 h | 1 | 1 / 0 / 0 | 3 | 계열이 하나뿐이라 순위는 의미가 없다. 약물이 가장 많은 조건은 10 µM · 6 h의 68개다 |
| GSE70138 | — | 0 | 0 | 0 | 성인 HCC 약물 자료 없음 |
| LINCS2020 | 10 µM · 24 h | 3 | 747 / 103 / **5** | 11 | Level 5 metadata 기준. Level 3는 서버에 없음 |
| Tahoe-100M | 5 µM (시간은 metadata에 없음, dataset card에는 24 h) | 1 | 379 / 0 / 0 | 490 | 세 농도 모두 계열 1개 |

LINCS2020에서 계열이 3개인 (농도, 시간) 쌍은 2,260개 중 4개뿐이다. 플랫폼이 제공하는 `nearest_dose`로 묶어 보면 10 µM · 24 h에서 3계열 공통 약물이 5개에서 6개가 된다. 이것은 참고 수치이며 판정에 쓰지 않았다(`lincs2020_nearest_dose_diagnostic.tsv`).

### 3.3 LOCO 지원: 한 계열을 보류했을 때 source에 같은 약물이 남는가

`loco_support_at_selected_dose_time.tsv`, LINCS2020 10 µM · 24 h:

| 보류 계열 | source 계열 수 | 대상 약물 수 | source ≥1계열에 있는 약물 | ≥2계열 | ≥3계열 | 대상에서 block ≥2인 약물 |
|---|---:|---:|---:|---:|---:|---:|
| Huh-7 (+Huh-7.5.1) | 2 | 73 | 8 | 5 | 0 | 58 |
| JHH-5 | 2 | 679 | 103 | 5 | 0 | 3 |
| JHH-7 | 2 | 103 | 100 | 5 | 0 | 0 |

- source 계열이 2개뿐이라 "각 약물이 source 3개 계열 이상"은 어느 fold에서도 0개다.
- inner 검증에 필요한 3개 group fold도 source 2계열로는 만들 수 없다.
- Huh-7을 보류하면 source에 같은 약물이 있는 조건이 8개뿐이다.

### 3.4 대조군과 반복

| release · 세포주 | plate 수 | vehicle 없는 plate | 조건 중 block ≥2 | 조건 중 서로 다른 plate 설계 ≥2 |
|---|---:|---:|---:|---:|
| GSE92742 · HUH7 | 6 (CVD001, 6 h·24 h 각 X1–X3) | 0 (plate당 vehicle well 중앙값 31.5) | 521 / 527 | 0 |
| LINCS2020 · HUH7 | 12 | 0 | 1,147 / 1,170 | 3 |
| LINCS2020 · JHH5 | 6 (모두 X1 한 장씩) | 0 | 3 / 2,140 | 3 |
| LINCS2020 · JHH7 | 1 | 0 | 0 / 353 | 0 |
| LINCS2020 · HUH751 | 1 | 0 | 0 / 29 | 0 |
| Tahoe · SNU-423 | 14 plate | 0 | 162 / 1,137 | 156 |

- LINCS2020의 대조군 연결은 "같은 detection plate를 가리키는 vehicle signature가 있다"는 수준으로만 확인했다. instance 파일이 서버에 없어 well 수는 세지 못했다.
- JHH-5와 JHH-7은 거의 모든 조건이 plate 한 장에서 한 번 측정됐다. 독립 반복으로 신호를 확인할 수 없다.
- Huh-7의 반복은 같은 plate 설계의 replicate plate 세 장이다. 서로 다른 설계에서 반복된 조건은 3개다.

### 3.5 release 사이의 중복

- GSE92742와 GSE70138은 detection plate를 공유하지 않는다(0장).
- LINCS2020의 HUH7 plate 12장 중 6장은 GSE92742의 CVD001 plate와 같은 plate다. 나머지 6장(LIVB001/002, 72 h)만 새 자료다. 두 release를 독립 반복으로 합치면 같은 측정을 두 번 세게 된다.
- LINCS2020의 HEPG2 plate 146장 중 53장이 GSE92742와, 7장이 GSE70138과 같다.

### 3.6 Level 3와 Level 5

- GSE92742와 GSE70138의 Level 3 행렬에는 감사한 간 세포주 instance가 모두 열로 들어 있다(누락 0, 각각 74,866개와 5,604개). landmark 유전자는 두 release 모두 978개다.
- 성인 HCC가 3계열인 유일한 release(LINCS2020)는 서버에 Level 5만 있다. 계획서 §4.2에 따라 이것을 Level 3 과제처럼 실행하지 않았다. 상태는 `CONTRACT_CHANGE_REQUIRED`다. Level 3를 받으려면 수십 GB의 새 내려받기가 필요하며 자동으로 하지 않았다.

### 3.7 Tahoe 엄격 LOCO

`tahoe_strict_loco.tsv`:

| 농도 | 대상 | 대상 조건 수 | 대상이 없는 well에 같은 약물·농도가 남는 조건 |
|---|---|---:|---:|
| 0.05 µM | SNU-423 | 379 | **0** |
| 0.5 µM | SNU-423 | 379 | **0** |
| 5 µM | SNU-423 | 379 | **0** |

- 1,137개 조건 중 975개는 well이 하나뿐이다. well이 여러 개인 162개 조건에서도 모든 well에 SNU-423 세포가 있다. 처리 well 1,316개의 세포주 수 중앙값은 50(전체)이다.
- 따라서 Tahoe는 `STRICT_LOCO_INELIGIBLE`이다. 성인 HCC 계열도 1개라 source가 될 다른 HCC 계열이 없다.

### 3.8 DepMap genotype 연결

성인 HCC로 인정된 네 계열은 모두 DepMap 모델 ID가 있고 WES·WGS profile과 MAF 행이 있다. 8개 후보 유전자의 상태(`genotype_support.tsv`):

| 세포주 (플랫폼) | TP53 | AXIN1 | ARID1A | CTNNB1 · RB1 · CDKN2A · PTEN · PIK3CA |
|---|---|---|---|---|
| Huh-7 (L1000) | LoF (p.Y220C) | 보고된 변이 없음 | 보고된 변이 없음 | 보고된 변이 없음 |
| JHH-5 (LINCS2020) | 미분류 coding 변이 (p.P190_H193del) | LoF (frameshift) | 보고된 변이 없음 | 보고된 변이 없음 |
| JHH-7 (LINCS2020) | GoF 표시 (p.R267P) | 보고된 변이 없음 | 보고된 변이 없음 | 보고된 변이 없음 |
| SNU-423 (Tahoe) | LoF (splice acceptor) | LoF (frameshift) | LoF (stop gained) | 보고된 변이 없음 |

- "보고된 변이 없음"은 `no_qualifying_variant_reported_with_assay_available` 상태이며 WT로 채우지 않았다. 이 MAF는 coding 변이 호출 결과라서 큰 결실이나 promoter 변이처럼 호출되지 않는 변화가 없다는 근거가 되지 않는다.
- LoF/GoF는 DepMap의 `LikelyLoF`, `TumorSuppressorHighImpact`, `OncogeneHighImpact`, `Hotspot` 표시만 따랐다. 기능 실험으로 확인한 것이 아니다.
- 계열이 3개인 LINCS2020 안에서 보면 8개 유전자 중 6개는 세 계열이 모두 같은 상태("보고된 변이 없음")다. 나머지 2개에서 AXIN1 LoF는 JHH-5에만 있고, TP53은 세 계열이 서로 다른 상태다. 즉 변이 상태마다 계열이 하나씩이며, 계획서의 "주요 genotype 상태가 사실상 한 계열에만 존재" 조건에 해당한다.

### 3.9 게이트 판정

| 경로 | 독립 성인 HCC 계열 | 판정 | 이유 |
|---|---:|---|---|
| L1000 GSE92742 Level 3 | 1 | `DATA_FAIL` | 계열 3개 미만. LOCO 불가능 |
| L1000 GSE70138 Level 3 | 0 | `DATA_FAIL` | 성인 HCC 약물 자료 없음 |
| L1000 LINCS2020 | 3 | `CONTRACT_CHANGE_REQUIRED`, 실행 불가 | Level 5만 있음. Level 3가 있어도 계열 3개로 `PILOT_ONLY` 상한이고, source ≥3계열 약물 0개, inner 3-fold 불가, 두 계열은 반복 없음, genotype 상태가 계열당 하나 |
| Tahoe-100M | 1 | `DATA_FAIL` | 계열 3개 미만, 엄격 LOCO 적격 조건 0개 |

**종합: `STOP_DATA_INSUFFICIENT`.**

## 4. 가능한 해석

- 관측된 사실은 "서버에 있는 L1000 두 GEO release와 Tahoe-100M에는 성인 HCC 계열이 각각 1개 이하이고, CMap 2020에는 3개"라는 것이다.
- 이것은 데이터 적격성 실패다. genotype이나 FiLM이 반응 예측에 도움이 되는지에 대해서는 아무것도 말해 주지 않는다. 가설은 시험되지 않았다.
- LINCS2020의 세 계열이 적은 이유(예: HCC 세포주가 특정 프로젝트 plate에만 들어갔다는 것)는 추정이며 확인하지 않았다.

## 5. 미검증·미실행

- LINCS2020 Level 3와 instance metadata: 서버에 없어 well 단위 대조군 수와 실제 값의 사용 가능 여부를 확인하지 못했다.
- 서버에 없는 다른 공개 자료(다른 L1000 release, 다른 약물 반응 RNA 자료)는 조사하지 않았다. 목표가 "L1000 한 release와 기존 Tahoe"로 한정돼 있기 때문이다.
- CNV, PROGENy 경로의 landmark 포함률, 약물 구조 파싱: 계열 수 게이트에서 멈췄으므로 하지 않았다.
- Phase 1–4 전체: R0–R3 학습, zero/global/same-drug/nearest-source/ridge 기준선, nested LOCO, genotype 교환 대조는 실행하지 않았다.
- 공개 모델 비교: `external_benchmark_not_run`. TranSiGen의 적격성 점검도 하지 않았다.
- 필수 검사 6개 중 이번 단계에 해당하는 것만 했다: 세포주 이름 → Cellosaurus → DepMap → 계열 ID의 연결(정확 일치만), 감사한 instance가 Level 3 행렬 열에 모두 있는지, plate ID가 release 사이에서 같은 plate를 가리키는지. 나머지(기준선 fixture, gradient, 공통 초기값 등)는 모델 단계의 검사라 대상이 없다.

## 6. 종료·진행 판단

**이 구성은 여기서 종료한다.** 목표의 6항에 따라 감사 결과와 실패 이유만 Git에 남겼다. pan-cancer 학습, 여러 플랫폼 통합, 환자 자료 전이, HepG2 포함, Level 5 과제로의 전환은 하지 않았다. 이들은 각각 연구 범위나 계약을 바꾸는 결정이므로 사용자가 정할 일이다.

## 7. 산출물과 hash

경로는 저장소 기준이다. 원자료의 서버 경로·크기·SHA-256은 `source_inventory.tsv`에 있다(Level 3·Level 5 행렬은 크기만 기록, hash 미계산). 각 표의 전체 SHA-256은 `phase0_summary.json`의 `outputs`에 있다.

| 파일 (`manifests/genotype_film_v3/`) | 내용 | 행 수 | SHA-256 앞 16자 |
|---|---|---:|---|
| `audit_policy.yaml` | 사전 고정 규칙과 수정 이력 | — | `24abf86cec7dad08` |
| `source_inventory.tsv` | 읽은 원자료 20개의 경로·크기·hash | 20 | `ab90fb0ff401477b` |
| `cellosaurus_lookup.tsv` | Cellosaurus 57.0 조회 결과 | 30 | `0e1e571ec09b76ec` |
| `cell_registry.tsv` | 플랫폼별 간 계열 후보와 판정 | 13 | `81704c2d3d44de1d` |
| `cell_condition_summary.tsv` | 세포주별 조건·약물 수 | 11 | `ebe268f6cab18a3d` |
| `condition_inventory.tsv` | (세포주, 약물, 농도, 시간) 조건 전체 | 31,671 | `866ad0ee1db98110` |
| `control_links.tsv` | plate별 처리 well·vehicle well 수 | 289 | `94a2e1ec2f331f62` |
| `replicate_registry.tsv` | 성인 HCC 조건의 block ID | 5,356 | `b4b231d2b488f7da` |
| `genotype_support.tsv` | 모델 × 8개 유전자 상태 | 48 | `ea338cf2db601af5` |
| `coverage_by_dose_time.tsv` | (농도, 시간)별 계열·약물·block 수 | 2,578 | `7cb2b099ff830732` |
| `loco_support_at_selected_dose_time.tsv` | 보류 계열별 source 지원 | 5 | `b4fc4f1ef00748c7` |
| `tahoe_strict_loco.tsv` | Tahoe 엄격 LOCO 결과 | 3 | `c5cce9a5aed3835f` |
| `tahoe_all_lines_lineage.tsv` | Tahoe 50개 세포주의 DepMap 계열 | 50 | `f9d917c6dd54331f` |
| `lincs2020_nearest_dose_diagnostic.tsv` | 참고용 nearest-dose 집계 (판정 미사용) | 56 | `fdd11dc0c20cac76` |
| `phase0_summary.json` | 검사 수치와 위 표들의 전체 hash | — | — |

코드: `src/genotype_film_v3/audit_metadata.py`, `src/genotype_film_v3/fetch_cellosaurus.py`, `experiments/genotype_film_v3/phase0_audit.sbatch`. 실행 로그: `reports/genotype_film_v3/logs/gf_p0_audit_23893.out`.

재실행: 서버 작업트리(`/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_genotype_film_v3`)에서 `sbatch experiments/genotype_film_v3/phase0_audit.sbatch`. CPU 4개로 약 1분 30초 걸린다.
