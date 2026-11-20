# 간암 Virtual Cell 모델링 연구 재기획서
## 기존 작업 아카이브 → 논문 정량 재현 → 동일 조건 튜닝 → 간암 관련 실측 데이터 검증

- 작성일: **2026-11-05**
- 작업 저장소: `https://github.com/honbul/vcell_liver`
- 권장 신규 브랜치: `research/reproduction-first-20261105`
- 문서 상태: **실행 기획서. 이 문서 작성만으로 아카이브·학습·평가를 실행한 것은 아니다.**
- 사용자 결정: 기존 튜토리얼 중심 작업을 현재 연구에서 폐기하고, 삭제 대신 아카이브로 보존한다.
- 핵심 실행 원칙: **논문에서 무엇을 재현할 것인지 먼저 고정하고, 그 기준점 위에서 모델을 변경한다.**

---

## 0. 이번에 무엇을 새로 연구하는가?

### 연구 주제

**공개 유전자 섭동 반응 예측 모델의 정량 재현과 적응 학습: GEARS·scGPT를 기준으로 한 모델 중심·데이터 중심 개선 및 간암 관련 적용 가능성 검증**

### 예측할 대상

> 대조군의 전사체와 유전자 개입 정보를 받아, 학습에서 반응을 보지 못한 개입 조건의 처리 후 전사체와 발현 변화를 예측한다.

```text
입력
  대조군 세포의 RNA 발현 + 개입 유전자/조합 + 알려진 실험 맥락
                         ↓
                 GEARS / scGPT
                         ↓
출력
  처리 후 유전자별 발현 또는 대조군 대비 변화
                         ↓
정답
  실제 개입 실험에서 측정한 처리군 RNA 발현
```

**이번 연구의 중심은 B세포·T세포·간세포 등 큰 세포 계열 분류가 아니다.** 분류 F1, 임베딩 생성, 유전자 토큰 삭제, 사용자 노트북 실행 성공을 섭동 반응 예측의 완료 결과로 대체하지 않는다.

### 연구 질문

| 질문 | 확인할 것 | 답을 얻는 실험 |
|---|---|---|
| Q1. 기존 논문 결과를 실제로 재현할 수 있는가? | 같은 데이터·분할·모델·평가 정의에서의 정량 일치와 차이 | R1·R2 원 논문 재현 |
| Q2. 같은 조건에서 모델을 적응시키면 개선되는가? | 원래 학습법 대비 decoder-only, 부분 미세조정, LoRA의 효과와 비용 | M 모델 중심 실험 |
| Q3. 학습 데이터의 제공 방식을 바꾸면 개선되는가? | 대조군 샘플링과 섭동별 학습 표본 구성의 효과 | D 데이터 중심 실험 |
| Q4. 개선은 두 변경의 결합에서도 유지되는가? | 데이터 효과·모델 효과·상호작용의 분리 | F 확인실험 |
| Q5. 간 관련 실측 개입 데이터에서도 적용되는가? | 동일 세포주 내 미관측 개입 예측의 이득, 실제 HCC 근거의 확보 여부 | H 간 관련 확장 |

**설계 선택:** 첫 재현 대상은 GEARS와 scGPT의 **perturbation prediction**으로 제한한다. Geneformer·scFoundation·LiverCT·scVIDR를 동시에 추가하는 넓은 모델 비교는 이번 필수 범위가 아니다. 이 선택은 사용자에게서 받은 간암 Virtual Cell 목표를 위한 새 기획이며, 기존 논문이 이 순서를 의무화한다는 뜻은 아니다.

---

## 1. 기존 작업의 폐기와 아카이브 범위

### 1.1 폐기의 의미

기존 작업을 삭제하거나 연구 기록을 숨기는 것이 아니라, **새 연구의 코드·기준선·성과·진행 목표에서 제외한다.**

| 기존 항목 | 처리 | 새 연구에서 허용되는 사용 |
|---|---|---|
| 기존 연구 계획, AGENTS 지침, README의 진행 목표 | 아카이브, 활성 지침 교체 | 과거 결정의 이력 확인만 |
| 자체 실험 코드·설정·스케줄러 스크립트 | 아카이브 | 새 실험의 실행 의존성으로 사용하지 않음 |
| 튜토리얼 00–10, 기존 설명 자료 | 아카이브 | 새 논문 재현 튜토리얼로 명칭만 변경하지 않음 |
| 기존 지표·학습 로그·예측·fine-tuned weights | 아카이브 | 새 baseline이나 새 실험 결과로 재사용하지 않음 |
| 공개 원본 데이터 및 저자 배포 사전학습 가중치 | 출처·버전·해시를 확인하여 읽기 전용 자원으로 보존 가능 | 같은 공개 bytes를 다시 다운로드하지 않기 위한 캐시로만 사용 |
| 기존 Python 환경 | 보존하되 비활성화 | 새 연구는 별도 환경 생성. 기존 환경을 수정하거나 실행 기준으로 상속하지 않음 |
| 이미 열람한 평가 결과에 대한 이력 | 짧은 exposure ledger로 보존 | 과거 관찰을 없던 일로 취급하지 않음 |

**공개 원본 데이터·저자 가중치의 캐시 재사용과, 이전 연구가 학습한 모델·예측의 재사용은 다르다.** 전자만 허용한다. 이전 자체 생성 embedding, 자체 전처리 결과, 선정된 hyperparameter는 새 기준선에 자동 승계하지 않는다.

### 1.2 에이전트가 실행할 아카이브 절차

1. `git status`, 현재 branch/HEAD, remote branch, 미커밋 변경, 프로젝트 전용 실행 작업을 조사한다. 사용자·다른 연구자의 변경을 덮어쓰지 않는다.
2. 기존 `main`과 연구 브랜치의 현재 HEAD를 각각 불변 archive ref/tag로 보존한다. 다음 값은 **본 기획서 작성 시 확인한 스냅샷**이며 실행 시 다시 확인한다. [G1]
   - `main`: `243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79`
   - `research/phase2-v2`: `4077a263fc0575402d1e9844d6308a1336bcda55`
3. 미커밋 코드·문서도 안전한 스냅샷으로 보존한다. 비밀값·비공개 원문은 public Git에 올리지 않는다.
4. 새 branch와 가능하면 새 worktree를 만든다. **현재 GitHub 저장소 자체를 archive/read-only 상태로 만들지 않는다.** 같은 저장소에서 새 연구를 계속해야 한다.
5. 새 branch의 이전 코드·설정·문서·노트북·보고서는 `archive/legacy_20261105/` 아래로 이동하거나, 동등한 archive snapshot과 경로 인덱스로 격리한다. `archive/README.md`에 복구 ref, 원 경로, 새 위치를 남긴다.
6. 기존 서버의 대용량 자산은 함부로 물리 이동하지 않는다. 공유 데이터, symlink 대상, 가상환경은 이동하면 다른 작업이나 절대경로가 깨질 수 있다. 읽기 전용 archive manifest로 논리 격리하고 새 출력 위치를 별도로 만든다.
7. 기존 프로젝트 전용 작업이 실행 중이면 checkpoint와 로그를 보존한 뒤 정리한다. 다른 사용자의 작업, 범용 서비스, 공유 데이터 처리 작업은 종료하지 않는다.
8. 활성 `README.md`, `AGENTS.md`, 실행 진입점은 새 기획서에 맞춰 교체한다. 기존 `.github` 자동화나 agent 지침이 옛 실험을 다시 실행하지 않게 정리한다.
9. archive ref와 manifest가 원격에 보존됐는지 확인한 뒤 새 연구를 시작한다. 권한이 없으면 로컬 보존·실패 내용을 보고하되, 허용되는 로컬 연구 작업은 진행한다.

**금지:** `git reset --hard`, `git clean -fdx`, 강제 push, 기존 branch 삭제, `rm -rf`로 연구 산출물 삭제, 비밀값을 포함한 일괄 업로드. Git 이력 제거 또는 저장소 전체 초기화도 하지 않는다.

아카이브는 연구 수행의 첫 작업이다. 별도의 대규모 보안 감사나 모든 대용량 파일의 재해싱을 선행 과제로 확대하지 않는다. 기존 검증 해시가 있으면 필요한 자산에서만 재확인한다.

### 1.3 아카이브해도 과거의 평가 관찰은 지워지지 않는다

이전 HCC 환자·외부 cohort·HepG2 및 공개 벤치마크의 결과를 이미 확인했다. 기록을 archive로 옮겨도 해당 데이터가 새 blind test가 되지는 않는다.

그러나 **공개 벤치마크는 원래 알려진 결과를 재현하는 것이 목적**이므로 재현에 다시 사용할 수 있다. 이를 데이터 사용 금지로 해석하지 않는다.

새 연구에서는 다음을 구분한다.

- `public_reproduction`: 알려진 공개 점수를 재현하는 데이터.
- `development_previously_exposed`: 과거 결과를 본 데이터에서 방법 개발·진단.
- `heldout_under_new_protocol`: 새 프로토콜에서 학습에는 제외됐으나, 연구 차원에서 완전히 처음 보는 데이터라고 보장할 수 없는 평가.
- `independent_confirmatory`: 선택·튜닝에 사용하지 않았음을 확인한 별도 연구/생물학적 반복의 확인 평가.

기존 결과를 읽어 만든 가설을 새로운 관측에서 처음 발견한 것처럼 쓰지 않는다. 그렇다고 평가 이력 관리 때문에 원 논문 재현이나 개발 실험을 중단하지도 않는다.

---

## 2. 재현할 논문과 모델을 먼저 고정한다

| 대상 | 원 연구의 역할 | 이번에 재현할 부분 | 제외할 부분 |
|---|---|---|---|
| **GEARS** — Roohani et al., Nature Biotechnology, 온라인 2023/권호 2024 | 유전자 관계 그래프를 이용한 단일·다중 유전자 섭동 반응 예측 | **Norman 데이터의 미관측 두 유전자 조합 평가**, 원 논문 Fig. 2e–f와 대응 분석 코드 | 모든 Figure 재현, 새 질환 플랫폼 구축 |
| **scGPT** — Cui et al., Nature Methods, 2024 | 사전학습 Transformer의 downstream 전이 | **Fig. 3a의 perturbation prediction 중 Norman 조건**과 연결되는 저자 학습·평가 경로 | MS 세포 유형 분류를 이번 핵심 재현으로 대체, 전체 사전학습 반복 |

GEARS는 논문 그림 재현용 별도 저장소 `yhr91/GEARS_misc`를 공개하고 있다. 단순 README 예제를 실행하는 데서 끝내지 말고 해당 그림에 연결된 데이터·split·분석 스크립트를 확인한다. [S1–S3]

scGPT의 공개 perturbation tutorial은 `TransformerGenerator`를 이용한 발현 예측 경로다. 확인한 문서의 기본 `data_name`은 **Adamson**이다. **데이터명만 Norman으로 바꿨다고 원 논문의 Norman 수치를 재현한 것으로 처리하면 안 된다.** Fig. 3a의 데이터 조건·checkpoint·split·집계를 원전과 대조한다. [S4–S6]

### 아직 확정하지 않은 것

본 기획서는 원 논문의 모든 수치 원자료와 실행 당시의 전체 환경까지 확보한 재현 결과가 아니다. scGPT의 일부 본문 접근은 제한됐고, 공개 서지·데이터/코드 안내·Fig. 3 설명·공식 tutorial을 확인했다. **정확한 목표 수치, 논문용 checkpoint, 모든 seed와 source-data 파일은 실행 초기의 reproduction contract에서 확정한다.**

우선순위는 `논문 source data → 저자의 figure 재현 코드/릴리스 → 저자의 저장된 출력 → 그림에서 읽은 근삿값`이다. 마지막 경우는 digitized approximation으로 표시하며 소수점 정밀 수치를 만들어 넣지 않는다.

Norman의 조합 평가는 다음처럼 구분한다. 저자 코드의 `seen`/`unseen` 이름이 반대로 표현될 수 있으므로 label 문자열이 아니라 실제 train 개입 목록으로 범주를 확인한다.

| 범주 | train에 있는 단일 유전자 반응 | test가 요구하는 것 |
|---|---|---|
| 두 유전자 모두 관측됨 | A와 B의 단독 반응이 train에 있음 | 학습에 없는 A+B 조합의 반응 |
| 한 유전자만 관측됨 | A 또는 B만 단독 반응이 있음 | 새 유전자 성분을 포함한 조합의 반응 |
| 두 유전자 모두 미관측 | A와 B의 단독 반응이 train에 없음 | 두 미관측 성분을 포함한 조합의 반응 |

모든 범주에서 정확한 A+B의 처리 세포는 test이면 train에 넣지 않는다. 같은 원 연구도 가공본에 따라 남는 조건 수가 달라질 수 있으므로 특정 총 개수를 임의로 고정하지 않는다.

Norman의 scGPT 논문용 자산을 확보하지 못하면, GEARS 재현은 계속 진행하고 scGPT는 공식 Adamson 조건으로 **공식 예제 재현**을 수행할 수 있다. 다만 이는 Norman 논문 결과 재현의 완료를 대신하지 않는다. 임의의 새 모델로 교체하지 않는다.

---

## 3. 데이터: 무엇이고, 왜 선택하는가?

| 데이터 | 내용 | 선택 이유 | 이번 역할·주의 |
|---|---|---|---|
| **Norman et al. / GSE133344** | K562에서 단일·두 유전자 조합을 활성화한 Perturb-seq | GEARS·scGPT 원 논문과 연결되고, 미관측 조합의 실제 RNA 정답이 있음 | 주 재현·튜닝 데이터. **간암 데이터가 아님**. 저자별 가공본의 세포·유전자·조건 수가 같다고 가정하지 않음 |
| **Adamson / GSE90546** | K562 유전자 억제 반응 데이터 | scGPT 공식 perturbation tutorial의 실제 기본 경로 | scGPT 원 경로 점검 또는 사전 선언한 보조 재현. Norman과 점수 합산 금지 |
| **GSE264667 HepG2** | 실제 유전자 개입 뒤 단일세포 RNA를 측정한 간 유래 세포주 자료 | 논문 재현에서 확보한 반응 예측 방법을 간 유래 실측 자료에 적용 가능 | 선택적 간 유래 확장. 과거 사용 데이터이므로 새 blind test 아님. HepG2는 성인 HCC 자체가 아닌 hepatoblastoma 유래 [S9–S10] |
| **새 사람 HCC 개입 자료** | 실제 HCC 세포주·환자 유래·오가노이드 등의 개입 전후 측정 | 간암이라는 최종 생물학적 맥락을 검증하기 위해 필요 | accession·실제 파일·control·개입·반복을 확인한 뒤 1개 선정. 이 기획서에서는 확보됐다고 가정하지 않음 |
| **GSE149614·GSE151530** | 환자 조직의 관측 scRNA-seq | 조직 상태를 연구할 수 있으나 이번 질문의 개입 후 정답은 아님 | 이전 작업과 함께 archive. 이번 핵심 실험의 6계열 분류 데이터로 다시 사용하지 않음 |

**세포 유형 라벨을 유전자·약물 개입 결과로 바꾸지 않는다.** Tumor/Normal 차이를 동일 세포의 개입 전후 변화로 취급하지 않는다. 유전자 하나의 count를 0으로 바꾼 것을 실제 CRISPR 결과로 사용하지 않는다.

### 실제 HCC 확장 자료의 최소 조건

다음 정보가 확인되는 자료를 우선한다.

1. 개입 종류와 표적이 확인되고, matched control과 처리군 RNA가 실제로 제공된다.
2. 처리 시간·용량(해당 시)·세포주/환자·배치가 해소된다.
3. 학습·검증·시험으로 나눌 개입 조건 수와 세포 수가 있다. 단일 약물의 한 비교만 있다면 미관측 약물 일반화 benchmark로 만들지 않는다.
4. 독립적인 생물학적 반복이 있으면 그 단위로 나눈다. lane, GEM group, library batch를 생물학적 반복이라고 바꾸어 적지 않는다.
5. 공개 권한과 실제 count/metadata 접근을 확인한다. 논문 제목에 HCC가 있다는 것만으로 채택하지 않는다.

유전자 섭동으로 재현한 모델에 약물 이름을 유전자 ID처럼 넣지 않는다. HCC에서 약물 데이터만 확보되면 **별도 chemical task 계약이 필요한 후속 단계**로 기록한다. 첫 연구를 자동으로 약물 모델 전체 비교로 확대하지 않는다.

---

## 4. ‘논문 재현’을 어떻게 판정할 것인가?

### 4.1 실행 전에 작성할 reproduction contract

모델별로 `protocols/reproduction/gears_norman.yaml`, `scgpt_norman.yaml`을 작성한다. 아래는 구현할 필드이며 이미 존재하는 파일이 아니다.

| 필드 | 반드시 적을 내용 |
|---|---|
| `claim` | 원 논문의 어떤 주장·그림·수치를 재현하는지 |
| `target_panel` | GEARS Fig. 2e–f / scGPT Fig. 3a의 해당 데이터 조건 |
| `reference_value` | 원 수치·오차막대·반복 수. 확보하지 못했으면 `unavailable` |
| `reference_source` | source-data 파일, table cell, 코드 위치 등 정확한 출처 |
| `data` | accession, 가공본 URL·버전·해시, 세포/유전자/개입 수 |
| `split` | train/validation/test 조건 목록과 생성 seed. 저장된 split 우선 |
| `model` | upstream commit/릴리스, 실제 architecture, checkpoint 해시 |
| `training` | loss, optimizer, scheduler, batch, epochs, early stopping, 정밀도 |
| `selection` | checkpoint 선택에 쓴 validation 지표 |
| `evaluation` | control reference, DE 정의, 유전자 집합, 셀/개입별 집계, 정규화 |
| `repetitions` | 원 연구의 split seed와 학습 seed를 구분 |
| `compatibility_changes` | 환경 문제로 바꾼 모든 부분. 의미 변경 여부 |
| `comparison_rule` | 결과를 보기 전에 정한 수치 비교·불일치 처리 규칙 |

**원 논문과 공식 tutorial의 설정이 충돌하면 숨기지 않는다.** `paper_condition`과 `tutorial_condition`을 분리한다. 논문 수치를 사용하면서 실제로는 다른 tutorial split을 실행하는 조합을 금지한다.

### 4.2 상태를 섞지 않는다

| 상태 | 의미 |
|---|---|
| `smoke_passed` | 모델이 로드되고 몇 batch가 실행됨. 성능 재현 아님 |
| `official_protocol_executed` | 지정한 공식 학습·예측·평가 경로를 실행함 |
| `numerically_consistent` | 사전에 고정한 비교 규칙상 원 결과와 정량적으로 일관됨 |
| `numerically_discrepant` | 비교 가능한 원 조건인데 의미 있는 차이가 남음 |
| `reference_unavailable` | 수치·split·원 자산 부족으로 정량 일치 판정 불가 |
| `modified_reproduction` | 알려진 환경·구현·정보 사용 차이가 있는 재현 |
| `new_benchmark` | 공동 비교를 위해 별도 설계한 새 평가 |

재현이 원 점수와 모든 소수점까지 같아야 한다는 뜻은 아니다. 반대로 임의로 “±5%면 성공”이라고 정하거나 CI가 겹친다는 이유만으로 동등성을 선언하지 않는다. 원 반복 설계를 따르고 효과 크기·분산·허용 가능한 구현 차이를 함께 보고한다. 실용적 동등성 한계가 필요하면 결과 전에 근거와 함께 고정한다.

### 4.3 원래 평가 정의를 먼저 복원한다

GEARS Fig. 2는 DE 유전자 오차, no-perturbation 대비 정규화, 일반화 범주 등을 사용하며 다섯 split으로 학습한 모델의 결과를 보고한다. [S1] 이를 새 전체 유전자 RMSE 한 개로 바꾼 뒤 원 논문과 일치한다고 하지 않는다.

scGPT Fig. 3a는 예측·실측 **발현 변화**의 Pearson correlation과 DE 유전자별 평가를 구분한다. [S5] 발현량 자체의 상관과 변화량 상관을 혼동하지 않는다.

원 코드에 held-out 정보의 부적절한 사용이 의심되면 실제로 그 값이 gradient, 모델 선택, 그래프 구축에 전달되는지 추적한다. **DE 순위가 평가에만 사용되는 것과 학습에 유입되는 것은 다르다.** 원본 동작을 재현하는 구간은 격리하여 명시하고, 엄격한 비교는 별도 구현·별도 이름으로 수행한다. test feedback이 학습 선택에 실제 사용된다면 이를 숨긴 채 재현 성공으로 처리하지 않는다.

### 4.4 불일치가 발생했을 때

`데이터/분할 → preprocessing → 가중치 로드 → loss/학습량 → backend/정밀도 → evaluator` 순서로 원인을 좁힌다. tensor key, shape, 실제 로드 비율과 예측 행 순서부터 확인한다.

한 모델의 동일 불일치에 대해 근거 없는 재시도를 무한히 하지 않는다. 첫 진단 뒤 **원인 가설이 다른 최대 두 차례의 추가 확인**을 수행하고, 해결되지 않으면 미해결 사유를 남긴다. 다른 모델의 재현은 진행할 수 있다.

정량 일치를 입증하지 못한 모델의 이후 실험은 `modified implementation 대비 개선`으로만 보고한다. **“원 논문보다 개선”이라는 주장은 해당 원 조건의 비교가 성립할 때만 한다.**

---

## 5. 전체 실험 순서와 산출물

| 단계 | 실험 | 데이터·모델 | 완료 시 답할 질문 | 핵심 산출물 |
|---|---|---|---|---|
| A0 | 기존 작업 아카이브 | 현재 저장소·서버 자산 | 무엇을 폐기하고 무엇을 보존했는가? | archive manifest·복구 ref·새 AGENTS |
| R1 | GEARS 논문 재현 | Norman·GEARS | 원 논문의 해당 조건 수치가 재현되는가? | reproduction contract·원 수치 대비표 |
| R2 | scGPT 논문 재현 | Norman·scGPT native response | 공개 가중치의 반응 예측 학습 결과가 재현되는가? | 동일 형식의 재현 보고서 |
| C0 | 공통 조건 기준선 | 두 모델·단순 기준선 | 입력·평가 조건을 맞추면 각 방법은 어느 수준인가? | common protocol·새 baseline |
| M | 모델 중심 튜닝 | scGPT 우선, GEARS 제한 tuning | 같은 데이터에서 무엇을 학습하는 것이 좋은가? | validation 선택표·최종 후보 |
| D | 데이터 중심 개선 | 같은 모델 고정 | 제공 표본/대조군 처리의 변경이 도움되는가? | 두 가설의 대조 결과 |
| F | 확인·결합 | 원본/데이터/모델/결합 | 개선이 반복되고 무엇의 효과인가? | 반복 결과·효과 분해·비용 |
| H | 간 관련 실측 확장 | 적격 개입 자료·고정된 방법 | 간 관련 맥락에서도 결론이 유지되는가? | 적용 보고서·HCC 범위 명시 |
| T | 재현 튜토리얼 | 완료된 실제 코드 | 다른 사용자가 같은 결과를 다시 만들 수 있는가? | 재현 명령·노트북·기대 결과표 |

R1·R2는 자원이 허용하면 병렬로 진행한다. **C0 이후의 튜닝은 재현 상태를 정리한 뒤** 시작한다. 정량 일치 미확인 모델을 미확인 상태로 유지한 채 탐색할 수는 있지만, 재현 실패를 감추기 위한 다른 benchmark 증설은 하지 않는다.

---

## 6. 공통 비교 C0: 원 논문 재현과 별도의 층

GEARS 원 논문 조건과 scGPT 원 논문 조건은 서로 다를 수 있다. 각각의 재현 점수를 그대로 한 표에서 순위화하지 않는다. 다음 단계에서 동일한 **새 common protocol**을 적용해 다시 baseline을 실행한다.

### 6.1 무엇을 맞추는가?

- 동일한 실제 cell ID, intervention ID, train/validation/test 역할.
- 동일한 정규화 스케일·주 출력 유전자 집합·평가 대상 개입 목록.
- 동일하게 제공되는 control pool, batch 조건, control ID manifest.
- 동일한 checkpoint 선택 원칙과 보고 지표.
- 가능한 한 같은 supervised training example budget. 모델별 update 수·노출 수·시간은 별도 기록.
- metadata로 공개된 target identity와 숨겨야 하는 treated outcome의 구분.

공통 출력 유전자 집합을 위해 특정 모델 vocabulary와의 교집합을 사용한다면 제외된 유전자·개입 수를 모두 공개한다. target token이 없거나 학습 시 gene sampling에서 빠진 경우, 개입 flag가 0인 채 진행하지 않는다. 임의 target token을 만들거나 다른 gene으로 매핑하지 않는다.

**모델별 가공본을 억지로 같다고 선언하지 않는다.** 원 논문 재현 데이터와 C0 공통 가공본은 서로 다른 버전으로 보존한다.

### 6.2 기준선

| 기준선 | 입력과 예측 | 사용 범위 |
|---|---|---|
| No-change | 제공된 control로 처리 후 발현을 예측 | 항상 유지 |
| Train-mean response | 학습 개입들의 평균 변화량을 control에 더함 | 항상 유지 |
| Additive response | 학습에서 관측된 단일 개입의 변화량을 합함 | 두 단일 개입이 실제 train에 있는 조합에서만. 미관측 단일 반응을 test에서 가져오지 않음 |
| 단순 선형/저차원 response 모델 | 공개 target prior와 train에서 학습한 response basis를 연결 | 강한 단순 기준선. 이전 자체 코드를 재사용하지 않고 새 protocol로 구현·fit |
| GEARS | 원 architecture를 공통 데이터 조건으로 fit | 새로운 공통 기준선 |
| scGPT | native expression decoder와 perturbation encoder를 포함한 원 학습 경로 | 새로운 공통 기준선 |

여기서 PCA는 **cell-type 분류용 PCA**가 아니라, 필요하면 학습 반응 벡터를 압축하는 계산 도구다. PCA만으로 유전자 개입 후 발현을 예측했다고 쓰지 않는다.

### 6.3 대조군 제공을 실제로 맞춘다

아래는 **새 설계 제안**이며 원 논문의 기본값이 아니다.

- 평가할 intervention/기술 배치마다 같은 고정 control pool을 제공한다.
- 기본 common 평가에서는 batch별 최대 **32개 control ID**를 비복원 추출하고, 개수가 적으면 모두 사용한다. 사용 수와 ID를 전 모델에서 고정한다.
- 신경망은 제공된 각 control에 대한 예측을 평균한다. 단순 기준선도 같은 control들의 평균을 사용한다.
- 처리군의 배치 구성으로 가중 평균을 만들면, 그 배치 구성을 모든 모델에 같은 허용 정보로 제공했다고 기록한다. 미래 개입의 구성이 알려졌다는 전향적 주장과 구분한다.
- `f(mean(control))`와 `mean(f(control))`는 비선형 모델에서 다른 계산이다. 둘을 변경하는 실험은 별도 ablation으로 표시한다.
- 기록된 control 평균에만 유리한 기준선을 만들거나, 일부 모델만 전체 control pool을 사용하지 않는다.

---

## 7. 모델 중심 실험 M: 무엇을 얼마나 학습할 것인가?

### 7.1 scGPT의 주 비교

**모든 조건의 출력은 처리 후 발현 벡터다.** 임베딩 뒤에 세포 계열 Logistic Regression을 붙이는 방식으로 대체하지 않는다.

| 조건 | 학습하는 부분 | 확인하는 질문 |
|---|---|---|
| M0 | 공통 조건에서 실행한 native response 학습 | 공식 학습 경로의 기준점 |
| M1 | pretrained backbone 동결 + native expression decoder·perturbation 입력 모듈 학습 | backbone을 업데이트하지 않아도 되는가? |
| M2 | 같은 decoder·perturbation 모듈 + 마지막 일부 Transformer block 학습 | 제한적인 backbone 적응이 유용한가? |
| M3 | 같은 decoder·perturbation 모듈 + 실제 attention projection의 LoRA | 작은 추가 파라미터로 성능·비용을 개선하는가? |
| M-random | 같은 architecture를 무작위 초기화하고 M0와 같은 방식으로 학습 | 사전학습 자체의 이득이 있는가? |

M0가 원래 full fine-tuning이면 그대로 기준선으로 사용한다. “새 기획이므로 반드시 전체 학습을 다시 설계”하는 것이 아니라 저자의 해당 task 학습법을 기준으로 한다.

LoRA는 모듈 이름을 확인해서 적용한다. GEARS GNN이나 fused QKV 구현에 BERT의 `query`/`value` 이름을 기계적으로 적용하지 않는다. 장착된 module 목록, 학습 파라미터 수, 실제 gradient/update, 저장·복원 후 예측을 작은 샘플에서 확인한다. trainable native decoder를 얼려 모델이 학습할 수 없는 대조군을 만들지 않는다.

### 7.2 탐색 예산

첫 screening은 고정된 train/validation 조건에서 한다. 다음 수치는 자원 폭주를 막기 위한 **제안값**이다.

- M1·M2·M3: 각 최대 2개 learning-rate 후보, 합계 6개 설정부터 시작.
- M2: 우선 마지막 2개 block. M3: 우선 rank 8. rank 16/추가 block은 validation 근거가 있을 때만 추가하여 총 신규 설정을 8개 이내로 제한.
- M-random: M0와 동일 학습 조건의 대조. 사전학습 모델만 유리하게 optimizer·데이터를 달리하지 않는다.
- GEARS: 먼저 원 기본 설정을 재현하고, 이후 learning rate·hidden size 등 최대 4개 명확한 설정만 비교. GEARS에 LoRA를 억지로 넣지 않는다.

학습 budget은 원 절차의 epoch를 우선하고, 조기종료는 validation만 사용한다. 데이터 수·batch·gradient accumulation을 바꿨다면 실제 update 및 example 수를 함께 기록한다. BatchNorm 등으로 microbatch 변경이 완전히 동등하지 않을 수 있으므로 “effective batch 같음=완전 동일”이라고 쓰지 않는다.

모델을 성능과 비용 두 축으로 비교한다. 높은 성능만을 고르는 후보와 같은 성능에서 비용을 줄이는 후보를 구분한다. 모든 모델·모든 설정을 모든 seed로 반복하지 말고 validation에서 선정한 후보만 확인한다.

---

## 8. 데이터 중심 실험 D: 모델은 고정한다

이번 첫 라운드는 합성 RNA 생성, 임의 세포 제거, 대규모 새 KG 구축으로 확대하지 않는다. 다음 두 가설만 검토한다.

| 조건 | 바꾸는 것 | 고정하는 것 | 해석 |
|---|---|---|---|
| D0 | 공통 기준선의 원 학습 표본·control 배정 | 기준 조건 | 비교점 |
| D1 | 같은 처리 세포에 연결할 matched control을 epoch마다 다시 추출 | 처리 세포 ID, 모델, 전체 training step, 평가 control ID | 특정 우연한 control pairing에 대한 의존이 줄어드는가? |
| D2 | 학습 중 intervention별 노출을 균등하게 만드는 sampling | 모델, 총 example/update budget, validation/test | 세포가 많은 개입에 치우친 학습을 줄이면 도움이 되는가? |

먼저 C0의 실제 control 배정 동작을 확인한다. 이미 epoch마다 재샘플링한다면 D1은 같은 작업을 다른 이름으로 반복하는 무효 ablation이므로 수행하지 않는다. 이 경우 한 처리 세포에 둘 이상의 독립 control을 제공하는 등 실제 차이가 있는 가설 하나로 사전 변경하고, 그 변경과 계산량을 명시한다.

D1은 진짜 같은 세포의 전후 pairing을 만든다는 뜻이 아니다. 서로 다른 control/treated 세포의 통계적 학습 조합이다. D2는 평가 분포를 균등하게 바꾸거나 모든 condition의 정답을 미리 참조하는 작업이 아니다.

동일 최적화 budget에서 screening하고, validation으로 D* 하나만 선정한다. 두 가설 모두 음성이면 그 결과를 보고하며, 억지로 개선된 것으로 표시하거나 새 가설을 끝없이 추가하지 않는다.

QC·guide 신뢰도 문제가 실제로 확인돼 D1/D2보다 중요하면 **train metadata만으로** 하나의 가설을 교체할 수 있다. 교체 이유·시점을 기록하고 test 결과를 본 뒤 유리한 표본을 제거하지 않는다. 합성 데이터는 이번 필수가 아니다.

---

## 9. 확인실험 F와 성능 개선 판정

### 9.1 기본 비교

validation에서 선택한 M*, D*로 다음 네 조건을 비교한다. 같은 시점의 새 연구 baseline을 쓰며 archive 결과를 대조군으로 끌어오지 않는다.

| 조건 | 데이터 | 모델 |
|---|---|---|
| D0M0 | 원 제공 방식 | 공통 기준 모델 |
| D*M0 | 선택된 데이터 변경 | 같은 기준 모델 |
| D0M* | 원 제공 방식 | 선택된 모델 변경 |
| D*M* | 선택된 데이터 변경 | 선택된 모델 변경 |

D 또는 M이 명백히 유망하지 않으면 조합 전체를 무조건 대규모 반복하지 않는다. 최소한 원본 대 선정 후보의 확인은 수행하고, 음성 가설은 screening 결과와 한계로 종료할 수 있다.

### 9.2 반복과 최종 test

- 선택은 train/validation에서 완료하고 후보·seed·주지표를 고정한 뒤 test를 평가한다.
- 기본 확인은 같은 split에서 학습 seed 3개를 사용한다. 이는 **학습 무작위성** 확인이며 새로운 독립 생물학 표본 3개가 아니다.
- 유망한 개선은 추가 split 2개에서 기준선과 후보만 확인한다. 공식 논문이 요구하는 반복 수가 별도로 있으면 재현 단계에서는 원 반복 수를 따른다.
- test를 여러 차례 보며 후보를 바꾸는 탐색은 `development`로 표시한다. 새 holdout이라고 부르지 않는다.

### 9.3 무엇을 개선이라고 보고할 것인가?

**주지표 제안:** common protocol의 사전 고정 출력 유전자 집합에 대해 intervention별 평균 발현의 ΔRMSE를 계산하고, 개입을 동일 가중치로 평균한다. 낮을수록 좋다. validation checkpoint 선택도 같은 집계를 기본으로 한다.

`mean(MSE)`와 `mean(sqrt(MSE))`는 다른 지표다. 원 논문 재현은 원 지표를 보존하고, common protocol에서 선택·보고 지표가 다르면 미리 이유를 명시한다.

보조지표는 ΔPearson, 반응 방향, evaluator-only 반응 유전자 순위 회복, no-change 대비 gain과 비용이다. top-20 absolute delta를 통계적 FDR DEG라고 부르지 않는다. 실제 DE 정답을 사용할 경우 그 계산법과 생물학적 반복의 유무를 명시한다.

raw expression Pearson 하나로 성능을 판단하지 않는다. 공통적인 처리 반응이나 baseline 발현 구조만 맞혀도 점수가 높아질 수 있다는 지적이 있으므로 단순 기준선·개입별 결과를 함께 본다. [S7–S8]

효과는 paired로 계산한다. 세포를 독립 생물학 표본처럼 bootstrap하지 않는다. 가능한 경우 생물학적 반복을 단위로 하고, 없으면 intervention 단위의 조건부 CI로 한정한다. 조합에서 유전자를 공유하거나 control을 공유하는 의존성도 적는다. 3개 seed의 평균·범위와 조건별 효과를 별도로 보고한다. 여러 후보를 검정하면 계획한 비교군에 대한 보정 여부를 명시한다.

“CI가 0을 포함함”은 효과가 정확히 없다는 증명이 아니다. 반대로 작은 양의 평균만으로 확정적 개선이라고 쓰지 않는다. **음성 결과도 완료된 연구 결과**다.

---

## 10. 간 관련 확장 H: 간암이라는 이름을 붙일 기준

Norman에서 학습한 GEARS를 그대로 HCC 환자에 넣어 일반화됐다고 하지 않는다. GEARS 공식 구현은 cross-cell-type transfer를 기본 지원 목표로 제시하지 않는다. [S2]

이번 확장의 기본 의미는 다음과 같다.

> 공개 benchmark에서 정리한 학습·튜닝 방법을 간 관련 개입 데이터에 적용하고, 해당 맥락에서 새로 학습한 원 방법과 개선 방법을 같은 조건으로 비교한다.

이는 **학습 방법의 이전**이지, 다른 세포주에 대한 zero-shot 모델 전이라고 자동 해석하지 않는다.

### 단계

1. 실제 HCC 자료 후보를 우선 3개 이내로 조사하고, 적합한 후보의 실제 파일·metadata를 확인한다. 유전자 개입과 화학물질 개입을 구분한다.
2. 적합한 자료가 확보되면 accession, 기원, control, guide/target, 시간, 반복, split을 고정한다. 원본 방법과 선정된 개선 방법의 두 조건을 먼저 비교한다.
3. HCC 개입 자료가 없거나 조건 수가 부족하면 `HCC_DATA_BLOCKED`로 명시한다. **간암 검증을 완료한 것으로 마감하지 않는다.**
4. 이 경우 HepG2 GSE264667을 간 유래 탐색 자료로 사용할 수는 있으나, 과거 노출 데이터라는 사실과 hepatoblastoma 기원을 함께 표시한다. 이전 모델·split·전처리 파일을 그대로 가져오지 않고 새 protocol로 실행한다.
5. 추가 학습 없이 쓸 수 없는 모델을 zero-shot 결과로 표시하지 않는다. 개입 방식·조건이 바뀌면 새 학습과 그 이유를 기록한다.

사람 HCC 개입 검증이 차단돼도 **R/M/D/F의 재현·모델링 연구는 독립적으로 완료·보고**한다. 간암 자료를 기다린다는 이유로 재현·튜닝 결과를 보고하지 않는 일은 피한다.

---

## 11. KG는 언제 붙이는가?

GEARS 원 모델에는 이미 유전자 관계 prior가 들어간다. 먼저 이 원래 구조를 재현한다. 별도의 RAG 시스템이나 거대한 biomedical KG를 먼저 만들지 않는다. [S1–S2]

추가 KG는 기본 재현·튜닝이 끝나고, **어떤 실패를 어떤 정보로 개선할지**가 명확할 때 하나의 후속 가설로 검토한다. 예를 들어 원 GO prior와 간 맥락 prior의 비교에는 실제/무작위 또는 구조 보존 대조가 필요하다.

검색 품질이나 문헌 설명이 좋아졌다는 것으로 RNA 반응 예측 성능 향상을 대신하지 않는다. 이번 첫 완료 조건에는 새 KG·RAG 서비스 구축을 포함하지 않는다.

---

## 12. 실행 운영과 재현 가능한 튜토리얼

### 12.1 자원 운영

- 서버 GPU·VRAM·CUDA·CPU·RAM·디스크·scheduler를 실제 조회한다. 과거 장비 기록만으로 현재 가용 자원을 단정하지 않는다.
- 모델별 환경을 새로 만들고 저자 버전 제약과 호환성을 기록한다. 모든 패키지를 최신 버전으로 맞추지 않는다.
- 원 학습 설정으로 실행 가능한지를 먼저 확인한다. OOM 때문에 sequence·genes·batch·정밀도를 바꾸면 변경 사실과 재현 상태를 갱신한다.
- 서로 독립적인 R1/R2, 이미 준비된 validation 실험은 허가된 자원 내에서 병렬 실행한다. GPU를 채우기 위한 무의미한 실험은 만들지 않는다.
- 같은 run의 중복 제출을 방지하고 checkpoint resume을 지원한다. 유료 API·cloud 추가 사용을 자동 승인하지 않는다.
- 긴 작업을 제출한 사실과 완료를 구분한다. 마지막 보고에서 `submitted/running/completed/failed/blocked`를 정확히 표시한다.

### 12.2 최소 저장 구조

```text
vcell_liver/
  README.md                       # 새 연구의 목적과 단일 진입점
  AGENTS.md                       # 새 지침만 활성화
  archive/
    README.md
    legacy_20261105/               # 구 계획/코드/튜토리얼/작은 보고서
  docs/
    REPRODUCTION_FIRST_RESEARCH_PLAN.md
  protocols/
    reproduction/                 # 논문별 원 조건
    common/                       # 새 비교 조건
  src/                            # 새 모델별 thin adapter와 공통 evaluator
  experiments/                    # R1/R2/C0/M/D/F/H 실행 진입점
  envs/                           # 새 환경과 성공한 lock
  manifests/                      # source·split·control IDs·exposure ledger
  reports/
    reproduction_comparison.tsv
    tuning_results.tsv
    liver_extension.md
    FINAL_REPORT.md
  tutorials/                      # 연구 실행과 같은 entrypoint 호출
```

원본 데이터·저자 가중치·새 학습 결과의 대용량 자산은 서버의 **새 연구 namespace**에 저장한다. archive run과 path가 섞이지 않게 한다. 공식 source code는 읽기 전용으로 pin하고 필요한 호환 변경은 patch로 관리한다. 이전 자체 scripts를 감싸기만 하고 새 구현이라고 부르지 않는다.

### 12.3 튜토리얼의 새 정의

튜토리얼은 연구의 주제가 아니라 **완료된 재현과 실험을 실행하는 사용 설명서**다.

| 튜토리얼 | 내용 | 실행 기본값 |
|---|---|---|
| 논문 재현 | 지정 논문 조건을 준비하고 실제 학습·예측·원 지표 계산 | `smoke`와 `full_reproduce`를 명확히 분리 |
| 튜닝 비교 | 같은 데이터에서 원 방법과 선택된 개선 방법을 실행 | 설정·학습 범위·출력을 보여줌 |
| 간 관련 적용 | 적격 데이터 준비부터 원 방법/개선 방법 비교 | HCC와 proxy 구분 |

노트북에서 저장된 TSV를 읽는 것은 결과 탐색이지 재현 실행이 아니다. **최소 하나의 full reproduction 명령을 실제로 실행한 로그·model·prediction·원 수치 대비표가 있어야 한다.** 사용자가 전체 재학습을 실수로 시작하지 않도록 비용이 드는 실행은 명시 옵션으로 둔다.

실행이 끝난 뒤 새 kernel에서 작은 입력 검증을 수행하고, 전체 재현의 기록을 연결한다. 숫자를 맞추기 위해 notebook output을 수동 입력하지 않는다.

---

## 13. 보고서와 완료 기준

### 보고서에는 네 가지 표를 반드시 둔다

**표 A — 논문 재현**

`논문/그림 | 데이터/분할 | 원 수치 | 재실행 값·반복 | 차이 | 변경사항 | 판정`

**표 B — 모델 튜닝**

`동일 task | M0/M1/M2/M3/random | 학습 파라미터 | validation 선택 | test 결과 | GPU/시간 | 효과`

**표 C — 데이터와 모델 효과**

`D0M0/D*M0/D0M*/D*M* | 데이터 ID | 동일 예측 대상 | 반복 결과 | paired 차이 | 한계`

**표 D — 간 관련 적용**

`생물학적 시스템 | 실제 개입 | train/val/test 단위 | 원 방법 | 개선 방법 | HCC인지 proxy인지 | 일반화 범위`

그 아래에서 실험마다 **목적 → 데이터와 선정 이유 → 모델과 선정 이유 → 학습·검증·시험 방식 → 결과 → 의미** 순서로 설명한다. 사용자가 파일 경로를 알아야만 이해할 수 있는 보고서는 만들지 않는다.

### 완료 기준

- 기존 작업을 archive하고 새 활성 지침·branch·출력 namespace를 분리했다.
- 선택한 두 모델의 원 논문 조건을 조사하고, 실행 가능한 조건을 실제 학습·평가했다.
- 논문 수치와의 일치/차이를 정량적으로 제시했다. 차단 모델은 이유와 실제 시도를 남겼다.
- 적어도 한 원 방법에 대해 실제 모델 튜닝을 수행했다. 단순 frozen embedding 분류로 대체하지 않았다.
- 데이터 중심 가설을 모델 고정 조건에서 비교하고, 유망 후보의 확인 또는 음성 종료 근거를 남겼다.
- 같은 입력·대조군·출력·평가 단위를 가진 단순 기준선이 있다.
- 간/HCC 확장은 실제 검증과 미확보를 구분해 보고했다. HCC 자료 미확보를 감추지 않았다.
- 튜토리얼은 실제 연구 실행 경로를 호출하고, 주요 실험을 재실행할 수 있다.
- 코드·config·작은 raw 결과·로그·source/split/model hash가 단계별 Git에 반영됐다.

**완료는 무조건 양의 성능 개선을 뜻하지 않는다.** 재현 불일치나 tuning 음성 결과도 원인·범위가 명확하면 연구 결과다. 다만 설치, 계획, 작은 smoke, 결과표 표시만으로 연구 완료라고 하지 않는다.

---

## 14. 근거와 설계 제안의 구분

본 기획서의 모델·원 논문 task·공개 자산 정보는 아래 출처에 근거한다. archive 방법, 실험 순서, common control 수, 제한된 tuning budget 등은 **이번 프로젝트를 위한 설계 제안**이다. 논문 기본값으로 인용하지 않는다.

- **[G1] 현재 저장소 branch 현황** — 2026-11-05 조회. `main`과 `research/phase2-v2`의 당시 HEAD 확인.
  - https://api.github.com/repos/honbul/vcell_liver/branches?per_page=100
- **[G2] 이전 공식 재현 기록** — 이전 기록을 archive할 범위를 확인한 자료. 새 실험 성능 근거로 재사용하지 않는다.
  - https://github.com/honbul/vcell_liver/blob/4077a263fc0575402d1e9844d6308a1336bcda55/reports/research_phase2/official_reproduction.md
- **[G3] 이전 활성 agent 지침** — 새 목표와 충돌하므로 archive·교체해야 하는 대상.
  - https://github.com/honbul/vcell_liver/blob/4077a263fc0575402d1e9844d6308a1336bcda55/AGENTS.md
- **[S1] Roohani et al. GEARS 원 논문** — Fig. 2e–f의 다중 섭동 평가, 원 평가 단위·반복 설계.
  - https://doi.org/10.1038/s41587-023-01905-6
- **[S2] GEARS 공식 구현** — 모델·데이터 API와 사용 범위. 실행 시 commit 고정 필요.
  - https://github.com/snap-stanford/GEARS
- **[S3] GEARS 그림·분석 재현 코드** — README가 원 논문의 figure/analysis 재현 및 legacy code를 포함한다고 명시. 해당 분석 파일과 버전을 실제로 해소할 것.
  - https://github.com/yhr91/GEARS_misc
- **[S4] Cui et al. scGPT 원 논문** — 공개 데이터·코드·Zenodo 안내 포함. 출판사 본문 접근 일부 제한.
  - https://doi.org/10.1038/s41592-024-02201-0
  - https://doi.org/10.5281/zenodo.10466117
- **[S5] scGPT Fig. 3 설명** — perturbation expression-change correlation과 DE 평가 구분.
  - https://www.nature.com/articles/s41592-024-02201-0/figures/3
- **[S6] scGPT 공식 perturbation tutorial** — TransformerGenerator 학습 경로. 확인한 기본값은 Adamson이며 논문 조건으로 자동 동일시하지 않음.
  - https://scgpt.readthedocs.io/en/stable/tutorial_perturbation.html
  - https://github.com/bowang-lab/scGPT
- **[S7] Ahlmann-Eltze et al., 2025** — 단순 선형·additive 기준선과의 비교 연구. 해당 조건에서의 결과이지 모든 모델·데이터에서의 우열 법칙은 아님.
  - https://doi.org/10.1038/s41592-025-02772-6
- **[S8] Viñas Torné et al., Systema** — 온라인 2025/권호 2026. 체계적 변동과 perturbation-specific 성능을 구분하는 평가 연구.
  - https://doi.org/10.1038/s41587-025-02777-8
- **[S9] GSE264667 공식 GEO 등록** — Jurkat/HepG2 Perturb-seq, day-7 scRNA 측정 등 실험 범위 확인.
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667
- **[S10] Cellosaurus Hep-G2** — hepatoblastoma 기원 명시.
  - https://www.cellosaurus.org/CVCL_0027

최종 원전·code snapshot·data manifest는 agent가 실제 획득한 bytes와 source revision을 기준으로 보완한다. 미확인 사항을 그럴듯한 설정이나 숫자로 채우지 않는다.
