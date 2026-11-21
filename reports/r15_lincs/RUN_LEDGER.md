# R15 실행 기록

학습 전 감사 결과는 `6ec6378`, [판정 사전 명세](PROTOCOL.md)는 `1213ffc`다. 사전 명세 SHA-256은 `ad8a19393c514d5b45cfa0bc98d34a3f871cad01e352c2f6df9113a42fd9a245`다. 서버 실행 루트는 `g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r15_lincs/v1`이다.

| 실행 | Slurm ID | 출력 |
|---|---|---|
| GSE92742와 GSE70138 위치 감사 | `29560`, `29561` | `audit/GSE92742/`, `audit/GSE70138/` |
| 노출 감사와 detection plate 확인 | `29562`, `29567` | `audit/exposure-verified/` |
| GSE92742와 GSE70138 선택 Level 3 추출 | `29570`, `29571` | `cache/GSE92742/`, `cache/GSE70138/` |
| 구조 전용 ChemBERTa 특징과 유전자 표 | `29572` | `model-inputs/` |
| LINCS 기존 VC CPU·GPU 앱 | `29574`, `29575` | `predict/{A,B}/` |
| LINCS 제안 예측 | `29610` | `predict/{A,B}/` |
| LINCS 동결 → 별도 채점 | `29579` → `29580` | `internal/` |
| Tahoe 외부 예측 → 동결 → 별도 채점 | `29611` → `29612` → `29613` | `external/` |
| Tahoe 외부 방향·크기 사후 기술 분석 | `29677` | `external/descriptive/` |

위치 감사 최초 두 작업 `29558`, `29559`는 GCTX 라이브러리 경로가 빠져 시작 전에 종료됐다. 기존 서버 환경을 다시 사용해 원본·배치 계약을 유지했다. 제안 예측 최초 작업 `29573`은 parent 평균 배열의 반환 형식을 잘못 해석해 출력 전에 실패했다. 반환 배열 처리만 수정한 `2b463ba`에서 해당 예측을 다시 제출했다. 같은 수정이 필요한 기존 외부 예측 의존성 `29576`–`29578`은 미채점 상태에서 교체했다. LINCS VC 학습은 최초 제출을 유지했다. GPU 배열 `29575`는 `%8`로 제한했다.

외부 평가의 HEPG2 모델은 Tahoe 평가 parent를 제외한 LINCS 학습 결과와 예측 배열을 동결한다. Tahoe의 T1/T2 비교군 원본 영수증도 같은 동결에서 대조한다. 원본 발현·예측 배열·모델은 서버에 보관한다.

| 범위 | 예측 동결 SHA-256 | 별도 채점 영수증 SHA-256 |
|---|---|---|
| LINCS 내부 96개 예측과 12개 fold·역할·시나리오 조합 | `02dbc1a8d9258b267863283009f8265d8db324f8d465080ab4fc47aa32aae9a7` | `ec8ea9105ef14539aa455fe1a4848139238044acd2f467ce97f9d60a4218fa40` |
| Tahoe 외부 4개 동결 모델과 12개 T1/T2 비교 조합 | `32c46e8d7a9c90eb29f6ee62cde650ae2765300920ddb0266ffe0db40a21aac0` | `44be0e43e85dba15b4695d9a2c4c27096829e4abbb07cb96a8083affe0be9622` |

[캐시 Slurm ID](data/cache-jobs.tsv) · [예측·동결·채점 Slurm ID](data/eval-jobs.tsv)
