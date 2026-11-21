# R17 실행 원장

`research/r14-cross-organ-data`에서 `fd60409`의 요청과 검토 진단을 동기화했다. 지원 등록 코드는 `1066903`, R17 사전 명세 커밋은 `09d83cb`, 예측 코드는 `5b1b16b`, 동결·채점·기술 분석 코드는 `9818a38`이다. 재고 집계는 `fba8f77`에서 구조 확인 반복군과 시간·용량별 미기록군 표를 확정했다.

서버 루트는 `g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r17_lincs/v1`이다. RNA plate별 DMSO 웰 배정·발현·예측 배열은 서버에 보존한다. 원 R15 및 R16의 학습·동결·채점 산출물을 교체하지 않았다. GPU fit 배열은 동시 8개 이하로 제출했다.

| 단계 | Slurm ID | 산출물 |
|---|---|---|
| 위치·역할 등록, 처리 지원 감사 | `30522` | `support/` |
| 역할별 DMSO 평균 스테이징 | `30525` | `vehicles/` |
| 커널·결합 및 CPU 비교군, GPU 비교군 재적합 | `30526`, `30527` | `predict/` |
| 예측·선택 동결 | `30549` | `freeze.json` |
| 별도 주 채점 | `30550` | `score/` |
| 같은 동결의 별도 사영 기술 분석 | `30551` | `projection/` |
| 로컬 메타데이터 재고, 반복 지원 확장 | `30576` → `30595` | `inventory-v2/` |

[작업 ID 원표](data/jobs.tsv) · [사전 명세](PROTOCOL.md) · [등록 지원](data/support/support.tsv)

| 산출물 | SHA-256 |
|---|---|
| DMSO 스테이징 | `fabd541d9c9bae7a3df2640aa55475438cdb90072dbae2251bc708d7b1526a39` |
| 18개 조합·90개 예측 영수증 동결 | `78753c1da59a11f10f7c0469369571d42b224eaa039348bdeb40f65baada06f5` |
| 주 채점 영수증 | `db0804aae2952e3af47074fdfef39a3b97a88c39cb77b92e82897e8db970f602` |
| 기술 사영 영수증 | `d61dcb16e1196f6c3ca383780a5b607d65293308d605b4dc3482b5a4cbf67d0d` |
