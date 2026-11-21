# P4 실행 기록

[사전 명세](PROTOCOL.md)는 커밋 `be86dd8`이다. 명세 SHA-256은 `38f7030d273fa719d1cab924d44819630afb72f17fcd0ff2541b70fadf70d993`이다. G1 실행 루트는 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_p4/v1`이다. `predict-code`는 `099f335`, `score-code`는 `7e14799`의 소스 스냅샷을 사용했다. 두 예측 배열은 각 36개 용량·최소세포·fold·DMSO 역할 조합을 포함한다. 모든 작업은 CPU 큐에서 실행했다.

| 단계 | Slurm ID | 의존성 | 서버 원표 위치 |
|---|---|---|---|
| P4 새 화학 arm 예측 | `29465` | 기존 T1·P1·P2 동결 | `predictions/chemical/` |
| 기존 VC 5종 P4 재적합 | `29466` | 기존 T2 동결 | `predictions/vc/` |
| 5 µM 동결 → 채점 | `29538` → `29539` | `29465`, `29466` 예측 성공 | `d5/` |
| 0.5 µM 동결 → 채점 | `29540` → `29541` | `29465`, `29466` 예측 성공 | `d0.5/` |
| 0.05 µM 동결 → 채점 | `29542` → `29543` | `29465`, `29466` 예측 성공 | `d0.05/` |

Slurm 제출 원장은 이 루트의 `predict-jobs.tsv`, `score-jobs.tsv`다. 각 동결은 12개 조합의 P4 새 예측과 과거 동결 예측의 영수증을 SHA-256으로 묶는다. 채점은 동결을 재검증한 뒤에만 개발 교차검증 처리 실측을 연다. 봉인·격리·예약 범위에 대한 새 접근은 없다.

| 용량 | 예측 동결 SHA-256 | 별도 채점 영수증 SHA-256 |
|---|---|---|
| 5 µM | `b45a19b17c9e3ef66a9dffe467761cf54914cb9417b03c21f63d0fefcf17b577` | `b63348d957b78dbe7f3e73ab91117819bedb5d61494307f4e728d94bd4674644` |
| 0.5 µM | `bcd019f9af8d6627d70cbad905edaa24aad48bd0863267bcc95ad835f35af49b` | `8913e4db278d5937f17d2c3c9c6edf36b1e90dae914750bf35910d7706a67d6d` |
| 0.05 µM | `5bca50440f35f96bfc52be07796312e60055cb188b7e351da193500de2d27d06` | `3f870990e4324dd33b9ae5f0b89f62c8b5bbd5c76cc7d71c6014ad67b82c2379` |

채점 영수증과 논문용 TSV를 `data/`에 복사하고 원본 해시를 대조했다. [예측 제출 원장](data/predict-jobs.tsv) · [동결·채점 제출 원장](data/score-jobs.tsv)
