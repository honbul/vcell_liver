# 이전 연구 복구

2026-11-05 새 연구 시작 전 다음 두 커밋을 변경하지 않는 아카이브 참조로 보존했다. 기존 브랜치도 보존한다. 이 참조와 경로 인덱스가 archive/legacy_20261105/ 전체 복제본을 대신한다.

| 참조 | 커밋 | 복구 범위 |
|---|---|---|
| `archive/legacy_20261105/main` | `243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79` | V1 코드·계획·설정·노트북·보고서 |
| `archive/legacy_20261105/phase2-v2` | `4077a263fc0575402d1e9844d6308a1336bcda55` | V2/V3까지의 코드·실험 기록·노트북·작은 결과 |

예: `git worktree add --detach ../vcell-liver-legacy archive/legacy_20261105/phase2-v2`. 원 경로는 `git ls-tree -r --name-only archive/legacy_20261105/phase2-v2`로 확인한다. 특정 파일은 `git show archive/legacy_20261105/phase2-v2:AGENTS.md`처럼 읽는다.

G1의 기존 workbench와 데이터 디렉터리를 그대로 보존한다. [자산 복구 인덱스](../manifests/legacy_assets.tsv)의 경로는 이동하지 않았다. 기존 소스 및 가상환경은 실행 경로에서 제외한다. 기존 체크포인트·예측·표현은 이전 연구 증거이며 이번 연구의 학습 입력이나 대조 결과로 쓰지 않는다. 공개 배포본만 출처와 실제 해시를 확인해 예외로 등록한다.

미커밋 V3 검토 원본은 로컬 프로젝트 상위의 `archive-private/legacy_20261105`에 복사했다. G1의 미커밋 환경 기록과 추적 파일 삭제 차이는 G1의 별도 제한 디렉터리 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/archive-private/legacy_20261105`에 보존했다. Git에 비공개 원본은 싣지 않는다. 아카이브 당시 gssong Slurm 목록은 비어 있었다.
