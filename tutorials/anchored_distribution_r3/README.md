# tutorials/anchored_distribution_r3

두 노트북 모두 실행 결과가 저장돼 있다.

1. `01_inspect_results.ipynb` — 커밋된 작은 표만 읽어 진단 gate, dev 복원 비교, 최종 결과를 다시 그린다. GPU·서버·세포 자료가 필요 없다(pandas, matplotlib). ⑥에 사용자 승인 사후 pilot(DDPM·CellFlow, 탐색적) 표가 있다.
2. `02_control_to_treated_example.ipynb` — SNU-423의 한 조건에서 대조군 세포 → 평균 이동 → 복원 v1/v2 → Energy distance를 한 단계씩 따라간다.
   실제 세포 캐시가 필요하므로 서버 worktree에서 실행한다(`experiments/anchored_distribution_r3/env.sh`). 저장된 출력에는 요약 수치와 그림만 있고 세포 단위 표는 없다.

대조군과 처리군은 서로 다른 세포다. 두 번째 노트북은 같은 세포의 전후를 보여 주는 것이 아니다.
