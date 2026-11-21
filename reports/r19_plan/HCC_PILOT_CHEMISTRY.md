# 96군 독립 HCC 파일럿의 화합물 명부 초안

**기존 등록 자료와 겹치지 않는 적격 484군에서 96군을 배정했다.** 가장 가까운 구조 유사도 층은 목표 32군 중 25군이었고, P4에서 이득이 집중됐던 Tanimoto 0.70 이상은 9군이었다. 사전 규칙으로 빈자리를 다른 층에 배정했다. 전체 96군은 규모와 반복 변동을 추정할 파일럿의 초안이다. [사전 명세](HCC_PILOT_CHEMISTRY_PROTOCOL.md) · [96군 초안](data/hcc_pilot_chemistry/audit/draft_slots.tsv)

| P4 203군에 대한 최대 ECFP4 Tanimoto | 구조 식별 가능 | 대조 등록표 일치 없음 | 단계까지 적격 | 초안 배정 |
|---|---:|---:|---:|---:|
| 0.35 미만 | 1,898 | 597 | 409 | 36 |
| 0.35 이상 0.55 미만 | 255 | 79 | 48 | 35 |
| 0.55 이상 | 146 | 32 | 27 | 25 |
| 전체 | 2,299 | 708 | 484 | 96 |

PRISM 19Q4의 4,686개 등록행을 canonical parent로 접고 구조가 충돌한 19군을 제외했다. 남은 2,299군 중 1,591군은 기존 카탈로그 한 곳 이상에서 구조, ID 또는 이름으로 일치했다. 초안에 배정된 단계는 PRISM 원본 라벨 기준 Launched 54군, Phase 3 18군, Phase 2 24군이다. 합산한 708개 미일치군에는 이 세 단계 밖의 224군도 포함된다. 출처별 일치 수는 같은 parent를 여러 등록표에서 셀 수 있다. 입력은 CC BY 4.0의 PRISM treatment-info이고, 공개 viability 값은 읽지 않았다. [단계·유사도 원표](data/hcc_pilot_chemistry/audit/support.tsv) · [출처별 중복](data/hcc_pilot_chemistry/audit/exclusion_counts.tsv) · [PRISM 등록 원본](https://doi.org/10.6084/m9.figshare.9393293.v4)

채택된 96군의 최대 Tanimoto 중앙값은 0.385였다. 기존 P4와 같은 비어 있지 않은 Bemis–Murcko scaffold가 있는 군은 9개였고, 초안 안의 비어 있지 않은 scaffold 80개는 서로 달랐다. 나머지 16군은 acyclic이다. P4에서 관측된 가까운 학습 유사체 효과를 이 명부 전체로 확장할 근거는 없다. 대조 등록표에서 일치 항목이 없다는 판정은 해당 목록에 한정되며 외부 모델별 사전학습 노출 감사가 별도로 필요하다. [P4 구조 지원 결과](HCC_SCAFFOLD_RESULTS.md)

초안의 SNU-423과 HUH7 처리 좌표는 각각 288개다. 각 parent는 세포주별 세 배양 슬롯을 사용하며 두 세포주를 합친 여섯 웰 좌표가 서로 다르다. 좌표는 기존 익명 배치와 대조했고 처리 표 안에서 vehicle 역할이 비어 있는 것도 확인했다. 실제 배양의 독립성, 물질 lot·염·입체화학, 용매 적합성 및 QC는 실측 명부로 확인할 항목이다. 물질을 주문하거나 발현·생존율을 측정하지 않았다. 예약 코호트 정보는 기존 차단 명부의 식별자만 사용했다. [처리 좌표](data/hcc_pilot_chemistry/audit/draft_treatment_coordinates.tsv) · [배치 검산](HCC_PILOT_LAYOUT.md)

명세 커밋은 `3ea69c3`, 실행 스냅샷은 `bed60ca`다. G1 작업 `31438`은 g3에서 6분 32초 만에 종료 코드 0으로 완료했다. CPU 2개를 할당했고 GPU 사용은 0개였다. 최대 RSS는 637,880 KiB였다. 전체 감사 원표에서 행 순서를 섞고 다시 배정해 같은 96자리를 재현했다. 로컬 원표와 전체 감사 파일의 SHA-256을 영수증과 대조했다. 전체 감사 파일은 G1의 `vcell_liver_r19_plan/hcc-pilot-chem.cQGX4JCR/audit/candidate_audit.tsv`에 있다. [실행 원장](data/hcc_pilot_chemistry/jobs.tsv) · [입출력 해시](data/hcc_pilot_chemistry/audit/receipt.json)
