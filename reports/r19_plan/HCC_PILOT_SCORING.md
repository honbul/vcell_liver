# 96군 독립 배양 파일럿 합성 검증

**실제 HCC 발현 없이 16개 plate의 검증과 별도 채점을 끝까지 실행했다.** P001–P096 익명 배치와 생성한 화합물·배양·발현 ID만 사용했다. 예측 동결 명령은 처리 발현 파일의 경로를 받지 않았다. 별도 채점 명령에서 동결과 입력 해시를 검증한 후 합성 처리값을 읽었다. 함수 실행과 두 개의 별도 명령, 같은 seed로 새로 생성한 실행 사이에서 채점 파일과 동결이 바이트 단위로 일치했다. [사전 명세](HCC_PILOT_SCORING_PROTOCOL.md) · [동결 영수증](data/hcc_pilot_scoring/freeze.json) · [채점 영수증](data/hcc_pilot_scoring/score/receipt.json)

| 세포주 | 전체 적격 화합물군 / 웰 / 배양 | 독립 반복 지원군 / 웰 / 배양 |
|---|---:|---:|
| SNU-423 | 96 / 288 / 8 | 96 / 288 / 8 |
| HUH7 | 96 / 251 / 7 | 95 / 250 / 7 |

HUH7 한 plate의 A vehicle 2개를 사전 QC에서 실패한 것으로 입력했다. 합성 실행의 역할당 최소 3개 조건에서 이 plate의 36 처리 웰이 빠졌다. 다른 plate의 처리 한 개를 추가로 QC 실패로 입력해 한 화합물은 적격 웰 하나만 남았다. 전체 적격 채점에는 이 웰이 포함됐고 다른 배양 반복과 비교하는 공통 표에서는 빠졌다. 총 539개 물리 웰의 세 vehicle 회전을 동일 관측 묶음으로 평가했다. [plate별 지원](data/hcc_pilot_scoring/score/support.tsv)

같은 culture ID를 두 plate slot에 부여하거나, 같은 parent ID를 두 화합물 slot에 부여하거나, 채점 vehicle을 예측 선택용으로 선언하거나, 동결 해시 또는 실측 유전자 순서를 바꾸면 실패했다. 동결 해시 오류는 측정 파일 접근 전에 차단했다. 이 검사는 공급한 access 명부의 일치와 독립성에 필요한 ID 관계를 검사한다. 실제 배양의 독립성과 예측 과정의 학습 노출은 실측 명부와 별도 실행 감사로 확인해야 한다. [오류 주입 체크](data/hcc_pilot_scoring/negative_checks.tsv)

고정 B0와 다른 위치·배양 반복, 가중 재표집마다 재선택한 최선 기존 VC 형식의 대조를 계산했다. parent만 또는 parent와 배양을 함께 재표집하는 두 99% 구간을 같은 웰의 대비로 검증했다. 합성 후보에는 생성 원신호를 제공했으므로 이 원표의 순위와 점수는 모델 성능 근거로 사용할 수 없다. 반복 참조 구간은 구성된 donor 벡터에 조건부다. [합성 점수](data/hcc_pilot_scoring/score/points.tsv) · [합성 대응 차이](data/hcc_pilot_scoring/score/paired.tsv)

합성 실행은 `python -m src.r19_plan.hcc_pilot_synthetic`에 `--root`, 기존 배치표의 `--layout` 및 `--layout-receipt`, 이 채점의 `--protocol`을 지정해 재현한다. 실제 실험에는 익명 자리의 화합물 매핑, 승인된 QC, 배양·plate·접근 명부 및 동결 예측이 필요하다. 현재 파일럿의 측정과 R18 예약 발현 채점은 수행되지 않았다.
