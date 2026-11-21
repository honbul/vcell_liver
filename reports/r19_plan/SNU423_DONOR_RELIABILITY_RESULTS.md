# SNU-423 외부 RNA의 독립 위치 반복 지원

**비간계열 RNA donor를 보유한 100군 중 다른 RNA plate·다른 웰 위치·같은 실측 농도 조건을 충족한 화합물은 38군이었다.** 사전 등록한 전체 50군 문턱에 미달해, source 반복 일치도로 가중하는 모델의 학습과 채점은 시작하지 않았다. [사전 명세](SNU423_DONOR_RELIABILITY_PROTOCOL.md) · [기존 donor 비교](SNU423_DONOR_RESULTS.md)

| 지원 | 전체 | Fold 0 · 1 · 2 |
|---|---:|---|
| 원 비간계열 donor 보유 | 100 | 34 · 31 · 35 |
| 같은 위치, 다른 plate 반복 보유 | 100 | 34 · 31 · 35 |
| 다른 위치, 다른 plate 반복 보유 | 38 | 12 · 14 · 12 |
| 다른 위치 반복을 보유한 donor 세포주가 둘 이상 | 24 | 5 · 11 · 8 |

원 모델이 고른 5 µM 최근접 실측 농도에서 386개 parent×세포주 입력 중 92개가 독립 위치 반복을 보유했다. 5–20 µM의 다른 등록 농도까지 확인해도 독립 위치 반복이 있는 화합물은 38군, 두 세포주 이상 지원은 24군이었다. 적격 parent×세포주만 92개에서 95개로 늘었다. SNU-423 평가와 동일한 **실측 5 µM**에서 독립 위치 반복은 0군이었다. 위치 기준을 적용하기 전 연구 결과에는 실측 5 µM donor가 2군 있었다. 이 감사는 반복 배치와 농도를 검증했으며, 반복 일치도나 모델의 효과는 측정하지 않았다. [fold별 지원](data/snu_donor_reliability/audit/support.tsv) · [농도 민감도](data/snu_donor_reliability/audit/dose_support.tsv)

명세 `aabb3ae`에서 시작했다. 최초 G1 작업 `31541`은 기존 메타데이터의 문자열 `24.0`을 정수 문자열과 비교하다 중단됐다. 숫자 비교로 고친 `31542`에서 원 지원을 확인했고 `31543`에서 다른 등록 농도의 지원을 추가 집계했다. 두 완료 작업의 공통 원표는 일치했다. 신규 수집, GPU, source·target·R18 발현 읽기는 없었다. [작업 원장](data/snu_donor_reliability/jobs.tsv) · [감사 영수증](data/snu_donor_reliability/audit/receipt.json) · [자원 내역](data/snu_donor_reliability/resource_usage.tsv)
